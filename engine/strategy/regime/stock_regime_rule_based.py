"""
Rule-based stock regime detector.

Classifies each stock in a universe into one of:
  TRENDING_UP, TRENDING_DOWN, RANGE_LOW_VOL, STRESSED_HIGH_VOL, NO_TRADE
using **pre-computed** per-symbol features from a
:class:`~engine.features.types.FeatureBundle`, market regime context,
hysteresis, and confidence gating.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .market_types import MarketRegimeState
from .stock_regime_config import StockRegimeConfig
from .stock_types import (
    FeatureBundle,
    StockRegimeLabel,
    StockRegimeMap,
    StockRegimeState,
    clamp01,
)

_SQRT_252: float = 15.874507866  # √252


# ---------------------------------------------------------------------------
# Per-symbol internal state
# ---------------------------------------------------------------------------

@dataclass
class _SymbolState:
    """Mutable state held per symbol for hysteresis / min-duration."""
    label: StockRegimeLabel = StockRegimeLabel.NO_TRADE
    confidence: float = 0.0
    switch_ts: Optional[pd.Timestamp] = None
    days_in_regime: int = 0


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class RuleBasedStockRegimeDetector:
    """
    Rule-based stock regime detector.

    Classifies each stock in a universe using per-stock features and
    market regime context.  Conforms to the ``StockRegimeDetector``
    protocol (structural subtyping — no explicit inheritance needed).

    Parameters
    ----------
    config : StockRegimeConfig, optional
        Detection thresholds and window parameters.

    Notes
    -----
    *   The detector is **stateful**: it stores per-symbol regime,
        confidence, and switch timestamp between calls to ``detect``.
    *   Call ``reset()`` before a new backtest run.
    *   When ``STRESSED_HIGH_VOL`` is the raw classification the
        min-duration lock is **bypassed** — stress always enters
        immediately (safety first).
    """

    def __init__(self, config: Optional[StockRegimeConfig] = None) -> None:
        self.config = config or StockRegimeConfig()
        self._state: Dict[str, _SymbolState] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all per-symbol state (call before a new backtest)."""
        self._state.clear()

    def detect(
        self,
        asof: pd.Timestamp,
        market_regime: MarketRegimeState,
        features: FeatureBundle,
        universe: List[str],
    ) -> StockRegimeMap:
        """
        Detect regime for every symbol in *universe* at *asof*.

        Parameters
        ----------
        asof : pd.Timestamp
            Assessment date.
        market_regime : MarketRegimeState
            Current market-level regime.
        features : FeatureBundle
            ``{"close": DataFrame, "rv20": DataFrame, "mom63": DataFrame, ...}``
            Each DataFrame has a ``DatetimeIndex`` (rows = dates) and
            columns = symbols.
        universe : List[str]
            Symbols to classify.

        Returns
        -------
        StockRegimeMap
        """
        cfg = self.config
        mkt_stressed = market_regime.is_stressed

        states: Dict[str, StockRegimeState] = {}
        for sym in universe:
            signals = self._extract_signals(asof, sym, features, cfg, mkt_stressed)
            if signals is None:
                # Insufficient data → NO_TRADE
                out = self._apply_stability(
                    sym, asof, StockRegimeLabel.NO_TRADE, 0.0,
                )
                states[sym] = out
                continue

            raw_label, raw_conf = self._classify(sym, signals, cfg, mkt_stressed)
            out = self._apply_stability(sym, asof, raw_label, raw_conf)
            # Attach diagnostic scores
            out = StockRegimeState(
                symbol=out.symbol,
                asof=out.asof,
                label=out.label,
                confidence=out.confidence,
                reasons=signals,
            )
            states[sym] = out

        return StockRegimeMap(asof=asof, states=states)

    # ------------------------------------------------------------------
    # Signal extraction (reads pre-computed features from FeatureBundle)
    # ------------------------------------------------------------------

    def _extract_signals(
        self,
        asof: pd.Timestamp,
        sym: str,
        features: FeatureBundle,
        cfg: StockRegimeConfig,
        mkt_stressed: bool,
    ) -> Optional[Dict[str, float]]:
        """
        Extract all per-stock signals at *asof* from *features*.

        Reads pre-computed scalar values from the
        :class:`~engine.features.types.FeatureBundle`.  Returns ``None``
        when required features (``normalized_slope``, ``rv20``) are
        missing or NaN, which causes the caller to classify the symbol
        as ``NO_TRADE``.
        """
        # ── Read required features ──────────────────────────────────────
        n_slope = features.get(sym, "normalized_slope")
        rv20_now = features.get(sym, "rv20")

        # Both are mandatory — without them classification is unreliable.
        if not math.isfinite(n_slope) or not math.isfinite(rv20_now):
            return None

        # ── Read optional features (NaN → safe default) ────────────────
        vol_z = features.get(sym, "rv20_z")
        max_dd = features.get(sym, "max_dd_90")
        mom63 = features.get(sym, "mom63")
        trend_gap = features.get(sym, "trend_gap")

        # Derive ma_trend from trend_gap (positive gap = bullish cross)
        if math.isfinite(trend_gap):
            ma_trend = 1.0 if trend_gap > 0 else -1.0
        else:
            ma_trend = 0.0

        # Derive daily_vol from annualised rv20 (diagnostic only)
        daily_vol = rv20_now / _SQRT_252

        return {
            "normalized_slope": n_slope,
            "daily_vol": daily_vol,
            "vol_z": vol_z if math.isfinite(vol_z) else 0.0,
            "max_dd": max_dd if math.isfinite(max_dd) else 0.0,
            "mom63": mom63 if math.isfinite(mom63) else 0.0,
            "ma_trend": ma_trend,
            "rv20": rv20_now,
        }

    # ------------------------------------------------------------------
    # Classification (pure, except for reading previous label)
    # ------------------------------------------------------------------

    def _classify(
        self,
        sym: str,
        sig: Dict[str, float],
        cfg: StockRegimeConfig,
        mkt_stressed: bool,
    ) -> Tuple[StockRegimeLabel, float]:
        """
        Classify a single symbol into a regime label + confidence.
        """
        prev = self._state.get(sym)
        prev_label = prev.label if prev else StockRegimeLabel.NO_TRADE

        # ── Effective stress thresholds (market conditioning) ───────────
        enter_z = cfg.enter_stress_z
        if mkt_stressed:
            enter_z -= cfg.market_stress_z_adj  # easier to enter stress

        # ── STRESS check (highest priority) ─────────────────────────────
        is_stressed, stress_conf = self._check_stress(
            sig, prev_label, enter_z, cfg,
        )
        if is_stressed:
            return StockRegimeLabel.STRESSED_HIGH_VOL, stress_conf

        # ── TREND check ─────────────────────────────────────────────────
        trend_label, trend_conf = self._check_trend(sig, prev_label, cfg)
        if trend_label is not None:
            conf = trend_conf
            if mkt_stressed:
                conf *= cfg.market_stress_conf_penalty
            return trend_label, clamp01(conf)

        # ── RANGE / LOW-VOL check ──────────────────────────────────────
        range_conf = self._check_range(sig, cfg)
        if range_conf > 0:
            return StockRegimeLabel.RANGE_LOW_VOL, clamp01(range_conf)

        # ── Fallback → RANGE_LOW_VOL with moderate confidence ──────────
        return StockRegimeLabel.RANGE_LOW_VOL, 0.45

    # ── Stress sub-check ────────────────────────────────────────────────

    @staticmethod
    def _check_stress(
        sig: Dict[str, float],
        prev_label: StockRegimeLabel,
        enter_z: float,
        cfg: StockRegimeConfig,
    ) -> Tuple[bool, float]:
        """
        Returns (is_stressed, confidence).

        Hysteresis: use different thresholds when already stressed.
        """
        vol_z = sig["vol_z"]
        max_dd = sig["max_dd"]
        currently_stressed = prev_label == StockRegimeLabel.STRESSED_HIGH_VOL

        if currently_stressed:
            z_thresh = cfg.exit_stress_z
            dd_thresh = cfg.exit_stress_dd
        else:
            z_thresh = enter_z
            dd_thresh = cfg.enter_stress_dd

        vol_hit = vol_z >= z_thresh
        dd_hit = max_dd <= dd_thresh

        if not (vol_hit or dd_hit):
            return False, 0.0

        # Confidence: how far past threshold
        vol_margin = max(0.0, (vol_z - z_thresh) / 2.0) if vol_hit else 0.0
        dd_margin = max(0.0, (dd_thresh - max_dd) / abs(dd_thresh) / 2.0) if dd_hit else 0.0
        conf = clamp01(0.55 + max(vol_margin, dd_margin) * 0.45)
        return True, conf

    # ── Trend sub-check ─────────────────────────────────────────────────

    @staticmethod
    def _check_trend(
        sig: Dict[str, float],
        prev_label: StockRegimeLabel,
        cfg: StockRegimeConfig,
    ) -> Tuple[Optional[StockRegimeLabel], float]:
        """
        Returns (label_or_None, confidence).

        Hysteresis on slope; momentum sign must agree.
        """
        slope = sig["normalized_slope"]
        mom63 = sig["mom63"]
        ma_trend = sig["ma_trend"]

        # -- Determine effective slope threshold --
        was_up = prev_label == StockRegimeLabel.TRENDING_UP
        was_down = prev_label == StockRegimeLabel.TRENDING_DOWN

        # Uptrend candidate
        if was_up:
            up_ok = slope >= cfg.exit_trend_slope
        else:
            up_ok = slope >= cfg.enter_trend_slope
        up_ok = up_ok and (mom63 > 0)

        # Downtrend candidate
        if was_down:
            down_ok = slope <= -cfg.exit_trend_slope
        else:
            down_ok = slope <= -cfg.enter_trend_slope
        down_ok = down_ok and (mom63 < 0)

        if not (up_ok or down_ok):
            return None, 0.0

        # -- Pick direction --
        if up_ok and down_ok:
            # Tie-break: whichever slope is stronger
            label = (
                StockRegimeLabel.TRENDING_UP
                if slope > 0
                else StockRegimeLabel.TRENDING_DOWN
            )
        elif up_ok:
            label = StockRegimeLabel.TRENDING_UP
        else:
            label = StockRegimeLabel.TRENDING_DOWN

        # -- Confidence --
        slope_mag = abs(slope)
        base = clamp01(0.5 + (slope_mag - cfg.exit_trend_slope) / 3.0)

        # Agreement bonus: MA trend agrees with direction
        direction = 1.0 if label == StockRegimeLabel.TRENDING_UP else -1.0
        agree_bonus = 0.1 if (ma_trend * direction > 0) else -0.05

        # Momentum magnitude bonus
        mom_bonus = clamp01(abs(mom63) * 2.0) * 0.1

        conf = clamp01(base + agree_bonus + mom_bonus)
        return label, conf

    # ── Range sub-check ─────────────────────────────────────────────────

    @staticmethod
    def _check_range(
        sig: Dict[str, float],
        cfg: StockRegimeConfig,
    ) -> float:
        """
        Returns confidence in RANGE_LOW_VOL (0 if criteria not met).
        """
        slope = abs(sig["normalized_slope"])
        rv20 = sig["rv20"]

        if slope > cfg.range_slope_ceil:
            return 0.0

        # Confidence increases as slope → 0 and rv20 → 0
        slope_score = clamp01(1.0 - slope / cfg.range_slope_ceil)
        vol_score = clamp01(1.0 - rv20 / cfg.low_vol_pct) if cfg.low_vol_pct > 0 else 0.5
        return clamp01(0.45 + 0.35 * slope_score + 0.20 * vol_score)

    # ------------------------------------------------------------------
    # Stability controls
    # ------------------------------------------------------------------

    def _apply_stability(
        self,
        sym: str,
        asof: pd.Timestamp,
        raw_label: StockRegimeLabel,
        raw_conf: float,
    ) -> StockRegimeState:
        """
        Apply min-duration and confidence-gate, then update per-symbol state.
        """
        cfg = self.config
        prev = self._state.get(sym)

        if prev is None:
            # First observation — accept raw classification
            self._state[sym] = _SymbolState(
                label=raw_label,
                confidence=raw_conf,
                switch_ts=asof,
                days_in_regime=1,
            )
            return StockRegimeState(
                symbol=sym, asof=asof, label=raw_label,
                confidence=clamp01(raw_conf),
            )

        # Same label → just increment counter
        if raw_label == prev.label:
            prev.days_in_regime += 1
            prev.confidence = raw_conf
            return StockRegimeState(
                symbol=sym, asof=asof, label=prev.label,
                confidence=clamp01(raw_conf),
            )

        # Attempting a switch — stress always allowed immediately
        if raw_label == StockRegimeLabel.STRESSED_HIGH_VOL:
            prev.label = raw_label
            prev.confidence = raw_conf
            prev.switch_ts = asof
            prev.days_in_regime = 1
            return StockRegimeState(
                symbol=sym, asof=asof, label=raw_label,
                confidence=clamp01(raw_conf),
            )

        # Min-duration check
        if prev.days_in_regime < cfg.min_regime_duration:
            prev.days_in_regime += 1
            return StockRegimeState(
                symbol=sym, asof=asof, label=prev.label,
                confidence=clamp01(raw_conf * 0.5),
            )

        # Confidence gate
        if raw_conf < cfg.confidence_gate:
            prev.days_in_regime += 1
            return StockRegimeState(
                symbol=sym, asof=asof, label=prev.label,
                confidence=clamp01(raw_conf),
            )

        # Switch allowed
        prev.label = raw_label
        prev.confidence = raw_conf
        prev.switch_ts = asof
        prev.days_in_regime = 1
        return StockRegimeState(
            symbol=sym, asof=asof, label=raw_label,
            confidence=clamp01(raw_conf),
        )
