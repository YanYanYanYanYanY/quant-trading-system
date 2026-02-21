"""
Low-volatility alpha model (signed).

Scores each symbol by the *negated* realised volatility, z-scored
cross-sectionally and winsorised to [-3, 3].  Lower-vol stocks receive
a positive signal; higher-vol stocks receive a negative signal.  The
long-only router downstream clips negative values.

Feature selection:
* Prefers ``rv60`` (60-day realised vol) when available.
* Falls back to ``rv20`` (20-day realised vol) when ``rv60`` is absent.

Confidence is a flat **0.7** for every symbol with data — the low-vol
anomaly is relatively stable cross-sectionally and doesn't need a
dynamic confidence model.

Pipeline
--------
1. For each symbol, read ``rv60`` (preferred) or ``rv20`` (fallback).
2. ``raw_signal = −rv`` (lower vol → positive).
3. Cross-sectional z-score across the universe (NaN-safe).
4. Winsorise z-scores to [-3, 3].
5. ``score = z``, ``raw = z`` (may be negative).
6. ``confidence = 0.7`` (constant when data present; 0 for missing).
"""

from __future__ import annotations

import math
from typing import Dict, List

import pandas as pd  # used for pd.Timestamp in predict() signature

from ..types import (
    AlphaModelName,
    AlphaScore,
    cross_sectional_zscore,
    winsorize,
)
from ...regime.stock_types import FeatureBundle  # re-exported canonical type

_CONFIDENCE: float = 0.7
"""Constant confidence for scored symbols."""


class LowVolAlpha:
    """
    Cross-sectional low-volatility alpha model.

    Conforms to the :class:`~engine.strategy.alpha.types.AlphaModel`
    protocol via structural subtyping (no explicit inheritance).

    The low-volatility anomaly — lower-risk stocks tend to deliver
    higher risk-adjusted returns — is one of the most robust cross-
    sectional factors.  This model captures it with a simple negated-vol
    z-score.

    Input features (in order of preference):

    * ``rv60`` — 60-day annualised realised volatility.
    * ``rv20`` — 20-day annualised realised volatility (fallback).

    Examples
    --------
    >>> model = LowVolAlpha()
    >>> scores = model.predict(
    ...     asof=pd.Timestamp("2024-06-15"),
    ...     features={"rv60": rv60_df},
    ...     universe=["AAPL", "MSFT", "TSLA"],
    ... )
    >>> scores["AAPL"].score   # winsorised z-score, may be < 0
    0.87
    """

    # -- AlphaModel protocol: name property ----------------------------------

    @property
    def name(self) -> str:
        """Return the canonical model name."""
        return AlphaModelName.LOWVOL.value

    # -- AlphaModel protocol: predict ----------------------------------------

    def predict(
        self,
        asof: pd.Timestamp,
        features: FeatureBundle,
        universe: List[str],
    ) -> Dict[str, AlphaScore]:
        """
        Generate per-symbol low-volatility alpha scores.

        Parameters
        ----------
        asof : pd.Timestamp
            Evaluation date.
        features : FeatureBundle
            Should contain ``"rv60"`` and/or ``"rv20"`` feature columns.
        universe : list[str]
            Symbols to score.

        Returns
        -------
        dict[str, AlphaScore]
            One entry per symbol in *universe*.  Symbols with no
            volatility data receive ``score=0, raw=0, confidence=0``
            with ``reasons["missing_data"] = 1``.
        """
        # ── Step 1: extract per-symbol vol (rv60 preferred, rv20 fallback)
        rv_vals: Dict[str, float] = {}
        rv_source: Dict[str, str] = {}
        missing: set[str] = set()

        for sym in universe:
            rv60 = features.get(sym, "rv60")
            if not math.isnan(rv60):
                rv_vals[sym] = rv60
                rv_source[sym] = "rv60"
                continue

            rv20 = features.get(sym, "rv20")
            if not math.isnan(rv20):
                rv_vals[sym] = rv20
                rv_source[sym] = "rv20"
                continue

            missing.add(sym)

        # ── Step 2: raw_signal = -rv (lower vol → positive) ───────────
        raw_signal: Dict[str, float] = {
            sym: -rv_vals[sym] for sym in rv_vals
        }

        # ── Step 3: cross-sectional z-score ───────────────────────────
        z_scores = cross_sectional_zscore(raw_signal)

        # ── Step 4: winsorise to [-3, 3] ──────────────────────────────
        z_scores = {s: winsorize(z) for s, z in z_scores.items()}

        # ── Step 5 & 6: build AlphaScore per symbol ───────────────────
        result: Dict[str, AlphaScore] = {}

        for sym in universe:
            if sym in missing:
                result[sym] = AlphaScore(
                    symbol=sym,
                    score=0.0,
                    raw=0.0,
                    confidence=0.0,
                    reasons={
                        "rv_used": 0.0,
                        "z": 0.0,
                        "missing_data": 1.0,
                    },
                )
                continue

            z = z_scores[sym]
            rv = rv_vals[sym]

            result[sym] = AlphaScore(
                symbol=sym,
                score=z,
                raw=z,
                confidence=_CONFIDENCE,
                reasons={
                    "rv_used": rv,
                    "rv_source": 60.0 if rv_source[sym] == "rv60" else 20.0,
                    "z": z,
                },
            )

        return result
