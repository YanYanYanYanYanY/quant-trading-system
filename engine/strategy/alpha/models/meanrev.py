"""
Mean-reversion alpha model (signed).

Scores each symbol by the *negated* 20-day deviation from the mean
(``-dev20``), z-scored cross-sectionally and winsorised to [-3, 3].
A stock trading **below** its recent mean gets a positive signal; one
trading **above** gets a negative signal.  The long-only router
downstream clips negative values — this model preserves the full
signed score.

Confidence is the product of two factors:
* *signal strength*: how far the stock has deviated (``|dev20| / 2``).
* *vol discount*: penalises high-volatility regimes where mean-
  reversion is less reliable (``1 − max(0, rv20_z) / 2``).

Pipeline
--------
1. Extract ``dev20`` and ``rv20_z`` at *asof* for every symbol.
2. ``raw_signal = −dev20`` (flip sign: below-mean → positive).
3. Cross-sectional z-score across the universe (NaN-safe).
4. Winsorise z-scores to [-3, 3].
5. ``score = z``, ``raw = z`` (may be negative).
6. ``confidence = clamp01(|dev20| / 2) × clamp01(1 − max(0, rv20_z) / 2)``.
"""

from __future__ import annotations

import math
from typing import Dict, List

import pandas as pd  # used for pd.Timestamp in predict() signature

from ..types import (
    AlphaModelName,
    AlphaScore,
    clamp01,
    cross_sectional_zscore,
    winsorize,
)
from ...regime.stock_types import FeatureBundle  # re-exported canonical type


class MeanReversionAlpha:
    """
    Cross-sectional mean-reversion alpha model.

    Conforms to the :class:`~engine.strategy.alpha.types.AlphaModel`
    protocol via structural subtyping (no explicit inheritance).

    Input features:

    * ``dev20`` — deviation of price from its 20-day mean, expressed in
      volatility units (z-score-like).  Positive = above mean; negative
      = below mean.
    * ``rv20_z`` — z-score of 20-day realised volatility vs its own
      trailing history.  Used to discount confidence when vol is elevated.

    Examples
    --------
    >>> model = MeanReversionAlpha()
    >>> scores = model.predict(
    ...     asof=pd.Timestamp("2024-06-15"),
    ...     features={"dev20": dev20_df, "rv20_z": rv20z_df},
    ...     universe=["AAPL", "MSFT", "TSLA"],
    ... )
    >>> scores["AAPL"].score   # winsorised z-score, may be < 0
    0.95
    """

    # -- AlphaModel protocol: name property ----------------------------------

    @property
    def name(self) -> str:
        """Return the canonical model name."""
        return AlphaModelName.MEANREV.value

    # -- AlphaModel protocol: predict ----------------------------------------

    def predict(
        self,
        asof: pd.Timestamp,
        features: FeatureBundle,
        universe: List[str],
    ) -> Dict[str, AlphaScore]:
        """
        Generate per-symbol mean-reversion alpha scores.

        Parameters
        ----------
        asof : pd.Timestamp
            Evaluation date.
        features : FeatureBundle
            Must contain ``"dev20"`` and ``"rv20_z"`` feature columns.
        universe : list[str]
            Symbols to score.

        Returns
        -------
        dict[str, AlphaScore]
            One entry per symbol in *universe*.  Symbols with missing
            ``dev20`` receive ``score=0, raw=0, confidence=0`` with
            ``reasons["missing_data"] = 1``.
        """
        # ── Step 1: extract per-symbol features ───────────────────────
        dev20_vals: Dict[str, float] = {}
        rv20z_vals: Dict[str, float] = {}
        missing: set[str] = set()

        for sym in universe:
            dev = features.get(sym, "dev20")
            if math.isnan(dev):
                missing.add(sym)
                continue

            dev20_vals[sym] = dev

            # rv20_z is optional; default to 0 (neutral) when absent
            rz = features.get(sym, "rv20_z")
            rv20z_vals[sym] = rz if not math.isnan(rz) else 0.0

        # ── Step 2: raw_signal = -dev20 ───────────────────────────────
        raw_signal: Dict[str, float] = {
            sym: -dev20_vals[sym] for sym in dev20_vals
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
                        "dev20": 0.0,
                        "rv20_z": 0.0,
                        "z": 0.0,
                        "missing_data": 1.0,
                    },
                )
                continue

            z = z_scores[sym]
            dev = dev20_vals[sym]
            rz = rv20z_vals[sym]

            # Confidence: signal strength × vol discount
            strength = clamp01(abs(dev) / 2.0)
            vol_discount = clamp01(1.0 - max(0.0, rz) / 2.0)
            conf = strength * vol_discount

            result[sym] = AlphaScore(
                symbol=sym,
                score=z,
                raw=z,
                confidence=conf,
                reasons={"dev20": dev, "rv20_z": rz, "z": z},
            )

        return result
