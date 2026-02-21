"""
Long-only momentum alpha model.

Scores each symbol by the cross-sectional z-score of its 63-day
momentum (``mom63``), winsorised to [-3, 3].  Negative z-scores are
**not** clipped here — the long-only router downstream is responsible
for that.

Pipeline
--------
1. Extract ``mom63`` at *asof* for every symbol in *universe*.
2. Cross-sectional z-score across the universe (NaN-safe).
3. Winsorise z-scores to [-3, 3].
4. ``raw = z``, ``score = z`` (may be negative).
5. ``confidence = clamp01(|z| / 3)``.
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


class MomentumAlpha:
    """
    Cross-sectional momentum alpha model.

    Conforms to the :class:`~engine.strategy.alpha.types.AlphaModel`
    protocol via structural subtyping (no explicit inheritance).

    The single input feature is ``mom63`` (63-day total return).  After
    cross-sectional z-scoring and winsorising the output is a signed
    z-score per symbol.  The downstream long-only router will clip
    ``score`` to ``>= 0``; this model intentionally preserves negative
    values so that the ensemble / routing layer has full information.

    Examples
    --------
    >>> model = MomentumAlpha()
    >>> scores = model.predict(
    ...     asof=pd.Timestamp("2024-06-15"),
    ...     features={"mom63": mom63_df},
    ...     universe=["AAPL", "MSFT", "TSLA"],
    ... )
    >>> scores["AAPL"].score   # winsorised z-score, may be < 0
    1.12
    """

    # -- AlphaModel protocol: name property ----------------------------------

    @property
    def name(self) -> str:
        """Return the canonical model name."""
        return AlphaModelName.MOMENTUM.value

    # -- AlphaModel protocol: predict ----------------------------------------

    def predict(
        self,
        asof: pd.Timestamp,
        features: FeatureBundle,
        universe: List[str],
    ) -> Dict[str, AlphaScore]:
        """
        Generate per-symbol momentum alpha scores.

        Parameters
        ----------
        asof : pd.Timestamp
            Evaluation date.
        features : FeatureBundle
            Must contain the ``"mom63"`` feature column.
        universe : list[str]
            Symbols to score.

        Returns
        -------
        dict[str, AlphaScore]
            One entry per symbol in *universe*.  Symbols with missing
            ``mom63`` data receive ``score=0, raw=0, confidence=0``
            with ``reasons["missing_data"] = 1``.
        """
        # ── Step 1: extract raw mom63 per symbol ───────────────────────
        raw_signal: Dict[str, float] = {}
        missing: set[str] = set()

        for sym in universe:
            val = features.get(sym, "mom63")
            if math.isnan(val):
                missing.add(sym)
            else:
                raw_signal[sym] = val

        # ── Step 2: cross-sectional z-score (NaN-safe) ────────────────
        z_scores = cross_sectional_zscore(raw_signal)

        # ── Step 3: winsorise to [-3, 3] ──────────────────────────────
        z_scores = {s: winsorize(z) for s, z in z_scores.items()}

        # ── Step 4 & 5: build AlphaScore per symbol ───────────────────
        result: Dict[str, AlphaScore] = {}

        for sym in universe:
            if sym in missing:
                result[sym] = AlphaScore(
                    symbol=sym,
                    score=0.0,
                    raw=0.0,
                    confidence=0.0,
                    reasons={"mom63": 0.0, "z": 0.0, "missing_data": 1.0},
                )
                continue

            z = z_scores[sym]
            mom_val = raw_signal[sym]
            conf = clamp01(abs(z) / 3.0)

            result[sym] = AlphaScore(
                symbol=sym,
                score=z,
                raw=z,
                confidence=conf,
                reasons={"mom63": mom_val, "z": z},
            )

        return result
