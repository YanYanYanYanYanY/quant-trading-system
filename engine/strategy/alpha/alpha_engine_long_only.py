"""
Long-only alpha ensemble engine.

Orchestrates the full alpha pipeline:

1. **Run models** — call every registered alpha sub-model to get
   signed per-symbol scores.
2. **Apply ensemble weights** — scale each model's scores by
   market-regime-dependent blend weights *before* routing.
3. **Route** — the :class:`LongOnlyAlphaRouter` applies stock-regime
   gates, ``keep_positive`` (long-only clamp), and confidence gating.
4. **Rank-normalise** (optional) — map final combined scores to
   ``[0, 1]`` percentile ranks for position selection.

The output is an :class:`AlphaVector` with one ``AlphaScore`` per
symbol, all ``score >= 0``.

Ensemble weight schedule (defaults)
------------------------------------
==========  ====  =======  ======
Regime      MOM   MEANREV  LOWVOL
==========  ====  =======  ======
CALM_TREND  0.80  0.05     0.15
CALM_RANGE  0.20  0.70     0.10
STRESS      0.20  0.10     0.70
CRISIS      0.00  0.00     0.00
==========  ====  =======  ======
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from .types import (
    AlphaModelName,
    AlphaScore,
    AlphaVector,
    clamp01,
    rank_to_unit,
)
from .models import LowVolAlpha, MeanReversionAlpha, MomentumAlpha
from .router_long_only import LongOnlyAlphaRouter, LongOnlyRouterConfig
from ..regime.market_types import MarketRegimeLabel, MarketRegimeState
from ..regime.stock_types import FeatureBundle, StockRegimeMap


# ---------------------------------------------------------------------------
# Default ensemble weight schedule
# ---------------------------------------------------------------------------

_MOM = AlphaModelName.MOMENTUM.value
_MR = AlphaModelName.MEANREV.value
_LV = AlphaModelName.LOWVOL.value

_DEFAULT_ENSEMBLE_WEIGHTS: Dict[MarketRegimeLabel, Dict[str, float]] = {
    MarketRegimeLabel.CALM_TREND: {_MOM: 0.80, _MR: 0.05, _LV: 0.15},
    MarketRegimeLabel.CALM_RANGE: {_MOM: 0.20, _MR: 0.70, _LV: 0.10},
    MarketRegimeLabel.STRESS:     {_MOM: 0.20, _MR: 0.10, _LV: 0.70},
    MarketRegimeLabel.CRISIS:     {_MOM: 0.00, _MR: 0.00, _LV: 0.00},
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AlphaEngineConfig:
    """
    Configuration for :class:`LongOnlyAlphaEngine`.

    Attributes
    ----------
    ensemble_weights : dict[MarketRegimeLabel, dict[str, float]]
        Market-regime → {model_name → blend weight}.
        Weights are applied to each model's raw score **before** the
        long-only router processes them.  They need not sum to 1 —
        the router and confidence gating normalise downstream.
    router_config : LongOnlyRouterConfig
        Configuration for the long-only router (stock-regime gates,
        momentum-in-range weight, market-stress dampening).
    normalize_to_ranks : bool
        If ``True`` (default), the final scores are re-mapped to
        ``[0, 1]`` percentile ranks after combination.  The raw
        combined score is preserved in ``reasons["combined_raw"]``.
    """

    ensemble_weights: Dict[MarketRegimeLabel, Dict[str, float]] = field(
        default_factory=lambda: dict(_DEFAULT_ENSEMBLE_WEIGHTS),
    )
    router_config: LongOnlyRouterConfig = field(
        default_factory=LongOnlyRouterConfig,
    )
    normalize_to_ranks: bool = True


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class LongOnlyAlphaEngine:
    """
    End-to-end long-only alpha ensemble engine.

    Instantiates the three alpha sub-models and the router, and
    exposes a single :meth:`generate` entry point that returns a
    ready-to-use :class:`AlphaVector`.

    Parameters
    ----------
    config : AlphaEngineConfig, optional
        Full configuration.  Defaults are sensible for US equities.

    Examples
    --------
    >>> engine = LongOnlyAlphaEngine()
    >>> alpha_vec = engine.generate(
    ...     asof=pd.Timestamp("2024-06-15"),
    ...     market_regime=market_state,
    ...     stock_regimes=stock_map,
    ...     features=features,
    ...     universe=["AAPL", "MSFT", "TSLA"],
    ... )
    >>> alpha_vec["AAPL"].score  # non-negative, rank-normalised
    0.85
    """

    def __init__(self, config: AlphaEngineConfig | None = None) -> None:
        self.config = config or AlphaEngineConfig()

        # Instantiate sub-models
        self._models = [
            MomentumAlpha(),
            MeanReversionAlpha(),
            LowVolAlpha(),
        ]

        # Instantiate router
        self._router = LongOnlyAlphaRouter(self.config.router_config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        asof: pd.Timestamp,
        market_regime: MarketRegimeState,
        stock_regimes: StockRegimeMap,
        features: FeatureBundle,
        universe: List[str],
    ) -> AlphaVector:
        """
        Run the full alpha pipeline and return an ``AlphaVector``.

        Parameters
        ----------
        asof : pd.Timestamp
            Evaluation date.
        market_regime : MarketRegimeState
            Current market-level regime.
        stock_regimes : StockRegimeMap
            Per-symbol stock-level regimes.
        features : FeatureBundle
            Pre-computed feature DataFrames.
        universe : list[str]
            Symbols to score.

        Returns
        -------
        AlphaVector
            Non-negative scores for every symbol in *universe* (or in
            *stock_regimes*, whichever is smaller).
        """
        model_outputs = self.run_models(asof, features, universe)
        final_scores = self.combine(
            asof, market_regime, stock_regimes, model_outputs,
        )
        return AlphaVector(asof=asof, scores=final_scores)

    def run_models(
        self,
        asof: pd.Timestamp,
        features: FeatureBundle,
        universe: List[str],
    ) -> Dict[str, Dict[str, AlphaScore]]:
        """
        Call every registered alpha model.

        Parameters
        ----------
        asof : pd.Timestamp
            Evaluation date.
        features : FeatureBundle
            Pre-computed feature DataFrames.
        universe : list[str]
            Symbols to score.

        Returns
        -------
        dict[str, dict[str, AlphaScore]]
            ``{model_name: {symbol: AlphaScore, ...}, ...}``
        """
        result: Dict[str, Dict[str, AlphaScore]] = {}
        for model in self._models:
            result[model.name] = model.predict(asof, features, universe)
        return result

    def combine(
        self,
        asof: pd.Timestamp,
        market_regime: MarketRegimeState,
        stock_regimes: StockRegimeMap,
        model_outputs: Dict[str, Dict[str, AlphaScore]],
    ) -> Dict[str, AlphaScore]:
        """
        Apply ensemble weights, route, and optionally rank-normalise.

        Parameters
        ----------
        asof : pd.Timestamp
            Evaluation date.
        market_regime : MarketRegimeState
            Current market-level regime.
        stock_regimes : StockRegimeMap
            Per-symbol stock-level regimes.
        model_outputs : dict[str, dict[str, AlphaScore]]
            Raw model outputs from :meth:`run_models`.

        Returns
        -------
        dict[str, AlphaScore]
            Final combined, non-negative scores.
        """
        mkt_label = market_regime.label
        ew = self.config.ensemble_weights.get(mkt_label, {})

        # ── Step 1: scale model scores by ensemble weights ────────────
        weighted_outputs = _apply_ensemble_weights(model_outputs, ew)

        # ── Step 2: route (stock-regime gates + keep_positive + conf) ─
        routed = self._router.route(
            market_regime, stock_regimes, weighted_outputs,
        )

        # ── Step 3: inject ensemble weight info into reasons ──────────
        for sym, alpha in routed.items():
            alpha.reasons["ensemble_weights"] = sum(ew.values())
            for mname, w in ew.items():
                alpha.reasons[f"ew_{mname}"] = w

        # ── Step 4: optional rank normalisation ───────────────────────
        if self.config.normalize_to_ranks:
            routed = _rank_normalize(routed)

        return routed


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_ensemble_weights(
    model_outputs: Dict[str, Dict[str, AlphaScore]],
    weights: Dict[str, float],
) -> Dict[str, Dict[str, AlphaScore]]:
    """
    Scale each model's scores by its ensemble weight.

    Creates *new* ``AlphaScore`` objects (does not mutate originals).
    Models not listed in *weights* receive weight 0.

    The weight is applied to ``score`` (and ``raw``) **before** the
    router's ``keep_positive`` — so a 0-weighted model genuinely
    contributes nothing, even if its original score was positive.
    """
    result: Dict[str, Dict[str, AlphaScore]] = {}

    for mname, sym_scores in model_outputs.items():
        w = weights.get(mname, 0.0)
        weighted: Dict[str, AlphaScore] = {}

        for sym, alpha in sym_scores.items():
            weighted[sym] = AlphaScore(
                symbol=sym,
                score=alpha.score * w,
                raw=alpha.raw * w,
                confidence=alpha.confidence,
                reasons={
                    **alpha.reasons,
                    "ensemble_w": w,
                },
            )

        result[mname] = weighted

    return result


def _rank_normalize(
    scores: Dict[str, AlphaScore],
) -> Dict[str, AlphaScore]:
    """
    Re-map combined scores to ``[0, 1]`` percentile ranks.

    Preserves the original combined score in ``reasons["combined_raw"]``
    and overwrites ``score`` with the rank.  Symbols with ``score == 0``
    (gated) are included in the ranking — they naturally land at the
    bottom.
    """
    # Extract score values for ranking
    score_vals: Dict[str, float] = {
        sym: alpha.score for sym, alpha in scores.items()
    }

    ranks = rank_to_unit(score_vals)

    result: Dict[str, AlphaScore] = {}
    for sym, alpha in scores.items():
        rank_score = ranks.get(sym, 0.0)
        result[sym] = AlphaScore(
            symbol=sym,
            score=rank_score,
            raw=alpha.raw,
            confidence=alpha.confidence,
            reasons={
                **alpha.reasons,
                "combined_score": alpha.score,
                "rank_score": rank_score,
            },
        )

    return result
