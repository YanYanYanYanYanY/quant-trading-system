"""
Long-only alpha router with market-regime and stock-regime gating.

Takes signed per-model ``AlphaScore`` dicts and produces a single
**non-negative** ``AlphaScore`` per symbol, applying:

1. **Market gate** — ``CRISIS`` flattens everything.
2. **Stock-regime gate** — ``NO_TRADE``, ``STRESSED_HIGH_VOL``, and
   ``TRENDING_DOWN`` are excluded for a long-only book.
3. **Model routing** — which alpha models are allowed (and at what
   weight) depends on the stock's regime.
4. **Long-only clamp** — ``keep_positive`` zeroes negative model scores.
5. **Confidence gating** — each component is scaled by its model
   confidence *and* the stock-regime confidence.
6. **Combination** — contributions are summed; the final confidence is
   the weighted-average of component confidences.

The router is deterministic, stateless, and fully configurable via
:class:`LongOnlyRouterConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .types import (
    AlphaModelName,
    AlphaScore,
    clamp01,
    keep_positive,
)
from ..regime.market_types import MarketRegimeLabel, MarketRegimeState
from ..regime.stock_types import (
    StockRegimeLabel,
    StockRegimeMap,
    StockRegimeState,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LongOnlyRouterConfig:
    """
    Tuneable knobs for the long-only alpha router.

    Attributes
    ----------
    momentum_in_range_weight : float
        Multiplier applied to the MOMENTUM model when the stock is in
        ``RANGE_LOW_VOL``.  Momentum signals are less reliable in
        range-bound conditions, so this is typically < 1.
    exclude_stressed : bool
        If ``True`` (default), stocks classified as
        ``STRESSED_HIGH_VOL`` receive score 0.  Set to ``False`` only
        if a downstream risk layer handles stressed names explicitly.
    stress_market_model_weights : dict[str, float]
        Per-model weight multipliers applied when the *market* regime
        is ``STRESS`` (but not ``CRISIS``).  Allows the router to
        dampen alpha during elevated macro vol without going fully
        flat.  Models not listed keep weight 1.0.
    """

    momentum_in_range_weight: float = 0.3
    exclude_stressed: bool = False
    stress_market_model_weights: Dict[str, float] = field(
        default_factory=lambda: {
            AlphaModelName.MOMENTUM.value: 0.5,
            AlphaModelName.MEANREV.value: 0.7,
            AlphaModelName.LOWVOL.value: 1.0,
        }
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.momentum_in_range_weight <= 1.0:
            raise ValueError(
                "momentum_in_range_weight must be in [0, 1], "
                f"got {self.momentum_in_range_weight}"
            )


# ---------------------------------------------------------------------------
# Routing tables
# ---------------------------------------------------------------------------

# Per stock-regime, which models are allowed and at what base weight.
# Weight 0 means the model is fully disallowed in that regime.
# A weight < 1 means the model is allowed but down-weighted.
# Missing models default to weight 1.0.

_REGIME_MODEL_WEIGHTS: Dict[StockRegimeLabel, Dict[str, float]] = {
    StockRegimeLabel.TRENDING_UP: {
        AlphaModelName.MOMENTUM.value: 1.0,
        AlphaModelName.MEANREV.value: 0.0,   # disallow mean-rev in trends
        AlphaModelName.LOWVOL.value: 1.0,
    },
    StockRegimeLabel.RANGE_LOW_VOL: {
        AlphaModelName.MOMENTUM.value: 0.3,   # placeholder; overridden by config
        AlphaModelName.MEANREV.value: 1.0,
        AlphaModelName.LOWVOL.value: 1.0,
    },
    StockRegimeLabel.STRESSED_HIGH_VOL: {
        AlphaModelName.MOMENTUM.value: 0.0,   # no momentum-chasing in stress
        AlphaModelName.MEANREV.value: 0.3,    # cautious mean-reversion
        AlphaModelName.LOWVOL.value: 0.5,     # prefer low-vol names
    },
    # Regimes that are hard-gated (all weights 0):
    StockRegimeLabel.TRENDING_DOWN: {},
    StockRegimeLabel.NO_TRADE: {},
}


def _get_model_weight(
    stock_label: StockRegimeLabel,
    model_name: str,
    cfg: LongOnlyRouterConfig,
) -> float:
    """
    Look up the base weight for *model_name* in *stock_label* regime.

    Applies the configurable ``momentum_in_range_weight`` override.
    """
    weights = _REGIME_MODEL_WEIGHTS.get(stock_label)
    if weights is None or len(weights) == 0:
        return 0.0  # hard-gated regime

    base = weights.get(model_name, 1.0)

    # Apply configurable momentum-in-range weight
    if (
        stock_label == StockRegimeLabel.RANGE_LOW_VOL
        and model_name == AlphaModelName.MOMENTUM.value
    ):
        base = cfg.momentum_in_range_weight

    return base


# ---------------------------------------------------------------------------
# Hard-gate sets
# ---------------------------------------------------------------------------

_STOCK_GATE_LABELS: frozenset[StockRegimeLabel] = frozenset({
    StockRegimeLabel.NO_TRADE,
    StockRegimeLabel.TRENDING_DOWN,
})

_STOCK_GATE_LABELS_WITH_STRESS: frozenset[StockRegimeLabel] = (
    _STOCK_GATE_LABELS | {StockRegimeLabel.STRESSED_HIGH_VOL}
)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class LongOnlyAlphaRouter:
    """
    Stateless long-only alpha router.

    Combines per-model signed ``AlphaScore`` dicts into a single
    non-negative ``AlphaScore`` per symbol, respecting market and
    stock regime gates.

    Parameters
    ----------
    config : LongOnlyRouterConfig, optional
        Routing parameters.  Uses defaults when omitted.

    Examples
    --------
    >>> router = LongOnlyAlphaRouter()
    >>> final = router.route(
    ...     market_regime=market_state,
    ...     stock_regimes=stock_map,
    ...     model_outputs={"momentum": mom_scores, "meanrev": mr_scores},
    ... )
    >>> final["AAPL"].score  # guaranteed >= 0
    1.42
    """

    def __init__(self, config: LongOnlyRouterConfig | None = None) -> None:
        self.config = config or LongOnlyRouterConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(
        self,
        market_regime: MarketRegimeState,
        stock_regimes: StockRegimeMap,
        model_outputs: Dict[str, Dict[str, AlphaScore]],
    ) -> Dict[str, AlphaScore]:
        """
        Route and combine alpha model outputs into final long-only scores.

        Parameters
        ----------
        market_regime : MarketRegimeState
            Current market-level regime.
        stock_regimes : StockRegimeMap
            Per-symbol stock regime states.
        model_outputs : dict[str, dict[str, AlphaScore]]
            ``{model_name: {symbol: AlphaScore, ...}, ...}``
            Each model's signed scores for the universe.

        Returns
        -------
        dict[str, AlphaScore]
            One entry per symbol found in *stock_regimes*.  Every
            ``score`` is ``>= 0``.
        """
        cfg = self.config
        mkt_label = market_regime.label

        # ── Market-level hard gate ────────────────────────────────────
        if mkt_label == MarketRegimeLabel.CRISIS:
            return self._flat_scores(
                stock_regimes.symbols,
                gate_reason="market_crisis",
            )

        # Pre-compute market-stress weight multipliers (identity when calm)
        mkt_is_stress = mkt_label == MarketRegimeLabel.STRESS
        mkt_model_weights: Dict[str, float] = {}
        if mkt_is_stress:
            mkt_model_weights = cfg.stress_market_model_weights

        # ── Determine stock-level gate set ────────────────────────────
        gate_labels = (
            _STOCK_GATE_LABELS_WITH_STRESS
            if cfg.exclude_stressed
            else _STOCK_GATE_LABELS
        )

        # ── Per-symbol routing ────────────────────────────────────────
        result: Dict[str, AlphaScore] = {}
        model_names = sorted(model_outputs.keys())

        for sym in stock_regimes.symbols:
            stock_state: StockRegimeState = stock_regimes[sym]
            stock_label = stock_state.label
            stock_conf = stock_state.confidence

            # Stock-level hard gate
            if stock_label in gate_labels:
                result[sym] = self._gated_score(
                    sym,
                    gate_reason=f"stock_{stock_label.value}",
                    market_label=mkt_label.value,
                    stock_label=stock_label.value,
                )
                continue

            # Accumulate weighted contributions across models
            combined_signed = 0.0
            combined_positive = 0.0
            weighted_conf_num = 0.0
            weighted_conf_den = 0.0
            component_reasons: Dict[str, float] = {}
            allowed_flags: Dict[str, float] = {}

            for mname in model_names:
                model_scores = model_outputs[mname]
                if sym not in model_scores:
                    continue

                alpha: AlphaScore = model_scores[sym]

                # Base weight from stock-regime routing table
                base_w = _get_model_weight(stock_label, mname, cfg)

                # Market-stress weight adjustment
                mkt_w = mkt_model_weights.get(mname, 1.0) if mkt_is_stress else 1.0

                effective_w = base_w * mkt_w
                allowed_flags[f"w_{mname}"] = effective_w

                if effective_w <= 0.0:
                    component_reasons[f"{mname}_contrib"] = 0.0
                    continue

                # Long-only clamp per model
                pos_score = keep_positive(alpha.score)

                # Confidence gating: model confidence × stock regime confidence
                gated_conf = alpha.confidence * stock_conf

                # Weighted contribution
                contrib = pos_score * effective_w * gated_conf
                combined_positive += contrib
                combined_signed += alpha.score * effective_w * gated_conf

                component_reasons[f"{mname}_contrib"] = contrib
                component_reasons[f"{mname}_raw"] = alpha.score
                component_reasons[f"{mname}_conf"] = alpha.confidence

                # For weighted-average confidence
                weighted_conf_num += gated_conf * effective_w
                weighted_conf_den += effective_w

            # Final confidence: weighted average of gated confidences
            final_conf = (
                clamp01(weighted_conf_num / weighted_conf_den)
                if weighted_conf_den > 0
                else 0.0
            )

            result[sym] = AlphaScore(
                symbol=sym,
                score=combined_positive,
                raw=combined_signed,
                confidence=final_conf,
                reasons={
                    "market_label": _label_to_float(mkt_label),
                    "stock_label": _label_to_float(stock_label),
                    "stock_conf": stock_conf,
                    **allowed_flags,
                    **component_reasons,
                },
            )

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flat_scores(
        symbols: List[str],
        gate_reason: str,
    ) -> Dict[str, AlphaScore]:
        """Return zero-score entries for every symbol (market gate)."""
        return {
            sym: AlphaScore(
                symbol=sym,
                score=0.0,
                raw=0.0,
                confidence=0.0,
                reasons={gate_reason: 1.0},
            )
            for sym in symbols
        }

    @staticmethod
    def _gated_score(
        sym: str,
        gate_reason: str,
        market_label: str,
        stock_label: str,
    ) -> AlphaScore:
        """Return a zero-score entry for a stock-level gated symbol."""
        return AlphaScore(
            symbol=sym,
            score=0.0,
            raw=0.0,
            confidence=0.0,
            reasons={
                "gate": 1.0,
                gate_reason: 1.0,
                "market_label": _MARKET_LABEL_MAP.get(market_label, -1.0),
                "stock_label": _STOCK_LABEL_MAP.get(stock_label, -1.0),
            },
        )


# ---------------------------------------------------------------------------
# Label → float mapping (reasons dict is Dict[str, float])
# ---------------------------------------------------------------------------

_MARKET_LABEL_MAP: Dict[str, float] = {
    MarketRegimeLabel.CALM_TREND.value: 0.0,
    MarketRegimeLabel.CALM_RANGE.value: 1.0,
    MarketRegimeLabel.STRESS.value: 2.0,
    MarketRegimeLabel.CRISIS.value: 3.0,
}

_STOCK_LABEL_MAP: Dict[str, float] = {
    StockRegimeLabel.TRENDING_UP.value: 0.0,
    StockRegimeLabel.TRENDING_DOWN.value: 1.0,
    StockRegimeLabel.RANGE_LOW_VOL.value: 2.0,
    StockRegimeLabel.STRESSED_HIGH_VOL.value: 3.0,
    StockRegimeLabel.NO_TRADE.value: 4.0,
}


def _label_to_float(
    label: MarketRegimeLabel | StockRegimeLabel,
) -> float:
    """Convert an enum label to a numeric code for the reasons dict."""
    val = label.value
    if val in _MARKET_LABEL_MAP:
        return _MARKET_LABEL_MAP[val]
    if val in _STOCK_LABEL_MAP:
        return _STOCK_LABEL_MAP[val]
    return -1.0
