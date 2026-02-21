"""
Unit tests for the long-only alpha router.

Tests validate routing behaviour without needing full FeatureBundle or
real alpha models.  We construct minimal fake ``AlphaScore`` dicts and
feed them directly to ``LongOnlyAlphaRouter.route()``.

Test cases
----------
1) CRISIS market regime  -> all final scores = 0.
2) stock_regime NO_TRADE -> score 0.
3) stock_regime STRESSED_HIGH_VOL (EVENT_VOL) -> score 0.
4) TRENDING_UP stock: meanrev component ignored, momentum allowed.
5) RANGE_LOW_VOL stock: meanrev allowed, momentum down-weighted.
6) Long-only clipping: negative model scores contribute 0.
7) Confidence gating: lower stock confidence reduces final score.
"""

from __future__ import annotations

import pandas as pd
import pytest

from engine.strategy.alpha.router_long_only import (
    LongOnlyAlphaRouter,
    LongOnlyRouterConfig,
)
from engine.strategy.alpha.types import AlphaModelName, AlphaScore
from engine.strategy.regime.market_types import MarketRegimeLabel, MarketRegimeState
from engine.strategy.regime.stock_types import (
    StockRegimeLabel,
    StockRegimeMap,
    StockRegimeState,
)


# ============================================================================
# Helpers
# ============================================================================

UNIVERSE = ["AAPL", "MSFT", "TSLA"]
ASOF = pd.Timestamp("2025-06-15")

_MOM = AlphaModelName.MOMENTUM.value
_MR = AlphaModelName.MEANREV.value
_LV = AlphaModelName.LOWVOL.value


def _market(label: MarketRegimeLabel, confidence: float = 0.80) -> MarketRegimeState:
    """Build a minimal ``MarketRegimeState``."""
    return MarketRegimeState(asof=ASOF, label=label, confidence=confidence)


def _stock_map(
    label_map: dict[str, StockRegimeLabel],
    confidence_map: dict[str, float] | None = None,
) -> StockRegimeMap:
    """Build a ``StockRegimeMap`` from {symbol: label} and optional confidences."""
    conf = confidence_map or {}
    states = {
        sym: StockRegimeState(
            symbol=sym,
            asof=ASOF,
            label=lbl,
            confidence=conf.get(sym, 0.80),
        )
        for sym, lbl in label_map.items()
    }
    return StockRegimeMap(asof=ASOF, states=states)


def _alpha(symbol: str, score: float, confidence: float = 0.90) -> AlphaScore:
    """Build a minimal ``AlphaScore``."""
    return AlphaScore(
        symbol=symbol,
        score=score,
        raw=score,
        confidence=confidence,
    )


def _uniform_model_outputs(
    scores: dict[str, float],
    confidence: float = 0.90,
) -> dict[str, dict[str, AlphaScore]]:
    """
    Create model_outputs where every model returns the *same* score per symbol.

    Useful when the test does not care about per-model variation.
    """
    return {
        model: {sym: _alpha(sym, sc, confidence) for sym, sc in scores.items()}
        for model in (_MOM, _MR, _LV)
    }


def _per_model_outputs(
    model_scores: dict[str, dict[str, float]],
    confidence: float = 0.90,
) -> dict[str, dict[str, AlphaScore]]:
    """
    Create model_outputs with different scores per model.

    Parameters
    ----------
    model_scores : dict[str, dict[str, float]]
        ``{model_name: {symbol: score, ...}, ...}``
    """
    return {
        model: {sym: _alpha(sym, sc, confidence) for sym, sc in sym_scores.items()}
        for model, sym_scores in model_scores.items()
    }


# ============================================================================
# 1) CRISIS market regime -> all final scores = 0
# ============================================================================


class TestCrisisMarketGate:
    """When market regime is CRISIS the router must flatten everything."""

    def test_all_scores_zero(self):
        """Every symbol should receive score == 0 under CRISIS."""
        router = LongOnlyAlphaRouter()
        market = _market(MarketRegimeLabel.CRISIS)
        stock_regimes = _stock_map({s: StockRegimeLabel.TRENDING_UP for s in UNIVERSE})
        model_outputs = _uniform_model_outputs(
            {s: 1.5 for s in UNIVERSE},
        )

        result = router.route(market, stock_regimes, model_outputs)

        for sym in UNIVERSE:
            assert result[sym].score == 0.0, f"{sym} should be 0 in CRISIS"

    def test_all_confidences_zero(self):
        """Confidence should also be 0 under CRISIS."""
        router = LongOnlyAlphaRouter()
        market = _market(MarketRegimeLabel.CRISIS)
        stock_regimes = _stock_map({s: StockRegimeLabel.RANGE_LOW_VOL for s in UNIVERSE})
        model_outputs = _uniform_model_outputs({s: 2.0 for s in UNIVERSE})

        result = router.route(market, stock_regimes, model_outputs)

        for sym in UNIVERSE:
            assert result[sym].confidence == 0.0

    def test_crisis_gate_reason(self):
        """Gated entries should carry a 'market_crisis' reason tag."""
        router = LongOnlyAlphaRouter()
        market = _market(MarketRegimeLabel.CRISIS)
        stock_regimes = _stock_map({s: StockRegimeLabel.TRENDING_UP for s in UNIVERSE})
        model_outputs = _uniform_model_outputs({s: 1.0 for s in UNIVERSE})

        result = router.route(market, stock_regimes, model_outputs)

        for sym in UNIVERSE:
            assert result[sym].reasons.get("market_crisis") == 1.0


# ============================================================================
# 2) stock_regime NO_TRADE -> score 0
# ============================================================================


class TestNoTradeStockGate:
    """Stocks with NO_TRADE stock regime should be gated to 0."""

    def test_no_trade_score_zero(self):
        router = LongOnlyAlphaRouter()
        market = _market(MarketRegimeLabel.CALM_TREND)
        stock_regimes = _stock_map({
            "AAPL": StockRegimeLabel.TRENDING_UP,
            "MSFT": StockRegimeLabel.NO_TRADE,
            "TSLA": StockRegimeLabel.TRENDING_UP,
        })
        model_outputs = _uniform_model_outputs(
            {"AAPL": 1.0, "MSFT": 1.0, "TSLA": 1.0},
        )

        result = router.route(market, stock_regimes, model_outputs)

        assert result["MSFT"].score == 0.0
        # The other two should be positive (they are TRENDING_UP with positive input)
        assert result["AAPL"].score > 0.0
        assert result["TSLA"].score > 0.0

    def test_no_trade_confidence_zero(self):
        router = LongOnlyAlphaRouter()
        market = _market(MarketRegimeLabel.CALM_TREND)
        stock_regimes = _stock_map({"AAPL": StockRegimeLabel.NO_TRADE})
        model_outputs = _uniform_model_outputs({"AAPL": 2.0})

        result = router.route(market, stock_regimes, model_outputs)

        assert result["AAPL"].confidence == 0.0


# ============================================================================
# 3) stock_regime STRESSED_HIGH_VOL (EVENT_VOL) -> score 0
# ============================================================================


class TestStressedStockGate:
    """Stocks with STRESSED_HIGH_VOL should be gated when exclude_stressed=True."""

    def test_stressed_score_zero_default(self):
        """Default config excludes stressed stocks."""
        router = LongOnlyAlphaRouter()  # exclude_stressed=True by default
        market = _market(MarketRegimeLabel.CALM_TREND)
        stock_regimes = _stock_map({
            "AAPL": StockRegimeLabel.STRESSED_HIGH_VOL,
            "MSFT": StockRegimeLabel.TRENDING_UP,
            "TSLA": StockRegimeLabel.STRESSED_HIGH_VOL,
        })
        model_outputs = _uniform_model_outputs(
            {"AAPL": 1.5, "MSFT": 1.5, "TSLA": 1.5},
        )

        result = router.route(market, stock_regimes, model_outputs)

        assert result["AAPL"].score == 0.0
        assert result["TSLA"].score == 0.0
        assert result["MSFT"].score > 0.0

    def test_stressed_allowed_when_configured(self):
        """With exclude_stressed=False, stressed stocks are routed normally."""
        cfg = LongOnlyRouterConfig(exclude_stressed=False)
        router = LongOnlyAlphaRouter(cfg)
        market = _market(MarketRegimeLabel.CALM_TREND)
        stock_regimes = _stock_map(
            {"AAPL": StockRegimeLabel.STRESSED_HIGH_VOL},
        )
        model_outputs = _uniform_model_outputs({"AAPL": 1.0})

        result = router.route(market, stock_regimes, model_outputs)

        # STRESSED_HIGH_VOL has empty weights in the routing table -> still gated
        # because _REGIME_MODEL_WEIGHTS maps it to {} (all 0).
        assert result["AAPL"].score == 0.0

    def test_trending_down_also_gated(self):
        """TRENDING_DOWN is likewise gated for a long-only book."""
        router = LongOnlyAlphaRouter()
        market = _market(MarketRegimeLabel.CALM_TREND)
        stock_regimes = _stock_map({"AAPL": StockRegimeLabel.TRENDING_DOWN})
        model_outputs = _uniform_model_outputs({"AAPL": 1.0})

        result = router.route(market, stock_regimes, model_outputs)

        assert result["AAPL"].score == 0.0


# ============================================================================
# 4) TRENDING_UP: meanrev ignored, momentum allowed
# ============================================================================


class TestTrendingRouting:
    """In TRENDING_UP regime, mean-reversion is zeroed and momentum passes."""

    def test_meanrev_zero_contribution(self):
        """Mean-reversion model should contribute 0 for a TRENDING_UP stock."""
        router = LongOnlyAlphaRouter()
        market = _market(MarketRegimeLabel.CALM_TREND)
        stock_regimes = _stock_map({"AAPL": StockRegimeLabel.TRENDING_UP})

        model_outputs = _per_model_outputs({
            _MOM: {"AAPL": 1.0},
            _MR:  {"AAPL": 2.0},   # large score — should be ignored
            _LV:  {"AAPL": 0.5},
        })

        result = router.route(market, stock_regimes, model_outputs)

        reasons = result["AAPL"].reasons
        assert reasons.get(f"{_MR}_contrib", 0.0) == 0.0, (
            "Meanrev should contribute 0 in TRENDING_UP"
        )

    def test_momentum_positive_contribution(self):
        """Momentum model should contribute positively for TRENDING_UP."""
        router = LongOnlyAlphaRouter()
        market = _market(MarketRegimeLabel.CALM_TREND)
        stock_regimes = _stock_map({"AAPL": StockRegimeLabel.TRENDING_UP})

        model_outputs = _per_model_outputs({
            _MOM: {"AAPL": 1.0},
            _MR:  {"AAPL": 0.0},
            _LV:  {"AAPL": 0.0},
        })

        result = router.route(market, stock_regimes, model_outputs)

        assert result["AAPL"].score > 0.0, "Momentum should pass through in TRENDING_UP"
        assert result["AAPL"].reasons.get(f"{_MOM}_contrib", 0.0) > 0.0

    def test_meanrev_weight_is_zero(self):
        """The effective weight for meanrev in TRENDING_UP should be 0."""
        router = LongOnlyAlphaRouter()
        market = _market(MarketRegimeLabel.CALM_TREND)
        stock_regimes = _stock_map({"AAPL": StockRegimeLabel.TRENDING_UP})

        model_outputs = _per_model_outputs({
            _MOM: {"AAPL": 1.0},
            _MR:  {"AAPL": 1.0},
            _LV:  {"AAPL": 1.0},
        })

        result = router.route(market, stock_regimes, model_outputs)

        assert result["AAPL"].reasons.get(f"w_{_MR}", -1) == 0.0


# ============================================================================
# 5) RANGE_LOW_VOL: meanrev allowed, momentum down-weighted
# ============================================================================


class TestRangingRouting:
    """In RANGE_LOW_VOL, meanrev is full weight; momentum is down-weighted."""

    def test_meanrev_allowed(self):
        """Mean-reversion should contribute positively for RANGE_LOW_VOL."""
        router = LongOnlyAlphaRouter()
        market = _market(MarketRegimeLabel.CALM_RANGE)
        stock_regimes = _stock_map({"AAPL": StockRegimeLabel.RANGE_LOW_VOL})

        model_outputs = _per_model_outputs({
            _MOM: {"AAPL": 0.0},
            _MR:  {"AAPL": 1.0},
            _LV:  {"AAPL": 0.0},
        })

        result = router.route(market, stock_regimes, model_outputs)

        assert result["AAPL"].score > 0.0, "Meanrev should pass through in RANGE_LOW_VOL"
        assert result["AAPL"].reasons.get(f"{_MR}_contrib", 0.0) > 0.0

    def test_momentum_downweighted(self):
        """Momentum weight in RANGE_LOW_VOL equals config.momentum_in_range_weight."""
        cfg = LongOnlyRouterConfig(momentum_in_range_weight=0.3)
        router = LongOnlyAlphaRouter(cfg)
        market = _market(MarketRegimeLabel.CALM_RANGE)
        stock_regimes = _stock_map({"AAPL": StockRegimeLabel.RANGE_LOW_VOL})

        model_outputs = _per_model_outputs({
            _MOM: {"AAPL": 1.0},
            _MR:  {"AAPL": 1.0},
            _LV:  {"AAPL": 1.0},
        })

        result = router.route(market, stock_regimes, model_outputs)

        w_mom = result["AAPL"].reasons.get(f"w_{_MOM}", -1)
        assert w_mom == pytest.approx(0.3), (
            f"Momentum weight should be 0.3, got {w_mom}"
        )

    def test_momentum_contrib_less_than_meanrev(self):
        """With equal raw scores, momentum contribution < meanrev contribution."""
        cfg = LongOnlyRouterConfig(momentum_in_range_weight=0.3)
        router = LongOnlyAlphaRouter(cfg)
        market = _market(MarketRegimeLabel.CALM_RANGE)
        stock_regimes = _stock_map({"AAPL": StockRegimeLabel.RANGE_LOW_VOL})

        model_outputs = _per_model_outputs({
            _MOM: {"AAPL": 1.0},
            _MR:  {"AAPL": 1.0},
            _LV:  {"AAPL": 0.0},
        })

        result = router.route(market, stock_regimes, model_outputs)

        mom_contrib = result["AAPL"].reasons.get(f"{_MOM}_contrib", 0.0)
        mr_contrib = result["AAPL"].reasons.get(f"{_MR}_contrib", 0.0)
        assert mom_contrib < mr_contrib, (
            f"momentum contrib ({mom_contrib}) should be < meanrev contrib ({mr_contrib})"
        )

    def test_custom_momentum_in_range_weight(self):
        """Different momentum_in_range_weight values should change the output."""
        market = _market(MarketRegimeLabel.CALM_RANGE)
        stock_regimes = _stock_map({"AAPL": StockRegimeLabel.RANGE_LOW_VOL})
        model_outputs = _per_model_outputs({
            _MOM: {"AAPL": 1.0},
            _MR:  {"AAPL": 0.0},
            _LV:  {"AAPL": 0.0},
        })

        r_lo = LongOnlyAlphaRouter(LongOnlyRouterConfig(momentum_in_range_weight=0.1))
        r_hi = LongOnlyAlphaRouter(LongOnlyRouterConfig(momentum_in_range_weight=0.9))

        score_lo = r_lo.route(market, stock_regimes, model_outputs)["AAPL"].score
        score_hi = r_hi.route(market, stock_regimes, model_outputs)["AAPL"].score

        assert score_lo < score_hi, (
            "Higher momentum_in_range_weight should yield higher score"
        )


# ============================================================================
# 6) Long-only clipping: negative model score -> contributes 0
# ============================================================================


class TestLongOnlyClipping:
    """Negative model scores must be clamped to 0 by keep_positive."""

    def test_negative_score_clipped(self):
        """A model returning a negative score contributes 0 to the total."""
        router = LongOnlyAlphaRouter()
        market = _market(MarketRegimeLabel.CALM_TREND)
        stock_regimes = _stock_map({"AAPL": StockRegimeLabel.TRENDING_UP})

        model_outputs = _per_model_outputs({
            _MOM: {"AAPL": -0.5},   # negative
            _MR:  {"AAPL": 0.0},
            _LV:  {"AAPL": 0.0},
        })

        result = router.route(market, stock_regimes, model_outputs)

        assert result["AAPL"].score == 0.0, "Negative-only input should yield 0"

    def test_mixed_positive_negative(self):
        """Only positive models should contribute; negatives are clipped."""
        router = LongOnlyAlphaRouter()
        market = _market(MarketRegimeLabel.CALM_TREND)
        stock_regimes = _stock_map({"AAPL": StockRegimeLabel.TRENDING_UP})

        model_outputs = _per_model_outputs({
            _MOM: {"AAPL": 2.0},    # positive — should contribute
            _MR:  {"AAPL": -1.0},   # negative — clipped (also meanrev blocked in TRENDING)
            _LV:  {"AAPL": -0.3},   # negative — clipped
        })

        result = router.route(market, stock_regimes, model_outputs)

        # Only momentum contributes (positive and allowed)
        assert result["AAPL"].score > 0.0
        # lowvol is allowed in TRENDING_UP but has negative score → clipped
        assert result["AAPL"].reasons.get(f"{_LV}_contrib", -1) == 0.0

    def test_all_negative_yields_zero(self):
        """If all model scores are negative, final score must be 0."""
        router = LongOnlyAlphaRouter()
        market = _market(MarketRegimeLabel.CALM_RANGE)
        stock_regimes = _stock_map({"AAPL": StockRegimeLabel.RANGE_LOW_VOL})

        model_outputs = _per_model_outputs({
            _MOM: {"AAPL": -1.0},
            _MR:  {"AAPL": -0.5},
            _LV:  {"AAPL": -0.2},
        })

        result = router.route(market, stock_regimes, model_outputs)

        assert result["AAPL"].score == 0.0

    def test_raw_preserves_signed_value(self):
        """The ``raw`` field should reflect signed (pre-clamp) accumulation."""
        router = LongOnlyAlphaRouter()
        market = _market(MarketRegimeLabel.CALM_TREND)
        stock_regimes = _stock_map({"AAPL": StockRegimeLabel.TRENDING_UP})

        model_outputs = _per_model_outputs({
            _MOM: {"AAPL": -1.0},
            _MR:  {"AAPL": 0.0},
            _LV:  {"AAPL": 0.5},
        })

        result = router.route(market, stock_regimes, model_outputs)

        # raw should carry the signed combination (momentum is negative, lowvol positive)
        # Both are allowed in TRENDING_UP (meanrev is not)
        # raw = sum of alpha.score * effective_w * gated_conf for each model
        assert result["AAPL"].raw != result["AAPL"].score or result["AAPL"].raw == 0.0


# ============================================================================
# 7) Confidence gating: lower stock confidence -> lower final score
# ============================================================================


class TestConfidenceGating:
    """Stock-level confidence scales every model's contribution."""

    def test_lower_confidence_lower_score(self):
        """Same model outputs, lower stock confidence → lower final score."""
        router = LongOnlyAlphaRouter()
        market = _market(MarketRegimeLabel.CALM_TREND)

        hi_conf = _stock_map(
            {"AAPL": StockRegimeLabel.TRENDING_UP},
            confidence_map={"AAPL": 0.90},
        )
        lo_conf = _stock_map(
            {"AAPL": StockRegimeLabel.TRENDING_UP},
            confidence_map={"AAPL": 0.30},
        )

        model_outputs = _per_model_outputs({
            _MOM: {"AAPL": 1.0},
            _MR:  {"AAPL": 0.0},
            _LV:  {"AAPL": 0.5},
        })

        score_hi = router.route(market, hi_conf, model_outputs)["AAPL"].score
        score_lo = router.route(market, lo_conf, model_outputs)["AAPL"].score

        assert score_lo < score_hi, (
            f"Lower stock confidence should yield lower score "
            f"({score_lo} vs {score_hi})"
        )

    def test_zero_confidence_zeroes_score(self):
        """Stock confidence of 0 should gate everything to 0."""
        router = LongOnlyAlphaRouter()
        market = _market(MarketRegimeLabel.CALM_TREND)
        stock_regimes = _stock_map(
            {"AAPL": StockRegimeLabel.TRENDING_UP},
            confidence_map={"AAPL": 0.0},
        )
        model_outputs = _uniform_model_outputs({"AAPL": 2.0})

        result = router.route(market, stock_regimes, model_outputs)

        assert result["AAPL"].score == 0.0

    def test_model_confidence_also_scales(self):
        """Lower *model* confidence should also reduce the contribution."""
        router = LongOnlyAlphaRouter()
        market = _market(MarketRegimeLabel.CALM_TREND)
        stock_regimes = _stock_map({"AAPL": StockRegimeLabel.TRENDING_UP})

        hi_model = _per_model_outputs({
            _MOM: {"AAPL": 1.0},
            _MR:  {"AAPL": 0.0},
            _LV:  {"AAPL": 0.0},
        }, confidence=0.95)
        lo_model = _per_model_outputs({
            _MOM: {"AAPL": 1.0},
            _MR:  {"AAPL": 0.0},
            _LV:  {"AAPL": 0.0},
        }, confidence=0.20)

        score_hi = router.route(market, stock_regimes, hi_model)["AAPL"].score
        score_lo = router.route(market, stock_regimes, lo_model)["AAPL"].score

        assert score_lo < score_hi

    def test_confidence_proportionality(self):
        """Halving stock confidence should roughly halve the score."""
        router = LongOnlyAlphaRouter()
        market = _market(MarketRegimeLabel.CALM_TREND)

        conf_full = _stock_map(
            {"AAPL": StockRegimeLabel.TRENDING_UP},
            confidence_map={"AAPL": 1.0},
        )
        conf_half = _stock_map(
            {"AAPL": StockRegimeLabel.TRENDING_UP},
            confidence_map={"AAPL": 0.5},
        )

        model_outputs = _per_model_outputs({
            _MOM: {"AAPL": 1.0},
            _MR:  {"AAPL": 0.0},
            _LV:  {"AAPL": 0.0},
        })

        score_full = router.route(market, conf_full, model_outputs)["AAPL"].score
        score_half = router.route(market, conf_half, model_outputs)["AAPL"].score

        # With a single model contributing, the relationship is linear:
        # contrib = pos_score * w * (model_conf * stock_conf)
        assert score_half == pytest.approx(score_full * 0.5, rel=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
