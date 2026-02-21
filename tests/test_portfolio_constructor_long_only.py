"""
Unit tests for the long-only rank-weight portfolio constructor.

Tests validate each stage of the construction pipeline using small,
deterministic ``AlphaVector`` fixtures with no dependency on real
market data.

Test cases
----------
1) Top-N selection: picks the correct top symbols by score.
2) Weights sum: ``gross`` ≈ ``gross_target``.
3) Long-only: every weight ``>= 0``.
4) Max weight cap: dominant scores are clipped and excess redistributed.
5) Allow cash: empty / insufficient candidates → weights empty or
   gross < gross_target.
6) Rebalance band: small weight changes are suppressed.
7) Turnover limit: total ``|Δw|`` respects the cap.
8) Score floor: scores at or below the floor are excluded.
9) Top-quantile selection mode.
10) Min-weight floor: tiny positions are dropped.
11) Risk multiplier overlay.
12) Deterministic tie-breaking (alphabetical).
13) Reasons dict populated.
"""

from __future__ import annotations

import math
from typing import Dict

import pandas as pd
import pytest

from engine.strategy.alpha.types import AlphaScore, AlphaVector
from engine.strategy.portfolio.constructor_long_only import (
    LongOnlyRankWeightConstructor,
    _compute_turnover,
)
from engine.strategy.portfolio.types import PortfolioConstraints


# ============================================================================
# Helpers
# ============================================================================

ASOF = pd.Timestamp("2025-06-15")


def _vec(score_map: Dict[str, float], confidence: float = 0.80) -> AlphaVector:
    """Build an ``AlphaVector`` from ``{symbol: score}``."""
    return AlphaVector(
        asof=ASOF,
        scores={
            s: AlphaScore(symbol=s, score=sc, raw=sc, confidence=confidence)
            for s, sc in score_map.items()
        },
    )


def _builder() -> LongOnlyRankWeightConstructor:
    return LongOnlyRankWeightConstructor()


def _turnover(
    new_w: Dict[str, float],
    old_w: Dict[str, float],
) -> float:
    """Total turnover helper."""
    return _compute_turnover(new_w, old_w)


# ============================================================================
# 1) Top-N selection
# ============================================================================


class TestTopNSelection:
    """Given 10 symbols with increasing scores, max_positions=3 picks top 3."""

    def test_picks_top_3(self):
        scores = {f"S{i:02d}": float(i) for i in range(1, 11)}
        alpha = _vec(scores)
        universe = list(scores.keys())
        constraints = PortfolioConstraints(
            max_positions=3, min_positions=1, max_weight=1.0,
        )

        tp = _builder().build_targets(ASOF, alpha, universe, constraints)

        assert tp.n_positions == 3
        assert set(tp.symbols) == {"S10", "S09", "S08"}

    def test_fewer_candidates_than_max(self):
        """When only 2 candidates, should hold 2 even if max_positions=10."""
        alpha = _vec({"A": 1.0, "B": 0.5})
        constraints = PortfolioConstraints(
            max_positions=10, min_positions=1, max_weight=1.0,
        )

        tp = _builder().build_targets(ASOF, alpha, ["A", "B"], constraints)

        assert tp.n_positions == 2

    def test_universe_filters_alpha(self):
        """Symbols not in universe are excluded even if scored."""
        alpha = _vec({"A": 1.0, "B": 0.5, "EXTRA": 9.0})
        constraints = PortfolioConstraints(
            max_positions=5, min_positions=1, max_weight=1.0,
        )

        tp = _builder().build_targets(ASOF, alpha, ["A", "B"], constraints)

        assert "EXTRA" not in tp.weights


# ============================================================================
# 2) Weights sum ≈ gross_target
# ============================================================================


class TestWeightsSum:
    """Weights should sum to approximately gross_target."""

    def test_default_gross_target(self):
        scores = {f"S{i}": float(i) for i in range(1, 6)}
        alpha = _vec(scores)
        constraints = PortfolioConstraints(
            max_positions=5, min_positions=1, max_weight=1.0,
        )

        tp = _builder().build_targets(ASOF, alpha, list(scores), constraints)

        assert tp.gross == pytest.approx(1.0, abs=1e-9)

    def test_custom_gross_target(self):
        alpha = _vec({"A": 1.0, "B": 1.0, "C": 1.0})
        constraints = PortfolioConstraints(
            gross_target=0.5, max_positions=5, min_positions=1,
            max_weight=1.0,
        )

        tp = _builder().build_targets(ASOF, alpha, ["A", "B", "C"], constraints)

        assert tp.gross == pytest.approx(0.5, abs=1e-9)

    def test_gross_equals_sum_of_weights(self):
        alpha = _vec({"A": 3.0, "B": 2.0, "C": 1.0})
        constraints = PortfolioConstraints(
            max_positions=5, min_positions=1, max_weight=1.0,
        )

        tp = _builder().build_targets(ASOF, alpha, ["A", "B", "C"], constraints)

        assert tp.gross == pytest.approx(sum(tp.weights.values()), abs=1e-12)


# ============================================================================
# 3) Long-only: all weights >= 0
# ============================================================================


class TestLongOnly:
    """Every weight in the output must be non-negative."""

    def test_all_weights_non_negative(self):
        scores = {f"S{i}": float(i) / 10 for i in range(1, 11)}
        alpha = _vec(scores)
        constraints = PortfolioConstraints(
            max_positions=5, min_positions=1, max_weight=0.05,
        )

        tp = _builder().build_targets(ASOF, alpha, list(scores), constraints)

        for sym, w in tp.weights.items():
            assert w >= 0.0, f"{sym} has negative weight {w}"

    def test_non_negative_after_turnover_limit(self):
        """Turnover scaling must not produce negative weights."""
        alpha = _vec({"A": 0.5, "B": 0.3, "C": 0.2})
        old_w = {"A": 0.10, "B": 0.50, "C": 0.40}
        constraints = PortfolioConstraints(
            max_positions=5, min_positions=1, max_weight=1.0,
            turnover_limit=0.05,
        )

        tp = _builder().build_targets(
            ASOF, alpha, ["A", "B", "C"], constraints,
            current_weights=old_w,
        )

        for sym, w in tp.weights.items():
            assert w >= 0.0, f"{sym} has negative weight {w}"


# ============================================================================
# 4) Max weight cap
# ============================================================================


class TestMaxWeightCap:
    """Dominant scores must be clipped and excess redistributed."""

    def test_cap_applied(self):
        alpha = _vec({"A": 100.0, "B": 1.0, "C": 1.0})
        constraints = PortfolioConstraints(
            max_positions=5, min_positions=1, max_weight=0.05,
        )

        tp = _builder().build_targets(ASOF, alpha, ["A", "B", "C"], constraints)

        assert tp.weights["A"] <= 0.05 + 1e-12
        assert tp.max_weight <= 0.05 + 1e-12

    def test_total_preserved_after_clip(self):
        """Clipping should redistribute excess — total stays at gross_target."""
        alpha = _vec({"A": 10.0, "B": 3.0, "C": 2.0, "D": 1.0})
        constraints = PortfolioConstraints(
            max_positions=10, min_positions=1, max_weight=0.30,
        )

        tp = _builder().build_targets(
            ASOF, alpha, ["A", "B", "C", "D"], constraints,
        )

        assert tp.gross == pytest.approx(1.0, abs=1e-9)
        for sym, w in tp.weights.items():
            assert w <= 0.30 + 1e-12, f"{sym} exceeds cap: {w}"

    def test_all_equal_under_tight_cap(self):
        """If max_weight * n_positions < gross and allow_cash, gross < target."""
        alpha = _vec({"A": 1.0, "B": 1.0, "C": 1.0})
        constraints = PortfolioConstraints(
            max_positions=3, min_positions=1,
            max_weight=0.10,  # 3 * 0.10 = 0.30 < 1.0
            allow_cash=True,
        )

        tp = _builder().build_targets(ASOF, alpha, ["A", "B", "C"], constraints)

        for w in tp.weights.values():
            assert w <= 0.10 + 1e-12
        # Gross can't exceed 3 * 0.10 = 0.30
        assert tp.gross <= 0.30 + 1e-9


# ============================================================================
# 5) Allow cash behaviour
# ============================================================================


class TestAllowCash:
    """With allow_cash=True, insufficient candidates → gross < gross_target."""

    def test_all_scores_zero(self):
        """All scores = 0 (at floor) → no positions, gross = 0."""
        alpha = _vec({"A": 0.0, "B": 0.0, "C": 0.0})
        constraints = PortfolioConstraints(
            max_positions=5, min_positions=1, max_weight=1.0,
            allow_cash=True,
        )

        tp = _builder().build_targets(ASOF, alpha, ["A", "B", "C"], constraints)

        assert tp.n_positions == 0
        assert tp.gross == 0.0

    def test_empty_universe(self):
        alpha = _vec({})
        constraints = PortfolioConstraints(
            max_positions=5, min_positions=1, max_weight=1.0,
        )

        tp = _builder().build_targets(ASOF, alpha, [], constraints)

        assert tp.n_positions == 0
        assert tp.gross == 0.0
        assert tp.weights == {}

    def test_few_candidates_below_min_positions(self):
        """2 candidates vs min_positions=10 → gross scaled down."""
        alpha = _vec({"A": 0.5, "B": 0.3})
        constraints = PortfolioConstraints(
            max_positions=50, min_positions=10, max_weight=1.0,
            allow_cash=True,
        )

        tp = _builder().build_targets(ASOF, alpha, ["A", "B"], constraints)

        assert tp.gross < 1.0
        # 2/10 * 1.0 = 0.2
        assert tp.gross == pytest.approx(0.2, abs=1e-9)

    def test_allow_cash_false_fully_invests(self):
        """allow_cash=False forces gross == gross_target even with few names."""
        alpha = _vec({"A": 0.5, "B": 0.3})
        constraints = PortfolioConstraints(
            max_positions=50, min_positions=10, max_weight=1.0,
            allow_cash=False,
        )

        tp = _builder().build_targets(ASOF, alpha, ["A", "B"], constraints)

        assert tp.gross == pytest.approx(1.0, abs=1e-9)


# ============================================================================
# 6) Rebalance band
# ============================================================================


class TestRebalanceBand:
    """Small weight differences should be suppressed when band > 0."""

    def test_small_change_suppressed(self):
        """Changes < band are held at old weight (after renorm)."""
        old_w = {"A": 0.40, "B": 0.35, "C": 0.25}
        # New scores that produce similar weights:
        alpha = _vec({"A": 0.40, "B": 0.35, "C": 0.25})
        constraints = PortfolioConstraints(
            max_positions=5, min_positions=1, max_weight=1.0,
            rebalance_band=0.10,  # large band → suppress everything
        )

        tp = _builder().build_targets(
            ASOF, alpha, ["A", "B", "C"], constraints,
            current_weights=old_w,
        )

        # Because all changes are < 0.10 band, banded weights start
        # from old values, then get renormalised.
        turnover = _turnover(tp.weights, old_w)
        assert turnover < 0.10 + 1e-9, f"turnover {turnover} should be small"

    def test_large_change_allowed(self):
        """Changes > band are not suppressed."""
        old_w = {"A": 0.80, "B": 0.10, "C": 0.10}
        alpha = _vec({"A": 0.10, "B": 0.80, "C": 0.10})
        constraints = PortfolioConstraints(
            max_positions=5, min_positions=1, max_weight=1.0,
            rebalance_band=0.05,
        )

        tp = _builder().build_targets(
            ASOF, alpha, ["A", "B", "C"], constraints,
            current_weights=old_w,
        )

        # B should have increased substantially
        assert tp.weights.get("B", 0) > old_w["B"]

    def test_band_zero_no_effect(self):
        """band=0 should behave identically to no banding."""
        alpha = _vec({"A": 0.50, "B": 0.30, "C": 0.20})
        old_w = {"A": 0.33, "B": 0.33, "C": 0.34}
        constraints_no = PortfolioConstraints(
            max_positions=5, min_positions=1, max_weight=1.0,
            rebalance_band=0.0,
        )
        constraints_none = PortfolioConstraints(
            max_positions=5, min_positions=1, max_weight=1.0,
        )

        tp_no = _builder().build_targets(
            ASOF, alpha, ["A", "B", "C"], constraints_no,
            current_weights=old_w,
        )
        tp_none = _builder().build_targets(
            ASOF, alpha, ["A", "B", "C"], constraints_none,
            current_weights=old_w,
        )

        for s in ["A", "B", "C"]:
            assert tp_no.weights.get(s, 0) == pytest.approx(
                tp_none.weights.get(s, 0), abs=1e-12,
            )


# ============================================================================
# 7) Turnover limit
# ============================================================================


class TestTurnoverLimit:
    """Total |Δw| must respect the configured limit."""

    def test_turnover_within_limit(self):
        old_w = {"A": 0.40, "B": 0.35, "C": 0.25}
        alpha = _vec({"A": 0.10, "B": 0.60, "C": 0.30})
        constraints = PortfolioConstraints(
            max_positions=5, min_positions=1, max_weight=1.0,
            turnover_limit=0.10,
        )

        tp = _builder().build_targets(
            ASOF, alpha, ["A", "B", "C"], constraints,
            current_weights=old_w,
        )

        turnover = _turnover(tp.weights, old_w)
        assert turnover <= 0.10 + 1e-9, f"turnover {turnover} > limit 0.10"

    def test_no_limit_allows_full_trade(self):
        old_w = {"A": 1.0}
        alpha = _vec({"B": 1.0})
        constraints = PortfolioConstraints(
            max_positions=5, min_positions=1, max_weight=1.0,
            turnover_limit=None,
        )

        tp = _builder().build_targets(
            ASOF, alpha, ["B"], constraints,
            current_weights=old_w,
        )

        # Full rotation: sell A, buy B → turnover = 2.0
        turnover = _turnover(tp.weights, old_w)
        assert turnover > 1.0

    def test_tight_limit_barely_moves(self):
        old_w = {"A": 0.50, "B": 0.50}
        alpha = _vec({"A": 0.10, "B": 0.90})
        constraints = PortfolioConstraints(
            max_positions=5, min_positions=1, max_weight=1.0,
            turnover_limit=0.01,
        )

        tp = _builder().build_targets(
            ASOF, alpha, ["A", "B"], constraints,
            current_weights=old_w,
        )

        turnover = _turnover(tp.weights, old_w)
        assert turnover <= 0.01 + 1e-9

    def test_turnover_in_reasons(self):
        """Turnover should appear in reasons when current_weights given."""
        old_w = {"A": 0.50, "B": 0.50}
        alpha = _vec({"A": 0.60, "B": 0.40})
        constraints = PortfolioConstraints(
            max_positions=5, min_positions=1, max_weight=1.0,
        )

        tp = _builder().build_targets(
            ASOF, alpha, ["A", "B"], constraints,
            current_weights=old_w,
        )

        assert "turnover" in tp.reasons


# ============================================================================
# 8) Score floor
# ============================================================================


class TestScoreFloor:
    """Scores at or below score_floor are excluded."""

    def test_zero_scores_excluded(self):
        alpha = _vec({"A": 0.5, "B": 0.0, "C": 0.3})
        constraints = PortfolioConstraints(
            max_positions=5, min_positions=1, max_weight=1.0,
            score_floor=0.0,
        )

        tp = _builder().build_targets(ASOF, alpha, ["A", "B", "C"], constraints)

        assert "B" not in tp.weights or tp.weights.get("B", 0) == 0.0
        assert tp.n_positions == 2

    def test_custom_floor(self):
        alpha = _vec({"A": 0.50, "B": 0.20, "C": 0.05})
        constraints = PortfolioConstraints(
            max_positions=5, min_positions=1, max_weight=1.0,
            score_floor=0.10,
        )

        tp = _builder().build_targets(ASOF, alpha, ["A", "B", "C"], constraints)

        assert "C" not in tp.weights or tp.weights.get("C", 0) == 0.0
        assert tp.n_positions == 2


# ============================================================================
# 9) Top-quantile selection mode
# ============================================================================


class TestTopQuantile:
    """selection_mode='top_quantile' picks the top fraction of candidates."""

    def test_top_25_pct_of_20(self):
        scores = {f"S{i:02d}": float(i) for i in range(1, 21)}
        alpha = _vec(scores)
        constraints = PortfolioConstraints(
            max_positions=50, min_positions=1, max_weight=0.20,
            selection_mode="top_quantile", top_quantile=0.25,
        )

        tp = _builder().build_targets(ASOF, alpha, list(scores), constraints)

        # ceil(20 * 0.25) = 5
        assert tp.n_positions == 5

    def test_quantile_capped_by_max_positions(self):
        scores = {f"S{i:02d}": float(i) for i in range(1, 101)}
        alpha = _vec(scores)
        constraints = PortfolioConstraints(
            max_positions=10, min_positions=1, max_weight=0.20,
            selection_mode="top_quantile", top_quantile=0.50,
        )

        tp = _builder().build_targets(ASOF, alpha, list(scores), constraints)

        # ceil(100 * 0.50) = 50, but capped to max_positions=10
        assert tp.n_positions == 10


# ============================================================================
# 10) Min-weight floor
# ============================================================================


class TestMinWeightFloor:
    """Positions below min_weight are dropped and weight redistributed."""

    def test_tiny_position_dropped(self):
        # A dominates → B and C get tiny weights that fall below min_weight
        alpha = _vec({"A": 100.0, "B": 1.0, "C": 0.5})
        constraints = PortfolioConstraints(
            max_positions=5, min_positions=1,
            max_weight=1.0, min_weight=0.02,
        )

        tp = _builder().build_targets(ASOF, alpha, ["A", "B", "C"], constraints)

        for sym, w in tp.weights.items():
            if w > 0:
                assert w >= 0.02, f"{sym} weight {w} < min_weight"

    def test_min_weight_zero_keeps_all(self):
        """min_weight=0 should not drop anything."""
        alpha = _vec({"A": 10.0, "B": 0.01})
        constraints = PortfolioConstraints(
            max_positions=5, min_positions=1,
            max_weight=1.0, min_weight=0.0,
        )

        tp = _builder().build_targets(ASOF, alpha, ["A", "B"], constraints)

        assert tp.n_positions == 2


# ============================================================================
# 11) Risk multiplier overlay
# ============================================================================


class TestRiskMultiplier:
    """Optional risk_multiplier adjusts scores before selection."""

    def test_none_is_no_op(self):
        alpha = _vec({"A": 0.5, "B": 0.3, "C": 0.2})
        constraints = PortfolioConstraints(
            max_positions=5, min_positions=1, max_weight=1.0,
        )

        tp_none = _builder().build_targets(
            ASOF, alpha, ["A", "B", "C"], constraints,
            risk_multiplier=None,
        )
        tp_omit = _builder().build_targets(
            ASOF, alpha, ["A", "B", "C"], constraints,
        )

        assert tp_none.weights == tp_omit.weights

    def test_zero_multiplier_removes_symbol(self):
        alpha = _vec({"A": 0.5, "B": 0.3, "C": 0.2})
        constraints = PortfolioConstraints(
            max_positions=5, min_positions=1, max_weight=1.0,
        )

        tp = _builder().build_targets(
            ASOF, alpha, ["A", "B", "C"], constraints,
            risk_multiplier={"A": 0.0},
        )

        assert "A" not in tp.weights or tp.weights.get("A", 0) == 0.0

    def test_downweight_changes_ranking(self):
        alpha = _vec({"A": 0.50, "B": 0.49})
        constraints = PortfolioConstraints(
            max_positions=1, min_positions=1, max_weight=1.0,
        )

        tp = _builder().build_targets(
            ASOF, alpha, ["A", "B"], constraints,
            risk_multiplier={"A": 0.1},  # A: 0.50*0.1=0.05 < B: 0.49
        )

        assert tp.symbols == ["B"]

    def test_values_clamped(self):
        """Multipliers > 1 clamped to 1; < 0 clamped to 0."""
        alpha = _vec({"A": 0.5, "B": 0.3})
        constraints = PortfolioConstraints(
            max_positions=5, min_positions=1, max_weight=1.0,
        )

        tp_over = _builder().build_targets(
            ASOF, alpha, ["A", "B"], constraints,
            risk_multiplier={"A": 5.0, "B": 1.5},
        )
        tp_base = _builder().build_targets(
            ASOF, alpha, ["A", "B"], constraints,
        )

        for s in ["A", "B"]:
            assert tp_over.weights[s] == pytest.approx(
                tp_base.weights[s], abs=1e-12,
            )


# ============================================================================
# 12) Deterministic tie-breaking
# ============================================================================


class TestDeterministicTieBreaking:
    """Equal scores should be broken alphabetically by symbol."""

    def test_alphabetical_tie_break(self):
        alpha = _vec({"Z": 1.0, "A": 1.0, "M": 1.0})
        constraints = PortfolioConstraints(
            max_positions=2, min_positions=1, max_weight=1.0,
        )

        tp = _builder().build_targets(
            ASOF, alpha, ["Z", "A", "M"], constraints,
        )

        assert tp.symbols == ["A", "M"]

    def test_stable_across_calls(self):
        alpha = _vec({"X": 0.5, "A": 0.5, "N": 0.5})
        constraints = PortfolioConstraints(
            max_positions=2, min_positions=1, max_weight=1.0,
        )

        results = [
            _builder().build_targets(ASOF, alpha, ["X", "A", "N"], constraints)
            for _ in range(5)
        ]

        for tp in results:
            assert tp.symbols == results[0].symbols


# ============================================================================
# 13) Reasons dict populated
# ============================================================================


class TestReasonsDict:
    """The TargetPortfolio reasons dict must contain expected keys."""

    def test_basic_keys_present(self):
        alpha = _vec({"A": 0.5, "B": 0.3, "C": 0.2})
        constraints = PortfolioConstraints(
            max_positions=5, min_positions=1, max_weight=1.0,
        )

        tp = _builder().build_targets(ASOF, alpha, ["A", "B", "C"], constraints)

        assert "selected_count" in tp.reasons
        assert "dropped_count" in tp.reasons
        assert "gross_target" in tp.reasons
        assert "cash_weight" in tp.reasons

    def test_selected_count_matches(self):
        alpha = _vec({"A": 0.5, "B": 0.3})
        constraints = PortfolioConstraints(
            max_positions=5, min_positions=1, max_weight=1.0,
        )

        tp = _builder().build_targets(ASOF, alpha, ["A", "B"], constraints)

        assert tp.reasons["selected_count"] == float(tp.n_positions)

    def test_cash_weight_correct(self):
        alpha = _vec({"A": 0.5, "B": 0.3})
        constraints = PortfolioConstraints(
            max_positions=50, min_positions=10, max_weight=1.0,
            allow_cash=True,
        )

        tp = _builder().build_targets(ASOF, alpha, ["A", "B"], constraints)

        expected_cash = max(0.0, constraints.gross_target - tp.gross)
        assert tp.reasons["cash_weight"] == pytest.approx(expected_cash, abs=1e-12)

    def test_turnover_absent_without_current_weights(self):
        alpha = _vec({"A": 0.5})
        constraints = PortfolioConstraints(
            max_positions=5, min_positions=1, max_weight=1.0,
        )

        tp = _builder().build_targets(ASOF, alpha, ["A"], constraints)

        assert "turnover" not in tp.reasons


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
