"""
Tests for the event-driven FeatureEngine.

Uses simple synthetic bars (linearly increasing prices) so expected
values are easy to reason about analytically.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pandas as pd
import pytest

from engine.features.definitions import FeatureSpec, default_feature_specs
from engine.features.feature_engine import FeatureEngine


# ============================================================================
# Helpers
# ============================================================================

_SQRT_252 = math.sqrt(252)


def _make_bar(symbol: str, close: float, volume: float, asof: pd.Timestamp):
    """Return a plain-dict bar."""
    return {
        "symbol": symbol,
        "close": close,
        "volume": volume,
        "asof": asof,
    }


def _feed_linear(
    engine: FeatureEngine,
    symbol: str,
    n: int,
    *,
    start_close: float = 100.0,
    step: float = 1.0,
    volume: float = 1000.0,
    start_date: str = "2020-01-01",
) -> pd.Timestamp:
    """Feed *n* bars with linearly increasing close prices.

    Returns the asof of the last bar.
    """
    dates = pd.bdate_range(start=start_date, periods=n)
    for i, dt in enumerate(dates):
        c = start_close + i * step
        engine.update(_make_bar(symbol, c, volume, dt))
    return dates[-1]


def _small_specs() -> List[FeatureSpec]:
    """Subset of specs for focused tests (keeps bundles small)."""
    keep = {"ret1", "sma20", "mom63", "rv20", "adv20"}
    return [s for s in default_feature_specs() if s.name in keep]


# ============================================================================
# Test: update() stores close & computes ret1 correctly
# ============================================================================


class TestUpdateAndRet1:
    """Verify that update() populates close and return windows."""

    def test_first_bar_has_nan_return(self):
        """The very first bar cannot compute a return → ret1 should be NaN."""
        engine = FeatureEngine(["A"], feature_specs=_small_specs())
        asof = pd.Timestamp("2020-01-02")
        engine.update(_make_bar("A", 100.0, 1000.0, asof))

        bundle = engine.get_feature_bundle(asof)
        ret1 = bundle.get("A", "ret1")
        assert math.isnan(ret1)

    def test_second_bar_ret1(self):
        """After two bars ret1 should equal close[1]/close[0] − 1."""
        engine = FeatureEngine(["A"], feature_specs=_small_specs())
        t0 = pd.Timestamp("2020-01-02")
        t1 = pd.Timestamp("2020-01-03")

        engine.update(_make_bar("A", 100.0, 1000.0, t0))
        engine.update(_make_bar("A", 105.0, 1000.0, t1))

        bundle = engine.get_feature_bundle(t1)
        ret1 = bundle.get("A", "ret1")
        assert ret1 == pytest.approx(105.0 / 100.0 - 1.0)

    def test_ret1_after_many_bars(self):
        """ret1 always reflects the last two closes, not older history."""
        engine = FeatureEngine(["A"], feature_specs=_small_specs())
        dates = pd.bdate_range("2020-01-01", periods=50)
        for i, dt in enumerate(dates):
            engine.update(_make_bar("A", 100.0 + i, 1000.0, dt))

        bundle = engine.get_feature_bundle(dates[-1])
        # close[-1] = 149, close[-2] = 148 → ret1 = 149/148 − 1
        expected = 149.0 / 148.0 - 1.0
        assert bundle.get("A", "ret1") == pytest.approx(expected)

    def test_unknown_symbol_ignored(self):
        """Bars for symbols not in the universe are silently dropped."""
        engine = FeatureEngine(["A"], feature_specs=_small_specs())
        engine.update(_make_bar("UNKNOWN", 100.0, 1000.0, pd.Timestamp("2020-01-02")))
        assert engine.bars_received["A"] == 0


# ============================================================================
# Test: sma20 readiness (NaN before 20 bars, finite after)
# ============================================================================


class TestSma20Readiness:
    """sma20 = mean(close[-20:]), needs nc >= 20."""

    def test_nan_before_20_bars(self):
        """With only 19 bars, sma20 must be NaN."""
        engine = FeatureEngine(["A"], feature_specs=_small_specs())
        asof = _feed_linear(engine, "A", 19)

        bundle = engine.get_feature_bundle(asof)
        assert math.isnan(bundle.get("A", "sma20"))

    def test_finite_at_20_bars(self):
        """With exactly 20 bars, sma20 should be the mean of those 20 closes."""
        engine = FeatureEngine(["A"], feature_specs=_small_specs())
        asof = _feed_linear(engine, "A", 20, start_close=1.0, step=1.0)

        bundle = engine.get_feature_bundle(asof)
        sma = bundle.get("A", "sma20")
        # closes are 1, 2, ..., 20 → mean = 10.5
        assert sma == pytest.approx(10.5)

    def test_sma20_rolling(self):
        """After 25 bars, sma20 uses only the last 20 closes."""
        engine = FeatureEngine(["A"], feature_specs=_small_specs())
        asof = _feed_linear(engine, "A", 25, start_close=1.0, step=1.0)

        bundle = engine.get_feature_bundle(asof)
        sma = bundle.get("A", "sma20")
        # closes 6..25 → mean = (6+25)/2 = 15.5
        assert sma == pytest.approx(15.5)


# ============================================================================
# Test: mom63 requires 64 closes
# ============================================================================


class TestMom63Readiness:
    """mom63 = close[-1] / close[-64] − 1, needs nc >= 64.

    Uses default specs (max_lookback=252) so the rolling-window can
    hold 64+ closes.  With ``_small_specs()`` the max lookback is only
    63, which is too small for mom63's ``ca[-64]`` access.
    """

    def test_nan_at_63_bars(self):
        """63 bars → only 63 closes → close[-64] doesn't exist → NaN."""
        engine = FeatureEngine(["A"])
        asof = _feed_linear(engine, "A", 63)

        bundle = engine.get_feature_bundle(asof)
        assert math.isnan(bundle.get("A", "mom63"))

    def test_finite_at_64_bars(self):
        """64 bars → close[-64] exists → mom63 is finite."""
        engine = FeatureEngine(["A"])
        asof = _feed_linear(engine, "A", 64, start_close=100.0, step=1.0)

        bundle = engine.get_feature_bundle(asof)
        mom = bundle.get("A", "mom63")
        assert math.isfinite(mom)
        # close[-1]=163, close[-64]=100 → mom = 163/100 − 1 = 0.63
        assert mom == pytest.approx(163.0 / 100.0 - 1.0)

    def test_mom63_value_large_n(self):
        """After 200 bars, mom63 still uses close[-1]/close[-64]."""
        engine = FeatureEngine(["A"])
        asof = _feed_linear(engine, "A", 200, start_close=10.0, step=0.5)

        bundle = engine.get_feature_bundle(asof)
        # close[-1] = 10 + 199*0.5 = 109.5
        # close[-64] = 10 + (199-63)*0.5 = 10 + 136*0.5 = 78.0
        expected = 109.5 / 78.0 - 1.0
        assert bundle.get("A", "mom63") == pytest.approx(expected)


# ============================================================================
# Test: rv20 becomes finite after enough returns
# ============================================================================


class TestRv20Readiness:
    """rv20 = std(ret[-20:], ddof=1) × √252, needs nr >= 20."""

    def test_nan_before_20_returns(self):
        """19 bars → 18 real returns + 1 NaN (first) → not enough."""
        engine = FeatureEngine(["A"], feature_specs=_small_specs())
        asof = _feed_linear(engine, "A", 19)

        bundle = engine.get_feature_bundle(asof)
        assert math.isnan(bundle.get("A", "rv20"))

    def test_finite_at_21_bars(self):
        """21 bars → 20 returns (first NaN + 19 real, but window
        slices last 20 which includes that NaN; after NaN-filter 19
        valid ≥ 15 → rv20 is finite)."""
        engine = FeatureEngine(["A"], feature_specs=_small_specs())
        asof = _feed_linear(engine, "A", 21, start_close=100.0, step=1.0)

        bundle = engine.get_feature_bundle(asof)
        rv20 = bundle.get("A", "rv20")
        assert math.isfinite(rv20)
        assert rv20 > 0.0

    def test_rv20_positive_for_varying_prices(self):
        """rv20 must be positive when returns are non-constant."""
        engine = FeatureEngine(["A"], feature_specs=_small_specs())
        asof = _feed_linear(engine, "A", 50, start_close=50.0, step=2.0)

        bundle = engine.get_feature_bundle(asof)
        rv20 = bundle.get("A", "rv20")
        assert math.isfinite(rv20)
        assert rv20 > 0.0

    def test_rv20_is_annualised(self):
        """rv20 should be daily std × √252."""
        engine = FeatureEngine(["A"], feature_specs=_small_specs())
        asof = _feed_linear(engine, "A", 30, start_close=100.0, step=1.0)

        bundle = engine.get_feature_bundle(asof)
        rv20 = bundle.get("A", "rv20")

        # With 30 bars, close = [100, 101, ..., 129],
        # ret window = [NaN, 1/100, 1/101, ..., 1/128] (30 entries).
        # ra[-20:] picks returns at bars 10..29 = [1/109, 1/110, ..., 1/128].
        last_20_rets = 1.0 / np.arange(109, 129, dtype=float)
        expected = float(np.std(last_20_rets, ddof=1)) * _SQRT_252
        assert rv20 == pytest.approx(expected, rel=1e-4)


# ============================================================================
# Test: get_feature_bundle shape and mask
# ============================================================================


class TestBundleShapeAndMask:
    """FeatureBundle.X must be (N, F) float32, mask marks NaN."""

    def test_shape_matches_universe_and_specs(self):
        specs = _small_specs()
        engine = FeatureEngine(["A", "B", "C"], feature_specs=specs)
        asof = _feed_linear(engine, "A", 5)

        bundle = engine.get_feature_bundle(asof)
        assert bundle.X.shape == (3, len(specs))
        assert bundle.X.dtype == np.float32

    def test_unfed_symbol_is_all_nan(self):
        """Symbols that never received a bar should be entirely NaN."""
        engine = FeatureEngine(["A", "B"], feature_specs=_small_specs())
        asof = _feed_linear(engine, "A", 30)

        bundle = engine.get_feature_bundle(asof)
        row_b = bundle.row("B")
        assert np.all(np.isnan(row_b))

    def test_mask_true_for_nan(self):
        """mask[i,j] must be True wherever X[i,j] is NaN."""
        engine = FeatureEngine(["A"], feature_specs=_small_specs())
        asof = _feed_linear(engine, "A", 10)

        bundle = engine.get_feature_bundle(asof)
        assert bundle.mask is not None
        assert np.array_equal(bundle.mask, np.isnan(bundle.X))

    def test_mask_false_for_finite(self):
        """mask[i,j] must be False wherever X[i,j] is finite."""
        engine = FeatureEngine(["A"], feature_specs=_small_specs())
        asof = _feed_linear(engine, "A", 100)

        bundle = engine.get_feature_bundle(asof)
        finite = np.isfinite(bundle.X)
        assert bundle.mask is not None
        assert np.all(bundle.mask[finite] == False)  # noqa: E712

    def test_subset_universe(self):
        """get_feature_bundle(universe=[...]) filters to that subset."""
        engine = FeatureEngine(["A", "B", "C"], feature_specs=_small_specs())
        _feed_linear(engine, "A", 5)
        _feed_linear(engine, "B", 5)
        asof = _feed_linear(engine, "C", 5)

        bundle = engine.get_feature_bundle(asof, universe=["B", "C"])
        assert bundle.symbols == ["B", "C"]
        assert bundle.X.shape[0] == 2

    def test_symbols_and_feature_names_match(self):
        specs = _small_specs()
        engine = FeatureEngine(["X", "Y"], feature_specs=specs)
        asof = _feed_linear(engine, "X", 5)

        bundle = engine.get_feature_bundle(asof)
        assert bundle.symbols == ["X", "Y"]
        assert bundle.feature_names == [s.name for s in specs]

    def test_asof_propagated(self):
        engine = FeatureEngine(["A"], feature_specs=_small_specs())
        ts = pd.Timestamp("2025-06-15")
        engine.update(_make_bar("A", 100.0, 1000.0, ts))

        bundle = engine.get_feature_bundle(ts)
        assert bundle.asof == ts


# ============================================================================
# Test: is_ready threshold
# ============================================================================


class TestIsReady:
    """is_ready requires >= readiness_threshold fraction of symbols to
    have bars_seen >= max_lookback."""

    def test_not_ready_initially(self):
        engine = FeatureEngine(["A", "B"], feature_specs=_small_specs())
        assert not engine.is_ready()

    def test_not_ready_after_few_bars(self):
        engine = FeatureEngine(["A"], feature_specs=_small_specs())
        _feed_linear(engine, "A", 10)
        assert not engine.is_ready()

    def test_ready_after_max_lookback(self):
        """One symbol, fed >= max_lookback bars → ready (100% > 80%)."""
        specs = _small_specs()
        engine = FeatureEngine(["A"], feature_specs=specs)
        _feed_linear(engine, "A", engine.max_lookback)
        assert engine.is_ready()

    def test_threshold_with_multiple_symbols(self):
        """5 symbols at 80% threshold → need 4 ready."""
        specs = [FeatureSpec(name="ret1", lookback=10, inputs=["close"])]
        syms = ["A", "B", "C", "D", "E"]
        engine = FeatureEngine(syms, feature_specs=specs, readiness_threshold=0.8)

        # Feed only 3 of 5 symbols enough bars → 60% < 80%
        for s in ["A", "B", "C"]:
            _feed_linear(engine, s, engine.max_lookback)
        assert not engine.is_ready()

        # Feed one more → 4/5 = 80% → ready
        _feed_linear(engine, "D", engine.max_lookback)
        assert engine.is_ready()

    def test_threshold_respects_value(self):
        """Lowering threshold to 0.5 → 3/5 = 60% suffices."""
        specs = [FeatureSpec(name="ret1", lookback=10, inputs=["close"])]
        syms = ["A", "B", "C", "D", "E"]
        engine = FeatureEngine(syms, feature_specs=specs, readiness_threshold=0.5)

        for s in ["A", "B", "C"]:
            _feed_linear(engine, s, engine.max_lookback)

        assert engine.is_ready()


# ============================================================================
# Test: adv20 (dollar volume feature)
# ============================================================================


class TestAdv20:
    """adv20 = mean(close × volume, last 20 bars)."""

    def test_nan_before_20(self):
        engine = FeatureEngine(["A"], feature_specs=_small_specs())
        asof = _feed_linear(engine, "A", 19, volume=500.0)

        assert math.isnan(engine.get_feature_bundle(asof).get("A", "adv20"))

    def test_value_at_20(self):
        engine = FeatureEngine(["A"], feature_specs=_small_specs())
        asof = _feed_linear(engine, "A", 20, start_close=10.0, step=0.0, volume=100.0)

        adv = engine.get_feature_bundle(asof).get("A", "adv20")
        # close=10 constant, volume=100 → dollar_vol=1000 → mean=1000
        assert adv == pytest.approx(1000.0)


# ============================================================================
# Test: reset clears all state
# ============================================================================


class TestReset:
    def test_reset_clears_bars(self):
        engine = FeatureEngine(["A"], feature_specs=_small_specs())
        _feed_linear(engine, "A", 50)
        assert engine.bars_received["A"] == 50

        engine.reset()
        assert engine.bars_received["A"] == 0
        assert engine.last_asof is None

    def test_features_nan_after_reset(self):
        engine = FeatureEngine(["A"], feature_specs=_small_specs())
        asof = _feed_linear(engine, "A", 50)

        engine.reset()
        bundle = engine.get_feature_bundle(asof)
        assert np.all(np.isnan(bundle.X))


# ============================================================================
# Test: properties
# ============================================================================


class TestProperties:
    def test_universe(self):
        engine = FeatureEngine(["X", "Y", "Z"], feature_specs=_small_specs())
        assert engine.universe == ["X", "Y", "Z"]

    def test_feature_names(self):
        specs = _small_specs()
        engine = FeatureEngine(["A"], feature_specs=specs)
        assert engine.feature_names == [s.name for s in specs]

    def test_max_lookback(self):
        specs = _small_specs()
        engine = FeatureEngine(["A"], feature_specs=specs)
        assert engine.max_lookback == max(s.lookback for s in specs)

    def test_empty_universe_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            FeatureEngine([], feature_specs=_small_specs())

    def test_repr(self):
        engine = FeatureEngine(["A", "B"], feature_specs=_small_specs())
        r = repr(engine)
        assert "FeatureEngine" in r
        assert "universe=2" in r


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
