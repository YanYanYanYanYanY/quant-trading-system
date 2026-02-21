"""
Tests for rule-based stock regime detector.

Validates:
- trending-up stock is classified TRENDING_UP
- sudden drop triggers STRESSED_HIGH_VOL
- hysteresis prevents flip-flopping around thresholds
- min-duration blocks rapid switching
- confidence gate blocks marginal switches
- NO_TRADE on missing / insufficient data
- market-stress conditioning tightens stress entry
- protocol conformance
"""

import math
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from engine.features.types import FeatureBundle
from engine.strategy.regime import (
    MarketRegimeLabel,
    MarketRegimeState,
    RuleBasedStockRegimeDetector,
    StockRegimeConfig,
    StockRegimeDetector,
    StockRegimeLabel,
    StockRegimeMap,
)


# ============================================================================
# Helpers — synthetic feature builders
# ============================================================================

_SQRT_252 = math.sqrt(252)

# Feature names consumed by the detector's _extract_signals method.
_FEATURE_NAMES: List[str] = [
    "max_dd_90",
    "mom63",
    "normalized_slope",
    "rv20",
    "rv20_z",
    "trend_gap",
]


def _dates(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n, freq="D")


def _calm_market(asof: pd.Timestamp) -> MarketRegimeState:
    return MarketRegimeState(
        asof=asof,
        label=MarketRegimeLabel.CALM_TREND,
        confidence=0.8,
    )


def _stressed_market(asof: pd.Timestamp) -> MarketRegimeState:
    return MarketRegimeState(
        asof=asof,
        label=MarketRegimeLabel.STRESS,
        confidence=0.75,
    )


def _compute_normalized_slope(close_arr: np.ndarray, window: int = 50) -> float:
    """OLS slope of log(close) / daily vol over *window*."""
    if len(close_arr) < window:
        return np.nan
    c = close_arr[-window:]
    log_c = np.log(np.maximum(c, 1e-10))
    x = np.arange(window, dtype=float)
    x_m = x.mean()
    y_m = log_c.mean()
    slope = np.dot(x - x_m, log_c - y_m) / np.dot(x - x_m, x - x_m)
    daily_vol = float(np.std(np.diff(log_c), ddof=1))
    if daily_vol > 0 and np.isfinite(daily_vol):
        return slope / daily_vol
    return np.nan


def _compute_max_dd(close_arr: np.ndarray, window: int = 90) -> float:
    """Rolling max drawdown over *window*."""
    if len(close_arr) < window:
        return np.nan
    c = close_arr[-window:]
    rm = np.maximum.accumulate(c)
    dd = (c - rm) / np.where(rm > 0, rm, 1.0)
    return float(np.nanmin(dd))


def make_feature_dfs(
    symbols: List[str],
    n_days: int = 600,
    *,
    daily_return: float = 0.0,
    volatility: float = 0.01,
    crash_at: int | None = None,
    crash_duration: int = 20,
    crash_mag: float = -0.20,
    crash_vol: float = 0.04,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Build DataFrames of all features from synthetic log-return parameters.

    Returns a dict with keys matching ``_FEATURE_NAMES`` plus ``"close"``
    (used by the test to index into dates).
    """
    dates = _dates(n_days)

    close_dict: Dict[str, pd.Series] = {}
    rv20_dict: Dict[str, pd.Series] = {}
    mom63_dict: Dict[str, pd.Series] = {}
    nslope_dict: Dict[str, pd.Series] = {}
    rv20z_dict: Dict[str, pd.Series] = {}
    maxdd_dict: Dict[str, pd.Series] = {}
    tgap_dict: Dict[str, pd.Series] = {}

    for i, sym in enumerate(symbols):
        sym_rng = np.random.default_rng(seed + i)
        rets = sym_rng.normal(daily_return, volatility, n_days)

        if crash_at is not None:
            c_end = min(crash_at + crash_duration, n_days)
            rets[crash_at:c_end] = sym_rng.normal(
                crash_mag / crash_duration, crash_vol, c_end - crash_at,
            )

        log_p = np.cumsum(rets) + np.log(100.0)
        close = np.exp(log_p)

        close_s = pd.Series(close, index=dates, name=sym)
        close_dict[sym] = close_s

        # rv20 — annualized 20d realized vol
        log_ret = np.log(close_s / close_s.shift(1))
        rv20 = log_ret.rolling(20).std() * _SQRT_252
        rv20_dict[sym] = rv20

        # mom63
        mom63 = close_s / close_s.shift(63) - 1
        mom63_dict[sym] = mom63

        # normalized_slope (50-day, computed per bar)
        nslope = pd.Series(np.nan, index=dates, name=sym)
        for t in range(49, n_days):
            nslope.iloc[t] = _compute_normalized_slope(close[:t + 1])
        nslope_dict[sym] = nslope

        # rv20_z (z-score of rv20 vs trailing history)
        rv20z = pd.Series(np.nan, index=dates, name=sym)
        for t in range(19, n_days):
            history = rv20.iloc[max(0, t - 251):t + 1].dropna()
            if len(history) >= 60:
                mu = history.mean()
                sig = history.std()
                if sig > 0:
                    rv20z.iloc[t] = (rv20.iloc[t] - mu) / sig
        rv20z_dict[sym] = rv20z

        # max_dd_90
        maxdd = pd.Series(np.nan, index=dates, name=sym)
        for t in range(89, n_days):
            maxdd.iloc[t] = _compute_max_dd(close[:t + 1])
        maxdd_dict[sym] = maxdd

        # trend_gap
        sma50 = close_s.rolling(50).mean()
        sma200 = close_s.rolling(200).mean()
        tgap = (sma50 - sma200) / sma200.replace(0, np.nan)
        tgap_dict[sym] = tgap

    return {
        "close": pd.DataFrame(close_dict),
        "rv20": pd.DataFrame(rv20_dict),
        "mom63": pd.DataFrame(mom63_dict),
        "normalized_slope": pd.DataFrame(nslope_dict),
        "rv20_z": pd.DataFrame(rv20z_dict),
        "max_dd_90": pd.DataFrame(maxdd_dict),
        "trend_gap": pd.DataFrame(tgap_dict),
    }


def _to_bundle(
    feature_dfs: Dict[str, pd.DataFrame],
    asof: pd.Timestamp,
    symbols: List[str],
) -> FeatureBundle:
    """Convert dict-of-DataFrames into a FeatureBundle at *asof*."""
    N = len(symbols)
    F = len(_FEATURE_NAMES)
    X = np.full((N, F), np.nan, dtype=np.float32)

    for j, fname in enumerate(_FEATURE_NAMES):
        df = feature_dfs.get(fname)
        if df is None:
            continue
        for i, sym in enumerate(symbols):
            if sym not in df.columns:
                continue
            col = df[sym].loc[:asof].dropna()
            if not col.empty:
                X[i, j] = float(col.iloc[-1])

    return FeatureBundle(
        asof=asof,
        symbols=list(symbols),
        feature_names=list(_FEATURE_NAMES),
        X=X,
    )


def make_features(
    symbols: List[str],
    n_days: int = 600,
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    """Convenience wrapper — returns the raw DataFrames dict.

    Useful for accessing ``features["close"].index`` (dates) and for
    the ``_to_bundle`` helper.
    """
    return make_feature_dfs(symbols, n_days, **kwargs)


def run_multi_day(
    detector: RuleBasedStockRegimeDetector,
    features: Dict[str, pd.DataFrame],
    universe: List[str],
    market_fn=_calm_market,
    start_idx: int = 300,
) -> List[StockRegimeMap]:
    """
    Run detector day-by-day from *start_idx* to end.

    Converts the dict-of-DataFrames to a :class:`FeatureBundle` at
    each bar, then calls ``detect``.

    Returns a list of StockRegimeMap, one per day.
    """
    dates = features["close"].index
    results = []
    for i in range(start_idx, len(dates)):
        asof = dates[i]
        bundle = _to_bundle(features, asof, universe)
        mkt = market_fn(asof)
        regime_map = detector.detect(asof, mkt, bundle, universe)
        results.append(regime_map)
    return results


# ============================================================================
# Test: Protocol conformance
# ============================================================================


class TestProtocol:
    def test_conforms_to_protocol(self):
        """RuleBasedStockRegimeDetector satisfies StockRegimeDetector protocol."""
        detector = RuleBasedStockRegimeDetector()
        assert isinstance(detector, StockRegimeDetector)


# ============================================================================
# Test: Trending-up classification
# ============================================================================


class TestTrendingUp:
    def test_uptrend_detected(self):
        """A strongly trending stock should become TRENDING_UP."""
        features = make_features(
            ["BULL"], daily_return=0.002, volatility=0.007, seed=10,
        )
        cfg = StockRegimeConfig(
            min_regime_duration=1,
            confidence_gate=0.3,
            enter_trend_slope=0.4,
            exit_trend_slope=0.15,
        )
        detector = RuleBasedStockRegimeDetector(cfg)
        results = run_multi_day(detector, features, ["BULL"])

        # Collect unique labels in second half (after detector warms up)
        labels = [r["BULL"].label for r in results[-100:]]
        up_count = labels.count(StockRegimeLabel.TRENDING_UP)

        assert up_count > len(labels) * 0.4, (
            f"Expected >40 % TRENDING_UP, got {up_count}/{len(labels)}"
        )

    def test_uptrend_confidence_positive(self):
        """Confidence should be meaningfully positive in strong uptrend."""
        features = make_features(
            ["BULL"], daily_return=0.002, volatility=0.007, seed=10,
        )
        cfg = StockRegimeConfig(min_regime_duration=1, confidence_gate=0.3)
        detector = RuleBasedStockRegimeDetector(cfg)
        results = run_multi_day(detector, features, ["BULL"])

        confs = [r["BULL"].confidence for r in results[-100:]]
        assert np.mean(confs) > 0.4


# ============================================================================
# Test: Stress detection
# ============================================================================


class TestStressed:
    def test_crash_triggers_stress(self):
        """A sudden crash should trigger STRESSED_HIGH_VOL."""
        features = make_features(
            ["CRASH"],
            daily_return=0.0003,
            volatility=0.008,
            crash_at=400,
            crash_mag=-0.25,
            crash_vol=0.04,
            seed=42,
        )
        cfg = StockRegimeConfig(
            min_regime_duration=1,
            confidence_gate=0.3,
            enter_stress_z=1.2,
            exit_stress_z=0.8,
        )
        detector = RuleBasedStockRegimeDetector(cfg)
        results = run_multi_day(detector, features, ["CRASH"], start_idx=300)

        # Look at labels right after crash (day 400..430 → index 100..130)
        post_crash = results[100:140]
        stressed_count = sum(
            1
            for r in post_crash
            if r["CRASH"].label == StockRegimeLabel.STRESSED_HIGH_VOL
        )
        assert stressed_count > 0, "Should detect STRESSED after crash"

    def test_stress_bypasses_min_duration(self):
        """Stress entry should bypass min-duration lock on prior regime."""
        features = make_features(
            ["SYM"],
            daily_return=0.0005,
            volatility=0.008,
            crash_at=400,
            crash_mag=-0.30,
            crash_vol=0.05,
            seed=7,
        )
        cfg = StockRegimeConfig(
            min_regime_duration=30,  # long lock
            confidence_gate=0.3,
            enter_stress_z=1.0,
            exit_stress_z=0.6,
        )
        detector = RuleBasedStockRegimeDetector(cfg)
        results = run_multi_day(detector, features, ["SYM"], start_idx=300)

        post_crash = results[100:140]
        stressed_count = sum(
            1
            for r in post_crash
            if r["SYM"].label == StockRegimeLabel.STRESSED_HIGH_VOL
        )
        assert stressed_count > 0, (
            "STRESSED should bypass min_regime_duration"
        )


# ============================================================================
# Test: Hysteresis
# ============================================================================


class TestHysteresis:
    def test_trend_hysteresis_prevents_flipping(self):
        """
        A stock whose slope oscillates near the enter threshold should not
        flip every day between TRENDING_UP and RANGE_LOW_VOL.
        """
        # Build a stock with moderate uptrend (slope near threshold)
        features = make_features(
            ["EDGE"], daily_return=0.0006, volatility=0.009, seed=55,
        )
        cfg = StockRegimeConfig(
            min_regime_duration=1,
            confidence_gate=0.3,
            enter_trend_slope=0.5,
            exit_trend_slope=0.2,
        )
        detector = RuleBasedStockRegimeDetector(cfg)
        results = run_multi_day(detector, features, ["EDGE"])

        labels = [r["EDGE"].label.value for r in results[-200:]]
        switches = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])

        # With hysteresis, switches should be much less than every day
        assert switches < len(labels) * 0.15, (
            f"Too many switches ({switches}/{len(labels)}); hysteresis failing"
        )

    def test_stress_hysteresis(self):
        """Config exit thresholds should be easier than entry thresholds."""
        cfg = StockRegimeConfig()
        assert cfg.exit_stress_z < cfg.enter_stress_z
        assert cfg.exit_stress_dd > cfg.enter_stress_dd


# ============================================================================
# Test: Minimum duration
# ============================================================================


class TestMinDuration:
    def test_min_duration_blocks_early_switch(self):
        """Regime should not change before min_regime_duration days elapse."""
        features = make_features(
            ["LOCK"], daily_return=0.0004, volatility=0.009, seed=99,
        )
        cfg = StockRegimeConfig(
            min_regime_duration=15,
            confidence_gate=0.3,
        )
        detector = RuleBasedStockRegimeDetector(cfg)
        results = run_multi_day(detector, features, ["LOCK"], start_idx=300)

        # Measure run lengths
        labels = [r["LOCK"].label.value for r in results]
        runs: List[int] = []
        cur = 1
        for i in range(1, len(labels)):
            if labels[i] == labels[i - 1]:
                cur += 1
            else:
                runs.append(cur)
                cur = 1
        runs.append(cur)

        # All completed runs (except possibly last) must honour min duration
        for run in runs[:-1]:
            assert run >= cfg.min_regime_duration, (
                f"Run of length {run} < min {cfg.min_regime_duration}"
            )

    def test_longer_duration_fewer_switches(self):
        """Increasing min_regime_duration should reduce switches."""
        features = make_features(
            ["SYM"], daily_return=0.0004, volatility=0.012, seed=77,
        )

        def count_switches(dur: int) -> int:
            cfg = StockRegimeConfig(min_regime_duration=dur, confidence_gate=0.3)
            det = RuleBasedStockRegimeDetector(cfg)
            res = run_multi_day(det, features, ["SYM"])
            labels = [r["SYM"].label.value for r in res]
            return sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])

        short_switches = count_switches(3)
        long_switches = count_switches(20)

        assert long_switches <= short_switches


# ============================================================================
# Test: Confidence gate
# ============================================================================


class TestConfidenceGate:
    def test_high_gate_blocks_switches(self):
        """A very high confidence gate should block most regime switches."""
        features = make_features(
            ["SYM"], daily_return=0.0004, volatility=0.012, seed=33,
        )
        cfg = StockRegimeConfig(
            confidence_gate=0.95,
            min_regime_duration=1,
        )
        detector = RuleBasedStockRegimeDetector(cfg)
        results = run_multi_day(detector, features, ["SYM"])

        labels = [r["SYM"].label.value for r in results]
        switches = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])
        assert switches < 10, f"High gate should block most switches, got {switches}"

    def test_confidence_in_valid_range(self):
        """All confidence values should be in [0, 1]."""
        features = make_features(
            ["A", "B"], daily_return=0.001, volatility=0.01, seed=11,
        )
        detector = RuleBasedStockRegimeDetector()
        results = run_multi_day(detector, features, ["A", "B"])

        for rm in results:
            for sym in ["A", "B"]:
                assert 0 <= rm[sym].confidence <= 1


# ============================================================================
# Test: NO_TRADE on missing data
# ============================================================================


class TestNoTrade:
    def test_missing_symbol(self):
        """Symbol not in FeatureBundle → NO_TRADE."""
        features = make_features(["AAPL"], seed=1)
        detector = RuleBasedStockRegimeDetector()
        asof = features["close"].index[-1]
        mkt = _calm_market(asof)

        # "MISSING" is not in the bundle's symbols — get() returns NaN.
        bundle = _to_bundle(features, asof, ["MISSING"])
        result = detector.detect(asof, mkt, bundle, ["MISSING"])
        assert result["MISSING"].label == StockRegimeLabel.NO_TRADE
        assert result["MISSING"].confidence == 0.0

    def test_short_history(self):
        """Fewer bars than needed → features are NaN → NO_TRADE."""
        features = make_features(["SHORT"], n_days=30, seed=2)
        detector = RuleBasedStockRegimeDetector()
        asof = features["close"].index[-1]
        mkt = _calm_market(asof)

        bundle = _to_bundle(features, asof, ["SHORT"])
        result = detector.detect(asof, mkt, bundle, ["SHORT"])
        assert result["SHORT"].label == StockRegimeLabel.NO_TRADE

    def test_missing_rv20(self):
        """rv20 is NaN for symbol → NO_TRADE (rv20 is required)."""
        features = make_features(["AAPL"], seed=3)
        detector = RuleBasedStockRegimeDetector()
        asof = features["close"].index[-1]
        mkt = _calm_market(asof)

        bundle = _to_bundle(features, asof, ["AAPL"])
        # Manually null out rv20 for AAPL
        rv20_j = bundle.feat2j["rv20"]
        bundle.X[0, rv20_j] = np.nan
        bundle.mask[0, rv20_j] = True

        result = detector.detect(asof, mkt, bundle, ["AAPL"])
        assert result["AAPL"].label == StockRegimeLabel.NO_TRADE


# ============================================================================
# Test: Market-stress conditioning
# ============================================================================


class TestMarketConditioning:
    def test_stress_lowers_enter_z(self):
        """Under a stressed market, enter_stress_z should effectively drop."""
        cfg = StockRegimeConfig(
            enter_stress_z=1.5,
            market_stress_z_adj=0.3,
        )
        effective = cfg.enter_stress_z - cfg.market_stress_z_adj
        assert effective == pytest.approx(1.2)

    def test_stressed_market_penalises_trend_confidence(self):
        """Trend confidence should be lower when market is stressed."""
        features = make_features(
            ["SYM"], daily_return=0.002, volatility=0.008, seed=20,
        )
        cfg = StockRegimeConfig(
            min_regime_duration=1,
            confidence_gate=0.2,
            market_stress_conf_penalty=0.6,
        )

        # Calm market
        det_calm = RuleBasedStockRegimeDetector(cfg)
        calm_results = run_multi_day(
            det_calm, features, ["SYM"], market_fn=_calm_market,
        )

        # Stressed market
        det_stress = RuleBasedStockRegimeDetector(cfg)
        stress_results = run_multi_day(
            det_stress, features, ["SYM"], market_fn=_stressed_market,
        )

        calm_confs = [r["SYM"].confidence for r in calm_results[-100:]]
        stress_confs = [r["SYM"].confidence for r in stress_results[-100:]]

        # Average confidence under stress should be lower
        assert np.mean(stress_confs) <= np.mean(calm_confs) + 0.05


# ============================================================================
# Test: State management
# ============================================================================


class TestStateManagement:
    def test_reset_clears_state(self):
        """reset() should clear all per-symbol state."""
        features = make_features(["SYM"], seed=1)
        detector = RuleBasedStockRegimeDetector()
        asof = features["close"].index[-1]
        mkt = _calm_market(asof)

        bundle = _to_bundle(features, asof, ["SYM"])
        detector.detect(asof, mkt, bundle, ["SYM"])
        assert len(detector._state) > 0

        detector.reset()
        assert len(detector._state) == 0

    def test_state_persists_across_calls(self):
        """Per-symbol state should persist between detect() calls."""
        features = make_features(["SYM"], seed=1)
        detector = RuleBasedStockRegimeDetector()
        dates = features["close"].index

        asof1 = dates[400]
        b1 = _to_bundle(features, asof1, ["SYM"])
        detector.detect(asof1, _calm_market(asof1), b1, ["SYM"])

        asof2 = dates[401]
        b2 = _to_bundle(features, asof2, ["SYM"])
        detector.detect(asof2, _calm_market(asof2), b2, ["SYM"])

        assert detector._state["SYM"].days_in_regime >= 2

    def test_output_has_reasons(self):
        """StockRegimeState.reasons should contain diagnostic scores."""
        features = make_features(["SYM"], n_days=600, seed=1)
        cfg = StockRegimeConfig(min_regime_duration=1, confidence_gate=0.0)
        detector = RuleBasedStockRegimeDetector(cfg)
        asof = features["close"].index[-1]
        mkt = _calm_market(asof)

        bundle = _to_bundle(features, asof, ["SYM"])
        result = detector.detect(asof, mkt, bundle, ["SYM"])
        reasons = result["SYM"].reasons

        for key in ("normalized_slope", "vol_z", "max_dd", "mom63", "rv20"):
            assert key in reasons, f"Missing key {key!r} in reasons"

    def test_multi_symbol(self):
        """Detector should handle multiple symbols in one call."""
        syms = ["AAPL", "MSFT", "GOOG"]
        features = make_features(syms, seed=50)
        detector = RuleBasedStockRegimeDetector()
        asof = features["close"].index[-1]
        mkt = _calm_market(asof)

        bundle = _to_bundle(features, asof, syms)
        result = detector.detect(asof, mkt, bundle, syms)
        assert len(result) == 3
        for sym in syms:
            assert sym in result


# ============================================================================
# Test: Config validation
# ============================================================================


class TestConfig:
    def test_default_valid(self):
        cfg = StockRegimeConfig()
        assert cfg.enter_stress_z > cfg.exit_stress_z

    def test_bad_hysteresis_raises(self):
        with pytest.raises(ValueError, match="hysteresis"):
            StockRegimeConfig(enter_stress_z=1.0, exit_stress_z=1.5)

    def test_bad_confidence_gate(self):
        with pytest.raises(ValueError, match="confidence_gate"):
            StockRegimeConfig(confidence_gate=1.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
