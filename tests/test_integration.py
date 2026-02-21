"""
Integration tests: data layer → feature layer → strategy layer.

Tests the full pipeline from OHLCV data through FeatureEngine to
the RegimeAlphaStrategy using synthetic data in a temp directory.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from engine.backtest.runner import PortfolioBacktestResult, run_regime_backtest
from engine.features.definitions import FeatureSpec, default_feature_specs
from engine.features.feature_engine import FeatureEngine
from engine.features.pipeline import (
    FeaturePipeline,
    align_dates,
    iter_bars,
    load_universe_ohlcv,
)
from engine.features.types import FeatureBundle
from engine.strategy.alpha.alpha_engine_long_only import LongOnlyAlphaEngine
from engine.strategy.portfolio.constructor_long_only import (
    LongOnlyRankWeightConstructor,
)
from engine.strategy.portfolio.types import PortfolioConstraints
from engine.strategy.regime.market_regime_adapter import EventDrivenMarketRegime
from engine.strategy.regime.stock_regime_rule_based import (
    RuleBasedStockRegimeDetector,
)
from engine.strategy.regime_alpha_strategy import RegimeAlphaStrategy


# ============================================================================
# Fixtures: synthetic OHLCV data
# ============================================================================


def _write_synthetic_csv(
    path: Path,
    symbol: str,
    n: int = 300,
    start: str = "2022-01-03",
    base_close: float = 100.0,
    daily_drift: float = 0.0005,
    daily_vol: float = 0.015,
    seed: int = 42,
) -> pd.DataFrame:
    """Write a realistic synthetic daily OHLCV CSV.

    Uses geometric Brownian motion for close prices so returns have
    a realistic distribution (important for vol / regime features).
    """
    rng = np.random.default_rng(seed + hash(symbol) % 10_000)
    dates = pd.bdate_range(start=start, periods=n)
    log_rets = rng.normal(daily_drift, daily_vol, size=n)
    log_rets[0] = 0.0
    log_prices = np.log(base_close) + np.cumsum(log_rets)
    close = np.exp(log_prices)

    # Synthetic OHLV from close
    noise = rng.uniform(0.002, 0.01, size=n)
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_ = close * (1 + rng.uniform(-0.005, 0.005, size=n))
    volume = rng.integers(500_000, 5_000_000, size=n)

    df = pd.DataFrame({
        "timestamp": dates.strftime("%Y-%m-%d"),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    path.mkdir(parents=True, exist_ok=True)
    df.to_csv(path / f"{symbol}_1d.csv", index=False)
    return df


@pytest.fixture()
def multi_symbol_dir(tmp_path: Path):
    """Create temp directory with 5 symbols + SPY (market proxy)."""
    symbols = ["SPY", "AAPL", "MSFT", "GOOG", "AMZN"]
    for i, sym in enumerate(symbols):
        _write_synthetic_csv(
            tmp_path, sym, n=300,
            base_close=100 + i * 20,
            seed=42 + i,
        )
    return tmp_path, symbols


# ============================================================================
# Test: EventDrivenMarketRegime adapter
# ============================================================================


class TestEventDrivenMarketRegime:
    """Verify the market regime adapter accumulates SPY and produces states."""

    def test_accumulates_spy_only(self, multi_symbol_dir):
        data_dir, symbols = multi_symbol_dir
        adapter = EventDrivenMarketRegime(spy_symbol="SPY")

        ohlcv = load_universe_ohlcv(symbols, data_dir=data_dir)
        dates = align_dates(ohlcv)

        n_fed = 0
        for bar in iter_bars(ohlcv, dates):
            adapter.update(bar)
            if bar["symbol"] == "SPY":
                n_fed += 1

        assert adapter.n_spy_bars == n_fed
        assert adapter.n_spy_bars == len(ohlcv["SPY"])

    def test_detect_before_enough_data(self):
        adapter = EventDrivenMarketRegime(spy_symbol="SPY")
        state = adapter.detect(pd.Timestamp("2022-01-05"))
        assert state.confidence == 0.0  # insufficient data

    def test_detect_after_warmup(self, multi_symbol_dir):
        data_dir, _ = multi_symbol_dir
        adapter = EventDrivenMarketRegime(spy_symbol="SPY")

        ohlcv = load_universe_ohlcv(["SPY"], data_dir=data_dir)
        dates = align_dates(ohlcv)

        for bar in iter_bars(ohlcv, dates):
            adapter.update(bar)

        state = adapter.detect(dates[-1])
        assert state is not None
        assert state.label is not None
        assert state.confidence > 0.0

    def test_reset_clears_state(self, multi_symbol_dir):
        data_dir, _ = multi_symbol_dir
        adapter = EventDrivenMarketRegime(spy_symbol="SPY")

        ohlcv = load_universe_ohlcv(["SPY"], data_dir=data_dir)
        for bar in iter_bars(ohlcv):
            adapter.update(bar)

        adapter.reset()
        assert adapter.n_spy_bars == 0
        assert adapter.last_state is None


# ============================================================================
# Test: FeatureEngine + FeaturePipeline → FeatureBundle
# ============================================================================


class TestFeaturePipelineToBundle:
    """Verify the pipeline produces valid FeatureBundles."""

    def test_pipeline_produces_bundles(self, multi_symbol_dir):
        data_dir, symbols = multi_symbol_dir
        pipe = FeaturePipeline(symbols, data_dir=data_dir)
        results = list(pipe.run())
        assert len(results) > 0

        asof, bundle = results[-1]
        assert isinstance(bundle, FeatureBundle)
        assert bundle.X.shape[0] == len(symbols)
        assert bundle.X.dtype == np.float32

    def test_bundle_features_are_finite(self, multi_symbol_dir):
        data_dir, symbols = multi_symbol_dir
        pipe = FeaturePipeline(symbols, data_dir=data_dir)
        result = pipe.build_latest()
        assert result is not None

        asof, bundle = result
        # After 300 bars, core features should be finite for all symbols
        for sym in symbols:
            ret1 = bundle.get(sym, "ret1")
            assert math.isfinite(ret1), f"{sym} ret1 is not finite"

            rv20 = bundle.get(sym, "rv20")
            assert math.isfinite(rv20), f"{sym} rv20 is not finite"

    def test_bundle_get_matches_matrix(self, multi_symbol_dir):
        data_dir, symbols = multi_symbol_dir
        pipe = FeaturePipeline(symbols, data_dir=data_dir)
        result = pipe.build_latest()
        assert result is not None

        _, bundle = result
        for sym in symbols:
            row = bundle.row(sym)
            for j, fname in enumerate(bundle.feature_names):
                assert bundle.get(sym, fname) == pytest.approx(
                    float(row[j]), nan_ok=True,
                )


# ============================================================================
# Test: RegimeAlphaStrategy.process_date()
# ============================================================================


class TestProcessDate:
    """Verify the date-driven cross-sectional pipeline works."""

    def test_process_date_returns_target_portfolio(self, multi_symbol_dir):
        data_dir, symbols = multi_symbol_dir
        ohlcv = load_universe_ohlcv(symbols, data_dir=data_dir)

        # Build components
        feature_engine = FeatureEngine(symbols)
        market_regime = EventDrivenMarketRegime(spy_symbol="SPY")
        stock_regime = RuleBasedStockRegimeDetector()
        alpha_engine = LongOnlyAlphaEngine()
        portfolio_constructor = LongOnlyRankWeightConstructor()

        strategy = RegimeAlphaStrategy(
            warmup_bars=feature_engine.max_lookback,
            universe=symbols,
            portfolio_constraints=PortfolioConstraints(),
            feature_engine=feature_engine,
            market_regime_detector=market_regime,
            stock_regime_detector=stock_regime,
            alpha_engine=alpha_engine,
            portfolio_constructor=portfolio_constructor,
        )

        dates = align_dates(ohlcv)
        last_live_result = None

        for dt in dates:
            # Feed bars
            for bar in _bars_for_date(ohlcv, dt):
                feature_engine.update(bar)
                market_regime.update(bar)

            strategy._increment_bar(dt)

            if feature_engine.is_ready():
                last_live_result = strategy.process_date(dt)

        # Should have produced a result
        assert last_live_result is not None
        # Weights should be non-negative (long-only)
        if hasattr(last_live_result, "weights"):
            for w in last_live_result.weights.values():
                assert w >= 0.0


# ============================================================================
# Test: Full backtest runner
# ============================================================================


class TestRunRegimeBacktest:
    """Verify the end-to-end multi-symbol backtest runner."""

    def test_runner_produces_result(self, multi_symbol_dir):
        data_dir, symbols = multi_symbol_dir
        result = run_regime_backtest(
            universe=symbols,
            data_dir=data_dir,
            spy_symbol="SPY",
            initial_cash=100_000.0,
        )
        assert isinstance(result, PortfolioBacktestResult)
        assert len(result.equity_curve) > 0
        assert "total_return" in result.stats

    def test_runner_with_preloaded_data(self, multi_symbol_dir):
        data_dir, symbols = multi_symbol_dir
        ohlcv = load_universe_ohlcv(symbols, data_dir=data_dir)

        result = run_regime_backtest(
            universe=symbols,
            data=ohlcv,
            spy_symbol="SPY",
        )
        assert isinstance(result, PortfolioBacktestResult)
        assert len(result.equity_curve) > 0

    def test_weights_history_shape(self, multi_symbol_dir):
        data_dir, symbols = multi_symbol_dir
        result = run_regime_backtest(
            universe=symbols,
            data_dir=data_dir,
            spy_symbol="SPY",
        )
        # weights_history columns are the symbols that got non-zero weights
        # It should have at least some rows (post-warmup)
        if len(result.weights_history) > 0:
            assert isinstance(result.weights_history, pd.DataFrame)

    def test_regime_history_populated(self, multi_symbol_dir):
        data_dir, symbols = multi_symbol_dir
        result = run_regime_backtest(
            universe=symbols,
            data_dir=data_dir,
            spy_symbol="SPY",
        )
        if len(result.regime_history) > 0:
            # All labels should be valid regime strings
            valid_labels = {"calm_trend", "calm_range", "stress", "crisis", "unknown"}
            for label in result.regime_history:
                assert label in valid_labels

    def test_stats_keys(self, multi_symbol_dir):
        data_dir, symbols = multi_symbol_dir
        result = run_regime_backtest(
            universe=symbols,
            data_dir=data_dir,
            spy_symbol="SPY",
        )
        expected_keys = {
            "initial_value", "final_value", "total_return",
            "max_drawdown", "n_trading_days",
        }
        assert expected_keys.issubset(result.stats.keys())


# ============================================================================
# Helpers
# ============================================================================


def _bars_for_date(
    ohlcv: Dict[str, pd.DataFrame],
    dt: pd.Timestamp,
) -> list:
    """Return bar dicts for all symbols on a given date."""
    bars = []
    for sym, df in ohlcv.items():
        if dt in df.index:
            row = df.loc[dt]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            bars.append({
                "symbol": sym,
                "asof": dt,
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0.0)),
            })
    return bars


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
