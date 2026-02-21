"""
Tests for engine.features.pipeline — OHLCV → FeatureEngine → FeatureBundle.

Uses a temporary directory with synthetic CSVs so tests don't depend on
real market data being present.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.features.definitions import FeatureSpec
from engine.features.pipeline import (
    FeaturePipeline,
    align_dates,
    iter_bars,
    load_ohlcv,
    load_universe_ohlcv,
)


# ============================================================================
# Fixtures: synthetic OHLCV CSVs in a temporary directory
# ============================================================================


def _write_csv(path: Path, symbol: str, n: int = 100, start: str = "2023-01-02"):
    """Write a synthetic daily OHLCV CSV with linearly increasing prices."""
    dates = pd.bdate_range(start=start, periods=n)
    close = 100.0 + np.arange(n, dtype=float)
    df = pd.DataFrame(
        {
            "timestamp": dates.strftime("%Y-%m-%d"),
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 1_000_000),
        }
    )
    path.mkdir(parents=True, exist_ok=True)
    df.to_csv(path / f"{symbol}_1d.csv", index=False)
    return dates, close


@pytest.fixture()
def data_dir(tmp_path: Path):
    """Create a temp directory with 3 synthetic symbol CSVs."""
    for sym in ("AAA", "BBB", "CCC"):
        _write_csv(tmp_path, sym)
    return tmp_path


@pytest.fixture()
def staggered_data_dir(tmp_path: Path):
    """Symbols with different start dates (partial overlap)."""
    _write_csv(tmp_path, "EARLY", n=120, start="2023-01-02")
    _write_csv(tmp_path, "LATE", n=60, start="2023-04-03")
    return tmp_path


# ============================================================================
# Tests: load_ohlcv
# ============================================================================


class TestLoadOhlcv:
    def test_load_csv(self, data_dir: Path):
        df = load_ohlcv("AAA", data_dir=data_dir)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert "close" in df.columns
        assert "volume" in df.columns
        assert len(df) == 100

    def test_date_filter_start(self, data_dir: Path):
        df = load_ohlcv("AAA", data_dir=data_dir, start="2023-03-01")
        assert df.index[0] >= pd.Timestamp("2023-03-01")

    def test_date_filter_end(self, data_dir: Path):
        df = load_ohlcv("AAA", data_dir=data_dir, end="2023-02-28")
        assert df.index[-1] <= pd.Timestamp("2023-02-28")

    def test_missing_symbol_raises(self, data_dir: Path):
        with pytest.raises(FileNotFoundError, match="No OHLCV file"):
            load_ohlcv("NONEXISTENT", data_dir=data_dir)

    def test_sorted_index(self, data_dir: Path):
        df = load_ohlcv("AAA", data_dir=data_dir)
        assert df.index.is_monotonic_increasing

    def test_parquet_preferred(self, data_dir: Path):
        """If both CSV and Parquet exist, Parquet wins."""
        df_csv = load_ohlcv("AAA", data_dir=data_dir)
        # Write a Parquet with different values
        df_parquet = df_csv.copy()
        df_parquet["close"] = 999.0
        df_parquet.to_parquet(data_dir / "AAA_1d.parquet")

        df_loaded = load_ohlcv("AAA", data_dir=data_dir)
        assert (df_loaded["close"] == 999.0).all()


# ============================================================================
# Tests: load_universe_ohlcv
# ============================================================================


class TestLoadUniverseOhlcv:
    def test_loads_all_available(self, data_dir: Path):
        ohlcv = load_universe_ohlcv(["AAA", "BBB", "CCC"], data_dir=data_dir)
        assert set(ohlcv.keys()) == {"AAA", "BBB", "CCC"}

    def test_skips_missing_by_default(self, data_dir: Path):
        ohlcv = load_universe_ohlcv(
            ["AAA", "MISSING"], data_dir=data_dir,
        )
        assert "AAA" in ohlcv
        assert "MISSING" not in ohlcv

    def test_strict_raises_on_missing(self, data_dir: Path):
        with pytest.raises(FileNotFoundError):
            load_universe_ohlcv(
                ["AAA", "MISSING"], data_dir=data_dir, strict=True,
            )


# ============================================================================
# Tests: align_dates
# ============================================================================


class TestAlignDates:
    def test_union_of_dates(self, staggered_data_dir: Path):
        ohlcv = load_universe_ohlcv(
            ["EARLY", "LATE"], data_dir=staggered_data_dir,
        )
        dates = align_dates(ohlcv)
        # Union should start at EARLY's first date
        assert dates[0] == ohlcv["EARLY"].index[0]
        # Union should include LATE's last date
        assert ohlcv["LATE"].index[-1] in dates

    def test_sorted_and_unique(self, staggered_data_dir: Path):
        ohlcv = load_universe_ohlcv(
            ["EARLY", "LATE"], data_dir=staggered_data_dir,
        )
        dates = align_dates(ohlcv)
        assert dates.is_monotonic_increasing
        assert dates.is_unique

    def test_empty_dict(self):
        dates = align_dates({})
        assert len(dates) == 0


# ============================================================================
# Tests: iter_bars
# ============================================================================


class TestIterBars:
    def test_yields_correct_fields(self, data_dir: Path):
        ohlcv = load_universe_ohlcv(["AAA"], data_dir=data_dir)
        bar = next(iter_bars(ohlcv))
        assert bar["symbol"] == "AAA"
        assert "asof" in bar
        assert "close" in bar
        assert "volume" in bar
        assert "open" in bar
        assert "high" in bar
        assert "low" in bar

    def test_chronological_order(self, data_dir: Path):
        ohlcv = load_universe_ohlcv(["AAA", "BBB"], data_dir=data_dir)
        bars = list(iter_bars(ohlcv))
        dates = [b["asof"] for b in bars]
        # Should be non-decreasing (multiple symbols per date)
        for i in range(1, len(dates)):
            assert dates[i] >= dates[i - 1]

    def test_all_symbols_per_date(self, data_dir: Path):
        ohlcv = load_universe_ohlcv(["AAA", "BBB"], data_dir=data_dir)
        bars = list(iter_bars(ohlcv))
        # Since AAA and BBB have the same dates, every date has 2 bars
        from collections import Counter
        counts = Counter(b["asof"] for b in bars)
        assert all(c == 2 for c in counts.values())

    def test_staggered_symbols(self, staggered_data_dir: Path):
        ohlcv = load_universe_ohlcv(
            ["EARLY", "LATE"], data_dir=staggered_data_dir,
        )
        bars = list(iter_bars(ohlcv))
        # Early dates should only have EARLY
        early_only_date = ohlcv["EARLY"].index[0]
        bars_on_early = [b for b in bars if b["asof"] == early_only_date]
        assert len(bars_on_early) == 1
        assert bars_on_early[0]["symbol"] == "EARLY"

    def test_total_bar_count(self, data_dir: Path):
        ohlcv = load_universe_ohlcv(["AAA"], data_dir=data_dir)
        bars = list(iter_bars(ohlcv))
        assert len(bars) == len(ohlcv["AAA"])


# ============================================================================
# Tests: FeaturePipeline
# ============================================================================


# A small spec set that has a short lookback so tests run quickly
_QUICK_SPECS = [
    FeatureSpec(name="ret1", lookback=1, inputs=["close"],
                description="daily return"),
    FeatureSpec(name="sma20", lookback=20, inputs=["close"],
                description="20-day SMA"),
    FeatureSpec(name="rv20", lookback=20, inputs=["close", "ret"],
                description="20-day realised vol"),
    FeatureSpec(name="adv20", lookback=20, inputs=["close", "volume"],
                description="20-day avg dollar volume"),
]


class TestFeaturePipeline:
    def test_run_yields_bundles(self, data_dir: Path):
        pipe = FeaturePipeline(
            ["AAA", "BBB"],
            feature_specs=_QUICK_SPECS,
            data_dir=data_dir,
        )
        results = list(pipe.run())
        assert len(results) > 0
        for asof, bundle in results:
            assert isinstance(asof, pd.Timestamp)
            assert bundle.X.shape == (2, len(_QUICK_SPECS))
            assert bundle.X.dtype == np.float32

    def test_bundle_symbols_match_loaded(self, data_dir: Path):
        pipe = FeaturePipeline(
            ["AAA", "BBB"],
            feature_specs=_QUICK_SPECS,
            data_dir=data_dir,
        )
        results = list(pipe.run())
        _, bundle = results[-1]
        assert set(bundle.symbols) == {"AAA", "BBB"}

    def test_features_are_finite_when_ready(self, data_dir: Path):
        pipe = FeaturePipeline(
            ["AAA"],
            feature_specs=_QUICK_SPECS,
            data_dir=data_dir,
        )
        results = list(pipe.run())
        _, last_bundle = results[-1]
        # After 100 bars with lookback 20, sma20 and ret1 should be finite
        assert math.isfinite(last_bundle.get("AAA", "ret1"))
        assert math.isfinite(last_bundle.get("AAA", "sma20"))

    def test_sma20_value(self, data_dir: Path):
        """Verify sma20 matches hand-computed value from synthetic data."""
        pipe = FeaturePipeline(
            ["AAA"],
            feature_specs=_QUICK_SPECS,
            data_dir=data_dir,
        )
        results = list(pipe.run())
        _, last_bundle = results[-1]
        sma = last_bundle.get("AAA", "sma20")
        # Last 20 closes: 180..199 → mean = 189.5
        assert sma == pytest.approx(189.5)

    def test_emit_frequency(self, data_dir: Path):
        pipe = FeaturePipeline(
            ["AAA"],
            feature_specs=_QUICK_SPECS,
            data_dir=data_dir,
        )
        results_1 = list(pipe.run(emit_frequency=1))
        pipe2 = FeaturePipeline(
            ["AAA"],
            feature_specs=_QUICK_SPECS,
            data_dir=data_dir,
        )
        results_5 = list(pipe2.run(emit_frequency=5))
        # emit_frequency=5 yields ~1/5 of the bundles
        assert len(results_5) < len(results_1)
        assert len(results_5) == pytest.approx(len(results_1) / 5, abs=1)

    def test_warmup_mode(self, data_dir: Path):
        pipe = FeaturePipeline(
            ["AAA"],
            feature_specs=_QUICK_SPECS,
            data_dir=data_dir,
        )
        results = list(pipe.run(warmup_only=True))
        assert len(results) == 0
        # But the engine should be warmed up
        assert pipe.engine is not None
        assert pipe.engine.bars_received["AAA"] == 100

    def test_build_latest(self, data_dir: Path):
        pipe = FeaturePipeline(
            ["AAA"],
            feature_specs=_QUICK_SPECS,
            data_dir=data_dir,
        )
        result = pipe.build_latest()
        assert result is not None
        asof, bundle = result
        assert isinstance(asof, pd.Timestamp)
        assert math.isfinite(bundle.get("AAA", "sma20"))

    def test_warmup_method(self, data_dir: Path):
        pipe = FeaturePipeline(
            ["AAA"],
            feature_specs=_QUICK_SPECS,
            data_dir=data_dir,
        )
        engine = pipe.warmup()
        assert engine.bars_received["AAA"] == 100

    def test_missing_symbol_graceful(self, data_dir: Path):
        """Pipeline skips symbols without data (non-strict)."""
        pipe = FeaturePipeline(
            ["AAA", "MISSING"],
            feature_specs=_QUICK_SPECS,
            data_dir=data_dir,
        )
        results = list(pipe.run())
        # Should still produce bundles, just with only AAA
        assert len(results) > 0
        _, bundle = results[-1]
        assert "AAA" in bundle.symbols

    def test_date_range_filter(self, data_dir: Path):
        pipe = FeaturePipeline(
            ["AAA"],
            feature_specs=_QUICK_SPECS,
            data_dir=data_dir,
        )
        results = list(pipe.run(start="2023-03-01", end="2023-04-30"))
        for asof, bundle in results:
            assert asof >= pd.Timestamp("2023-03-01")
            assert asof <= pd.Timestamp("2023-04-30")

    def test_staggered_symbols(self, staggered_data_dir: Path):
        """Symbols with different date ranges are handled correctly."""
        pipe = FeaturePipeline(
            ["EARLY", "LATE"],
            feature_specs=_QUICK_SPECS,
            data_dir=staggered_data_dir,
        )
        results = list(pipe.run())
        assert len(results) > 0

    def test_ohlcv_property(self, data_dir: Path):
        pipe = FeaturePipeline(
            ["AAA"],
            feature_specs=_QUICK_SPECS,
            data_dir=data_dir,
        )
        assert pipe.ohlcv is None  # before run
        list(pipe.run())
        assert pipe.ohlcv is not None
        assert "AAA" in pipe.ohlcv

    def test_repr(self, data_dir: Path):
        pipe = FeaturePipeline(
            ["AAA", "BBB"],
            feature_specs=_QUICK_SPECS,
            data_dir=data_dir,
        )
        r = repr(pipe)
        assert "FeaturePipeline" in r
        assert "universe=2" in r


# ============================================================================
# Test: with real AAPL data (skipped if file doesn't exist)
# ============================================================================

_AAPL_CSV = Path(__file__).resolve().parents[1] / "engine" / "data" / "raw" / "stocks" / "AAPL_1d.csv"


@pytest.mark.skipif(not _AAPL_CSV.exists(), reason="AAPL_1d.csv not present")
class TestWithRealData:
    def test_load_aapl(self):
        df = load_ohlcv(
            "AAPL",
            data_dir=_AAPL_CSV.parent,
        )
        assert len(df) > 200
        assert "close" in df.columns
        assert df.index.is_monotonic_increasing

    def test_pipeline_aapl(self):
        pipe = FeaturePipeline(
            ["AAPL"],
            data_dir=_AAPL_CSV.parent,
        )
        result = pipe.build_latest()
        if result is not None:
            asof, bundle = result
            assert math.isfinite(bundle.get("AAPL", "ret1"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
