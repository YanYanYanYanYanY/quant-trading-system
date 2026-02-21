"""
Multi-symbol portfolio backtester.

Connects the full stack: **DataEngine** (data layer) ->
**FeatureEngine** (feature layer) -> **RegimeAlphaStrategy**
(strategy layer) into a single runnable backtest.

Data flow
---------
::

    DataEngine
         |  .sync() -- download missing / update stale
         |  .load_universe() -- dict[str, DataFrame]
         v
    align_dates + iter_bars
         |
         v
    Bar stream (chronological, all symbols per date)
         |
    +----+------------------------------+
    |  For each date:                   |
    |    1. Feed bars -> FeatureEngine  |
    |    2. Feed bars -> MarketRegime   |
    |    3. strategy.process_date()     |
    |    4. Mark-to-market portfolio    |
    +-----------------------------------+
         |
         v
    PortfolioBacktestResult
        equity_curve, weights_history, stats

Entry points
------------
- :func:`run_regime_backtest` -- programmatic API
- :func:`main` -- CLI / script entry point (``python -m engine.backtest.runner``)
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # reads .env from project root into os.environ

from ..data.data_engine import DataEngine
from ..features.definitions import FeatureSpec
from ..features.feature_engine import FeatureEngine
from ..features.pipeline import align_dates, iter_bars
from ..strategy.alpha.alpha_engine_long_only import (
    AlphaEngineConfig,
    LongOnlyAlphaEngine,
)
from ..strategy.portfolio.constructor_long_only import (
    LongOnlyRankWeightConstructor,
)
from ..strategy.portfolio.types import PortfolioConstraints, TargetPortfolio
from ..strategy.regime.config import RegimeConfig
from ..strategy.regime.market_regime_adapter import EventDrivenMarketRegime
from ..strategy.regime.stock_regime_config import StockRegimeConfig
from ..strategy.regime.stock_regime_rule_based import RuleBasedStockRegimeDetector
from ..strategy.regime_alpha_strategy import RegimeAlphaStrategy

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PortfolioBacktestResult:
    """Output of :func:`run_regime_backtest`.

    Attributes
    ----------
    equity_curve : pd.Series
        DatetimeIndex -> portfolio value (mark-to-market).
    weights_history : pd.DataFrame
        Rows = dates, columns = symbols, values = target weights.
    regime_history : pd.Series
        DatetimeIndex -> market regime label.
    stats : dict[str, float]
        Summary statistics (total return, max drawdown, etc.).
    """

    equity_curve: pd.Series
    weights_history: pd.DataFrame
    regime_history: pd.Series
    stats: Dict[str, float]


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def run_regime_backtest(
    universe: List[str] | None = None,
    *,
    data: Dict[str, pd.DataFrame] | None = None,
    data_engine: DataEngine | None = None,
    auto_sync: bool = False,
    spy_symbol: str = "SPY",
    initial_cash: float = 100_000.0,
    feature_specs: List[FeatureSpec] | None = None,
    regime_config: RegimeConfig | None = None,
    stock_regime_config: StockRegimeConfig | None = None,
    alpha_config: AlphaEngineConfig | None = None,
    portfolio_constraints: PortfolioConstraints | None = None,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
    data_dir: Path | None = None,
    timeframe: str = "1d",
    readiness_threshold: float = 0.80,
) -> PortfolioBacktestResult:
    """Run a multi-symbol portfolio backtest with the regime-alpha pipeline.

    This is the **primary entry point** for a full backtest.  It:

    1. Resolves data — via *data_engine*, pre-loaded *data*, or raw
       files on disk.  When *auto_sync* is ``True`` the DataEngine
       downloads any missing / stale OHLCV files first.
    2. Builds the full component stack (feature engine, regime
       detectors, alpha engine, portfolio constructor, strategy).
    3. Iterates bars date-by-date, feeding the feature engine and
       market regime adapter.
    4. After warmup, calls ``strategy.process_date(asof)`` to get
       target weights and tracks mark-to-market equity.
    5. Returns a :class:`PortfolioBacktestResult` with equity curve,
       weights history, regime labels, and summary statistics.

    Parameters
    ----------
    universe : list[str], optional
        Symbols to trade.  If ``None`` and *data_engine* is provided,
        the engine's full universe is used.
    data : dict[str, pd.DataFrame], optional
        Pre-loaded OHLCV DataFrames keyed by symbol.  Highest priority
        — if provided, *data_engine* and *data_dir* are ignored.
    data_engine : DataEngine, optional
        A :class:`DataEngine` instance.  Used to load (and optionally
        sync) OHLCV data.  If ``None``, one is created on the fly when
        *data* is also ``None``.
    auto_sync : bool
        If ``True`` and a *data_engine* is available, call
        ``data_engine.sync()`` before loading data.  Requires a valid
        Polygon API key.
    spy_symbol : str
        Market proxy symbol for regime detection (default ``"SPY"``).
    initial_cash : float
        Starting portfolio value.
    feature_specs : list[FeatureSpec], optional
        Feature catalogue (defaults to :func:`default_feature_specs`).
    regime_config : RegimeConfig, optional
        Market regime detector config.
    stock_regime_config : StockRegimeConfig, optional
        Stock regime detector config.
    alpha_config : AlphaEngineConfig, optional
        Alpha engine config (ensemble weights, router, etc.).
    portfolio_constraints : PortfolioConstraints, optional
        Position count, weight caps, turnover limits.
    start, end : str or pd.Timestamp, optional
        Date range filter (inclusive).
    data_dir : Path, optional
        Override default OHLCV data directory.
    timeframe : str
        Bar resolution (default ``"1d"``).
    readiness_threshold : float
        Fraction of universe that must be warm before trading.

    Returns
    -------
    PortfolioBacktestResult
    """
    # ── 1. Resolve data ───────────────────────────────────────────────
    #   Always include spy_symbol in the load so that the market regime
    #   adapter receives bars even when the user trades a small subset.
    load_universe: List[str] | None = None
    if universe is not None:
        load_universe = list(dict.fromkeys(universe + [spy_symbol]))
    ohlcv = _resolve_data(
        universe=load_universe,
        data=data,
        data_engine=data_engine,
        auto_sync=auto_sync,
        data_dir=data_dir,
        timeframe=timeframe,
        start=start,
        end=end,
    )

    if not ohlcv:
        raise ValueError("No OHLCV data available for the given universe")

    # Trading universe = only the symbols the user requested (or all
    # loaded except SPY when universe was not specified).
    if universe is not None:
        loaded_symbols = [s for s in universe if s in ohlcv]
    else:
        loaded_symbols = [s for s in ohlcv if s != spy_symbol]

    if not loaded_symbols:
        raise ValueError(
            "No OHLCV data available for the trading universe. "
            "Check that data files exist on disk or run with --sync."
        )

    has_spy = spy_symbol in ohlcv
    log.info(
        "Loaded %d tradeable symbols + SPY=%s: %s",
        len(loaded_symbols), has_spy, loaded_symbols[:10],
    )
    if not has_spy:
        log.warning(
            "%s data not found — market regime detection will use defaults. "
            "Add %s to your data directory or run with --sync.",
            spy_symbol, spy_symbol,
        )

    # ── 2. Build component stack ──────────────────────────────────────
    feature_engine = FeatureEngine(
        universe=loaded_symbols,
        feature_specs=feature_specs,
        readiness_threshold=readiness_threshold,
    )

    market_regime = EventDrivenMarketRegime(
        spy_symbol=spy_symbol,
        config=regime_config,
    )

    stock_regime = RuleBasedStockRegimeDetector(config=stock_regime_config)

    alpha_engine = LongOnlyAlphaEngine(config=alpha_config)

    portfolio_constructor = LongOnlyRankWeightConstructor()

    # Auto-scale constraints for the actual universe size when no
    # custom constraints were provided.  The defaults assume ~50 stocks;
    # with fewer symbols, max_weight must increase so that (a) the
    # portfolio can be fully invested, and (b) alpha scores still
    # differentiate positions (max_weight > 1/n gives room for tilts).
    if portfolio_constraints is not None:
        constraints = portfolio_constraints
    else:
        n = len(loaded_symbols)
        # Allow the top stock up to ~2× equal-weight, capped at 25%
        auto_max_weight = min(0.25, max(0.05, round(2.0 / max(n, 1), 4)))
        constraints = PortfolioConstraints(
            max_positions=min(50, n),
            min_positions=min(5, n),
            max_weight=auto_max_weight,
        )
        log.info(
            "Auto-scaled constraints: max_positions=%d, min_positions=%d, "
            "max_weight=%.2f%%",
            constraints.max_positions, constraints.min_positions,
            constraints.max_weight * 100,
        )

    strategy = RegimeAlphaStrategy(
        warmup_bars=feature_engine.warmup_bars,
        universe=loaded_symbols,
        portfolio_constraints=constraints,
        feature_engine=feature_engine,
        market_regime_detector=market_regime,
        stock_regime_detector=stock_regime,
        alpha_engine=alpha_engine,
        portfolio_constructor=portfolio_constructor,
    )

    # ── 3. Align dates ────────────────────────────────────────────────
    dates = align_dates(ohlcv)
    n_dates = len(dates)
    log.info(
        "Backtest: %s -> %s (%d trading days), warmup=%d",
        dates[0].date() if n_dates else "?",
        dates[-1].date() if n_dates else "?",
        n_dates,
        feature_engine.warmup_bars,
    )

    # ── 4. Pre-index bars by date for fast iteration ──────────────────
    bars_by_date: Dict[pd.Timestamp, List[dict]] = {}
    for bar in iter_bars(ohlcv, dates):
        dt = bar["asof"]
        bars_by_date.setdefault(dt, []).append(bar)

    # ── 5. Latest close prices for mark-to-market ─────────────────────
    latest_close: Dict[str, float] = {}
    prev_close: Dict[str, float] = {}

    # ── 6. Main loop ──────────────────────────────────────────────────
    equity_values: List[float] = []
    equity_dates: List[pd.Timestamp] = []
    weights_rows: List[Dict[str, float]] = []
    weights_dates: List[pd.Timestamp] = []
    regime_labels: List[str] = []
    regime_dates: List[pd.Timestamp] = []

    portfolio_value = float(initial_cash)
    current_weights: Dict[str, float] = {}

    for dt in dates:
        day_bars = bars_by_date.get(dt, [])

        # 6a. Feed all bars for this date to feature engine + market adapter
        prev_close = dict(latest_close)
        for bar in day_bars:
            feature_engine.update(bar)
            market_regime.update(bar)
            latest_close[bar["symbol"]] = float(bar["close"])

        # 6b. Increment strategy bar counter
        strategy._increment_bar(dt)

        # 6c. Mark-to-market using previous weights and today's returns
        if current_weights and prev_close:
            portfolio_value = _mark_to_market(
                current_weights, prev_close, latest_close, portfolio_value,
            )

        # 6d. Check if engine is ready
        if not feature_engine.is_ready():
            equity_values.append(portfolio_value)
            equity_dates.append(dt)
            continue

        # 6e. Run the cross-sectional pipeline
        try:
            result = strategy.process_date(dt)
        except Exception as exc:
            log.warning("process_date failed on %s: %s", dt.date(), exc)
            equity_values.append(portfolio_value)
            equity_dates.append(dt)
            continue

        # 6f. Extract target weights
        if isinstance(result, TargetPortfolio):
            new_weights = result.weights
        elif isinstance(result, dict):
            new_weights = result
        elif hasattr(result, "weights"):
            new_weights = result.weights
        else:
            new_weights = current_weights

        current_weights = dict(new_weights)

        # 6g. Record
        equity_values.append(portfolio_value)
        equity_dates.append(dt)

        weights_rows.append(dict(new_weights))
        weights_dates.append(dt)

        mkt_label = (
            market_regime.last_state.label.value
            if market_regime.last_state
            else "unknown"
        )
        regime_labels.append(mkt_label)
        regime_dates.append(dt)

    # ── 7. Build result ───────────────────────────────────────────────
    equity_curve = pd.Series(
        equity_values, index=pd.DatetimeIndex(equity_dates), name="equity",
    )

    weights_history = pd.DataFrame(
        weights_rows, index=pd.DatetimeIndex(weights_dates),
    ).fillna(0.0)

    regime_history = pd.Series(
        regime_labels, index=pd.DatetimeIndex(regime_dates), name="regime",
    )

    stats = _compute_stats(equity_curve, weights_history)

    log.info(
        "Backtest done: %d days, return=%.2f%%, maxDD=%.2f%%, sharpe=%.2f",
        stats.get("n_trading_days", 0),
        stats.get("total_return", 0) * 100,
        stats.get("max_drawdown", 0) * 100,
        stats.get("sharpe_ratio", 0),
    )

    return PortfolioBacktestResult(
        equity_curve=equity_curve,
        weights_history=weights_history,
        regime_history=regime_history,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Data resolution
# ---------------------------------------------------------------------------


def _resolve_data(
    universe: List[str] | None,
    data: Dict[str, pd.DataFrame] | None,
    data_engine: DataEngine | None,
    auto_sync: bool,
    data_dir: Path | None,
    timeframe: str,
    start: pd.Timestamp | str | None,
    end: pd.Timestamp | str | None,
) -> Dict[str, pd.DataFrame]:
    """Resolve OHLCV data from the best available source.

    Priority: *data* > *data_engine* > fallback DataEngine.
    """
    # Priority 1: pre-loaded DataFrames
    if data is not None:
        return dict(data)

    # Priority 2: explicit DataEngine
    if data_engine is not None:
        if auto_sync:
            log.info("Auto-syncing data via DataEngine...")
            report = data_engine.sync(symbols=universe)
            log.info("Sync: %s", report)
        ohlcv = data_engine.load_universe(symbols=universe)
        return _filter_dates(ohlcv, start, end)

    # Priority 3: create a DataEngine on the fly
    try:
        de = DataEngine(data_dir=data_dir)
        if auto_sync:
            log.info("Auto-syncing data via DataEngine...")
            report = de.sync(symbols=universe)
            log.info("Sync: %s", report)
        syms = universe or de.universe
        ohlcv = de.load_universe(symbols=syms)
        return _filter_dates(ohlcv, start, end)
    except FileNotFoundError:
        pass

    # Priority 4: raw load (no DataEngine available)
    if universe is not None:
        from ..features.pipeline import load_universe_ohlcv
        return load_universe_ohlcv(
            universe, timeframe, data_dir=data_dir, start=start, end=end,
        )

    raise ValueError(
        "Cannot resolve data: provide universe, data, or data_engine"
    )


def _filter_dates(
    ohlcv: Dict[str, pd.DataFrame],
    start: pd.Timestamp | str | None,
    end: pd.Timestamp | str | None,
) -> Dict[str, pd.DataFrame]:
    """Apply optional date filters to loaded DataFrames."""
    if start is None and end is None:
        return ohlcv

    filtered: Dict[str, pd.DataFrame] = {}
    for sym, df in ohlcv.items():
        d = df
        if start is not None:
            d = d.loc[pd.Timestamp(start):]
        if end is not None:
            d = d.loc[:pd.Timestamp(end)]
        if not d.empty:
            filtered[sym] = d
    return filtered


# ---------------------------------------------------------------------------
# Mark-to-market
# ---------------------------------------------------------------------------


def _mark_to_market(
    weights: Dict[str, float],
    prev_close: Dict[str, float],
    curr_close: Dict[str, float],
    prev_nav: float,
) -> float:
    """Compute new NAV from yesterday's weights and today's returns.

    Assumes the portfolio was rebalanced to *weights* at yesterday's
    close.  Today's NAV = prev_NAV * (1 + weighted portfolio return).
    """
    if not weights or prev_nav <= 0:
        return prev_nav

    port_ret = 0.0
    for sym, w in weights.items():
        if w <= 0:
            continue
        p0 = prev_close.get(sym)
        p1 = curr_close.get(sym)
        if p0 and p1 and p0 > 0:
            port_ret += w * (p1 / p0 - 1.0)

    return prev_nav * (1.0 + port_ret)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _compute_stats(
    equity: pd.Series,
    weights: pd.DataFrame,
) -> Dict[str, float]:
    """Compute summary statistics from the equity curve."""
    if len(equity) < 2:
        return {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "n_trading_days": len(equity),
            "n_rebalances": len(weights),
        }

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)

    # Max drawdown
    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = float(dd.min())

    # Annualised return and vol
    daily_rets = equity.pct_change().dropna()
    n_days = len(daily_rets)
    ann_ret = float(daily_rets.mean() * 252) if n_days > 0 else 0.0
    ann_vol = float(daily_rets.std() * np.sqrt(252)) if n_days > 1 else 0.0
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    # Average positions held
    avg_positions = float(
        (weights > 0).sum(axis=1).mean()
    ) if len(weights) > 0 else 0.0

    return {
        "initial_value": float(equity.iloc[0]),
        "final_value": float(equity.iloc[-1]),
        "total_return": total_return,
        "annualised_return": ann_ret,
        "annualised_vol": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "n_trading_days": n_days,
        "n_rebalances": len(weights),
        "avg_positions": avg_positions,
    }


# ---------------------------------------------------------------------------
# CLI / main entry point
# ---------------------------------------------------------------------------


def main(argv: List[str] | None = None) -> PortfolioBacktestResult:
    """Run a backtest from the command line.

    Examples
    --------
    ::

        # Use ticker file universe, sync missing data, run full backtest
        python -m engine.backtest.runner --sync

        # Specific symbols, date range
        python -m engine.backtest.runner --symbols AAPL MSFT GOOG --start 2024-01-01

        # Custom initial cash and readiness threshold
        python -m engine.backtest.runner --cash 500000 --readiness 0.6

    """
    parser = argparse.ArgumentParser(
        description="Run a multi-symbol regime-alpha portfolio backtest.",
    )
    parser.add_argument(
        "--symbols", nargs="*", default=None,
        help="Symbols to backtest (default: all from ticker file).",
    )
    parser.add_argument(
        "--sync", action="store_true",
        help="Download missing / update stale OHLCV data before running.",
    )
    parser.add_argument(
        "--spy", default="SPY",
        help="Market proxy symbol (default: SPY).",
    )
    parser.add_argument(
        "--cash", type=float, default=100_000.0,
        help="Initial cash (default: 100000).",
    )
    parser.add_argument(
        "--start", default=None,
        help="Backtest start date, e.g. 2023-01-01.",
    )
    parser.add_argument(
        "--end", default=None,
        help="Backtest end date, e.g. 2024-12-31.",
    )
    parser.add_argument(
        "--readiness", type=float, default=0.80,
        help="Fraction of universe that must be warm (default: 0.80).",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Override OHLCV data directory.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    args = parser.parse_args(argv)

    # ── Logging ───────────────────────────────────────────────────────
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)-30s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── DataEngine ────────────────────────────────────────────────────
    data_dir = Path(args.data_dir) if args.data_dir else None
    de = DataEngine(data_dir=data_dir)

    print(f"\n{'='*60}")
    print("Regime-Alpha Portfolio Backtest")
    print(f"{'='*60}")
    de.status()

    # ── Resolve universe ──────────────────────────────────────────────
    universe = args.symbols or de.universe

    # Check which tickers are missing data on disk BEFORE sync so we
    # can report what needs downloading.
    present_pre = set(de.present())
    missing = [s for s in universe if s not in present_pre]

    if missing:
        print(f"\n  Missing data for {len(missing)} ticker(s): {missing}")

    if args.sync:
        # Sync everything in the requested universe (downloads missing
        # and updates stale tickers).
        print(f"\n--- Syncing data ({len(universe)} tickers) ---")
        report = de.sync(symbols=universe)
        print(f"Sync result: {report}\n")
    elif missing:
        print(
            f"  Run with --sync to download them automatically.\n"
        )

    # Recompute after potential sync
    present = set(de.present())
    tradeable = [s for s in universe if s in present]

    if not tradeable:
        print(
            "\nNo OHLCV data on disk for the requested symbols.\n"
            "Run with --sync to download, or check your ticker file."
        )
        sys.exit(1)

    # Report if any tickers are still missing after sync attempt
    still_missing = [s for s in universe if s not in present]
    if still_missing:
        print(
            f"  Warning: {len(still_missing)} ticker(s) still have no data "
            f"and will be excluded: {still_missing}"
        )

    # Ensure SPY is available for market regime detection.
    # SPY may not be in the ticker file universe, so check on disk directly.
    spy_on_disk = (
        (de._data_dir / f"{args.spy}_{de._timeframe}.csv").exists()
        or (de._data_dir / f"{args.spy}_{de._timeframe}.parquet").exists()
    )
    if not spy_on_disk:
        if args.sync:
            print(f"\n--- Downloading {args.spy} for regime detection ---")
            spy_report = de.sync(symbols=[args.spy])
            print(f"SPY sync: {spy_report}")
            spy_on_disk = True
        else:
            print(
                f"\nNote: {args.spy} data not found. "
                f"Market regime detection will use defaults.\n"
                f"Run with --sync to download {args.spy}."
            )

    print(f"\nUniverse: {tradeable}")
    print(f"Date range: {args.start or 'earliest'} -> {args.end or 'latest'}")
    print(f"Initial cash: ${args.cash:,.0f}")
    print(f"Readiness threshold: {args.readiness:.0%}")

    # ── Run ───────────────────────────────────────────────────────────
    print(f"\n--- Running backtest ---\n")

    result = run_regime_backtest(
        universe=tradeable,
        data_engine=de,
        spy_symbol=args.spy,
        initial_cash=args.cash,
        start=args.start,
        end=args.end,
        readiness_threshold=args.readiness,
    )

    # ── Print results ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    for k, v in result.stats.items():
        if isinstance(v, float):
            if "return" in k or "drawdown" in k:
                print(f"  {k:<25s} {v:>10.2%}")
            elif "ratio" in k:
                print(f"  {k:<25s} {v:>10.2f}")
            elif "value" in k or "cash" in k:
                print(f"  {k:<25s} ${v:>12,.2f}")
            else:
                print(f"  {k:<25s} {v:>10.2f}")
        else:
            print(f"  {k:<25s} {v!s:>10s}")

    if len(result.weights_history) > 0:
        last_w = result.weights_history.iloc[-1]
        active = last_w[last_w > 0].sort_values(ascending=False)
        if not active.empty:
            print(f"\nLast portfolio ({result.weights_history.index[-1].date()}):")
            for sym, w in active.items():
                print(f"  {sym:<8s} {w:>7.2%}")

    if len(result.regime_history) > 0:
        print(f"\nRegime distribution:")
        for label, count in result.regime_history.value_counts().items():
            pct = count / len(result.regime_history)
            print(f"  {label:<15s} {count:>4d} ({pct:.0%})")

    print(f"\n{'='*60}\n")

    return result


if __name__ == "__main__":
    main()
