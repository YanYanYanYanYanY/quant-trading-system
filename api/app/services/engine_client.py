from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from app.schemas.requests import BacktestRequest, PlaceOrderRequest
from engine.backtest.backtest import run_backtest, BacktestResult
from engine.data.storage import RAW_DIR, load_csv, load_parquet, file_exists
from engine.data.update_daily_aggregates import get_tickers
from engine.strategy.mean_revert import MeanReversionStrategy
from engine.strategy.rules import MeanRevertParams


# Strategy registry: maps strategy names to factory functions
STRATEGY_REGISTRY: Dict[str, Callable] = {
    "mean_revert_v1": lambda: MeanReversionStrategy(MeanRevertParams(), name="mean_revert_v1"),
}


class EngineClient:
    """
    Boundary object between API and engine.
    
    Wired to the engine module for:
      - Data: loads OHLCV from engine.data.storage (parquet/csv)
      - Symbols: reads from ticker file or scans data directory
      - Strategies: uses STRATEGY_REGISTRY
      - Backtests: runs real backtests via engine.backtest.run_backtest
      
    Trading methods (place_order, start_trading, etc.) still use in-memory stubs.
    Wire to engine.execution when ready for paper/live trading.
    """

    def __init__(self) -> None:
        self._mode = "stopped"  # stopped/paper/live
        self._backtests: Dict[str, Dict[str, Any]] = {}
        self._orders: List[Dict[str, Any]] = []
        self._positions: Dict[str, Dict[str, Any]] = {}

        self._symbols = ["AAPL", "MSFT", "NVDA", "SPY"]
        self._strategies = ["mean_revert_v1", "trend_v1"]

        self._event_sink: Optional[Callable[[Dict[str, Any]], None]] = None

    # -----------------
    # Helpers
    # -----------------
    def utcnow_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # -----------------
    # Status
    # -----------------
    def get_status(self) -> Dict[str, Any]:
        # ws_clients + last_event_ts are owned by WSManager; API merges them
        return {"mode": self._mode, "engine_ok": True}

    # -----------------
    # Data
    # -----------------
    def list_symbols(self) -> List[str]:
        """Return available symbols from ticker file or data directory."""
        # Try to read from ticker file
        ticker_file = RAW_DIR / "stocks" / "good_quality_stock_tickers_200.txt"
        if ticker_file.exists():
            try:
                return get_tickers(ticker_file)
            except Exception:
                pass
        
        # Fallback: scan data directory for existing files
        data_dir = RAW_DIR / "stocks"
        if data_dir.exists():
            symbols = set()
            for f in data_dir.glob("*_1d.parquet"):
                symbol = f.stem.replace("_1d", "")
                symbols.add(symbol)
            for f in data_dir.glob("*_1d.csv"):
                symbol = f.stem.replace("_1d", "")
                symbols.add(symbol)
            if symbols:
                return sorted(symbols)
        
        # Final fallback to hardcoded list
        return list(self._symbols)

    def get_candles(
        self,
        symbol: str,
        tf: str,
        start: Optional[str],
        end: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Load candles from engine storage (parquet/csv)."""
        symbol = symbol.upper()
        
        # Determine file path based on timeframe
        if tf == "1d" or tf == "day":
            rel_path_parquet = f"raw/stocks/{symbol}_1d.parquet"
            rel_path_csv = f"raw/stocks/{symbol}_1d.csv"
        else:
            rel_path_parquet = f"raw/stocks/{symbol}_{tf}.parquet"
            rel_path_csv = f"raw/stocks/{symbol}_{tf}.csv"
        
        # Try to load data (parquet first, then CSV)
        df = None
        try:
            if file_exists(rel_path_parquet):
                df = load_parquet(rel_path_parquet)
            elif file_exists(rel_path_csv):
                df = load_csv(rel_path_csv)
        except Exception as e:
            print(f"Error loading candles for {symbol}: {e}")
            return []
        
        if df is None or df.empty:
            return []
        
        # Ensure timestamp column is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp")
        
        # Filter by date range
        if start and "timestamp" in df.columns:
            start_dt = pd.to_datetime(start, utc=True)
            df = df[df["timestamp"] >= start_dt]
        if end and "timestamp" in df.columns:
            end_dt = pd.to_datetime(end, utc=True)
            df = df[df["timestamp"] <= end_dt]
        
        # Apply limit (take last N rows)
        if limit and len(df) > limit:
            df = df.tail(limit)
        
        # Convert to list of dicts with standard column names
        candles: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            ts = row.get("timestamp")
            candles.append({
                "ts": ts.isoformat() if pd.notna(ts) else self.utcnow_iso(),
                "open": float(row.get("open", row.get("o", 0))),
                "high": float(row.get("high", row.get("h", 0))),
                "low": float(row.get("low", row.get("l", 0))),
                "close": float(row.get("close", row.get("c", 0))),
                "volume": float(row.get("volume", row.get("v", 0))),
            })
        
        return candles

    # -----------------
    # Strategies
    # -----------------
    def list_strategies(self) -> List[str]:
        """Return available strategies from the registry."""
        return list(STRATEGY_REGISTRY.keys())

    # -----------------
    # Backtests
    # -----------------
    def start_backtest(self, req: BacktestRequest) -> str:
        job_id = uuid.uuid4().hex
        self._backtests[job_id] = {
            "job_id": job_id,
            "state": "queued",
            "progress": 0.0,
            "result": None,
            "error": None,
            "req": req.model_dump(),
        }
        return job_id

    def get_backtest_status(self, job_id: str) -> Dict[str, Any]:
        bt = self._backtests.get(job_id)
        if bt is None:
            return {
                "job_id": job_id,
                "state": "failed",
                "progress": 0.0,
                "result": None,
                "error": "unknown job_id",
            }
        return {
            "job_id": bt["job_id"],
            "state": bt["state"],
            "progress": bt["progress"],
            "result": bt["result"],
            "error": bt["error"],
        }

    def simulate_backtest_run(self, job_id: str, emit: Callable[[Dict[str, Any]], None]) -> None:
        """
        Run a real backtest using the engine module.
        Loads data, creates strategy, runs backtest, and emits events.
        """
        bt = self._backtests.get(job_id)
        if bt is None:
            return

        bt["state"] = "running"
        emit({"type": "log", "ts": self.utcnow_iso(), "data": {"msg": f"backtest {job_id} started"}})

        try:
            req = bt["req"]
            symbol = req["symbol"].upper()
            initial_cash = req.get("initial_cash", 10_000.0)
            strategy_name = req["strategy"]
            
            # 1. Load data
            emit({"type": "log", "ts": self.utcnow_iso(), "data": {"msg": f"Loading data for {symbol}..."}})
            
            rel_path_parquet = f"raw/stocks/{symbol}_1d.parquet"
            rel_path_csv = f"raw/stocks/{symbol}_1d.csv"
            
            df = None
            if file_exists(rel_path_parquet):
                df = load_parquet(rel_path_parquet)
            elif file_exists(rel_path_csv):
                df = load_csv(rel_path_csv)
            
            if df is None or df.empty:
                raise ValueError(f"No data found for {symbol}")
            
            # 2. Prepare data
            # Ensure 'close' column exists
            if "close" not in df.columns:
                if "c" in df.columns:
                    df["close"] = df["c"]
                else:
                    raise ValueError("Data must have 'close' or 'c' column")
            
            # Add features needed by mean revert strategy (sma20, std20)
            df["sma20"] = df["close"].rolling(20).mean()
            df["std20"] = df["close"].rolling(20).std()
            
            # Drop rows with NaN (from rolling calculations)
            df = df.dropna()
            
            if df.empty:
                raise ValueError(f"Not enough data for {symbol} after feature calculation")
            
            # Set timestamp as index if available
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp")
            
            emit({"type": "log", "ts": self.utcnow_iso(), "data": {"msg": f"Loaded {len(df)} bars for {symbol}"}})
            bt["progress"] = 0.1
            
            # 3. Get strategy from registry
            if strategy_name not in STRATEGY_REGISTRY:
                raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(STRATEGY_REGISTRY.keys())}")
            
            strategy = STRATEGY_REGISTRY[strategy_name]()
            emit({"type": "log", "ts": self.utcnow_iso(), "data": {"msg": f"Using strategy: {strategy_name}"}})
            bt["progress"] = 0.2
            
            # 4. Run backtest
            emit({"type": "log", "ts": self.utcnow_iso(), "data": {"msg": "Running backtest..."}})
            
            result = run_backtest(
                data=df,
                strategy=strategy,
                initial_cash=initial_cash,
                slippage_bps=5.0,
            )
            
            bt["progress"] = 0.9
            
            # 5. Emit final results
            emit({
                "type": "pnl_update",
                "ts": self.utcnow_iso(),
                "data": {
                    "job_id": job_id,
                    "equity": result.stats["final_equity"],
                    "progress": 1.0,
                },
            })
            
            # 6. Store result
            bt["state"] = "done"
            bt["progress"] = 1.0
            bt["result"] = {
                "job_id": job_id,
                "final_equity": result.stats["final_equity"],
                "return_pct": result.stats["total_return"] * 100.0,
                "max_drawdown": result.stats["max_drawdown"] * 100.0,
                "num_trades": int(result.stats["num_trades"]),
                "win_rate": result.stats["win_rate"] * 100.0,
                "avg_trade_pnl": result.stats["avg_trade_pnl"],
            }
            
            emit({"type": "log", "ts": self.utcnow_iso(), "data": {
                "msg": f"backtest {job_id} done: {result.stats['num_trades']:.0f} trades, "
                       f"{result.stats['total_return']*100:.2f}% return"
            }})

        except Exception as e:
            bt["state"] = "failed"
            bt["error"] = str(e)
            emit({"type": "log", "ts": self.utcnow_iso(), "data": {"msg": f"backtest {job_id} failed: {e}"}})

    # -----------------
    # Live/Paper trading
    # -----------------
    def start_trading(self, emit: Callable[[Dict[str, Any]], None]) -> None:
        self._mode = "paper"
        self._event_sink = emit
        emit({"type": "log", "ts": self.utcnow_iso(), "data": {"msg": "paper trading started"}})

    def stop_trading(self) -> None:
        self._mode = "stopped"
        self._event_sink = None

    def place_order(self, req: PlaceOrderRequest) -> Dict[str, Any]:
        order_id = uuid.uuid4().hex
        order = {
            "id": order_id,
            "symbol": req.symbol,
            "side": req.side,
            "qty": float(req.qty),
            "type": req.order_type,
            "limit_price": req.limit_price,
            "status": "submitted",
            "ts": self.utcnow_iso(),
        }
        self._orders.append(order)

        # Simulate immediate fill for demo
        fill_price = 100.0 if req.order_type == "market" else float(req.limit_price or 100.0)
        order["status"] = "filled"
        order["fill_price"] = fill_price

        pos = self._positions.get(req.symbol, {"symbol": req.symbol, "qty": 0.0, "avg_price": 0.0})
        signed_qty = req.qty if req.side == "buy" else -req.qty
        new_qty = pos["qty"] + signed_qty

        # naive avg price update (demo only)
        if new_qty != 0 and req.side == "buy":
            pos["avg_price"] = (pos["avg_price"] * pos["qty"] + fill_price * req.qty) / (pos["qty"] + req.qty)
        pos["qty"] = new_qty
        self._positions[req.symbol] = pos

        if self._event_sink:
            self._event_sink({"type": "order_filled", "ts": self.utcnow_iso(), "data": order})
            self._event_sink({"type": "position_update", "ts": self.utcnow_iso(), "data": pos})

        return order

    def get_orders(self) -> List[Dict[str, Any]]:
        return list(self._orders)

    def get_positions(self) -> List[Dict[str, Any]]:
        return list(self._positions.values())