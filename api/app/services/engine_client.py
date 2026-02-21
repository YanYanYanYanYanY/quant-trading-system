"""
Boundary object between the FastAPI layer and the engine module.

Wired to real engine components:
  - **Data**: ``engine.data.storage`` for OHLCV candles, ``DataEngine`` for
    universe resolution.
  - **Strategies**: ``RegimeAlphaStrategy`` and the regime-alpha pipeline.
  - **Backtests**: ``run_regime_backtest`` for portfolio-level backtests.
  - **Trading**: ``AlpacaBroker`` → ``ExecutionEngine`` for paper/live
    order submission, fill polling, account snapshots, and positions.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from app.schemas.requests import BacktestRequest, PlaceOrderRequest
from engine.data.storage import RAW_DIR, load_csv, load_parquet, file_exists
from engine.data.update_daily_aggregates import get_tickers
from engine.execution.broker_alpaca import AlpacaBroker, AlpacaBrokerConfig
from engine.execution.execution_engine import ExecutionEngine
from engine.execution.order_types import (
    Fill,
    Order,
    OrderStatus,
    OrderType,
    Side,
)
from engine.execution.risk import RiskLimits, RiskManager

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strategy registry (names surfaced to the UI)
# ---------------------------------------------------------------------------

STRATEGY_NAMES: List[str] = [
    "regime_alpha_v1",
]


class EngineClient:
    """Boundary object between API and engine.

    Wired to the engine module for:
      - Data: loads OHLCV from ``engine.data.storage`` (parquet/csv)
      - Symbols: reads from ticker file or scans data directory
      - Strategies: exposes strategy names from the registry
      - Backtests: runs portfolio-level backtests via ``run_regime_backtest``
      - Trading: real paper/live trading via ``AlpacaBroker`` → ``ExecutionEngine``

    The broker is lazily initialised on ``start_trading()`` so that the
    API can boot even when Alpaca keys are not yet configured.
    """

    def __init__(self) -> None:
        self._mode: str = "stopped"
        self._backtests: Dict[str, Dict[str, Any]] = {}

        # Broker / execution (created on start_trading)
        self._broker: Optional[AlpacaBroker] = None
        self._engine: Optional[ExecutionEngine] = None
        self._risk: Optional[RiskManager] = None
        self._fill_poller: Optional[threading.Thread] = None
        self._poll_stop = threading.Event()

        self._event_sink: Optional[Callable[[Dict[str, Any]], None]] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def utcnow_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _ensure_broker(self) -> AlpacaBroker:
        """Return the active broker, or raise if not trading."""
        if self._broker is None:
            raise RuntimeError(
                "Broker not initialised. Call start_trading first."
            )
        return self._broker

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        broker_ok = self._broker is not None
        return {
            "mode": self._mode,
            "engine_ok": True,
            "broker_connected": broker_ok,
        }

    # ------------------------------------------------------------------
    # Data (already wired to engine.data.storage)
    # ------------------------------------------------------------------

    def list_symbols(self) -> List[str]:
        """Return available symbols from ticker file or data directory."""
        ticker_file = RAW_DIR / "stocks" / "good_quality_stock_tickers_200.txt"
        if ticker_file.exists():
            try:
                return get_tickers(ticker_file)
            except Exception:
                pass

        data_dir = RAW_DIR / "stocks"
        if data_dir.exists():
            symbols: set[str] = set()
            for f in data_dir.glob("*_1d.parquet"):
                symbols.add(f.stem.replace("_1d", ""))
            for f in data_dir.glob("*_1d.csv"):
                symbols.add(f.stem.replace("_1d", ""))
            if symbols:
                return sorted(symbols)

        return ["AAPL", "MSFT", "NVDA", "SPY"]

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

        if tf in ("1d", "day"):
            rel_pq = f"raw/stocks/{symbol}_1d.parquet"
            rel_csv = f"raw/stocks/{symbol}_1d.csv"
        else:
            rel_pq = f"raw/stocks/{symbol}_{tf}.parquet"
            rel_csv = f"raw/stocks/{symbol}_{tf}.csv"

        df: Optional[pd.DataFrame] = None
        try:
            if file_exists(rel_pq):
                df = load_parquet(rel_pq)
            elif file_exists(rel_csv):
                df = load_csv(rel_csv)
        except Exception as exc:
            log.warning("Error loading candles for %s: %s", symbol, exc)
            return []

        if df is None or df.empty:
            return []

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp")

        if start and "timestamp" in df.columns:
            df = df[df["timestamp"] >= pd.to_datetime(start, utc=True)]
        if end and "timestamp" in df.columns:
            df = df[df["timestamp"] <= pd.to_datetime(end, utc=True)]

        if limit and len(df) > limit:
            df = df.tail(limit)

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

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def list_strategies(self) -> List[str]:
        return list(STRATEGY_NAMES)

    # ------------------------------------------------------------------
    # Backtests
    # ------------------------------------------------------------------

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

    def simulate_backtest_run(
        self, job_id: str, emit: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Run a portfolio backtest using the regime-alpha pipeline."""
        bt = self._backtests.get(job_id)
        if bt is None:
            return

        bt["state"] = "running"
        emit({
            "type": "log",
            "ts": self.utcnow_iso(),
            "data": {"msg": f"backtest {job_id} started"},
        })

        try:
            from engine.backtest.runner import run_regime_backtest

            req = bt["req"]
            symbol = req["symbol"].upper()
            initial_cash = req.get("initial_cash", 100_000.0)

            emit({
                "type": "log",
                "ts": self.utcnow_iso(),
                "data": {"msg": f"Running regime-alpha backtest for {symbol}..."},
            })
            bt["progress"] = 0.2

            result = run_regime_backtest(
                universe=[symbol],
                initial_cash=initial_cash,
                auto_sync=False,
            )

            bt["progress"] = 0.9

            stats = result.stats
            emit({
                "type": "pnl_update",
                "ts": self.utcnow_iso(),
                "data": {
                    "job_id": job_id,
                    "equity": stats.get("final_equity", initial_cash),
                    "progress": 1.0,
                },
            })

            bt["state"] = "done"
            bt["progress"] = 1.0
            bt["result"] = {
                "job_id": job_id,
                "final_equity": stats.get("final_equity", initial_cash),
                "return_pct": stats.get("total_return", 0) * 100.0,
                "max_drawdown": stats.get("max_drawdown", 0) * 100.0,
                "sharpe": stats.get("sharpe", 0),
                "n_rebalances": stats.get("n_rebalances", 0),
                "avg_positions": stats.get("avg_positions", 0),
            }

            emit({
                "type": "log",
                "ts": self.utcnow_iso(),
                "data": {
                    "msg": (
                        f"backtest {job_id} done: "
                        f"{stats.get('total_return', 0)*100:.2f}% return, "
                        f"sharpe={stats.get('sharpe', 0):.2f}"
                    )
                },
            })

        except Exception as exc:
            bt["state"] = "failed"
            bt["error"] = str(exc)
            emit({
                "type": "log",
                "ts": self.utcnow_iso(),
                "data": {"msg": f"backtest {job_id} failed: {exc}"},
            })

    # ------------------------------------------------------------------
    # Live / Paper trading — wired to AlpacaBroker + ExecutionEngine
    # ------------------------------------------------------------------

    def start_trading(
        self, emit: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Initialise broker, risk manager, and execution engine.

        Reads Alpaca credentials from environment variables and starts
        a background fill-polling thread.
        """
        api_key = os.getenv("ALPACA_API_KEY", "")
        api_secret = os.getenv("ALPACA_API_SECRET", "")
        paper = os.getenv("ALPACA_PAPER", "true").lower() in ("true", "1", "yes")

        if not api_key or not api_secret:
            raise RuntimeError(
                "ALPACA_API_KEY and ALPACA_API_SECRET must be set in .env"
            )

        cfg = AlpacaBrokerConfig(paper=paper)
        self._broker = AlpacaBroker(
            api_key=api_key, api_secret=api_secret, cfg=cfg,
        )
        self._risk = RiskManager(RiskLimits())
        self._engine = ExecutionEngine(
            broker=self._broker, risk=self._risk,
        )
        self._event_sink = emit
        self._mode = "paper" if paper else "live"

        # Start background fill poller
        self._poll_stop.clear()
        self._fill_poller = threading.Thread(
            target=self._poll_fills_loop, daemon=True,
        )
        self._fill_poller.start()

        log.info("Trading started (mode=%s)", self._mode)
        emit({
            "type": "log",
            "ts": self.utcnow_iso(),
            "data": {"msg": f"{self._mode} trading started — broker connected"},
        })

    def stop_trading(self) -> None:
        """Stop trading, cancel open orders, and shut down the fill poller."""
        if self._broker is not None:
            try:
                self._broker.cancel_all_orders()
            except Exception as exc:
                log.warning("Error cancelling orders on stop: %s", exc)

        self._poll_stop.set()
        if self._fill_poller and self._fill_poller.is_alive():
            self._fill_poller.join(timeout=5)

        self._mode = "stopped"
        self._broker = None
        self._engine = None
        self._risk = None
        self._event_sink = None
        log.info("Trading stopped")

    def _poll_fills_loop(self) -> None:
        """Background thread that polls broker for fills every 2 seconds."""
        while not self._poll_stop.is_set():
            try:
                if self._engine is not None:
                    fills = self._engine.poll()
                    for fill in fills:
                        self._emit_fill(fill)
            except Exception as exc:
                log.warning("Fill poll error: %s", exc)
            self._poll_stop.wait(2.0)

    def _emit_fill(self, fill: Fill) -> None:
        """Push a fill event through the WebSocket event sink."""
        if self._event_sink is None:
            return
        self._event_sink({
            "type": "order_filled",
            "ts": self.utcnow_iso(),
            "data": {
                "order_id": fill.order_id,
                "symbol": fill.symbol,
                "side": fill.side.value,
                "qty": fill.qty,
                "price": fill.price,
                "fee": fill.fee,
            },
        })

    # ------------------------------------------------------------------
    # Orders — wired to ExecutionEngine → AlpacaBroker
    # ------------------------------------------------------------------

    def place_order(self, req: PlaceOrderRequest) -> Dict[str, Any]:
        """Submit an order through the execution engine to Alpaca."""
        broker = self._ensure_broker()
        engine = self._engine
        if engine is None:
            raise RuntimeError("Execution engine not initialised")

        side = Side.BUY if req.side == "buy" else Side.SELL
        order_type = (
            OrderType.LIMIT if req.order_type == "limit" else OrderType.MARKET
        )

        order = Order(
            symbol=req.symbol.upper(),
            side=side,
            qty=float(req.qty),
            order_type=order_type,
            limit_price=float(req.limit_price) if req.limit_price else None,
        )

        positions = broker.get_positions()
        result = engine.place_order(order, positions)

        order_dict = {
            "id": result.id,
            "symbol": result.symbol,
            "side": result.side.value,
            "qty": result.qty,
            "type": result.order_type.value,
            "limit_price": result.limit_price,
            "status": result.status.value,
            "reason": result.reason,
            "ts": self.utcnow_iso(),
        }

        if self._event_sink:
            self._event_sink({
                "type": "order_submitted",
                "ts": self.utcnow_iso(),
                "data": order_dict,
            })

        return order_dict

    def get_orders(self) -> List[Dict[str, Any]]:
        """Return open + recent orders from the execution engine."""
        if self._engine is None:
            return []

        orders: List[Dict[str, Any]] = []
        for oid, order in self._engine.open_orders.items():
            orders.append({
                "id": order.id,
                "symbol": order.symbol,
                "side": order.side.value,
                "qty": order.qty,
                "type": order.order_type.value,
                "status": order.status.value,
                "filled_qty": order.filled_qty,
                "avg_fill_price": order.avg_fill_price,
                "reason": order.reason,
                "ts": datetime.fromtimestamp(
                    order.created_ts, tz=timezone.utc,
                ).isoformat(),
            })

        # Also include open orders from Alpaca directly
        if self._broker is not None:
            try:
                alpaca_open = self._broker.get_open_orders()
                seen_ids = {o["id"] for o in orders}
                for ao in alpaca_open:
                    if ao["alpaca_id"] not in seen_ids:
                        orders.append({
                            "id": ao["alpaca_id"],
                            "symbol": ao["symbol"],
                            "side": ao["side"],
                            "qty": ao["qty"],
                            "type": ao["type"],
                            "status": ao["status"],
                            "filled_qty": ao["filled_qty"],
                            "avg_fill_price": None,
                            "reason": None,
                            "ts": ao["created_at"],
                        })
            except Exception as exc:
                log.warning("Failed to fetch Alpaca open orders: %s", exc)

        return orders

    # ------------------------------------------------------------------
    # Positions — wired to AlpacaBroker
    # ------------------------------------------------------------------

    def get_positions(self) -> List[Dict[str, Any]]:
        """Return positions from Alpaca with live market prices."""
        if self._broker is None:
            return []

        try:
            raw = self._broker._trading.get_all_positions()
        except Exception as exc:
            log.warning("Failed to fetch positions: %s", exc)
            return []

        positions: List[Dict[str, Any]] = []
        for pos in raw:
            qty = float(pos.qty)
            if str(pos.side) == "short":
                qty = -qty
            positions.append({
                "symbol": str(pos.symbol),
                "qty": qty,
                "avg_price": float(pos.avg_entry_price),
                "market_price": float(pos.current_price),
                "market_value": float(pos.market_value),
                "unrealized_pnl": float(pos.unrealized_pl),
                "unrealized_pnl_pct": float(pos.unrealized_plpc) * 100,
                "side": str(pos.side),
            })

        return positions

    # ------------------------------------------------------------------
    # Account — wired to AlpacaBroker
    # ------------------------------------------------------------------

    def get_account(self) -> Dict[str, Any]:
        """Return the Alpaca account snapshot."""
        if self._broker is None:
            return {
                "equity": None,
                "cash": None,
                "buying_power": None,
                "portfolio_value": None,
                "daily_pnl": None,
            }
        try:
            acct = self._broker.get_account()
            return {
                "equity": acct["equity"],
                "cash": acct["cash"],
                "buying_power": acct["buying_power"],
                "portfolio_value": acct["portfolio_value"],
                "daily_pnl": None,
            }
        except Exception as exc:
            log.warning("Failed to fetch account: %s", exc)
            return {
                "equity": None,
                "cash": None,
                "buying_power": None,
                "portfolio_value": None,
                "daily_pnl": None,
            }

    # ------------------------------------------------------------------
    # Market clock — wired to AlpacaBroker
    # ------------------------------------------------------------------

    def get_clock(self) -> Dict[str, Any]:
        """Return market clock info from Alpaca."""
        if self._broker is None:
            return {"is_open": False, "next_open": None, "next_close": None}
        try:
            return self._broker.get_clock()
        except Exception as exc:
            log.warning("Failed to fetch clock: %s", exc)
            return {"is_open": False, "next_open": None, "next_close": None}

    # ------------------------------------------------------------------
    # Flatten all — wired to AlpacaBroker
    # ------------------------------------------------------------------

    def flatten_all(self) -> Dict[str, Any]:
        """Cancel all orders and close all positions via Alpaca."""
        if self._broker is None:
            return {"cancelled": 0, "closed": 0}
        cancelled = self._broker.cancel_all_orders()
        closed = self._broker.close_all_positions()
        if self._event_sink:
            self._event_sink({
                "type": "log",
                "ts": self.utcnow_iso(),
                "data": {
                    "msg": f"Flattened: cancelled {cancelled} orders, "
                           f"closed {closed} positions"
                },
            })
        return {"cancelled": cancelled, "closed": closed}
