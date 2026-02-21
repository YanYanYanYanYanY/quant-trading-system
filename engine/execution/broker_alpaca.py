"""
Alpaca broker adapter using the modern ``alpaca-py`` SDK.

Provides a concrete :class:`BrokerBase` implementation that connects to
Alpaca's Trading API (paper or live) and Market Data API for real-time
quotes.

Setup
-----
1. ``pip install alpaca-py``
2. Set env vars (or pass directly)::

       ALPACA_API_KEY=PK...
       ALPACA_API_SECRET=...
       ALPACA_PAPER=true          # "true" for paper, "false" for live

Usage
-----
::

    from engine.execution.broker_alpaca import AlpacaBroker, AlpacaBrokerConfig

    broker = AlpacaBroker(
        api_key="PK...",
        api_secret="...",
        cfg=AlpacaBrokerConfig(paper=True),
    )
    acct = broker.get_account()
    print(acct)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
)
from alpaca.trading.enums import (
    OrderSide as AlpacaOrderSide,
    OrderStatus as AlpacaOrderStatus,
    QueryOrderStatus,
    TimeInForce,
)
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    StockLatestQuoteRequest,
    StockLatestTradeRequest,
)

from .broker_base import BrokerBase
from .order_types import Fill, Order, OrderStatus, OrderType, Side

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlpacaBrokerConfig:
    """Alpaca connection and behaviour settings.

    Secrets (``api_key``, ``api_secret``) should be passed at runtime
    from environment variables — **never** hard-code them here.

    Attributes
    ----------
    paper : bool
        ``True`` for the paper-trading sandbox (default), ``False`` for
        the live production endpoint.
    time_in_force : str
        Default time-in-force for submitted orders (``"day"``, ``"gtc"``,
        ``"ioc"``, ``"fok"``).
    fill_poll_interval : float
        Minimum seconds between consecutive :meth:`poll_fills` network
        calls (avoids hammering the API).
    """

    paper: bool = True
    time_in_force: str = "day"
    fill_poll_interval: float = 1.0


# ---------------------------------------------------------------------------
# Time-in-force mapping helper
# ---------------------------------------------------------------------------

_TIF_MAP: Dict[str, TimeInForce] = {
    "day": TimeInForce.DAY,
    "gtc": TimeInForce.GTC,
    "ioc": TimeInForce.IOC,
    "fok": TimeInForce.FOK,
}


def _resolve_tif(tif_str: str) -> TimeInForce:
    """Convert a string time-in-force to the Alpaca enum."""
    result = _TIF_MAP.get(tif_str.lower())
    if result is None:
        raise ValueError(
            f"Unknown time_in_force {tif_str!r}. "
            f"Valid: {list(_TIF_MAP.keys())}"
        )
    return result


# ---------------------------------------------------------------------------
# Broker
# ---------------------------------------------------------------------------


class AlpacaBroker(BrokerBase):
    """Alpaca broker adapter (``alpaca-py`` SDK).

    Implements all :class:`BrokerBase` methods against Alpaca's Trading
    and Market Data REST APIs.

    Parameters
    ----------
    api_key : str
        Alpaca API key (``ALPACA_API_KEY``).
    api_secret : str
        Alpaca API secret (``ALPACA_API_SECRET``).
    cfg : AlpacaBrokerConfig
        Connection and behaviour settings.

    Notes
    -----
    *   **Thread safety** — the underlying ``httpx`` clients used by
        ``alpaca-py`` are *not* thread-safe.  Each thread should own its
        own :class:`AlpacaBroker` instance.
    *   **Fill tracking** — :meth:`poll_fills` fetches *closed* orders
        from Alpaca and converts newly-seen fills into
        :class:`~engine.execution.order_types.Fill` objects.  Call it
        periodically (e.g. every 1-5 s) to keep fill state current.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        cfg: AlpacaBrokerConfig | None = None,
    ) -> None:
        self.cfg = cfg or AlpacaBrokerConfig()

        # ── Trading client (orders, account, positions) ──────────────
        self._trading: TradingClient = TradingClient(
            api_key=api_key,
            secret_key=api_secret,
            paper=self.cfg.paper,
        )

        # ── Market-data client (quotes, trades) ─────────────────────
        self._data: StockHistoricalDataClient = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=api_secret,
        )

        # ── Internal bookkeeping ─────────────────────────────────────
        # Maps our internal order-id → Alpaca order UUID string.
        self._order_map: Dict[str, str] = {}
        # Reverse map: Alpaca UUID → internal order-id.
        self._reverse_map: Dict[str, str] = {}
        # Set of Alpaca order UUIDs whose fills we already emitted.
        self._seen_fill_ids: Set[str] = set()
        # Rate-limit guard for poll_fills.
        self._last_poll_ts: float = 0.0

        log.info(
            "AlpacaBroker initialised (paper=%s)", self.cfg.paper,
        )

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_last_price(self, symbol: str) -> float:
        """Return the latest trade price for *symbol*."""
        req = StockLatestTradeRequest(symbol_or_symbols=symbol)
        trades = self._data.get_stock_latest_trade(req)
        trade = trades[symbol]
        return float(trade.price)

    def get_bid_ask(self, symbol: str) -> tuple[float, float]:
        """Return ``(bid, ask)`` from the latest NBBO quote."""
        req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quotes = self._data.get_stock_latest_quote(req)
        quote = quotes[symbol]
        return float(quote.bid_price), float(quote.ask_price)

    # ------------------------------------------------------------------
    # Account & positions
    # ------------------------------------------------------------------

    def get_account(self) -> Dict:
        """Return account summary: cash, equity, buying_power."""
        acct = self._trading.get_account()
        return {
            "cash": float(acct.cash),
            "equity": float(acct.equity),
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
            "currency": str(acct.currency),
            "status": str(acct.status),
        }

    def get_positions(self) -> Dict[str, float]:
        """Return ``{symbol: signed_qty}`` for all open positions."""
        positions = self._trading.get_all_positions()
        result: Dict[str, float] = {}
        for pos in positions:
            qty = float(pos.qty)
            # Alpaca qty is always positive; side indicates direction.
            if str(pos.side) == "short":
                qty = -qty
            result[pos.symbol] = qty
        return result

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def submit_order(self, order: Order) -> Order:
        """Submit an order to Alpaca and return the updated order."""
        side = (
            AlpacaOrderSide.BUY
            if order.side == Side.BUY
            else AlpacaOrderSide.SELL
        )
        tif = _resolve_tif(self.cfg.time_in_force)

        try:
            if order.order_type == OrderType.MARKET:
                req = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=order.qty,
                    side=side,
                    time_in_force=tif,
                )
            elif order.order_type == OrderType.LIMIT:
                if order.limit_price is None:
                    order.status = OrderStatus.REJECTED
                    order.reason = "Limit order requires limit_price"
                    return order
                req = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=order.qty,
                    side=side,
                    time_in_force=tif,
                    limit_price=order.limit_price,
                )
            else:
                order.status = OrderStatus.REJECTED
                order.reason = f"Unsupported order_type: {order.order_type}"
                return order

            resp = self._trading.submit_order(req)

        except Exception as exc:
            order.status = OrderStatus.REJECTED
            order.reason = f"Alpaca submit_order failed: {exc}"
            log.warning(
                "Order %s rejected: %s", order.id[:8], exc,
            )
            return order

        # Store the mapping.
        alpaca_id = str(resp.id)
        self._order_map[order.id] = alpaca_id
        self._reverse_map[alpaca_id] = order.id
        order.status = OrderStatus.ACCEPTED

        log.info(
            "Order %s accepted → Alpaca %s (%s %s %.1f %s)",
            order.id[:8], alpaca_id[:8],
            order.side.value, order.symbol, order.qty,
            order.order_type.value,
        )
        return order

    def cancel_order(self, order_id: str) -> None:
        """Cancel an order by our internal ID.  Non-fatal if already done."""
        alpaca_id = self._order_map.get(order_id)
        if not alpaca_id:
            log.debug("cancel_order: unknown internal id %s", order_id[:8])
            return
        try:
            self._trading.cancel_order_by_id(alpaca_id)
            log.info(
                "Cancelled order %s (Alpaca %s)",
                order_id[:8], alpaca_id[:8],
            )
        except Exception as exc:
            # Already filled / cancelled / not found — non-fatal.
            log.debug(
                "cancel_order %s ignored: %s", order_id[:8], exc,
            )

    # ------------------------------------------------------------------
    # Fills
    # ------------------------------------------------------------------

    def poll_fills(self) -> List[Fill]:
        """Poll Alpaca for newly filled orders and return new fills.

        Uses ``get_orders(status='closed')`` filtered to orders we
        submitted.  Deduplicates by Alpaca order UUID so repeated
        calls are safe.

        Returns
        -------
        list[Fill]
            Zero or more new fills since the last poll.
        """
        now = time.time()
        if now - self._last_poll_ts < self.cfg.fill_poll_interval:
            return []
        self._last_poll_ts = now

        fills: List[Fill] = []

        try:
            req = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                limit=200,
            )
            closed_orders = self._trading.get_orders(req)
        except Exception as exc:
            log.warning("poll_fills failed: %s", exc)
            return fills

        for alp_order in closed_orders:
            alp_id = str(alp_order.id)

            # Skip orders we didn't submit or already processed.
            if alp_id not in self._reverse_map:
                continue
            if alp_id in self._seen_fill_ids:
                continue

            # Only process fully or partially filled orders.
            status = str(alp_order.status)
            if status not in ("filled", "partially_filled"):
                continue

            self._seen_fill_ids.add(alp_id)

            internal_id = self._reverse_map[alp_id]
            filled_qty = float(alp_order.filled_qty or 0)
            filled_price = float(alp_order.filled_avg_price or 0)

            if filled_qty <= 0:
                continue

            side_str = str(alp_order.side).lower()
            side = Side.BUY if side_str == "buy" else Side.SELL

            # Alpaca charges zero commission on most equity orders.
            ts = time.time()
            if alp_order.filled_at is not None:
                try:
                    ts = alp_order.filled_at.timestamp()
                except Exception:
                    pass

            fills.append(
                Fill(
                    order_id=internal_id,
                    symbol=str(alp_order.symbol),
                    side=side,
                    qty=filled_qty,
                    price=filled_price,
                    fee=0.0,
                    ts=ts,
                )
            )

            log.info(
                "Fill: %s %s %.1f @ %.2f (order %s)",
                side.value, alp_order.symbol,
                filled_qty, filled_price, internal_id[:8],
            )

        return fills

    # ------------------------------------------------------------------
    # Extended helpers (not in BrokerBase, but useful)
    # ------------------------------------------------------------------

    def get_open_orders(self) -> List[Dict]:
        """Return a list of currently open Alpaca orders."""
        try:
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            orders = self._trading.get_orders(req)
            return [
                {
                    "alpaca_id": str(o.id),
                    "symbol": str(o.symbol),
                    "side": str(o.side),
                    "qty": float(o.qty),
                    "type": str(o.type),
                    "status": str(o.status),
                    "filled_qty": float(o.filled_qty or 0),
                    "created_at": str(o.created_at),
                }
                for o in orders
            ]
        except Exception as exc:
            log.warning("get_open_orders failed: %s", exc)
            return []

    def is_market_open(self) -> bool:
        """Return ``True`` if the US equity market is currently open."""
        try:
            clock = self._trading.get_clock()
            return bool(clock.is_open)
        except Exception as exc:
            log.warning("is_market_open check failed: %s", exc)
            return False

    def get_clock(self) -> Dict:
        """Return market clock info (open, next_open, next_close)."""
        clock = self._trading.get_clock()
        return {
            "is_open": bool(clock.is_open),
            "next_open": str(clock.next_open),
            "next_close": str(clock.next_close),
            "timestamp": str(clock.timestamp),
        }

    def cancel_all_orders(self) -> int:
        """Cancel all open orders.  Returns count cancelled."""
        try:
            responses = self._trading.cancel_orders()
            cancelled = len(responses) if responses else 0
            log.info("Cancelled %d open orders", cancelled)
            return cancelled
        except Exception as exc:
            log.warning("cancel_all_orders failed: %s", exc)
            return 0

    def close_all_positions(self) -> int:
        """Liquidate all open positions.  Returns count closed."""
        try:
            responses = self._trading.close_all_positions(cancel_orders=True)
            closed = len(responses) if responses else 0
            log.info("Closed %d positions", closed)
            return closed
        except Exception as exc:
            log.warning("close_all_positions failed: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        mode = "paper" if self.cfg.paper else "LIVE"
        n_orders = len(self._order_map)
        n_fills = len(self._seen_fill_ids)
        return (
            f"AlpacaBroker(mode={mode}, orders={n_orders}, fills={n_fills})"
        )
