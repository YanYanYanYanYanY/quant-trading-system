from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import time

from alpaca_trade_api import REST  # pip install alpaca-trade-api

from .broker_base import BrokerBase
from .order_types import Order, Fill, Side, OrderStatus, OrderType


@dataclass(frozen=True)
class AlpacaBrokerConfig:
    """
    Secrets should NOT go here. Pass api_key/api_secret from env vars.
    """
    base_url: str = "https://paper-api.alpaca.markets"  # or live: https://api.alpaca.markets
    api_version: str = "v2"
    # Polling parameters
    activities_page_size: int = 100
    lookback_seconds: int = 120  # safety window to avoid missing fills
    time_in_force: str = "day"   # "day", "gtc", ...


class AlpacaBroker(BrokerBase):
    """
    Alpaca broker adapter using alpaca-trade-api REST client.

    Key properties:
    - submit_order places orders via Alpaca Trading API
    - poll_fills polls /v2/account/activities?activity_types=FILL (idempotent)
    - broker is the source of truth for executions
    """

    def __init__(self, api_key: str, api_secret: str, cfg: AlpacaBrokerConfig):
        self.cfg = cfg
        self.api = REST(
            key_id=api_key,
            secret_key=api_secret,
            base_url=cfg.base_url,
            api_version=cfg.api_version,
        )

        # internal_id -> alpaca_order_id
        self._order_map: Dict[str, str] = {}

        # Idempotency: keep seen activity ids
        self._seen_activity_ids: Set[str] = set()

        # Timestamp cursor for polling
        self._last_poll_ts: float = 0.0

    # ---------- Market data ----------
    def get_last_price(self, symbol: str) -> float:
        trade = self.api.get_latest_trade(symbol)
        return float(trade.price)

    def get_bid_ask(self, symbol: str) -> tuple[float, float]:
        q = self.api.get_latest_quote(symbol)
        return float(q.bp), float(q.ap)  # bid price, ask price (alpaca_trade_api fields)

    # ---------- Account ----------
    def get_account(self) -> Dict:
        acct = self.api.get_account()
        return {
            "cash": float(acct.cash),
            "equity": float(acct.equity),
            "buying_power": float(acct.buying_power),
        }

    # ---------- Orders ----------
    def submit_order(self, order: Order) -> Order:
        side = "buy" if order.side == Side.BUY else "sell"
        tif = (self.cfg.time_in_force or "day").lower()

        try:
            if order.order_type == OrderType.MARKET:
                resp = self.api.submit_order(
                    symbol=order.symbol,
                    qty=order.qty,
                    side=side,
                    type="market",
                    time_in_force=tif,
                )
            elif order.order_type == OrderType.LIMIT:
                if order.limit_price is None:
                    order.status = OrderStatus.REJECTED
                    order.reason = "Limit order missing limit_price"
                    return order

                resp = self.api.submit_order(
                    symbol=order.symbol,
                    qty=order.qty,
                    side=side,
                    type="limit",
                    limit_price=str(order.limit_price),
                    time_in_force=tif,
                )
            else:
                order.status = OrderStatus.REJECTED
                order.reason = f"Unsupported order_type: {order.order_type}"
                return order

        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.reason = f"Alpaca submit_order failed: {e}"
            return order
        
        if resp is None or not hasattr(resp, "id"):
           order.status = OrderStatus.REJECTED
           order.reason = "Alpaca submit_order returned no order id"
           return order

        self._order_map[order.id] = str(resp.id)
        order.status = OrderStatus.ACCEPTED
        return order

    def cancel_order(self, order_id: str) -> None:
        alp_id = self._order_map.get(order_id)
        if not alp_id:
            return
        try:
            self.api.cancel_order(alp_id)
        except Exception:
            # already filled/canceled or not found -> non-fatal
            return

    # ---------- Fills ----------
    def poll_fills(self) -> List[Fill]:
        """
        Poll Alpaca account activities for new fills.

        Uses /v2/account/activities with pagination; Alpaca docs describe page_token/page_size. :contentReference[oaicite:3]{index=3}
        We deduplicate by activity id so repeated polls are safe.
        """
        now = time.time()
        if self._last_poll_ts <= 0:
            after_ts = now - self.cfg.lookback_seconds
        else:
            # small overlap to be safe
            after_ts = self._last_poll_ts - 2.0

        self._last_poll_ts = now
        after_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(after_ts))

        fills: List[Fill] = []
        page_token: Optional[str] = None

        while True:
            try:
                # alpaca_trade_api provides get_activities; pass after/page_size/page_token.
                acts = self.api.get_activities(
                    activity_types="FILL",
                    after=after_iso,
                    page_size=self.cfg.activities_page_size,
                    page_token=page_token,
                )
            except Exception:
                break

            if not acts:
                break

            for act in acts:
                act_id = str(getattr(act, "id", "") or "")
                if not act_id or act_id in self._seen_activity_ids:
                    continue
                self._seen_activity_ids.add(act_id)

                symbol = str(getattr(act, "symbol", "") or "")
                side_raw = str(getattr(act, "side", "") or "").lower()
                qty = float(getattr(act, "qty", 0.0) or 0.0)
                price = float(getattr(act, "price", 0.0) or 0.0)

                alp_order_id = str(getattr(act, "order_id", "") or "")
                internal_id = self._find_internal_order_id(alp_order_id) or alp_order_id or act_id

                # Alpaca returns transaction_time in many activity models; keep best-effort.
                tx_time = getattr(act, "transaction_time", None)
                if tx_time is None:
                   ts = time.time()
                elif hasattr(tx_time, "timestamp"):
                    ts = float(tx_time.timestamp())
                else:
                    ts = time.time()
                fills.append(
                    Fill(
                        order_id=internal_id,
                        symbol=symbol,
                        side=Side.BUY if side_raw == "buy" else Side.SELL,
                        qty=qty,
                        price=price,
                        fee=0.0,
                        ts=ts,
                    )
                )

            # Pagination: Alpaca uses page_token/page_size (docs). :contentReference[oaicite:4]{index=4}
            # alpaca_trade_api usually exposes next_page_token on the response object or via headers,
            # but it can vary; safest is: if len(acts) < page_size, we stop.
            if len(acts) < self.cfg.activities_page_size:
                break

            # If your REST client exposes a token, set it here; otherwise, stopping on length is fine.
            page_token = getattr(acts, "next_page_token", None)  # may be None depending on SDK

            if not page_token:
                break

        return fills

    # ---------- Helpers ----------
    def _find_internal_order_id(self, alp_order_id: str) -> Optional[str]:
        if not alp_order_id:
            return None
        for internal_id, mapped_alp_id in self._order_map.items():
            if mapped_alp_id == alp_order_id:
                return internal_id
        return None