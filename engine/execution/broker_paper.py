from __future__ import annotations
from typing import Dict, List, Optional
from collections import deque

from .broker_base import BrokerBase
from .order_types import Order, Fill, OrderStatus, OrderType, Side
from .costs import CostModel


class PaperBroker(BrokerBase):
    """
    Config-driven paper broker.

    price_feed expected per symbol keys:
      {"last": float, "bid": float, "ask": float}
    Optionally for bar-based limit fill rule "bar_high_low":
      {"high": float, "low": float}  (for current bar)
    """

    def __init__(
        self,
        price_feed: Dict[str, Dict[str, float]],
        cost_model: CostModel,
        starting_cash: float = 100_000.0,
        *,
        # paper behavior knobs (from YAML)
        fill_on: str = "next_poll",  # "immediate" | "next_poll"
        use_spread_for_market: bool = True,
        extra_slippage_bps: float = 2.0,
        allow_partial_fills: bool = False,
        partial_fill_ratio: float = 0.6,
        limit_fill_rule: str = "touch",  # "touch" | "bar_high_low"
        max_fill_delay_polls: int = 0,
    ):
        self.price_feed = price_feed
        self.costs = cost_model

        self.cash = float(starting_cash)
        self.equity = float(starting_cash)
        self.buying_power = float(starting_cash)

        self.fill_on = fill_on
        self.use_spread_for_market = use_spread_for_market
        self.extra_slippage_bps = float(extra_slippage_bps)
        self.allow_partial_fills = bool(allow_partial_fills)
        self.partial_fill_ratio = float(partial_fill_ratio)
        self.limit_fill_rule = limit_fill_rule
        self.max_fill_delay_polls = int(max_fill_delay_polls)

        self._orders: Dict[str, Order] = {}
        self._fills = deque()  # type: deque[Fill]
        self._delay_counter: Dict[str, int] = {}  # order_id -> polls waited

    # ---- market data ----

    def get_last_price(self, symbol: str) -> float:
        return float(self.price_feed[symbol]["last"])

    def get_bid_ask(self, symbol: str) -> tuple[float, float]:
        bid = float(self.price_feed[symbol]["bid"])
        ask = float(self.price_feed[symbol]["ask"])
        return bid, ask

    def get_account(self) -> Dict:
        return {"cash": self.cash, "equity": self.equity, "buying_power": self.buying_power}

    # ---- order API ----

    def submit_order(self, order: Order) -> Order:
        order.status = OrderStatus.ACCEPTED
        self._orders[order.id] = order
        self._delay_counter[order.id] = 0

        # If configured, market orders fill now; otherwise on next poll.
        if order.order_type == OrderType.MARKET and self.fill_on == "immediate":
            self._try_fill_market(order)

        return order

    def cancel_order(self, order_id: str) -> None:
        o = self._orders.get(order_id)
        if o and o.status in (OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED):
            o.status = OrderStatus.CANCELED

    def poll_fills(self) -> List[Fill]:
        # Attempt fills for eligible orders
        for o in list(self._orders.values()):
            if o.status not in (OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED):
                continue

            # delay logic
            waited = self._delay_counter.get(o.id, 0)
            if waited < self.max_fill_delay_polls:
                self._delay_counter[o.id] = waited + 1
                continue

            if o.order_type == OrderType.MARKET:
                if self.fill_on == "next_poll":
                    self._try_fill_market(o)
            elif o.order_type == OrderType.LIMIT:
                self._try_fill_limit(o)

        fills: List[Fill] = []
        while self._fills:
            fills.append(self._fills.popleft())
        return fills

    # ---- fill logic ----

    def _try_fill_market(self, order: Order) -> None:
        bid, ask = self.get_bid_ask(order.symbol)
        last = self.get_last_price(order.symbol)

        # Reference price for slippage model
        ref = ask if (self.use_spread_for_market and order.side == Side.BUY) else \
              bid if (self.use_spread_for_market and order.side == Side.SELL) else \
              (bid + ask) / 2.0

        # base slippage from CostModel (optional) + extra_slippage_bps (paper knob)
        base_slip = self.costs.slippage_per_share(order.qty, ref)
        extra_slip = abs(ref) * (self.extra_slippage_bps / 10_000.0)
        slip_per_share = base_slip + extra_slip

        fill_price = ref + slip_per_share if order.side == Side.BUY else ref - slip_per_share
        self._fill_order(order, fill_price)

    def _try_fill_limit(self, order: Order) -> None:
        if order.limit_price is None:
            return

        if self.limit_fill_rule == "touch":
            last = self.get_last_price(order.symbol)
            if order.side == Side.BUY and last <= order.limit_price:
                self._fill_order(order, order.limit_price)
            elif order.side == Side.SELL and last >= order.limit_price:
                self._fill_order(order, order.limit_price)
            return

        if self.limit_fill_rule == "bar_high_low":
            bar = self.price_feed[order.symbol]
            high_val = bar.get("high")
            low_val  = bar.get("low")
            last_val = bar.get("last")

            if last_val is None:
                raise ValueError("bar must contain 'last' price")
 
            high = float(high_val if high_val is not None else last_val)
            low  = float(low_val  if low_val  is not None else last_val)
            if order.side == Side.BUY and low <= order.limit_price:
                self._fill_order(order, order.limit_price)
            elif order.side == Side.SELL and high >= order.limit_price:
                self._fill_order(order, order.limit_price)
            return

        raise ValueError(f"Unknown limit_fill_rule: {self.limit_fill_rule}")

    def _fill_order(self, order: Order, fill_price: float) -> None:
        remaining = max(order.qty - order.filled_qty, 0.0)
        if remaining <= 0:
            return

        # partial fills
        if self.allow_partial_fills and remaining > 0:
            first_qty = remaining * self.partial_fill_ratio
            # keep it sensible
            fill_qty = first_qty if order.filled_qty == 0 else remaining
        else:
            fill_qty = remaining

        fee = self.costs.commission(fill_qty, fill_price)
        notional = fill_qty * fill_price

        # cash bookkeeping (simple cash account)
        if order.side == Side.BUY:
            self.cash -= (notional + fee)
        else:
            self.cash += (notional - fee)

        # update order avg price
        new_filled = order.filled_qty + fill_qty
        if order.avg_fill_price is None:
            order.avg_fill_price = fill_price
        else:
            # volume-weighted avg
            order.avg_fill_price = (order.avg_fill_price * order.filled_qty + fill_price * fill_qty) / new_filled

        order.filled_qty = new_filled
        if abs(order.filled_qty - order.qty) < 1e-9:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED

        self._fills.append(Fill(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            qty=fill_qty,
            price=fill_price,
            fee=fee,
        ))