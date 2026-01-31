from __future__ import annotations
from typing import Dict, List, Optional

from .broker_base import BrokerBase
from .order_types import Order, Fill, OrderStatus, OrderType
from .risk import RiskManager


class ExecutionEngine:
    def __init__(self, broker: BrokerBase, risk: RiskManager, default_order_type: OrderType = OrderType.MARKET):
        self.broker = broker
        self.risk = risk
        self.default_order_type = default_order_type
        self.open_orders: Dict[str, Order] = {}

    def place_order(self, order: Order, positions: Dict[str, float]) -> Order:
        # Apply default order type if caller didn’t set one
        if order.order_type is None:
            order.order_type = self.default_order_type

        last = self.broker.get_last_price(order.symbol)
        account = self.broker.get_account()
        open_orders_count = sum(
            1 for o in self.open_orders.values()
            if o.status in (OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED)
        )

        ok, reason = self.risk.check(
            order,
            last_price=last,
            account=account,
            positions=positions,
            open_orders_count=open_orders_count,
        )
        if not ok:
            order.status = OrderStatus.REJECTED
            order.reason = reason
            return order

        accepted = self.broker.submit_order(order)
        self.open_orders[accepted.id] = accepted
        return accepted

    def poll(self) -> List[Fill]:
        return self.broker.poll_fills()