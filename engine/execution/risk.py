from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
from .order_types import Order, Side


@dataclass(frozen=True)
class RiskLimits:
    max_notional_per_trade: float = 10_000.0
    max_position_notional: float = 30_000.0
    min_cash_buffer: float = 1_000.0
    max_open_orders: int = 20
    allow_short: bool = False


class RiskManager:
    def __init__(self, limits: RiskLimits):
        self.limits = limits

    def check(
        self,
        order: Order,
        last_price: float,
        account: Dict,
        positions: Dict[str, float],
        open_orders_count: int = 0,
    ) -> tuple[bool, str | None]:
        if open_orders_count >= self.limits.max_open_orders:
            return False, "Too many open orders"

        notional = abs(order.qty * last_price)
        if notional > self.limits.max_notional_per_trade:
            return False, f"Trade notional {notional:.2f} > max_notional_per_trade"

        cash = float(account.get("cash", 0.0))
        if order.side == Side.BUY and cash - notional < self.limits.min_cash_buffer:
            return False, "Insufficient cash buffer"

        pos_qty = float(positions.get(order.symbol, 0.0))
        new_qty = pos_qty + (order.qty if order.side == Side.BUY else -order.qty)

        if not self.limits.allow_short and new_qty < 0:
            return False, "Shorting not allowed"

        new_pos_notional = abs(new_qty * last_price)
        if new_pos_notional > self.limits.max_position_notional:
            return False, "Position limit exceeded"

        return True, None