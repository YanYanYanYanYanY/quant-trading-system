from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List
from .order_types import Order, Fill


class BrokerBase(ABC):
    """Abstract broker interface.

    Every concrete broker (paper, Alpaca, IBKR, ...) must implement
    these methods so the :class:`ExecutionEngine` can work uniformly.
    """

    # ── Orders ────────────────────────────────────────────────────────
    @abstractmethod
    def submit_order(self, order: Order) -> Order:
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        ...

    @abstractmethod
    def poll_fills(self) -> List[Fill]:
        ...

    # ── Market data ───────────────────────────────────────────────────
    @abstractmethod
    def get_last_price(self, symbol: str) -> float:
        ...

    @abstractmethod
    def get_bid_ask(self, symbol: str) -> tuple[float, float]:
        ...

    # ── Account & positions ───────────────────────────────────────────
    @abstractmethod
    def get_account(self) -> Dict:
        """Return ``{'cash': float, 'equity': float, 'buying_power': float}``."""
        ...

    @abstractmethod
    def get_positions(self) -> Dict[str, float]:
        """Return ``{symbol: signed_qty}`` for all open positions."""
        ...