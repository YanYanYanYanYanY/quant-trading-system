from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List
from .order_types import Order, Fill

class BrokerBase(ABC):
    @abstractmethod
    def submit_order(self, order: Order) -> Order:
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        ...

    @abstractmethod
    def poll_fills(self) -> List[Fill]:
        ...

    @abstractmethod
    def get_last_price(self, symbol: str) -> float:
        ...

    @abstractmethod
    def get_bid_ask(self, symbol: str) -> tuple[float, float]:
        ...

    @abstractmethod
    def get_account(self) -> Dict:
        """cash, equity, buying_power, etc."""
        ...