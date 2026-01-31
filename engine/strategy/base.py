from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional
from abc import ABC, abstractmethod
import pandas as pd


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class Order:
    """
    A simple market order executed on the current bar's price (plus slippage).
    qty: positive integer
    side: BUY or SELL
    reason: optional tag for logging/debugging
    """
    side: Side
    qty: int
    reason: str = ""


@dataclass
class StrategyState:
    """
    Mutable state carried through the backtest.
    You can extend this later (e.g., trailing stops, pyramiding, etc.)
    """
    position: int = 0                 # number of units held (long-only in our starter)
    entry_price: Optional[float] = None
    entry_index: Optional[pd.Timestamp] = None
    hold_bars: int = 0                # how many bars held since entry


class Strategy(ABC):
    """
    Base class for strategies.

    Convention:
    - data is a DataFrame indexed by datetime (or integer index),
      with at least a 'close' column and whatever features you computed.
    - on_bar returns an Order or None.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def warmup_bars(self) -> int:
        """Number of bars needed before the strategy can trade (e.g., SMA window)."""
        raise NotImplementedError

    @abstractmethod
    def on_bar(
        self,
        i: int,
        row: pd.Series,
        data: pd.DataFrame,
        state: StrategyState,
        context: Dict,
    ) -> Optional[Order]:
        """Called once per bar. Return an Order to execute on this bar, or None."""
        raise NotImplementedError