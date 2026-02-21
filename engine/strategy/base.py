"""
Strategy base module.

Provides the abstract :class:`Strategy` base class with a two-phase
bar lifecycle (warmup → live), plus supporting primitives
(:class:`Order`, :class:`Side`, :class:`StrategyState`) consumed by
the backtest and execution layers.

Lifecycle
---------
1. **Warmup phase** — the first ``warmup_bars`` bars are dispatched to
   :meth:`Strategy.warmup_bar`.  The strategy should update any
   internal state (e.g. rolling windows) but must **not** emit
   orders.
2. **Live phase** — once ``bar_count >= warmup_bars``, each bar is
   dispatched to :meth:`Strategy.on_bar`, which runs the full
   decision pipeline and may return orders.

The caller (backtest engine, live runner, etc.) is responsible for
calling ``warmup_bar`` / ``on_bar`` in sequence; the :class:`Strategy`
base class tracks ``bar_count``, ``last_asof``, and ``ready`` so
subclasses can query them at any point.

Design notes
------------
* ``warmup_bars`` is a **required** constructor argument — every
  concrete strategy must declare how much history it needs.
* Bar and order types are deliberately generic (``Any``) so the base
  class carries no dependency on a specific data layout.  Concrete
  strategies narrow the types in their overrides.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Side(str, Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


# ---------------------------------------------------------------------------
# Order
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Order:
    """
    A simple market order.

    Executed on the current bar's price (plus slippage) by the
    backtest or execution engine.

    Attributes
    ----------
    side : Side
        ``BUY`` or ``SELL``.
    qty : int
        Number of units.  Must be positive.
    reason : str
        Optional tag for logging / debugging (e.g. ``"entry"``,
        ``"stop_loss"``).

    Examples
    --------
    >>> Order(side=Side.BUY, qty=10, reason="momentum_entry")
    """

    side: Side
    qty: int
    reason: str = ""


# ---------------------------------------------------------------------------
# StrategyState
# ---------------------------------------------------------------------------


@dataclass
class StrategyState:
    """
    Mutable state carried through a backtest or live session.

    This is the *position-level* bookkeeping that the execution
    engine manages on behalf of the strategy.  Strategies read from
    it (e.g. to decide exit timing) but should only mutate it via
    the engine's order-execution path.

    Attributes
    ----------
    position : int
        Current number of units held (``>= 0`` for long-only).
    entry_price : float or None
        Execution price of the most recent entry.
    entry_index : pd.Timestamp or None
        Timestamp of the most recent entry.
    hold_bars : int
        Number of bars elapsed since the last entry.
    """

    position: int = 0
    entry_price: Optional[float] = None
    entry_index: Optional[pd.Timestamp] = None
    hold_bars: int = 0


# ---------------------------------------------------------------------------
# Strategy ABC
# ---------------------------------------------------------------------------


class Strategy(ABC):
    """
    Abstract base class for bar-driven strategies.

    Subclasses must implement :meth:`warmup_bar` and :meth:`on_bar`.
    The constructor requires ``warmup_bars`` — the minimum number of
    bars the strategy needs before it can start trading.

    Parameters
    ----------
    warmup_bars : int
        Number of bars required before :attr:`ready` becomes ``True``.
        Must be ``>= 0``.

    Attributes
    ----------
    warmup_bars : int
        *(read-only)* Number of warmup bars configured at construction.
    bar_count : int
        *(read-only)* Total bars processed so far (warmup + live).
    last_asof : pd.Timestamp or None
        *(read-only)* Timestamp of the most recent bar, or ``None``
        if no bars have been processed.
    ready : bool
        *(read-only)* ``True`` once ``bar_count >= warmup_bars``.

    Examples
    --------
    >>> class MomentumStrategy(Strategy):
    ...     def __init__(self):
    ...         super().__init__(warmup_bars=20)
    ...
    ...     def warmup_bar(self, bar) -> None:
    ...         # accumulate rolling features
    ...         pass
    ...
    ...     def on_bar(self, bar):
    ...         # full signal + order logic
    ...         return [Order(side=Side.BUY, qty=1, reason="go")]
    ...
    >>> s = MomentumStrategy()
    >>> s.warmup_bars
    20
    >>> s.ready
    False
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, warmup_bars: int) -> None:
        if warmup_bars < 0:
            raise ValueError(
                f"warmup_bars must be >= 0, got {warmup_bars}"
            )
        self._warmup_bars: int = warmup_bars
        self._bar_count: int = 0
        self._last_asof: Optional[pd.Timestamp] = None

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def warmup_bars(self) -> int:
        """Number of warmup bars required before trading."""
        return self._warmup_bars

    @property
    def bar_count(self) -> int:
        """Total bars processed so far (warmup + live)."""
        return self._bar_count

    @property
    def last_asof(self) -> Optional[pd.Timestamp]:
        """Timestamp of the most recent bar, or ``None``."""
        return self._last_asof

    @property
    def ready(self) -> bool:
        """``True`` once enough warmup bars have been processed."""
        return self.is_ready()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """
        Return ``True`` when ``bar_count >= warmup_bars``.

        Equivalent to :attr:`ready` but available as a regular method
        for callers that prefer an explicit function call.

        Returns
        -------
        bool
        """
        return self._bar_count >= self._warmup_bars

    def _increment_bar(self, asof: pd.Timestamp) -> None:
        """
        Advance the bar counter and record the timestamp.

        This should be called **once per bar** by the concrete
        strategy (typically at the top of :meth:`warmup_bar` and
        :meth:`on_bar`), or by the engine driving the lifecycle.

        Parameters
        ----------
        asof : pd.Timestamp
            Timestamp of the current bar.
        """
        self._bar_count += 1
        self._last_asof = asof

    def reset(self) -> None:
        """
        Reset bar counter and timestamp to initial state.

        Call before a new backtest run or live session to clear
        accumulated lifecycle state.  Subclasses that carry their
        own state should override this and call ``super().reset()``.
        """
        self._bar_count = 0
        self._last_asof = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def warmup_bar(self, bar: Any) -> None:
        """
        Process a single bar during the warmup phase.

        Called for each of the first ``warmup_bars`` bars.  The
        strategy should update any internal state it needs (rolling
        averages, feature buffers, etc.) but must **not** emit
        orders.

        Parameters
        ----------
        bar : Any
            A single bar of market data.  The concrete type depends
            on the engine (e.g. ``pd.Series``, ``dict``, a typed
            bar object).
        """
        ...

    @abstractmethod
    def on_bar(self, bar: Any) -> Any:
        """
        Process a single bar during the live phase.

        Called for every bar after warmup is complete
        (``bar_count >= warmup_bars``).  Runs the full decision
        pipeline — feature update, signal generation, and order
        construction.

        Parameters
        ----------
        bar : Any
            A single bar of market data (same type as
            :meth:`warmup_bar`).

        Returns
        -------
        list[Order] | list[dict] | Any
            Zero or more orders to execute.  The exact return type
            is determined by the execution engine; common choices
            include ``list[Order]``, a single ``Order | None``, or
            a list of dicts for downstream conversion.
        """
        ...
