"""
Rolling-window primitives for an event-driven feature engine.

All classes are pure-Python + NumPy — **no pandas dependency** — so they can
sit on a hot path that receives one tick / bar at a time.

Typical usage inside a feature builder::

    window = RollingWindow(20)
    for price in stream:
        window.add(price)
        if window.is_ready():
            feat = window.mean()
"""

from __future__ import annotations

import math
from collections import deque
from typing import Final

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_div(a: float, b: float, default: float = np.nan) -> float:
    """Return *a / b*, falling back to *default* when *b* is zero or NaN.

    >>> safe_div(10, 2)
    5.0
    >>> safe_div(1, 0)
    nan
    """
    try:
        if b == 0 or math.isnan(b):
            return default
        return a / b
    except (TypeError, ValueError):
        return default


def clip(x: float, lo: float, hi: float) -> float:
    """Clamp *x* into [*lo*, *hi*].

    Faster than ``np.clip`` for scalar values.

    >>> clip(-0.5, 0.0, 1.0)
    0.0
    >>> clip(0.7, 0.0, 1.0)
    0.7
    >>> clip(1.5, 0.0, 1.0)
    1.0
    """
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


# ---------------------------------------------------------------------------
# RollingWindow
# ---------------------------------------------------------------------------

class RollingWindow:
    """Fixed-size FIFO window over the last *N* ``float`` observations.

    Backed by :class:`collections.deque` for O(1) append / eviction.
    Expensive aggregations (``mean``, ``std``) are computed via NumPy on
    a snapshot array, which is cache-friendly and vectorised.

    Parameters
    ----------
    size : int
        Maximum number of values to keep.  Must be >= 1.

    Examples
    --------
    >>> w = RollingWindow(3)
    >>> w.add(1.0); w.add(2.0); w.add(3.0)
    >>> w.is_ready()
    True
    >>> w.mean()
    2.0
    """

    __slots__ = ("_size", "_buf")

    def __init__(self, size: int) -> None:
        if size < 1:
            raise ValueError(f"size must be >= 1, got {size}")
        self._size: Final[int] = size
        self._buf: deque[float] = deque(maxlen=size)

    # -- mutators -----------------------------------------------------------

    def add(self, x: float) -> None:
        """Append *x* to the window, evicting the oldest value if full."""
        self._buf.append(x)

    def reset(self) -> None:
        """Drop all stored observations."""
        self._buf.clear()

    # -- state queries ------------------------------------------------------

    def is_ready(self) -> bool:
        """``True`` once the window contains exactly *size* observations."""
        return len(self._buf) == self._size

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def size(self) -> int:
        """Configured window size (read-only)."""
        return self._size

    # -- accessors ----------------------------------------------------------

    def values(self) -> np.ndarray:
        """Return a **copy** of the buffer as a 1-D float64 array.

        Callers may mutate the returned array without affecting the window.
        """
        return np.array(self._buf, dtype=np.float64)

    def last(self) -> float:
        """Most recently added value, or ``np.nan`` if empty."""
        if not self._buf:
            return np.nan
        return self._buf[-1]

    # -- aggregations -------------------------------------------------------

    def mean(self) -> float:
        """Arithmetic mean of the current buffer, or ``np.nan`` if empty."""
        if not self._buf:
            return np.nan
        return float(np.mean(self.values()))

    def std(self, ddof: int = 0) -> float:
        """Standard deviation of the current buffer.

        Parameters
        ----------
        ddof : int
            Delta degrees of freedom (default 0 = population std).

        Returns ``np.nan`` when the buffer has fewer values than ``ddof + 1``
        (the minimum needed for a finite result).
        """
        n = len(self._buf)
        if n == 0 or n <= ddof:
            return np.nan
        return float(np.std(self.values(), ddof=ddof))

    def sum(self) -> float:
        """Sum of the current buffer, or ``np.nan`` if empty."""
        if not self._buf:
            return np.nan
        return float(np.sum(self.values()))

    def min(self) -> float:
        """Minimum value in the buffer, or ``np.nan`` if empty."""
        if not self._buf:
            return np.nan
        return float(np.min(self.values()))

    def max(self) -> float:
        """Maximum value in the buffer, or ``np.nan`` if empty."""
        if not self._buf:
            return np.nan
        return float(np.max(self.values()))

    # -- dunder -------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RollingWindow(size={self._size}, "
            f"filled={len(self._buf)}/{self._size})"
        )


# ---------------------------------------------------------------------------
# RollingEWMA — exponentially weighted moving average (recursive form)
# ---------------------------------------------------------------------------

class RollingEWMA:
    """Recursive exponentially weighted moving average.

    The update rule is::

        ewma_{t} = alpha * x_{t}  +  (1 - alpha) * ewma_{t-1}

    where *alpha* in (0, 1] controls responsiveness.  A higher *alpha*
    gives more weight to recent observations.

    Parameters
    ----------
    alpha : float
        Smoothing factor, must be in (0, 1].
    warmup : int
        Number of observations before the EWMA is considered ready.
        Defaults to 1 (ready immediately after the first update).

    Examples
    --------
    >>> e = RollingEWMA(alpha=0.5)
    >>> e.update(10.0); e.update(20.0)
    >>> round(e.value, 4)
    15.0
    """

    __slots__ = ("_alpha", "_one_minus_alpha", "_value", "_n", "_warmup")

    def __init__(self, alpha: float, warmup: int = 1) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if warmup < 1:
            raise ValueError(f"warmup must be >= 1, got {warmup}")
        self._alpha: Final[float] = alpha
        self._one_minus_alpha: Final[float] = 1.0 - alpha
        self._value: float = np.nan
        self._n: int = 0
        self._warmup: Final[int] = warmup

    # -- mutators -----------------------------------------------------------

    def update(self, x: float) -> None:
        """Incorporate a new observation *x*."""
        self._n += 1
        if self._n == 1:
            # Initialise to the first value (unbiased start).
            self._value = x
        else:
            self._value = self._alpha * x + self._one_minus_alpha * self._value

    def reset(self) -> None:
        """Reset to pristine state."""
        self._value = np.nan
        self._n = 0

    # -- accessors ----------------------------------------------------------

    @property
    def value(self) -> float:
        """Current EWMA value, or ``np.nan`` before warmup."""
        if self._n < self._warmup:
            return np.nan
        return self._value

    def is_ready(self) -> bool:
        """``True`` once at least *warmup* observations have been fed."""
        return self._n >= self._warmup

    @property
    def count(self) -> int:
        """Total observations fed so far."""
        return self._n

    # -- dunder -------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RollingEWMA(alpha={self._alpha}, "
            f"n={self._n}, value={self._value:.6g})"
        )
