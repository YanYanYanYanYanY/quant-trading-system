"""
Portfolio construction types and interfaces for a long-only strategy.

Defines the data structures that flow between the alpha layer and the
execution / risk layer:

* :class:`TargetPortfolio` — the desired weight vector at a point in time.
* :class:`PortfolioConstraints` — tuneable limits on position count,
  weight concentration, turnover, etc.
* :class:`PortfolioConstructor` — protocol that every portfolio-builder
  must satisfy.

Helper functions
----------------
* :func:`normalize_positive` — rescale non-negative values to a target sum.
* :func:`clip_weights` — cap individual weights and redistribute excess.
* :func:`enforce_max_positions` — truncate a ranked symbol list.

Design notes
------------
* Weights are **non-negative** (long-only).  Any negative value that
  leaks in is an error and will be caught by ``TargetPortfolio``
  validation.
* ``gross`` tracks the actual sum of weights.  When ``allow_cash`` is
  ``True`` the gross may be less than ``gross_target``; when ``False``
  the builder must fully invest.
* The ``reasons`` dict on ``TargetPortfolio`` mirrors the pattern used
  in ``AlphaScore`` and ``StockRegimeState`` — a flat or nested dict
  of numeric diagnostics for attribution and debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol, Union, runtime_checkable

import pandas as pd

from ..alpha.types import AlphaVector


# ---------------------------------------------------------------------------
# TargetPortfolio
# ---------------------------------------------------------------------------


@dataclass
class TargetPortfolio:
    """
    Desired portfolio weights at a point in time.

    All weights are non-negative (long-only).  The ``gross`` field
    records the actual sum of weights, which may be less than the
    constraint's ``gross_target`` when cash is allowed and there are
    too few qualifying candidates.

    Attributes
    ----------
    asof : pd.Timestamp
        Rebalance timestamp.
    weights : dict[str, float]
        Symbol → target weight.  Every value must be ``>= 0``.
    gross : float
        Sum of all weights (``sum(weights.values())``).  Must be
        ``>= 0``.
    reasons : dict[str, float | dict[str, float]]
        Optional diagnostic breakdown.  Top-level keys may map to
        scalars (e.g. ``"n_positions": 25``) or to per-symbol dicts
        (e.g. ``"alpha_score": {"AAPL": 0.82, ...}``).

    Examples
    --------
    >>> tp = TargetPortfolio(
    ...     asof=pd.Timestamp("2025-06-15"),
    ...     weights={"AAPL": 0.04, "MSFT": 0.03, "GOOG": 0.02},
    ...     gross=0.09,
    ... )
    >>> tp.n_positions
    3
    >>> tp.top(2)
    [('AAPL', 0.04), ('MSFT', 0.03)]
    """

    asof: pd.Timestamp
    weights: Dict[str, float]
    gross: float
    reasons: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate long-only invariants after initialisation."""
        for sym, w in self.weights.items():
            if w < 0.0:
                raise ValueError(
                    f"Long-only weight for {sym!r} must be >= 0, got {w}"
                )
        if self.gross < 0.0:
            raise ValueError(f"gross must be >= 0, got {self.gross}")

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TargetPortfolio(asof={self.asof}, "
            f"n_positions={self.n_positions}, gross={self.gross:.4f})"
        )

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------

    @property
    def n_positions(self) -> int:
        """Number of symbols with strictly positive weight."""
        return sum(1 for w in self.weights.values() if w > 0.0)

    @property
    def symbols(self) -> List[str]:
        """Symbols with positive weight, sorted descending by weight."""
        return [
            sym
            for sym, _ in sorted(
                self.weights.items(), key=lambda kv: kv[1], reverse=True,
            )
            if self.weights[sym] > 0.0
        ]

    @property
    def max_weight(self) -> float:
        """Largest individual weight (0 if portfolio is empty)."""
        return max(self.weights.values()) if self.weights else 0.0

    @property
    def cash(self) -> float:
        """Implied cash weight (``1.0 - gross``), floored at 0."""
        return max(0.0, 1.0 - self.gross)

    def top(self, n: int) -> List[tuple[str, float]]:
        """
        Return the *n* largest positions as ``(symbol, weight)`` pairs.

        Parameters
        ----------
        n : int
            Number of top positions to return.

        Returns
        -------
        list[tuple[str, float]]
        """
        ranked = sorted(
            self.weights.items(), key=lambda kv: kv[1], reverse=True,
        )
        return ranked[:n]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """Serialise to a plain dictionary."""
        return {
            "asof": str(self.asof),
            "weights": dict(self.weights),
            "gross": self.gross,
            "n_positions": self.n_positions,
            "reasons": self.reasons,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert to a single-row-per-symbol ``DataFrame``.

        Returns
        -------
        pd.DataFrame
            Columns: ``weight``.  Indexed by symbol, sorted descending.
        """
        if not self.weights:
            return pd.DataFrame(columns=["weight"])
        series = pd.Series(self.weights, name="weight").sort_values(
            ascending=False,
        )
        return series.to_frame()


# ---------------------------------------------------------------------------
# PortfolioConstraints
# ---------------------------------------------------------------------------


@dataclass
class PortfolioConstraints:
    """
    Tuneable constraints for portfolio construction.

    These parameters govern how many positions to hold, how
    concentrated the portfolio may be, and how aggressively it may
    turn over between rebalances.

    Attributes
    ----------
    gross_target : float
        Target gross exposure (sum of weights).  Typically ``1.0``
        for a fully-invested long-only book.
    max_positions : int
        Maximum number of positions.  Overrides ``top_quantile``
        when ``selection_mode == "top_n"``.
    min_positions : int
        Minimum positions to hold (if enough candidates exist).
        When fewer candidates pass ``score_floor``, the portfolio
        may hold fewer if ``allow_cash`` is ``True``.
    max_weight : float
        Hard cap on any single position weight.  After clipping,
        excess weight is redistributed proportionally.
    min_weight : float
        Minimum weight for an included position.  Symbols whose
        computed weight falls below this are dropped.
    score_floor : float
        Alpha scores at or below this threshold are excluded before
        selection.  Useful for removing gated (zero-score) names.
    selection_mode : ``"top_n"`` | ``"top_quantile"``
        How to select which symbols enter the portfolio.
        ``"top_n"`` takes the best *max_positions* symbols.
        ``"top_quantile"`` takes the top *top_quantile* fraction
        of the scored universe.
    top_quantile : float
        Fraction of the scored universe to include when
        ``selection_mode == "top_quantile"``.  Ignored otherwise.
    allow_cash : bool
        If ``True`` (default), the gross may be below
        ``gross_target`` when there are too few qualifying
        candidates.  If ``False``, the builder must scale up
        weights to hit ``gross_target`` regardless.
    rebalance_band : float
        Weight change below this threshold is suppressed (set to
        the current weight).  ``0.0`` disables banding.
    turnover_limit : float or None
        Maximum one-way turnover per rebalance, measured as
        ``sum(|w_new - w_old|)``.  ``None`` disables the limit.

    Examples
    --------
    >>> c = PortfolioConstraints(max_positions=30, max_weight=0.04)
    >>> c.selection_mode
    'top_n'
    """

    gross_target: float = 1.0
    max_positions: int = 50
    min_positions: int = 10
    max_weight: float = 0.05
    min_weight: float = 0.0
    score_floor: float = 0.0
    selection_mode: Literal["top_n", "top_quantile"] = "top_n"
    top_quantile: float = 0.2
    allow_cash: bool = True
    rebalance_band: float = 0.0
    turnover_limit: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate constraint invariants."""
        if self.gross_target < 0.0:
            raise ValueError(
                f"gross_target must be >= 0, got {self.gross_target}"
            )
        if self.max_positions < 1:
            raise ValueError(
                f"max_positions must be >= 1, got {self.max_positions}"
            )
        if self.min_positions < 0:
            raise ValueError(
                f"min_positions must be >= 0, got {self.min_positions}"
            )
        if self.min_positions > self.max_positions:
            raise ValueError(
                f"min_positions ({self.min_positions}) must be "
                f"<= max_positions ({self.max_positions})"
            )
        if not 0.0 < self.max_weight <= 1.0:
            raise ValueError(
                f"max_weight must be in (0, 1], got {self.max_weight}"
            )
        if self.min_weight < 0.0:
            raise ValueError(
                f"min_weight must be >= 0, got {self.min_weight}"
            )
        if self.min_weight > self.max_weight:
            raise ValueError(
                f"min_weight ({self.min_weight}) must be "
                f"<= max_weight ({self.max_weight})"
            )
        if not 0.0 < self.top_quantile <= 1.0:
            raise ValueError(
                f"top_quantile must be in (0, 1], got {self.top_quantile}"
            )
        if self.rebalance_band < 0.0:
            raise ValueError(
                f"rebalance_band must be >= 0, got {self.rebalance_band}"
            )
        if self.turnover_limit is not None and self.turnover_limit < 0.0:
            raise ValueError(
                f"turnover_limit must be >= 0, got {self.turnover_limit}"
            )


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class PortfolioConstructor(Protocol):
    """
    Protocol for portfolio construction implementations.

    Accepts an :class:`AlphaVector` (non-negative scores from the
    long-only router) and produces a :class:`TargetPortfolio` of
    desired weights subject to :class:`PortfolioConstraints`.

    Implementations may range from simple score-proportional weighting
    to full mean-variance optimisation.  They simply need to expose
    the ``build_targets`` method with the signature below.

    Parameters
    ----------
    asof : pd.Timestamp
        Rebalance date.
    alpha : AlphaVector
        Scored universe (scores ``>= 0`` after long-only routing).
    universe : list[str]
        Eligible symbols.  The builder should only allocate to
        symbols present in both *alpha* and *universe*.
    constraints : PortfolioConstraints
        Position-count, weight, and turnover limits.
    current_weights : dict[str, float] or None
        Current portfolio weights.  ``None`` on the first rebalance.
        Used for turnover and rebalance-band calculations.

    Returns
    -------
    TargetPortfolio
        Desired weights for the next holding period.

    Examples
    --------
    >>> class ScoreWeighter:
    ...     def build_targets(
    ...         self,
    ...         asof: pd.Timestamp,
    ...         alpha: AlphaVector,
    ...         universe: list[str],
    ...         constraints: PortfolioConstraints,
    ...         current_weights: dict[str, float] | None = None,
    ...     ) -> TargetPortfolio:
    ...         ...
    ...
    >>> builder: PortfolioConstructor = ScoreWeighter()
    """

    def build_targets(
        self,
        asof: pd.Timestamp,
        alpha: AlphaVector,
        universe: List[str],
        constraints: PortfolioConstraints,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> TargetPortfolio:
        """Generate target portfolio weights from alpha scores."""
        ...


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def normalize_positive(
    values: Dict[str, float],
    target_sum: float = 1.0,
) -> Dict[str, float]:
    """
    Rescale non-negative values so they sum to *target_sum*.

    Zero and negative entries are kept at ``0.0``.  If all values are
    ``<= 0`` the function returns all zeros (avoids division by zero).

    Parameters
    ----------
    values : dict[str, float]
        Symbol → raw positive value.
    target_sum : float, default ``1.0``
        Desired sum of the output values.

    Returns
    -------
    dict[str, float]
        Rescaled values summing to *target_sum* (or all zeros).

    Raises
    ------
    ValueError
        If *target_sum* is negative.

    Examples
    --------
    >>> normalize_positive({"A": 3.0, "B": 1.0, "C": 0.0}, target_sum=1.0)
    {'A': 0.75, 'B': 0.25, 'C': 0.0}
    >>> normalize_positive({"A": 0.0, "B": 0.0}, target_sum=1.0)
    {'A': 0.0, 'B': 0.0}
    """
    if target_sum < 0.0:
        raise ValueError(f"target_sum must be >= 0, got {target_sum}")

    positive = {k: max(v, 0.0) for k, v in values.items()}
    total = sum(positive.values())

    if total <= 0.0:
        return {k: 0.0 for k in values}

    scale = target_sum / total
    return {k: v * scale for k, v in positive.items()}


def clip_weights(
    weights: Dict[str, float],
    max_weight: float,
) -> Dict[str, float]:
    """
    Cap individual weights at *max_weight*, redistributing excess.

    The function iteratively clips weights that exceed *max_weight*
    and spreads the freed capacity proportionally among un-clipped
    positions.  This preserves the total sum of weights while
    respecting the cap.

    Parameters
    ----------
    weights : dict[str, float]
        Symbol → weight (must be non-negative).
    max_weight : float
        Upper bound for any single weight.  Must be in ``(0, 1]``.

    Returns
    -------
    dict[str, float]
        Adjusted weights where no entry exceeds *max_weight*.

    Raises
    ------
    ValueError
        If *max_weight* is not in ``(0, 1]``.

    Examples
    --------
    >>> clip_weights({"A": 0.60, "B": 0.30, "C": 0.10}, max_weight=0.40)
    {'A': 0.4, 'B': 0.4, 'C': 0.2}

    Notes
    -----
    The iterative approach converges because each pass either clips at
    least one new symbol or leaves the solution unchanged (stable).
    In the worst case the number of iterations equals the number of
    symbols.
    """
    if not 0.0 < max_weight <= 1.0:
        raise ValueError(f"max_weight must be in (0, 1], got {max_weight}")

    result = dict(weights)
    total = sum(result.values())

    if total <= 0.0:
        return result

    # Iterative clipping with redistribution.
    # Symbols that have been clipped are "locked" at *max_weight* so
    # subsequent redistribution rounds cannot push them back over.
    locked: set[str] = set()

    for _ in range(len(result)):
        excess = 0.0
        newly_locked: Dict[str, float] = {}
        free: Dict[str, float] = {}

        for sym, w in result.items():
            if sym in locked:
                continue  # already capped in a prior round
            if w > max_weight:
                newly_locked[sym] = max_weight
                excess += w - max_weight
            else:
                free[sym] = w

        if excess <= 0.0:
            break  # nothing to clip

        # Lock newly clipped symbols and write their capped values
        locked.update(newly_locked)
        for sym, w in newly_locked.items():
            result[sym] = w

        # Redistribute excess proportionally among free positions
        free_total = sum(free.values())
        if free_total > 0.0:
            scale = 1.0 + excess / free_total
            for sym in free:
                result[sym] = free[sym] * scale
        else:
            break  # no free capacity left

    return result


def enforce_max_positions(
    sorted_symbols: List[str],
    max_positions: int,
) -> List[str]:
    """
    Truncate a ranked symbol list to at most *max_positions* entries.

    The input list should already be sorted in descending order of
    desirability (e.g. by alpha score).  This function simply takes
    the top slice.

    Parameters
    ----------
    sorted_symbols : list[str]
        Symbols ranked from best to worst.
    max_positions : int
        Maximum number of positions to retain.

    Returns
    -------
    list[str]
        The top *max_positions* symbols (or fewer if the input is
        shorter).

    Raises
    ------
    ValueError
        If *max_positions* is less than 1.

    Examples
    --------
    >>> enforce_max_positions(["AAPL", "MSFT", "GOOG", "TSLA"], 2)
    ['AAPL', 'MSFT']
    >>> enforce_max_positions(["AAPL"], 5)
    ['AAPL']
    """
    if max_positions < 1:
        raise ValueError(f"max_positions must be >= 1, got {max_positions}")
    return sorted_symbols[:max_positions]
