"""
Alpha layer types and utilities for a long-only strategy.

Defines alpha model scoring primitives, the ``AlphaModel`` protocol, and
cross-sectional helper functions used by every alpha sub-model and the
ensemble router.

Design notes
------------
* Scores are *unsigned* after routing (``score >= 0``).  The ``raw`` field
  preserves the original model output which may be negative (e.g. a
  mean-reversion z-score).
* ``confidence`` lives in [0, 1] and gates downstream sizing / ensemble
  weight.
* Utility helpers (``winsorize``, ``cross_sectional_zscore``, …) are
  pure functions with no side-effects so they compose cleanly in
  feature pipelines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Protocol, runtime_checkable

import pandas as pd

from ..regime.stock_types import FeatureBundle

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AlphaModelName(str, Enum):
    """
    Registry of available alpha sub-models.

    Each value corresponds to a concrete ``AlphaModel`` implementation
    that can be selected by the strategy's configuration layer.

    Values
    ------
    MOMENTUM : str
        Trend-following / relative-strength model.
    MEANREV : str
        Mean-reversion model (e.g. z-score of price vs moving average).
    LOWVOL : str
        Low-volatility anomaly model (favour lower-risk names).
    """

    MOMENTUM = "momentum"
    MEANREV = "meanrev"
    LOWVOL = "lowvol"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AlphaScore:
    """
    Alpha score for a single symbol produced by one alpha model.

    Attributes
    ----------
    symbol : str
        Ticker symbol this score belongs to.
    score : float
        Model score.  May be negative when produced by an alpha model;
        the long-only router downstream clamps to ``>= 0`` before the
        portfolio optimiser / sizer consumes it.
    raw : float
        Raw model output *before* any transformation.  May be negative
        (e.g. a mean-reversion z-score of −1.5).  Preserved for
        diagnostics and ensemble blending.
    confidence : float
        Model-reported confidence in ``[0, 1]``.  A value of 0 means the
        model has no conviction; 1 means full conviction.  Downstream
        layers may scale position size by this factor.
    reasons : dict[str, float]
        Breakdown of contributing sub-signals.  Keys are human-readable
        feature names; values are the signal magnitudes.  Useful for
        attribution and debugging.

    Examples
    --------
    >>> s = AlphaScore(
    ...     symbol="AAPL",
    ...     score=0.82,
    ...     raw=0.82,
    ...     confidence=0.70,
    ...     reasons={"mom21": 0.45, "mom63": 0.37},
    ... )
    """

    symbol: str
    score: float
    raw: float
    confidence: float
    reasons: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate invariants after initialisation."""
        if not self.symbol:
            raise ValueError("symbol cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )

    def __repr__(self) -> str:
        return (
            f"AlphaScore(symbol={self.symbol!r}, score={self.score:.4f}, "
            f"raw={self.raw:.4f}, confidence={self.confidence:.2f})"
        )


@dataclass
class AlphaVector:
    """
    Cross-sectional collection of alpha scores at a point in time.

    This is the primary output of a single alpha model's ``predict``
    call, and also the input to the ensemble / routing layer.

    Attributes
    ----------
    asof : pd.Timestamp
        Observation timestamp.  All scores share this reference date.
    scores : dict[str, AlphaScore]
        Mapping from symbol → ``AlphaScore``.

    Examples
    --------
    >>> vec = AlphaVector(
    ...     asof=pd.Timestamp("2024-06-15"),
    ...     scores={"AAPL": AlphaScore(...), "MSFT": AlphaScore(...)},
    ... )
    >>> vec["AAPL"].score
    0.82
    """

    asof: pd.Timestamp
    scores: Dict[str, AlphaScore] = field(default_factory=dict)

    # -- container helpers ---------------------------------------------------

    def __len__(self) -> int:
        """Return number of scored symbols."""
        return len(self.scores)

    def __contains__(self, symbol: str) -> bool:
        """Check whether *symbol* has a score."""
        return symbol in self.scores

    def __getitem__(self, symbol: str) -> AlphaScore:
        """Retrieve the ``AlphaScore`` for *symbol*."""
        return self.scores[symbol]

    def __iter__(self):
        """Iterate over scored symbols."""
        return iter(self.scores)

    def get(
        self, symbol: str, default: AlphaScore | None = None
    ) -> AlphaScore | None:
        """Get score for *symbol* with an optional *default*."""
        return self.scores.get(symbol, default)

    # -- derived views -------------------------------------------------------

    @property
    def symbols(self) -> List[str]:
        """Return list of all scored symbols."""
        return list(self.scores.keys())

    def top(self, n: int) -> List[AlphaScore]:
        """
        Return the *n* highest-scoring symbols (descending by ``score``).

        Parameters
        ----------
        n : int
            Number of top scores to return.

        Returns
        -------
        list[AlphaScore]
        """
        return sorted(
            self.scores.values(), key=lambda s: s.score, reverse=True
        )[:n]

    def to_dict(self) -> Dict:
        """Serialise to a plain dictionary."""
        return {
            "asof": str(self.asof),
            "scores": {
                sym: {
                    "symbol": sc.symbol,
                    "score": sc.score,
                    "raw": sc.raw,
                    "confidence": sc.confidence,
                    "reasons": sc.reasons,
                }
                for sym, sc in self.scores.items()
            },
        }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert to a ``DataFrame`` for analysis.

        Returns
        -------
        pd.DataFrame
            Columns: ``score``, ``raw``, ``confidence``, plus one column
            per unique reason key.  Indexed by symbol.
        """
        if not self.scores:
            return pd.DataFrame(
                columns=["score", "raw", "confidence"]
            )

        records = []
        for sym, sc in self.scores.items():
            records.append(
                {
                    "symbol": sym,
                    "score": sc.score,
                    "raw": sc.raw,
                    "confidence": sc.confidence,
                    **sc.reasons,
                }
            )
        return pd.DataFrame(records).set_index("symbol")


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class AlphaModel(Protocol):
    """
    Protocol that every alpha sub-model must satisfy.

    Implementations are free to carry any internal state (lookback
    buffers, trained weights, etc.).  They simply need to expose a
    ``name`` property and a ``predict`` method with the signature below.

    The return value is a *per-symbol* score dict.  Ensemble routing
    (blending weights, long-only clamping, regime gating) happens in a
    separate layer so that individual models stay clean and testable.

    Examples
    --------
    >>> class MomentumAlpha:
    ...     @property
    ...     def name(self) -> str:
    ...         return AlphaModelName.MOMENTUM.value
    ...
    ...     def predict(
    ...         self,
    ...         asof: pd.Timestamp,
    ...         features: FeatureBundle,
    ...         universe: list[str],
    ...     ) -> dict[str, AlphaScore]:
    ...         ...
    ...
    >>> model: AlphaModel = MomentumAlpha()  # structural match
    """

    @property
    def name(self) -> str:
        """Human-readable model identifier (e.g. ``'momentum'``)."""
        ...

    def predict(
        self,
        asof: pd.Timestamp,
        features: FeatureBundle,
        universe: List[str],
    ) -> Dict[str, AlphaScore]:
        """
        Generate per-symbol alpha scores.

        Parameters
        ----------
        asof : pd.Timestamp
            Evaluation date.  Models must not peek beyond this date.
        features : FeatureBundle
            Cross-sectional feature snapshot.  Use
            ``features.get(symbol, feature_name)`` for scalar lookups.
        universe : list[str]
            Symbols to score.  The model should return a key for every
            symbol it has an opinion on (may be a subset of *universe*
            if data is missing).

        Returns
        -------
        dict[str, AlphaScore]
            Per-symbol scores.  Routing / clamping to ``score >= 0``
            is done downstream, so ``raw`` may be negative.
        """
        ...


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def clamp01(x: float) -> float:
    """
    Clamp *x* to the ``[0, 1]`` interval.

    Parameters
    ----------
    x : float
        Input value.

    Returns
    -------
    float

    Examples
    --------
    >>> clamp01(0.5)
    0.5
    >>> clamp01(-0.3)
    0.0
    >>> clamp01(1.7)
    1.0
    """
    return max(0.0, min(1.0, x))


def keep_positive(x: float) -> float:
    """
    Return ``max(x, 0)`` — the long-only half-rectifier.

    Useful for converting a signed alpha signal into a long-only score.

    Parameters
    ----------
    x : float
        Raw signal value.

    Returns
    -------
    float

    Examples
    --------
    >>> keep_positive(1.2)
    1.2
    >>> keep_positive(-0.5)
    0.0
    """
    return max(x, 0.0)


def winsorize(x: float, lo: float = -3.0, hi: float = 3.0) -> float:
    """
    Clip *x* to ``[lo, hi]``.

    Winsorising prevents extreme outliers from dominating downstream
    z-scores or portfolio weights.

    Parameters
    ----------
    x : float
        Input value.
    lo : float, default ``-3.0``
        Lower bound.
    hi : float, default ``3.0``
        Upper bound.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If ``lo > hi``.

    Examples
    --------
    >>> winsorize(5.0)
    3.0
    >>> winsorize(-4.0)
    -3.0
    >>> winsorize(1.5)
    1.5
    """
    if lo > hi:
        raise ValueError(f"lo ({lo}) must be <= hi ({hi})")
    return max(lo, min(hi, x))


def cross_sectional_zscore(
    values: Dict[str, float],
) -> Dict[str, float]:
    """
    Cross-sectional z-score normalisation.

    Computes ``(x - mean) / std`` across the *non-NaN* values in the
    dict.  If the standard deviation is effectively zero (< 1e-12) the
    function returns zeros for every key to avoid division-by-zero
    artefacts.

    Parameters
    ----------
    values : dict[str, float]
        Mapping from symbol → raw value.  ``NaN`` entries are passed
        through as ``NaN`` in the output.

    Returns
    -------
    dict[str, float]
        Same keys, z-scored values.

    Examples
    --------
    >>> cross_sectional_zscore({"A": 1.0, "B": 3.0, "C": 5.0})
    {'A': -1.2247..., 'B': 0.0, 'C': 1.2247...}
    >>> cross_sectional_zscore({"A": 2.0, "B": float('nan')})
    {'A': 0.0, 'B': nan}
    """
    finite_vals = [v for v in values.values() if not math.isnan(v)]

    if len(finite_vals) < 2:
        # Not enough data — return zeros (or NaN passthrough).
        return {
            k: (float("nan") if math.isnan(v) else 0.0)
            for k, v in values.items()
        }

    mean = sum(finite_vals) / len(finite_vals)
    var = sum((v - mean) ** 2 for v in finite_vals) / len(finite_vals)
    std = math.sqrt(var)

    if std < 1e-12:
        return {
            k: (float("nan") if math.isnan(v) else 0.0)
            for k, v in values.items()
        }

    return {
        k: (float("nan") if math.isnan(v) else (v - mean) / std)
        for k, v in values.items()
    }


def rank_to_unit(values: Dict[str, float]) -> Dict[str, float]:
    """
    Map values to ``[0, 1]`` percentile ranks (cross-sectional).

    ``NaN`` entries are passed through unchanged.  Ties receive the
    average rank.

    Parameters
    ----------
    values : dict[str, float]
        Symbol → raw value mapping.

    Returns
    -------
    dict[str, float]
        Symbol → percentile rank in ``[0, 1]``.

    Examples
    --------
    >>> rank_to_unit({"A": 10.0, "B": 30.0, "C": 20.0})
    {'A': 0.0, 'B': 1.0, 'C': 0.5}
    >>> rank_to_unit({"A": 5.0, "B": float('nan'), "C": 5.0})
    {'A': 0.5, 'B': nan, 'C': 0.5}
    """
    # Separate finite from NaN.
    finite: Dict[str, float] = {
        k: v for k, v in values.items() if not math.isnan(v)
    }
    n = len(finite)

    if n <= 1:
        # One or zero finite values — everyone gets 0 (or NaN).
        return {
            k: (float("nan") if math.isnan(v) else 0.0)
            for k, v in values.items()
        }

    # Sort keys by value to assign ordinal ranks (0-based).
    sorted_keys = sorted(finite, key=lambda k: finite[k])

    # Build rank lookup — average rank for ties.
    rank_map: Dict[str, float] = {}
    i = 0
    while i < n:
        j = i
        # Find block of ties.
        while j < n and finite[sorted_keys[j]] == finite[sorted_keys[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0
        for idx in range(i, j):
            rank_map[sorted_keys[idx]] = avg_rank
        i = j

    # Normalise to [0, 1].
    max_rank = n - 1.0
    return {
        k: (
            float("nan")
            if math.isnan(v)
            else rank_map[k] / max_rank
        )
        for k, v in values.items()
    }
