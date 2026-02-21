"""
Cross-sectional utilities for feature snapshots.

All functions operate on 1-D ``np.ndarray`` inputs, preserve ``NaN``
values, and handle edge cases such as zero standard deviation or
all-NaN slices.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Cross-sectional z-score
# ---------------------------------------------------------------------------


def cross_sectional_zscore(
    x: np.ndarray,
    *,
    ddof: int = 0,
    min_finite: int = 2,
) -> np.ndarray:
    """Standardise a cross-sectional vector to zero mean, unit variance.

    Parameters
    ----------
    x : np.ndarray (N,)
        Raw feature values.  ``NaN`` entries are preserved.
    ddof : int
        Delta degrees of freedom for ``np.std``.
    min_finite : int
        Minimum number of finite values required.  If fewer exist, the
        entire output is ``NaN``.

    Returns
    -------
    np.ndarray (N,)
        Z-scores.  ``NaN`` for originally-missing entries.  If the
        standard deviation is ≈ 0, finite entries are returned as 0.
    """
    out = np.full_like(x, np.nan, dtype=np.float64)
    mask = np.isfinite(x)
    n_finite = int(mask.sum())

    if n_finite < min_finite:
        return out

    vals = x[mask]
    mu = float(np.mean(vals))
    sigma = float(np.std(vals, ddof=ddof))

    if sigma < 1e-12:
        out[mask] = 0.0
    else:
        out[mask] = (vals - mu) / sigma

    return out


# ---------------------------------------------------------------------------
# Percentile rank
# ---------------------------------------------------------------------------


def percentile_rank(x: np.ndarray) -> np.ndarray:
    """Rank values to ``[0, 1]`` using average-rank tie-breaking.

    Parameters
    ----------
    x : np.ndarray (N,)
        Raw values.  ``NaN`` entries are preserved.

    Returns
    -------
    np.ndarray (N,)
        Percentile ranks in ``[0, 1]``.  ``NaN`` for missing entries.
        With a single finite value, returns 0.5 for that entry.
    """
    out = np.full_like(x, np.nan, dtype=np.float64)
    mask = np.isfinite(x)
    n_finite = int(mask.sum())

    if n_finite == 0:
        return out

    vals = x[mask]
    # Average-rank without scipy: sort + inverse mapping
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(n_finite, dtype=np.float64)

    # Adjust for ties: group identical values and assign average rank
    sorted_vals = vals[order]
    i = 0
    while i < n_finite:
        j = i + 1
        while j < n_finite and sorted_vals[j] == sorted_vals[i]:
            j += 1
        # Indices i..j-1 share the same value → average rank
        if j > i + 1:
            avg_rank = np.mean(np.arange(i, j, dtype=np.float64))
            for idx in range(i, j):
                ranks[order[idx]] = avg_rank
        i = j

    # Scale to [0, 1]
    if n_finite == 1:
        out[mask] = 0.5
    else:
        out[mask] = ranks / (n_finite - 1)

    return out


# ---------------------------------------------------------------------------
# Winsorize
# ---------------------------------------------------------------------------


def winsorize(
    x: np.ndarray,
    lo: float = -3.0,
    hi: float = 3.0,
) -> np.ndarray:
    """Clip finite values to ``[lo, hi]``, preserving ``NaN``.

    Parameters
    ----------
    x : np.ndarray (N,)
        Input values.
    lo, hi : float
        Lower and upper clip bounds.

    Returns
    -------
    np.ndarray (N,)
        Clipped copy.
    """
    out = x.copy().astype(np.float64)
    mask = np.isfinite(out)
    out[mask] = np.clip(out[mask], lo, hi)
    return out
