"""
Typed containers for cross-sectional feature data.

The central object is :class:`FeatureBundle` — an immutable-ish snapshot of
every feature value for every symbol at a single point in time.  It is the
canonical hand-off format between the feature engine and downstream consumers
(alpha models, regime detectors, portfolio constructors).

Design goals
------------
* **O(1) lookups** by (symbol, feature) via pre-built index dicts.
* **NumPy-native** storage (``float32`` matrix) so downstream code can stay
  vectorised without copies.
* **Explicit missingness** via a boolean ``mask`` array — callers never have
  to guess whether a NaN is "real" or "hasn't arrived yet".
* Minimal dependencies: only ``numpy`` and ``pandas`` (for ``pd.Timestamp``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# FeatureBundle
# ---------------------------------------------------------------------------

@dataclass
class FeatureBundle:
    """Cross-sectional snapshot of features at a single timestamp.

    Parameters
    ----------
    asof : pd.Timestamp
        The bar / event timestamp this snapshot corresponds to.
    symbols : list[str]
        Ordered list of symbols (length *N*).  The row order of ``X``
        matches this list exactly.
    feature_names : list[str]
        Ordered list of feature names (length *F*).  The column order
        of ``X`` matches this list exactly.
    X : np.ndarray, shape (N, F), dtype float32
        Feature matrix.  Missing values are stored as ``np.nan``.
    mask : np.ndarray, shape (N, F), dtype bool, optional
        ``True`` where the corresponding entry in ``X`` is **missing**
        (i.e. should be treated as unavailable).  Built automatically
        from ``X`` in ``__post_init__`` if not supplied.

    Derived attributes (built in ``__post_init__``)
    ------------------------------------------------
    sym2i : dict[str, int]
        Maps each symbol to its row index.
    feat2j : dict[str, int]
        Maps each feature name to its column index.

    Examples
    --------
    >>> fb = FeatureBundle(
    ...     asof=pd.Timestamp("2024-01-02"),
    ...     symbols=["AAPL", "MSFT"],
    ...     feature_names=["mom_20", "vol_20"],
    ...     X=np.array([[0.05, 0.12], [0.03, np.nan]], dtype=np.float32),
    ... )
    >>> fb.get("AAPL", "vol_20")
    0.11999...
    >>> fb.valid_ratio("vol_20")
    0.5
    """

    asof: pd.Timestamp
    symbols: List[str]
    feature_names: List[str]
    X: np.ndarray
    mask: np.ndarray | None = None

    # -- derived indices (populated by __post_init__) -----------------------
    sym2i: Dict[str, int] = field(init=False, repr=False)
    feat2j: Dict[str, int] = field(init=False, repr=False)

    # -- post-init validation & index building ------------------------------

    def __post_init__(self) -> None:
        N = len(self.symbols)
        F = len(self.feature_names)

        # Coerce X to float32 contiguous array.
        if not isinstance(self.X, np.ndarray):
            self.X = np.asarray(self.X, dtype=np.float32)
        elif self.X.dtype != np.float32:
            self.X = self.X.astype(np.float32, copy=False)

        # Shape validation.
        if self.X.ndim != 2:
            raise ValueError(
                f"X must be 2-D, got shape {self.X.shape}"
            )
        if self.X.shape != (N, F):
            raise ValueError(
                f"X shape {self.X.shape} does not match "
                f"({N} symbols, {F} features)"
            )

        # Build or validate mask.
        if self.mask is None:
            self.mask = np.isnan(self.X)
        else:
            if not isinstance(self.mask, np.ndarray):
                self.mask = np.asarray(self.mask, dtype=bool)
            elif self.mask.dtype != bool:
                self.mask = self.mask.astype(bool, copy=False)
            if self.mask.shape != (N, F):
                raise ValueError(
                    f"mask shape {self.mask.shape} does not match "
                    f"X shape {self.X.shape}"
                )

        # Duplicate checks.
        if len(set(self.symbols)) != N:
            raise ValueError("symbols list contains duplicates")
        if len(set(self.feature_names)) != F:
            raise ValueError("feature_names list contains duplicates")

        # Build O(1) index dicts.
        self.sym2i = {s: i for i, s in enumerate(self.symbols)}
        self.feat2j = {f: j for j, f in enumerate(self.feature_names)}

    # -- scalar access ------------------------------------------------------

    def get(self, symbol: str, feature: str) -> float:
        """Return the value for (*symbol*, *feature*), or ``np.nan`` if
        the symbol or feature is unknown or the entry is masked.

        Parameters
        ----------
        symbol : str
        feature : str

        Returns
        -------
        float
        """
        i = self.sym2i.get(symbol)
        j = self.feat2j.get(feature)
        if i is None or j is None:
            return np.nan
        return float(self.X[i, j])

    # -- slice access -------------------------------------------------------

    def row(self, symbol: str) -> np.ndarray:
        """Return the feature vector for *symbol* as a 1-D float32 array.

        Returns a **copy** — callers may mutate freely.

        Raises
        ------
        KeyError
            If *symbol* is not in the bundle.
        """
        i = self.sym2i[symbol]
        return self.X[i, :].copy()

    def col(self, feature: str) -> np.ndarray:
        """Return the cross-section for *feature* as a 1-D float32 array.

        Returns a **copy** — callers may mutate freely.

        Raises
        ------
        KeyError
            If *feature* is not in the bundle.
        """
        j = self.feat2j[feature]
        return self.X[:, j].copy()

    # -- quality metrics ----------------------------------------------------

    def valid_ratio(self, feature: str) -> float:
        """Fraction of symbols that have a **non-missing** value for *feature*.

        Returns 0.0 if the feature is unknown or the bundle is empty.

        Parameters
        ----------
        feature : str

        Returns
        -------
        float
            Value in [0.0, 1.0].
        """
        j = self.feat2j.get(feature)
        if j is None or len(self.symbols) == 0:
            return 0.0
        assert self.mask is not None          # guaranteed by __post_init__
        n_valid = int(np.count_nonzero(~self.mask[:, j]))
        return n_valid / len(self.symbols)

    # -- serialisation helpers ----------------------------------------------

    def to_dict_of_dicts(self) -> Dict[str, Dict[str, float]]:
        """Convert to ``{symbol: {feature: value, …}, …}``.

        Useful for logging, JSON serialisation, or handing off to code
        that expects plain Python dicts.  Missing entries are included as
        ``float('nan')``.
        """
        return {
            sym: {
                feat: float(self.X[i, j])
                for j, feat in enumerate(self.feature_names)
            }
            for i, sym in enumerate(self.symbols)
        }

    # -- convenience --------------------------------------------------------

    @property
    def n_symbols(self) -> int:
        """Number of symbols (rows)."""
        return len(self.symbols)

    @property
    def n_features(self) -> int:
        """Number of features (columns)."""
        return len(self.feature_names)

    @property
    def shape(self) -> tuple[int, int]:
        """(N, F) tuple mirroring ``X.shape``."""
        return (self.n_symbols, self.n_features)

    def __repr__(self) -> str:
        assert self.mask is not None
        pct = 100.0 * (1.0 - self.mask.mean()) if self.mask.size else 100.0
        return (
            f"FeatureBundle(asof={self.asof}, "
            f"shape=({self.n_symbols}, {self.n_features}), "
            f"valid={pct:.1f}%)"
        )
