"""
Event-driven feature engine.

Ingests bars one at a time via :meth:`update`, maintains per-symbol
rolling windows, and materialises a cross-sectional
:class:`~engine.features.types.FeatureBundle` on demand via
:meth:`get_feature_bundle`.

Architecture
------------
::

    bars (one-by-one, per symbol)
           │
           ▼
      ┌──────────────────────┐
      │     FeatureEngine    │
      │  per-symbol rolling  │
      │  windows (close,     │
      │  ret, vol, dvol,     │
      │  rv20_hist)          │
      └──────────┬───────────┘
                 │  get_feature_bundle(asof)
                 ▼
          FeatureBundle(N, F)
            float32 matrix

*   ``update()`` is **O(1)** per bar — it appends to deques and pushes
    one rv20 value.  No feature matrix is built until requested.
*   ``get_feature_bundle()`` is **O(N × F)** — it snapshots each
    symbol's windows once and computes all features via NumPy slices.
*   Not thread-safe.  Designed for a single-threaded event loop.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .definitions import (
    FeatureSpec,
    default_feature_specs,
    max_lookback as _max_lookback_from_specs,
)
from .rolling import RollingWindow, safe_div
from .types import FeatureBundle

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SQRT_252: float = 15.874507866          # √252 — daily → annual vol
_EMPTY: np.ndarray = np.empty(0, dtype=np.float64)
_MIN_VALID_FRAC_20: int = 15             # min valid returns for rv20 (75 %)
_MIN_VALID_FRAC_60: int = 45             # min valid returns for rv60 (75 %)
_MIN_RV20_HIST: int = 60                 # min rv20 samples for rv20_z


# ---------------------------------------------------------------------------
# Per-symbol mutable state
# ---------------------------------------------------------------------------

@dataclass
class _SymbolState:
    """Rolling-window state for a single symbol.

    All windows are sized to ``max_lookback`` so every feature can
    freely slice into the last *K* values of any series.
    """

    close: RollingWindow
    ret: RollingWindow
    volume: RollingWindow
    dollar_vol: RollingWindow
    rv20_hist: RollingWindow        # stores computed rv20 (ann.) per bar
    last_close: float = np.nan      # previous bar's close for return calc
    bars_seen: int = 0              # total bars ingested


# ---------------------------------------------------------------------------
# Bar accessor
# ---------------------------------------------------------------------------

def _get(bar: Any, field: str, default: Any = None) -> Any:
    """Extract *field* from *bar* — tries attribute then dict access."""
    if hasattr(bar, field):
        return getattr(bar, field)
    if isinstance(bar, dict):
        return bar.get(field, default)
    return default


# ---------------------------------------------------------------------------
# FeatureEngine
# ---------------------------------------------------------------------------

class FeatureEngine:
    """Event-driven per-symbol feature engine.

    Parameters
    ----------
    universe : list[str]
        Fixed, ordered list of tradeable symbols.  This determines the
        row order of every :class:`FeatureBundle` produced.
    feature_specs : list[FeatureSpec], optional
        Feature catalogue.  Defaults to :func:`default_feature_specs`.
    readiness_threshold : float
        Fraction of universe that must have ``bars_seen >= max_lookback``
        before :meth:`is_ready` returns ``True``.  Default 0.80 (80 %).

    Examples
    --------
    >>> engine = FeatureEngine(["AAPL", "MSFT", "GOOG"])
    >>> for bar in daily_bar_stream:
    ...     engine.update(bar)
    >>> if engine.is_ready():
    ...     bundle = engine.get_feature_bundle(bar.asof)
    ...     print(bundle.get("AAPL", "mom63"))
    """

    def __init__(
        self,
        universe: List[str],
        feature_specs: List[FeatureSpec] | None = None,
        readiness_threshold: float = 0.80,
        warmup_bars: int | None = None,
    ) -> None:
        if not universe:
            raise ValueError("universe must be non-empty")

        self._specs: List[FeatureSpec] = list(
            feature_specs or default_feature_specs()
        )
        self._feature_names: List[str] = [s.name for s in self._specs]
        self._feat2j: Dict[str, int] = {
            name: j for j, name in enumerate(self._feature_names)
        }
        self._universe: List[str] = list(universe)
        self._sym2i: Dict[str, int] = {
            s: i for i, s in enumerate(self._universe)
        }
        self._readiness_threshold: float = readiness_threshold

        # Derived sizes
        self._max_lb: int = _max_lookback_from_specs(self._specs) or 252
        self._n_sym: int = len(self._universe)
        self._n_feat: int = len(self._feature_names)

        # Warmup: how many bars a symbol needs before the engine is
        # considered "ready".  Defaults to 200 (enough for sma200 and
        # trend_gap).  Features that need more history (e.g. rv20_z
        # with lookback 252) will simply be NaN until they accumulate
        # enough data; the engine can start producing bundles earlier.
        self._warmup_bars: int = (
            warmup_bars if warmup_bars is not None
            else min(self._max_lb, 200)
        )

        # Per-symbol state
        self._state: Dict[str, _SymbolState] = {
            sym: self._make_state() for sym in self._universe
        }

        # Global bookkeeping
        self._last_asof: Optional[pd.Timestamp] = None
        self._asof_count: int = 0  # how many symbols updated at current asof

    # ------------------------------------------------------------------
    # Internal factory
    # ------------------------------------------------------------------

    def _make_state(self) -> _SymbolState:
        """Create a fresh per-symbol state with correctly sized windows."""
        sz = self._max_lb
        return _SymbolState(
            close=RollingWindow(sz),
            ret=RollingWindow(sz),
            volume=RollingWindow(sz),
            dollar_vol=RollingWindow(sz),
            rv20_hist=RollingWindow(sz),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def universe(self) -> List[str]:
        """Ordered symbol list (copy)."""
        return list(self._universe)

    @property
    def feature_names(self) -> List[str]:
        """Ordered feature-name list (copy)."""
        return list(self._feature_names)

    @property
    def last_asof(self) -> Optional[pd.Timestamp]:
        """Timestamp of the most recent bar ingested."""
        return self._last_asof

    @property
    def n_features(self) -> int:
        """Number of features in the catalogue."""
        return self._n_feat

    @property
    def max_lookback(self) -> int:
        """Maximum lookback across all feature specs."""
        return self._max_lb

    @property
    def warmup_bars(self) -> int:
        """Minimum bars per symbol before the engine is ready."""
        return self._warmup_bars

    @property
    def bars_received(self) -> Dict[str, int]:
        """Per-symbol count of bars ingested so far."""
        return {sym: st.bars_seen for sym, st in self._state.items()}

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------

    def update(self, bar: Any) -> None:
        """Ingest a single bar and update rolling windows.

        Parameters
        ----------
        bar
            Must expose ``symbol`` (str), ``close`` (float),
            ``volume`` (float), and ``asof`` (pd.Timestamp).
            Supports both attribute and dict-style access.

        Notes
        -----
        *   Bars for symbols outside the engine's universe are silently
            ignored.
        *   The daily return is computed as ``close / last_close − 1``.
            The very first bar for a symbol produces a ``NaN`` return
            (no previous close).
        *   After every bar, if the return window has ≥ 20 values, a
            fresh ``rv20`` is computed and pushed to ``rv20_hist`` for
            later use by ``rv20_z``.
        """
        sym: str | None = _get(bar, "symbol")
        if sym is None or sym not in self._sym2i:
            return

        close: float = float(_get(bar, "close"))
        volume: float = float(_get(bar, "volume", 0.0))
        asof = _get(bar, "asof")

        st = self._state[sym]

        # ── Asof tracking ────────────────────────────────────────────
        if asof != self._last_asof:
            self._last_asof = asof
            self._asof_count = 1
        else:
            self._asof_count += 1

        # ── Raw series ───────────────────────────────────────────────
        st.close.add(close)
        st.volume.add(volume)
        st.dollar_vol.add(close * volume)

        # ── Return (simple) ──────────────────────────────────────────
        if math.isfinite(st.last_close) and st.last_close > 0.0:
            ret = close / st.last_close - 1.0
        else:
            ret = np.nan
        st.ret.add(ret)

        # ── Push rv20 into history (for rv20_z downstream) ───────────
        n_ret = len(st.ret)
        if n_ret >= 20:
            r20 = st.ret.values()[-20:]
            valid = r20[~np.isnan(r20)]
            if len(valid) >= _MIN_VALID_FRAC_20:
                rv20_val = float(np.std(valid, ddof=1)) * _SQRT_252
                st.rv20_hist.add(rv20_val)

        # ── Bookkeeping ──────────────────────────────────────────────
        st.last_close = close
        st.bars_seen += 1

    # ------------------------------------------------------------------
    # is_ready
    # ------------------------------------------------------------------

    def is_ready(self, asof: Optional[pd.Timestamp] = None) -> bool:
        """Check whether the engine has enough data to produce features.

        A symbol is *ready* when ``bars_seen >= warmup_bars``.  The
        engine is ready when at least ``readiness_threshold`` of the
        universe meets this criterion.

        Note: ``warmup_bars`` (default 200) is deliberately lower than
        ``max_lookback`` (252).  Features that require longer history
        (e.g. ``rv20_z``) will be NaN until their own lookback is
        satisfied, but the engine can start producing bundles earlier
        so the strategy has more trading days.

        Parameters
        ----------
        asof : pd.Timestamp, optional
            Unused; accepted for interface symmetry.

        Returns
        -------
        bool
        """
        n_ready = sum(
            1 for st in self._state.values()
            if st.bars_seen >= self._warmup_bars
        )
        return n_ready >= self._n_sym * self._readiness_threshold

    # ------------------------------------------------------------------
    # get_feature_bundle
    # ------------------------------------------------------------------

    def get_feature_bundle(
        self,
        asof: pd.Timestamp,
        universe: List[str] | None = None,
    ) -> FeatureBundle:
        """Build a cross-sectional feature snapshot at *asof*.

        Parameters
        ----------
        asof : pd.Timestamp
            Timestamp to stamp the bundle with.
        universe : list[str], optional
            Subset of symbols to include.  Defaults to the full engine
            universe.

        Returns
        -------
        FeatureBundle
            Matrix ``X`` has shape ``(N, F)`` with ``dtype=float32``.
            Missing / uncomputable entries are ``NaN`` with the mask
            set to ``True``.
        """
        syms = universe if universe is not None else self._universe
        N = len(syms)
        F = self._n_feat

        X = np.full((N, F), np.nan, dtype=np.float32)

        for i, sym in enumerate(syms):
            st = self._state.get(sym)
            if st is None or st.bars_seen == 0:
                continue  # row stays all-NaN
            self._compute_symbol_features(st, X[i, :])

        return FeatureBundle(
            asof=asof,
            symbols=list(syms),
            feature_names=list(self._feature_names),
            X=X,
        )

    # ------------------------------------------------------------------
    # Per-symbol feature computation (hot path)
    # ------------------------------------------------------------------

    def _compute_symbol_features(
        self,
        st: _SymbolState,
        out: np.ndarray,
    ) -> None:
        """Compute all features for one symbol into *out* (shape F,).

        The output array is pre-filled with ``NaN``; this method only
        overwrites entries it can compute.  Shared intermediates (SMA,
        rv) are computed once and reused.
        """
        # ── Snapshot arrays (one copy each) ──────────────────────────
        ca = st.close.values() if len(st.close) > 0 else _EMPTY
        ra = st.ret.values() if len(st.ret) > 0 else _EMPTY
        da = st.dollar_vol.values() if len(st.dollar_vol) > 0 else _EMPTY
        ha = st.rv20_hist.values() if len(st.rv20_hist) > 0 else _EMPTY

        nc = len(ca)
        nr = len(ra)
        nd = len(da)
        nh = len(ha)

        if nc == 0:
            return

        cur: float = float(ca[-1])

        # ── Shared intermediates ─────────────────────────────────────
        sma20: float = float(np.mean(ca[-20:])) if nc >= 20 else np.nan
        sma50: float = float(np.mean(ca[-50:])) if nc >= 50 else np.nan
        sma200: float = float(np.mean(ca[-200:])) if nc >= 200 else np.nan

        # rv20 (daily + annualised)
        if nr >= 20:
            v20 = ra[-20:]
            v20 = v20[~np.isnan(v20)]
            if len(v20) >= _MIN_VALID_FRAC_20:
                rv20_daily: float = float(np.std(v20, ddof=1))
                rv20_ann: float = rv20_daily * _SQRT_252
            else:
                rv20_daily = np.nan
                rv20_ann = np.nan
        else:
            rv20_daily = np.nan
            rv20_ann = np.nan

        # rv60 (annualised)
        if nr >= 60:
            v60 = ra[-60:]
            v60 = v60[~np.isnan(v60)]
            rv60_ann: float = (
                float(np.std(v60, ddof=1)) * _SQRT_252
                if len(v60) >= _MIN_VALID_FRAC_60
                else np.nan
            )
        else:
            rv60_ann = np.nan

        # ── Write features by catalogue index ────────────────────────
        _j = self._feat2j.get

        # -- ret1: close[t] / close[t-1] − 1 -------------------------
        j = _j("ret1")
        if j is not None and nc >= 2 and ca[-2] > 0:
            out[j] = ca[-1] / ca[-2] - 1.0

        # -- mom5: close[t] / close[t-5] − 1 -------------------------
        j = _j("mom5")
        if j is not None and nc >= 6 and ca[-6] > 0:
            out[j] = ca[-1] / ca[-6] - 1.0

        # -- mom21: close[t] / close[t-21] − 1 -----------------------
        j = _j("mom21")
        if j is not None and nc >= 22 and ca[-22] > 0:
            out[j] = ca[-1] / ca[-22] - 1.0

        # -- mom63: close[t] / close[t-63] − 1 -----------------------
        j = _j("mom63")
        if j is not None and nc >= 64 and ca[-64] > 0:
            out[j] = ca[-1] / ca[-64] - 1.0

        # -- sma20 ----------------------------------------------------
        j = _j("sma20")
        if j is not None and math.isfinite(sma20):
            out[j] = sma20

        # -- sma50 ----------------------------------------------------
        j = _j("sma50")
        if j is not None and math.isfinite(sma50):
            out[j] = sma50

        # -- sma200 ---------------------------------------------------
        j = _j("sma200")
        if j is not None and math.isfinite(sma200):
            out[j] = sma200

        # -- trend_gap: (sma50 − sma200) / sma200 --------------------
        j = _j("trend_gap")
        if (
            j is not None
            and math.isfinite(sma50)
            and math.isfinite(sma200)
        ):
            out[j] = safe_div(sma50 - sma200, sma200)

        # -- rv20: std(ret, 20, ddof=1) × √252 -----------------------
        j = _j("rv20")
        if j is not None and math.isfinite(rv20_ann):
            out[j] = rv20_ann

        # -- rv60: std(ret, 60, ddof=1) × √252 -----------------------
        j = _j("rv60")
        if j is not None and math.isfinite(rv60_ann):
            out[j] = rv60_ann

        # -- rv20_z: z-score of rv20 vs trailing history ---------------
        j = _j("rv20_z")
        if j is not None and nh >= _MIN_RV20_HIST:
            mu = float(np.mean(ha))
            sig = float(np.std(ha, ddof=1))
            if sig > 0 and math.isfinite(sig):
                out[j] = (float(ha[-1]) - mu) / sig

        # -- adv20: mean(close × volume, 20) --------------------------
        j = _j("adv20")
        if j is not None and nd >= 20:
            out[j] = float(np.mean(da[-20:]))

        # -- dev20: (close − sma20) / (sma20 × rv20_daily) ------------
        j = _j("dev20")
        if (
            j is not None
            and math.isfinite(sma20)
            and math.isfinite(rv20_daily)
        ):
            denom = sma20 * rv20_daily
            out[j] = safe_div(cur - sma20, denom)

        # -- max_dd_90: rolling 90d max drawdown -----------------------
        j = _j("max_dd_90")
        if j is not None and nc >= 90:
            out[j] = _max_drawdown(ca[-90:])

        # -- normalized_slope: OLS slope(log close, 50) / daily_vol ----
        j = _j("normalized_slope")
        if j is not None and nc >= 50:
            out[j] = _normalized_slope(ca[-50:])

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all per-symbol state.  Call before a new backtest."""
        self._state = {
            sym: self._make_state() for sym in self._universe
        }
        self._last_asof = None
        self._asof_count = 0

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_ready = sum(
            1 for st in self._state.values()
            if st.bars_seen >= self._max_lb
        )
        return (
            f"FeatureEngine(universe={self._n_sym}, "
            f"features={self._n_feat}, "
            f"ready={n_ready}/{self._n_sym}, "
            f"max_lb={self._max_lb})"
        )


# ---------------------------------------------------------------------------
# Module-level pure helpers (used by _compute_symbol_features)
# ---------------------------------------------------------------------------


def _max_drawdown(close: np.ndarray) -> float:
    """Peak-to-trough / peak drawdown over *close*.

    Returns a non-positive float (e.g. ``-0.12`` for −12 %).
    """
    if len(close) < 2:
        return np.nan
    running_max = np.maximum.accumulate(close)
    safe_max = np.where(running_max > 0, running_max, 1.0)
    dd = (close - running_max) / safe_max
    return float(np.nanmin(dd))


def _normalized_slope(close: np.ndarray) -> float:
    """OLS slope of ``log(close)`` divided by daily volatility.

    Parameters
    ----------
    close : np.ndarray
        Window of close prices (length ≥ 2).

    Returns
    -------
    float
        Normalised slope, or ``NaN`` when daily vol ≈ 0.
    """
    n = len(close)
    if n < 2:
        return np.nan
    log_c = np.log(np.maximum(close, 1e-10))
    x = np.arange(n, dtype=np.float64)
    x_mean = x.mean()
    y_mean = log_c.mean()
    x_diff = x - x_mean
    denom = float(np.dot(x_diff, x_diff))
    if denom <= 0:
        return np.nan
    slope = float(np.dot(x_diff, log_c - y_mean)) / denom
    log_rets = np.diff(log_c)
    daily_vol = (
        float(np.std(log_rets, ddof=1)) if len(log_rets) > 1 else np.nan
    )
    return safe_div(slope, daily_vol)
