"""
Feature pipeline — connects the data layer to the feature engine.

Loads OHLCV CSV / Parquet files for a universe of symbols, aligns
dates, feeds bars chronologically into :class:`FeatureEngine`, and
yields :class:`FeatureBundle` snapshots.

Typical usage
-------------
::

    from engine.features.pipeline import FeaturePipeline

    pipe = FeaturePipeline(["AAPL", "MSFT", "GOOG"])
    for asof, bundle in pipe.run():
        print(asof.date(), bundle.X.shape, bundle.valid_ratio("mom63"))

Data layout
-----------
Raw files must live under ``engine/data/raw/stocks/`` with names
``{SYMBOL}_{timeframe}.csv`` or ``.parquet``.  CSV columns expected:

    timestamp, open, high, low, close, volume

The ``timestamp`` column is parsed to ``pd.Timestamp`` (UTC-naive
business dates for daily data).  Both CSV and Parquet are supported;
Parquet is preferred when both exist.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from .definitions import FeatureSpec, default_feature_specs
from .feature_engine import FeatureEngine
from .types import FeatureBundle

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_DATA_DIR: Path = Path(__file__).resolve().parents[1] / "data"
_RAW_STOCKS: Path = _DATA_DIR / "raw" / "stocks"


# ---------------------------------------------------------------------------
# OHLCV loaders
# ---------------------------------------------------------------------------


def load_ohlcv(
    symbol: str,
    timeframe: str = "1d",
    *,
    data_dir: Path | None = None,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Load a single symbol's OHLCV data from disk.

    Tries Parquet first, then CSV.  Returns a DataFrame indexed by
    ``pd.DatetimeIndex`` with columns ``[open, high, low, close, volume]``.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g. ``"AAPL"``).
    timeframe : str
        Bar resolution string (e.g. ``"1d"``).
    data_dir : Path, optional
        Override the default ``engine/data/raw/stocks/`` directory.
    start, end : pd.Timestamp | str, optional
        Date filters (inclusive).

    Returns
    -------
    pd.DataFrame
        DatetimeIndex, columns ``[open, high, low, close, volume]``.

    Raises
    ------
    FileNotFoundError
        If neither CSV nor Parquet exists for the symbol.
    """
    base = data_dir or _RAW_STOCKS
    stem = f"{symbol}_{timeframe}"

    parquet_path = base / f"{stem}.parquet"
    csv_path = base / f"{stem}.csv"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"No OHLCV file for {symbol} ({timeframe}) in {base}"
        )

    # ── Normalise column names ──────────────────────────────────────
    # Only rename Polygon short-hand columns that don't clash with
    # existing standard names (avoids duplicate 'open', 'close', etc.)
    col_map = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    safe_renames = {
        k: v for k, v in col_map.items()
        if k in df.columns and v not in df.columns
    }
    if safe_renames:
        df.rename(columns=safe_renames, inplace=True)
    # Drop Polygon raw columns that weren't renamed (still short names)
    extra_cols = {"o", "h", "l", "c", "v", "vw", "t", "n"} & set(df.columns)
    if extra_cols:
        df.drop(columns=list(extra_cols), inplace=True)

    # ── Build datetime index ────────────────────────────────────────
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    elif "t" in df.columns:
        # Polygon format: Unix milliseconds
        df["t"] = pd.to_datetime(df["t"], unit="ms")
        df.set_index("t", inplace=True)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Normalize to date-only (midnight) so that daily bars from
    # different sources (e.g. 00:00 UTC vs 05:00 UTC) align.
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()

    df.index.name = "timestamp"
    df.sort_index(inplace=True)
    # Drop duplicate dates (keep last) in case of overlap
    df = df[~df.index.duplicated(keep="last")]

    # ── Date filter ─────────────────────────────────────────────────
    if start is not None:
        df = df.loc[pd.Timestamp(start):]
    if end is not None:
        df = df.loc[:pd.Timestamp(end)]

    # ── Ensure required columns exist ───────────────────────────────
    for col in ("close", "volume"):
        if col not in df.columns:
            raise ValueError(
                f"{symbol} OHLCV file missing required column '{col}'"
            )

    return df


def load_universe_ohlcv(
    symbols: List[str],
    timeframe: str = "1d",
    *,
    data_dir: Path | None = None,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
    strict: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Load OHLCV data for every symbol in the universe.

    Parameters
    ----------
    symbols : list[str]
        Ticker symbols.
    timeframe : str
        Bar resolution.
    data_dir : Path, optional
        Override the default raw-data directory.
    start, end : pd.Timestamp | str, optional
        Date filters (inclusive).
    strict : bool
        If ``True``, raise on any missing file.  If ``False`` (default),
        log a warning and skip the symbol.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from symbol to its OHLCV DataFrame.
    """
    ohlcv: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            ohlcv[sym] = load_ohlcv(
                sym, timeframe, data_dir=data_dir, start=start, end=end,
            )
        except FileNotFoundError:
            if strict:
                raise
            log.warning("No OHLCV data for %s — skipping", sym)
    return ohlcv


# ---------------------------------------------------------------------------
# Date alignment
# ---------------------------------------------------------------------------


def align_dates(
    ohlcv: Dict[str, pd.DataFrame],
) -> pd.DatetimeIndex:
    """Return sorted union of all trading dates across symbols.

    Parameters
    ----------
    ohlcv : dict[str, pd.DataFrame]
        Symbol → OHLCV DataFrame (DatetimeIndex).

    Returns
    -------
    pd.DatetimeIndex
        Sorted, deduplicated dates.
    """
    if not ohlcv:
        return pd.DatetimeIndex([])
    all_dates = pd.DatetimeIndex([]).append(
        [df.index for df in ohlcv.values()]
    )
    return all_dates.drop_duplicates().sort_values()


# ---------------------------------------------------------------------------
# Bar iterator
# ---------------------------------------------------------------------------


def iter_bars(
    ohlcv: Dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex | None = None,
) -> Iterator[dict]:
    """Yield bar dicts in chronological order (all symbols per date).

    For each date, a bar is yielded for every symbol that has data on
    that date.  Bar dict keys: ``symbol``, ``asof``, ``open``, ``high``,
    ``low``, ``close``, ``volume``.

    Parameters
    ----------
    ohlcv : dict[str, pd.DataFrame]
        Symbol → OHLCV DataFrame (DatetimeIndex).
    dates : pd.DatetimeIndex, optional
        Restrict to these dates.  Defaults to ``align_dates(ohlcv)``.

    Yields
    ------
    dict
        One bar per symbol per date.
    """
    if dates is None:
        dates = align_dates(ohlcv)

    for dt in dates:
        for sym, df in ohlcv.items():
            if dt not in df.index:
                continue
            row = df.loc[dt]
            # If duplicate timestamps exist, take the last row
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            yield {
                "symbol": sym,
                "asof": dt,
                "open": float(row.get("open", np.nan)),
                "high": float(row.get("high", np.nan)),
                "low": float(row.get("low", np.nan)),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0.0)),
            }


# ---------------------------------------------------------------------------
# FeaturePipeline
# ---------------------------------------------------------------------------


class FeaturePipeline:
    """End-to-end: OHLCV files → FeatureEngine → FeatureBundle stream.

    Parameters
    ----------
    universe : list[str]
        Ticker symbols to load and process.
    feature_specs : list[FeatureSpec], optional
        Override default feature catalogue.
    timeframe : str
        Bar resolution (e.g. ``"1d"``).
    data_dir : Path, optional
        Override the default raw-data directory.
    readiness_threshold : float
        Fraction of universe that must be warm before yielding bundles.
        Passed to :class:`FeatureEngine`.

    Examples
    --------
    >>> pipe = FeaturePipeline(["AAPL", "MSFT"])
    >>> for asof, bundle in pipe.run():
    ...     print(asof.date(), bundle.get("AAPL", "mom63"))
    """

    def __init__(
        self,
        universe: List[str],
        feature_specs: List[FeatureSpec] | None = None,
        timeframe: str = "1d",
        data_dir: Path | None = None,
        readiness_threshold: float = 0.80,
    ) -> None:
        self._universe = list(universe)
        self._specs = feature_specs
        self._timeframe = timeframe
        self._data_dir = data_dir
        self._readiness_threshold = readiness_threshold

        self._engine: FeatureEngine | None = None
        self._ohlcv: Dict[str, pd.DataFrame] | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def engine(self) -> FeatureEngine | None:
        """The underlying FeatureEngine (available after :meth:`run`)."""
        return self._engine

    @property
    def ohlcv(self) -> Dict[str, pd.DataFrame] | None:
        """Loaded OHLCV DataFrames (available after :meth:`run`)."""
        return self._ohlcv

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        *,
        emit_frequency: int = 1,
        warmup_only: bool = False,
    ) -> Iterator[Tuple[pd.Timestamp, FeatureBundle]]:
        """Load data, feed the engine, and yield feature bundles.

        Parameters
        ----------
        start, end : pd.Timestamp | str, optional
            Date range for the OHLCV data (inclusive).
        emit_frequency : int
            Yield a bundle every *N* trading days.  ``1`` = every day
            (after the engine is ready).  ``5`` = weekly.
        warmup_only : bool
            If ``True``, feed all bars but never yield bundles.  Useful
            for warming up the engine before a live session.

        Yields
        ------
        tuple[pd.Timestamp, FeatureBundle]
            ``(asof, bundle)`` pairs, one per emission date.
        """
        # ── 1. Load data ────────────────────────────────────────────
        self._ohlcv = load_universe_ohlcv(
            self._universe,
            self._timeframe,
            data_dir=self._data_dir,
            start=start,
            end=end,
        )
        if not self._ohlcv:
            log.warning("No OHLCV data loaded — nothing to process")
            return

        loaded_symbols = list(self._ohlcv.keys())
        log.info(
            "Loaded %d/%d symbols: %s",
            len(loaded_symbols),
            len(self._universe),
            loaded_symbols[:10],
        )

        # ── 2. Build engine (use loaded symbols as effective universe) ─
        self._engine = FeatureEngine(
            universe=loaded_symbols,
            feature_specs=self._specs,
            readiness_threshold=self._readiness_threshold,
        )

        # ── 3. Align dates & iterate ────────────────────────────────
        dates = align_dates(self._ohlcv)
        log.info(
            "Date range: %s → %s (%d trading days)",
            dates[0].date() if len(dates) > 0 else "?",
            dates[-1].date() if len(dates) > 0 else "?",
            len(dates),
        )

        last_emit_dt: pd.Timestamp | None = None
        bars_by_date: Dict[pd.Timestamp, int] = {}
        emit_counter = 0

        for bar in iter_bars(self._ohlcv, dates):
            self._engine.update(bar)

            dt = bar["asof"]
            bars_by_date[dt] = bars_by_date.get(dt, 0) + 1

            # ── Emit logic: once all symbols for this date are fed ─
            # Detect date boundary: we've moved past a date
            if last_emit_dt is not None and dt != last_emit_dt:
                # Previous date is complete — evaluate for emission
                if not warmup_only and self._engine.is_ready():
                    emit_counter += 1
                    if emit_counter % emit_frequency == 0:
                        bundle = self._engine.get_feature_bundle(last_emit_dt)
                        yield last_emit_dt, bundle

            last_emit_dt = dt

        # ── Final date (last date in the stream) ────────────────────
        if (
            last_emit_dt is not None
            and not warmup_only
            and self._engine.is_ready()
        ):
            emit_counter += 1
            if emit_counter % emit_frequency == 0:
                bundle = self._engine.get_feature_bundle(last_emit_dt)
                yield last_emit_dt, bundle

    # ------------------------------------------------------------------
    # Convenience: build a single bundle for the last date
    # ------------------------------------------------------------------

    def build_latest(
        self,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
    ) -> Tuple[pd.Timestamp, FeatureBundle] | None:
        """Load data, warm up, and return the final-date bundle.

        A convenience wrapper that runs the full pipeline and returns
        only the last emitted ``(asof, FeatureBundle)`` pair.

        Returns ``None`` if the engine never becomes ready.
        """
        result: Tuple[pd.Timestamp, FeatureBundle] | None = None
        for asof, bundle in self.run(start=start, end=end):
            result = (asof, bundle)
        return result

    # ------------------------------------------------------------------
    # Convenience: warm up the engine without emitting
    # ------------------------------------------------------------------

    def warmup(
        self,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
    ) -> FeatureEngine:
        """Load data and feed all bars without yielding bundles.

        Returns the warmed-up :class:`FeatureEngine` for use in a
        live / forward session.
        """
        # Exhaust the generator
        for _ in self.run(start=start, end=end, warmup_only=True):
            pass  # pragma: no cover
        if self._engine is None:
            raise RuntimeError("No data loaded — engine not created")
        return self._engine

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"FeaturePipeline(universe={len(self._universe)}, "
            f"timeframe={self._timeframe!r}, "
            f"engine={'ready' if self._engine else 'not built'})"
        )
