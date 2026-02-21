"""
Data engine — manages the OHLCV data lifecycle.

Reads the ticker universe from a text file, inventories which symbols
already have raw data on disk, identifies gaps, and downloads missing
or stale data via the Polygon API.

Typical usage
-------------
::

    from engine.data.data_engine import DataEngine

    de = DataEngine()
    de.status()                       # print inventory
    de.sync()                         # download everything missing
    de.sync(symbols=["MSFT", "META"]) # download specific symbols only
    ohlcv = de.load_universe()        # dict[str, pd.DataFrame]

Integration with feature pipeline
----------------------------------
::

    from engine.data.data_engine import DataEngine
    from engine.features.pipeline import FeaturePipeline

    de = DataEngine()
    de.sync()
    pipe = FeaturePipeline(de.universe, data_dir=de.data_dir)
    for asof, bundle in pipe.run():
        ...
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # reads .env from project root into os.environ

from .polygon_agg_download import download_aggregates
from .storage import RAW_DIR, load_csv, load_parquet, save_csv, save_parquet
from .update_daily_aggregates import (
    FIRST_MARKET_DATE,
    get_last_date,
    get_tickers,
    standardize_dataframe,
)

log = logging.getLogger(__name__)

_STOCKS_DIR: Path = RAW_DIR / "stocks"
_TICKERS_FILE: Path = _STOCKS_DIR / "good_quality_stock_tickers_200.txt"


# ---------------------------------------------------------------------------
# Inventory dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolStatus:
    """Per-symbol data status.

    Attributes
    ----------
    symbol : str
        Ticker symbol.
    has_csv : bool
        True if ``{SYMBOL}_1d.csv`` exists.
    has_parquet : bool
        True if ``{SYMBOL}_1d.parquet`` exists.
    n_rows : int
        Number of rows in the best available file (0 if missing).
    first_date : pd.Timestamp | None
        Earliest timestamp in the data.
    last_date : pd.Timestamp | None
        Latest timestamp in the data.
    days_stale : int
        Business days between *last_date* and today.  0 if up-to-date
        or if no data exists.
    """

    symbol: str
    has_csv: bool = False
    has_parquet: bool = False
    n_rows: int = 0
    first_date: Optional[pd.Timestamp] = None
    last_date: Optional[pd.Timestamp] = None
    days_stale: int = 0

    @property
    def exists(self) -> bool:
        """True if any data file exists."""
        return self.has_csv or self.has_parquet

    @property
    def is_stale(self) -> bool:
        """True if data is more than 1 business day behind today."""
        return self.days_stale > 1


@dataclass
class SyncReport:
    """Summary returned by :meth:`DataEngine.sync`.

    Attributes
    ----------
    downloaded : list[str]
        Symbols that were successfully downloaded / updated.
    skipped : list[str]
        Symbols already up-to-date (skipped).
    failed : dict[str, str]
        Symbols that failed → error message.
    """

    downloaded: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    failed: Dict[str, str] = field(default_factory=dict)

    @property
    def n_total(self) -> int:
        return len(self.downloaded) + len(self.skipped) + len(self.failed)

    def __repr__(self) -> str:
        return (
            f"SyncReport(downloaded={len(self.downloaded)}, "
            f"skipped={len(self.skipped)}, "
            f"failed={len(self.failed)})"
        )


# ---------------------------------------------------------------------------
# DataEngine
# ---------------------------------------------------------------------------


class DataEngine:
    """Manages OHLCV data for a universe of stock tickers.

    Parameters
    ----------
    ticker_file : Path, optional
        Path to the newline-separated ticker file.
        Defaults to ``engine/data/raw/stocks/good_quality_stock_tickers_200.txt``.
    data_dir : Path, optional
        Directory where ``{SYMBOL}_1d.csv`` / ``.parquet`` files live.
        Defaults to ``engine/data/raw/stocks/``.
    api_key : str, optional
        Polygon API key.  Falls back to ``POLYGON_API_KEY`` env var.
    timeframe : str
        Bar resolution string (default ``"1d"``).
    """

    def __init__(
        self,
        ticker_file: Path | None = None,
        data_dir: Path | None = None,
        api_key: str | None = None,
        timeframe: str = "1d",
    ) -> None:
        self._ticker_file = ticker_file or _TICKERS_FILE
        self._data_dir = data_dir or _STOCKS_DIR
        self._api_key = api_key or os.getenv("POLYGON_API_KEY", "")
        self._timeframe = timeframe

        # Load ticker universe
        self._universe: List[str] = get_tickers(self._ticker_file)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def universe(self) -> List[str]:
        """Ordered list of ticker symbols."""
        return list(self._universe)

    @property
    def data_dir(self) -> Path:
        """Directory containing raw OHLCV files."""
        return self._data_dir

    @property
    def ticker_file(self) -> Path:
        return self._ticker_file

    # ------------------------------------------------------------------
    # Inventory
    # ------------------------------------------------------------------

    def inventory(self) -> Dict[str, SymbolStatus]:
        """Scan disk and return per-symbol data status.

        Returns
        -------
        dict[str, SymbolStatus]
            Mapping from symbol to its :class:`SymbolStatus`.
        """
        today = pd.Timestamp(datetime.now(timezone.utc).date(), tz="UTC")
        result: Dict[str, SymbolStatus] = {}

        for sym in self._universe:
            csv_path = self._data_dir / f"{sym}_{self._timeframe}.csv"
            parquet_path = self._data_dir / f"{sym}_{self._timeframe}.parquet"

            has_csv = csv_path.exists()
            has_parquet = parquet_path.exists()

            if not has_csv and not has_parquet:
                result[sym] = SymbolStatus(symbol=sym)
                continue

            # Load the best available file
            df = self._load_symbol(sym)
            if df is None or df.empty:
                result[sym] = SymbolStatus(
                    symbol=sym, has_csv=has_csv, has_parquet=has_parquet,
                )
                continue

            # Parse dates
            first_dt = self._parse_first_date(df)
            last_dt = self._parse_last_date(df)

            # Staleness: business days between last_date and today
            days_stale = 0
            if last_dt is not None:
                # Ensure both ends are tz-aware (UTC) for pd.bdate_range
                last_utc = (
                    last_dt.tz_localize("UTC")
                    if last_dt.tzinfo is None
                    else last_dt.tz_convert("UTC")
                )
                bdays = pd.bdate_range(
                    start=last_utc + pd.Timedelta(days=1), end=today,
                )
                days_stale = len(bdays)

            result[sym] = SymbolStatus(
                symbol=sym,
                has_csv=has_csv,
                has_parquet=has_parquet,
                n_rows=len(df),
                first_date=first_dt,
                last_date=last_dt,
                days_stale=days_stale,
            )

        return result

    def missing(self) -> List[str]:
        """Return symbols that have no data on disk."""
        inv = self.inventory()
        return [sym for sym, st in inv.items() if not st.exists]

    def stale(self, max_days: int = 1) -> List[str]:
        """Return symbols whose data is more than *max_days* business days old."""
        inv = self.inventory()
        return [sym for sym, st in inv.items() if st.days_stale > max_days]

    def present(self) -> List[str]:
        """Return symbols that have data on disk."""
        inv = self.inventory()
        return [sym for sym, st in inv.items() if st.exists]

    # ------------------------------------------------------------------
    # Status report
    # ------------------------------------------------------------------

    def status(self) -> str:
        """Print and return a human-readable inventory report."""
        inv = self.inventory()

        lines: List[str] = []
        lines.append(f"DataEngine — {len(self._universe)} symbols from {self._ticker_file.name}")
        lines.append(f"Data dir:  {self._data_dir}")
        lines.append(f"API key:   {'set' if self._api_key else 'NOT SET'}")
        lines.append("")

        present = [s for s, st in inv.items() if st.exists]
        missing = [s for s, st in inv.items() if not st.exists]
        stale = [s for s, st in inv.items() if st.is_stale]

        lines.append(f"  Present: {len(present)}/{len(inv)}")
        lines.append(f"  Missing: {len(missing)}/{len(inv)}")
        lines.append(f"  Stale:   {len(stale)}/{len(inv)}")
        lines.append("")

        # Detail table
        lines.append(f"  {'Symbol':<8} {'Status':<10} {'Rows':>7} {'First':>12} {'Last':>12} {'Stale':>6}")
        lines.append(f"  {'-'*8} {'-'*10} {'-'*7} {'-'*12} {'-'*12} {'-'*6}")

        for sym in self._universe:
            st = inv[sym]
            if st.exists:
                status_str = "ok" if not st.is_stale else "STALE"
                first = st.first_date.strftime("%Y-%m-%d") if st.first_date else "?"
                last = st.last_date.strftime("%Y-%m-%d") if st.last_date else "?"
                stale_str = str(st.days_stale) + "d" if st.days_stale > 0 else "-"
            else:
                status_str = "MISSING"
                first = "-"
                last = "-"
                stale_str = "-"

            lines.append(
                f"  {sym:<8} {status_str:<10} {st.n_rows:>7} {first:>12} {last:>12} {stale_str:>6}"
            )

        report = "\n".join(lines)
        print(report)
        return report

    # ------------------------------------------------------------------
    # Sync (download missing / update stale)
    # ------------------------------------------------------------------

    def sync(
        self,
        symbols: List[str] | None = None,
        *,
        force: bool = False,
        start_date: str = FIRST_MARKET_DATE,
        delay_seconds: float = 13.0,
    ) -> SyncReport:
        """Download missing and update stale data.

        Parameters
        ----------
        symbols : list[str], optional
            Subset of universe to sync.  Defaults to the full universe.
        force : bool
            If ``True``, re-download even if data exists and is fresh.
        start_date : str
            Earliest date to download for new symbols.
        delay_seconds : float
            Pause between API calls (rate limiting).

        Returns
        -------
        SyncReport
            Summary of what was downloaded, skipped, or failed.

        Raises
        ------
        ValueError
            If no Polygon API key is configured.
        """
        if not self._api_key:
            raise ValueError(
                "No Polygon API key configured.  Set POLYGON_API_KEY env var "
                "or pass api_key= to DataEngine()."
            )

        targets = symbols or self._universe
        inv = self.inventory()
        report = SyncReport()

        for i, sym in enumerate(targets, 1):
            st = inv.get(sym)

            # Decide whether to skip
            if not force and st is not None and st.exists and not st.is_stale:
                report.skipped.append(sym)
                log.debug("[%d/%d] %s — up to date, skipping", i, len(targets), sym)
                continue

            # Determine download range
            if st is not None and st.exists and st.last_date is not None:
                dl_start = (st.last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                mode = "update"
            else:
                dl_start = start_date
                mode = "new"

            dl_end = datetime.now(timezone.utc).date().strftime("%Y-%m-%d")

            if dl_start > dl_end:
                report.skipped.append(sym)
                continue

            log.info(
                "[%d/%d] %s — %s from %s to %s",
                i, len(targets), sym, mode, dl_start, dl_end,
            )

            try:
                new_df = download_aggregates(
                    symbol=sym,
                    multiplier=1,
                    timespan="day",
                    start=dl_start,
                    end=dl_end,
                    api_key=self._api_key,
                )
            except Exception as exc:
                report.failed[sym] = f"download error: {exc}"
                log.warning("%s download failed: %s", sym, exc)
                time.sleep(delay_seconds)
                continue

            if new_df.empty:
                report.skipped.append(sym)
                log.info("%s — no new data for %s to %s", sym, dl_start, dl_end)
                time.sleep(delay_seconds)
                continue

            # Standardise
            try:
                new_df = standardize_dataframe(new_df)
            except Exception as exc:
                report.failed[sym] = f"standardize error: {exc}"
                log.warning("%s standardize failed: %s", sym, exc)
                time.sleep(delay_seconds)
                continue

            # Merge with existing
            existing_df = self._load_symbol(sym)
            if existing_df is not None and not existing_df.empty:
                existing_df = standardize_dataframe(existing_df)
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                combined = combined.drop_duplicates(
                    subset=["timestamp"], keep="last",
                )
                combined = combined.sort_values("timestamp").reset_index(drop=True)
            else:
                combined = new_df

            # Save both formats
            rel_prefix = f"raw/stocks/{sym}_{self._timeframe}"
            save_csv(combined, f"{rel_prefix}.csv")
            try:
                save_parquet(combined, f"{rel_prefix}.parquet")
            except Exception:
                pass  # parquet optional if pyarrow not installed

            report.downloaded.append(sym)
            log.info(
                "%s — %s done: +%d rows → %d total",
                sym, mode, len(new_df), len(combined),
            )

            time.sleep(delay_seconds)

        # Summary log
        log.info(
            "Sync complete: %d downloaded, %d skipped, %d failed",
            len(report.downloaded),
            len(report.skipped),
            len(report.failed),
        )
        return report

    # ------------------------------------------------------------------
    # Load helpers
    # ------------------------------------------------------------------

    def load_universe(
        self,
        symbols: List[str] | None = None,
    ) -> Dict[str, pd.DataFrame]:
        """Load OHLCV DataFrames for all (or specified) symbols.

        Only returns symbols that actually have data on disk.

        Returns
        -------
        dict[str, pd.DataFrame]
            Symbol → OHLCV DataFrame (DatetimeIndex).
        """
        targets = symbols or self._universe
        result: Dict[str, pd.DataFrame] = {}

        for sym in targets:
            df = self._load_symbol(sym)
            if df is not None and not df.empty:
                # Ensure DatetimeIndex
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df.set_index("timestamp", inplace=True)
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                # Normalize to date-only (midnight, tz-naive) so that
                # daily bars from different sources align correctly.
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                df.index = df.index.normalize()
                df.sort_index(inplace=True)
                df = df[~df.index.duplicated(keep="last")]
                result[sym] = df

        return result

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_symbol(self, sym: str) -> Optional[pd.DataFrame]:
        """Try loading parquet, then CSV for a symbol."""
        parquet_rel = f"raw/stocks/{sym}_{self._timeframe}.parquet"
        csv_rel = f"raw/stocks/{sym}_{self._timeframe}.csv"

        parquet_path = self._data_dir / f"{sym}_{self._timeframe}.parquet"
        csv_path = self._data_dir / f"{sym}_{self._timeframe}.csv"

        if parquet_path.exists():
            try:
                return load_parquet(parquet_rel)
            except Exception:
                pass

        if csv_path.exists():
            try:
                return load_csv(csv_rel)
            except Exception:
                pass

        return None

    @staticmethod
    def _parse_first_date(df: pd.DataFrame) -> Optional[pd.Timestamp]:
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"])
            return ts.min()
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
            return df.index.min()
        return None

    @staticmethod
    def _parse_last_date(df: pd.DataFrame) -> Optional[pd.Timestamp]:
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"])
            return ts.max()
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
            return df.index.max()
        return None

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_present = len(self.present())
        return (
            f"DataEngine(universe={len(self._universe)}, "
            f"present={n_present}, "
            f"data_dir={self._data_dir.name!r})"
        )
