"""
Data layer package.

Public API
----------
- :class:`DataEngine` — manages OHLCV data lifecycle (inventory, sync, load).
- :func:`load_csv` / :func:`load_parquet` — low-level file loaders.
- :func:`save_csv` / :func:`save_parquet` — low-level file writers.
"""

from .data_engine import DataEngine, SymbolStatus, SyncReport
from .storage import load_csv, load_parquet, save_csv, save_parquet

__all__ = [
    "DataEngine",
    "SymbolStatus",
    "SyncReport",
    "load_csv",
    "load_parquet",
    "save_csv",
    "save_parquet",
]
