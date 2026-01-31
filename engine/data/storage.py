from pathlib import Path
import pandas as pd


# Project root = folder that contains main.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _ensure_dir(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


# ---------- SAVE FUNCTIONS ----------

def save_parquet(df: pd.DataFrame, relative_path: str) -> Path:
    """
    Save DataFrame as Parquet under data/raw or data/processed.

    Example:
        save_parquet(df, "raw/stocks/AAPL_1m_2024-01.parquet")
    """
    full_path = PROJECT_ROOT / "data" / relative_path
    _ensure_dir(full_path.parent)

    df.to_parquet(full_path, index=False, engine="pyarrow")
    return full_path


def save_csv(df: pd.DataFrame, relative_path: str) -> Path:
    """
    Save DataFrame as CSV.

    Example:
        save_csv(df, "processed/stocks/AAPL_1m_clean.csv")
    """
    full_path = PROJECT_ROOT / "data" / relative_path
    _ensure_dir(full_path.parent)

    df.to_csv(full_path, index=False)
    return full_path


# ---------- LOAD FUNCTIONS ----------

def load_parquet(relative_path: str) -> pd.DataFrame:
    """
    Load Parquet file from data directory.
    """
    full_path = PROJECT_ROOT / "data" / relative_path
    return pd.read_parquet(full_path, engine="pyarrow")


def load_csv(relative_path: str) -> pd.DataFrame:
    """
    Load CSV file from data directory.
    """
    full_path = PROJECT_ROOT / "data" / relative_path
    return pd.read_csv(full_path)


# ---------- UTILS ----------

def file_exists(relative_path: str) -> bool:
    return (PROJECT_ROOT / "data" / relative_path).exists()