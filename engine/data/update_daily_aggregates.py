"""
Daily stock aggregates downloader and updater.

Downloads daily aggregates for tickers listed in good_quality_stock_tickers_200.txt.
Automatically updates existing files by appending new data, or downloads from scratch if file doesn't exist.
Saves both parquet and CSV formats.

Usage:
    # Set environment variable
    export POLYGON_API_KEY=your_api_key
    
    # Run from command line
    python -m engine.data.update_daily_aggregates
    
    # Or from Python
    from engine.data.update_daily_aggregates import update_all_daily_aggregates
    results = update_all_daily_aggregates()
    
    # Or with explicit API key
    results = update_all_daily_aggregates(api_key="your_api_key")

The script will:
1. Read tickers from data/raw/stocks/good_quality_stock_tickers_200.txt
2. For each ticker, check if {SYMBOL}_1d.parquet or {SYMBOL}_1d.csv exists
3. If exists: load it, find last date, download from last_date+1 to today
4. If not exists: download from 1970-01-01 to today
5. Append new data, remove duplicates, save both parquet and CSV
6. Handle rate limiting with configurable delay between API calls
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from .polygon_agg_download import download_aggregates
from .storage import RAW_DIR, file_exists, load_csv, load_parquet, save_csv, save_parquet


# First trading date (approximate - NYSE started in 1792, but we use a safe early date)
FIRST_MARKET_DATE = "1970-01-01"


def get_tickers(ticker_file: Path) -> list[str]:
    """Read ticker symbols from file."""
    if not ticker_file.exists():
        raise FileNotFoundError(f"Ticker file not found: {ticker_file}")
    
    with open(ticker_file, "r") as f:
        tickers = [line.strip().upper() for line in f if line.strip()]
    return tickers


def get_last_date(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    """Get the last timestamp from DataFrame."""
    if df.empty or "timestamp" not in df.columns:
        return None
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    
    return df["timestamp"].max()


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize DataFrame format:
    - Rename Polygon columns to standard OHLCV format
    - Ensure timestamp column exists
    - Sort by timestamp
    - Keep both original and standardized column names for compatibility
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Ensure timestamp exists
    if "timestamp" not in df.columns:
        if "t" in df.columns:
            # Convert Unix milliseconds to datetime
            df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        else:
            raise ValueError("DataFrame must have 'timestamp' or 't' column")
    else:
        # Ensure timestamp is datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    
    # Map Polygon API columns to standard names (keep both for compatibility)
    # Polygon: o, h, l, c, v, vw, n, t
    # Standard: open, high, low, close, volume
    column_mapping = {
        "o": "open",
        "h": "high", 
        "l": "low",
        "c": "close",
        "v": "volume",
    }
    
    # Add standard columns if they don't exist
    for polygon_col, standard_col in column_mapping.items():
        if polygon_col in df.columns:
            if standard_col not in df.columns:
                df[standard_col] = df[polygon_col]
        elif standard_col not in df.columns:
            # If neither exists, we can't proceed
            raise ValueError(f"Missing required column: {polygon_col} or {standard_col}")
    
    # Ensure we have the essential columns
    required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after standardization: {missing}")
    
    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    return df


def download_and_update_ticker(
    symbol: str,
    api_key: str,
    data_dir: Path,
    start_date: str = FIRST_MARKET_DATE,
    delay_seconds: float = 0.1,  # Rate limiting for API
) -> tuple[bool, str]:
    """
    Download or update daily aggregates for a single ticker.
    
    Returns:
        (success: bool, message: str)
    """
    symbol = symbol.upper()
    
    # File paths
    csv_path = data_dir / f"{symbol}_1d.csv"
    parquet_path = data_dir / f"{symbol}_1d.parquet"
    
    # Check if file exists (prefer parquet, fallback to CSV)
    existing_df = None
    file_exists_flag = False
    
    if parquet_path.exists():
        try:
            existing_df = load_parquet(f"raw/stocks/{symbol}_1d.parquet")
            file_exists_flag = True
        except Exception as e:
            print(f"  Warning: Could not load parquet for {symbol}: {e}")
    
    if existing_df is None and csv_path.exists():
        try:
            existing_df = load_csv(f"raw/stocks/{symbol}_1d.csv")
            file_exists_flag = True
        except Exception as e:
            print(f"  Warning: Could not load CSV for {symbol}: {e}")
    
    # Determine date range
    if file_exists_flag and existing_df is not None and not existing_df.empty:
        existing_df = standardize_dataframe(existing_df)
        last_date = get_last_date(existing_df)
        
        if last_date is not None:
            # Start from day after last date
            start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            mode = "update"
        else:
            start = start_date
            mode = "full_download"
    else:
        start = start_date
        mode = "new_download"
        existing_df = None
    
    # End date is today
    today = datetime.now(timezone.utc).date()
    end = today.strftime("%Y-%m-%d")
    
    # Skip if start > end (already up to date)
    if start > end:
        last_date_str = last_date.strftime("%Y-%m-%d") if last_date else "N/A"
        return True, f"{symbol}: Already up to date (last date: {last_date_str})"
    
    try:
        # Download new data
        print(f"  {symbol}: Downloading {mode} from {start} to {end}...")
        
        try:
            new_df = download_aggregates(
                symbol=symbol,
                multiplier=1,
                timespan="day",
                start=start,
                end=end,
                api_key=api_key,
            )
        except Exception as e:
            return False, f"{symbol}: Download failed - {str(e)}"
        
        if new_df.empty:
            # No new data is not necessarily an error (e.g., weekend, holiday)
            return True, f"{symbol}: No new data available for {start} to {end}"
        
        # Standardize new data
        new_df = standardize_dataframe(new_df)
        
        # Combine with existing data
        if existing_df is not None and not existing_df.empty:
            # Combine and remove duplicates
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            # Remove duplicates based on timestamp (keep last)
            combined_df = combined_df.drop_duplicates(subset=["timestamp"], keep="last")
            combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)
        else:
            combined_df = new_df
        
        # Save both formats
        save_parquet(combined_df, f"raw/stocks/{symbol}_1d.parquet")
        save_csv(combined_df, f"raw/stocks/{symbol}_1d.csv")
        
        new_rows = len(new_df)
        total_rows = len(combined_df)
        
        return True, f"{symbol}: {mode} successful - added {new_rows} rows, total {total_rows} rows"
        
    except Exception as e:
        return False, f"{symbol}: Error - {str(e)}"
    
    finally:
        # Rate limiting
        time.sleep(delay_seconds)


def update_all_daily_aggregates(
    api_key: Optional[str] = None,
    ticker_file: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    delay_seconds: float = 0.1,
) -> dict[str, tuple[bool, str]]:
    """
    Update daily aggregates for all tickers in the ticker file.
    
    Args:
        api_key: Polygon API key (defaults to POLYGON_API_KEY env var)
        ticker_file: Path to ticker file (defaults to data/raw/stocks/good_quality_stock_tickers_200.txt)
        data_dir: Directory to save data (defaults to data/raw/stocks)
        delay_seconds: Delay between API calls for rate limiting
    
    Returns:
        dict mapping symbol -> (success: bool, message: str)
    """
    # Get API key
    if api_key is None:
        api_key = os.getenv("POLYGON_API_KEY")
        if not api_key:
            raise ValueError(
                "Polygon API key not provided. Set POLYGON_API_KEY environment variable "
                "or pass api_key parameter."
            )
    
    # Set default paths
    if ticker_file is None:
        ticker_file = RAW_DIR / "stocks" / "good_quality_stock_tickers_200.txt"
    
    if data_dir is None:
        data_dir = RAW_DIR / "stocks"
    
    # Ensure data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Get tickers
    tickers = get_tickers(ticker_file)
    print(f"Found {len(tickers)} tickers in {ticker_file}")
    print(f"Starting update at {datetime.now(timezone.utc).isoformat()}\n")
    
    results = {}
    success_count = 0
    fail_count = 0
    
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] Processing {ticker}...")
        success, message = download_and_update_ticker(
            symbol=ticker,
            api_key=api_key,
            data_dir=data_dir,
            delay_seconds=delay_seconds,
        )
        results[ticker] = (success, message)
        
        if success:
            success_count += 1
        else:
            fail_count += 1
        
        print(f"  {message}\n")
    
    # Summary
    print("=" * 60)
    print(f"Update complete: {success_count} succeeded, {fail_count} failed")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    import sys
    
    # Allow API key to be passed as command line argument
    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    
    results = update_all_daily_aggregates(api_key=api_key)
    
    # Exit with error code if any failed
    failed = [ticker for ticker, (success, _) in results.items() if not success]
    if failed:
        print(f"\nFailed tickers: {', '.join(failed)}")
        sys.exit(1)
