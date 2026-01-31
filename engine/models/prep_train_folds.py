from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Iterator


def select_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    *,
    keep_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    (1) Tailor the input by selecting feature columns.

    - feature_cols: columns you want as model inputs (X)
    - keep_cols: extra columns to keep for later (e.g., ['date','symbol','close'])
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    keep_cols = keep_cols or []
    cols = list(dict.fromkeys(keep_cols + feature_cols))  # preserve order, remove dups
    return df.loc[:, cols].copy()


def add_next_day_direction_label(
    df: pd.DataFrame,
    *,
    close_col: str = "close",
    group_col: Optional[str] = "symbol",
    label_col: str = "y",
    fwd_ret_col: str = "ret_fwd_1",
) -> pd.DataFrame:
    """
    (2) Generate label: up/down next day from close price.

    y = 1 if next day's close return > 0 else 0

    Assumes df contains at least: [group_col, close_col] (group_col can be None for single asset)
    """
    out = df.copy()

    if close_col not in out.columns:
        raise ValueError(f"'{close_col}' not found in df")

    if group_col and group_col in out.columns:
        out[fwd_ret_col] = out.groupby(group_col)[close_col].pct_change(-1)
    else:
        out[fwd_ret_col] = out[close_col].pct_change(-1)

    out[label_col] = (out[fwd_ret_col] > 0).astype(int)
    return out

def walk_forward_splits(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    train_days: int = 252 * 2,   # ~2 years of trading days
    test_days: int = 21,         # ~1 month of trading days
    step_days: Optional[int] = None,  # move window forward by this many days (defaults to test_days)
    gap_days: int = 0,           # optional gap between train and test to reduce leakage
    anchored: bool = True,       # True = expanding train, False = rolling train
) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate walk-forward train/test splits using day-counts on unique sorted dates.

    Parameters
    ----------
    train_days : number of unique dates in the training window
    test_days  : number of unique dates in the test window
    step_days  : how many unique dates to advance each iteration (default = test_days)
    gap_days   : number of unique dates to skip between train end and test start
    anchored   : expanding (anchored) vs rolling training window

    Yields
    ------
    (train_df, test_df) for each walk-forward fold
    """
    if date_col not in df.columns:
        raise ValueError(f"'{date_col}' not found in df")

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.sort_values(date_col)

    # Use unique dates so multi-symbol data works correctly
    dates = pd.Index(d[date_col].dropna().unique()).sort_values()
    if len(dates) < (train_days + gap_days + test_days):
        raise ValueError(
            f"Not enough unique dates ({len(dates)}) for train_days={train_days}, "
            f"gap_days={gap_days}, test_days={test_days}"
        )

    step_days = step_days or test_days

    # Initial indices (on the unique date array)
    train_start_idx = 0
    train_end_idx = train_start_idx + train_days          # exclusive
    test_start_idx = train_end_idx + gap_days
    test_end_idx = test_start_idx + test_days             # exclusive

    while test_end_idx <= len(dates):
        train_start_date = dates[train_start_idx]
        train_end_date_excl = dates[train_end_idx]        # exclusive bound
        test_start_date = dates[test_start_idx]
        test_end_date_excl = dates[test_end_idx]          # exclusive bound

        if anchored:
            # expanding train: always start from the beginning
            cur_train_start_date = dates[0]
        else:
            # rolling train: move start forward with end
            cur_train_start_date = train_start_date

        train_mask = (d[date_col] >= cur_train_start_date) & (d[date_col] < train_end_date_excl)
        test_mask = (d[date_col] >= test_start_date) & (d[date_col] < test_end_date_excl)

        train_df = d.loc[train_mask].copy()
        test_df = d.loc[test_mask].copy()

        yield train_df, test_df

        # advance
        train_start_idx += step_days
        train_end_idx += step_days
        test_start_idx = train_end_idx + gap_days
        test_end_idx = test_start_idx + test_days


def generate_training_folds(
    df: pd.DataFrame,
    *,
    feature_cols: List[str],
    date_col: str = "date",
    close_col: str = "close",
    symbol_col: str = "symbol",
    label_col: str = "y",
    keep_cols: Optional[List[str]] = None,
    # walk-forward params (unique-date based)
    train_days: int = 252 * 2,
    test_days: int = 21,
    step_days: Optional[int] = None,
    gap_days: int = 0,
    anchored: bool = True,
) -> Iterator[Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
    """
    Combine:
      1) select_features
      2) add_next_day_direction_label (from close)
      3) walk_forward_splits

    Yields per fold:
      (X_train, y_train, X_test, y_test)

    Requirements:
      df includes: date_col, close_col, symbol_col (optional if single asset), plus feature_cols
    """

    # --- 1) Select needed columns (keep minimal extras for label & sorting) ---
    base_keep = [date_col, close_col]
    if symbol_col in df.columns:
        base_keep.append(symbol_col)
    if keep_cols:
        base_keep.extend(keep_cols)

    d = select_features(df, feature_cols, keep_cols=base_keep)

    # Ensure datetime + sort for correct label and folds
    d[date_col] = pd.to_datetime(d[date_col])
    sort_cols = [date_col] if symbol_col not in d.columns else [symbol_col, date_col]
    d = d.sort_values(sort_cols)

    # --- 2) Label: next-day direction based on close ---
    group_col = symbol_col if symbol_col in d.columns else None
    d = add_next_day_direction_label(
        d,
        close_col=close_col,
        group_col=group_col,
        label_col=label_col,
        fwd_ret_col="ret_fwd_1",
    )

    # Drop rows that can’t be used (NaNs from rolling features + last day per symbol)
    d = d.dropna(subset=feature_cols + [label_col])

    # --- 3) Walk-forward splits and yield X/y ---
    for train_df, test_df in walk_forward_splits(
        d,
        date_col=date_col,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
        gap_days=gap_days,
        anchored=anchored,
    ):
        X_train = train_df[feature_cols]
        y_train = train_df[label_col].astype(int)

        X_test = test_df[feature_cols]
        y_test = test_df[label_col].astype(int)

        yield X_train, y_train, X_test, y_test