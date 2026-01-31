from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass(frozen=True)
class MeanRevertParams:
    sma_col: str = "sma20"
    std_col: str = "std20"
    close_col: str = "close"

    entry_k: float = 2.0        # enter when close < sma - entry_k*std
    exit_to_mean: bool = True   # exit when close >= sma

    stop_loss_pct: float = 0.03 # 3% stop loss
    max_hold_bars: int = 10     # time stop (bars)

    qty: int = 1                # fixed position size (units)


def can_trade(row: pd.Series, cols: list[str]) -> bool:
    """Return True if all required columns are present and non-NaN on this row."""
    for c in cols:
        if c not in row.index:
            return False
        v = row[c]
        if v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v):
            return False
    return True


def mean_revert_entry_signal(row: pd.Series, p: MeanRevertParams) -> bool:
    if not can_trade(row, [p.close_col, p.sma_col, p.std_col]):
        return False
    close = float(row[p.close_col])
    sma = float(row[p.sma_col])
    std = float(row[p.std_col])
    return close < (sma - p.entry_k * std)


def mean_revert_exit_signal(
    row: pd.Series,
    entry_price: float,
    hold_bars: int,
    p: MeanRevertParams,
) -> Optional[str]:
    """
    Returns a string reason if we should exit, else None.
    """
    if not can_trade(row, [p.close_col, p.sma_col]):
        return None

    close = float(row[p.close_col])
    sma = float(row[p.sma_col])

    # 1) exit to mean
    if p.exit_to_mean and close >= sma:
        return "exit_to_mean"

    # 2) stop loss
    # long-only: loss% = (entry - close)/entry
    loss_pct = (entry_price - close) / entry_price
    if loss_pct >= p.stop_loss_pct:
        return "stop_loss"

    # 3) time stop
    if hold_bars >= p.max_hold_bars:
        return "time_stop"

    return None