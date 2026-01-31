from __future__ import annotations

from typing import Dict, Optional
import pandas as pd

from .base import Strategy, StrategyState, Order, Side
from .rules import MeanRevertParams, mean_revert_entry_signal, mean_revert_exit_signal


class MeanReversionStrategy(Strategy):
    """
    Starter mean-reversion strategy (long-only):
    - Enter when close < SMA - k*STD
    - Exit when close >= SMA OR stop loss OR max hold bars
    """

    def __init__(self, params: MeanRevertParams = MeanRevertParams(), name: str = "mean_revert_v1"):
        super().__init__(name=name)
        self.p = params

    def warmup_bars(self) -> int:
        # Require SMA/STD to be available; assume 20 by default.
        # If you change to SMA(50), set your feature cols and warmup accordingly.
        return 20

    def on_bar(
        self,
        i: int,
        row: pd.Series,
        data: pd.DataFrame,
        state: StrategyState,
        context: Dict,
    ) -> Optional[Order]:
        close_col = self.p.close_col
        if close_col not in row.index or pd.isna(row[close_col]):
            return None

        # Update holding counter if in a position
        if state.position > 0:
            state.hold_bars += 1

            # exit?
            assert state.entry_price is not None
            reason = mean_revert_exit_signal(row, state.entry_price, state.hold_bars, self.p)
            if reason is not None:
                return Order(side=Side.SELL, qty=state.position, reason=reason)

            return None

        # If flat, check entry
        if mean_revert_entry_signal(row, self.p):
            return Order(side=Side.BUY, qty=self.p.qty, reason="mr_entry")

        return None