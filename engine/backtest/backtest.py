from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd

from ..strategy.base import Strategy, StrategyState, Order, Side


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    qty: int
    entry_price: float
    exit_price: float
    reason: str
    pnl: float
    pnl_pct: float
    holding_bars: int


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: List[Trade]
    stats: Dict[str, float]


def _exec_price(price: float, side: Side, slippage_bps: float) -> float:
    """
    slippage_bps: e.g., 5 means 5 basis points (0.05%)
    BUY executes a bit higher, SELL executes a bit lower.
    """
    slip = (slippage_bps / 10_000.0)
    if side == Side.BUY:
        return price * (1.0 + slip)
    else:
        return price * (1.0 - slip)


def run_backtest(
    data: pd.DataFrame,
    strategy: Strategy,
    initial_cash: float = 10_000.0,
    commission_per_trade: float = 0.0,  # flat fee each order execution
    slippage_bps: float = 0.0,
    context: Optional[Dict] = None,
) -> BacktestResult:
    """
    A simple single-asset backtester:
    - Executes market orders on the same bar's close (with optional slippage).
    - Long-only starter mechanics.
    - Tracks equity curve (mark-to-market using close).
    """
    if context is None:
        context = {}

    if "close" not in data.columns:
        raise ValueError("data must contain a 'close' column")

    data = data.copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        # still works, but trades will carry whatever index type you use
        pass

    state = StrategyState()
    cash = float(initial_cash)

    equity = []
    idxs = list(data.index)
    close = data["close"].astype(float)

    trades: List[Trade] = []
    open_trade: Optional[Dict] = None

    warmup = strategy.warmup_bars()

    for i in range(len(data)):
        row = data.iloc[i]
        price = float(row["close"])

        # Strategy decision (after warmup)
        order: Optional[Order] = None
        if i >= warmup:
            order = strategy.on_bar(i=i, row=row, data=data, state=state, context=context)

        # Execute order
        if order is not None:
            if order.qty <= 0:
                raise ValueError("order.qty must be positive")

            exec_px = _exec_price(price, order.side, slippage_bps)

            if order.side == Side.BUY:
                # Long-only: open or add (we keep it simple; allow only if flat)
                if state.position != 0:
                    # keep simple for v1
                    raise ValueError("This backtest v1 expects long-only and no adding; must be flat to BUY.")

                cost = exec_px * order.qty + commission_per_trade
                if cost > cash:
                    # insufficient cash -> skip (or you could partial fill)
                    order = None
                else:
                    cash -= cost
                    state.position += order.qty
                    state.entry_price = exec_px
                    state.entry_index = idxs[i]
                    state.hold_bars = 0

                    open_trade = {
                        "entry_time": idxs[i],
                        "qty": order.qty,
                        "entry_price": exec_px,
                    }

            elif order.side == Side.SELL:
                if state.position <= 0:
                    # nothing to sell
                    order = None
                else:
                    qty = min(order.qty, state.position)
                    proceeds = exec_px * qty - commission_per_trade
                    cash += proceeds
                    state.position -= qty

                    # close trade fully for v1
                    if state.position == 0 and open_trade is not None:
                        entry_time = open_trade["entry_time"]
                        entry_price = float(open_trade["entry_price"])
                        holding_bars = int(state.hold_bars)

                        pnl = (exec_px - entry_price) * qty - 2 * commission_per_trade
                        pnl_pct = (exec_px - entry_price) / entry_price

                        trades.append(
                            Trade(
                                entry_time=entry_time,
                                exit_time=idxs[i],
                                qty=qty,
                                entry_price=entry_price,
                                exit_price=exec_px,
                                reason=order.reason or "exit",
                                pnl=float(pnl),
                                pnl_pct=float(pnl_pct),
                                holding_bars=holding_bars,
                            )
                        )

                        open_trade = None
                        state.entry_price = None
                        state.entry_index = None
                        state.hold_bars = 0

        # Mark-to-market equity using close price
        pos_value = state.position * price
        equity.append(cash + pos_value)

    equity_curve = pd.Series(equity, index=data.index, name="equity")

    # Basic stats
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0
    max_dd = _max_drawdown(equity_curve)
    win_rate = (sum(1 for t in trades if t.pnl > 0) / len(trades)) if trades else 0.0
    avg_trade = (sum(t.pnl for t in trades) / len(trades)) if trades else 0.0

    stats = {
        "initial_cash": float(initial_cash),
        "final_equity": float(equity_curve.iloc[-1]),
        "total_return": float(total_return),
        "max_drawdown": float(max_dd),
        "num_trades": float(len(trades)),
        "win_rate": float(win_rate),
        "avg_trade_pnl": float(avg_trade),
    }

    return BacktestResult(equity_curve=equity_curve, trades=trades, stats=stats)


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())