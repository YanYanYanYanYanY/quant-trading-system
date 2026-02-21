"""
Unit tests for backtest module.
"""
import pandas as pd
import pytest

from engine.backtest.backtest import run_backtest, BacktestResult, _max_drawdown
from engine.strategy.base import Strategy, StrategyState, Order, Side


class SimpleTestStrategy(Strategy):
    """Simple test strategy for backtesting."""
    
    def __init__(self):
        super().__init__(warmup_bars=0)
    
    def warmup_bar(self, bar) -> None:
        pass
    
    def on_bar(self, bar):
        """Buy on first bar, sell on 5th bar."""
        i = bar["_i"]
        state = bar["_state"]
        if i == 0 and state.position == 0:
            return Order(side=Side.BUY, qty=1, reason="entry")
        elif i == 4 and state.position > 0:
            return Order(side=Side.SELL, qty=state.position, reason="exit")
        return None


class TestBacktest:
    """Test suite for backtest module."""
    
    def test_run_backtest_basic(self, sample_ohlcv_data_with_features):
        """Test basic backtest execution."""
        strategy = SimpleTestStrategy()
        data = sample_ohlcv_data_with_features.copy()
        
        result = run_backtest(
            data=data,
            strategy=strategy,
            initial_cash=10000.0,
            slippage_bps=0.0,
        )
        
        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) == len(data)
        assert result.stats["initial_cash"] == 10000.0
        assert result.stats["final_equity"] > 0
    
    def test_run_backtest_with_trades(self, sample_ohlcv_data_with_features):
        """Test backtest that generates trades."""
        strategy = SimpleTestStrategy()
        data = sample_ohlcv_data_with_features.copy()
        
        result = run_backtest(
            data=data,
            strategy=strategy,
            initial_cash=10000.0,
        )
        
        assert len(result.trades) > 0
        assert result.stats["num_trades"] > 0
    
    def test_run_backtest_insufficient_cash(self, sample_ohlcv_data_with_features):
        """Test backtest with insufficient cash."""
        strategy = SimpleTestStrategy()
        data = sample_ohlcv_data_with_features.copy()
        
        # Very low initial cash
        result = run_backtest(
            data=data,
            strategy=strategy,
            initial_cash=1.0,  # Too low to buy
        )
        
        # Should complete without error, but no trades
        assert isinstance(result, BacktestResult)
        assert result.stats["num_trades"] == 0
    
    def test_run_backtest_missing_close_column(self):
        """Test backtest with missing close column."""
        strategy = SimpleTestStrategy()
        data = pd.DataFrame({
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            # Missing "close"
        })
        
        with pytest.raises(ValueError, match="close"):
            run_backtest(data=data, strategy=strategy)
    
    def test_max_drawdown(self):
        """Test max drawdown calculation."""
        # Create equity curve with drawdown
        equity = pd.Series([100, 110, 105, 120, 115, 130])
        dd = _max_drawdown(equity)
        
        assert dd < 0  # Should be negative
        assert dd >= -0.15  # Max drawdown should be reasonable
    
    def test_max_drawdown_no_drawdown(self):
        """Test max drawdown with no drawdown."""
        equity = pd.Series([100, 110, 120, 130])  # Always increasing
        dd = _max_drawdown(equity)
        
        assert dd == 0.0
    
    def test_backtest_result_stats(self, sample_ohlcv_data_with_features):
        """Test that backtest result contains expected stats."""
        strategy = SimpleTestStrategy()
        data = sample_ohlcv_data_with_features.copy()
        
        result = run_backtest(
            data=data,
            strategy=strategy,
            initial_cash=10000.0,
        )
        
        stats = result.stats
        assert "initial_cash" in stats
        assert "final_equity" in stats
        assert "total_return" in stats
        assert "max_drawdown" in stats
        assert "num_trades" in stats
        assert "win_rate" in stats
        assert "avg_trade_pnl" in stats
        
        assert stats["initial_cash"] == 10000.0
        assert stats["final_equity"] > 0
        assert isinstance(stats["num_trades"], float)
