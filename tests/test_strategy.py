"""
Unit tests for strategy module.
"""
import pandas as pd
import pytest

from engine.strategy.base import Strategy, StrategyState, Order, Side
from engine.strategy.mean_revert import MeanReversionStrategy
from engine.strategy.rules import MeanRevertParams, mean_revert_entry_signal, mean_revert_exit_signal


class TestMeanReversionStrategy:
    """Test suite for MeanReversionStrategy."""
    
    def test_init(self):
        """Test strategy initialization."""
        strategy = MeanReversionStrategy()
        assert strategy.name == "mean_revert_v1"
        assert strategy.warmup_bars() == 20
    
    def test_warmup_bars(self):
        """Test warmup bars calculation."""
        strategy = MeanReversionStrategy()
        assert strategy.warmup_bars() == 20
    
    def test_on_bar_no_signal(self, sample_ohlcv_data_with_features):
        """Test strategy when no signal is generated."""
        strategy = MeanReversionStrategy()
        state = StrategyState()
        data = sample_ohlcv_data_with_features.copy()
        
        # Use a row where close is not below SMA - 2*STD
        row = data.iloc[50]  # Middle of dataset
        order = strategy.on_bar(50, row, data, state, {})
        
        assert order is None
    
    def test_on_bar_entry_signal(self):
        """Test strategy entry signal generation."""
        # Create data where close < SMA - 2*STD
        data = pd.DataFrame({
            "close": [95.0] * 30,  # Below SMA
            "sma20": [100.0] * 30,
            "std20": [2.0] * 30,
        })
        
        row = data.iloc[25]  # After warmup
        params = MeanRevertParams(entry_k=2.0)
        
        signal = mean_revert_entry_signal(row, params)
        assert signal is True
    
    def test_on_bar_exit_signal(self):
        """Test strategy exit signal generation."""
        # Create data where close >= SMA (exit to mean)
        data = pd.DataFrame({
            "close": [102.0],  # Above SMA
            "sma20": [100.0],
        })
        
        row = data.iloc[0]
        entry_price = 95.0
        hold_bars = 5
        
        params = MeanRevertParams(exit_to_mean=True)
        reason = mean_revert_exit_signal(row, entry_price, hold_bars, params)
        
        assert reason == "exit_to_mean"
    
    def test_exit_signal_stop_loss(self):
        """Test stop loss exit signal."""
        data = pd.DataFrame({
            "close": [92.0],  # 3% below entry
            "sma20": [100.0],
        })
        
        row = data.iloc[0]
        entry_price = 95.0  # 3% loss
        hold_bars = 1
        
        params = MeanRevertParams(stop_loss_pct=0.03)
        reason = mean_revert_exit_signal(row, entry_price, hold_bars, params)
        
        assert reason == "stop_loss"
    
    def test_exit_signal_time_stop(self):
        """Test time-based exit signal."""
        data = pd.DataFrame({
            "close": [96.0],
            "sma20": [100.0],
        })
        
        row = data.iloc[0]
        entry_price = 95.0
        hold_bars = 10  # Max hold bars
        
        params = MeanRevertParams(max_hold_bars=10)
        reason = mean_revert_exit_signal(row, entry_price, hold_bars, params)
        
        assert reason == "time_stop"


class TestStrategyState:
    """Test suite for StrategyState."""
    
    def test_init(self):
        """Test StrategyState initialization."""
        state = StrategyState()
        assert state.position == 0
        assert state.entry_price is None
        assert state.entry_index is None
        assert state.hold_bars == 0
    
    def test_state_mutation(self):
        """Test that StrategyState can be mutated."""
        state = StrategyState()
        state.position = 10
        state.entry_price = 100.0
        state.hold_bars = 5
        
        assert state.position == 10
        assert state.entry_price == 100.0
        assert state.hold_bars == 5


class TestOrder:
    """Test suite for Order dataclass."""
    
    def test_order_creation(self):
        """Test Order creation."""
        order = Order(side=Side.BUY, qty=10, reason="test")
        assert order.side == Side.BUY
        assert order.qty == 10
        assert order.reason == "test"
    
    def test_order_immutability(self):
        """Test that Order is immutable."""
        order = Order(side=Side.BUY, qty=10)
        
        with pytest.raises(Exception):  # dataclass frozen raises exception
            order.qty = 20
