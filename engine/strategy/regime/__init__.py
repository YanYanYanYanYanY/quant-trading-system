"""
Regime detection module.

Provides market-level (SPY) and stock-level regime classification
for use by strategy layer and alpha ensemble.
"""

from .config import RegimeConfig
from .market_regime import MarketRegimeDetector
from .market_types import MarketRegimeLabel, MarketRegimeState
from .stock_regime_config import StockRegimeConfig
from .stock_regime_rule_based import RuleBasedStockRegimeDetector
from .stock_types import (
    FeatureBundle,
    StockRegimeDetector,
    StockRegimeLabel,
    StockRegimeMap,
    StockRegimeState,
    clamp01,
)

__all__ = [
    # SPY regime detector
    "MarketRegimeDetector",
    "RegimeConfig",
    # Market regime types
    "MarketRegimeLabel",
    "MarketRegimeState",
    # Stock regime types & protocol
    "StockRegimeLabel",
    "StockRegimeState",
    "StockRegimeMap",
    "StockRegimeDetector",
    "FeatureBundle",
    "clamp01",
    # Stock regime implementation
    "StockRegimeConfig",
    "RuleBasedStockRegimeDetector",
]
