"""
Strategy module.

Contains base strategy classes, specific strategy implementations,
market/stock regime detection, and the alpha scoring layer.
"""

from .base import Order, Side, Strategy, StrategyState
from .regime_alpha_strategy import RegimeAlphaStrategy
from .regime import (
    FeatureBundle,
    MarketRegimeDetector,
    MarketRegimeLabel,
    MarketRegimeState,
    RegimeConfig,
    RuleBasedStockRegimeDetector,
    StockRegimeConfig,
    StockRegimeDetector,
    StockRegimeLabel,
    StockRegimeMap,
    StockRegimeState,
    clamp01,
)
from .alpha import (
    AlphaModel,
    AlphaModelName,
    AlphaScore,
    AlphaVector,
    cross_sectional_zscore,
    keep_positive,
    rank_to_unit,
    winsorize,
)

__all__ = [
    # Strategy base
    "Order",
    "Side",
    "Strategy",
    "StrategyState",
    # Concrete strategy
    "RegimeAlphaStrategy",
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
    # Alpha layer
    "AlphaModelName",
    "AlphaScore",
    "AlphaVector",
    "AlphaModel",
    "winsorize",
    "cross_sectional_zscore",
    "keep_positive",
    "rank_to_unit",
]
