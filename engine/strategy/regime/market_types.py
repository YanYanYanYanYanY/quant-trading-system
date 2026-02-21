"""
Market regime types and interfaces.

Defines the market-level regime classification output that feeds into
stock-level regime detection and strategy decisions.

This module provides a minimal interface layer. The actual detection
logic lives in engine.strategy.regime.market_regime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import pandas as pd


class MarketRegimeLabel(str, Enum):
    """
    Market-level regime classification.

    These labels represent broad market conditions derived from SPY analysis.

    Values
    ------
    CALM_TREND : str
        Low volatility with clear directional trend (up or down).
        Favorable for momentum and trend-following strategies.
    CALM_RANGE : str
        Low volatility without clear trend direction.
        Favorable for mean-reversion and range-bound strategies.
    STRESS : str
        Elevated volatility and/or moderate drawdown.
        Requires position sizing adjustments and tighter risk controls.
    CRISIS : str
        Extreme volatility and/or severe drawdown.
        May warrant defensive positioning or trading halt.
    """
    CALM_TREND = "calm_trend"
    CALM_RANGE = "calm_range"
    STRESS = "stress"
    CRISIS = "crisis"


@dataclass
class MarketRegimeState:
    """
    Output from market regime detection.

    Represents the current market regime assessment at a point in time.
    This feeds into stock-level regime detection and strategy decisions.

    Attributes
    ----------
    asof : pd.Timestamp
        Timestamp of the regime assessment.
    label : MarketRegimeLabel
        Current regime classification.
    confidence : float
        Confidence in the regime assignment, in [0, 1].
        Higher values indicate clearer regime signals.
    scores : Dict[str, Any]
        Dictionary of underlying scores/metrics used in classification:
        - rv20: 20-day realized volatility (annualized)
        - z_vol: Volatility z-score vs history
        - max_dd: Maximum drawdown over trailing window
        - normalized_slope: Normalized trend strength
        - ma_trend_flag: MA crossover direction (+1 / -1 / 0)
        - raw_regime: Internal regime label before stability controls
        - raw_confidence: Confidence before stability controls
        - days_in_regime: Days spent in current regime
        - Additional detector-specific metrics

    Examples
    --------
    >>> state = MarketRegimeState(
    ...     asof=pd.Timestamp("2024-01-15"),
    ...     label=MarketRegimeLabel.CALM_TREND,
    ...     confidence=0.82,
    ...     scores={"rv20": 0.12, "z_vol": -0.5, "max_dd": -0.03}
    ... )
    """
    asof: pd.Timestamp
    label: MarketRegimeLabel
    confidence: float
    scores: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    def __repr__(self) -> str:
        return (
            f"MarketRegimeState(asof={self.asof}, label={self.label.value}, "
            f"confidence={self.confidence:.2f})"
        )

    @property
    def is_stressed(self) -> bool:
        """Return True if market is in stress or crisis regime."""
        return self.label in (MarketRegimeLabel.STRESS, MarketRegimeLabel.CRISIS)

    @property
    def is_calm(self) -> bool:
        """Return True if market is in a calm regime."""
        return self.label in (MarketRegimeLabel.CALM_TREND, MarketRegimeLabel.CALM_RANGE)

    @property
    def is_trending(self) -> bool:
        """Return True if market shows clear trend."""
        return self.label == MarketRegimeLabel.CALM_TREND

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "asof": str(self.asof),
            "label": self.label.value,
            "confidence": self.confidence,
            "scores": self.scores,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MarketRegimeState":
        """Create instance from dictionary."""
        return cls(
            asof=pd.Timestamp(data["asof"]),
            label=MarketRegimeLabel(data["label"]),
            confidence=data["confidence"],
            scores=data.get("scores", {}),
        )
