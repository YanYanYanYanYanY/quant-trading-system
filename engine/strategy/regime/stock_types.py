"""
Stock regime types and interfaces.

Defines stock-level regime classification for individual securities.
The stock regime detector classifies each stock in a universe based on
its own characteristics plus the market regime context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Protocol, runtime_checkable

import pandas as pd

if TYPE_CHECKING:
    from .market_types import MarketRegimeState


class StockRegimeLabel(str, Enum):
    """
    Stock-level regime classification.

    These labels determine how a strategy should interact with each stock.

    Values
    ------
    TRENDING_UP : str
        Stock shows clear upward trend with favorable risk characteristics.
        Suitable for momentum long entries.
    TRENDING_DOWN : str
        Stock shows clear downward trend.
        Suitable for short-side strategies or avoidance for long-only.
    RANGE_LOW_VOL : str
        Stock is range-bound with low volatility, mean-reverting behavior.
        Suitable for mean-reversion strategies at range extremes.
    STRESSED_HIGH_VOL : str
        Stock experiencing elevated volatility and/or deep drawdown.
        Requires special handling: wider stops, reduced size, or avoidance.
    NO_TRADE : str
        Stock should not be traded due to:
        - Insufficient data or history
        - Data quality issues (NaN features)
        - Extreme conditions outside model bounds
    """
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE_LOW_VOL = "range_low_vol"
    STRESSED_HIGH_VOL = "stressed_high_vol"
    NO_TRADE = "no_trade"


def clamp01(x: float) -> float:
    """
    Clamp a value to the [0, 1] interval.

    Parameters
    ----------
    x : float
        Input value to clamp.

    Returns
    -------
    float
        Value clamped to [0, 1].

    Examples
    --------
    >>> clamp01(0.5)
    0.5
    >>> clamp01(-0.3)
    0.0
    >>> clamp01(1.7)
    1.0
    """
    return max(0.0, min(1.0, x))


@dataclass
class StockRegimeState:
    """
    Regime state for a single stock.

    Represents the regime classification for one symbol at a point in time,
    including the confidence level and underlying reason scores.

    Attributes
    ----------
    symbol : str
        Stock ticker symbol.
    asof : pd.Timestamp
        Timestamp of the regime assessment.
    label : StockRegimeLabel
        Current regime classification for this stock.
    confidence : float
        Confidence in the regime assignment, in [0, 1].
        Higher values indicate clearer regime signals.
    reasons : Dict[str, float]
        Dictionary of scores/metrics contributing to the classification:
        - rv20_z: 20-day realized vol z-score vs stock's own history
        - trend_gap: Distance from trend (e.g., price vs 50d MA, normalized)
        - mom63: 63-day (quarterly) momentum score
        - adv20: 20-day average daily volume (liquidity)
        - shock_z: Recent return shock z-score (for event detection)
        - spread_bps: Typical bid-ask spread in basis points
        - Additional detector-specific metrics

    Examples
    --------
    >>> state = StockRegimeState(
    ...     symbol="AAPL",
    ...     asof=pd.Timestamp("2024-01-15"),
    ...     label=StockRegimeLabel.TRENDING,
    ...     confidence=0.75,
    ...     reasons={"rv20_z": -0.3, "trend_gap": 0.02, "mom63": 0.15}
    ... )
    """
    symbol: str
    asof: pd.Timestamp
    label: StockRegimeLabel
    confidence: float
    reasons: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if not self.symbol:
            raise ValueError("symbol cannot be empty")

    def __repr__(self) -> str:
        return (
            f"StockRegimeState(symbol={self.symbol!r}, asof={self.asof}, "
            f"label={self.label.value}, confidence={self.confidence:.2f})"
        )

    @property
    def is_tradeable(self) -> bool:
        """Return True if stock is in a tradeable regime."""
        return self.label != StockRegimeLabel.NO_TRADE

    @property
    def is_trending(self) -> bool:
        """Return True if stock is in a trending regime (up or down)."""
        return self.label in (
            StockRegimeLabel.TRENDING_UP,
            StockRegimeLabel.TRENDING_DOWN,
        )

    @property
    def is_trending_up(self) -> bool:
        """Return True if stock is trending up."""
        return self.label == StockRegimeLabel.TRENDING_UP

    @property
    def is_trending_down(self) -> bool:
        """Return True if stock is trending down."""
        return self.label == StockRegimeLabel.TRENDING_DOWN

    @property
    def is_ranging(self) -> bool:
        """Return True if stock is in ranging / low-vol regime."""
        return self.label == StockRegimeLabel.RANGE_LOW_VOL

    @property
    def is_stressed(self) -> bool:
        """Return True if stock has elevated volatility / drawdown."""
        return self.label == StockRegimeLabel.STRESSED_HIGH_VOL

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "asof": str(self.asof),
            "label": self.label.value,
            "confidence": self.confidence,
            "reasons": self.reasons,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "StockRegimeState":
        """Create instance from dictionary."""
        return cls(
            symbol=data["symbol"],
            asof=pd.Timestamp(data["asof"]),
            label=StockRegimeLabel(data["label"]),
            confidence=data["confidence"],
            reasons=data.get("reasons", {}),
        )


@dataclass
class StockRegimeMap:
    """
    Collection of stock regime states for a universe.

    Maps symbols to their regime states at a common point in time.
    This is the primary output of the stock regime detector.

    Attributes
    ----------
    asof : pd.Timestamp
        Timestamp of the regime assessment (common for all stocks).
    states : Dict[str, StockRegimeState]
        Mapping from symbol to its regime state.

    Examples
    --------
    >>> regime_map = StockRegimeMap(
    ...     asof=pd.Timestamp("2024-01-15"),
    ...     states={
    ...         "AAPL": StockRegimeState(...),
    ...         "MSFT": StockRegimeState(...),
    ...     }
    ... )
    >>> aapl_state = regime_map["AAPL"]
    >>> trending_stocks = regime_map.filter_by_label(StockRegimeLabel.TRENDING)
    """
    asof: pd.Timestamp
    states: Dict[str, StockRegimeState] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return number of stocks in the map."""
        return len(self.states)

    def __contains__(self, symbol: str) -> bool:
        """Check if symbol is in the map."""
        return symbol in self.states

    def __getitem__(self, symbol: str) -> StockRegimeState:
        """Get regime state for a symbol."""
        return self.states[symbol]

    def __iter__(self):
        """Iterate over symbols."""
        return iter(self.states)

    def get(self, symbol: str, default: StockRegimeState = None) -> StockRegimeState:
        """Get regime state for a symbol with optional default."""
        return self.states.get(symbol, default)

    @property
    def symbols(self) -> List[str]:
        """Return list of all symbols in the map."""
        return list(self.states.keys())

    def filter_by_label(self, label: StockRegimeLabel) -> Dict[str, StockRegimeState]:
        """
        Filter states by regime label.

        Parameters
        ----------
        label : StockRegimeLabel
            Regime label to filter by.

        Returns
        -------
        Dict[str, StockRegimeState]
            Subset of states matching the label.
        """
        return {
            sym: state for sym, state in self.states.items()
            if state.label == label
        }

    def filter_tradeable(self) -> Dict[str, StockRegimeState]:
        """
        Filter to only tradeable stocks (not NO_TRADE).

        Returns
        -------
        Dict[str, StockRegimeState]
            Subset of states that are tradeable.
        """
        return {
            sym: state for sym, state in self.states.items()
            if state.is_tradeable
        }

    def filter_by_confidence(
        self, min_confidence: float = 0.5
    ) -> Dict[str, StockRegimeState]:
        """
        Filter states by minimum confidence threshold.

        Parameters
        ----------
        min_confidence : float
            Minimum confidence threshold (default 0.5).

        Returns
        -------
        Dict[str, StockRegimeState]
            Subset of states meeting confidence threshold.
        """
        return {
            sym: state for sym, state in self.states.items()
            if state.confidence >= min_confidence
        }

    def summary(self) -> Dict[str, int]:
        """
        Return count of stocks by regime label.

        Returns
        -------
        Dict[str, int]
            Mapping from label name to count.
        """
        counts = {label.value: 0 for label in StockRegimeLabel}
        for state in self.states.values():
            counts[state.label.value] += 1
        return counts

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert to DataFrame for analysis.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: symbol, label, confidence, plus reason columns.
        """
        if not self.states:
            return pd.DataFrame(columns=["symbol", "label", "confidence"])

        records = []
        for sym, state in self.states.items():
            record = {
                "symbol": sym,
                "label": state.label.value,
                "confidence": state.confidence,
                **state.reasons,
            }
            records.append(record)

        return pd.DataFrame(records).set_index("symbol")

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "asof": str(self.asof),
            "states": {sym: state.to_dict() for sym, state in self.states.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "StockRegimeMap":
        """Create instance from dictionary."""
        return cls(
            asof=pd.Timestamp(data["asof"]),
            states={
                sym: StockRegimeState.from_dict(state_data)
                for sym, state_data in data.get("states", {}).items()
            },
        )


# Re-export the canonical FeatureBundle from the features module.
# Consumers that import FeatureBundle from this module continue to work.
from ...features.types import FeatureBundle as FeatureBundle  # noqa: F401


@runtime_checkable
class StockRegimeDetector(Protocol):
    """
    Protocol for stock regime detection.

    Implementations classify each stock in a universe into a regime
    based on stock-specific features and market context.

    This is a Protocol (structural subtyping) so implementations
    don't need to explicitly inherit - they just need to implement
    the required method signature.

    Examples
    --------
    >>> class RuleBasedDetector:
    ...     def detect(
    ...         self,
    ...         asof: pd.Timestamp,
    ...         market_regime: MarketRegimeState,
    ...         features: FeatureBundle,
    ...         universe: List[str],
    ...     ) -> StockRegimeMap:
    ...         # Implementation here
    ...         pass
    ...
    >>> detector: StockRegimeDetector = RuleBasedDetector()
    """

    def detect(
        self,
        asof: pd.Timestamp,
        market_regime: "MarketRegimeState",
        features: FeatureBundle,
        universe: List[str],
    ) -> StockRegimeMap:
        """
        Detect regime for each stock in the universe.

        Parameters
        ----------
        asof : pd.Timestamp
            Timestamp for the regime assessment.
        market_regime : MarketRegimeState
            Current market regime from market-level detector.
            Used to adjust stock regime thresholds and confidence.
        features : FeatureBundle
            Cross-sectional feature snapshot (``N × F`` matrix).
            Use ``features.get(symbol, feature_name)`` for scalar
            lookups.  Expected features include ``rv20``, ``mom63``,
            ``normalized_slope``, ``rv20_z``, ``max_dd_90``,
            ``trend_gap``, ``dev20``, ``rv60``, ``adv20``, etc.
        universe : List[str]
            List of symbols to classify.

        Returns
        -------
        StockRegimeMap
            Collection of regime states for all symbols in universe.

        Notes
        -----
        Implementations should:
        1. Handle missing data gracefully (classify as NO_TRADE)
        2. Apply smoothing (min duration, hysteresis)
        3. Compute meaningful confidence scores
        4. Consider market regime when setting thresholds
        """
        ...
