"""
Alpha layer module.

Provides alpha model scoring types, the ``AlphaModel`` protocol,
cross-sectional utility functions, and concrete alpha model
implementations for a long-only strategy.
"""

from .types import (
    AlphaModel,
    AlphaModelName,
    AlphaScore,
    AlphaVector,
    clamp01,
    cross_sectional_zscore,
    keep_positive,
    rank_to_unit,
    winsorize,
)
from .models import LowVolAlpha, MeanReversionAlpha, MomentumAlpha
from .router_long_only import LongOnlyAlphaRouter, LongOnlyRouterConfig
from .alpha_engine_long_only import AlphaEngineConfig, LongOnlyAlphaEngine

__all__ = [
    # Enum
    "AlphaModelName",
    # Dataclasses
    "AlphaScore",
    "AlphaVector",
    # Protocol
    "AlphaModel",
    # Concrete models
    "LowVolAlpha",
    "MeanReversionAlpha",
    "MomentumAlpha",
    # Router
    "LongOnlyAlphaRouter",
    "LongOnlyRouterConfig",
    # Engine
    "AlphaEngineConfig",
    "LongOnlyAlphaEngine",
    # Utilities
    "clamp01",
    "keep_positive",
    "winsorize",
    "cross_sectional_zscore",
    "rank_to_unit",
]
