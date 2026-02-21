"""
Feature engine package.

Public API
----------
- :class:`FeatureEngine` — event-driven per-symbol feature computation.
- :class:`FeatureBundle` — cross-sectional feature snapshot (N×F matrix).
- :class:`FeatureSpec` / :func:`default_feature_specs` — feature catalogue.
- :class:`FeaturePipeline` — OHLCV files → FeatureEngine → FeatureBundle.
- :func:`load_ohlcv` / :func:`load_universe_ohlcv` — data loaders.
- :func:`cross_sectional_zscore`, :func:`percentile_rank`, :func:`winsorize`
  — cross-sectional utilities.
"""

from .cross_section import cross_sectional_zscore, percentile_rank, winsorize
from .definitions import FeatureSpec, default_feature_specs
from .feature_engine import FeatureEngine
from .pipeline import (
    FeaturePipeline,
    load_ohlcv,
    load_universe_ohlcv,
)
from .types import FeatureBundle

__all__ = [
    "FeatureEngine",
    "FeatureBundle",
    "FeaturePipeline",
    "FeatureSpec",
    "default_feature_specs",
    "load_ohlcv",
    "load_universe_ohlcv",
    "cross_sectional_zscore",
    "percentile_rank",
    "winsorize",
]
