"""
Pytest configuration and shared fixtures.
"""
import sys
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, Mock

# Add project root and api directory to path so "app" and "engine" can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
API_DIR = PROJECT_ROOT / "api"
for path in (str(PROJECT_ROOT), str(API_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import pandas as pd
import pytest


def pytest_configure(config):
    """Ensure tmp dir exists so --basetemp=tmp/pytest can be created."""
    tmp_base = PROJECT_ROOT / "tmp"
    tmp_base.mkdir(exist_ok=True)


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Create sample OHLCV DataFrame for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
    prices = 100.0 + pd.Series(range(100)) * 0.5 + pd.Series(range(100)).apply(lambda x: (x % 10 - 5) * 0.1)
    
    df = pd.DataFrame({
        "timestamp": dates,
        "open": prices + 0.1,
        "high": prices + 0.5,
        "low": prices - 0.5,
        "close": prices,
        "volume": [1000 + i * 10 for i in range(100)],
        "c": prices,  # Polygon format
        "o": prices + 0.1,
        "h": prices + 0.5,
        "l": prices - 0.5,
        "v": [1000 + i * 10 for i in range(100)],
    })
    
    # Add features for mean revert strategy
    df["sma20"] = df["close"].rolling(20).mean()
    df["std20"] = df["close"].rolling(20).std()
    
    return df.dropna()


@pytest.fixture
def sample_ohlcv_data_with_features(sample_ohlcv_data: pd.DataFrame) -> pd.DataFrame:
    """Create sample OHLCV DataFrame with features for strategy testing."""
    df = sample_ohlcv_data.copy()
    df = df.set_index("timestamp")
    return df


@pytest.fixture
def mock_emit() -> Generator[MagicMock, None, None]:
    """Mock event emitter function."""
    emit = MagicMock()
    yield emit


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create temporary data directory for testing."""
    data_dir = tmp_path / "data" / "raw" / "stocks"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture
def mock_engine_client():
    """Create a mock EngineClient for testing."""
    from app.services.engine_client import EngineClient
    client = EngineClient()
    return client
