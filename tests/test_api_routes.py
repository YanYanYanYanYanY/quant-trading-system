"""
Unit tests for API routes.
"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas.requests import BacktestRequest
from app.deps import get_engine_client


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def mock_engine_client():
    """Create mock engine client."""
    client = MagicMock()
    client.list_symbols.return_value = ["AAPL", "MSFT"]
    client.list_strategies.return_value = ["mean_revert_v1"]
    client.get_candles.return_value = []
    client.start_backtest.return_value = "test_job_id"
    client.get_backtest_status.return_value = {
        "job_id": "test_job_id",
        "state": "queued",
        "progress": 0.0,
        "result": None,
        "error": None,
    }
    return client


class TestHealthRoute:
    """Test suite for health check route."""
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200


class TestDataRoutes:
    """Test suite for data routes."""
    
    @patch("app.routes.data.get_engine_client")
    def test_list_symbols(self, mock_get_client, client, mock_engine_client):
        """Test listing symbols."""
        mock_get_client.return_value = mock_engine_client
        
        response = client.get("/data/symbols")
        assert response.status_code == 200
        data = response.json()
        assert "symbols" in data
        assert "AAPL" in data["symbols"]
    
    @patch("app.routes.data.get_engine_client")
    def test_get_candles(self, mock_get_client, client, mock_engine_client):
        """Test getting candles."""
        mock_get_client.return_value = mock_engine_client
        mock_engine_client.get_candles.return_value = [
            {
                "ts": "2024-01-01T00:00:00Z",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 101.0,
                "volume": 1000.0,
            }
        ]
        
        response = client.get("/data/candles?symbol=AAPL&tf=1d&limit=10")
        assert response.status_code == 200
        data = response.json()
        assert "candles" in data


class TestBacktestRoutes:
    """Test suite for backtest routes."""

    @patch("app.routes.backtests.get_ws_manager")
    def test_start_backtest(self, mock_get_ws, client, mock_engine_client):
        """Test starting a backtest."""
        app = client.app
        app.dependency_overrides[get_engine_client] = lambda: mock_engine_client

        try:
            request_data = {
                "strategy": "mean_revert_v1",
                "symbol": "AAPL",
                "tf": "1d",
                "start": "2024-01-01",
                "end": "2024-01-31",
                "initial_cash": 10000.0
            }

            response = client.post("/backtests", json=request_data)
            assert response.status_code == 200
            data = response.json()
            assert "job_id" in data
            assert data["job_id"] == "test_job_id"
        finally:
            app.dependency_overrides.pop(get_engine_client, None)

    def test_get_backtest_status(self, client, mock_engine_client):
        """Test getting backtest status."""
        app = client.app
        app.dependency_overrides[get_engine_client] = lambda: mock_engine_client

        try:
            response = client.get("/backtests/test_job_id")
            assert response.status_code == 200
            data = response.json()
            assert data["job_id"] == "test_job_id"
            assert data["state"] == "queued"
        finally:
            app.dependency_overrides.pop(get_engine_client, None)


class TestStrategiesRoute:
    """Test suite for strategies route."""
    
    @patch("app.routes.strategies.get_engine_client")
    def test_list_strategies(self, mock_get_client, client, mock_engine_client):
        """Test listing strategies."""
        mock_get_client.return_value = mock_engine_client
        
        response = client.get("/strategies")
        assert response.status_code == 200
        data = response.json()
        assert "strategies" in data
        assert "mean_revert_v1" in data["strategies"]
