"""
Unit tests for EngineClient.
"""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from app.schemas.requests import BacktestRequest, PlaceOrderRequest
from app.services.engine_client import EngineClient, STRATEGY_REGISTRY


class TestEngineClient:
    """Test suite for EngineClient class."""
    
    def test_init(self):
        """Test EngineClient initialization."""
        client = EngineClient()
        assert client._mode == "stopped"
        assert client._backtests == {}
        assert client._orders == []
        assert client._positions == {}
        assert client._event_sink is None
    
    def test_utcnow_iso(self):
        """Test UTC timestamp generation."""
        client = EngineClient()
        timestamp = client.utcnow_iso()
        assert isinstance(timestamp, str)
        assert "T" in timestamp  # ISO format
        assert "Z" in timestamp or "+00:00" in timestamp  # UTC indicator
    
    def test_get_status(self):
        """Test status retrieval."""
        client = EngineClient()
        status = client.get_status()
        assert status["mode"] == "stopped"
        assert status["engine_ok"] is True
        
        client._mode = "paper"
        status = client.get_status()
        assert status["mode"] == "paper"
    
    def test_list_strategies(self):
        """Test strategy listing."""
        client = EngineClient()
        strategies = client.list_strategies()
        assert isinstance(strategies, list)
        assert "mean_revert_v1" in strategies
        assert len(strategies) == len(STRATEGY_REGISTRY)
    
    @patch("app.services.engine_client.get_tickers")
    @patch("app.services.engine_client.RAW_DIR")
    def test_list_symbols_from_ticker_file(self, mock_raw_dir, mock_get_tickers):
        """Test listing symbols from ticker file."""
        mock_ticker_file = Mock()
        mock_ticker_file.exists.return_value = True
        mock_stocks = Mock()
        mock_stocks.__truediv__ = Mock(return_value=mock_ticker_file)
        mock_raw_dir.__truediv__ = Mock(return_value=mock_stocks)
        mock_get_tickers.return_value = ["AAPL", "MSFT", "GOOGL"]

        client = EngineClient()
        symbols = client.list_symbols()

        assert symbols == ["AAPL", "MSFT", "GOOGL"]
        mock_get_tickers.assert_called_once()
    
    @patch("app.services.engine_client.RAW_DIR")
    def test_list_symbols_from_data_directory(self, mock_raw_dir):
        """Test listing symbols by scanning data directory."""
        mock_ticker_file = Mock()
        mock_ticker_file.exists.return_value = False

        mock_file1 = Mock()
        mock_file1.stem = "AAPL_1d"
        mock_file2 = Mock()
        mock_file2.stem = "MSFT_1d"

        mock_stocks = Mock()
        mock_stocks.exists.return_value = True
        mock_stocks.glob.return_value = [mock_file1, mock_file2]
        mock_stocks.__truediv__ = Mock(return_value=mock_ticker_file)

        mock_raw_dir.__truediv__ = Mock(return_value=mock_stocks)

        client = EngineClient()
        symbols = client.list_symbols()

        assert "AAPL" in symbols
        assert "MSFT" in symbols
    
    def test_list_symbols_fallback(self):
        """Test fallback to hardcoded symbols."""
        with patch("app.services.engine_client.RAW_DIR") as mock_raw_dir:
            mock_ticker_file = Mock()
            mock_ticker_file.exists.return_value = False

            mock_stocks = Mock()
            mock_stocks.exists.return_value = False
            mock_stocks.glob.return_value = []
            mock_stocks.__truediv__ = Mock(return_value=mock_ticker_file)

            mock_raw_dir.__truediv__ = Mock(return_value=mock_stocks)

            client = EngineClient()
            symbols = client.list_symbols()

            assert symbols == list(client._symbols)
    
    @patch("app.services.engine_client.file_exists")
    @patch("app.services.engine_client.load_parquet")
    def test_get_candles_from_parquet(self, mock_load_parquet, mock_file_exists):
        """Test loading candles from parquet file."""
        mock_file_exists.return_value = True
        
        # Create sample data
        dates = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
        df = pd.DataFrame({
            "timestamp": dates,
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.0] * 10,
            "volume": [1000] * 10,
        })
        mock_load_parquet.return_value = df
        
        client = EngineClient()
        candles = client.get_candles("AAPL", "1d", None, None, 10)
        
        assert len(candles) == 10
        assert all("ts" in c for c in candles)
        assert all("open" in c for c in candles)
        assert all("close" in c for c in candles)
        mock_load_parquet.assert_called_once()
    
    @patch("app.services.engine_client.file_exists")
    @patch("app.services.engine_client.load_csv")
    def test_get_candles_from_csv(self, mock_load_csv, mock_file_exists):
        """Test loading candles from CSV file."""
        mock_file_exists.side_effect = lambda x: "csv" in x
        
        dates = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
        df = pd.DataFrame({
            "timestamp": dates,
            "c": [100.0] * 5,  # Polygon format
            "o": [99.5] * 5,
            "h": [100.5] * 5,
            "l": [99.0] * 5,
            "v": [1000] * 5,
        })
        mock_load_csv.return_value = df
        
        client = EngineClient()
        candles = client.get_candles("AAPL", "day", None, None, 5)
        
        assert len(candles) == 5
        assert candles[0]["close"] == 100.0
        mock_load_csv.assert_called_once()
    
    def test_get_candles_with_date_filter(self):
        """Test candles filtering by date range."""
        with patch("app.services.engine_client.file_exists", return_value=True):
            with patch("app.services.engine_client.load_parquet") as mock_load:
                dates = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
                df = pd.DataFrame({
                    "timestamp": dates,
                    "open": [100.0] * 10,
                    "high": [101.0] * 10,
                    "low": [99.0] * 10,
                    "close": [100.0] * 10,
                    "volume": [1000] * 10,
                })
                mock_load.return_value = df
                
                client = EngineClient()
                candles = client.get_candles(
                    "AAPL", "1d",
                    start="2024-01-05",
                    end="2024-01-08",
                    limit=10
                )
                
                assert len(candles) <= 4  # Filtered range
    
    def test_get_candles_empty_result(self):
        """Test handling of missing data."""
        with patch("app.services.engine_client.file_exists", return_value=False):
            client = EngineClient()
            candles = client.get_candles("INVALID", "1d", None, None, 10)
            assert candles == []
    
    def test_start_backtest(self):
        """Test starting a backtest job."""
        client = EngineClient()
        req = BacktestRequest(
            strategy="mean_revert_v1",
            symbol="AAPL",
            tf="1d",
            start="2024-01-01",
            end="2024-01-31",
            initial_cash=10000.0
        )
        
        job_id = client.start_backtest(req)
        
        assert isinstance(job_id, str)
        assert len(job_id) > 0
        assert job_id in client._backtests
        assert client._backtests[job_id]["state"] == "queued"
        assert client._backtests[job_id]["progress"] == 0.0
    
    def test_get_backtest_status(self):
        """Test retrieving backtest status."""
        client = EngineClient()
        req = BacktestRequest(
            strategy="mean_revert_v1",
            symbol="AAPL",
            tf="1d",
            start="2024-01-01",
            end="2024-01-31"
        )
        job_id = client.start_backtest(req)
        
        status = client.get_backtest_status(job_id)
        
        assert status["job_id"] == job_id
        assert status["state"] == "queued"
        assert status["progress"] == 0.0
    
    def test_get_backtest_status_invalid_job_id(self):
        """Test retrieving status for non-existent job."""
        client = EngineClient()
        status = client.get_backtest_status("invalid_job_id")
        
        assert status["state"] == "failed"
        assert status["error"] == "unknown job_id"
    
    @patch("app.services.engine_client.run_backtest")
    @patch("app.services.engine_client.file_exists")
    @patch("app.services.engine_client.load_parquet")
    def test_simulate_backtest_run_success(
        self, mock_load, mock_exists, mock_run_backtest, sample_ohlcv_data_with_features, mock_emit
    ):
        """Test successful backtest run."""
        mock_exists.return_value = True
        mock_load.return_value = sample_ohlcv_data_with_features.reset_index()
        
        # Mock backtest result
        from engine.backtest.backtest import BacktestResult
        mock_result = BacktestResult(
            equity_curve=pd.Series([10000, 10100, 10200]),
            trades=[],
            stats={
                "final_equity": 10200.0,
                "total_return": 0.02,
                "max_drawdown": -0.01,
                "num_trades": 5.0,
                "win_rate": 0.6,
                "avg_trade_pnl": 10.0,
            }
        )
        mock_run_backtest.return_value = mock_result
        
        client = EngineClient()
        req = BacktestRequest(
            strategy="mean_revert_v1",
            symbol="AAPL",
            tf="1d",
            start="2024-01-01",
            end="2024-01-31"
        )
        job_id = client.start_backtest(req)
        
        client.simulate_backtest_run(job_id, mock_emit)
        
        # Verify backtest completed
        assert client._backtests[job_id]["state"] == "done"
        assert client._backtests[job_id]["progress"] == 1.0
        assert client._backtests[job_id]["result"] is not None
        
        # Verify events were emitted
        assert mock_emit.call_count > 0
    
    def test_simulate_backtest_run_invalid_job_id(self, mock_emit):
        """Test backtest run with invalid job ID."""
        client = EngineClient()
        client.simulate_backtest_run("invalid_id", mock_emit)
        
        # Should not raise error, just return early
        assert mock_emit.call_count == 0
    
    @patch("app.services.engine_client.file_exists")
    def test_simulate_backtest_run_no_data(self, mock_exists, mock_emit):
        """Test backtest run when no data is available."""
        mock_exists.return_value = False
        
        client = EngineClient()
        req = BacktestRequest(
            strategy="mean_revert_v1",
            symbol="INVALID",
            tf="1d",
            start="2024-01-01",
            end="2024-01-31"
        )
        job_id = client.start_backtest(req)
        
        client.simulate_backtest_run(job_id, mock_emit)
        
        assert client._backtests[job_id]["state"] == "failed"
        assert "No data found" in client._backtests[job_id]["error"]
    
    def test_start_trading(self, mock_emit):
        """Test starting trading mode."""
        client = EngineClient()
        client.start_trading(mock_emit)
        
        assert client._mode == "paper"
        assert client._event_sink == mock_emit
        mock_emit.assert_called_once()
    
    def test_stop_trading(self, mock_emit):
        """Test stopping trading."""
        client = EngineClient()
        client.start_trading(mock_emit)
        client.stop_trading()
        
        assert client._mode == "stopped"
        assert client._event_sink is None
    
    def test_place_order(self, mock_emit):
        """Test placing an order."""
        client = EngineClient()
        client.start_trading(mock_emit)
        
        req = PlaceOrderRequest(
            symbol="AAPL",
            side="buy",
            qty=10.0,
            order_type="market"
        )
        
        order = client.place_order(req)
        
        assert order["symbol"] == "AAPL"
        assert order["side"] == "buy"
        assert order["qty"] == 10.0
        assert order["status"] == "filled"
        assert "id" in order
        assert len(client._orders) == 1
        
        # Verify position was updated
        assert "AAPL" in client._positions
        assert client._positions["AAPL"]["qty"] == 10.0
    
    def test_get_orders(self):
        """Test retrieving orders."""
        client = EngineClient()
        req = PlaceOrderRequest(symbol="AAPL", side="buy", qty=10.0)
        client.place_order(req)
        
        orders = client.get_orders()
        assert len(orders) == 1
        assert orders[0]["symbol"] == "AAPL"
    
    def test_get_positions(self):
        """Test retrieving positions."""
        client = EngineClient()
        req = PlaceOrderRequest(symbol="AAPL", side="buy", qty=10.0)
        client.place_order(req)
        
        positions = client.get_positions()
        assert len(positions) == 1
        assert positions[0]["symbol"] == "AAPL"
        assert positions[0]["qty"] == 10.0
