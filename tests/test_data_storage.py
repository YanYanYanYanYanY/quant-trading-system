"""
Unit tests for data storage module.
"""
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from engine.data.storage import save_csv, save_parquet, load_csv, load_parquet, file_exists


class TestDataStorage:
    """Test suite for data storage functions."""
    
    def test_save_and_load_csv(self, tmp_path):
        """Test saving and loading CSV files."""
        # Create test data
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            "open": [100.0] * 5,
            "close": [101.0] * 5,
        })
        
        # Save
        rel_path = "test/test_data.csv"
        with patch("engine.data.storage.PROJECT_ROOT", tmp_path):
            save_csv(df, rel_path)
            
            # Verify file exists
            full_path = tmp_path / "data" / rel_path
            assert full_path.exists()
            
            # Load
            loaded_df = load_csv(rel_path)
            assert len(loaded_df) == 5
            assert "open" in loaded_df.columns
            assert "close" in loaded_df.columns
    
    def test_save_and_load_parquet(self, tmp_path):
        """Test saving and loading Parquet files."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            "open": [100.0] * 5,
            "close": [101.0] * 5,
        })
        
        rel_path = "test/test_data.parquet"
        with patch("engine.data.storage.PROJECT_ROOT", tmp_path):
            save_parquet(df, rel_path)
            
            full_path = tmp_path / "data" / rel_path
            assert full_path.exists()
            
            loaded_df = load_parquet(rel_path)
            assert len(loaded_df) == 5
            assert "open" in loaded_df.columns
    
    def test_file_exists(self, tmp_path):
        """Test file_exists function."""
        with patch("engine.data.storage.PROJECT_ROOT", tmp_path):
            # Create a test file
            test_file = tmp_path / "data" / "test" / "exists.csv"
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_text("test")
            
            assert file_exists("test/exists.csv") is True
            assert file_exists("test/nonexistent.csv") is False
    
    def test_save_creates_directory(self, tmp_path):
        """Test that save functions create directories."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        rel_path = "new_dir/deep/nested/file.csv"
        
        with patch("engine.data.storage.PROJECT_ROOT", tmp_path):
            save_csv(df, rel_path)
            
            full_path = tmp_path / "data" / rel_path
            assert full_path.exists()
            assert full_path.parent.exists()
