# Test Suite

This directory contains unit tests for the quant trading project.

## Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures and configuration
├── test_engine_client.py    # Tests for EngineClient
├── test_backtest.py         # Tests for backtest module
├── test_strategy.py         # Tests for strategy module
├── test_data_storage.py     # Tests for data storage
└── test_api_routes.py       # Tests for API routes
```

## Running Tests

### Install dependencies

```bash
pip install pytest pytest-mock pytest-asyncio
```

### Run all tests

```bash
pytest
```

### Run specific test file

```bash
pytest tests/test_engine_client.py
```

### Run specific test

```bash
pytest tests/test_engine_client.py::TestEngineClient::test_list_symbols
```

### Run with coverage

```bash
pip install pytest-cov
pytest --cov=app --cov=engine --cov-report=html
```

### Run with verbose output

```bash
pytest -v
```

## Test Coverage

Current test coverage includes:

- ✅ EngineClient methods (list_symbols, get_candles, backtests, trading)
- ✅ Backtest execution and results
- ✅ Strategy signals and logic
- ✅ Data storage (save/load CSV/Parquet)
- ✅ API routes (health, data, backtests, strategies)

## Writing New Tests

1. Create test file: `tests/test_<module_name>.py`
2. Import pytest and your module
3. Use fixtures from `conftest.py` when available
4. Follow naming convention: `test_<functionality>`
5. Use descriptive test names

Example:

```python
def test_my_function():
    """Test description."""
    result = my_function(input)
    assert result == expected
```

## Mocking

Use `unittest.mock` or `pytest-mock` for mocking:

```python
from unittest.mock import patch, MagicMock

@patch("module.function")
def test_with_mock(mock_function):
    mock_function.return_value = "mocked"
    result = my_code()
    assert result == "mocked"
```
