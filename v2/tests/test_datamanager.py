# tests/test_datamanager.py

import pandas as pd
import pytest
from unittest.mock import MagicMock
from data.DataManager import DataManager


@pytest.fixture
def dummy_config(tmp_path):
    config = {
        "data_sources": {
            "local": {
                "type": "localfile",
                "path": str(tmp_path),
                "format": "parquet",
                "use_duckdb": False
            },
            "binance": {
                "type": "binanceus",
                "api_key": "fake",
                "api_secret": "fake"
            }
        }
    }
    return config


@pytest.fixture
def manager_with_mocks(monkeypatch, dummy_config):
    # Patch the source classes BEFORE DataManager instantiation
    from data import DataManager as dm_module
    dm_module.BinanceUSSource = MagicMock()
    dm_module.LocalFileSource = MagicMock()

    dm = dm_module.DataManager(config=dummy_config)
    return dm


def test_initialize_sources(manager_with_mocks):
    assert "local" in manager_with_mocks.sources
    assert "binance" in manager_with_mocks.sources


def test_get_price_data_delegation(manager_with_mocks):
    local_mock = manager_with_mocks.sources["local"]
    local_mock.get_price_data.return_value = pd.DataFrame({"BTC-USD_close": [1, 2, 3]})

    df = manager_with_mocks.get_price_data("local", ["BTC-USD"], "2022-01-01", "2022-01-03")

    assert isinstance(df, pd.DataFrame)
    assert "BTC-USD_close" in df.columns
    local_mock.get_price_data.assert_called_once()


def test_get_price_data_with_fallback(manager_with_mocks):
    local = manager_with_mocks.sources["local"]
    binance = manager_with_mocks.sources["binance"]

    local.get_price_data.side_effect = FileNotFoundError
    binance.get_price_data.return_value = pd.DataFrame({
        "BTC-USD_close": [1.0, 2.0],
        "datetime": pd.date_range("2024-01-01", periods=2, freq="D")
    }).set_index("datetime")

    manager_with_mocks.store_price_data = MagicMock()

    df = manager_with_mocks.get_price_data_with_fallback(
        primary="local",
        fallback="binance",
        symbols=["BTC-USD"],
        start_date="2024-01-01",
        end_date="2024-01-02"
    )

    assert isinstance(df, pd.DataFrame)
    assert "BTC-USD_close" in df.columns
    manager_with_mocks.store_price_data.assert_called_once()


def test_store_price_data_calls_underlying(manager_with_mocks):
    mock = manager_with_mocks.sources["local"]
    manager_with_mocks.store_price_data("local", "BTC-USD", pd.DataFrame(), "1d")

    mock.store_price_data.assert_called_once()


def test_get_live_price(manager_with_mocks):
    binance = manager_with_mocks.sources["binance"]
    binance.get_live_price.return_value = 42000.0

    price = manager_with_mocks.get_live_price("binance", "BTC-USD")
    assert price == 42000.0


def test_get_available_frequencies(manager_with_mocks):
    local = manager_with_mocks.sources["local"]
    local.get_available_frequencies.return_value = ["1m", "1d"]

    freqs = manager_with_mocks.get_available_frequencies("local", "BTC-USD")
    assert "1m" in freqs


def test_sync_data_from_to(manager_with_mocks):
    # Use fake multiindex DataFrame as if returned from get_price_data
    df = pd.concat({
        "BTC-USD": pd.DataFrame({
            "close": [1.0, 2.0],
            "datetime": pd.date_range("2024-01-01", periods=2)
        }).set_index("datetime")
    }, axis=1)

    manager_with_mocks.get_price_data = MagicMock(return_value=df)
    manager_with_mocks.store_price_data = MagicMock()

    manager_with_mocks.sync_data_from_to(
        from_source="binance",
        to_source="local",
        symbols=["BTC-USD"],
        start_date="2024-01-01",
        end_date="2024-01-02"
    )

    manager_with_mocks.store_price_data.assert_called_once()
