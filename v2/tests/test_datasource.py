# tests/test_datasource.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.DataSource import DataSource


class DummySource(DataSource):
    """
    Minimal subclass of DataSource that mocks get_price_data.
    Used for testing base methods like get_returns and resample_data.
    """
    def get_price_data(self, symbols, start_date, end_date, frequency="1d", fields=["close"]):
        index = pd.date_range(start="2022-01-01", periods=5, freq="D")
        data = pd.DataFrame(
            {
                ("BTC-USD", "close"): [100, 101, 102, 103, 104]
            },
            index=index
        )
        data.columns = pd.MultiIndex.from_tuples(data.columns)
        return data


def test_get_returns_linear():
    ds = DummySource()
    df = ds.get_returns("BTC-USD", "2022-01-01", "2022-01-05", frequency="1d", log=False)

    expected_len = 4  # one less than number of rows
    assert len(df) == expected_len
    assert isinstance(df, pd.Series)
    assert not df.isnull().any()


def test_get_returns_log():
    ds = DummySource()
    df = ds.get_returns("BTC-USD", "2022-01-01", "2022-01-05", frequency="1d", log=True)

    assert len(df) == 4
    assert isinstance(df, pd.Series)
    expected = np.log(101 / 100)
    actual = df.iloc[0]
    print(f"Expected: {expected}, Actual: {actual}")
    assert np.isclose(actual, expected, atol=1e-6)



def test_resample_data_ohlc():
    ds = DummySource()
    index = pd.date_range("2022-01-01", periods=6, freq="h")
    df = pd.DataFrame({"close": [1, 2, 3, 4, 5, 6]}, index=index)

    resampled = ds.resample_data(df, frequency="3h", method="ohlc")
    assert resampled.shape[0] == 2
    assert "close" in resampled.columns or "close" in resampled.columns.get_level_values(0)


def test_resample_data_mean():
    ds = DummySource()
    index = pd.date_range("2022-01-01", periods=6, freq="h")
    df = pd.DataFrame({"close": [1, 2, 3, 4, 5, 6]}, index=index)

    resampled = ds.resample_data(df, frequency="3h", method="mean")
    assert resampled.shape[0] == 2
    assert np.isclose(resampled.iloc[0]["close"], 2.0)


def test_get_available_frequencies_raises():
    ds = DummySource()
    try:
        ds.get_available_frequencies("BTC-USD")
    except NotImplementedError:
        assert True
    else:
        assert False, "Expected NotImplementedError"
