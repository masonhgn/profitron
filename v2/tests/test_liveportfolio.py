# tests/test_live_portfolio.py

import pytest
import pandas as pd
from portfolio.LivePortfolio import LivePortfolio


@pytest.fixture
def portfolio():
    return LivePortfolio(starting_cash=10000.0)


@pytest.fixture
def mock_broker_data():
    return {
        "BTC-USDT": {"units": 0.1, "avg_cost": 30000},
        "ETH-USDT": {"units": 1.0, "avg_cost": 2000}
    }


@pytest.fixture
def mock_prices():
    return {
        "BTC-USDT": 35000,
        "ETH-USDT": 2500
    }


def test_sync_from_api(portfolio, mock_broker_data):
    portfolio.sync_from_api(mock_broker_data, cash=5000.0)

    assert portfolio.cash == 5000.0
    assert portfolio.positions["BTC-USDT"] == 0.1
    assert portfolio.avg_cost["ETH-USDT"] == 2000


def test_get_exposure(portfolio, mock_broker_data):
    portfolio.sync_from_api(mock_broker_data, cash=0)
    btc_exp = portfolio.get_exposure("BTC-USDT", current_price=34000)
    eth_exp = portfolio.get_exposure("ETH-USDT", current_price=2200)

    assert btc_exp == 0.1 * 34000
    assert eth_exp == 1.0 * 2200


def test_get_total_equity(portfolio, mock_broker_data, mock_prices):
    portfolio.sync_from_api(mock_broker_data, cash=5000.0)
    total_equity = portfolio.get_total_equity(mock_prices)

    expected_equity = 5000 + (0.1 * 35000) + (1.0 * 2500)
    assert total_equity == pytest.approx(expected_equity)


def test_summary_dataframe(portfolio, mock_broker_data, mock_prices):
    portfolio.sync_from_api(mock_broker_data, cash=5000.0)
    summary = portfolio.summary(mock_prices)

    assert isinstance(summary, pd.DataFrame)
    assert set(summary.columns) >= {"symbol", "units", "avg_cost", "price", "unrealized_pnl", "cash", "total_equity"}
    
    btc_row = summary[summary["symbol"] == "BTC-USDT"].iloc[0]
    assert btc_row["unrealized_pnl"] == pytest.approx((35000 - 30000) * 0.1)

    # Ensure summary has correct equity value repeated
    assert all(summary["total_equity"] == portfolio.get_total_equity(mock_prices))
