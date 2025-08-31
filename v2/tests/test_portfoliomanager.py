# tests/test_portfolio_manager.py

import pytest
import pandas as pd
from portfolio.PortfolioManager import PortfolioManager


@pytest.fixture
def manager():
    return PortfolioManager(capital=10000, max_position_pct=0.5)


@pytest.fixture
def sample_prices():
    return pd.Series({
        "AAPL": 200,
        "MSFT": 250,
        "GOOG": 1000
    })


def test_update_positions_basic(manager, sample_prices):
    weights = pd.Series({
        "AAPL": 0.5,
        "MSFT": -0.5,
        "GOOG": 0.0
    })

    positions = manager.update_positions(weights, sample_prices)

    assert positions["AAPL"] == pytest.approx((0.5 * 10000) / 200)
    assert positions["MSFT"] == pytest.approx((-0.5 * 10000) / 250)
    assert positions["GOOG"] == 0.0


def test_weights_exceed_max(manager, sample_prices):
    weights = pd.Series({
        "AAPL": 1.0,   # exceeds max
        "MSFT": -2.0,  # exceeds negative max
        "GOOG": 0.0
    })

    positions = manager.update_positions(weights, sample_prices)

    assert positions["AAPL"] == pytest.approx((0.5 * 10000) / 200)
    assert positions["MSFT"] == pytest.approx((-0.5 * 10000) / 250)


def test_all_zero_weights(manager, sample_prices):
    weights = pd.Series({
        "AAPL": 0.0,
        "MSFT": 0.0,
        "GOOG": 0.0
    })

    positions = manager.update_positions(weights, sample_prices)
    assert positions.sum() == 0
    assert all(positions == 0)
    assert all(manager.current_positions == 0)


def test_position_is_persisted(manager, sample_prices):
    weights = pd.Series({
        "AAPL": 0.25,
        "MSFT": 0.0,
        "GOOG": 0.25
    })
    manager.update_positions(weights, sample_prices)

    assert isinstance(manager.current_positions, pd.Series)
    assert manager.current_positions["AAPL"] > 0
    assert manager.current_positions["MSFT"] == 0
