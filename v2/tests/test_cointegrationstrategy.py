import pytest
import pandas as pd
import numpy as np
from strategy.mean_reversion.CointegrationStrategy import CointegrationStrategy
from signals.Signal import Signal


@pytest.fixture
def sample_data():
    dates = pd.date_range("2023-01-01", periods=110)
    prices = pd.DataFrame({
        "A": np.linspace(100, 110, 110) + np.random.normal(0, 0.5, 110),
        "B": np.linspace(50, 55, 110) + np.random.normal(0, 0.5, 110)
    }, index=dates)
    return prices


@pytest.fixture
def strategy():
    return CointegrationStrategy(
        asset_1="A",
        asset_2="B",
        lookback=100,
        entry_z=1.0,
        exit_z=0.5,
        hedge_ratio_method="ols"
    )


def test_no_signal_if_not_enough_data(strategy, sample_data):
    small_data = sample_data.iloc[:50]
    signals = strategy.generate_signals(small_data)
    assert signals == []


def test_signal_generation_positive_z(strategy, sample_data):
    # artificially stretch spread to induce high z-score
    sample_data.loc[sample_data.index[-1], "A"] += 10

    signals = strategy.generate_signals(sample_data)

    assert len(signals) == 2
    assert any(s.symbol == "A" and s.action == "sell" for s in signals)
    assert any(s.symbol == "B" and s.action == "buy" for s in signals)


def test_signal_generation_negative_z(strategy, sample_data):
    # shrink spread to induce low z-score
    sample_data.loc[sample_data.index[-1], "A"] -= 10

    signals = strategy.generate_signals(sample_data)

    assert len(signals) == 2
    assert any(s.symbol == "A" and s.action == "buy" for s in signals)
    assert any(s.symbol == "B" and s.action == "sell" for s in signals)


def test_exit_signal(strategy, sample_data):
    # manipulate data to make zscore near 0
    idx = sample_data.index[-1]
    sample_data.loc[idx, "A"] = sample_data["A"].mean()
    sample_data.loc[idx, "B"] = sample_data["B"].mean()

    signals = strategy.generate_signals(sample_data)

    assert len(signals) == 2
    assert all(s.action == "sell" and s.target_pct == 0.0 for s in signals)


def test_signal_objects_have_required_fields(strategy, sample_data):
    sample_data.loc[sample_data.index[-1], "A"] += 10  # force signal
    signals = strategy.generate_signals(sample_data)

    for s in signals:
        assert isinstance(s, Signal)
        assert s.symbol in {"A", "B"}
        assert s.action in {"buy", "sell"}
        assert s.target_pct in {0.0, 0.5}
        assert s.asset_type == "equity"
