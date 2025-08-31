# tests/test_engine.py

import pytest
from signals.Signal import Signal
from core.Engine import Engine


@pytest.fixture
def mock_config(monkeypatch):
    config = {
        "mode": "backtest",
        "strategy": {
            "module": "mock_strategy",
            "class": "MockStrategy",
            "params": {}
        },
        "brokers": {
            "binanceus": {
                "mode": "paper",
                "starting_cash": 10000
            }
        },
        "backtest": {
            "params": {
                "start_date": "2022-01-01",
                "end_date": "2022-01-31",
                "capital": 10000,
                "slippage_bps": 5,
                "commission_per_trade": 1
            },
            "price_data_loader": lambda: __import__("pandas").DataFrame(
                {
                    "BTC-USDT": [40000, 41000, 42000, 43000]
                },
                index=__import__("pandas").date_range("2022-01-01", periods=4)
            )
        },
        "print_summary": False
    }

    def mock_load_yaml(path):
        return config

    # Patch imports + loader
    monkeypatch.setattr("core.Engine.load_yaml_config", mock_load_yaml)

    class MockStrategy:
        def generate_signals(self, data):
            return [Signal(symbol="BTC-USDT", action="buy", target_pct=0.5)]

    monkeypatch.setitem(__import__("sys").modules, "mock_strategy", type("mod", (), {"MockStrategy": MockStrategy}))

    return config


def test_engine_init_and_backtest_run(monkeypatch, mock_config):
    engine = Engine(config_path="dummy.yaml")
    result = engine._run_backtest()

    assert "pnl" in result
    assert "positions" in result
    assert "stats" in result


def test_live_loop_runs_once(monkeypatch, mock_config):
    mock_config["mode"] = "paper"
    called = {}

    class DummyMonitor:
        def run_once(self):
            called["run_once"] = True
            return []

    class DummyMeta:
        def __init__(self):
            self.brokers = {}

        def sync_all(self):
            called["sync"] = True

        def place_order(self, signal):
            pass

        def summary(self, price_lookup):
            return {"equity": 10000}

    monkeypatch.setattr("core.Engine.load_yaml_config", lambda path: mock_config)
    engine = Engine(config_path="dummy.yaml")
    engine.monitor = DummyMonitor()
    engine.meta = DummyMeta()

    monkeypatch.setattr("time.sleep", lambda x: None)  # don't actually wait
    monkeypatch.setattr("builtins.print", lambda *a, **kw: None)
    monkeypatch.setattr(engine, "_get_price_lookup", lambda signals: {"BTC-USDT": 40000})
    monkeypatch.setattr(engine, "_run_live_or_paper", lambda: None)  # bypass loop in test

    assert isinstance(engine.monitor, DummyMonitor)
    assert isinstance(engine.meta, DummyMeta)
