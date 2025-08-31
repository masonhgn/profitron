from typing import List
import pandas as pd
from strategy.TradingStrategy import TradingStrategy
from signals.Signal import Signal
from strategy.equities.mean_reversion.CointegrationStrategy import CointegrationStrategy

class PairsTradingStrategyCollection(TradingStrategy):
    def __init__(self, pairs_file: str, strategy_params: dict):
        self.strategies: List[CointegrationStrategy] = []
        self._load_pairs(pairs_file, strategy_params)

    def _load_pairs(self, filepath: str, strategy_params: dict):
        df = pd.read_csv(filepath)
        for _, row in df.iterrows():
            strat = CointegrationStrategy(
                asset_1={"symbol": row["asset_1"], "type": row["type_1"]},
                asset_2={"symbol": row["asset_2"], "type": row["type_2"]},
                **strategy_params
            )
            self.strategies.append(strat)

    def get_assets(self):
        assets = []
        for strat in self.strategies:
            assets.extend(strat.get_assets())
        return assets

    def get_frequency(self):
        return self.strategies[0].get_frequency()  # assume same for all

    def get_fields(self):
        return self.strategies[0].get_fields()

    def get_poll_interval(self):
        return self.strategies[0].get_poll_interval()

    def get_lookback_window(self):
        return self.strategies[0].get_lookback_window()

    def on_event(self, data: pd.DataFrame) -> List[Signal]:
        all_signals = []
        for strat in self.strategies:
            all_signals.extend(strat.on_event(data))
        return all_signals
