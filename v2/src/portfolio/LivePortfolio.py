import pandas as pd
from typing import Dict


class LivePortfolio:
    def __init__(self, starting_cash: float = 0.0):
        self.cash = starting_cash
        self.positions: Dict[str, float] = {}
        self.avg_cost: Dict[str, float] = {}
        self.realized_pnl: float = 0.0  # optionally pulled or inferred

    def sync_from_api(self, broker_positions: Dict[str, Dict[str, float]], cash: float):
        """
        sync portfolio with live brokerage api data

        Args:
            broker_positions: Dict[symbol, {"units": float, "avg_cost": float}]
            cash (float): total available cash from broker
        """
        self.positions = {symbol: data["units"] for symbol, data in broker_positions.items()}
        self.avg_cost = {symbol: data["avg_cost"] for symbol, data in broker_positions.items()}
        self.cash = cash

    def get_exposure(self, symbol: str, current_price: float) -> float:
        units = self.positions.get(symbol, 0.0)
        return units * current_price

    def get_total_equity(self, price_lookup: Dict[str, float]) -> float:
        equity = self.cash
        for symbol, units in self.positions.items():
            price = price_lookup.get(symbol, 0.0)
            equity += units * price
        return equity

    def summary(self, price_lookup: Dict[str, float]) -> pd.DataFrame:
        rows = []
        for symbol, units in self.positions.items():
            price = price_lookup.get(symbol, 0.0)
            cost = self.avg_cost.get(symbol, 0.0)
            unrealized = (price - cost) * units
            rows.append({
                "symbol": symbol,
                "units": units,
                "avg_cost": cost,
                "price": price,
                "unrealized_pnl": unrealized
            })

        df = pd.DataFrame(rows)
        df["cash"] = self.cash
        df["total_equity"] = self.get_total_equity(price_lookup)
        return df
