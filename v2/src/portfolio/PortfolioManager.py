# src/portfolio/PortfolioManager.py

from typing import Optional
import pandas as pd


class PortfolioManager:
    def __init__(self, capital: float, max_position_pct: float = 1.0):
        """
        Args:
            capital (float): total portfolio capital
            max_position_pct (float): max capital % per asset (1.0 = 100%)
        """
        self.capital = capital
        self.max_position_pct = max_position_pct
        self.current_positions = pd.Series(dtype=float)

    def update_positions(self, target_weights: pd.Series, prices: pd.Series) -> pd.Series:
        """
        convert target weights into position sizes (e.g. shares, units)

        Args:
            target_weights (pd.Series): target portfolio weights per asset (-1 to 1)
            prices (pd.Series): current prices per asset

        Returns:
            pd.Series: position sizes in units (same index as weights)
        """
        active = target_weights[target_weights != 0]
        if active.empty:
            self.current_positions = pd.Series(0, index=target_weights.index)
            return self.current_positions

        positions = pd.Series(0.0, index=target_weights.index)
        for asset in active.index:
            weight = target_weights[asset]
            capped_weight = max(-self.max_position_pct, min(weight, self.max_position_pct))
            allocation_dollars = capped_weight * self.capital
            price = prices.get(asset, 1.0)
            positions[asset] = allocation_dollars / price

        self.current_positions = positions
        return positions
