# src/strategy/TradingStrategy.py

from abc import ABC, abstractmethod
from typing import List, Dict
import pandas as pd
from src.signals.Signal import Signal


class TradingStrategy(ABC):
    """parent class for strats"""

    @abstractmethod
    def on_event(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate a list of Signal objects based on historical data.

        Args:
            data (pd.DataFrame): Time-indexed price data for relevant assets.

        Returns:
            List[Signal]: Structured trade signals (long, short, etc.)
        """
        pass
    
    def get_assets(self) -> List[Dict[str, str]]:
        """Get list of assets this strategy trades"""
        raise NotImplementedError("Subclasses must implement get_assets()")
    
    def get_frequency(self) -> str:
        """Get data frequency for this strategy"""
        raise NotImplementedError("Subclasses must implement get_frequency()")
    
    def get_lookback_window(self) -> int:
        """Get lookback window for this strategy"""
        raise NotImplementedError("Subclasses must implement get_lookback_window()")
    
    def get_fields(self) -> List[str]:
        """Get required data fields for this strategy"""
        raise NotImplementedError("Subclasses must implement get_fields()")
    
    def get_poll_interval(self) -> int:
        """Get poll interval for live trading"""
        raise NotImplementedError("Subclasses must implement get_poll_interval()")
