from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import asyncio
import websockets
import json



class EquityDataSource(ABC):
    """Base class for equity data sources (stocks, ETFs, etc.)"""
    
    @abstractmethod
    def get_historical_ohlcv(
        self, 
        symbols: List[str], 
        start_date: str, 
        end_date: str, 
        frequency: str = "1d"
    ) -> pd.DataFrame:
        """Get historical OHLCV data for equity symbols"""
        pass
    
    @abstractmethod
    def get_live_price(self, symbols: List[str]) -> Dict[str, float]:
        """Get live prices for equity symbols"""
        pass
    
    def get_available_frequencies(self) -> List[str]:
        """Return available data frequencies for this source"""
        return ["1m", "5m", "15m", "1h", "1d"]


class CryptoDataSource(ABC):
    """Base class for cryptocurrency data sources"""
    
    @abstractmethod
    def get_historical_ohlcv(
        self, 
        symbols: List[str], 
        start_date: str, 
        end_date: str, 
        frequency: str = "1d"
    ) -> pd.DataFrame:
        """Get historical OHLCV data for crypto symbols"""
        pass
    
    @abstractmethod
    def get_live_price(self, symbols: List[str]) -> Dict[str, float]:
        """Get live prices for crypto symbols"""
        pass
    
    def get_available_frequencies(self) -> List[str]:
        """Return available data frequencies for this source"""
        return ["1m", "5m", "15m", "1h", "4h", "1d"]


# Legacy base class for backward compatibility (can be removed later)
class DataSource(ABC):
    """Legacy base class - use EquityDataSource or CryptoDataSource instead"""
    
    def __init__(self, frequencies: List[str], asset_types: List[str]) -> None:
        self.frequencies = frequencies
        self.asset_types = asset_types

    @abstractmethod
    def get_live_price(self, symbols: List[str]) -> float:
        """Legacy method - use asset-specific sources instead"""
        pass

    def get_historical_equity_ohlcv(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        frequency: str = "1d" 
    ) -> pd.DataFrame:
        """Legacy method - use EquityDataSource instead"""
        raise NotImplementedError("Use EquityDataSource instead")

    def get_historical_crypto_ohlcv(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        frequency: str = "1d"             
    ) -> pd.DataFrame:
        """Legacy method - use CryptoDataSource instead"""
        raise NotImplementedError("Use CryptoDataSource instead")

    def get_available_frequencies(self, symbol: str) -> List[str]:
        """Legacy method"""
        return self.frequencies
    
    def get_available_asset_types(self) -> List[str]:
        """Legacy method"""
        return self.asset_types