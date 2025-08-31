from typing import List, Dict, Any
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import Adjustment
from alpaca.data.timeframe import TimeFrame
import requests
from ..DataSource import EquityDataSource


class AlpacaEquitySource(EquityDataSource):
    """Alpaca data source for US equity data"""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.client = StockHistoricalDataClient(api_key, api_secret)
        
        self.freq_map = {
            "1m": TimeFrame.Minute,
            "5m": TimeFrame(5, TimeFrame.Minute),
            "15m": TimeFrame(15, TimeFrame.Minute),
            "1h": TimeFrame.Hour,
            "1d": TimeFrame.Day
        }

    def get_historical_ohlcv(
        self, 
        symbols: List[str], 
        start_date: str, 
        end_date: str, 
        frequency: str = "1d"
    ) -> pd.DataFrame:
        """Get historical OHLCV data for equity symbols"""
        timeframe = self.freq_map.get(frequency)
        if not timeframe:
            raise ValueError(f"Unsupported frequency: {frequency}")

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            start=pd.to_datetime(start_date),
            end=pd.to_datetime(end_date),
            timeframe=timeframe,
            adjustment=Adjustment.ALL
        )

        barset = self.client.get_stock_bars(request).df
        
        # Handle MultiIndex if present
        if isinstance(barset.index, pd.MultiIndex):
            barset = barset.reset_index()

        # Clean up timestamp
        barset["timestamp"] = pd.to_datetime(barset["timestamp"]).dt.tz_localize(None)
        barset.set_index("timestamp", inplace=True)

        # Format columns with symbol prefixes
        result = pd.concat({
            sym: group[["open", "high", "low", "close", "volume"]].add_prefix(f"{sym}_")
            for sym, group in barset.groupby("symbol")
        }, axis=1)
        
        return result

    def get_live_price(self, symbols: List[str]) -> Dict[str, float]:
        """Get live prices for equity symbols"""
        prices = {}
        
        for symbol in symbols:
            try:
                # Use Alpaca's latest quote endpoint
                base_url = "https://data.alpaca.markets/v2/stocks"
                headers = {
                    "APCA-API-KEY-ID": self.api_key,
                    "APCA-API-SECRET-KEY": self.api_secret
                }
                url = f"{base_url}/{symbol}/quotes/latest"
                resp = requests.get(url, headers=headers)
                
                if resp.status_code == 200:
                    data = resp.json()
                    prices[symbol] = data["quote"]["ask_price"]
                else:
                    print(f"Warning: Failed to get live price for {symbol}: {resp.text}")
                    prices[symbol] = 0.0
                    
            except Exception as e:
                print(f"Error getting live price for {symbol}: {e}")
                prices[symbol] = 0.0
        
        return prices

    def get_available_frequencies(self) -> List[str]:
        """Return available data frequencies"""
        return list(self.freq_map.keys()) 