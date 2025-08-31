import ccxt
import pandas as pd
from typing import List, Dict, Any
from ..DataSource import CryptoDataSource


class BinanceUSCryptoSource(CryptoDataSource):
    """Binance US data source for cryptocurrency data"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.client = ccxt.binanceus({
            'apiKey': api_key or "",
            'secret': api_secret or "",
            'enableRateLimit': True
        })
        self.max_limit = 1000  # Binance US allows 1000 candles per request
        
        self.timeframe_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d"
        }

    def get_historical_ohlcv(
        self, 
        symbols: List[str], 
        start_date: str, 
        end_date: str, 
        frequency: str = "1d"
    ) -> pd.DataFrame:
        """Get historical OHLCV data for crypto symbols"""
        df_list = []
        
        for symbol in symbols:
            market_symbol = symbol.replace("-", "/").upper()
            df = self._fetch_symbol_data(market_symbol, start_date, end_date, frequency)
            df_list.append(df)

        if df_list:
            return pd.concat(df_list, axis=1)
        else:
            return pd.DataFrame()

    def get_live_price(self, symbols: List[str]) -> Dict[str, float]:
        """Get live prices for crypto symbols"""
        prices = {}
        
        for symbol in symbols:
            try:
                market_symbol = symbol.replace("-", "/").upper()
                ticker = self.client.fetch_ticker(market_symbol)
                prices[symbol] = ticker["last"]
            except Exception as e:
                print(f"Error getting live price for {symbol}: {e}")
                prices[symbol] = 0.0
        
        return prices

    def _fetch_symbol_data(self, symbol: str, start: str, end: str, frequency: str) -> pd.DataFrame:
        """Fetch historical data for a single symbol"""
        since = self.client.parse8601(pd.to_datetime(start).isoformat())
        end_ts = pd.to_datetime(end)

        all_candles = []
        timeframe = self.timeframe_map.get(frequency)
        if not timeframe:
            raise ValueError(f"Unsupported frequency: {frequency}")

        while True:
            candles = self.client.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=self.max_limit)
            if not candles:
                break

            df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("datetime", inplace=True)
            df.drop("timestamp", axis=1, inplace=True)

            all_candles.append(df)
            since = candles[-1][0] + 1

            if pd.to_datetime(since, unit="ms") > end_ts:
                break

        if not all_candles:
            return pd.DataFrame()

        full_df = pd.concat(all_candles)
        full_df = full_df.loc[start:end]
        
        # Add symbol prefix to columns
        symbol_clean = symbol.replace("/", "-")
        full_df.columns = [f"{symbol_clean}_{col}" for col in full_df.columns]
        
        return full_df

    def get_available_frequencies(self) -> List[str]:
        """Return available data frequencies"""
        return list(self.timeframe_map.keys()) 