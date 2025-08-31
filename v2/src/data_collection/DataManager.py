# src/data/DataManager.py

from typing import Dict, Any, List, Optional
import pandas as pd
from .DataSource import EquityDataSource, CryptoDataSource
from .sources.AlpacaEquitySource import AlpacaEquitySource
from .sources.BinanceUSCryptoSource import BinanceUSCryptoSource


class DataManager:
    """Clean data manager that routes requests to appropriate asset-specific sources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.equity_source: Optional[EquityDataSource] = None
        self.crypto_source: Optional[CryptoDataSource] = None
        
        self._initialize_sources()

    def _initialize_sources(self):
        """Initialize data sources based on configuration"""
        data_sources = self.config.get("data_sources", {})
        
        # Initialize equity source
        equity_config = data_sources.get("equity", {})
        provider = equity_config.get("provider")
        
        if provider == "alpaca":
            api_key = equity_config.get("api_key")
            api_secret = equity_config.get("api_secret")
            if api_key and api_secret and api_key != "${ALPACA_API_KEY}":
                try:
                    self.equity_source = AlpacaEquitySource(api_key, api_secret)
                    print("[DataManager] Initialized Alpaca equity source")
                except Exception as e:
                    print(f"[DataManager] Error initializing Alpaca: {e}")
                    raise Exception(f"Failed to initialize Alpaca equity source: {e}")
            else:
                raise Exception("Alpaca credentials not provided or not resolved from environment variables")
        
        # Initialize crypto source
        crypto_config = data_sources.get("crypto", {})
        if crypto_config.get("provider") == "binanceus":
            api_key = crypto_config.get("api_key")
            api_secret = crypto_config.get("api_secret")
            if api_key and api_secret and api_key != "${BINANCEUS_API_KEY}":
                try:
                    self.crypto_source = BinanceUSCryptoSource(api_key, api_secret)
                    print("[DataManager] Initialized BinanceUS crypto source")
                except Exception as e:
                    print(f"[DataManager] Error initializing BinanceUS: {e}")
                    raise Exception(f"Failed to initialize BinanceUS crypto source: {e}")
            else:
                print("[DataManager] BinanceUS credentials not provided, crypto source not initialized")

    def get_price_data(
        self,
        assets: List[Dict[str, str]],
        start_date: str,
        end_date: str,
        frequency: str = "1d",
        fields: List[str] = ["close"]
    ) -> pd.DataFrame:
        """
        Get price data for multiple assets, routing to appropriate sources.
        
        Args:
            assets: List of dicts with 'symbol' and 'type' keys
            start_date: Start date string
            end_date: End date string  
            frequency: Data frequency (e.g., "1d", "1h")
            fields: List of fields to retrieve (e.g., ["close", "volume"])
            
        Returns:
            DataFrame with price data for all assets
        """
        # Group assets by type
        equity_symbols = [asset["symbol"] for asset in assets if asset["type"] == "equity"]
        crypto_symbols = [asset["symbol"] for asset in assets if asset["type"] == "crypto"]
        
        dfs = []
        
        # Get equity data
        if equity_symbols and self.equity_source:
            try:
                equity_df = self.equity_source.get_historical_ohlcv(
                    symbols=equity_symbols,
                    start_date=start_date,
                    end_date=end_date,
                    frequency=frequency
                )
                # Flatten MultiIndex columns if present, but avoid duplication
                if isinstance(equity_df.columns, pd.MultiIndex):
                    # Check if the second level already contains the symbol prefix
                    new_columns = []
                    for col in equity_df.columns:
                        if col[1].startswith(f"{col[0]}_"):
                            # Already has symbol prefix, just use the second level
                            new_columns.append(col[1])
                        else:
                            # Need to add symbol prefix
                            new_columns.append(f"{col[0]}_{col[1]}")
                    equity_df.columns = new_columns
                dfs.append(equity_df)
            except Exception as e:
                print(f"[DataManager] Error getting equity data: {e}")
        
        # Get crypto data
        if crypto_symbols and self.crypto_source:
            try:
                crypto_df = self.crypto_source.get_historical_ohlcv(
                    symbols=crypto_symbols,
                    start_date=start_date,
                    end_date=end_date,
                    frequency=frequency
                )
                # Flatten MultiIndex columns if present, but avoid duplication
                if isinstance(crypto_df.columns, pd.MultiIndex):
                    # Check if the second level already contains the symbol prefix
                    new_columns = []
                    for col in crypto_df.columns:
                        if col[1].startswith(f"{col[0]}_"):
                            # Already has symbol prefix, just use the second level
                            new_columns.append(col[1])
                        else:
                            # Need to add symbol prefix
                            new_columns.append(f"{col[0]}_{col[1]}")
                    crypto_df.columns = new_columns
                dfs.append(crypto_df)
            except Exception as e:
                print(f"[DataManager] Error getting crypto data: {e}")
        
        # Combine dataframes
        if dfs:
            return pd.concat(dfs, axis=1)
        else:
            print("[DataManager] Warning: No data sources available for the requested assets")
            return pd.DataFrame()

    def get_live_price(self, assets: List[Dict[str, str]]) -> Dict[str, float]:
        """
        Get live prices for assets.
        
        Args:
            assets: List of dicts with 'symbol' and 'type' keys
            
        Returns:
            Dict mapping symbol to price
        """
        prices = {}
        
        # Group assets by type
        equity_symbols = [asset["symbol"] for asset in assets if asset["type"] == "equity"]
        crypto_symbols = [asset["symbol"] for asset in assets if asset["type"] == "crypto"]
        
        # Get equity prices
        if equity_symbols and self.equity_source:
            try:
                equity_prices = self.equity_source.get_live_price(equity_symbols)
                prices.update(equity_prices)
            except Exception as e:
                print(f"[DataManager] Error getting equity live prices: {e}")
        
        # Get crypto prices
        if crypto_symbols and self.crypto_source:
            try:
                crypto_prices = self.crypto_source.get_live_price(crypto_symbols)
                prices.update(crypto_prices)
            except Exception as e:
                print(f"[DataManager] Error getting crypto live prices: {e}")
        
        return prices

    def get_available_frequencies(self, asset_type: str) -> List[str]:
        """Get available frequencies for a given asset type"""
        if asset_type == "equity" and self.equity_source:
            return self.equity_source.get_available_frequencies()
        elif asset_type == "crypto" and self.crypto_source:
            return self.crypto_source.get_available_frequencies()
        else:
            return []






