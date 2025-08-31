# src/monitor/SignalMonitor.py

import time
from typing import Dict, Any, List
import pandas as pd

from strategies.TradingStrategy import TradingStrategy
from ..data_collection.DataManager import DataManager
from .Signal import Signal


class SignalMonitor:
    def __init__(self, strategy: TradingStrategy, data_manager: DataManager, config: Dict[str, Any]):
        self.strategy = strategy
        self.data_manager = data_manager
        self.config = config


        self.frequency = config.get("frequency", "1h")
        self.fields = config.get("fields", ["close"])
        self.lookback_window = config.get("lookback_window", 100)
        self.poll_interval = config.get("poll_interval", 60)  # seconds

    def run_once(self) -> List[Signal]:
        """
        fetches recent data and generates signal(s) once

        Returns:
            List[Signal]: Latest generated signals
        """
        timestamp = pd.Timestamp.now()
        print(f"[{timestamp}] [SignalMonitor] Scanning market for signals...")
        
        end_time = timestamp
        start_time = end_time - pd.Timedelta(f"{self.lookback_window}{self.frequency}")

        print(f"[{timestamp}] [SignalMonitor] Fetching data from {start_time} to {end_time}")
        print(f"[{timestamp}] [SignalMonitor] Assets: {[asset['symbol'] for asset in self.strategy.get_assets()]}")

        # Fetch latest window of price data
        df = self.data_manager.get_price_data(
            assets=self.strategy.get_assets(),
            start_date=start_time.strftime("%Y-%m-%d %H:%M:%S"),
            end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
            frequency=self.frequency,
            fields=self.fields
        )

        print(f"[{timestamp}] [SignalMonitor] Data fetched: {len(df)} observations")
        if not df.empty:
            print(f"[{timestamp}] [SignalMonitor] Latest prices: {df.iloc[-1].to_dict()}")

        # Generate and return latest signal objects
        signals = self.strategy.on_event(df)
        print(f"[{timestamp}] [SignalMonitor] Generated {len(signals)} signals")
        
        return signals

    def run_forever(self):
        """
        continuously poll data and generate signals.
        """
        print("[SignalMonitor] Starting polling loop...")
        while True:
            try:
                signals = self.run_once()
                timestamp = pd.Timestamp.now()
                print(f"[{timestamp}] latest signals:")
                for sig in signals:
                    print(f"  {sig}")
            except Exception as e:
                print(f"[SignalMonitor] error: {e}")

            time.sleep(self.poll_interval)
