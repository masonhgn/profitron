# src/brokers/BrokerContext.py

from typing import Dict, Any
from ..portfolio.LivePortfolio import LivePortfolio
from ..signals.Signal import Signal
from ..execution.OrderManager import OrderManager
from alpaca.trading.client import TradingClient
import os
from dotenv import load_dotenv

load_dotenv()

class BrokerContext:
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.mode = config.get("mode", "paper")
        self.api = self._init_api()
        # For live trading, we'll get actual balance from broker
        # For paper trading, use starting_cash if provided
        starting_cash = config.get("starting_cash", 100_000) if self.mode == "paper" else 0.0
        self.portfolio = LivePortfolio(starting_cash=starting_cash)
        self.order_manager = OrderManager(self.api, timeout=config.get("order_timeout", 30)) if self.api else None

    def _init_api(self):
        if self.name == "alpaca":
            if self.mode == "paper":
                api_key = os.environ.get("ALPACA_PAPER_API_KEY")
                api_secret = os.environ.get("ALPACA_PAPER_API_SECRET")
            else:
                api_key = os.environ.get("ALPACA_API_KEY")
                api_secret = os.environ.get("ALPACA_API_SECRET")
            return TradingClient(api_key, api_secret, paper=(self.mode == "paper"))
        return None  # Only support Alpaca for now

    def sync(self):
        # No-op for Alpaca for now
        return

    def place_order(self, signal: Signal):
        """executes or queues the order based on signal metadata"""
        if self.order_manager:
            self.order_manager.submit_order(signal)
        else:
            print(f"[BrokerContext] No order manager available for broker: {self.name}")
