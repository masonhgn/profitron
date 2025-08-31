# src/portfolio/MetaPortfolio.py

from typing import Dict, List
from ..brokers.BrokerContext import BrokerContext
from ..signals.Signal import Signal


class MetaPortfolio:
    def __init__(self):
        self.brokers: Dict[str, BrokerContext] = {}

    def register_broker(self, name: str, ctx: BrokerContext):
        self.brokers[name] = ctx
        print(f"[MetaPortfolio] registered broker: {name}")

    def sync_all(self):
        """
        sync all portfolios together
        """
        for name, ctx in self.brokers.items():
            ctx.sync() #dispatch sync 

    def place_order(self, signal: Signal):
        """dispatches the signal to the appropriate broker context based on asset_type or routing logic"""
        broker = self._select_broker(signal)

        if broker:
            broker.place_order(signal)
        else:
            print(f"[MetaPortfolio] no broker found for signal: {signal.symbol} ({signal.asset_type})")

    def _select_broker(self, signal: Signal) -> BrokerContext:
        """selects a broker for a specific asset class. Like if we're trading crypto we use a crypto exchange"""
        # Basic default: asset_type → broker name match
        mapping = {
            "crypto": "binanceus",
            "equity": "alpaca",
            "option": "tradier"
        }
        broker_name = mapping.get(signal.asset_type)
        return self.brokers.get(broker_name)

    def get_total_equity(self, price_lookup: Dict[str, float]) -> float:
        """gets all equity for specific   """
        return sum(ctx.portfolio.get_total_equity(price_lookup) for ctx in self.brokers.values())

    def get_exposure(self, symbol: str, price: float) -> float:
        """what's my exposure to xyz"""
        return sum(ctx.portfolio.get_exposure(symbol, price) for ctx in self.brokers.values())

    def summary(self, price_lookup: Dict[str, float]) -> Dict[str, float]:
        """summary stats for all positions across all brokers"""
        equity = self.get_total_equity(price_lookup)
        exposures = {symbol: self.get_exposure(symbol, price_lookup[symbol]) for symbol in price_lookup}
        return {
            "total_equity": equity,
            "exposures": exposures
        }
