# src/execution/Executor.py

from typing import Dict, Any, List
from ..signals.SignalMonitor import SignalMonitor
from ..signals.Signal import Signal
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
import os
from dotenv import load_dotenv
from ..utils.telegram_alerts import TelegramAlerts

load_dotenv()

class Executor:
    def __init__(self, monitor: SignalMonitor, config: Dict[str, Any]):
        self.monitor = monitor
        self.config = config
        self.mode = config.get("mode", "paper")  # "paper" or "live"
        # Always load Alpaca keys from environment
        if self.mode == "paper":
            self.api_key = os.environ.get("ALPACA_PAPER_API_KEY")
            self.api_secret = os.environ.get("ALPACA_PAPER_API_SECRET")
        else:
            self.api_key = os.environ.get("ALPACA_API_KEY")
            self.api_secret = os.environ.get("ALPACA_API_SECRET")
        print(f"[Executor] Initialized in {self.mode} mode")
        self.paper = True  # Always use paper trading for safety
        self.client = TradingClient(self.api_key, self.api_secret, paper=self.paper)
        self.data_client = StockHistoricalDataClient(self.api_key, self.api_secret)
        # In-memory position tracker: symbol -> shares
        self.positions: Dict[str, int] = {}
        
        # Initialize Telegram alerts
        self.alerts = TelegramAlerts()

    def execute(self):
        signals: List[Signal] = self.monitor.run_once()
        print(f"[Executor] {len(signals)} signals received.")
        
        for sig in signals:
            print(f"[DEBUG] Signal: symbol={sig.symbol}, action={sig.action}, target_pct={sig.target_pct}, asset_type={sig.asset_type}")
            
            # Send Telegram alert for each signal
            if self.alerts.enabled:
                signal_data = {
                    'symbol': sig.symbol,
                    'action': sig.action,
                    'target_pct': sig.target_pct,
                    'asset_type': sig.asset_type,
                    'strategy': 'Cointegration',
                    'z_score': 'N/A',  # Will be updated if available
                    'position': 'N/A'  # Will be updated if available
                }
                self.alerts.send_signal_alert(signal_data)
            
            if sig.asset_type == "equity":
                self._handle_equity(sig)
            else:
                print(f"[Executor] Unsupported asset type '{sig.asset_type}' — skipping signal: {sig}")

    def _handle_equity(self, sig: Signal):
        # Determine target position in shares
        target_shares = self._determine_target_shares(sig)
        current_shares = self.positions.get(sig.symbol, 0)
        delta = target_shares - current_shares
        print(f"[DEBUG] {sig.symbol}: current={current_shares}, target={target_shares}, delta={delta}")
        if delta == 0:
            print(f"[Executor] No order needed for {sig.symbol} (already at target position)")
            return
        action = "buy" if delta > 0 else "sell"
        order_qty = abs(delta)
        if self.mode == "live":
            try:
                print(f"[LIVE] (Alpaca) {action.upper()} {sig.symbol} | Qty: {order_qty}")
                order_side = OrderSide.BUY if action == "buy" else OrderSide.SELL
                order = MarketOrderRequest(
                    symbol=sig.symbol,
                    qty=order_qty,
                    side=order_side,
                    time_in_force=TimeInForce.DAY
                )
                self.client.submit_order(order)
            except Exception as e:
                print(f"[ERROR] Failed to place equity order: {e}")
        else:
            print(f"[PAPER] (Alpaca) {action.upper()} {sig.symbol} | Qty: {order_qty}")
        # Update in-memory position
        self.positions[sig.symbol] = target_shares

    def _determine_target_shares(self, sig: Signal) -> int:
        if sig.quantity is not None:
            return int(sig.quantity)
        elif sig.target_pct is not None:
            capital = self.config.get("paper_capital", 10000)
            try:
                req = StockLatestTradeRequest(symbol_or_symbols=sig.symbol)
                latest_trade = self.data_client.get_stock_latest_trade(req)
                price = latest_trade[sig.symbol].price
            except Exception:
                price = 1.0  # fallback
            shares = (sig.target_pct * capital) / price
            return int(round(shares))
        else:
            return 0
