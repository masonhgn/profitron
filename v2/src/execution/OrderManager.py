import time
from typing import Dict, Optional
from ..signals.Signal import Signal
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

class OrderManager:
    def __init__(self, client: TradingClient, timeout: int = 30):
        """
        Args:
            client: Alpaca TradingClient
            timeout: seconds before canceling a stale order
        """
        self.client = client
        self.timeout = timeout
        self.open_orders: Dict[str, Dict] = {}  # order_id -> {signal, time, status}

    def submit_order(self, signal: Signal) -> Optional[str]:
        try:
            symbol = signal.symbol
            side = OrderSide.BUY if signal.action == "buy" else OrderSide.SELL
            quantity = int(signal.quantity or self._infer_quantity(signal))
            limit_price = signal.meta.get("limit_price")
            order = None
            if limit_price:
                order_req = LimitOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=side,
                    limit_price=limit_price,
                    time_in_force=TimeInForce.DAY
                )
                order = self.client.submit_order(order_req)
                print(f"[OrderManager] submitted LIMIT {side.value.upper()} {symbol} x{quantity} @ {limit_price}")
            else:
                order_req = MarketOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY
                )
                order = self.client.submit_order(order_req)
                print(f"[OrderManager] submitted MARKET {side.value.upper()} {symbol} x{quantity}")
            if order:
                order_id = getattr(order, 'id', None)
                if order_id is not None:
                    order_id = str(order_id)
                    status = getattr(order, 'status', 'unknown')
                    self.open_orders[order_id] = {
                        "signal": signal,
                        "timestamp": time.time(),
                        "status": status
                    }
                    return order_id
                else:
                    print("[OrderManager] Warning: order.id not found in Alpaca order response.")
                    return None
            return None
        except Exception as e:
            print(f"[OrderManager] failed to submit order: {e}")
            return None

    def _infer_quantity(self, signal: Signal) -> float:
        # For simplicity, fallback to 1 share if not specified
        return 1

    def check_and_cancel_stale_orders(self):
        """check if an order is older than timeout, cancel if so"""
        now = time.time()
        to_cancel = []
        for order_id, meta in self.open_orders.items():
            age = now - meta["timestamp"]
            if age > self.timeout and meta["status"] == "open":
                to_cancel.append(order_id)
        for oid in to_cancel:
            try:
                self.client.cancel_order_by_id(oid)
                print(f"[OrderManager] canceled stale order: {oid}")
                signal = self.open_orders[oid]["signal"]
                if signal.meta.get("replace_with_market", False):
                    self._replace_with_market(signal)
                self.open_orders[oid]["status"] = "cancelled"
            except Exception as e:
                print(f"[OrderManager] failed to cancel order {oid}: {e}")

    def _replace_with_market(self, signal: Signal):
        try:
            symbol = signal.symbol
            side = OrderSide.BUY if signal.action == "buy" else OrderSide.SELL
            quantity = int(signal.quantity or self._infer_quantity(signal))
            order_req = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            self.client.submit_order(order_req)
            print(f"[OrderManager] replaced with MARKET {side.value.upper()} {symbol} x{quantity}")
        except Exception as e:
            print(f"[OrderManager] failed to replace with market order: {e}")
