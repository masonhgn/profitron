# src/brokers/IBKRBrokerContext.py

import threading
import time
import json
import os
from typing import Dict, Optional, List, Any
from pathlib import Path
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.common import OrderId, TickerId
from ibapi.ticktype import TickTypeEnum
from .BrokerContext import BrokerContext
from ..signals.Signal import Signal
from ..execution.OrderManager import OrderManager
from ..portfolio.LivePortfolio import LivePortfolio

class IBKRBrokerContext(EClient, EWrapper, BrokerContext):
    """
    Interactive Brokers broker context using TWS API
    Handles authentication, session management, order execution, and market data
    """
    
    def __init__(self, name: str, config: dict):
        # Initialize EClient and EWrapper
        EClient.__init__(self, self)
        
        # Initialize BrokerContext
        BrokerContext.__init__(self, name, config)
        
        # IBKR specific configuration
        self.host = config.get("host", "127.0.0.1")
        self.port = config.get("port", 7497)  # 7497 for TWS, 4001 for IB Gateway
        self.client_id = config.get("client_id", 1)
        self.paper_trading = config.get("mode", "paper") == "paper"
        
        # Set account ID based on mode
        if self.paper_trading:
            self.account_id = config.get("paper_account_id") or os.getenv("IBKR_PAPER_ACCOUNT_ID")
        else:
            self.account_id = config.get("live_account_id") or os.getenv("IBKR_LIVE_ACCOUNT_ID")
        
        # Connection state
        self.connected = False
        self.authenticated = False
        self.next_order_id = None
        
        # Data storage
        self.positions = []
        self.orders = []
        self.account_info = {}
        self.market_data = {}
        self.historical_data = {}
        
        # Contract cache
        self.cache_file = Path("data/ibkr_contract_cache.json")
        self.cache_file.parent.mkdir(exist_ok=True)
        self.contract_cache = self._load_contract_cache()
        
        # Initialize components
        self.order_manager = OrderManager(self)
        self.portfolio = LivePortfolio(self)
        
        # Connect and authenticate
        self._connect()
        
        # Sync portfolio with actual account balance for live trading
        if self.authenticated and not self.paper_trading:
            self._sync_portfolio_balance()
    
    def _load_contract_cache(self) -> Dict[str, int]:
        """Load contract cache from JSON file"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[IBKR] Error loading contract cache: {e}")
        return {}
    
    def _save_contract_cache(self):
        """Save contract cache to JSON file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.contract_cache, f, indent=2)
        except Exception as e:
                print(f"[IBKR] Error saving contract cache: {e}")
    
    def _connect(self):
        """Connect to TWS/IB Gateway"""
        try:
            print(f"[IBKR] Connecting to TWS at {self.host}:{self.port}")
            self.connect(self.host, self.port, self.client_id)
            
            # Start the message processing thread
            self.thread = threading.Thread(target=self.run)
            self.thread.daemon = True
            self.thread.start()
            
            # Wait for connection
            timeout = 10
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if self.connected:
                print("[IBKR] Connected to TWS successfully")
                self.authenticated = True
            else:
                print("[IBKR] Failed to connect to TWS")
                
        except Exception as e:
            print(f"[IBKR] Connection error: {e}")
    
    def disconnect(self):
        """Disconnect from TWS"""
        if self.connected:
            self.disconnect()
            self.connected = False
            print("[IBKR] Disconnected from TWS")
    
    # EWrapper callback methods
    def nextValidId(self, orderId: OrderId):
        """Called when TWS assigns the next valid order ID"""
        self.next_order_id = orderId
        print(f"[IBKR] Next valid order ID: {orderId}")
    
    def error(self, reqId: TickerId, errorCode: int, errorString: str, advancedOrderReject=""):
        """Called when TWS reports an error"""
        print(f"[IBKR] Error - reqId: {reqId}, errorCode: {errorCode}, errorString: {errorString}")
        if advancedOrderReject:
            print(f"[IBKR] Advanced order reject: {advancedOrderReject}")
    
    def connectionClosed(self):
        """Called when the connection to TWS is closed"""
        self.connected = False
        print("[IBKR] Connection to TWS closed")
    
    def connectAck(self):
        """Called when connection is acknowledged"""
        self.connected = True
        print("[IBKR] Connection acknowledged")
    
    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        """Called when position data is received"""
        position_data = {
            "account": account,
            "symbol": contract.symbol,
            "secType": contract.secType,
            "currency": contract.currency,
            "exchange": contract.exchange,
            "position": position,
            "avgCost": avgCost
        }
        self.positions.append(position_data)
    
    def positionEnd(self):
        """Called when all position data has been received"""
        print(f"[IBKR] Received {len(self.positions)} positions")
    
    def orderStatus(self, orderId: OrderId, status: str, filled: float, remaining: float, avgFillPrice: float, permId: int, parentId: int, lastFillPrice: float, clientId: int, whyHeld: str, mktCapPrice: float):
        """Called when order status changes"""
        order_data = {
            "orderId": orderId,
            "status": status,
            "filled": filled,
            "remaining": remaining,
            "avgFillPrice": avgFillPrice,
            "permId": permId,
            "parentId": parentId,
            "lastFillPrice": lastFillPrice,
            "clientId": clientId,
            "whyHeld": whyHeld,
            "mktCapPrice": mktCapPrice
        }
        
        # Update existing order or add new one
        for i, order in enumerate(self.orders):
            if order.get("orderId") == orderId:
                self.orders[i] = order_data
                break
        else:
            self.orders.append(order_data)
        
        print(f"[IBKR] Order {orderId} status: {status}, filled: {filled}, remaining: {remaining}")
    
    def tickPrice(self, reqId: TickerId, tickType: int, price: float, attrib):
        """Called when price data is received"""
        tick_type_str = TickTypeEnum.toStr(tickType)
        if reqId not in self.market_data:
            self.market_data[reqId] = {}
        self.market_data[reqId][tick_type_str] = price
        print(f"[IBKR] Tick Price - reqId: {reqId}, tickType: {tick_type_str}, price: {price}")
    
    def tickSize(self, reqId: TickerId, tickType: int, size: int):
        """Called when size data is received"""
        tick_type_str = TickTypeEnum.toStr(tickType)
        if reqId not in self.market_data:
            self.market_data[reqId] = {}
        self.market_data[reqId][tick_type_str] = size
        print(f"[IBKR] Tick Size - reqId: {reqId}, tickType: {tick_type_str}, size: {size}")
    
    def historicalData(self, reqId: int, bar):
        """Called when historical data is received"""
        if reqId not in self.historical_data:
            self.historical_data[reqId] = []
        
        bar_data = {
            "date": bar.date,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "barCount": bar.barCount,
            "wap": bar.wap
        }
        self.historical_data[reqId].append(bar_data)
    
    def historicalDataEnd(self, reqId: int, start: str, end: str):
        """Called when historical data request is complete"""
        print(f"[IBKR] Historical data complete for reqId {reqId}: {len(self.historical_data.get(reqId, []))} bars")
    
    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        """Called when account summary data is received"""
        if reqId not in self.account_info:
            self.account_info[reqId] = {}
        self.account_info[reqId][tag] = {"value": value, "currency": currency}
    
    def accountSummaryEnd(self, reqId: int):
        """Called when account summary request is complete"""
        print(f"[IBKR] Account summary complete for reqId {reqId}")
    
    def nextId(self) -> int:
        """Get the next available order ID"""
        if self.next_order_id is None:
            return 1
        self.next_order_id += 1
        return self.next_order_id
    
    def get_conid(self, symbol: str, asset_type: str = "STK") -> Optional[int]:
        """Get conid for symbol, with caching"""
        cache_key = f"{symbol}_{asset_type}"
        
        if cache_key in self.contract_cache:
            return self.contract_cache[cache_key]
        
        # For now, we'll use a simple approach - in a real implementation,
        # you would use reqContractDetails to get the actual conid
        # This is a placeholder that would need to be implemented with proper contract details
        print(f"[IBKR] Contract details lookup not implemented for {symbol}")
        return None
    
    def create_contract(self, symbol: str, asset_type: str = "STK", currency: str = "USD", exchange: str = "SMART") -> Contract:
        """Create a contract object for the given symbol"""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = asset_type
        contract.currency = currency
        contract.exchange = exchange
        return contract
    
    def place_order(self, signal: Signal) -> Dict:
        """Place order using TWS API"""
        try:
            if not self.authenticated:
                print("[IBKR] Not authenticated, cannot place order")
                return {"success": False, "error": "Not authenticated"}
            
            # Create contract
            contract = self.create_contract(signal.symbol, signal.asset_type)
            
            # Create order
            order = Order()
            order.action = "BUY" if signal.action == "buy" else "SELL"
            order.orderType = "MKT"  # Market order for simplicity
            order.totalQuantity = int(signal.target_pct * 100) if signal.target_pct > 0 else 0
            order.tif = "DAY"
            
            # Get next order ID
            order_id = self.nextId()
            
            # Submit order
            self.placeOrder(order_id, contract, order)
            
            print(f"[IBKR] Order placed: {order_id} for {signal.symbol}")
            return {
                "success": True,
                "order_id": order_id,
                "status": "SUBMITTED"
            }
            
        except Exception as e:
            print(f"[IBKR] Error placing order: {e}")
            return {"success": False, "error": str(e)}
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        try:
            if not self.authenticated:
                return {"error": "Not authenticated"}
            
            # Request account summary
            req_id = self.nextId()
            self.reqAccountSummary(req_id, "All", "NetLiquidation,BuyingPower,TotalCashValue")
            
            # Wait for data (in a real implementation, you'd use proper async handling)
            time.sleep(1)
            
            return self.account_info.get(req_id, {})
                
        except Exception as e:
            return {"error": str(e)}
    
    def _sync_portfolio_balance(self):
        """Sync portfolio with actual account balance from IBKR"""
        try:
            account_info = self.get_account_info()
            if "error" not in account_info:
                # Look for cash balance in account info
                for tag, data in account_info.items():
                    if tag == "TotalCashValue":
                        cash_balance = float(data["value"])
                        self.portfolio.cash = cash_balance
                        print(f"[IBKR] Synced portfolio balance: ${cash_balance:,.2f}")
                        return
                
                print("[IBKR] Could not find cash balance in account info")
            else:
                print(f"[IBKR] Failed to get account info: {account_info['error']}")
                
        except Exception as e:
            print(f"[IBKR] Error syncing portfolio balance: {e}")
    
    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        try:
            if not self.authenticated:
                return []
            
            # Clear previous positions
            self.positions = []
            
            # Request positions
            self.reqPositions()
            
            # Wait for data (in a real implementation, you'd use proper async handling)
            time.sleep(1)
            
            return self.positions
                
        except Exception as e:
            print(f"[IBKR] Error getting positions: {e}")
            return []
    
    def get_orders(self) -> List[Dict]:
        """Get current orders"""
        try:
            if not self.authenticated:
                return []
            
            # Note: TWS API doesn't have a direct method to get all orders
            # This would need to be implemented by tracking orders as they're placed
            return self.orders
                
        except Exception as e:
            print(f"[IBKR] Error getting orders: {e}")
            return []
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            if not self.authenticated:
                return False
            
            self.cancelOrder(int(order_id))
            print(f"[IBKR] Order {order_id} cancellation requested")
            return True
                
        except Exception as e:
            print(f"[IBKR] Error cancelling order {order_id}: {e}")
            return False
    
    def get_market_data(self, symbol: str, fields: List[str] = None) -> Dict:
        """Get market data for a symbol"""
        try:
            if not self.authenticated:
                return {"error": "Not authenticated"}
            
            # Create contract
            contract = self.create_contract(symbol)
            
            # Generate request ID
            req_id = self.nextId()
            
            # Request market data
            generic_tick_list = "232"  # Mark price
            snapshot = False
            regulatory_snapshot = False
            mkt_data_options = []
            
            self.reqMktData(req_id, contract, generic_tick_list, snapshot, regulatory_snapshot, mkt_data_options)
            
            # Wait for data (in a real implementation, you'd use proper async handling)
            time.sleep(1)
            
            return self.market_data.get(req_id, {})
                
        except Exception as e:
            return {"error": str(e)}
    
    def get_historical_data(self, symbol: str, duration: str = "1 D", bar_size: str = "1 hour", 
                          what_to_show: str = "TRADES", use_rth: bool = True) -> List[Dict]:
        """Get historical data for a symbol"""
        try:
            if not self.authenticated:
                return []
            
            # Create contract
            contract = self.create_contract(symbol)
            
            # Generate request ID
            req_id = self.nextId()
            
            # Request historical data
            end_date_time = ""  # Current time
            format_date = 1  # String format
            
            self.reqHistoricalData(req_id, contract, end_date_time, duration, bar_size, 
                                 what_to_show, use_rth, format_date, False, [])
            
            # Wait for data (in a real implementation, you'd use proper async handling)
            time.sleep(2)
            
            return self.historical_data.get(req_id, [])
                
        except Exception as e:
            print(f"[IBKR] Error getting historical data: {e}")
            return []
    
    def __del__(self):
        """Cleanup on destruction"""
        self.disconnect() 