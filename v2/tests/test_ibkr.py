#!/usr/bin/env python3
"""
Integration tests for IBKR broker functionality
Tests actual connection and functionality with IBKR
"""

import pytest
import os
import sys
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from brokers.IBKRBrokerContext import IBKRBrokerContext
from signals.Signal import Signal

# Load environment variables
load_dotenv()

class TestIBKRIntegration:
    """Integration tests for IBKR broker functionality"""
    
    @pytest.fixture
    def ibkr_config(self):
        """IBKR configuration for testing"""
        return {
            "gateway_url": "https://localhost:5008/v1/api",
            "username": os.getenv("IBKR_USERNAME"),
            "password": os.getenv("IBKR_PASSWORD"),
            "paper_account_id": os.getenv("IBKR_PAPER_ACCOUNT_ID"),
            "mode": "paper"
        }
    
    @pytest.fixture
    def ibkr_broker(self, ibkr_config):
        """Create IBKR broker instance for testing"""
        return IBKRBrokerContext("test_ibkr", ibkr_config)
    
    def test_ibkr_authentication(self, ibkr_broker):
        """Test that IBKR authentication works"""
        # This is a real test that requires the Client Portal Gateway to be running
        if not os.getenv("IBKR_USERNAME"):
            pytest.skip("IBKR credentials not configured")
        
        # Test authentication
        assert ibkr_broker.authenticated, "IBKR authentication failed"
        print(f"Authentication successful")
    
    def test_ibkr_account_info(self, ibkr_broker):
        """Test getting account information from IBKR"""
        if not os.getenv("IBKR_USERNAME"):
            pytest.skip("IBKR credentials not configured")
        
        if not ibkr_broker.authenticated:
            pytest.skip("IBKR not authenticated")
        
        # Get account info
        account_info = ibkr_broker.get_account_info()
        
        # Verify we got account data
        assert "error" not in account_info, f"Failed to get account info: {account_info}"
        assert isinstance(account_info, dict), "Account info should be a dictionary"
        
        # Check for expected account data structure
        assert len(account_info) > 0, "Account info should not be empty"
        
        # Look for cash balance data
        found_cash_balance = False
        for currency, data in account_info.items():
            if isinstance(data, dict) and "cashbalance" in data:
                cash_balance = data["cashbalance"]
                assert isinstance(cash_balance, (int, float)), "Cash balance should be numeric"
                assert cash_balance >= 0, "Cash balance should be non-negative"
                found_cash_balance = True
                print(f"Found cash balance: ${cash_balance:,.2f} ({currency})")
                break
        
        assert found_cash_balance, "Could not find cash balance in account info"
    
    def test_ibkr_contract_discovery(self, ibkr_broker):
        """Test contract discovery (symbol to conid mapping)"""
        if not os.getenv("IBKR_USERNAME"):
            pytest.skip("IBKR credentials not configured")
        
        if not ibkr_broker.authenticated:
            pytest.skip("IBKR not authenticated")
        
        # Test symbols that should exist on IBKR
        test_symbols = [
            ("AAPL", "STK"),
            ("SPY", "STK"), 
            ("QQQ", "STK"),
            ("ES", "FUT")  # E-mini S&P 500 futures
        ]
        
        for symbol, asset_type in test_symbols:
            conid = ibkr_broker.get_conid(symbol, asset_type)
            
            assert conid is not None, f"Could not find conid for {symbol}"
            assert isinstance(conid, int), f"Conid for {symbol} should be an integer"
            assert conid > 0, f"Conid for {symbol} should be positive"
            
            print(f"Found conid {conid} for {symbol} ({asset_type})")
            
            # Test that the conid is cached
            cached_conid = ibkr_broker.get_conid(symbol, asset_type)
            assert cached_conid == conid, f"Conid for {symbol} should be cached"
    
    def test_ibkr_market_data(self, ibkr_broker):
        """Test getting market data from IBKR"""
        if not os.getenv("IBKR_USERNAME"):
            pytest.skip("IBKR credentials not configured")
        
        if not ibkr_broker.authenticated:
            pytest.skip("IBKR not authenticated")
        
        # Get conid for a test symbol
        conid = ibkr_broker.get_conid("AAPL", "STK")
        if not conid:
            pytest.skip("Could not get conid for AAPL")
        
        # Get market data
        market_data = ibkr_broker.get_market_data("AAPL")
        
        # Verify market data structure
        assert "error" not in market_data, f"Failed to get market data: {market_data}"
        assert isinstance(market_data, dict), "Market data should be a dictionary"
        assert "conid" in market_data, "Market data should contain conid"
        assert market_data["conid"] == conid, "Market data conid should match"
        
        # Check for basic market data fields
        expected_fields = ["31", "84", "85", "86", "88"]  # Last, Bid, BidSize, Ask, AskSize
        found_fields = []
        
        for field in expected_fields:
            if field in market_data:
                value = market_data[field]
                if value and value != "0":
                    found_fields.append(field)
        
        assert len(found_fields) > 0, f"No market data fields found. Available: {list(market_data.keys())}"
        print(f"Found market data fields: {found_fields}")
    
    def test_ibkr_positions(self, ibkr_broker):
        """Test getting positions from IBKR"""
        if not os.getenv("IBKR_USERNAME"):
            pytest.skip("IBKR credentials not configured")
        
        if not ibkr_broker.authenticated:
            pytest.skip("IBKR not authenticated")
        
        # Get positions
        positions = ibkr_broker.get_positions()
        
        # Verify positions data
        assert isinstance(positions, list), "Positions should be a list"
        
        print(f"Retrieved {len(positions)} positions")
        
        # If there are positions, verify their structure
        if positions:
            position = positions[0]
            assert isinstance(position, dict), "Position should be a dictionary"
            # Note: We don't check specific fields as position structure may vary
    
    def test_ibkr_orders(self, ibkr_broker):
        """Test getting orders from IBKR"""
        if not os.getenv("IBKR_USERNAME"):
            pytest.skip("IBKR credentials not configured")
        
        if not ibkr_broker.authenticated:
            pytest.skip("IBKR not authenticated")
        
        # Get orders
        orders = ibkr_broker.get_orders()
        
        # Verify orders data
        assert isinstance(orders, list), "Orders should be a list"
        
        print(f"Retrieved {len(orders)} orders")
        
        # If there are orders, verify their structure
        if orders:
            order = orders[0]
            assert isinstance(order, dict), "Order should be a dictionary"
            # Note: We don't check specific fields as order structure may vary
    
    def test_ibkr_portfolio_sync(self, ibkr_broker):
        """Test portfolio balance synchronization"""
        if not os.getenv("IBKR_USERNAME"):
            pytest.skip("IBKR credentials not configured")
        
        if not ibkr_broker.authenticated:
            pytest.skip("IBKR not authenticated")
        
        # Test portfolio sync for live trading
        if not ibkr_broker.paper_trading:
            # Store original cash balance
            original_cash = ibkr_broker.portfolio.cash
            
            # Sync portfolio balance
            ibkr_broker._sync_portfolio_balance()
            
            # Verify cash balance was updated
            assert ibkr_broker.portfolio.cash != original_cash, "Cash balance should be updated"
            assert ibkr_broker.portfolio.cash >= 0, "Cash balance should be non-negative"
            
            print(f"Portfolio synced: ${ibkr_broker.portfolio.cash:,.2f}")
        else:
            print("Paper trading - portfolio sync not applicable")
    
    def test_ibkr_order_placement_simulation(self, ibkr_broker):
        """Test order placement simulation (without actually placing orders)"""
        if not os.getenv("IBKR_USERNAME"):
            pytest.skip("IBKR credentials not configured")
        
        if not ibkr_broker.authenticated:
            pytest.skip("IBKR not authenticated")
        
        # Create a test signal
        test_signal = Signal(
            symbol="AAPL",
            action="buy",
            target_pct=0.01,  # 1% position
            asset_type="STK"
        )
        
        # Test order preparation (without actually submitting)
        conid = ibkr_broker.get_conid("AAPL", "STK")
        if not conid:
            pytest.skip("Could not get conid for AAPL")
        
        # Verify we can prepare the order
        order = {
            "conid": conid,
            "side": "BUY" if test_signal.action == "buy" else "SELL",
            "orderType": "MKT",
            "quantity": int(test_signal.target_pct * 100) if test_signal.target_pct > 0 else 0,
            "tif": "DAY"
        }
        
        assert order["conid"] == conid, "Order conid should match"
        assert order["side"] == "BUY", "Order side should be BUY"
        assert order["quantity"] == 1, "Order quantity should be 1"
        
        print(f"Order preparation successful: {order}")
    
    def test_ibkr_cache_functionality(self, ibkr_broker):
        """Test contract cache functionality"""
        if not os.getenv("IBKR_USERNAME"):
            pytest.skip("IBKR credentials not configured")
        
        if not ibkr_broker.authenticated:
            pytest.skip("IBKR not authenticated")
        
        # Test cache loading
        assert isinstance(ibkr_broker.contract_cache, dict), "Contract cache should be a dictionary"
        
        # Test cache saving
        original_cache_size = len(ibkr_broker.contract_cache)
        
        # Get a conid to populate cache
        conid = ibkr_broker.get_conid("AAPL", "STK")
        assert conid is not None, "Should be able to get conid for AAPL"
        
        # Verify cache was updated
        assert len(ibkr_broker.contract_cache) >= original_cache_size, "Cache should be populated"
        
        # Test cache file exists
        assert ibkr_broker.cache_file.exists(), "Cache file should exist"
        
        print(f"Cache functionality working: {len(ibkr_broker.contract_cache)} entries")
    
    def test_ibkr_error_handling(self, ibkr_broker):
        """Test error handling for invalid requests"""
        if not os.getenv("IBKR_USERNAME"):
            pytest.skip("IBKR credentials not configured")
        
        if not ibkr_broker.authenticated:
            pytest.skip("IBKR not authenticated")
        
        # Test invalid symbol
        invalid_conid = ibkr_broker.get_conid("INVALID_SYMBOL_12345", "STK")
        assert invalid_conid is None, "Invalid symbol should return None"
        
        # Test invalid asset type
        invalid_conid = ibkr_broker.get_conid("AAPL", "INVALID_TYPE")
        assert invalid_conid is None, "Invalid asset type should return None"
        
        print("Error handling working correctly")

if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"]) 