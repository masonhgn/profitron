#!/usr/bin/env python3
"""
Test script for IBKR TWS API implementation
"""

import os
import time

from brokers.IBKRBrokerContext import IBKRBrokerContext

def test_ibkr_connection():
    """Test basic connection to TWS"""
    print("Testing IBKR TWS API connection...")
    
    # Configuration for testing
    config = {
        "host": "127.0.0.1",
        "port": 4002,  # Your TWS API port
        "client_id": 1,
        "mode": "paper",
        "paper_account_id": os.getenv("IBKR_PAPER_ACCOUNT_ID", "DU1234567")
    }
    
    try:
        # Create broker context
        broker = IBKRBrokerContext("test_ibkr", config)
        
        # Wait a bit for connection
        time.sleep(2)
        
        if broker.authenticated:
            print("✅ Successfully connected to TWS!")
            
            # Test basic functionality
            print("\nTesting account info...")
            account_info = broker.get_account_info()
            print(f"Account info: {account_info}")
            
            print("\nTesting positions...")
            positions = broker.get_positions()
            print(f"Positions: {positions}")
            
            print("\nTesting market data for AAPL...")
            market_data = broker.get_market_data("AAPL")
            print(f"Market data: {market_data}")
            
            print("\nTesting historical data for AAPL...")
            historical_data = broker.get_historical_data("AAPL", duration="1 D", bar_size="1 hour")
            print(f"Historical data bars: {len(historical_data)}")
            if historical_data:
                print(f"Sample bar: {historical_data[0]}")
            
            # Cleanup
            broker.disconnect()
            print("\n✅ All tests passed!")
            
        else:
            print("❌ Failed to connect to TWS")
            print("Make sure TWS or IB Gateway is running and API connections are enabled")
            print("Check that the port and client ID are correct")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ibkr_connection() 