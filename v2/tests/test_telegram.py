#!/usr/bin/env python3
"""
Test script for Telegram alerts
Run this to verify your bot is working correctly
"""

import os
from dotenv import load_dotenv
from src.utils.telegram_alerts import TelegramAlerts

def test_telegram_alerts():
    """Test all Telegram alert functions"""
    
    # Load environment variables
    load_dotenv()
    
    print("Testing Telegram Alerts...")
    print("=" * 50)
    
    # Initialize alerts
    alerts = TelegramAlerts()
    
    if not alerts.enabled:
        print("❌ Telegram alerts not enabled!")
        print("Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in your .env file")
        return False
    
    print("✅ Telegram alerts initialized")
    
    # Test connection
    print("\nTesting connection...")
    if alerts.test_connection():
        print("✅ Connection test successful")
    else:
        print("❌ Connection test failed")
        return False
    
    # Test system status
    print("\nTesting system status alert...")
    if alerts.send_system_status("Test system status", "This is a test message"):
        print("✅ System status alert sent")
    else:
        print("❌ System status alert failed")
    
    # Test signal alert
    print("\nTesting signal alert...")
    test_signal = {
        'symbol': 'ETHA',
        'action': 'buy',
        'target_pct': 0.25,
        'asset_type': 'equity',
        'strategy': 'Cointegration',
        'z_score': 2.1,
        'position': 'long_spread'
    }
    
    if alerts.send_signal_alert(test_signal):
        print("✅ Signal alert sent")
    else:
        print("❌ Signal alert failed")
    
    # Test error alert
    print("\nTesting error alert...")
    if alerts.send_error_alert("Test error message", "This is a test error context"):
        print("✅ Error alert sent")
    else:
        print("❌ Error alert failed")
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")
    print("Check your Telegram for the test messages.")
    
    return True

if __name__ == "__main__":
    test_telegram_alerts() 