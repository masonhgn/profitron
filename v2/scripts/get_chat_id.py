#!/usr/bin/env python3
"""
Simple script to get your Telegram chat ID
"""

import requests

def get_chat_id(bot_token):
    """Get chat ID from bot updates"""
    
    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        print("Bot API Response:")
        print("=" * 50)
        print(f"Status: {data.get('ok')}")
        print(f"Updates: {len(data.get('result', []))}")
        
        if data.get('result'):
            print("\nFound updates:")
            for i, update in enumerate(data['result']):
                if 'message' in update:
                    message = update['message']
                    chat = message.get('chat', {})
                    print(f"\nUpdate {i+1}:")
                    print(f"  Chat ID: {chat.get('id')}")
                    print(f"  Name: {chat.get('first_name', 'N/A')}")
                    print(f"  Username: {chat.get('username', 'N/A')}")
                    print(f"  Message: {message.get('text', 'N/A')}")
        else:
            print("\nNo updates found!")
            print("Please send a message to your bot first, then run this script again.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    bot_token = "7309059762:AAEHeeoo6VCCqjaHAqSY3Eo-xSBmEHnC8RI"
    print("Getting chat ID for your bot...")
    get_chat_id(bot_token) 