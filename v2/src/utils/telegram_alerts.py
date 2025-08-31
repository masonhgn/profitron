import requests
import os
from typing import Optional, Dict, Any
from datetime import datetime
import json

class TelegramAlerts:
    """Telegram bot for sending trading alerts"""
    
    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        self.bot_token = bot_token or os.environ.get('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.environ.get('TELEGRAM_CHAT_ID')
        
        if not self.bot_token or not self.chat_id:
            print("[TelegramAlerts] Warning: Missing bot token or chat ID. Alerts will be disabled.")
            self.enabled = False
        else:
            self.enabled = True
            self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
            print(f"[TelegramAlerts] Initialized with chat ID: {self.chat_id}")
    
    def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        Send a message to Telegram
        
        Args:
            message: Message text
            parse_mode: HTML or Markdown formatting
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        if not self.enabled:
            return False
            
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            
            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()
            
            return True
            
        except Exception as e:
            print(f"[TelegramAlerts] Error sending message: {e}")
            return False
    
    def send_signal_alert(self, signal_data: Dict[str, Any]) -> bool:
        """
        Send a trading signal alert
        
        Args:
            signal_data: Dictionary containing signal information
        """
        if not self.enabled:
            return False
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create emoji based on action
        action_emoji = {
            "buy": "🟢",
            "sell": "🔴",
            "exit": "🟡"
        }
        
        emoji = action_emoji.get(signal_data.get('action', '').lower(), "📊")
        
        message = f"""
{emoji} <b>TRADING SIGNAL</b> {emoji}

📅 <b>Time:</b> {timestamp}
💱 <b>Symbol:</b> {signal_data.get('symbol', 'N/A')}
🎯 <b>Action:</b> {signal_data.get('action', 'N/A').upper()}
📊 <b>Target %:</b> {signal_data.get('target_pct', 0):.2%}
💰 <b>Asset Type:</b> {signal_data.get('asset_type', 'N/A')}

📈 <b>Strategy:</b> {signal_data.get('strategy', 'Cointegration')}
🔢 <b>Z-Score:</b> {signal_data.get('z_score', 'N/A')}
📊 <b>Position:</b> {signal_data.get('position', 'N/A')}
        """.strip()
        
        return self.send_message(message)
    
    def send_system_status(self, status: str, details: Optional[str] = None) -> bool:
        """
        Send system status update
        
        Args:
            status: Status message
            details: Optional details
        """
        if not self.enabled:
            return False
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Choose emoji based on status
        if "error" in status.lower() or "failed" in status.lower():
            emoji = "❌"
        elif "start" in status.lower() or "running" in status.lower():
            emoji = "🟢"
        elif "stop" in status.lower() or "shutdown" in status.lower():
            emoji = "🔴"
        else:
            emoji = "ℹ️"
        
        message = f"""
{emoji} <b>SYSTEM STATUS</b> {emoji}

📅 <b>Time:</b> {timestamp}
📊 <b>Status:</b> {status}
        """.strip()
        
        if details:
            message += f"\n\n📝 <b>Details:</b>\n{details}"
        
        return self.send_message(message)
    
    def send_error_alert(self, error: str, context: Optional[str] = None) -> bool:
        """
        Send error alert
        
        Args:
            error: Error message
            context: Optional context information
        """
        if not self.enabled:
            return False
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"""
🚨 <b>ERROR ALERT</b> 🚨

📅 <b>Time:</b> {timestamp}
❌ <b>Error:</b> {error}
        """.strip()
        
        if context:
            message += f"\n\n📋 <b>Context:</b>\n{context}"
        
        return self.send_message(message)
    
    def send_daily_summary(self, summary_data: Dict[str, Any]) -> bool:
        """
        Send daily trading summary
        
        Args:
            summary_data: Dictionary containing summary information
        """
        if not self.enabled:
            return False
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"""
📊 <b>DAILY SUMMARY</b> 📊

📅 <b>Date:</b> {timestamp}
💰 <b>P&L:</b> ${summary_data.get('pnl', 0):.2f}
📈 <b>Total Return:</b> {summary_data.get('total_return', 0):.2%}
📊 <b>Sharpe Ratio:</b> {summary_data.get('sharpe_ratio', 0):.2f}
📉 <b>Max Drawdown:</b> {summary_data.get('max_drawdown', 0):.2%}
🔄 <b>Total Trades:</b> {summary_data.get('total_trades', 0)}
✅ <b>Win Rate:</b> {summary_data.get('win_rate', 0):.1%}
        """.strip()
        
        return self.send_message(message)
    
    def test_connection(self) -> bool:
        """
        Test the Telegram bot connection
        
        Returns:
            bool: True if connection successful
        """
        if not self.enabled:
            print("[TelegramAlerts] Bot not enabled - skipping connection test")
            return False
            
        try:
            url = f"{self.base_url}/getMe"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            bot_info = response.json()
            if bot_info.get('ok'):
                print(f"[TelegramAlerts] Connection test successful - Bot: {bot_info['result']['username']}")
                return True
            else:
                print(f"[TelegramAlerts] Connection test failed: {bot_info}")
                return False
                
        except Exception as e:
            print(f"[TelegramAlerts] Connection test error: {e}")
            return False 