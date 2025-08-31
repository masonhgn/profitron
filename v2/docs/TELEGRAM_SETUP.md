# Telegram Bot Setup Guide

## Quick Setup (5 minutes)

### Step 1: Create Bot (2 minutes)
1. Open Telegram and search for `@BotFather`
2. Send `/newbot`
3. Choose a name: `Profitron Trading Bot`
4. Choose a username: `profitron_trading_bot` (must end in 'bot')
5. **Save the bot token** you receive

### Step 2: Get Chat ID (1 minute)
1. Send any message to your bot
2. Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
3. Find your `chat_id` in the response (it's a number)

### Step 3: Add to Environment (1 minute)
Add to your `.env` file:
```bash
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=987654321
```

### Step 4: Test (1 minute)
Run the test script:
```bash
python test_telegram.py
```

## What You'll Receive

### Trading Signals
```
🟢 TRADING SIGNAL 🟢

📅 Time: 2025-01-13 10:00:00
💱 Symbol: ETHA
🎯 Action: BUY
📊 Target %: 25.00%
💰 Asset Type: equity

📈 Strategy: Cointegration
🔢 Z-Score: 2.1
📊 Position: long_spread
```

### System Status
```
🟢 SYSTEM STATUS 🟢

📅 Time: 2025-01-13 10:00:00
📊 Status: Trading system started in PAPER mode

📝 Details:
Strategy: CointegrationStrategy
Assets: ['ETHA', 'ETHV']
Polling: 1.0 hours
```

### Error Alerts
```
🚨 ERROR ALERT 🚨

📅 Time: 2025-01-13 10:00:00
❌ Error: API connection failed

📋 Context:
Main trading loop
```

## Features

- ✅ **Trading Signals**: Get notified of all buy/sell signals
- ✅ **System Status**: Startup, shutdown, and status updates
- ✅ **Error Alerts**: Immediate notification of any errors
- ✅ **Daily Summaries**: End-of-day performance reports
- ✅ **HTML Formatting**: Beautiful, readable messages
- ✅ **Emoji Support**: Visual indicators for different alert types
- ✅ **Error Handling**: Graceful fallback if bot is unavailable

## Troubleshooting

### Bot not working?
1. Check your bot token is correct
2. Make sure you sent a message to the bot first
3. Verify your chat ID is correct
4. Run `python test_telegram.py` to test

### No messages received?
1. Check your internet connection
2. Verify the bot token and chat ID
3. Make sure the bot is not blocked
4. Check if Telegram is accessible from your server

### Messages delayed?
- Telegram API has rate limits
- Messages are sent asynchronously
- Network latency can cause delays

## Security Notes

- ✅ Bot token is stored in environment variables
- ✅ No sensitive trading data in messages
- ✅ Only you can receive messages (private bot)
- ✅ Messages are sent over HTTPS

## Customization

You can customize the alert messages by editing `src/utils/telegram_alerts.py`:

- Change emoji indicators
- Modify message format
- Add new alert types
- Adjust timing and frequency

## Integration Points

The Telegram alerts are integrated into:

1. **Engine**: System startup/shutdown and error alerts
2. **Executor**: Trading signal alerts
3. **Strategy**: Can be extended for strategy-specific alerts
4. **Portfolio**: Can be added for position updates

## Next Steps

1. Set up your bot following the steps above
2. Test with `python test_telegram.py`
3. Run your trading system
4. Receive real-time alerts on your phone!

The alerts will work automatically once configured - no additional setup needed for your trading system. 