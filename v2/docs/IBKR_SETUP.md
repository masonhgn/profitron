# IBKR Integration Setup Guide

This guide will help you set up Interactive Brokers (IBKR) integration with your trading engine using the Client Portal Gateway.

## Prerequisites

1. **IBKR Account**: You need a live IBKR account with paper trading enabled
2. **Client Portal Gateway**: Download and install the IBKR Client Portal Gateway
3. **Python Environment**: Your trading engine environment with required dependencies

## Step 1: Install and Run Client Portal Gateway

**IMPORTANT**: The Client Portal Gateway is a **Java application that must be running locally** for the IBKR API to work. This is not optional - it's required for all API access.

### Download and Install

1. Download the Client Portal Gateway from IBKR:
   - Go to https://www.interactivebrokers.com/en/trading/ib-api.html
   - Download "Client Portal Gateway" for your operating system
   - Extract the downloaded file

### Run the Gateway

**Before running your trading engine, you MUST start the Client Portal Gateway:**

#### On Windows:
```bash
# Navigate to the extracted folder
cd Client\ Portal\ Gateway

# Run the gateway
java -jar bin/run.bat
```

#### On macOS/Linux:
```bash
# Navigate to the extracted folder
cd Client\ Portal\ Gateway

# Make the script executable
chmod +x bin/run.sh

# Run the gateway
./bin/run.sh
```

### Verify Gateway is Running

1. Open your web browser and go to: `https://localhost:5000`
2. You should see the IBKR Client Portal Gateway interface
3. The gateway runs on `https://localhost:5000` by default

### Gateway Configuration

The gateway will prompt you to:
1. **Login with your IBKR credentials** (username/password)
2. **Enable API access** if not already enabled
3. **Configure settings** (usually defaults are fine)

### Keep Gateway Running

- **The gateway must stay running** while your trading engine is active
- If the gateway stops, your trading engine will lose connection to IBKR
- You can run the gateway in a separate terminal window
- Consider using a process manager or service to keep it running

## Step 2: Configure Environment Variables

Add the following environment variables to your `.env` file:

```bash
# IBKR Credentials
IBKR_USERNAME=your_ibkr_username
IBKR_PASSWORD=your_ibkr_password
IBKR_PAPER_ACCOUNT_ID=your_paper_account_id
IBKR_LIVE_ACCOUNT_ID=your_live_account_id
```

**Note**: 
- Use your IBKR username (not account ID) for `IBKR_USERNAME`
- Use your IBKR password for `IBKR_PASSWORD`
- Use your paper account ID (e.g., "U1234567") for `IBKR_PAPER_ACCOUNT_ID`
- Use your live account ID (e.g., "U1234567") for `IBKR_LIVE_ACCOUNT_ID`
- The system will automatically use the appropriate account based on your trading mode
- **Live trading**: Uses actual account balance from IBKR
- **Paper trading**: Uses starting cash amount from config

## Step 3: Test the Integration

**Make sure the Client Portal Gateway is running first!**

Then run the test script to verify your setup:

```bash
# Run the comprehensive test suite
PYTHONPATH=/path/to/project python -m pytest tests/test_ibkr.py -v

# Or run a specific test
PYTHONPATH=/path/to/project python -m pytest tests/test_ibkr.py::TestIBKRIntegration::test_ibkr_authentication -v
```

This will test:
- Authentication with IBKR
- Account information retrieval
- Contract discovery (symbol to conid mapping)
- Market data access
- Position and order retrieval
- Portfolio synchronization
- Order placement simulation
- Error handling
- Contract cache functionality

## Step 4: Configure Trading

**Make sure the Client Portal Gateway is running first!**

### Simple Usage

You can now run the trading engine with just a strategy config:

```bash
# Paper trading (default)
PYTHONPATH=/path/to/project python src/main.py --config configs/strategies/cointegration_etha_ethv.yaml

# Live trading
PYTHONPATH=/path/to/project python src/main.py --config configs/strategies/cointegration_etha_ethv.yaml --mode live

# Or specify the engine config directly
PYTHONPATH=/path/to/project python src/main.py --config configs/engine/Engine.yaml
```

## Configuration Files

### Engine Config

- `configs/engine/Engine.yaml` - Single consolidated engine config supporting both paper and live modes

### Strategy Configs

- `configs/strategies/cointegration_etha_ethv.yaml` - ETHA/ETHV cointegration strategy
- `configs/strategies/cointegration_btc_eth.yaml` - BTC/ETH cointegration strategy

### Usage

The engine automatically detects the type of config file:
- **Strategy config**: Automatically loads with default engine config
- **Engine config**: Uses the specified engine configuration
- **Main config**: Legacy support for main config files

## Features

### ✅ Implemented

- **Authentication**: Automatic login via Client Portal Gateway
- **Session Management**: Handles read-only and brokerage sessions
- **Contract Discovery**: Symbol to conid mapping with caching
- **Order Execution**: Market orders with automatic reply message handling
- **Position Tracking**: Real-time position monitoring
- **Market Data**: Basic market data retrieval
- **Error Handling**: Robust error handling and logging

### 🔄 Contract Cache

The system automatically caches contract IDs (conids) in `data/ibkr_contract_cache.json` to avoid repeated API calls for the same symbols.

### 📊 Order Reply Messages

IBKR may send order reply messages that require confirmation. The integration automatically confirms these messages to ensure smooth order execution.

## Troubleshooting

### Common Issues

1. **Authentication Failed**
   - **Ensure Client Portal Gateway is running** (most common issue!)
   - Verify username/password in `.env` file
   - Check that your IBKR account is active
   - Make sure you logged into the gateway with your IBKR credentials

2. **Gateway Connection Error**
   - **Verify Client Portal Gateway is running** on `https://localhost:5000`
   - Check that you can access the gateway in your browser
   - Check firewall settings
   - Ensure SSL certificates are trusted
   - Make sure Java is installed and working

3. **Contract Not Found**
   - Verify symbol is correct
   - Check that the symbol is available on IBKR
   - Try clearing the contract cache: `rm data/ibkr_contract_cache.json`

4. **Order Placement Failed**
   - Check account permissions
   - Verify sufficient funds
   - Ensure market is open for the instrument

### Debug Mode

Enable debug logging by setting the log level in your configuration:

```yaml
logging:
  level: DEBUG
```

## Security Notes

- Never commit your `.env` file to version control
- Use paper trading for testing
- Monitor your account regularly
- Keep your IBKR credentials secure

## Rate Limits

The integration is designed to stay well within IBKR's rate limits:
- 50 requests per second for direct API access
- Conservative implementation uses ~40 requests per second maximum

## Support

For IBKR-specific issues:
- Contact IBKR API Support
- Check IBKR API documentation
- Use IBKR's Client Portal Gateway support

For integration issues:
- Check the logs for detailed error messages
- Verify your configuration
- Test with the provided test script 