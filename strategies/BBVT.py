
from .TradingStrategy import TradingStrategy
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from data_collection.data_collection import *
from tools.technical_indicators import rolling_sma

class BBVT(TradingStrategy):
    """
    
    Strategy: Bollinger Bands Volatility Threshold
    
    We will go long on stocks which cross 
    
    """
    def __init__(self):
        self.prices = None

    def calculate(self):
        vmp = 1 #volatility multiplier
        self.universe = ['COIN']

        for ticker in self.universe:

            prices = prices_df([ticker],365)
            
            prices['Log Return'] = np.log(prices['Close'] / prices['Close'].shift(1))
            daily_volatility = prices['Log Return'].std()

            prices['15SMA'] = prices['Close'].rolling(window=15).mean()
            prices['25SMA'] = prices['Close'].rolling(window=25).mean()
            prices['Lower'] = prices['15SMA'] * (1 - daily_volatility * vmp)
            prices['Upper'] = prices['15SMA'] * (1 + daily_volatility * vmp)
            prices['Signal'] = np.where(prices['Close'] < prices['Lower'], 1, np.where(prices['Close'] > prices['Upper'], -1, 0))
            prices = prices.dropna()
            df = pd.DataFrame(prices)
            #df['Date'] = pd.to_datetime(df['Date'])
            #df.set_index('Date', inplace=True)
            plt.figure(figsize=(10, 5))
            plt.plot(df.index, df['Close'], label='Close Price', color='blue', marker='.')
            plt.plot(df.index, df['15SMA'], label='15-day MA', color='red')
            plt.plot(df.index, df['25SMA'], label='25-day MA', color='cyan')
            #plt.plot(df.index, df['Upper'], label='Upper Bollinger Band', linestyle='--', color='red')
            #plt.plot(df.index, df['Lower'], label='Lower Bollinger Band', linestyle='--', color='red')

            # Highlighting signals
            # for i in df.index[df['Signal'] == 1]:
            #     plt.scatter(i, df.loc[i, 'Close'], color='gold', marker='^', s=100)  # Buy signal
            # for i in df.index[df['Signal'] == -1]:
            #     plt.scatter(i, df.loc[i, 'Close'], color='black', marker='v', s=100)  # Sell signal

            plt.title('Bollinger Bands and Trading Signals')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.show()

        print(prices)


