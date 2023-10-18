from .TradingStrategy import TradingStrategy
from tools.technical_indicators import rolling_sma, rolling_price
import pandas as pd

class MomentumTradingStrategy(TradingStrategy):
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
    def calculate(self):
        short_sma = rolling_sma(self.ticker, 30, self.start_date, self.end_date)
        long_sma = rolling_sma(self.ticker, 90, self.start_date, self.end_date)
        price_data = rolling_price(self.ticker, self.start_date, self.end_date)
        data = pd.merge(price_data, pd.merge(short_sma, long_sma, on='Date', how='inner'), on='Date', how='inner')

        data.loc[data['30SMA'] > data['90SMA'], 'Signal'] = 1
        data.loc[data['30SMA'] < data['90SMA'], 'Signal'] = 0

        self.data = data
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        print(self.data)
        