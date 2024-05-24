import pandas as pd
import pandas_datareader as pdr
import numpy as np
import yfinance as yf
from datetime import datetime as dt


'''
STATISTICAL ARBITRAGE PAIRS TRADING STRATEGY

SOME KEY INFO:

lookback = period of time in which we use to calculate hedge ratio, zscore and spreads

the lookback window needs to have a lookback window as well because:
	- to calculate spread of a single day, we need to calculate hedge ratio, which
	is based on the lookback period until that day. So for the first day of a 20 day
	lookback period, we need to get the lookback of 20 days before that. so we need a
	look back of lookback * 2


STRATEGY:

for each pair:

	1. get 60 days of historical data for both tickers

	2. calculate hedge ratio and spread for last 21 days

	3. calculate zscore based on spreads

	4.  if we are short the spread for this pair and zscore < 0, exit

        elif we are long the spread for this pair and zscore > 0, exit

        elif we are not long and zscore < -1, enter long position

        elif we are not short and zscore > 1 enter short position

'''


class StatArb(object):

    def __init__(self):
        self.universe: list = ['XOM','CVX',]
        self.frequency: str = '3d'
        self.next_trading_date = dt.now()

    def rebalance(self) -> None:
        """rebalances portfolio, calls generate_signal"""
        pass

    

    
    def generate_signal(self) -> dict:
        """core of trading strategy, generates signals for every ticker in universe"""
        pass


    def backtest(self, start_date: str, end_date: str) -> dict:
        """backtests the trading strategy by calling generate_signal() from start_date to end_date with respect to self.frequency"""
        pass

    
