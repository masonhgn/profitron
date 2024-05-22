import pandas as pd
import pandas_datareader as pdr
import numpy as np
import yfinance as yf
from datetime import datetime as dt



class StatArb(object):

    def __init__(self):
        self.universe: list = []
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

    
