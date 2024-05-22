import pandas as pd
import pandas_datareader as pdr
import numpy as np
import yfinance as yf
from datetime import datetime as dt


class DataCollection(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(DataCollection, cls).__new__(cls)
            return cls.instance


    def __init__(self):
        self.dataframes: list = []


    def single_last_closing_price(self, ticker: str) -> float:
        """returns the last closing price of a single equity """
        try:
            price = yf.Ticker(ticker)
            history = price.history(period="1d")
            return history['Close'].iloc[-1]
        except: return None

    def closing_prices(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """returns closing prices from a specific period"""
        return pdr.DataReader(ticker, 'yahoo', pd.to_datetime(start_date), pd.to_datetime(end_date))

    
    def ohlcv(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """returns pd.DataFrame of open, high, low, close and volume for a specific period, for a single stock"""
        return yf.download(ticker, start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), progress=False)


    def all_sp_500_tickers(self) -> list[str]:
        """ gathers all S&P 500 tickers from wikipedia page on S&P 500"""
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        table = pd.read_html(url)
        df = table[0]
        tickers = df['Symbol'].tolist()
        return tickers


    def filter_last_closing_price(self, limit: float, tickers: list[str], lower: bool = True) -> list[str]:
        """filters list of tickers by closing price"""
        result = []
        for ticker in tickers:
            price = single_last_closing_price(ticker)
            if price: #if price was retrieved properly
                if lower: #if we want to filter for prices lower than the threshold
                    if price <= limit: result.append(ticker) 
                else:
                    if price >= limit: result.append(ticker)
        return result



    

    

    


    


    
    



