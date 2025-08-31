import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
import yfinance as yf
from datetime import datetime as dt


class DataCollection(object):
    # def __new__(cls):
    #     if not hasattr(cls, 'instance'):
    #         cls.instance = super(DataCollection, cls).__new__(cls)
    #         return cls.instance


    def __init__(self):
        self.dataframes: list = []


    def single_last_closing_price(self, ticker: str) -> float:
        """returns the last closing price of a single equity"""
        try:
            price = yf.Ticker(ticker)
            history = price.history(period="1d")
            return history['Close'].iloc[-1]
        except: return None

    def closing_prices(self, ticker: str, start_date: str, end_date: str) -> np.array:
        """returns a np.array of closing prices from a specific period"""
        return list(yf.download(ticker, start= start_date, end = end_date, progress = False)['Close'])

    def closing_prices_multi(self,tickers: list[str], start_date: str, end_date: str) -> dict:
        result = {}
        for ticker in tickers:
            prices = self.closing_prices(ticker, start_date, end_date)
            if len(prices): result[ticker] = prices
        return result

    
    def ohlcv(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """returns pd.DataFrame of open, high, low, close and volume for a specific period, for a single stock"""
        return yf.download(ticker, start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), progress=False)


    def ohlcv_multi(self, tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
        """returns dict of pd.DataFrames of open, high, low, close and volume for a specific period, for multiple stocks"""
        result = {}
        for ticker in tickers:
            df = yf.download(ticker, start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), progress=False)
            result[ticker] = df
        return result



    def all_sp_500_tickers(self) -> list[str]:
        """ gathers all S&P 500 tickers from wikipedia page on S&P 500"""
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        table = pd.read_html(url)
        df = table[0]
        tickers = df['Symbol'].tolist()
        return tickers



    def get_risk_free_rate(self):
        #ten year treasury yield
        tnx = yf.Ticker("^TNX")
        
        tnx_data = tnx.history(period="1d")
    
        risk_free_rate = tnx_data['Close'].iloc[-1] / 100  # Convert to a decimal rate
        
        #print(f"Risk-Free Rate (10-Year Treasury Yield): {risk_free_rate:.4%}")
        return risk_free_rate




    def ishares_energy_tickers(self) -> list[str]:
        return ["XOM", "CVX", "SHEL", "TTE", "COP", "BP.", "CNQ", "ENB", "EOG", "SLB", "MPC", "PSX", "VLO", "SU", "WMB", "OKE", "HES", "OXY", "TRP", "KMI", "ENI", "WDS", "FANG", "HAL", "BKR", "PBRA", "DVN", "EQNR", "PBR", "CVE", "TRGP", "CCO", "CTRA", "857", "PPL", "REP", "EQT", "STO", "TOU", "5020", "MRO", "1605", "IMO", "APA", "NESTE", "GALP", "TEN", "AKRBP", "OMV", "ALD", "COPEC", "EC"]



    def filter_last_closing_price(self, limit: float, tickers: list[str], lower: bool = True) -> list[str]:
        """filters list of tickers by closing price"""
        result = []
        for ticker in tickers:
            price = self.single_last_closing_price(ticker)
            if price: #if price was retrieved properly
                if lower: #if we want to filter for prices lower than the threshold
                    if price <= limit: result.append(ticker) 
                else:
                    if price >= limit: result.append(ticker)
        return result


    def all_nyse_tickers(self, filename='nyse_tickers.csv'):
        """get all NYSE tickers"""
        try:
            df = pd.read_csv(filename)
            tickers = df['ACT Symbol'].tolist()
            return tickers
        except Exception as e:
            print(f"An error occurred while reading the CSV file: {e}")
            return []


    def log_returns(self, prices):
        prices = np.array(prices)
        price_ratios = prices[1:] / prices[:-1]
        log_returns = np.log(price_ratios)
        return log_returns


    def write_to_txt_file(self, tickers, file_name):
        try:
            with open(file_name, 'w') as f:
                for ticker in tickers:
                    f.write(f"{ticker}\n")
            print('wrote tickers to file ' + file_name)
        except Exception as e:
            print(e)

    def read_from_txt_file(self, file_name):
        try:
            with open(file_name, 'r') as f:
                tickers = f.read().splitlines()
            print(f"read from '{file_name}'.")
            return tickers
        except Exception as e:
            print(f"error occurred while reading the file: {e}")
            return []
    



