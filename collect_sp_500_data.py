import pandas as pd
import pandas_datareader as pdr

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def collect_data():
    """ gathers all S&P 500 tickers from wikipedia page listed above"""
    table = pd.read_html(url)
    df = table[0]
    tickers = df['Symbol'].tolist()
    return tickers




