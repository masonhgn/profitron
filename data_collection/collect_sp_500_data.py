import pandas as pd
import pandas_datareader as pdr

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def collect_sp_500_tickers():
    """ gathers all S&P 500 tickers from wikipedia page listed above"""
    table = pd.read_html(url)
    df = table[0]
    tickers = df['Symbol'].tolist()
    return tickers





import yfinance as yf

def get_last_closing_price(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        history = stock.history(period="1d")
        last_closing_price = history['Close'].iloc[-1]
        return last_closing_price
    except:
        return None



def collect_tickers_with_limit(limit):
    tickers = collect_sp_500_tickers()
    results = []
    for ticker in tickers:
        price = get_last_closing_price(ticker)
        if price is not None and price <= limit:
            results.append(ticker)
    return results
