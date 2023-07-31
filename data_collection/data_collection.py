
import pandas as pd
import pandas_datareader as pdr


def collect_sp_500_tickers():
    """ gathers all S&P 500 tickers from wikipedia page on S&P 500"""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)
    df = table[0]
    tickers = df['Symbol'].tolist()
    return tickers



def collect_all_tickers():
    """gathers all tickers from nasdaq, nyse and amex using provided csv downloaded from nasdaq website"""
    csv_file = "data_collection/nasdaq_info.csv"
    df = pd.read_csv(csv_file)
    tickers = df.iloc[:, 0].tolist()
    return tickers


def last_closing_price(ticker):
    """returns last closing price of a security"""
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period="1d")
        last_closing_price = history['Close'].iloc[-1]
        return last_closing_price
    except:
        return None


def price_filter(limit, tickers):
    """filters out a list of tickers with a price limit using last closing price. (i.e. from these 500 stocks return all that are < $50)"""
    results = []
    for ticker in tickers:
        price = get_last_closing_price(ticker)
        if price is not None and price <= limit:
            results.append(ticker)
    return results
