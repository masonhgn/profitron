
import pandas as pd
import pandas_datareader as pdr
import numpy as np
import yfinance as yf
from datetime import datetime as dt
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

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
        price = last_closing_price(ticker)
        if price is not None and price <= limit:
            results.append(ticker)
    return results


def prices_df(tickers, period):
    """creates pandas dataframe for prices from a specific period"""
    prices = []
    for t in tickers:
        temp = yf.download(t,start=dt.now() - pd.Timedelta(days=period), end = dt.now(), progress = False)
        temp[t] = np.log(temp['Adj Close'] / temp['Adj Close'].shift(1))
        prices.append(temp)
    df = pd.concat(prices, axis=1).dropna()
    return df

def plot_df(df, ticker_x, ticker_y):
    plt.figure(figsize = (10, 6))
    plt.rcParams.update({'font.size': 14})
    
    # Scatter plot
    plt.scatter(df[ticker_x], df[ticker_y])
    
    # Best fit line
    m, b = np.polyfit(df[ticker_x], df[ticker_y], 1)  # 1 is the degree of the polynomial (straight line)
    plt.plot(df[ticker_x], m*df[ticker_x] + b, color='red')
    
    # Display equation on the plot
    equation = f"y = {m:.2f}x + {b:.2f}"
    plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    
    plt.xlabel(f"{ticker_x} returns")
    plt.ylabel(f"{ticker_y} returns")
    plt.show()

    return m, b


def generate_data(*args, ticker, period):
    merged_df = pd.DataFrame(columns=['Date'])
    for func in args:
        df = func(ticker, period)
        merged_df = pd.concat([merged_df, df], ignore_index = True)

    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    
    merged_df = merged_df.sort_values(by='Date')

    merged_df = merged_df.reset_index(drop=True)

    return merged_df







