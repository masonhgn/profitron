import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def basic_correlation(stock1, stock2, start_date=datetime.now()-timedelta(days=30), end_date = datetime.now()):

    data = yf.download([stock1, stock2], start=start_date, end=end_date)['Adj Close']

    price1 = data[stock1]
    price2 = data[stock2]


    #percent change per day, with null values removed from the dataframe
    returns1 = price1.pct_change().dropna()
    returns2 = price2.pct_change().dropna()

    #concatenate returns dataframes
    returns_df = pd.concat([returns1, returns2], axis=1)

    #get correlation coefficient
    correlation = returns_df.corr().iloc[0, 1]

    #convert to percentage
    similarity_percentage = abs(correlation) * 100

    return similarity_percentage



def bollinger_bands(ticker, period=20, stdev=2):
    stock = yf.Ticker(ticker)
    history = stock.history(period=f"{period}d")

    #get rolling mean and std. deviation
    prices = history['Close']
    rolling_mean = prices.rolling(window=period).mean()
    rolling_std = prices.rolling(window=period).std()

    #upper and lower bands calculated by adding/subtracting (stdev) standard devations from rolling mean
    upper_band = rolling_mean + (rolling_std * stdev)
    lower_band = rolling_mean - (rolling_std * stdev)

    #get last price
    current_price = prices[-1]
    
    return [current_price, lower_band[-1], upper_band[-1]]
