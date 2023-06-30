import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def basic_correlation(start_date=datetime.now()-timedelta(days=30), end_date = datetime.now(), stock1, stock2):

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
