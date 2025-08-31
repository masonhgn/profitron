import pandas as pd
import requests




#interval: 1min, 5min, 15min, 30min, 60min

def year_intraday(ticker, year, interval):
    """
    Collects intraday stock prices for every minute of a given year.

    @PARAMS:
        - ticker (str): The stock ticker you are observing.
        - year (int): The specified year from which you are collecting data.
        - interval (str): The intraday interval (1min, 5min, 15min, 30min, 60min).
    @RETURNS:
        A pandas dataframe with the necessary data.
    """

    for month in range(1,13):
        str_month = str(month) if month > 9 else '0' + str(month)
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={ticker}&interval={interval}&month={year}-{str_month}&outputsize=full&apikey=6M8CGVK4JAOSXG24&datatype=csv'
        data = pd.read_csv(url)
        data.to_csv('data.csv',index=False)
        return data



def month_intraday(ticker, year, month, interval):
    """
    Collects intraday stock prices for every minute of a given month.

    @PARAMS:
        - ticker (str): The stock ticker you are observing.
        - year (int): The specified year from which you are collecting data.
        - month (int): The specified month from which you are collecting data.
        - interval (str): The intraday interval (1min, 5min, 15min, 30min, 60min).
    @RETURNS:
        A pandas dataframe with the necessary data.
    """

    if month < 1 or month > 12:
        print('ERROR: month_intraday() INVALID MONTH')
    str_month = str(month) if month > 9 else '0' + str(month)
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={ticker}&interval={interval}&month={year}-{str_month}&outputsize=full&apikey=6M8CGVK4JAOSXG24&datatype=csv'
    print(url)
    data = pd.read_csv(url)
    data.to_csv('data.csv',index=False)
    return data





#print(month_intraday('AAPL',2012, 4,'1min'))

print(year_intraday('MSFT', '2011', '5min'))



