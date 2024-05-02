import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta



def calculate_rsi(symbol, period=14):
    #get historical data for underlying security
    stock = yf.Ticker(symbol)
    history = stock.history(period=f"{period+1}d")

    #get closing prices from dataframe
    prices = history['Close'].to_list()

    #price change per day
    changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]

    #separate gains and losses into separate lists
    gains = [change if change > 0 else 0 for change in changes]
    losses = [-change if change < 0 else 0 for change in changes]

    #average it out over the period
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    if avg_loss == 0:
        return 100 if avg_gain > 0 else 50  # Max RSI value is 100 if there are no losses; 50 indicates a neutral value

    # Calculate the running average
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:  # Check again while looping through
            return 100 if avg_gain > 0 else 50

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi




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



def rolling_price(ticker, start, end):
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    
    # download data
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # get open/close prices
    open_prices = data['Open']
    close_prices = data['Close']
    
    # calculate return  based on open and close price
    returns = (close_prices / open_prices) - 1.0
    
    #add to result dataframe
    data['Open'] = open_prices
    data['Close'] = close_prices
    data['Change'] = returns
    
    # drop NaN values
    data.dropna(inplace=True)
    
    return data[['Open', 'Close', 'Change']]






def rolling_sma(ticker, period, start, end):
    '''
    RETURNS: a rolling simple moving average
    PARAMS:
        - ticker: the stock ticker you want to get the simple moving average for (i.e. 'AAPL')
        - period: moving average period, (i.e. 5 would provide a 5 day simple moving avg.)
        - start, end: start date, end date (i.e. "2023-10-01")
    '''
    #convert to pd datetime objects for use with dataframe
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)

    title = str(period) + 'SMA'

    #this will be what we return
    result_df = pd.DataFrame(columns=['Date', title])

    #we need this to precisely get the moving average backdate
    window_df = yf.download(ticker, start=start_date, end=end_date)

    # get dataframe with backdate to create moving average
    increment = 0
    df = yf.download(ticker, start=start_date - pd.DateOffset(days=period + increment), end=end_date)

    while (df.shape[0] < window_df.shape[0]+period): #the original backdate is off because we aren't accounting for weekends/non trading days
        increment += 1 #increment number of days until start day is exactly start_date - period
        df = yf.download(ticker, start=start_date - pd.DateOffset(days=period + increment), end=end_date) #fetch new dataframe

    for date in range(period-1,df.shape[0]): #amount of total days - period = the window we provided
        moving_average = round(df.iloc[date-period+1:date+1]['Close'].mean(),2)
        daily_result = pd.DataFrame({'Date': [df.index[date]], title: [moving_average]})
        result_df = pd.concat([result_df, daily_result], ignore_index=True)
    return result_df






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




#def adf_stationarity(ticker, start_date, end_date)
