import yfinance as yf

def calculate_rsi(symbol, period=14):
    #get historical data for underlying security
    stock = yf.Ticker(symbol)
    history = stock.history(period=f"{period+1}d")

    #get closing prices from dataframe
    prices = history['Close'].to_list()



    #price change per day
    changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    
    #separate gains and losses into separate dataframes
    gains = [change if change > 0 else 0 for change in changes]
    losses = [-change if change < 0 else 0 for change in changes]


    #average it out over the period
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period


    #idk??
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

