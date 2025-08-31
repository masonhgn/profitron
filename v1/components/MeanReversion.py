import pandas as pd
import pandas_datareader as pdr
import numpy as np
import yfinance as yf
import datetime
from components.DataCollection import DataCollection
from components.Analysis import Analysis
from components.ApiBridge import ApiBridge
import random
from Portfolio import Portfolio

'''
MEAN REVERSION TRADING STRATEGY


STRATEGY:

1. FILTERING:
    for each security in our investment universe, we filter by these criteria:
        a. price <= $75
        b. log normal returns pass the adf stationarity test (lookback defined previously, pvalue=0.05)
        c. price <= X standard deviations below Y day mean
        d. volatility level >= Z
        e. daily liquidity >= P

2. TRADING:
    rebalance portfolio with a basket of Q securities, equally weighted
    (weight by volatility later)

rebalance portfolio every week

'''


class MeanReversion(object):
    def __init__(self):
        self.rebalance_period = 14 #how many days in between rebalancing portfolio
        self.price_limit = 75 #1a highest cap stock we are willing to buy
        self.lookback = 30 #lookback period
        self.z_score_threshold = -2 #how many standard deviations below the mean price will trigger a buy signal
        self.volatility_threshold = 0.05 #minimum volatility required
        self.liquidity_threshold = 1 #minimum liquidity required
        self.basket_size = 10 #diversity

        self.next_trading_date = datetime.datetime.now()
        self.data = DataCollection()
        self.analysis = Analysis()
        self.api = ApiBridge(production=False)

        self.universe = self.data.read_from_txt_file('low_cap_stationary_tickers_2024-09-14')
        #self.universe = self.data.read_from_txt_file('low_cap2')
        #self.universe = random.sample(self.universe, 100)


    def rebalance(self) -> None:
        """rebalances portfolio by calling generate_signal for the current date"""
        '''
            1. liquidate entire portfolio
            2. call generate signal for current date
            3. buy securities based on the baskets
        '''
        pass

    def generate_basket(self, price_data: dict) -> list:
        '''
        @params
        - price_data: dictionary with k = ticker and v = prices

        returns:
        - a list of tickers
        '''
        basket = {}


        #get last closing price and determine if we are <= (mean - 1 std dev.)
        for ticker, prices in price_data.items():
            mean_price = np.mean(prices)
            volatility = np.std(self.data.log_returns(prices))
            target_price = mean_price - (np.std(prices)*1)
            z_score = (prices[-1] - mean_price) / np.std(prices)

            if prices[-1] <= target_price:
                basket[ticker] = {
                    'volatility': volatility,
                    'z_score': z_score, 
                    'current_price': prices[-1]
                }

        basket = sorted(basket.items(), key = lambda x: x[1]['volatility'], reverse=True)
        return basket


    
    def backtest(self, start_date: datetime.date, end_date: datetime.date) -> dict:

        # get start_date - lookback
        lookback_dates = pd.bdate_range(end=start_date, periods=self.lookback + 1)[:-1]  # Exclude start_date itself
        # #get start_date to end_date
        common_dates = pd.bdate_range(start=start_date, end=end_date, freq='B')

        # combine lookback_dates with common_dates
        all_dates = lookback_dates.append(common_dates)
        #all_dates = [date.date() for date in all_dates]

        historical_data = self.data.ohlcv_multi(self.universe, all_dates[0], all_dates[-1]) #get a ohlcv DF for each ticker from start_date to end_date

        min_size = max(len(df) for df in historical_data.values())
        historical_data = {ticker: df for ticker, df in historical_data.items() if len(df) >= min_size}

        portfolio = Portfolio(100000)


        #go through every date in the backtest. we have dates defined for start_date - lookback
        rebalance_counter = 0
        for i in range(self.lookback, len(all_dates)):
            current_date = all_dates[i]

            current_date_as_timestamp = pd.Timestamp(current_date)

            """######## IF IT IS REBALANCE DAY ########"""
            ###########################################################

            if rebalance_counter == 0 or (rebalance_counter >= self.rebalance_period and rebalance_counter % self.rebalance_period == 0): #if it's time to rebalance
                #rebalance portfolio
                print('rebalancing on date ' + str(current_date))

                price_data = {}
                #gather price info from current_date - lookback to current_date
                for ticker, df in historical_data.items():
                    #avoid errors by trying to fetch data that isn't there

                    prices = df['Adj Close'].loc[:current_date].tail(self.lookback).tolist() #get the last (lookback) days of prices for this security
                    price_data[ticker] = prices                    

                #generate basket for that price data
                basket = self.generate_basket(price_data)
                #get the top basket_size securities from the basket
                basket = basket[:self.basket_size]

                if len(basket) == 0:
                    print('length of selection is zero')
                    continue


                #this is just reformatting the basket so that the portfolio can understand it
                basket_arg = {ticker: details['current_price'] for ticker, details in basket}

                #liquidate entire portfolio to buy new basket
                portfolio.liquidate()

                #buy securities in newly generated basket
                portfolio.equal_weight_allocation(basket_arg)


            ###########################################################


            #update portfolio
            for ticker in portfolio.get_tickers_owned():

                print('updating '+ ticker + ' value...')

                if current_date_as_timestamp in df.index:
                    row = historical_data[ticker].loc[current_date_as_timestamp]


                    updated_price = row['Adj Close']

                    print('old share price: ' + str(portfolio.holdings[ticker]['current_price']) + ', new price: ' + str(updated_price))
                    portfolio.update_price(ticker, updated_price)

            
            if portfolio.balance == 0:
                print('BLEW ACCOUNT')
                return

            #save state of portfolio
            portfolio.save_state(current_date)
            print('current value: ' + str(portfolio.get_total_value()))

            #update rebalance counter every iteration so we rebalance on the right dates
            rebalance_counter += 1

        rfr = self.data.get_risk_free_rate()
        results = portfolio.calculate_metrics(rfr)
        
        return results