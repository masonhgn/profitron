import pandas as pd
import pandas_datareader as pdr
import numpy as np
import yfinance as yf
import datetime
from DataCollection import DataCollection
from Analysis import Analysis
from ApiBridge import ApiBridge


'''
STATISTICAL ARBITRAGE PAIRS TRADING STRATEGY

SOME KEY INFO:

lookback = period of time in which we use to calculate hedge ratio, zscore and spreads

the lookback window needs to have a lookback window as well because:
	- to calculate spread of a single day, we need to calculate hedge ratio, which
	is based on the lookback period until that day. So for the first day of a 20 day
	lookback period, we need to get the lookback of 20 days before that. so we need a
	look back of lookback * 2


STRATEGY:

for each pair:

	1. get 60 days of historical data for both tickers

	2. calculate hedge ratio and spread for last 21 days

	3. calculate zscore based on spreads

	4.  if we are short the spread for this pair and zscore < 0, exit

        elif we are long the spread for this pair and zscore > 0, exit

        elif we are not long and zscore < -1, enter long position

        elif we are not short and zscore > 1 enter short position

'''


class StatArb(object):

    def __init__(self):
        self.universe: list = ['XOM','CVX']
        self.pairs = [['XOM','CVX']]
        self.frequency: str = '3d'
        self.next_trading_date = datetime.datetime.now()
        self.data = DataCollection()
        self.analysis = Analysis()
        self.api = ApiBridge(production=False)
        self.lookback = 20
        

    def rebalance(self) -> None:
        """rebalances portfolio, calls generate_signal"""
        pass

    

    
    def generate_signal(self, date: datetime.date = datetime.date.today(), last_signal: int = 0) -> dict:
        """core of trading strategy, generates signals for every ticker in universe"""
        
        #gather historical data needed to create current day's signal
        historical = self.data.closing_prices_multi(self.universe, date - datetime.timedelta(days=self.lookback*3), date)
        hedge_ratios = []
        for i in range(len(self.pairs)):
            X, Y = historical[self.pairs[i][0]], historical[self.pairs[i][1]]

            assert len(X) == len(Y)

            for j in range(-self.lookback, 0): hedge_ratios.append(self.analysis.hedge_ratio(X[j-self.lookback:j],Y[j-self.lookback:j])[0][1])

            X, Y, hedge_ratios = np.array(X[-20:]), np.array(Y[-20:]), np.array(hedge_ratios)

            spreads = Y - hedge_ratios * X

            cur_spread = spreads[-1]

            zscore = (cur_spread - spreads.mean()) / spreads.std()

            if last_signal == -1 and zscore < 0: #if we are currently shorting the spread and the spread is < historical average, exit
                signal = 0
            elif last_signal == 1 and zscore > 0: #if we are currently long the spread and the spread is > historical average, exit
                signal = 0
            elif last_signal != 1 and zscore < -1: #if we are not long the spread and spread is >= 1 std dev. below historical average, enter long poisition
                signal = 1
            elif last_signal != -1 and zscore > 1: #if we are not short and the spread is >= 1 std dev. above historical average, enter short position
                signal = -1

            return signal



                
        


    def backtest(self, start_date: str, end_date: str) -> dict:
        """backtests the trading strategy by calling generate_signal() from start_date to end_date with respect to self.frequency"""
        pass





if __name__ == "__main__":
    arb = StatArb()
    print(datetime.datetime.now())
    arb.generate_signal()