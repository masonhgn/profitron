from trading_strategies import TradingStrategy
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from data_collection.collect_sp_500_data import collect_tickers_with_limit
import os.path
import json



class TopTenMomentum(TradingStrategy.TradingStrategy):
    def __init__(self):
        TradingStrategy.TradingStrategy.__init__(self)

    def generate_portfolio(self):

        tickers = collect_tickers_with_limit(30)

        #print('tickers: ',tickers)
    
        def create_10_day_momentum_map():
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
                  
            momentum_map = {}
            for ticker in tickers:
                #get the last 10 trading days of closing prices
                data = yf.download(ticker, start=start_date, end=end_date)
                #print(ticker, data)
                if len(data) < 10:
                    print('length of data < 10')
                    continue
                last_10_days = data['Close'].values.tolist()[-10:]
                momentum = (last_10_days[-1] - last_10_days[0]) / last_10_days[-1]
                #print('momentum: ',momentum)
                momentum_map[ticker] = momentum
            return momentum_map

        print('10 day momentum: ', create_10_day_momentum_map())

        sp_500_momentums = self.create_map_file('sp_500_momentums', create_10_day_momentum_map)
        sp_500_momentums = sorted(sp_500_momentums.items(), key=lambda x:x[1], reverse = True) #sorts by value in decreasing order

        top_10_list = self.get_top_n_percent(sp_500_momentums, 0.10)
        self.save_map_file('generated_portfolio',top_10_list)
        print('top_ten_list: ')
        print(top_10_list)
        self.portfolio = top_10_list

