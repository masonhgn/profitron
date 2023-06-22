from trading_strategies import TradingStrategy
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from data_collection import collect_sp_500_data
import os.path
import json



class TopTenMomentum(TradingStrategy.TradingStrategy):
    def __init__(self):
        TradingStrategy.__init__(self)

    def generate_portfolio(self):

        tickers = collect_sp_500_data.collect_data()


        def create_10_day_momentum_map():
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")

            momentum_map = {}
            for ticker in tickers:
                #get the last 10 trading days of closing prices
                data = yf.download(ticker, start=start_date, end=end_date)
                if len(data) < 10: continue
                last_10_days = data['Close'].values.tolist()[-10:]
                momentum = (last_10_days[-1] - last_10_days[0]) / last_10_days[-1]
                momentum_map[ticker] = momentum
            return momentum_map

        sp_500_momentums = self.create_map_file('sp_500_momentums', create_10_day_momentum_map)
        sp_500_momentums = sorted(sp_500_momentums.items(), key=lambda x:x[1], reverse = True)

        top_10_list = self.get_top_n_percent(sp_500_momentums, 0.02)
        self.save_map_file('generated_portfolio',top_10_list)

        self.portfolio = top_10_list

