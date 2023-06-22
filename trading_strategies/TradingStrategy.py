import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from data_collection import collect_sp_500_data
import os.path
import json



class TradingStrategy:

    def __init__(self):
        self.portfolio = dict()

    def file_to_map(self, file_name):
        if os.path.isfile('data/'+file_name):
            with open('data/'+file_name, 'r') as json_file:
                retrieved_data = json.load(json_file)
                return retrieved_data
        else:
            print('file not found during calling of file_to_map() function. aborting.')

    def save_map_file(self, file_type_name, map_obj):
        today_file_name = datetime.now().strftime("%Y-%m-%d_") + file_type_name +  '.json'
        if os.path.isfile('data/'+today_file_name):
            print('file already exists. please just use that.')
            return
        else:
            print('Up to date file not found. Creating new one...')
            with open('data/'+today_file_name, 'w') as json_file:
                json.dump(map_obj, json_file)


    def create_map_file(self, file_type_name, map_func):
        today_file_name = datetime.now().strftime("%Y-%m-%d_") + file_type_name +  '.json'
        if os.path.isfile('data/'+today_file_name):
            with open('data/'+today_file_name, 'r') as json_file:
                retrieved_data = json.load(json_file)
                return retrieved_data
        else:
            print('Up to date file not found. Creating new one...')
            new_map = map_func()
            with open('data/'+today_file_name, 'w') as json_file:
                json.dump(new_map, json_file)
            return new_map
    
    def generate_trade_signals(self):
        #take the current generated portfolio and yesterday's generated portfolio and generate trade signals
        today_file_name = datetime.now().strftime("%Y-%m-%d") +  '_generated_portfolio.json'
        date_ptr = today_file_name
        date_found = False
        for i in range(1,7):
            date_ptr = (datetime.now()- timedelta(days=i)).strftime("%Y-%m-%d") +  '_generated_portfolio.json'
            if os.path.isfile('data/'+date_ptr):
                date_found=True
                break
        if not date_found:
            print('could not find previous portfolio files to generate trade signals. aborting generate_trade_signals() function.')
            return

        buy_tickers, sell_tickers = [], []
        last_portfolio = self.file_to_map(date_ptr)
        today_portfolio = self.file_to_map(today_file_name)
        
        trades = []

        for item in last_portfolio:
            if not item in today_portfolio:
                trades.append('SELL ' + item[0])
        for item in today_portfolio:
            if not item in last_portfolio:
                trades.append('BUY ' + item[0])

        return trades
        

    def get_top_n_percent(self, map, n):
        if n > 0.99 or n < 0.01:
            print('cannot get '+n+'% of numbers. please input a different value for n (0.01 <= n <= 0.99')
            return
        if len(map) < 1:
            print('map passed into get_top_n_percent is empty')
            return
        result = []
        num_securities = int(len(map) * n)
        count = 1
        for item in map:
            if count > num_securities: break
            result.append(item)
            count += 1
        return result
    def print_portfolio(self):
        for item in self.portfolio:
            print(item[0])
