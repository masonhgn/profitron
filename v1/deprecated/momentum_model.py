import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import collect_sp_500_data
import os.path
import json
tickers_file = "tickers.txt"

def open_tickers_file(tickers_file_arg):
    with open(tickers_file_arg, 'r') as file:
        tickers_list = file.read().splitlines()
        return tickers_list


#uses my special high volatility list of tickers
#tickers = open_tickers_file(tickers_file)



#uses the s&p 500
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
            

def create_file(file_type_name, map_func):
    today_file_name = datetime.now().strftime("%Y-%m-%d_") + file_type_name +  '.json'
    if os.path.isfile(today_file_name):
        with open(today_file_name, 'r') as json_file:
            retrieved_data = json.load(json_file)
            return retrieved_data
    else:
        print('Up to date file not found. Creating new one...')
        new_map = map_func()
        with open(today_file_name, 'w') as json_file:
            json.dump(new_map, json_file)
        return new_map


sp_500_momentums = create_file('sp_500_momentums', create_10_day_momentum_map)
sp_500_momentums = sorted(sp_500_momentums.items(), key=lambda x:x[1], reverse = True)


def get_top_n_percent(map, n):
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



top_10_list = create_file(get_top_n_percent(sp_500_momentums, 0.1))
for item in top_10_list:
    print(item)

