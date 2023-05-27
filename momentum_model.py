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


def calculate_momentum():
    today_file_name = datetime.now().strftime("%Y-%m-%d") + '_momentum_map.json'
    if os.path.isfile(today_file_name):
        with open(today_file_name, 'r') as json_file:
            retrieved_data = json.load(json_file)
            return retrieved_data
    else:
        print('Up to date file of S&P500 momentums not found. Creating new one...')
        new_map = create_10_day_momentum_map()
        with open(today_file_name, 'w') as json_file:
            json.dump(new_map, json_file)
        return new_map


#price_map = create_10_day_momentum_map()
#price_map = sorted(price_map.items(), key=lambda x:x[1], reverse = True)

#for item in price_map:
#    print(item)

price_map = calculate_momentum()
price_map = sorted(price_map.items(), key=lambda x:x[1], reverse = True)

def get_top_10_percent():
    result = []
    num_securities = int(len(price_map) / 10)
    count = 1
    for item in price_map:
        if count > 10: break
        result.append(item)
        count += 1
    return result

top_ten_list = get_top_10_percent()
for item in top_ten_list:
    print(item)












