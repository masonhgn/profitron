from components.DataCollection import DataCollection
from components.Analysis import Analysis
import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
import yfinance as yf
from datetime import datetime as dt
import json
from matplotlib import pyplot as plt


if __name__ == "__main__":
    analysis,data_collection  = Analysis(), DataCollection()
    #tickers = data_collection.all_sp_500_tickers()
    tickers = data_collection.ishares_energy_tickers()
    
    data = {}

    with open('energy_prices.json') as json_file: data = json.load(json_file)

    # for i in range(len(tickers)):
    #     print('getting price data for '+tickers[i] +', '+str(i/len(tickers)*100)+'%')
    #     prices = data_collection.closing_prices(tickers[i],'2024-01-22','2024-05-22')
    #     if len(prices): data[tickers[i]] = prices

    #with open("energy_prices.json", "w") as outfile: json.dump(data, outfile)

    pairs = analysis.filter_pairs([analysis.check_cointegration_pv, analysis.delta_stationarity_pv],data,0.98)


    print(pairs)

    #a,b = data_collection.closing_prices('XOM','2024-01-22','2024-05-22'),data_collection.closing_prices('CVX','2024-01-22','2024-05-22')
    #a, b = [1,2,3,4,5],[2,4,6,8,10]
    #print(analysis.delta_stationarity(a,b))
    