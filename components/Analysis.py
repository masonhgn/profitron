import pandas as pd
import pandas_datareader as pdr
import numpy as np
import yfinance as yf
from datetime import datetime as dt
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from matplotlib import pyplot as plt


class Analysis(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Analysis, cls).__new__(cls)
            return cls.instance


    def __init__(self):
        self.dataframes: list = []


    def delta_stationarity(self, series1: np.array, series2: np.array) -> list:
        """gets stationarity of spread, returns entire statistical results"""
        series1, series2 = np.array(series1), np.array(series2)
        adjusted = sm.OLS(series2,sm.add_constant(series1)).fit()

        intercept = adjusted.params[0]
        slope = adjusted.params[1]

        spread = series2 - (intercept + slope * series1)

        # plt.plot(series1)
        # plt.plot(series2)
        # plt.plot(spread)
        # plt.show()

        #spread -= np.mean(spread)
        return adfuller(spread)


    def delta_stationarity_pv(self, series1: np.array, series2: np.array) -> float:
        """gets stationarity of spread, returns p-value only"""
        return self.delta_stationarity(series1,series2)[1]





    def check_cointegration(self, series1: np.array, series2: np.array) -> list:
        """checks cointegration between to series, returns entire statistical results"""
        return coint(series1,series2)


    def check_cointegration_pv(self, series1: np.array, series2: np.array) -> float:
        """checks cointegration between to series, returns p-value only"""
        return self.check_cointegration(series1,series2)[1]






    def find_cointegrated_pairs(self, data, ci) -> list:
        """takes in dict[k=ticker,v=series] and returns a list of likely cointegrated pairs based on provided confidence interval ci"""
        n = len(data)
        scores, pvalues, pairs = np.zeros((n,n)), np.ones((n,n)), []
        keys = list(data.keys())
        count = 0
        for i in range(n):
            for j in range(i+1,n):
                count += 1
                s1, s2 = data[keys[i]][:min(len(data[keys[i]]),len(data[keys[j]]))], data[keys[j]][:min(len(data[keys[i]]),len(data[keys[j]]))]
                print(str(round((count/(n**2))*100,2)) + '%')
                result = coint(s1,s2)
                scores[i,j], pvalues[i,j] = result[0], result[1]
                if result[1] < 1 - ci: pairs.append((keys[i],keys[j]))

        return scores, pvalues, pairs




    def filter_pairs(self, tests, data, ci) -> list:
        """takes in dict[k=ticker,v=series] and returns a list of pairs which pass all tests in tests parameter with confidence interval ci"""
        n, pairs, keys, count = len(data), [], list(data.keys()), 0

        for i in range(n):
            for j in range(i+1,n):
                count += 1
                s1, s2 = data[keys[i]][:min(len(data[keys[i]]),len(data[keys[j]]))], data[keys[j]][:min(len(data[keys[i]]),len(data[keys[j]]))]
                if not (len(s1) and len(s2)): continue #one of the tickers is missing data, skip
                print(str(round((count/(n**2))*100,2)) + '%')
                results = []
                failed = False
                for test in tests:
                    result = test(s1,s2)
                    if result >= 1 - ci:
                        failed = True
                        break

                if not failed:
                    #passed all tests
                    pairs.append((keys[i],keys[j]))

        return pairs




