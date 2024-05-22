import pandas as pd
import pandas_datareader as pdr
import numpy as np
import yfinance as yf
from datetime import datetime as dt



class Analysis(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Analysis, cls).__new__(cls)
            return cls.instance

    '''
    TODO: implement this class
    '''


    def __init__(self):
        self.dataframes: list = []


    def cointegration_test(self, series1: pd.Series, series2: pd.Series) -> bool:
        pass

    def cointegration_matrix(self, list[list[pd.Series]]) -> list[list[bool]]:
        pass

