
from tradier_api import ApiBridge
from tools.technical_indicators import calculate_rsi, bollinger_bands, basic_correlation, rolling_sma
from data_collection.data_collection import *
from strategies.momentum_strategies import MomentumTradingStrategy
from strategies.BBVT import BBVT
from statsmodels.tsa.stattools import adfuller
def main():
    print('welcome to the trading desk.')

    strategy = BBVT()
    strategy.calculate()



    # prices = prices_df(['AAPL'],30)
    # prices['Log Return'] = np.log(prices['Close'] / prices['Close'].shift(1))
    # daily_volatility = prices['Log Return'].std()
    # print(daily_volatility) #we might set a stop loss at 2x the daily volatility
    #calculate middle band, upper band and lower band



    #current = MomentumTradingStrategy("AAPL", "2022-10-18", "2023-10-18")
    #current.calculate()
    #current.backtest()

    #print(rolling_sma('AAPL', 50, '2023-09-01','2023-10-16'))

    #strat = TopTenMomentum()
    #strat.generate_portfolio()
    #strat.print_portfolio()
    #print(basic_correlation('F','AAPL'))
    #print(calculate_rsi('AAPL'))

    '''
    #make a map of stocks sorted by rsi
    tickers = price_filter(50, collect_all_tickers())


    rsi_map = {}
    for ticker in tickers:
        rsi_map[ticker] = calculate_rsi(ticker)

    rsi_map = sorted(rsi_map.items(), key=lambda x:x[1], reverse = True)

    print(rsi_map)
    '''    



    '''
    #do linear regression between two stocks
    ticker1, ticker2 = 'AAL', 'UAL'

    fun_df = prices_df([ticker1, ticker2], 800)
    m, b = plot_df(fun_df, ticker1, ticker2)
    print(f"Relationship equation: y = {m:.2f}x + {b:.2f}")
    '''


if __name__ == "__main__":
    main()
