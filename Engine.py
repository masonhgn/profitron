from trading_strategies.TopTenMomentum import TopTenMomentum
from TradeMaker import TradeMaker
from tools.alpha_factors import calculate_rsi
from tools.basic_tools import bollinger_bands, basic_correlation

def main():
    print('welcome to the trading desk.')
    #strat = TopTenMomentum()
    #strat.generate_portfolio()
    #strat.print_portfolio()
    #print(basic_correlation('F','AAPL'))
    print(calculate_rsi('AAPL'))

if __name__ == "__main__":
    main()
