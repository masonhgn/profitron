from trading_strategies.TopTenMomentum import TopTenMomentum
from TradeMaker import TradeMaker

def main():
    print('welcome to the trading desk.')
    strat = TopTenMomentum()
    strat.generate_portfolio()
    strat.print_portfolio()

if __name__ == "__main__":
    main()
