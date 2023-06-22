from trading_strategies.TopTenMomentum import TopTenMomentum
from TradeMaker import TradeMaker

def main():
    print('welcome to the trading desk.')
    trader = TradeMaker()
    trader.get_cash_available()

if __name__ == "__main__":
    main()
