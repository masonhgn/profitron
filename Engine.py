from TopTenMomentum import TopTenMomentum
from TradeMaker import TradeMaker

def main():
    print('welcome to the trading desk.')
    #cur_strat = TopTenMomentum()
    #portfolio = cur_strat.generate_portfolio()
    #cur_strat.print_portfolio()
    #trades = cur_strat.generate_trade_signals()
    #for t in trades: print(t)
    trader = TradeMaker()
    trader.getAccountInfo()

if __name__ == "__main__":
    main()
