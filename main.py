from components import StatArb


if __name__ == "__main__":
    
    strategy = StatArb()
    now = dt.datetime.now()

    if now >= strategy.next_trading_date:
        current_strategy.rebalance()
        days = int(strategy.frequency[:-1])
        strategy.next_trading_date = now + dt.timedelta(days=days)
        print(f"Next trading date set to: {strategy.next_trading_date}")