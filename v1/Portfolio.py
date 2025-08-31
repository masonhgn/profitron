
import datetime
import numpy as np


class Portfolio:
    def __init__(self, balance):
        self.balance = balance
        self.holdings = {}


        self.performance_history = []


    def buy(self, ticker, share_price, amount):
        assert self.balance >= share_price*amount

        print('buying ' + ticker + ', balance = ' + str(self.balance))

        self.balance -= share_price*amount

        if ticker in self.holdings:
            old_amount, old_cost_basis = self.holdings[ticker]['amount'], self.holdings[ticker]['cost_basis']
            self.holdings[ticker]['cost_basis'] = ((old_cost_basis*old_amount) + (share_price*amount))/ (old_amount+amount)
            self.holdings[ticker]['amount'] += amount
            self.holdings[ticker]['current_price'] = share_price
            self.holdings[ticker]['total_value'] = self.holdings[ticker]['amount'] * self.holdings[ticker]['current_price']
        else:
            self.holdings[ticker] = {
                'amount': amount,
                'cost_basis': share_price,
                'current_price': share_price,
                'total_value': amount * share_price
            }

        print('after buying ' + ticker + ' balance = ' + str(self.balance))


    def sell(self, ticker, amount):
        assert ticker in self.holdings and self.holdings[ticker]['amount'] >= amount

        print('selling ' + ticker)

        self.balance += self.holdings[ticker]['current_price'] * amount

        if self.holdings[ticker]['amount'] == amount:
            del self.holdings[ticker]
        else:
            self.holdings[ticker]['amount'] -= amount
            self.holdings[ticker]['total_value'] -= (amount*self.holdings[ticker]['current_price'])


    def liquidate(self):
        tickers = list(self.holdings.keys())
        for ticker in tickers:
            self.sell(ticker, self.holdings[ticker]['amount'])



    def equal_weight_allocation(self, basket):
        """basket is dict with k=ticker, v=current_price"""
        assert len(self.holdings.keys()) == 0 #we should have already liquidated our portfolio before this
        assert self.balance > 0 #we can't have no money

        allocation_per_security = self.balance / len(basket.keys()) #how much money per security are we investing?

        for ticker in basket.keys():
            if basket[ticker] > allocation_per_security: #if we don't have enough money to allocate for even 1 share, there's something wrong
                print('NOT ENOUGH MONEY TO BUY ' + ticker + ', current balance = ' + str(self.balance))
                continue

            #we cannot buy fractional shares
            num_shares = allocation_per_security // basket[ticker]
            if num_shares >= 1:
                self.buy(ticker, basket[ticker], num_shares)


    def update_price(self, ticker, new_price):
        assert ticker in self.holdings
        self.holdings[ticker]['current_price'] = new_price
        self.holdings[ticker]['total_value'] = self.holdings[ticker]['current_price'] * self.holdings[ticker]['amount']


    def save_state(self, date):
        """save the current value of the portfolio, so we can keep track of the portfolio's performance over time"""
        total_value = self.get_total_value()
        state = {
            'date': date,
            'balance': self.balance,
            'total_value': total_value,
            'holdings': self.holdings.copy(),  # Make a copy of the current holdings to avoid mutability issues
        }
        self.performance_history.append(state)

    def get_total_value(self):
        total = 0
        for ticker in self.holdings.keys():
            total += self.holdings[ticker]['total_value']
        return total + self.balance

    
    def get_tickers_owned(self):
        return list(self.holdings.keys())




    def calculate_metrics(self, risk_free_rate):
        """get key metrics for portfolio"""
        if len(self.performance_history) < 2:
            raise ValueError("not enough dates")

        #get total portfolio values over time 
        values = np.array([state['total_value'] for state in self.performance_history])


        returns = np.diff(values) / values[:-1]


        cumulative_return = (values[-1] / values[0]) - 1

        #volatility is just std dev. of returns
        volatility = np.std(returns)


        #get max drawdown
        cumulative_returns = (values / values[0]) - 1
        # running_max = np.maximum.accumulate(cumulative_returns)
        # drawdown = np.where(running_max != 0, (cumulative_returns - running_max) / running_max, 0)
        # max_drawdown = drawdown.min()

        #get sharpe ratio
        excess_returns = returns - risk_free_rate
        sharpe_ratio = np.mean(excess_returns) / volatility if volatility > 0 else 0
    
        metrics = {
            'cumulative_return': cumulative_return,
            'volatility': volatility,
            #'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }

        return metrics



    
        