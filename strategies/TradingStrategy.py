import pandas as pd

class TradingStrategy:

    def calculate(self):
        '''
        1. determine asset class / index and create a list of tickers to work from
        2. from that, use tools library to generate a dataframe for necessary stocks with all relevant indicators
        3. generate a 'Signal' column for the dataframe based on relevant indicators
        
        '''
        raise NotImplementedError("Subclasses must implement this method.")

    def backtest(self):
        self.data['Returns'] = self.data['Signal'].shift(1) * self.data['Change']

        cumulative_returns = (1 + self.data['Returns']).cumprod() - 1

        # Calculate annualized returns correctly
        total_return = cumulative_returns.iloc[-1]  # Total return over the entire period
        num_years = len(self.data) / 252  # Assuming 252 trading days in a year
        annualized_returns = (1 + total_return) ** (1 / num_years) - 1

        standard_deviation = self.data['Returns'].std()
        sharpe_ratio = (annualized_returns - 0.03) / (standard_deviation * (252 ** 0.5))  # Assuming a risk-free rate of 3%
        
        max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
        calmar_ratio = annualized_returns / abs(max_drawdown)
        
        win_loss_ratio = (self.data['Returns'] > 0).sum() / (self.data['Returns'] < 0).sum()
        
        print({
            'Cumulative Returns': cumulative_returns.iloc[-1],
            'Annualized Returns': annualized_returns,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Standard Deviation': standard_deviation,
            'Calmar Ratio': calmar_ratio,
            'Win-Loss Ratio': win_loss_ratio,
        })
