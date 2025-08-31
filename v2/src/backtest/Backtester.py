# src/backtesting/Backtester.py

from typing import List, Dict, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
from tqdm import tqdm

from ..portfolio.PortfolioManager import PortfolioManager
from strategies.TradingStrategy import TradingStrategy
from .PerformanceStats import PerformanceStats
from ..signals.Signal import Signal


@dataclass
class BacktestConfig:
    start_date: str
    end_date: str
    capital: float
    slippage_bps: float
    commission_per_trade: float
    rebalance_frequency: str
    bid_ask_spread_bps: float = 5.0
    min_trade_size: float = 100.0
    max_position_size: float = 0.5


def combine_signals(signals: List[Signal], all_assets: List[str], max_gross_exposure: float = 1.0) -> pd.Series:

    # FIXED: Create proper mapping from symbol to column name
    # Signals have symbols like "SPY", but we need to find columns like "SPY_close"
    asset_to_column = {}
    for col in all_assets:
        if col.endswith('_close'):
            symbol = col.replace('_close', '')
            asset_to_column[symbol] = col
    
    weights = pd.Series(0.0, index=all_assets)

    for sig in signals:
        pct = sig.target_pct or 0.0
        column_name = asset_to_column.get(sig.symbol)
        if column_name is None:
            print(f"Warning: Signal symbol {sig.symbol} does not match any data column")
            continue
        if sig.action == "buy":
            weights[column_name] += pct
        elif sig.action == "sell":
            weights[column_name] -= pct

    # normalize total gross exposure to max_gross_exposure
    gross = weights.abs().sum()
    if gross > max_gross_exposure:
        weights *= max_gross_exposure / gross

    result = weights.clip(-1.0, 1.0)
    return result


def calculate_effective_prices(day_prices: pd.Series, signal_sum: pd.Series, config: BacktestConfig) -> pd.Series:
    """
    Calculate realistic execution prices considering bid/ask spreads and slippage
    
    Args:
        day_prices: Current prices (close prices)
        signal_sum: Target position weights
        config: Backtest configuration
        
    Returns:
        pd.Series: Effective execution prices
    """
    effective_prices = day_prices.copy()
    
    for i, (asset, signal) in enumerate(signal_sum.items()):
        if abs(signal) < 0.001:  # No trade
            continue
            
        # Get OHLCV data if available
        base_symbol = asset.replace('_close', '')
        
        # Calculate bid/ask spread
        spread_bps = config.bid_ask_spread_bps / 10000
        current_price = day_prices[asset]
        
        if signal > 0:  # Buy - pay ask price
            # Ask price = close + half spread
            ask_price = current_price * (1 + spread_bps / 2)
            # Add slippage (positive for buys)
            slippage = current_price * (config.slippage_bps / 10000)
            effective_prices[asset] = ask_price + slippage
        else:  # Sell - receive bid price
            # Bid price = close - half spread
            bid_price = current_price * (1 - spread_bps / 2)
            # Subtract slippage (negative for sells)
            slippage = current_price * (config.slippage_bps / 10000)
            effective_prices[asset] = bid_price - slippage
    
    return effective_prices


def calculate_trading_costs(signal_sum: pd.Series, effective_prices: pd.Series, config: BacktestConfig) -> float:
    """
    Calculate total trading costs including commission and spread costs
    
    Args:
        signal_sum: Target position weights
        effective_prices: Execution prices
        config: Backtest configuration
        
    Returns:
        float: Total trading costs
    """
    total_cost = 0.0
    
    for asset, signal in signal_sum.items():
        if abs(signal) < 0.001:  # No trade
            continue
            
        # Commission cost
        commission_cost = config.commission_per_trade
        
        # Spread cost (difference between effective price and mid price)
        base_symbol = asset.replace('_close', '')
        mid_price = effective_prices[asset]  # This should be close price
        
        # Calculate spread cost
        if signal > 0:  # Buy
            spread_cost = effective_prices[asset] - mid_price
        else:  # Sell
            spread_cost = mid_price - effective_prices[asset]
        
        # Total cost for this trade
        trade_cost = commission_cost + (abs(signal) * spread_cost)
        total_cost += trade_cost
    
    return total_cost


class Backtester:
    def __init__(self, strategies: List[TradingStrategy], price_data: pd.DataFrame, config: BacktestConfig):
        self.strategies = strategies
        self.data = price_data
        self.config = config
        self.pm = PortfolioManager(config.capital)

        # Store original string dates for metadata
        self.original_start_date = self.config.start_date
        self.original_end_date = self.config.end_date
        
        # Convert to timestamps for processing
        self.start_timestamp = pd.Timestamp(self.config.start_date)
        self.end_timestamp = pd.Timestamp(self.config.end_date)

    def run(self) -> Dict[str, pd.DataFrame]:
        date_index = self.data.index
        positions = pd.DataFrame(index=date_index, columns=self.data.columns, dtype=float).fillna(0)
        pnl = pd.Series(index=date_index, dtype=float)
        trading_costs = pd.Series(index=date_index, dtype=float)

        for t, date in tqdm(enumerate(date_index), total=len(date_index), desc="Backtesting"):
            if date < self.start_timestamp or date > self.end_timestamp:
                continue

            day_prices = self.data.loc[date]

            #1 collect all signals from all strategies
            all_signals: List[Signal] = []
            for strat in self.strategies:
                signals = strat.on_event(self.data.loc[:date])
                all_signals.extend(signals)

            #2 convert to unified target weights
            signal_sum = combine_signals(all_signals, self.data.columns.tolist(), max_gross_exposure=1.0)

            #3 get close prices for trading
            if isinstance(day_prices.index, pd.MultiIndex):
                try:
                    close_prices = day_prices.loc[(slice(None), "close")]
                    close_prices.index = close_prices.index.droplevel(1)  # drop "close" level
                except KeyError:
                    raise ValueError("[Backtester] 'close' prices not found in MultiIndex day_prices!")
            else:
                close_prices = day_prices  # assume it's already flat, with columns like ["BTC-USDT", "ETH-USDT"]

            #4 calculate realistic execution prices
            effective_prices = calculate_effective_prices(close_prices, signal_sum, self.config)

            #5 calculate trading costs
            daily_trading_cost = calculate_trading_costs(signal_sum, effective_prices, self.config)
            trading_costs[date] = daily_trading_cost

            #6 position sizing
            current_position = self.pm.update_positions(signal_sum, effective_prices)

            #7 mark-to-market PnL 
            if t > 0:
                price_diff = close_prices - self.data.loc[self.data.index[t-1]]
                daily_pnl = (positions.iloc[t-1] * price_diff).sum() - daily_trading_cost
                pnl[date] = daily_pnl

            positions.loc[date] = current_position

        cumulative_pnl = pnl.dropna().cumsum()
        cumulative_costs = trading_costs.dropna().cumsum()
        
        # Extract ticker information from data columns for metadata
        tickers = []
        for col in self.data.columns:
            if col.endswith('_close'):
                ticker = col.replace('_close', '')
                tickers.append(ticker)
        
        # Create metadata for plotting
        metadata = {
            'tickers': tickers,
            'start_date': self.original_start_date,
            'end_date': self.original_end_date,
            'strategy_name': self._get_strategy_name(),
            'initial_capital': self.config.capital,
            'slippage_bps': self.config.slippage_bps,
            'commission_per_trade': self.config.commission_per_trade,
            'bid_ask_spread_bps': self.config.bid_ask_spread_bps,
            'max_position_size': self.config.max_position_size,
            'total_trading_costs': cumulative_costs.iloc[-1] if len(cumulative_costs) > 0 else 0.0
        }
        
        # Add strategy parameters if available
        if self.strategies and len(self.strategies) > 0:
            strategy = self.strategies[0]
            metadata.update(self._extract_strategy_parameters(strategy))
        
        results = {
            "pnl": cumulative_pnl,
            "positions": positions,
            "trading_costs": cumulative_costs,
            "stats": PerformanceStats.compute(self.config.capital, cumulative_pnl)
        }
        
        print(f"\n=== BACKTEST RESULTS ===")
        print(f"Total PnL: ${cumulative_pnl.iloc[-1]:.2f}" if len(cumulative_pnl) > 0 else "Total PnL: $0.00")
        print(f"Total Trading Costs: ${cumulative_costs.iloc[-1]:.2f}" if len(cumulative_costs) > 0 else "Total Trading Costs: $0.00")
        print(f"Net PnL: ${(cumulative_pnl.iloc[-1] - cumulative_costs.iloc[-1]):.2f}" if len(cumulative_pnl) > 0 and len(cumulative_costs) > 0 else "Net PnL: $0.00")
        
        PerformanceStats.plot(cumulative_pnl, title=self._get_strategy_name(), metadata=metadata, trading_costs=cumulative_costs, save_image=False)
        return results

    def _get_strategy_name(self) -> str:
        """Extract strategy name from the first strategy"""
        if self.strategies and len(self.strategies) > 0:
            strategy = self.strategies[0]
            # Try to get the class name
            strategy_name = strategy.__class__.__name__
            # Remove 'Strategy' suffix if present
            if strategy_name.endswith('Strategy'):
                strategy_name = strategy_name[:-8]  # Remove 'Strategy'
            return strategy_name
        return "Unknown Strategy"

    def _extract_strategy_parameters(self, strategy) -> Dict[str, Any]:
        """Extract strategy parameters dynamically"""
        params = {}
        
        # Get all attributes that might be strategy parameters
        for attr_name in dir(strategy):
            if not attr_name.startswith('_') and not callable(getattr(strategy, attr_name)):
                attr_value = getattr(strategy, attr_name)
                # Only include simple types (str, int, float, bool)
                if isinstance(attr_value, (str, int, float, bool)):
                    params[attr_name] = attr_value
        
        return params
