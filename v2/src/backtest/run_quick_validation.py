#!/usr/bin/env python3
"""
quick validation script for the cointegration strategy
runs a subset of validation tests for fast feedback
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # use non-interactive backend
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.equities.mean_reversion.CointegrationStrategy import CointegrationStrategy
from .Backtester import Backtester, BacktestConfig
from ..data_collection.DataManager import DataManager
from ..utils import load_environment, load_yaml_config, resolve_config_values

def main():
    """run quick validation tests"""
    
    print("running quick validation...")
    
    # load configuration
    load_environment()
    config = load_yaml_config("src/core/config/Engine.yaml")
    config = resolve_config_values(config)
    
    # get data
    dm = DataManager(config)
    bt_params = config['backtest']['params']
    
    data = dm.get_price_data(
        assets=[
            {'symbol': 'ETHA', 'type': 'equity'},
            {'symbol': 'ETHV', 'type': 'equity'}
        ],
        start_date=bt_params.get('start_date', '2024-07-23'),
        end_date=bt_params.get('end_date', '2025-04-09'),
        frequency=bt_params.get('frequency', '1h'),
        fields=bt_params.get('fields', ['open', 'high', 'low', 'close', 'volume'])
    )
    
    if data.empty:
        print("no data available for validation")
        return
    
    price_data = data
    print(f"date range: {price_data.index.min()} to {price_data.index.max()}")
    print(f"assets: {list(price_data.columns)}")
    
    # base configuration
    base_config = BacktestConfig(
        start_date=bt_params.get('start_date', '2024-07-23'),
        end_date=bt_params.get('end_date', '2025-04-09'),
        capital=bt_params.get('capital', 100000),
        slippage_bps=bt_params.get('slippage_bps', 5.0),
        commission_per_trade=bt_params.get('commission_per_trade', 1.0),
        rebalance_frequency=bt_params.get('rebalance_frequency', 'daily'),
        bid_ask_spread_bps=bt_params.get('bid_ask_spread_bps', 5.0),
        min_trade_size=bt_params.get('min_trade_size', 100.0),
        max_position_size=bt_params.get('max_position_size', 0.5)
    )
    
    # test parameters
    test_params = [
        {'lookback_bars': 10, 'entry_z': 1.5, 'exit_z': 0.5},
        {'lookback_bars': 15, 'entry_z': 1.8, 'exit_z': 0.5},
        {'lookback_bars': 20, 'entry_z': 2.0, 'exit_z': 0.5},
        {'lookback_bars': 25, 'entry_z': 2.2, 'exit_z': 0.6},
        {'lookback_bars': 30, 'entry_z': 2.0, 'exit_z': 0.7},
    ]
    
    results = []
    
    print("1. parameter sensitivity test")
    print("-" * 40)
    
    for i, params in enumerate(test_params):
        print(f"test {i+1}: {params}")
        
        strategy = CointegrationStrategy(
            asset_1={'symbol': 'ETHA', 'type': 'equity'},
            asset_2={'symbol': 'ETHV', 'type': 'equity'},
            **params
        )
        
        backtester = Backtester(
            strategies=[strategy],
            price_data=price_data,
            config=base_config
        )
        
        backtest_results = backtester.run()
        
        # extract metrics
        pnl_series = backtest_results.get('pnl', pd.Series([0]))
        total_return = pnl_series.iloc[-1]
        returns = pnl_series.diff().dropna()
        
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        max_drawdown = (pnl_series - pnl_series.expanding().max()).min()
        
        trading_costs = backtest_results.get('trading_costs', pd.Series([0]))
        total_costs = trading_costs.iloc[-1] if len(trading_costs) > 0 else 0
        
        net_return = total_return - total_costs
        
        num_trades = len(returns[returns != 0])
        win_rate = len(returns[returns > 0]) / num_trades if num_trades > 0 else 0
        
        results.append({
            'test': f'param_test_{i+1}',
            'params': params,
            'total_return': total_return,
            'net_return': net_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_costs': total_costs,
            'win_rate': win_rate
        })
        
        print(f"  net return: ${net_return:.2f}, sharpe: {sharpe_ratio:.2f}, win rate: {win_rate:.1%}")
    
    print("\n2. time period analysis")
    print("-" * 40)
    
    # test different time periods
    num_quarters = 4
    data_length = len(price_data)
    quarter_length = data_length // num_quarters
    
    for i in range(num_quarters):
        start_idx = i * quarter_length
        end_idx = (i + 1) * quarter_length if i < num_quarters - 1 else data_length
        
        quarter_data = price_data.iloc[start_idx:end_idx]
        quarter_start = quarter_data.index.min()
        quarter_end = quarter_data.index.max()
        
        print(f"quarter {i+1}: {quarter_start.date()} to {quarter_end.date()}")
        
        quarter_config = BacktestConfig(
            start_date=quarter_start.strftime('%Y-%m-%d'),
            end_date=quarter_end.strftime('%Y-%m-%d'),
            capital=base_config.capital,
            slippage_bps=base_config.slippage_bps,
            commission_per_trade=base_config.commission_per_trade,
            rebalance_frequency=base_config.rebalance_frequency,
            bid_ask_spread_bps=base_config.bid_ask_spread_bps,
            min_trade_size=base_config.min_trade_size,
            max_position_size=base_config.max_position_size
        )
        
        strategy = CointegrationStrategy(
            asset_1={'symbol': 'ETHA', 'type': 'equity'},
            asset_2={'symbol': 'ETHV', 'type': 'equity'},
            lookback_bars=15,
            entry_z=1.8,
            exit_z=0.5
        )
        
        backtester = Backtester(
            strategies=[strategy],
            price_data=quarter_data,
            config=quarter_config
        )
        
        quarter_results = backtester.run()
        
        # extract metrics
        pnl_series = quarter_results.get('pnl', pd.Series([0]))
        total_return = pnl_series.iloc[-1]
        returns = pnl_series.diff().dropna()
        
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        max_drawdown = (pnl_series - pnl_series.expanding().max()).min()
        
        trading_costs = quarter_results.get('trading_costs', pd.Series([0]))
        total_costs = trading_costs.iloc[-1] if len(trading_costs) > 0 else 0
        
        net_return = total_return - total_costs
        
        num_trades = len(returns[returns != 0])
        win_rate = len(returns[returns > 0]) / num_trades if num_trades > 0 else 0
        
        results.append({
            'test': f'quarter_{i+1}',
            'params': {'lookback_bars': 15, 'entry_z': 1.8, 'exit_z': 0.5},
            'total_return': total_return,
            'net_return': net_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_costs': total_costs,
            'win_rate': win_rate
        })
        
        print(f"  net return: ${net_return:.2f}, sharpe: {sharpe_ratio:.2f}, win rate: {win_rate:.1%}")
    
    print("\n3. validation summary")
    print("-" * 40)
    
    # analyze results
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        print("no results to analyze")
        return
    
    print(f"total tests: {len(df)}")
    print(f"mean net return: ${df['net_return'].mean():.2f}")
    print(f"std net return: ${df['net_return'].std():.2f}")
    print(f"median net return: ${df['net_return'].median():.2f}")
    print(f"positive return rate: {(df['net_return'] > 0).mean():.1%}")
    print(f"mean sharpe ratio: {df['sharpe_ratio'].mean():.2f}")
    print(f"mean max drawdown: {df['max_drawdown'].mean():.1%}")
    print(f"mean win rate: {df['win_rate'].mean():.1%}")
    print(f"mean trading costs: ${df['total_costs'].mean():.2f}")
    
    # best and worst results
    best_result = df.loc[df['net_return'].idxmax()]
    print(f"\nbest result ({best_result['test']}):")
    print(f"net return: ${best_result['net_return']:.2f}")
    print(f"sharpe ratio: {best_result['sharpe_ratio']:.2f}")
    print(f"win rate: {best_result['win_rate']:.1%}")
    
    worst_result = df.loc[df['net_return'].idxmin()]
    print(f"\nworst result ({worst_result['test']}):")
    print(f"net return: ${worst_result['net_return']:.2f}")
    print(f"sharpe ratio: {worst_result['sharpe_ratio']:.2f}")
    print(f"win rate: {worst_result['win_rate']:.1%}")
    
    print("\n4. saving results")
    print("-" * 40)
    
    # save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # save summary
    summary_path = f"quick_validation_{timestamp}.txt"
    with open(summary_path, 'w') as f:
        f.write(f"date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"strategy: cointegration\n")
        f.write(f"assets: ETHA/ETHV\n")
        f.write(f"frequency: {bt_params.get('frequency', '1h')}\n\n")
        
        f.write(f"summary statistics:\n")
        f.write(f"mean net return: ${df['net_return'].mean():.2f}\n")
        f.write(f"std net return: ${df['net_return'].std():.2f}\n")
        f.write(f"positive return rate: {(df['net_return'] > 0).mean():.1%}\n")
        f.write(f"mean sharpe ratio: {df['sharpe_ratio'].mean():.2f}\n")
        f.write(f"mean max drawdown: {df['max_drawdown'].mean():.1%}\n")
        f.write(f"mean win rate: {df['win_rate'].mean():.1%}\n\n")
        
        f.write(f"best result ({best_result['test']}):\n")
        f.write(f"net return: ${best_result['net_return']:.2f}\n")
        f.write(f"sharpe ratio: {best_result['sharpe_ratio']:.2f}\n")
        f.write(f"win rate: {best_result['win_rate']:.1%}\n\n")
        
        f.write(f"detailed results:\n")
        for _, row in df.iterrows():
            f.write(f"{row['test']}: net return=${row['net_return']:.2f}, sharpe={row['sharpe_ratio']:.2f}, win rate={row['win_rate']:.1%}\n")
    
    # save csv
    csv_path = f"quick_validation_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"results saved to {summary_path} and {csv_path}")
    print("quick validation complete!")

if __name__ == "__main__":
    main() 