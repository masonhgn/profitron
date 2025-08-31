#!/usr/bin/env python3
"""
Ensemble Methods for Cointegration Strategy
Reduces overfitting by combining multiple parameter sets and strategies
"""

import numpy as np
import pandas as pd
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import using the same pattern as the working overfitting check script
from strategies.equities.mean_reversion.CointegrationStrategy import CointegrationStrategy
from backtest.Backtester import Backtester
from data_collection.DataManager import DataManager
from utils.utilities import load_yaml_config, load_environment, resolve_config_values


def load_config():
    """Load configuration"""
    load_environment()
    
    # Load engine config for data sources
    engine_config = load_yaml_config("configs/engine/Engine.yaml")
    
    # Load strategy config
    strategy_config = load_yaml_config("configs/strategies/cointegration_btc_eth.yaml")
    
    # Merge them
    config = {**engine_config, **strategy_config}
    
    # Resolve environment variables in the config
    config = resolve_config_values(config)
    
    return config


def get_price_data(config):
    """Get price data"""
    strategy_config = config['strategy']['params']
    
    # Initialize data manager
    dm = DataManager(config)
    
    # Convert asset strings to proper format
    assets = [
        {'symbol': strategy_config['asset_1'], 'type': 'equity'},
        {'symbol': strategy_config['asset_2'], 'type': 'equity'}
    ]
    
    # Get data
    data = dm.get_price_data(
        assets=assets,
        start_date=config['backtest']['params']['start_date'],
        end_date=config['backtest']['params']['end_date'],
        frequency=strategy_config['frequency'],
        fields=strategy_config['fields']
    )
    
    return data


def create_backtest_config(config):
    """Create backtest configuration"""
    return {
        'initial_capital': 100000,
        'commission': 0.001,
        'slippage': 0.0005,
        'rebalance_frequency': '1H'
    }


def create_ensemble_strategies(config):
    """Create multiple strategy variations for ensemble"""
    base_params = {
        'asset_1': {'symbol': config['strategy']['params']['asset_1'], 'type': 'equity'},
        'asset_2': {'symbol': config['strategy']['params']['asset_2'], 'type': 'equity'},
        'frequency': config['strategy']['params']['frequency'],
        'poll_interval': config['strategy']['params']['poll_interval'],
        'fields': config['strategy']['params']['fields'],
        'hedge_ratio_method': 'ols'
    }
    
    # Define parameter variations for ensemble
    parameter_sets = [
        # Conservative set
        {'lookback_bars': 30, 'entry_z': 1.5, 'exit_z': 0.5, 'weight': 0.3},
        # Moderate set
        {'lookback_bars': 25, 'entry_z': 1.8, 'exit_z': 0.4, 'weight': 0.3},
        # Aggressive set
        {'lookback_bars': 20, 'entry_z': 2.0, 'exit_z': 0.3, 'weight': 0.2},
        # Very conservative set
        {'lookback_bars': 35, 'entry_z': 1.2, 'exit_z': 0.6, 'weight': 0.2},
    ]
    
    strategies = []
    for params in parameter_sets:
        strategy_params = {**base_params, **{k: v for k, v in params.items() if k != 'weight'}}
        weight = params['weight']
        
        strategy = CointegrationStrategy(**strategy_params)
        strategies.append({
            'strategy': strategy,
            'weight': weight,
            'params': params
        })
    
    return strategies


def run_ensemble_validation():
    """Main function to run ensemble validation"""
    print("ENSEMBLE METHODS VALIDATION")
    print("=" * 60)
    
    # Load configuration and data
    config = load_config()
    price_data = get_price_data(config)
    base_config = create_backtest_config(config)
    
    print(f"Data loaded: {len(price_data)} observations")
    print(f"Date range: {price_data.index.min()} to {price_data.index.max()}")
    
    # Create ensemble strategies
    strategies = create_ensemble_strategies(config)
    
    print(f"\nCreated {len(strategies)} ensemble strategies:")
    for i, strategy_info in enumerate(strategies):
        weight = strategy_info['weight']
        params = strategy_info['params']
        print(f"  Strategy {i+1}: Weight {weight:.1%}, Params: {params}")
    
    # Run individual backtests
    print("\n" + "=" * 60)
    print("INDIVIDUAL STRATEGY PERFORMANCE")
    print("=" * 60)
    
    individual_results = []
    for i, strategy_info in enumerate(strategies):
        strategy = strategy_info['strategy']
        weight = strategy_info['weight']
        params = strategy_info['params']
        
        print(f"\nStrategy {i+1} (Weight: {weight:.1%}):")
        print(f"  Parameters: {params}")
        
        # Run backtest
        backtester = Backtester([strategy], price_data, base_config)
        result = backtester.run()
        
        # Calculate metrics
        final_pnl = result['pnl'].iloc[-1] if len(result['pnl']) > 0 else 0
        total_trades = len(result['trades']) if 'trades' in result else 0
        
        individual_results.append({
            'strategy_id': i + 1,
            'weight': weight,
            'params': params,
            'final_pnl': final_pnl,
            'total_trades': total_trades,
            'pnl_series': result['pnl']
        })
        
        print(f"  Final PnL: ${final_pnl:,.2f}")
        print(f"  Total Trades: {total_trades}")
    
    # Calculate weighted ensemble performance
    weighted_pnl = sum(result['final_pnl'] * result['weight'] for result in individual_results)
    
    print(f"\nENSEMBLE RESULTS:")
    print(f"  Weighted PnL: ${weighted_pnl:,.2f}")
    
    # Run out-of-sample test
    print("\n" + "=" * 60)
    print("OUT-OF-SAMPLE ENSEMBLE TEST")
    print("=" * 60)
    
    # Split data 80/20
    split_idx = int(len(price_data) * 0.8)
    train_data = price_data.iloc[:split_idx]
    test_data = price_data.iloc[split_idx:]
    
    print(f"Training period: {train_data.index.min()} to {train_data.index.max()}")
    print(f"Testing period: {test_data.index.min()} to {test_data.index.max()}")
    
    # Run ensemble on training data
    strategy_list = [info['strategy'] for info in strategies]
    train_backtester = Backtester(strategy_list, train_data, base_config)
    train_result = train_backtester.run()
    train_pnl = train_result['pnl'].iloc[-1] if len(train_result['pnl']) > 0 else 0
    
    # Run ensemble on test data
    test_backtester = Backtester(strategy_list, test_data, base_config)
    test_result = test_backtester.run()
    test_pnl = test_result['pnl'].iloc[-1] if len(test_result['pnl']) > 0 else 0
    
    print(f"\nTraining PnL: ${train_pnl:,.2f}")
    print(f"Test PnL: ${test_pnl:,.2f}")
    
    if train_pnl != 0:
        degradation = (test_pnl - train_pnl) / train_pnl * 100
        print(f"Performance degradation: {degradation:.1f}%")
        
        if degradation > -20:
            print("ENSEMBLE: Good out-of-sample performance")
        elif degradation > -50:
            print("ENSEMBLE: Moderate overfitting detected")
        else:
            print("ENSEMBLE: Significant overfitting detected")
    
    # Final assessment
    print("\n" + "=" * 80)
    print("ENSEMBLE ASSESSMENT")
    print("=" * 80)
    
    print(f"Ensemble Summary:")
    print(f"  • Total Strategies: {len(strategies)}")
    print(f"  • Training PnL: ${train_pnl:,.2f}")
    print(f"  • Test PnL: ${test_pnl:,.2f}")
    print(f"  • Weighted PnL: ${weighted_pnl:,.2f}")
    
    if train_pnl != 0 and test_pnl > train_pnl * 0.8:
        print("  • ENSEMBLE: Good out-of-sample performance")
    else:
        print("  • ENSEMBLE: Poor out-of-sample performance")
    
    print(f"\nEnsemble Benefits:")
    print(f"  • Combines multiple parameter sets")
    print(f"  • Reduces single-strategy overfitting")
    print(f"  • Provides diversification")
    print(f"  • More stable performance")


if __name__ == "__main__":
    run_ensemble_validation() 