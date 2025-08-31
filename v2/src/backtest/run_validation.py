#!/usr/bin/env python3
"""
Comprehensive Strategy Validation Script
Runs walk-forward analysis, cross-validation, bootstrap testing, and parameter stability analysis
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import matplotlib.pyplot as plt

from strategies.equities.mean_reversion.CointegrationStrategy import CointegrationStrategy
from .ValidationFramework import ValidationFramework
from .MonteCarloSimulation import MonteCarloSimulation
from .Backtester import Backtester, BacktestConfig
from ..data_collection.DataManager import DataManager
from ..utils import load_environment, load_yaml_config, resolve_config_values


def load_config():
    """Load configuration"""
    load_environment()
    config = load_yaml_config("src/core/config/Engine.yaml")
    return resolve_config_values(config)


def get_price_data(config):
    """Get price data for validation"""
    dm = DataManager(config)
    
    # Get strategy assets
    strategy_config = config['strategy']['params']
    assets = [
        strategy_config['asset_1'],
        strategy_config['asset_2']
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
    bt_params = config['backtest']['params']
    return BacktestConfig(
        start_date=bt_params['start_date'],
        end_date=bt_params['end_date'],
        capital=bt_params['capital'],
        slippage_bps=bt_params['slippage_bps'],
        commission_per_trade=bt_params['commission_per_trade'],
        rebalance_frequency=bt_params.get('rebalance_frequency', 'daily'),
        bid_ask_spread_bps=bt_params.get('bid_ask_spread_bps', 5.0),
        min_trade_size=bt_params.get('min_trade_size', 100.0),
        max_position_size=bt_params.get('max_position_size', 0.5)
    )


def run_comprehensive_validation():
    """Run comprehensive validation tests"""
    print("=" * 80)
    print("COMPREHENSIVE STRATEGY VALIDATION")
    print("=" * 80)
    
    # Load configuration
    config = load_config()
    price_data = get_price_data(config)
    base_config = create_backtest_config(config)
    
    print(f"Data loaded: {len(price_data)} observations")
    print(f"Date range: {price_data.index.min()} to {price_data.index.max()}")
    print(f"Assets: {list(price_data.columns)}")
    
    # Create validation frameworks
    validation_framework = ValidationFramework(CointegrationStrategy, price_data, base_config)
    monte_carlo_framework = MonteCarloSimulation(CointegrationStrategy, price_data, base_config)
    
    # Define parameter combinations for testing
    base_params = {
        'asset_1': config['strategy']['params']['asset_1'],
        'asset_2': config['strategy']['params']['asset_2'],
        'lookback_bars': 20,
        'entry_z': 2.0,
        'exit_z': 0.5,
        'frequency': config['strategy']['params']['frequency'],
        'poll_interval': config['strategy']['params']['poll_interval'],
        'fields': config['strategy']['params']['fields'],
        'hedge_ratio_method': 'ols'
    }
    
    # Parameter combinations for walk-forward analysis
    param_combinations = [
        {**base_params, 'lookback_bars': 10, 'entry_z': 1.5, 'exit_z': 0.5},
        {**base_params, 'lookback_bars': 20, 'entry_z': 2.0, 'exit_z': 0.5},
        {**base_params, 'lookback_bars': 30, 'entry_z': 2.5, 'exit_z': 0.8},
        {**base_params, 'lookback_bars': 50, 'entry_z': 2.0, 'exit_z': 0.5},
        {**base_params, 'lookback_bars': 20, 'entry_z': 1.5, 'exit_z': 0.3},
        {**base_params, 'lookback_bars': 20, 'entry_z': 2.5, 'exit_z': 0.8},
    ]
    
    all_results = []
    
    # 1. Walk-Forward Analysis
    print("\n" + "=" * 60)
    print("1. WALK-FORWARD ANALYSIS")
    print("=" * 60)
    
    wf_results = validation_framework.run_walk_forward_analysis(
        param_combinations=param_combinations,
        train_days=252,  # 1 year
        test_days=63,    # 3 months
        step_days=21,    # 1 month
        min_test_days=30
    )
    all_results.extend(wf_results)
    
    # 2. Time Series Cross-Validation
    print("\n" + "=" * 60)
    print("2. TIME SERIES CROSS-VALIDATION")
    print("=" * 60)
    
    cv_results = validation_framework.run_time_series_cv(
        params=base_params,
        n_splits=5,
        test_size=0.2
    )
    all_results.extend(cv_results)
    
    # 3. Bootstrap Analysis
    print("\n" + "=" * 60)
    print("3. BOOTSTRAP ANALYSIS")
    print("=" * 60)
    
    bootstrap_results = validation_framework.run_bootstrap_analysis(
        params=base_params,
        n_bootstrap=500,  # Reduced for speed
        sample_size=None
    )
    all_results.extend(bootstrap_results)
    
    # 4. Parameter Stability Test
    print("\n" + "=" * 60)
    print("4. PARAMETER STABILITY TEST")
    print("=" * 60)
    
    param_ranges = {
        'lookback_bars': [10, 20, 30, 50],
        'entry_z': [1.5, 2.0, 2.5],
        'exit_z': [0.3, 0.5, 0.8]
    }
    
    stability_results = validation_framework.run_parameter_stability_test(
        base_params=base_params,
        param_ranges=param_ranges,
        test_periods=5
    )
    all_results.extend(stability_results)
    
    # 5. Monte Carlo Simulations
    print("\n" + "=" * 60)
    print("5. MONTE CARLO SIMULATIONS")
    print("=" * 60)
    
    mc_results = monte_carlo_framework.simulate_market_conditions(
        base_params=base_params,
        n_simulations=200,  # Reduced for speed
        scenarios=['normal', 'volatile', 'trending', 'mean_reverting', 'crisis']
    )
    all_results.extend(mc_results)
    
    # 6. Stress Tests
    print("\n" + "=" * 60)
    print("6. STRESS TESTS")
    print("=" * 60)
    
    stress_results = monte_carlo_framework.run_stress_tests(
        base_params=base_params
    )
    all_results.extend(stress_results)
    
    # 7. Analyze Results
    print("\n" + "=" * 60)
    print("7. VALIDATION RESULTS ANALYSIS")
    print("=" * 60)
    
    # Analyze validation results
    validation_summary = validation_framework.analyze_results(all_results)
    
    # Analyze Monte Carlo results separately
    mc_only_results = [r for r in all_results if r.method in ['monte_carlo', 'stress_test']]
    mc_summary = monte_carlo_framework.analyze_monte_carlo_results(mc_only_results)
    
    # Combine summaries
    summary = {
        **validation_summary,
        'monte_carlo_summary': mc_summary
    }
    
    print(f"\nSUMMARY STATISTICS:")
    print(f"Total tests: {summary['total_tests']}")
    print(f"Methods: {summary['methods']}")
    print(f"Mean net return: ${summary['mean_net_return']:.2f}")
    print(f"Std net return: ${summary['std_net_return']:.2f}")
    print(f"Median net return: ${summary['median_net_return']:.2f}")
    print(f"Positive return rate: {summary['positive_return_rate']:.1%}")
    print(f"Mean Sharpe ratio: {summary['mean_sharpe']:.2f}")
    print(f"Mean max drawdown: {summary['mean_max_drawdown']:.1%}")
    print(f"Mean win rate: {summary['mean_win_rate']:.1%}")
    print(f"Mean profit factor: {summary['mean_profit_factor']:.2f}")
    print(f"Total costs ratio: {summary['total_costs_ratio']:.1%}")
    
    print(f"\nCONFIDENCE INTERVALS (95%):")
    print(f"Net return: ${summary['confidence_intervals']['net_return_95'][0]:.2f} to ${summary['confidence_intervals']['net_return_95'][1]:.2f}")
    print(f"Sharpe ratio: {summary['confidence_intervals']['sharpe_95'][0]:.2f} to {summary['confidence_intervals']['sharpe_95'][1]:.2f}")
    print(f"Max drawdown: {summary['confidence_intervals']['max_dd_95'][0]:.1%} to {summary['confidence_intervals']['max_dd_95'][1]:.1%}")
    
    # 8. Plot Results
    print("\n" + "=" * 60)
    print("8. GENERATING PLOTS")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"validation_results_{timestamp}.png"
    
    # Plot validation results
    validation_framework.plot_results(all_results, save_path=plot_path)
    
    # Plot Monte Carlo results separately
    if mc_only_results:
        mc_plot_path = f"monte_carlo_results_{timestamp}.png"
        monte_carlo_framework.plot_monte_carlo_results(mc_only_results, save_path=mc_plot_path)
    
    # 9. Save Results
    print("\n" + "=" * 60)
    print("9. SAVING RESULTS")
    print("=" * 60)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame([{
        'method': r.method,
        'total_return': r.total_return,
        'sharpe_ratio': r.sharpe_ratio,
        'max_drawdown': r.max_drawdown,
        'win_rate': r.win_rate,
        'profit_factor': r.profit_factor,
        'avg_trade': r.avg_trade,
        'num_trades': r.num_trades,
        'total_costs': r.total_costs,
        'net_return': r.net_return,
        'params': str(r.params),
        'metadata': str(r.metadata)
    } for r in all_results])
    
    csv_path = f"validation_results_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Save summary
    summary_path = f"validation_summary_{timestamp}.txt"
    with open(summary_path, 'w') as f:
        f.write("COMPREHENSIVE STRATEGY VALIDATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Strategy: ETHV/ETHA Cointegration\n")
        f.write(f"Frequency: {base_params['frequency']}\n\n")
        
        f.write("SUMMARY STATISTICS:\n")
        f.write(f"Total tests: {summary['total_tests']}\n")
        f.write(f"Methods: {summary['methods']}\n")
        f.write(f"Mean net return: ${summary['mean_net_return']:.2f}\n")
        f.write(f"Std net return: ${summary['std_net_return']:.2f}\n")
        f.write(f"Median net return: ${summary['median_net_return']:.2f}\n")
        f.write(f"Positive return rate: {summary['positive_return_rate']:.1%}\n")
        f.write(f"Mean Sharpe ratio: {summary['mean_sharpe']:.2f}\n")
        f.write(f"Mean max drawdown: {summary['mean_max_drawdown']:.1%}\n")
        f.write(f"Mean win rate: {summary['mean_win_rate']:.1%}\n")
        f.write(f"Mean profit factor: {summary['mean_profit_factor']:.2f}\n")
        f.write(f"Total costs ratio: {summary['total_costs_ratio']:.1%}\n\n")
        
        f.write("CONFIDENCE INTERVALS (95%):\n")
        f.write(f"Net return: ${summary['confidence_intervals']['net_return_95'][0]:.2f} to ${summary['confidence_intervals']['net_return_95'][1]:.2f}\n")
        f.write(f"Sharpe ratio: {summary['confidence_intervals']['sharpe_95'][0]:.2f} to {summary['confidence_intervals']['sharpe_95'][1]:.2f}\n")
        f.write(f"Max drawdown: {summary['confidence_intervals']['max_dd_95'][0]:.1%} to {summary['confidence_intervals']['max_dd_95'][1]:.1%}\n")
    
    print(f"Summary saved to {summary_path}")
    
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE!")
    print("=" * 80)
    
    return all_results, summary


if __name__ == "__main__":
    run_comprehensive_validation() 