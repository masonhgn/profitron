#!/usr/bin/env python3
"""
Streamlined Strategy Validation Script
Runs essential validation tests with minimal output for faster execution
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


def run_fast_validation():
    """Run streamlined validation tests"""
    print("=" * 80)
    print("STREAMLINED STRATEGY VALIDATION")
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
    
    # Control flags
    SAVE_IMAGES = False  # Set to True if you want to save images
    SAVE_PLOTS = False   # Set to True if you want to save plots

    # 1. Walk-Forward Analysis (Reduced)
    print("\n" + "=" * 60)
    print("1. WALK-FORWARD ANALYSIS (3 windows)")
    print("=" * 60)
    
    wf_results = validation_framework.run_walk_forward_analysis(
        param_combinations=param_combinations,
        train_days=252,  # 1 year
        test_days=63,    # 3 months
        step_days=126,   # 6 months (fewer windows)
        min_test_days=30
    )
    all_results.extend(wf_results)
    
    # 2. Time Series CV (Reduced)
    print("\n" + "=" * 60)
    print("2. TIME SERIES CROSS-VALIDATION (3 splits)")
    print("=" * 60)
    
    cv_results = validation_framework.run_time_series_cv(
        params=base_params,
        n_splits=3,  # Reduced from 5
        test_size=0.2
    )
    all_results.extend(cv_results)
    
    # 3. Bootstrap Analysis (Reduced)
    print("\n" + "=" * 60)
    print("3. BOOTSTRAP ANALYSIS (100 samples)")
    print("=" * 60)
    
    bootstrap_results = validation_framework.run_bootstrap_analysis(
        params=base_params,
        n_bootstrap=100,  # Reduced from 500
        sample_size=None
    )
    all_results.extend(bootstrap_results)
    
    # 4. Monte Carlo (Reduced)
    print("\n" + "=" * 60)
    print("4. MONTE CARLO SIMULATIONS (50 per scenario)")
    print("=" * 60)
    
    mc_results = monte_carlo_framework.simulate_market_conditions(
        base_params=base_params,
        n_simulations=50,  # Reduced from 200
        scenarios=['normal', 'volatile', 'crisis']  # Reduced scenarios
    )
    all_results.extend(mc_results)
    
    # 5. Stress Tests (Reduced)
    print("\n" + "=" * 60)
    print("5. STRESS TESTS (20 per scenario)")
    print("=" * 60)
    
    stress_results = monte_carlo_framework.run_stress_tests(
        base_params=base_params
    )
    all_results.extend(stress_results)
    
    # 6. Analyze Results
    print("\n" + "=" * 60)
    print("6. VALIDATION RESULTS ANALYSIS")
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
    print(f"Positive return rate: {summary['positive_return_rate']:.1%}")
    print(f"Mean Sharpe ratio: {summary['mean_sharpe']:.2f}")
    print(f"Mean max drawdown: {summary['mean_max_drawdown']:.1%}")
    print(f"95% VaR: ${summary['confidence_intervals']['net_return_95'][0]:.2f}")
    
    # 7. Generate Single Consolidated Plot
    print("\n" + "=" * 60)
    print("7. GENERATING CONSOLIDATED PLOT")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"validation_summary_{timestamp}.png"
    
    # Create a single comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Strategy Validation Summary', fontsize=16)
    
    # Convert results to DataFrame
    df = pd.DataFrame([{
        'method': r.method,
        'net_return': r.net_return,
        'sharpe_ratio': r.sharpe_ratio,
        'max_drawdown': r.max_drawdown,
        'win_rate': r.win_rate
    } for r in all_results])
    
    # Net Return Distribution
    axes[0, 0].hist(df['net_return'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(df['net_return'].mean(), color='red', linestyle='--', label=f'Mean: ${df["net_return"].mean():.0f}')
    axes[0, 0].set_title('Net Return Distribution')
    axes[0, 0].set_xlabel('Net Return ($)')
    axes[0, 0].legend()
    
    # Method Comparison
    method_means = df.groupby('method')['net_return'].mean()
    axes[0, 1].bar(range(len(method_means)), method_means.values)
    axes[0, 1].set_title('Average Net Return by Method')
    axes[0, 1].set_xticks(range(len(method_means)))
    axes[0, 1].set_xticklabels(method_means.index, rotation=45)
    axes[0, 1].set_ylabel('Average Net Return ($)')
    
    # Sharpe Ratio vs Win Rate
    axes[1, 0].scatter(df['win_rate'], df['sharpe_ratio'], alpha=0.6)
    axes[1, 0].set_title('Win Rate vs Sharpe Ratio')
    axes[1, 0].set_xlabel('Win Rate')
    axes[1, 0].set_ylabel('Sharpe Ratio')
    
    # Return vs Drawdown
    axes[1, 1].scatter(df['max_drawdown'], df['net_return'], alpha=0.6)
    axes[1, 1].set_title('Max Drawdown vs Net Return')
    axes[1, 1].set_xlabel('Max Drawdown')
    axes[1, 1].set_ylabel('Net Return ($)')
    
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Consolidated plot saved to {plot_path}")
    else:
        print("Plot display disabled (SAVE_PLOTS=False)")
    plt.close()  # Don't show, just save
    
    # 8. Save Results
    print("\n" + "=" * 60)
    print("8. SAVING RESULTS")
    print("=" * 60)
    
    # Save summary to file
    summary_path = f"validation_summary_{timestamp}.txt"
    with open(summary_path, 'w') as f:
        f.write("STREAMLINED STRATEGY VALIDATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Strategy: ETHV/ETHA Cointegration\n")
        f.write(f"Frequency: {base_params['frequency']}\n\n")
        
        f.write("SUMMARY STATISTICS:\n")
        f.write(f"Total tests: {summary['total_tests']}\n")
        f.write(f"Methods: {summary['methods']}\n")
        f.write(f"Mean net return: ${summary['mean_net_return']:.2f}\n")
        f.write(f"Std net return: ${summary['std_net_return']:.2f}\n")
        f.write(f"Positive return rate: {summary['positive_return_rate']:.1%}\n")
        f.write(f"Mean Sharpe ratio: {summary['mean_sharpe']:.2f}\n")
        f.write(f"Mean max drawdown: {summary['mean_max_drawdown']:.1%}\n")
        f.write(f"95% VaR: ${summary['confidence_intervals']['net_return_95'][0]:.2f}\n\n")
        
        if 'monte_carlo_summary' in summary and summary['monte_carlo_summary']:
            f.write("MONTE CARLO RESULTS:\n")
            mc_sum = summary['monte_carlo_summary']
            f.write(f"Total simulations: {mc_sum.get('total_simulations', 0)}\n")
            f.write(f"Overall positive rate: {mc_sum.get('overall_positive_rate', 0):.1%}\n")
            f.write(f"95% VaR: ${mc_sum.get('overall_var_95', 0):.2f}\n")
    
    print(f"Summary saved to {summary_path}")
    
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE!")
    print("=" * 80)
    
    return all_results, summary


if __name__ == "__main__":
    run_fast_validation() 