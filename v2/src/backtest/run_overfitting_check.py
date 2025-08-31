"""
Simple Overfitting Detection Script
Focuses on key overfitting red flags without parameter optimization
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


def load_config():
    """Load configuration"""
    # Load environment variables
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


def run_overfitting_check():
    """Run focused overfitting detection tests"""
    print("=" * 80)
    print("OVERFITTING DETECTION ANALYSIS")
    print("=" * 80)
    
    # Load configuration and data
    config = load_config()
    price_data = get_price_data(config)
    base_config = create_backtest_config(config)
    
    if price_data.empty:
        print("ERROR: No data loaded. Check your configuration and data sources.")
        return
    
    print(f"Data loaded: {len(price_data)} observations")
    print(f"Date range: {price_data.index.min()} to {price_data.index.max()}")
    
    # Strategy parameters (more conservative settings to reduce overfitting)
    strategy_params = {
        'asset_1': {'symbol': config['strategy']['params']['asset_1'], 'type': 'equity'},
        'asset_2': {'symbol': config['strategy']['params']['asset_2'], 'type': 'equity'},
        'lookback_bars': 30,  # Increased from 20 for more stability
        'entry_z': 1.5,       # Reduced from 2.0 for less aggressive entries
        'exit_z': 0.5,        # Increased from 0.3 for earlier exits
        'frequency': config['strategy']['params']['frequency'],
        'poll_interval': config['strategy']['params']['poll_interval'],
        'fields': config['strategy']['params']['fields'],
        'hedge_ratio_method': 'ols'
    }
    
    print(f"\nStrategy Parameters (Conservative): {strategy_params}")
    
    # Test 1: Out-of-Sample Performance
    print("\n" + "=" * 60)
    print("TEST 1: OUT-OF-SAMPLE PERFORMANCE")
    print("=" * 60)
    
    # Split data into train/test (80/20) - longer out-of-sample period
    split_idx = int(len(price_data) * 0.8)
    train_data = price_data.iloc[:split_idx]
    test_data = price_data.iloc[split_idx:]
    
    print(f"Training period: {train_data.index.min()} to {train_data.index.max()}")
    print(f"Testing period: {test_data.index.min()} to {test_data.index.max()}")
    
    # Run backtest on training data
    train_strategy = CointegrationStrategy(**strategy_params)
    train_backtester = Backtester([train_strategy], train_data, base_config)
    train_results = train_backtester.run()
    
    # Run backtest on test data
    test_strategy = CointegrationStrategy(**strategy_params)
    test_backtester = Backtester([test_strategy], test_data, base_config)
    test_results = test_backtester.run()
    
    # Compare performance
    train_pnl = train_results['pnl'].iloc[-1] if len(train_results['pnl']) > 0 else 0
    test_pnl = test_results['pnl'].iloc[-1] if len(test_results['pnl']) > 0 else 0
    
    print(f"\nTraining PnL: ${train_pnl:,.2f}")
    print(f"Test PnL: ${test_pnl:,.2f}")
    print(f"Performance degradation: {((test_pnl - train_pnl) / train_pnl * 100):.1f}%" if train_pnl != 0 else "N/A")
    
    # Test 2: Parameter Sensitivity
    print("\n" + "=" * 60)
    print("TEST 2: PARAMETER SENSITIVITY")
    print("=" * 60)
    
    # Test slight parameter variations (more conservative ranges)
    variations = [
        {'lookback_bars': 25, 'entry_z': 1.3, 'exit_z': 0.6},
        {'lookback_bars': 35, 'entry_z': 1.7, 'exit_z': 0.4},
        {'lookback_bars': 30, 'entry_z': 1.2, 'exit_z': 0.7},
        {'lookback_bars': 30, 'entry_z': 1.8, 'exit_z': 0.3},
    ]
    
    results = []
    for i, variation in enumerate(variations):
        params = {**strategy_params, **variation}
        strategy = CointegrationStrategy(**params)
        backtester = Backtester([strategy], price_data, base_config)
        result = backtester.run()
        
        pnl = result['pnl'].iloc[-1] if len(result['pnl']) > 0 else 0
        results.append({
            'variation': i + 1,
            'params': variation,
            'pnl': pnl
        })
        
        print(f"Variation {i+1}: {variation} -> PnL: ${pnl:,.2f}")
    
    # Calculate parameter sensitivity
    base_pnl = train_pnl  # Use training performance as baseline
    sensitivities = []
    for result in results:
        degradation = (result['pnl'] - base_pnl) / base_pnl * 100 if base_pnl != 0 else 0
        sensitivities.append(degradation)
    
    avg_sensitivity = np.mean(sensitivities)
    print(f"\nAverage performance degradation: {avg_sensitivity:.1f}%")
    
    # Test 3: Time Period Stability
    print("\n" + "=" * 60)
    print("TEST 3: TIME PERIOD STABILITY")
    print("=" * 60)
    
    # Test on different time periods
    periods = [
        (0, 0.5),      # First half
        (0.5, 1.0),    # Second half
        (0.25, 0.75),  # Middle half
    ]
    
    period_results = []
    for i, (start_frac, end_frac) in enumerate(periods):
        start_idx = int(len(price_data) * start_frac)
        end_idx = int(len(price_data) * end_frac)
        period_data = price_data.iloc[start_idx:end_idx]
        
        strategy = CointegrationStrategy(**strategy_params)
        backtester = Backtester([strategy], period_data, base_config)
        result = backtester.run()
        
        pnl = result['pnl'].iloc[-1] if len(result['pnl']) > 0 else 0
        period_results.append({
            'period': f"Period {i+1}",
            'date_range': f"{period_data.index.min().date()} to {period_data.index.max().date()}",
            'pnl': pnl
        })
        
        print(f"Period {i+1}: {period_data.index.min().date()} to {period_data.index.max().date()} -> PnL: ${pnl:,.2f}")
    
    # Calculate stability
    pnls = [r['pnl'] for r in period_results]
    stability = np.std(pnls) / np.mean(pnls) if np.mean(pnls) != 0 else float('inf')
    print(f"\nPerformance stability (CV): {stability:.2f}")
    
    # Test 4: Random Subset Performance
    print("\n" + "=" * 60)
    print("TEST 4: RANDOM SUBSET PERFORMANCE")
    print("=" * 60)
    
    # Test on random 50% subsets
    np.random.seed(42)  # For reproducibility
    subset_results = []
    
    for i in range(5):
        # Random 50% subset
        subset_mask = np.random.choice([True, False], size=len(price_data), p=[0.5, 0.5])
        subset_data = price_data[subset_mask]
        
        if len(subset_data) > 100:  # Only test if enough data
            strategy = CointegrationStrategy(**strategy_params)
            backtester = Backtester([strategy], subset_data, base_config)
            result = backtester.run()
            
            pnl = result['pnl'].iloc[-1] if len(result['pnl']) > 0 else 0
            subset_results.append(pnl)
            
            print(f"Random subset {i+1}: {len(subset_data)} observations -> PnL: ${pnl:,.2f}")
    
    if subset_results:
        subset_mean = np.mean(subset_results)
        subset_std = np.std(subset_results)
        print(f"\nRandom subset performance: ${subset_mean:,.2f} ± ${subset_std:,.2f}")
    
    # OVERFITTING ASSESSMENT
    print("\n" + "=" * 80)
    print("OVERFITTING ASSESSMENT")
    print("=" * 80)
    
    # Red flags
    red_flags = []
    
    # Flag 1: Large performance degradation in out-of-sample
    if train_pnl != 0 and test_pnl < train_pnl * 0.5:
        red_flags.append("LARGE: Out-of-sample performance < 50% of in-sample")
    elif train_pnl != 0 and test_pnl < train_pnl * 0.8:
        red_flags.append("MEDIUM: Out-of-sample performance < 80% of in-sample")
    
    # Flag 2: High parameter sensitivity
    if abs(avg_sensitivity) > 50:
        red_flags.append("LARGE: Performance highly sensitive to parameter changes")
    elif abs(avg_sensitivity) > 20:
        red_flags.append("MEDIUM: Performance moderately sensitive to parameter changes")
    
    # Flag 3: Poor time period stability
    if stability > 1.0:
        red_flags.append("LARGE: Performance varies significantly across time periods")
    elif stability > 0.5:
        red_flags.append("MEDIUM: Performance shows some instability across time periods")
    
    # Flag 4: Random subset inconsistency
    if subset_results and np.std(subset_results) > np.mean(subset_results) * 0.5:
        red_flags.append("MEDIUM: Performance inconsistent across random data subsets")
    
    # Summary
    if not red_flags:
        print("NO MAJOR OVERFITTING RED FLAGS DETECTED")
        print("Your strategy appears to be reasonably robust.")
    else:
        print("OVERFITTING RED FLAGS DETECTED:")
        for flag in red_flags:
            print(f"  • {flag}")
        print("\nRecommendations:")
        print("  • Consider using more conservative parameters")
        print("  • Test on longer out-of-sample periods")
        print("  • Reduce strategy complexity")
        print("  • Use ensemble methods")
    
    print(f"\nDetailed Results:")
    print(f"  • In-sample PnL: ${train_pnl:,.2f}")
    print(f"  • Out-of-sample PnL: ${test_pnl:,.2f}")
    print(f"  • Parameter sensitivity: {avg_sensitivity:.1f}%")
    print(f"  • Time stability (CV): {stability:.2f}")
    if subset_results:
        print(f"  • Random subset consistency: {np.std(subset_results):.1f} std dev")
    
    return {
        'train_pnl': train_pnl,
        'test_pnl': test_pnl,
        'parameter_sensitivity': avg_sensitivity,
        'time_stability': stability,
        'red_flags': red_flags
    }


if __name__ == "__main__":
    run_overfitting_check() 