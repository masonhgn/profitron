#!/usr/bin/env python3
"""
hyperparameter optimization script for the cointegration strategy
"""

import yaml
import subprocess
import sys
import pandas as pd
from datetime import datetime

def run_backtest_with_params(lookback, entry_z, exit_z):
    """run backtest with given parameters and extract results"""
    
    # load current config
    config_path = "src/core/config/Engine.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # update strategy parameters
    config['strategy']['params']['lookback_bars'] = lookback
    config['strategy']['params']['entry_z'] = entry_z
    config['strategy']['params']['exit_z'] = exit_z
    
    # save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # run backtest
    result = subprocess.run([sys.executable, "src/main.py"],
                          capture_output=True, text=True, timeout=300)
    
    if result.returncode == 0:
        # parse output to extract metrics
        output = result.stdout
        
        # extract metrics from output
        lines = output.split('\n')
        
        # initialize default values
        total_return = 0.0
        sharpe_ratio = 0.0
        max_drawdown = 0.0
        
        for line in lines:
            if line.strip() and not line.startswith('['):
                if 'total return' in line.lower():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.replace('.', '').replace('-', '').isdigit():
                            total_return = float(part)
                            break
                elif 'sharpe' in line.lower():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.replace('.', '').replace('-', '').isdigit():
                            sharpe_ratio = float(part)
                            break
        
        return {
            'success': True,
            'lookback': lookback,
            'entry_z': entry_z,
            'exit_z': exit_z,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': 0.5,  # placeholder
            'profit_factor': 1.0,  # placeholder
        }
    else:
        return {
            'success': False,
            'lookback': lookback,
            'entry_z': entry_z,
            'exit_z': exit_z,
            'error': result.stderr
        }

def main():
    """main optimization function"""
    
    print("starting hyperparameter optimization...")
    
    # define parameter ranges to test
    lookback_values = [10, 15, 20, 25, 30]
    entry_z_values = [1.0, 1.5, 2.0]
    exit_z_values = [0.2, 0.5]
    
    results = []
    
    # test all combinations
    for lookback in lookback_values:
        for entry_z in entry_z_values:
            for exit_z in exit_z_values:
                print(f"\ntesting: lookback={lookback}, entry_z={entry_z}, exit_z={exit_z}")
                
                result = run_backtest_with_params(lookback, entry_z, exit_z)
                results.append(result)
                
                if result['success']:
                    print(f"  success: return: {result['total_return']:.2f}")
                else:
                    print(f"  failed: {result.get('error', 'unknown error')}")
    
    # analyze results
    df = pd.DataFrame(results)
    
    # filter successful runs
    valid_results = df[df['success'] == True].copy()
    
    if len(valid_results) == 0:
        print("\nno valid results found!")
        return
    
    # sort by total return
    valid_results = valid_results.sort_values(by='total_return', ascending=False)
    
    print("\nbest performing parameters:")
    print(valid_results[['lookback', 'entry_z', 'exit_z', 'total_return']].head(10))
    
    # get best result
    best_result = valid_results.iloc[0]
    
    print(f"\nbest configuration:")
    print(f"lookback: {best_result['lookback']}")
    print(f"entry_z: {best_result['entry_z']}")
    print(f"exit_z: {best_result['exit_z']}")
    print(f"total return: {best_result['total_return']:.2f}")
    
    # save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    valid_results.to_csv(f'optimization_results_{timestamp}.csv', index=False)
    print(f"\nresults saved to optimization_results_{timestamp}.csv")

if __name__ == "__main__":
    main() 