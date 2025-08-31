#!/usr/bin/env python3
"""
script to validate cointegration for different asset pairs
run this before trading to ensure the pairs are suitable for cointegration strategies
"""

import yaml
from src.data_collection.DataManager import DataManager
from strategies.equities.mean_reversion.CointegrationStrategy import CointegrationStrategy
from src.utils.utilities import load_environment, load_yaml_config, resolve_config_values

def validate_asset_pair(asset_1: dict, asset_2: dict, start_date: str = "2020-01-01", end_date: str = "2024-12-31"):
    """
    validate cointegration for a specific asset pair
    
    args:
        asset_1: first asset dict with symbol and type
        asset_2: second asset dict with symbol and type
        start_date: start date for validation
        end_date: end date for validation
    """
    print(f"\n{'='*60}")
    print(f"validating: {asset_1['symbol']} vs {asset_2['symbol']}")
    print(f"{'='*60}")
    
    # load config and initialize data manager
    load_environment()
    config = load_yaml_config("src/core/config/Engine.yaml")
    config = resolve_config_values(config)
    
    dm = DataManager(config)
    
    # get historical data
    try:
        data = dm.get_price_data(
            assets=[asset_1, asset_2],
            start_date=start_date,
            end_date=end_date,
            frequency="1d",
            fields=["close"]
        )
        
        if data.empty:
            print(f"no data available for {asset_1['symbol']}/{asset_2['symbol']}")
            return False
            
        print(f"data range: {data.index.min()} to {data.index.max()}")
        print(f"data points: {len(data)}")
        
    except Exception as e:
        print(f"error getting data: {e}")
        return False
    
    # create strategy instance for validation
    strategy_params = {
        'asset_1': asset_1,
        'asset_2': asset_2,
        'lookback': 20,
        'entry_z': 2.0,
        'exit_z': 0.5,
        'frequency': "1d",
        'fields': ["close"],
        'poll_interval': 60,
        'hedge_ratio_method': "ols",
        'cointegration_pvalue_threshold': 0.05,
        'max_position_size': 0.5,
        'min_correlation': 0.7,
        'max_spread_volatility': 0.1,
        'risk_free_rate': 0.02
    }
    
    strategy = CointegrationStrategy(**strategy_params)
    
    # validate cointegration
    try:
        validation_result = strategy.validate_cointegration(data)
        
        if validation_result['is_valid']:
            print(f"validation passed - {asset_1['symbol']}/{asset_2['symbol']} is suitable for cointegration trading")
            print(f"   correlation: {validation_result['correlation']:.4f}")
            print(f"   r-squared: {validation_result['r_squared']:.4f}")
            print(f"   hedge ratio: {validation_result['hedge_ratio']:.4f}")
            print(f"   spread volatility: {validation_result['spread_volatility']:.4f}")
            return True
        else:
            print(f"validation failed - {asset_1['symbol']}/{asset_2['symbol']} is not suitable for cointegration trading")
            if not validation_result['is_cointegrated']:
                print(f"   not cointegrated")
            if not validation_result['is_correlated']:
                print(f"   insufficient correlation: {validation_result['correlation']:.4f}")
            return False
            
    except Exception as e:
        print(f"error during validation: {e}")
        return False

def main():
    """test multiple asset pairs"""
    
    # define pairs to test
    pairs_to_test = [
        # vanguard etfs
        ({'symbol': 'VOO', 'type': 'equity'}, {'symbol': 'VTI', 'type': 'equity'}),
        
        # technology etfs
        ({'symbol': 'QQQ', 'type': 'equity'}, {'symbol': 'TQQQ', 'type': 'equity'}),
        
        # oil etfs
        ({'symbol': 'USO', 'type': 'equity'}, {'symbol': 'XOP', 'type': 'equity'}),
        
        # bond etfs
        ({'symbol': 'TLT', 'type': 'equity'}, {'symbol': 'IEI', 'type': 'equity'}),
        
        # international etfs
        ({'symbol': 'EFA', 'type': 'equity'}, {'symbol': 'EWJ', 'type': 'equity'}),
        
        # s&p 500 vs nasdaq
        ({'symbol': 'SPY', 'type': 'equity'}, {'symbol': 'QQQ', 'type': 'equity'}),
        
        # russell 2000 vs s&p 500
        ({'symbol': 'IWM', 'type': 'equity'}, {'symbol': 'SPY', 'type': 'equity'}),
        
        # energy sector vs exxon
        ({'symbol': 'XLE', 'type': 'equity'}, {'symbol': 'XOM', 'type': 'equity'}),
        
        # financial sector vs jpmorgan
        ({'symbol': 'XLF', 'type': 'equity'}, {'symbol': 'JPM', 'type': 'equity'}),
        
        # gold etfs
        ({'symbol': 'GLD', 'type': 'equity'}, {'symbol': 'IAU', 'type': 'equity'}),
    ]
    
    print("cointegration validation for asset pairs")
    print("=" * 60)
    
    valid_pairs = []
    
    for asset_1, asset_2 in pairs_to_test:
        is_valid = validate_asset_pair(asset_1, asset_2)
        if is_valid:
            valid_pairs.append((asset_1, asset_2))
    
    print(f"\n{'='*60}")
    print(f"summary: {len(valid_pairs)} out of {len(pairs_to_test)} pairs are suitable for cointegration trading")
    print(f"{'='*60}")
    
    if valid_pairs:
        print("\nvalid pairs for trading:")
        for i, (asset_1, asset_2) in enumerate(valid_pairs, 1):
            print(f"   {i}. {asset_1['symbol']} / {asset_2['symbol']}")
    else:
        print("\nno valid pairs found. consider:")
        print("   - adjusting correlation threshold")
        print("   - using different time periods")
        print("   - testing different asset classes")

if __name__ == "__main__":
    main() 