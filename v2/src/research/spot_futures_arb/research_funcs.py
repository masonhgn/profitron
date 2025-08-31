import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# --- Data Loading Functions ---
def load_spot_data(filename='spot_gold_data.csv'):
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.rename(columns={'Datetime': 'timestamp', 'Close': 'spot_price'})
    return df[['timestamp', 'spot_price']]

def load_futures_data(filename='futures_long_format.csv', contract=None):
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if contract:
        df = df[df['symbol'] == contract]
    return df[['timestamp', 'symbol', 'price']].rename(columns={'price': 'future_price'})

# --- Data Alignment ---
def align_data(spot_df, futures_df):
    merged = pd.merge_asof(
        futures_df.sort_values('timestamp'),
        spot_df.sort_values('timestamp'),
        on='timestamp',
        direction='backward'
    )
    merged = merged.dropna(subset=['spot_price', 'future_price'])
    return merged

# --- Research Functions ---
def calculate_hedge_ratio(df):
    X = sm.add_constant(df['future_price'])
    y = df['spot_price']
    model = sm.OLS(y, X).fit()
    return model.params['future_price'], model

def calculate_spread(df, hedge_ratio):
    return df['spot_price'] - hedge_ratio * df['future_price']

def adf_test(series):
    result = adfuller(series.dropna())
    return {
        'adf_stat': result[0],
        'p_value': result[1],
        'crit_values': result[4]
    }

# --- Visualization ---
def plot_spread(df, spread, title='Spot-Futures Spread'):
    plt.figure(figsize=(12, 4))
    plt.plot(df['timestamp'], spread, label='Spread')
    plt.axhline(spread.mean(), color='red', linestyle='--', label='Mean')
    plt.title(title)
    plt.xlabel('Timestamp')
    plt.ylabel('Spread')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # Load data
    spot_df = load_spot_data()
    # Pick a contract (e.g., most active or front month)
    contract = 'GCQ5'  # Change as needed
    futures_df = load_futures_data(contract=contract)
    
    # Align data
    df = align_data(spot_df, futures_df)
    
    # Calculate hedge ratio
    hedge_ratio, model = calculate_hedge_ratio(df)
    print(f"Hedge ratio (spot ~ future): {hedge_ratio:.4f}")
    
    # Calculate spread
    spread = calculate_spread(df, hedge_ratio)
    
    # Plot spread
    plot_spread(df, spread)
    
    # ADF test
    adf_result = adf_test(spread)
    print(f"ADF Statistic: {adf_result['adf_stat']:.4f}")
    print(f"p-value: {adf_result['p_value']:.4g}")
    for key, value in adf_result['crit_values'].items():
        print(f"Critical Value {key}: {value}")

if __name__ == "__main__":
    main() 