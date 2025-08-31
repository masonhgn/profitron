# strategies/equities/mean_reversion/CointegrationStrategy.py

from dataclasses import dataclass
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from typing import List, Dict, Optional
from strategies.TradingStrategy import TradingStrategy
from src.signals.Signal import Signal
import os

@dataclass
class CointegrationStrategy(TradingStrategy):
    """
    see explore.ipynb
    """
    asset_1: Dict[str, str]  # {'symbol': 'ETH', 'type': 'crypto'}
    asset_2: Dict[str, str]
    lookback_bars: int
    entry_z: float
    exit_z: float
    frequency: str = "1h"
    poll_interval: int = 60
    fields: Optional[List[str]] = None
    hedge_ratio_method: str = "ols"
    cointegration_pvalue_threshold: float = 0.05  # ADF test p-value threshold
    max_position_size: float = 0.5  # Maximum position size per asset (50%)
    min_correlation: float = 0.7  # Minimum correlation threshold
    max_spread_volatility: float = 0.1  # Maximum spread volatility (10%)
    risk_free_rate: float = 0.02  # Risk-free rate for Sharpe ratio calculation
    
    # New parameters to reduce overfitting
    hedge_ratio_window_multiplier: float = 3.0  # Use 3x lookback for hedge ratio
    max_position_duration: int = 48  # Maximum position duration in hours
    min_hedge_ratio_stability: float = 0.8  # Minimum R-squared for hedge ratio stability
    adaptive_thresholds: bool = True  # Use adaptive thresholds
    regime_detection: bool = True  # Enable regime detection

    def __post_init__(self):
        self.current_position = "flat"  # can be 'long_spread', 'short_spread', or 'flat'
        self.position_start_time = None  # Track when position was opened
        self.hedge_ratio_history = []  # Track hedge ratio stability
        self.regime_stability_score = 1.0  # Market regime stability score

    def get_assets(self) -> List[Dict[str, str]]:
        return [self.asset_1, self.asset_2]

    def get_frequency(self) -> str:
        return self.frequency

    def get_lookback_window(self) -> int:
        return self.lookback_bars

    def get_fields(self) -> List[str]:
        return self.fields or ["close"]

    def get_poll_interval(self) -> int:
        return self.poll_interval

    def _test_cointegration(self, y: pd.Series, x: pd.Series, hedge_ratio: float) -> bool:
        """
        Test if the residuals are stationary (cointegrated)
        
        Args:
            y: Dependent variable series
            x: Independent variable series  
            hedge_ratio: OLS hedge ratio
            
        Returns:
            bool: True if cointegrated (stationary residuals)
        """
        residuals = y - hedge_ratio * x
        
        # Perform Augmented Dickey-Fuller test
        adf_result = adfuller(residuals.dropna())
        p_value = adf_result[1]
        
        # Check if residuals are stationary (p-value < threshold)
        is_cointegrated = bool(p_value < self.cointegration_pvalue_threshold)
        
        if not is_cointegrated:
            print(f"[Cointegration] Assets not cointegrated (p-value: {p_value:.4f} > {self.cointegration_pvalue_threshold})")
        
        return is_cointegrated

    def _check_correlation(self, y: pd.Series, x: pd.Series) -> bool:
        """
        Check if assets have sufficient correlation
        
        Args:
            y: First asset series
            x: Second asset series
            
        Returns:
            bool: True if correlation is above threshold
        """
        correlation = y.corr(x)
        is_correlated = abs(correlation) >= self.min_correlation
        
        if not is_correlated:
            print(f"[Correlation] Assets insufficiently correlated (corr: {correlation:.4f} < {self.min_correlation})")
        
        return is_correlated

    def _calculate_adaptive_thresholds(self, data: pd.DataFrame) -> tuple:
        """
        Calculate adaptive thresholds based on recent market conditions
        
        Args:
            data: Historical price data
            
        Returns:
            tuple: (adaptive_entry_z, adaptive_exit_z, adaptive_correlation)
        """
        if not self.adaptive_thresholds:
            return self.entry_z, self.exit_z, self.min_correlation
        
        # Calculate recent volatility
        recent_vol = data.std().mean()
        historical_vol = data.expanding().std().mean().iloc[-1]
        
        # Adjust thresholds based on volatility
        vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1.0
        
        # More volatile = more conservative thresholds
        adaptive_entry_z = self.entry_z * (1 + (vol_ratio - 1) * 0.5)
        adaptive_exit_z = self.exit_z * (1 + (vol_ratio - 1) * 0.3)
        adaptive_correlation = min(self.min_correlation * 1.1, 0.9)  # Slightly higher correlation requirement
        
        return adaptive_entry_z, adaptive_exit_z, adaptive_correlation

    def _detect_market_regime(self, data: pd.DataFrame) -> float:
        """
        Detect market regime stability
        
        Args:
            data: Historical price data
            
        Returns:
            float: Regime stability score (0-1, higher = more stable)
        """
        if not self.regime_detection:
            return 1.0
        
        try:
            # Calculate rolling volatility
            rolling_vol = data.rolling(window=20).std()
            vol_mean = rolling_vol.mean().mean()  # Average across all columns
            vol_std = rolling_vol.std().mean()    # Standard deviation across time
            
            if vol_mean > 0:
                vol_stability = 1.0 - (vol_std / vol_mean)
            else:
                vol_stability = 0.0
            
            # Calculate rolling correlation stability
            if len(data.columns) >= 2:
                rolling_corr = data.iloc[:, 0].rolling(window=20).corr(data.iloc[:, 1])
                corr_std = rolling_corr.std()
                corr_stability = 1.0 - corr_std if not pd.isna(corr_std) else 0.0
            else:
                corr_stability = 0.0
            
            # Combine stability metrics
            regime_score = (vol_stability + corr_stability) / 2
            regime_score = np.clip(regime_score, 0.0, 1.0)
            
            return regime_score
            
        except Exception as e:
            print(f"[Regime Detection] Error: {e}")
            return 1.0

    def _validate_hedge_ratio_stability(self, hedge_ratio: float, r_squared: float) -> bool:
        """
        Validate hedge ratio stability
        
        Args:
            hedge_ratio: Current hedge ratio
            r_squared: R-squared from OLS regression
            
        Returns:
            bool: True if hedge ratio is stable
        """
        # Check R-squared threshold
        if r_squared < self.min_hedge_ratio_stability:
            print(f"[Hedge Ratio] Unstable relationship (R²: {r_squared:.3f} < {self.min_hedge_ratio_stability})")
            return False
        
        # Check hedge ratio stability over time
        if len(self.hedge_ratio_history) >= 5:
            recent_ratios = self.hedge_ratio_history[-5:]
            ratio_std = np.std(recent_ratios)
            ratio_mean = np.mean(recent_ratios)
            
            if ratio_mean != 0:
                coefficient_of_variation = ratio_std / abs(ratio_mean)
                if coefficient_of_variation > 0.2:  # More than 20% variation
                    print(f"[Hedge Ratio] High variation in hedge ratio (CV: {coefficient_of_variation:.3f})")
                    return False
        
        return True

    def _check_position_duration(self, current_time: pd.Timestamp) -> bool:
        """
        Check if position has been held too long
        
        Args:
            current_time: Current timestamp
            
        Returns:
            bool: True if position should be closed due to duration
        """
        if self.current_position == "flat" or self.position_start_time is None:
            return False
        
        duration_hours = (current_time - self.position_start_time).total_seconds() / 3600
        
        if duration_hours > self.max_position_duration:
            print(f"[Position Duration] Position held for {duration_hours:.1f} hours, closing due to duration limit")
            return True
        
        return False

    def _calculate_position_size(self, zscore: float, spread_std: float) -> float:
        """
        Calculate position size with risk management
        
        Args:
            zscore: Current z-score
            spread_std: Spread standard deviation
            
        Returns:
            float: Position size as fraction of capital
        """
        # Base position size based on z-score
        base_weight = self.max_position_size
        
        # Z-score multiplier (higher conviction = larger position)
        z_multiplier = min(abs(zscore) / self.entry_z, 2.0)
        
        # Volatility adjustment (higher vol = smaller position)
        vol_multiplier = 1.0 / (1.0 + spread_std)
        vol_multiplier = min(vol_multiplier, 1.0)
        
        # Risk adjustment based on spread volatility
        if spread_std > self.max_spread_volatility:
            risk_multiplier = self.max_spread_volatility / spread_std
        else:
            risk_multiplier = 1.0
        
        # Regime stability adjustment
        regime_multiplier = self.regime_stability_score
        
        # Calculate final position size
        target_pct = base_weight * z_multiplier * vol_multiplier * risk_multiplier * regime_multiplier
        
        # Apply safety caps
        target_pct = np.clip(target_pct, 0.0, self.max_position_size)
        
        return target_pct

    def validate_cointegration(self, data: pd.DataFrame) -> dict:
        """
        Validate cointegration and correlation for the asset pair using historical data
        
        Args:
            data: Historical price data with columns [asset_1_close, asset_2_close]
            
        Returns:
            dict: Validation results including cointegration status, correlation, and hedge ratio
        """
        sym_1 = self.asset_1["symbol"]
        sym_2 = self.asset_2["symbol"]
        col_1 = f"{sym_1}_close"
        col_2 = f"{sym_2}_close"

        if col_1 not in data.columns or col_2 not in data.columns:
            raise ValueError(f"missing close columns: {col_1}, {col_2}")

        # Prepare data
        close_data = data[[col_1, col_2]].copy()
        close_data.columns = ["asset_1", "asset_2"]
        close_data = close_data.dropna()
        
        if len(close_data) < 100:  # Need sufficient historical data
            return {
                'is_valid': False,
                'error': 'Insufficient historical data (need at least 100 observations)'
            }

        # Calculate log returns
        log_data = np.log(close_data)
        if not isinstance(log_data, pd.DataFrame):
            log_data = pd.DataFrame(log_data, columns=close_data.columns, index=close_data.index)
        
        # Calculate hedge ratio using full historical data
        y = log_data["asset_1"].dropna()
        x = log_data["asset_2"].dropna()
        
        # Ensure alignment
        common_index = y.index.intersection(x.index)
        y = y.loc[common_index]
        x = x.loc[common_index]
        
        x_const = sm.add_constant(x)
        model = sm.OLS(y, x_const).fit()
        hedge_ratio = model.params.iloc[1]
        
        # Test cointegration
        is_cointegrated = self._test_cointegration(y, x, hedge_ratio)
        
        # Test correlation
        is_correlated = self._check_correlation(y, x)
        
        # Calculate spread statistics
        spread = y - hedge_ratio * x
        spread_std = spread.std()
        spread_mean = spread.mean()
        
        # Calculate additional metrics
        correlation = y.corr(x)
        r_squared = model.rsquared
        
        validation_result = {
            'is_valid': is_cointegrated and is_correlated,
            'is_cointegrated': is_cointegrated,
            'is_correlated': is_correlated,
            'correlation': correlation,
            'hedge_ratio': hedge_ratio,
            'r_squared': r_squared,
            'spread_mean': spread_mean,
            'spread_std': spread_std,
            'spread_volatility': spread_std,
            'num_observations': len(close_data),
            'date_range': {
                'start': close_data.index.min(),
                'end': close_data.index.max()
            }
        }
        
        # Print validation summary
        print(f"\n=== COINTEGRATION VALIDATION FOR {sym_1}/{sym_2} ===")
        print(f"Correlation: {correlation:.4f} (threshold: {self.min_correlation})")
        print(f"R-squared: {r_squared:.4f}")
        print(f"Hedge ratio: {hedge_ratio:.4f}")
        print(f"Spread volatility: {spread_std:.4f}")
        print(f"Cointegrated: {is_cointegrated}")
        print(f"Sufficiently correlated: {is_correlated}")
        print(f"Overall valid: {validation_result['is_valid']}")
        
        return validation_result

    def on_event(self, data: pd.DataFrame) -> List[Signal]:
        """
        generate trade signals based on z-score of the cointegration spread
        """
        timestamp = pd.Timestamp.now()
        sym_1 = self.asset_1["symbol"]
        sym_2 = self.asset_2["symbol"]
        col_1 = f"{sym_1}_close"
        col_2 = f"{sym_2}_close"

        print(f"[{timestamp}] [CointegrationStrategy] Analyzing {sym_1}/{sym_2} pair...")

        if col_1 not in data.columns or col_2 not in data.columns:
            raise ValueError(f"missing close columns: {col_1}, {col_2}")

        close_data = data[[col_1, col_2]].copy()
        close_data.columns = ["asset_1", "asset_2"]
        close_data = close_data.dropna()
        
        if len(close_data) <= self.lookback_bars + 1:
            print(f"[{timestamp}] [CointegrationStrategy] Insufficient data: {len(close_data)} observations")
            return []
            
        if close_data["asset_1"].std() == 0 or close_data["asset_2"].std() == 0:
            print(f"[{timestamp}] [CointegrationStrategy] Insufficient price variation detected")
            return []

        # Calculate adaptive thresholds
        adaptive_entry_z, adaptive_exit_z, adaptive_correlation = self._calculate_adaptive_thresholds(close_data)
        
        # Detect market regime
        self.regime_stability_score = self._detect_market_regime(close_data)
        
        # Check position duration limit
        if self._check_position_duration(timestamp):
            signals = []
            if self.current_position != "flat":
                print(f"[{timestamp}] [CointegrationStrategy] SIGNAL: Exit position (duration limit)")
                signals.append(Signal(symbol=sym_1, action="sell", target_pct=0.0, asset_type=self.asset_1["type"]))
                signals.append(Signal(symbol=sym_2, action="sell", target_pct=0.0, asset_type=self.asset_2["type"]))
                self.current_position = "flat"
                self.position_start_time = None
            return signals

        log_data = np.log(close_data)
        if not isinstance(log_data, pd.DataFrame):
            log_data = pd.DataFrame(log_data, columns=close_data.columns, index=close_data.index)
        
        t = len(log_data) - 1
        
        # Use longer window for hedge ratio calculation to reduce overfitting
        hedge_ratio_window = min(int(self.lookback_bars * self.hedge_ratio_window_multiplier), len(log_data) - 1)
        hedge_ratio_data = log_data.iloc[(t - hedge_ratio_window):t]
        
        y = hedge_ratio_data["asset_1"].dropna()
        x = hedge_ratio_data["asset_2"].dropna()
        
        if len(y) < 20 or len(x) < 20:  # Need more data for stable hedge ratio
            print(f"[{timestamp}] [CointegrationStrategy] Insufficient data for hedge ratio calculation")
            return []
            
        common_index = y.index.intersection(x.index)
        if len(common_index) < 20:
            print(f"[{timestamp}] [CointegrationStrategy] Insufficient common data for hedge ratio")
            return []
            
        y = y.loc[common_index]
        x = x.loc[common_index]
        
        # Calculate hedge ratio with stability validation
        if hasattr(self, "hedge_ratio_method") and self.hedge_ratio_method == "ols":
            x_const = sm.add_constant(x)
            model = sm.OLS(y, x_const).fit()
            hedge_ratio = model.params.iloc[1]
            r_squared = model.rsquared
            
            # Validate hedge ratio stability
            if not self._validate_hedge_ratio_stability(hedge_ratio, r_squared):
                print(f"[{timestamp}] [CointegrationStrategy] Hedge ratio unstable, skipping trade")
                return []
            
            # Track hedge ratio history
            self.hedge_ratio_history.append(hedge_ratio)
            if len(self.hedge_ratio_history) > 20:  # Keep last 20 values
                self.hedge_ratio_history.pop(0)
                
        else:
            raise NotImplementedError("Only OLS hedge ratio is implemented!")
        
        # Calculate spread using current data point
        current_spread = log_data.iloc[t]["asset_1"] - hedge_ratio * log_data.iloc[t]["asset_2"]
        
        # Calculate spread statistics using lookback window for z-score
        window = log_data.iloc[(t - self.lookback_bars):t]
        y_window = window["asset_1"].dropna()
        x_window = window["asset_2"].dropna()
        
        common_index_window = y_window.index.intersection(x_window.index)
        if len(common_index_window) < 10:
            print(f"[{timestamp}] [CointegrationStrategy] Insufficient data for spread calculation")
            return []
            
        y_window = y_window.loc[common_index_window]
        x_window = x_window.loc[common_index_window]
        
        spread_series = y_window - hedge_ratio * x_window
        spread_std = spread_series.std()
        
        if np.isnan(spread_std) or spread_std == 0:
            print(f"[Z-score] Cannot compute z-score — std={spread_std}")
            return []
            
        zscore = (current_spread - spread_series.mean()) / spread_std
        zscore = float(zscore)
        
        print(f"[{timestamp}] [CointegrationStrategy] Z-score: {zscore:.3f} (entry: {adaptive_entry_z:.2f}, exit: {adaptive_exit_z:.2f})")
        print(f"[{timestamp}] [CointegrationStrategy] Current position: {self.current_position}")
        print(f"[{timestamp}] [CointegrationStrategy] Regime stability: {self.regime_stability_score:.3f}")
        
        signals: List[Signal] = []
        target_pct = self._calculate_position_size(zscore, spread_std)
        
        # Stateful logic: only emit entry/exit if position change is needed
        if zscore > adaptive_entry_z:
            if self.current_position != "short_spread":
                print(f"[{timestamp}] [CointegrationStrategy] SIGNAL: Short spread (z-score: {zscore:.3f} > {adaptive_entry_z:.2f})")
                signals.append(Signal(symbol=sym_1, action="sell", target_pct=target_pct, asset_type=self.asset_1["type"]))
                signals.append(Signal(symbol=sym_2, action="buy", target_pct=target_pct, asset_type=self.asset_2["type"]))
                self.current_position = "short_spread"
                self.position_start_time = timestamp
        elif zscore < -adaptive_entry_z:
            if self.current_position != "long_spread":
                print(f"[{timestamp}] [CointegrationStrategy] SIGNAL: Long spread (z-score: {zscore:.3f} < -{adaptive_entry_z:.2f})")
                signals.append(Signal(symbol=sym_1, action="buy", target_pct=target_pct, asset_type=self.asset_1["type"]))
                signals.append(Signal(symbol=sym_2, action="sell", target_pct=target_pct, asset_type=self.asset_2["type"]))
                self.current_position = "long_spread"
                self.position_start_time = timestamp
        elif abs(zscore) < adaptive_exit_z:
            if self.current_position != "flat":
                print(f"[{timestamp}] [CointegrationStrategy] SIGNAL: Exit position (z-score: {zscore:.3f} < {adaptive_exit_z:.2f})")
                signals.append(Signal(symbol=sym_1, action="sell", target_pct=0.0, asset_type=self.asset_1["type"]))
                signals.append(Signal(symbol=sym_2, action="sell", target_pct=0.0, asset_type=self.asset_2["type"]))
                self.current_position = "flat"
                self.position_start_time = None
        else:
            print(f"[{timestamp}] [CointegrationStrategy] No signal (z-score: {zscore:.3f} within bounds)")
            
        return signals