"""
Comprehensive Strategy Validation Framework
Includes walk-forward analysis, cross-validation, bootstrap testing, and parameter stability analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

from .Backtester import Backtester, BacktestConfig
from strategies.TradingStrategy import TradingStrategy
from ..signals.Signal import Signal


@dataclass
class ValidationResult:
    """Container for validation results"""
    method: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_trade: float
    num_trades: int
    total_costs: float
    net_return: float
    params: Dict
    pnl_series: pd.Series
    metadata: Dict


class ValidationFramework:
    """Comprehensive strategy validation framework"""
    
    def __init__(self, strategy_class, price_data: pd.DataFrame, base_config: BacktestConfig):
        self.strategy_class = strategy_class
        self.price_data = price_data
        self.base_config = base_config
        self.results = []
        
    def run_walk_forward_analysis(
        self, 
        param_combinations: List[Dict],
        train_days: int = 252,  # 1 year
        test_days: int = 63,    # 3 months
        step_days: int = 21,    # 1 month
        min_test_days: int = 30
    ) -> List[ValidationResult]:
        """
        Walk-forward analysis with parameter optimization
        
        Args:
            param_combinations: List of parameter dictionaries to test
            train_days: Training window size in days
            test_days: Testing window size in days
            step_days: Step size between windows
            min_test_days: Minimum test period required
        """
        print(f"Running Walk-Forward Analysis...")
        print(f"Training window: {train_days} days, Test window: {test_days} days")
        print(f"Step size: {step_days} days, Parameter combinations: {len(param_combinations)}")
        
        results = []
        total_windows = 0
        
        # Convert days to timestamps
        train_timedelta = pd.Timedelta(days=train_days)
        test_timedelta = pd.Timedelta(days=test_days)
        step_timedelta = pd.Timedelta(days=step_days)
        
        start_date = self.price_data.index.min()
        end_date = self.price_data.index.max()
        
        current_start = start_date
        
        while current_start + train_timedelta + test_timedelta <= end_date:
            train_end = current_start + train_timedelta
            test_end = train_end + test_timedelta
            
            # Skip if test period is too short
            if (test_end - train_end).days < min_test_days:
                current_start += step_timedelta
                continue
                
            print(f"\nWindow {total_windows + 1}: Train {current_start.date()} - {train_end.date()}, Test {train_end.date()} - {test_end.date()}")
            
            # Get data for this window
            train_data = self.price_data.loc[current_start:train_end]
            test_data = self.price_data.loc[train_end:test_end]
            
            if len(train_data) < 100 or len(test_data) < 30:
                current_start += step_timedelta
                continue
            
            # Test each parameter combination
            best_params = None
            best_train_return = -np.inf
            
            for params in param_combinations:
                try:
                    # Create strategy with current parameters
                    strategy = self.strategy_class(**params)
                    
                    # Run backtest on training data
                    train_config = BacktestConfig(
                        start_date=train_data.index.min().strftime('%Y-%m-%d'),
                        end_date=train_data.index.max().strftime('%Y-%m-%d'),
                        capital=self.base_config.capital,
                        slippage_bps=self.base_config.slippage_bps,
                        commission_per_trade=self.base_config.commission_per_trade,
                        rebalance_frequency=self.base_config.rebalance_frequency,
                        bid_ask_spread_bps=self.base_config.bid_ask_spread_bps,
                        min_trade_size=self.base_config.min_trade_size,
                        max_position_size=self.base_config.max_position_size
                    )
                    
                    train_backtester = Backtester([strategy], train_data, train_config)
                    train_results = train_backtester.run()
                    train_return = train_results['pnl'].iloc[-1] if len(train_results['pnl']) > 0 else 0
                    
                    if train_return > best_train_return:
                        best_train_return = train_return
                        best_params = params
                        
                except Exception as e:
                    print(f"Error testing params {params}: {e}")
                    continue
            
            if best_params is None:
                current_start += step_timedelta
                continue
            
            # Test best parameters on test data
            try:
                strategy = self.strategy_class(**best_params)
                test_config = BacktestConfig(
                    start_date=test_data.index.min().strftime('%Y-%m-%d'),
                    end_date=test_data.index.max().strftime('%Y-%m-%d'),
                    capital=self.base_config.capital,
                    slippage_bps=self.base_config.slippage_bps,
                    commission_per_trade=self.base_config.commission_per_trade,
                    rebalance_frequency=self.base_config.rebalance_frequency,
                    bid_ask_spread_bps=self.base_config.bid_ask_spread_bps,
                    min_trade_size=self.base_config.min_trade_size,
                    max_position_size=self.base_config.max_position_size
                )
                
                test_backtester = Backtester([strategy], test_data, test_config)
                test_results = test_backtester.run()
                
                # Calculate metrics
                pnl_series = test_results['pnl']
                if len(pnl_series) > 0:
                    total_return = pnl_series.iloc[-1]
                    returns = pnl_series.diff().dropna()
                    
                    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                    max_drawdown = (pnl_series - pnl_series.expanding().max()).min()
                    
                    # Calculate trade statistics
                    trading_costs = test_results.get('trading_costs', pd.Series([0]))
                    total_costs = trading_costs.iloc[-1] if len(trading_costs) > 0 else 0
                    net_return = total_return - total_costs
                    
                    # Estimate number of trades (simplified)
                    num_trades = len(returns[returns != 0])
                    win_rate = len(returns[returns > 0]) / num_trades if num_trades > 0 else 0
                    avg_trade = returns.mean() if num_trades > 0 else 0
                    
                    # Profit factor
                    gross_profit = returns[returns > 0].sum()
                    gross_loss = abs(returns[returns < 0].sum())
                    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
                    
                    result = ValidationResult(
                        method="walk_forward",
                        total_return=total_return,
                        sharpe_ratio=sharpe_ratio,
                        max_drawdown=max_drawdown,
                        win_rate=win_rate,
                        profit_factor=profit_factor,
                        avg_trade=avg_trade,
                        num_trades=num_trades,
                        total_costs=total_costs,
                        net_return=net_return,
                        params=best_params,
                        pnl_series=pnl_series,
                        metadata={
                            'train_start': current_start,
                            'train_end': train_end,
                            'test_start': train_end,
                            'test_end': test_end,
                            'window': total_windows + 1
                        }
                    )
                    results.append(result)
                    
                    print(f"  Best params: {best_params}")
                    print(f"  Test return: ${total_return:.2f}, Net: ${net_return:.2f}, Sharpe: {sharpe_ratio:.2f}")
                    
            except Exception as e:
                print(f"Error in test period: {e}")
            
            total_windows += 1
            current_start += step_timedelta
        
        print(f"\nWalk-Forward Analysis Complete: {len(results)} valid windows")
        return results
    
    def run_time_series_cv(
        self, 
        params: Dict,
        n_splits: int = 5,
        test_size: float = 0.2
    ) -> List[ValidationResult]:
        """
        Time series cross-validation
        
        Args:
            params: Strategy parameters
            n_splits: Number of CV splits
            test_size: Fraction of data for testing
        """
        print(f"Running Time Series Cross-Validation...")
        
        results = []
        strategy = self.strategy_class(**params)
        
        # Create time series splits
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=int(len(self.price_data) * test_size))
        
        for i, (train_idx, test_idx) in enumerate(tscv.split(self.price_data)):
            train_data = self.price_data.iloc[train_idx]
            test_data = self.price_data.iloc[test_idx]
            
            if len(train_data) < 100 or len(test_data) < 30:
                continue
                
            print(f"CV Split {i+1}: Train {len(train_data)} samples, Test {len(test_data)} samples")
            
            try:
                # Test on this split
                test_config = BacktestConfig(
                    start_date=test_data.index.min().strftime('%Y-%m-%d'),
                    end_date=test_data.index.max().strftime('%Y-%m-%d'),
                    capital=self.base_config.capital,
                    slippage_bps=self.base_config.slippage_bps,
                    commission_per_trade=self.base_config.commission_per_trade,
                    rebalance_frequency=self.base_config.rebalance_frequency,
                    bid_ask_spread_bps=self.base_config.bid_ask_spread_bps,
                    min_trade_size=self.base_config.min_trade_size,
                    max_position_size=self.base_config.max_position_size
                )
                
                test_backtester = Backtester([strategy], test_data, test_config)
                test_results = test_backtester.run()
                
                # Calculate metrics
                pnl_series = test_results['pnl']
                if len(pnl_series) > 0:
                    total_return = pnl_series.iloc[-1]
                    returns = pnl_series.diff().dropna()
                    
                    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                    max_drawdown = (pnl_series - pnl_series.expanding().max()).min()
                    
                    trading_costs = test_results.get('trading_costs', pd.Series([0]))
                    total_costs = trading_costs.iloc[-1] if len(trading_costs) > 0 else 0
                    net_return = total_return - total_costs
                    
                    num_trades = len(returns[returns != 0])
                    win_rate = len(returns[returns > 0]) / num_trades if num_trades > 0 else 0
                    avg_trade = returns.mean() if num_trades > 0 else 0
                    
                    gross_profit = returns[returns > 0].sum()
                    gross_loss = abs(returns[returns < 0].sum())
                    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
                    
                    result = ValidationResult(
                        method="time_series_cv",
                        total_return=total_return,
                        sharpe_ratio=sharpe_ratio,
                        max_drawdown=max_drawdown,
                        win_rate=win_rate,
                        profit_factor=profit_factor,
                        avg_trade=avg_trade,
                        num_trades=num_trades,
                        total_costs=total_costs,
                        net_return=net_return,
                        params=params,
                        pnl_series=pnl_series,
                        metadata={'split': i+1, 'n_splits': n_splits}
                    )
                    results.append(result)
                    
                    print(f"  Return: ${total_return:.2f}, Net: ${net_return:.2f}, Sharpe: {sharpe_ratio:.2f}")
                    
            except Exception as e:
                print(f"Error in CV split {i+1}: {e}")
        
        print(f"Time Series CV Complete: {len(results)} valid splits")
        return results
    
    def run_bootstrap_analysis(
        self, 
        params: Dict,
        n_bootstrap: int = 1000,
        sample_size: Optional[int] = None
    ) -> List[ValidationResult]:
        """
        Bootstrap analysis to test strategy robustness
        
        Args:
            params: Strategy parameters
            n_bootstrap: Number of bootstrap samples
            sample_size: Size of each bootstrap sample (None = same as original)
        """
        print(f"Running Bootstrap Analysis...")
        
        results = []
        strategy = self.strategy_class(**params)
        
        if sample_size is None:
            sample_size = len(self.price_data)
        
        for i in range(n_bootstrap):
            if i % 100 == 0:
                print(f"Bootstrap {i+1}/{n_bootstrap}")
            
            # Resample with replacement
            bootstrap_indices = np.random.choice(len(self.price_data), size=sample_size, replace=True)
            bootstrap_data = self.price_data.iloc[bootstrap_indices].copy()
            bootstrap_data = bootstrap_data.sort_index()  # Maintain time order
            
            try:
                # Run backtest on bootstrap sample
                bootstrap_config = BacktestConfig(
                    start_date=bootstrap_data.index.min().strftime('%Y-%m-%d'),
                    end_date=bootstrap_data.index.max().strftime('%Y-%m-%d'),
                    capital=self.base_config.capital,
                    slippage_bps=self.base_config.slippage_bps,
                    commission_per_trade=self.base_config.commission_per_trade,
                    rebalance_frequency=self.base_config.rebalance_frequency,
                    bid_ask_spread_bps=self.base_config.bid_ask_spread_bps,
                    min_trade_size=self.base_config.min_trade_size,
                    max_position_size=self.base_config.max_position_size
                )
                
                bootstrap_backtester = Backtester([strategy], bootstrap_data, bootstrap_config)
                bootstrap_results = bootstrap_backtester.run()
                
                # Calculate metrics
                pnl_series = bootstrap_results['pnl']
                if len(pnl_series) > 0:
                    total_return = pnl_series.iloc[-1]
                    returns = pnl_series.diff().dropna()
                    
                    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                    max_drawdown = (pnl_series - pnl_series.expanding().max()).min()
                    
                    trading_costs = bootstrap_results.get('trading_costs', pd.Series([0]))
                    total_costs = trading_costs.iloc[-1] if len(trading_costs) > 0 else 0
                    net_return = total_return - total_costs
                    
                    num_trades = len(returns[returns != 0])
                    win_rate = len(returns[returns > 0]) / num_trades if num_trades > 0 else 0
                    avg_trade = returns.mean() if num_trades > 0 else 0
                    
                    gross_profit = returns[returns > 0].sum()
                    gross_loss = abs(returns[returns < 0].sum())
                    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
                    
                    result = ValidationResult(
                        method="bootstrap",
                        total_return=total_return,
                        sharpe_ratio=sharpe_ratio,
                        max_drawdown=max_drawdown,
                        win_rate=win_rate,
                        profit_factor=profit_factor,
                        avg_trade=avg_trade,
                        num_trades=num_trades,
                        total_costs=total_costs,
                        net_return=net_return,
                        params=params,
                        pnl_series=pnl_series,
                        metadata={'bootstrap_sample': i+1}
                    )
                    results.append(result)
                    
            except Exception as e:
                if i < 10:  # Only print first few errors
                    print(f"Error in bootstrap {i+1}: {e}")
        
        print(f"Bootstrap Analysis Complete: {len(results)} valid samples")
        return results
    
    def run_parameter_stability_test(
        self,
        base_params: Dict,
        param_ranges: Dict[str, List],
        test_periods: int = 5
    ) -> List[ValidationResult]:
        """
        Test parameter stability across different time periods
        
        Args:
            base_params: Base parameter set
            param_ranges: Dictionary of parameter ranges to test
            test_periods: Number of time periods to test
        """
        print(f"Running Parameter Stability Test...")
        
        results = []
        total_periods = len(self.price_data)
        period_size = total_periods // test_periods
        
        for period in range(test_periods):
            start_idx = period * period_size
            end_idx = (period + 1) * period_size if period < test_periods - 1 else total_periods
            
            period_data = self.price_data.iloc[start_idx:end_idx]
            
            if len(period_data) < 100:
                continue
                
            print(f"Period {period+1}: {period_data.index.min().date()} - {period_data.index.max().date()}")
            
            # Test each parameter combination
            for param_name, param_values in param_ranges.items():
                for param_value in param_values:
                    test_params = base_params.copy()
                    test_params[param_name] = param_value
                    
                    try:
                        strategy = self.strategy_class(**test_params)
                        
                        period_config = BacktestConfig(
                            start_date=period_data.index.min().strftime('%Y-%m-%d'),
                            end_date=period_data.index.max().strftime('%Y-%m-%d'),
                            capital=self.base_config.capital,
                            slippage_bps=self.base_config.slippage_bps,
                            commission_per_trade=self.base_config.commission_per_trade,
                            rebalance_frequency=self.base_config.rebalance_frequency,
                            bid_ask_spread_bps=self.base_config.bid_ask_spread_bps,
                            min_trade_size=self.base_config.min_trade_size,
                            max_position_size=self.base_config.max_position_size
                        )
                        
                        period_backtester = Backtester([strategy], period_data, period_config)
                        period_results = period_backtester.run()
                        
                        # Calculate metrics
                        pnl_series = period_results['pnl']
                        if len(pnl_series) > 0:
                            total_return = pnl_series.iloc[-1]
                            returns = pnl_series.diff().dropna()
                            
                            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                            max_drawdown = (pnl_series - pnl_series.expanding().max()).min()
                            
                            trading_costs = period_results.get('trading_costs', pd.Series([0]))
                            total_costs = trading_costs.iloc[-1] if len(trading_costs) > 0 else 0
                            net_return = total_return - total_costs
                            
                            num_trades = len(returns[returns != 0])
                            win_rate = len(returns[returns > 0]) / num_trades if num_trades > 0 else 0
                            avg_trade = returns.mean() if num_trades > 0 else 0
                            
                            gross_profit = returns[returns > 0].sum()
                            gross_loss = abs(returns[returns < 0].sum())
                            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
                            
                            result = ValidationResult(
                                method="parameter_stability",
                                total_return=total_return,
                                sharpe_ratio=sharpe_ratio,
                                max_drawdown=max_drawdown,
                                win_rate=win_rate,
                                profit_factor=profit_factor,
                                avg_trade=avg_trade,
                                num_trades=num_trades,
                                total_costs=total_costs,
                                net_return=net_return,
                                params=test_params,
                                pnl_series=pnl_series,
                                metadata={
                                    'period': period+1,
                                    'tested_param': param_name,
                                    'param_value': param_value
                                }
                            )
                            results.append(result)
                            
                    except Exception as e:
                        print(f"Error testing {param_name}={param_value} in period {period+1}: {e}")
        
        print(f"Parameter Stability Test Complete: {len(results)} valid tests")
        return results
    
    def analyze_results(self, results: List[ValidationResult]) -> Dict:
        """Analyze validation results and generate summary statistics"""
        if not results:
            return {}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([{
            'method': r.method,
            'total_return': r.total_return,
            'sharpe_ratio': r.sharpe_ratio,
            'max_drawdown': r.max_drawdown,
            'win_rate': r.win_rate,
            'profit_factor': r.profit_factor,
            'avg_trade': r.avg_trade,
            'num_trades': r.num_trades,
            'total_costs': r.total_costs,
            'net_return': r.net_return
        } for r in results])
        
        # Calculate summary statistics
        summary = {
            'total_tests': len(results),
            'methods': df['method'].value_counts().to_dict(),
            'mean_net_return': df['net_return'].mean(),
            'std_net_return': df['net_return'].std(),
            'median_net_return': df['net_return'].median(),
            'positive_return_rate': (df['net_return'] > 0).mean(),
            'mean_sharpe': df['sharpe_ratio'].mean(),
            'std_sharpe': df['sharpe_ratio'].std(),
            'mean_max_drawdown': df['max_drawdown'].mean(),
            'mean_win_rate': df['win_rate'].mean(),
            'mean_profit_factor': df['profit_factor'].mean(),
            'total_costs_ratio': (df['total_costs'] / df['total_return'].abs()).mean(),
            'confidence_intervals': {
                'net_return_95': np.percentile(df['net_return'], [2.5, 97.5]).tolist(),
                'sharpe_95': np.percentile(df['sharpe_ratio'], [2.5, 97.5]).tolist(),
                'max_dd_95': np.percentile(df['max_drawdown'], [2.5, 97.5]).tolist()
            }
        }
        
        return summary
    
    def plot_results(self, results: List[ValidationResult], save_path: str = None):
        """Plot validation results"""
        if not results:
            print("No results to plot")
            return
        
        df = pd.DataFrame([{
            'method': r.method,
            'total_return': r.total_return,
            'sharpe_ratio': r.sharpe_ratio,
            'max_drawdown': r.max_drawdown,
            'win_rate': r.win_rate,
            'profit_factor': r.profit_factor,
            'net_return': r.net_return,
            'total_costs': r.total_costs
        } for r in results])
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Strategy Validation Results', fontsize=16)
        
        # Net Return Distribution
        axes[0, 0].hist(df['net_return'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(df['net_return'].mean(), color='red', linestyle='--', label=f'Mean: ${df["net_return"].mean():.2f}')
        axes[0, 0].set_title('Net Return Distribution')
        axes[0, 0].set_xlabel('Net Return ($)')
        axes[0, 0].legend()
        
        # Sharpe Ratio Distribution
        axes[0, 1].hist(df['sharpe_ratio'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(df['sharpe_ratio'].mean(), color='red', linestyle='--', label=f'Mean: {df["sharpe_ratio"].mean():.2f}')
        axes[0, 1].set_title('Sharpe Ratio Distribution')
        axes[0, 1].set_xlabel('Sharpe Ratio')
        axes[0, 1].legend()
        
        # Max Drawdown Distribution
        axes[0, 2].hist(df['max_drawdown'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 2].axvline(df['max_drawdown'].mean(), color='red', linestyle='--', label=f'Mean: {df["max_drawdown"].mean():.2f}')
        axes[0, 2].set_title('Max Drawdown Distribution')
        axes[0, 2].set_xlabel('Max Drawdown')
        axes[0, 2].legend()
        
        # Win Rate vs Net Return
        axes[1, 0].scatter(df['win_rate'], df['net_return'], alpha=0.6)
        axes[1, 0].set_title('Win Rate vs Net Return')
        axes[1, 0].set_xlabel('Win Rate')
        axes[1, 0].set_ylabel('Net Return ($)')
        
        # Profit Factor vs Net Return
        axes[1, 1].scatter(df['profit_factor'], df['net_return'], alpha=0.6)
        axes[1, 1].set_title('Profit Factor vs Net Return')
        axes[1, 1].set_xlabel('Profit Factor')
        axes[1, 1].set_ylabel('Net Return ($)')
        
        # Method Comparison
        method_means = df.groupby('method')['net_return'].mean()
        axes[1, 2].bar(method_means.index, method_means.values)
        axes[1, 2].set_title('Average Net Return by Method')
        axes[1, 2].set_xlabel('Validation Method')
        axes[1, 2].set_ylabel('Average Net Return ($)')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Validation results plot saved to {save_path}")
        
        plt.show() 