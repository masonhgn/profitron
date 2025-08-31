"""
Monte Carlo Simulation for Strategy Robustness Testing
Tests strategy performance under various market conditions and parameter uncertainties
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from .Backtester import Backtester, BacktestConfig
from strategies.TradingStrategy import TradingStrategy


@dataclass
class MonteCarloResult:
    """Container for Monte Carlo simulation results"""
    scenario: str
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
    market_conditions: Dict
    metadata: Dict


class MonteCarloSimulation:
    """Monte Carlo simulation framework for strategy testing"""
    
    def __init__(self, strategy_class, base_price_data: pd.DataFrame, base_config: BacktestConfig):
        self.strategy_class = strategy_class
        self.base_price_data = base_price_data
        self.base_config = base_config
        self.results = []
        
    def simulate_market_conditions(
        self,
        base_params: Dict,
        n_simulations: int = 1000,
        scenarios: Optional[List[str]] = None
    ) -> List[MonteCarloResult]:
        """
        Simulate different market conditions and test strategy robustness
        
        Args:
            base_params: Base strategy parameters
            n_simulations: Number of Monte Carlo simulations
            scenarios: List of market scenarios to test
        """
        if scenarios is None:
            scenarios = ['normal', 'volatile', 'trending', 'mean_reverting', 'crisis']
        
        print(f"Running Monte Carlo Simulations...")
        print(f"Scenarios: {scenarios}")
        print(f"Simulations per scenario: {n_simulations}")
        
        all_results = []
        
        for scenario in scenarios:
            print(f"\nSimulating {scenario} market conditions...")
            scenario_results = self._run_scenario_simulations(
                base_params, scenario, n_simulations
            )
            all_results.extend(scenario_results)
        
        return all_results
    
    def _run_scenario_simulations(
        self,
        base_params: Dict,
        scenario: str,
        n_simulations: int
    ) -> List[MonteCarloResult]:
        """Run simulations for a specific market scenario"""
        results = []
        
        for i in range(n_simulations):
            if i % 100 == 0:
                print(f"  Simulation {i+1}/{n_simulations}")
            
            try:
                # Generate synthetic market data based on scenario
                synthetic_data = self._generate_synthetic_data(scenario)
                
                # Add parameter uncertainty
                uncertain_params = self._add_parameter_uncertainty(base_params)
                
                # Run backtest
                result = self._run_single_simulation(
                    uncertain_params, synthetic_data, scenario
                )
                
                if result is not None:
                    results.append(result)
                    
            except Exception as e:
                if i < 10:  # Only print first few errors
                    print(f"Error in simulation {i+1}: {e}")
        
        print(f"  Completed {len(results)} valid simulations for {scenario}")
        return results
    
    def _generate_synthetic_data(self, scenario: str) -> pd.DataFrame:
        """Generate synthetic price data based on market scenario"""
        base_data = self.base_price_data.copy()
        
        if scenario == 'normal':
            # Normal market conditions - add small random noise
            noise = np.random.normal(0, 0.001, base_data.shape)
            synthetic_data = base_data * (1 + noise)
            
        elif scenario == 'volatile':
            # High volatility - increase price movements
            volatility_multiplier = np.random.uniform(1.5, 3.0)
            returns = base_data.pct_change().dropna()
            volatile_returns = returns * volatility_multiplier
            synthetic_data = base_data.iloc[0] * (1 + volatile_returns).cumprod()
            
        elif scenario == 'trending':
            # Trending market - add persistent trend
            trend_strength = np.random.uniform(0.0001, 0.001)
            trend_direction = np.random.choice([-1, 1])
            trend = np.arange(len(base_data)) * trend_strength * trend_direction
            synthetic_data = base_data * (1 + trend.reshape(-1, 1))
            
        elif scenario == 'mean_reverting':
            # Mean reverting - add oscillating component
            frequency = np.random.uniform(0.01, 0.05)
            oscillation = np.sin(2 * np.pi * frequency * np.arange(len(base_data)))
            synthetic_data = base_data * (1 + 0.002 * oscillation.reshape(-1, 1))
            
        elif scenario == 'crisis':
            # Crisis scenario - large negative shocks
            crisis_prob = 0.05
            crisis_shock = np.random.choice([0, -0.1, -0.2], size=len(base_data), p=[1-crisis_prob, crisis_prob/2, crisis_prob/2])
            synthetic_data = base_data * (1 + crisis_shock.reshape(-1, 1))
            
        else:
            # Default to normal
            synthetic_data = base_data.copy()
        
        # Ensure positive prices
        synthetic_data = synthetic_data.abs()
        
        # Add bid/ask spreads if available
        if 'bid' in base_data.columns and 'ask' in base_data.columns:
            spread_multiplier = np.random.uniform(0.8, 1.2)
            synthetic_data['bid'] = synthetic_data['bid'] * (1 - 0.0005 * spread_multiplier)
            synthetic_data['ask'] = synthetic_data['ask'] * (1 + 0.0005 * spread_multiplier)
        
        return synthetic_data
    
    def _add_parameter_uncertainty(self, base_params: Dict) -> Dict:
        """Add uncertainty to strategy parameters"""
        uncertain_params = base_params.copy()
        
        # Add uncertainty to key parameters
        if 'lookback_bars' in uncertain_params:
            uncertainty = np.random.normal(0, 2)
            uncertain_params['lookback_bars'] = max(5, int(uncertain_params['lookback_bars'] + uncertainty))
        
        if 'entry_z' in uncertain_params:
            uncertainty = np.random.normal(0, 0.2)
            uncertain_params['entry_z'] = max(0.5, uncertain_params['entry_z'] + uncertainty)
        
        if 'exit_z' in uncertain_params:
            uncertainty = np.random.normal(0, 0.1)
            uncertain_params['exit_z'] = max(0.1, uncertain_params['exit_z'] + uncertainty)
        
        return uncertain_params
    
    def _run_single_simulation(
        self,
        params: Dict,
        price_data: pd.DataFrame,
        scenario: str
    ) -> Optional[MonteCarloResult]:
        """Run a single Monte Carlo simulation"""
        try:
            strategy = self.strategy_class(**params)
            
            # Create backtest config
            config = BacktestConfig(
                start_date=price_data.index.min().strftime('%Y-%m-%d'),
                end_date=price_data.index.max().strftime('%Y-%m-%d'),
                capital=self.base_config.capital,
                slippage_bps=self.base_config.slippage_bps,
                commission_per_trade=self.base_config.commission_per_trade,
                rebalance_frequency=self.base_config.rebalance_frequency,
                bid_ask_spread_bps=self.base_config.bid_ask_spread_bps,
                min_trade_size=self.base_config.min_trade_size,
                max_position_size=self.base_config.max_position_size
            )
            
            # Run backtest
            backtester = Backtester([strategy], price_data, config)
            results = backtester.run()
            
            # Calculate metrics
            pnl_series = results['pnl']
            if len(pnl_series) == 0:
                return None
            
            total_return = pnl_series.iloc[-1]
            returns = pnl_series.diff().dropna()
            
            if len(returns) == 0:
                return None
            
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            max_drawdown = (pnl_series - pnl_series.expanding().max()).min()
            
            trading_costs = results.get('trading_costs', pd.Series([0]))
            total_costs = trading_costs.iloc[-1] if len(trading_costs) > 0 else 0
            net_return = total_return - total_costs
            
            num_trades = len(returns[returns != 0])
            win_rate = len(returns[returns > 0]) / num_trades if num_trades > 0 else 0
            avg_trade = returns.mean() if num_trades > 0 else 0
            
            gross_profit = returns[returns > 0].sum()
            gross_loss = abs(returns[returns < 0].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
            
            # Market conditions metadata
            market_conditions = {
                'scenario': scenario,
                'volatility': returns.std() * np.sqrt(252),
                'mean_return': returns.mean() * 252,
                'skewness': stats.skew(returns),
                'kurtosis': stats.kurtosis(returns)
            }
            
            result = MonteCarloResult(
                scenario=scenario,
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
                market_conditions=market_conditions,
                metadata={'simulation_id': len(self.results)}
            )
            
            return result
            
        except Exception as e:
            return None
    
    def run_stress_tests(
        self,
        base_params: Dict,
        stress_scenarios: Optional[Dict] = None
    ) -> List[MonteCarloResult]:
        """
        Run stress tests under extreme market conditions
        
        Args:
            base_params: Base strategy parameters
            stress_scenarios: Dictionary of stress test scenarios
        """
        if stress_scenarios is None:
            stress_scenarios = {
                'flash_crash': {'volatility_multiplier': 5.0, 'trend_strength': -0.01},
                'liquidity_crisis': {'spread_multiplier': 10.0, 'volatility_multiplier': 3.0},
                'correlation_breakdown': {'correlation_shift': -0.8},
                'regime_change': {'trend_strength': 0.005, 'volatility_multiplier': 2.0},
                'black_swan': {'shock_probability': 0.1, 'shock_magnitude': -0.3}
            }
        
        print(f"Running Stress Tests...")
        print(f"Scenarios: {list(stress_scenarios.keys())}")
        
        results = []
        
        for scenario_name, scenario_params in stress_scenarios.items():
            print(f"\nStress testing: {scenario_name}")
            
            for i in range(100):  # 100 stress tests per scenario
                try:
                    # Generate extreme market data
                    stress_data = self._generate_stress_data(scenario_name, scenario_params)
                    
                    # Run backtest
                    result = self._run_single_simulation(
                        base_params, stress_data, f"stress_{scenario_name}"
                    )
                    
                    if result is not None:
                        results.append(result)
                        
                except Exception as e:
                    if i < 5:  # Only print first few errors
                        print(f"Error in stress test {i+1}: {e}")
        
        print(f"Stress tests completed: {len(results)} valid results")
        return results
    
    def _generate_stress_data(self, scenario: str, params: Dict) -> pd.DataFrame:
        """Generate extreme market data for stress testing"""
        base_data = self.base_price_data.copy()
        
        if scenario == 'flash_crash':
            # Sudden large price drops
            volatility_multiplier = params.get('volatility_multiplier', 5.0)
            trend_strength = params.get('trend_strength', -0.01)
            
            returns = base_data.pct_change().dropna()
            stress_returns = returns * volatility_multiplier
            trend = np.arange(len(base_data)) * trend_strength
            stress_data = base_data.iloc[0] * (1 + stress_returns + trend.reshape(-1, 1)).cumprod()
            
        elif scenario == 'liquidity_crisis':
            # Wide spreads and high volatility
            spread_multiplier = params.get('spread_multiplier', 10.0)
            volatility_multiplier = params.get('volatility_multiplier', 3.0)
            
            returns = base_data.pct_change().dropna()
            stress_returns = returns * volatility_multiplier
            stress_data = base_data.iloc[0] * (1 + stress_returns).cumprod()
            
            # Widen spreads
            if 'bid' in stress_data.columns and 'ask' in stress_data.columns:
                stress_data['bid'] = stress_data['bid'] * (1 - 0.005 * spread_multiplier)
                stress_data['ask'] = stress_data['ask'] * (1 + 0.005 * spread_multiplier)
                
        elif scenario == 'correlation_breakdown':
            # Break correlation between assets
            correlation_shift = params.get('correlation_shift', -0.8)
            
            # Add uncorrelated noise to break correlation
            noise = np.random.normal(0, 0.01, base_data.shape)
            stress_data = base_data * (1 + noise * correlation_shift)
            
        elif scenario == 'regime_change':
            # Sudden change in market regime
            trend_strength = params.get('trend_strength', 0.005)
            volatility_multiplier = params.get('volatility_multiplier', 2.0)
            
            # Split data into two regimes
            mid_point = len(base_data) // 2
            returns = base_data.pct_change().dropna()
            
            # First regime: normal
            regime1_returns = returns.iloc[:mid_point]
            # Second regime: high volatility and trend
            regime2_returns = returns.iloc[mid_point:] * volatility_multiplier
            trend = np.arange(len(base_data) - mid_point) * trend_strength
            
            stress_data = base_data.iloc[0] * (1 + pd.concat([regime1_returns, regime2_returns]) + trend.reshape(-1, 1)).cumprod()
            
        elif scenario == 'black_swan':
            # Rare extreme events
            shock_prob = params.get('shock_probability', 0.1)
            shock_magnitude = params.get('shock_magnitude', -0.3)
            
            shocks = np.random.choice([0, shock_magnitude], size=len(base_data), p=[1-shock_prob, shock_prob])
            stress_data = base_data * (1 + shocks.reshape(-1, 1))
            
        else:
            stress_data = base_data.copy()
        
        # Ensure positive prices
        stress_data = stress_data.abs()
        return stress_data
    
    def analyze_monte_carlo_results(self, results: List[MonteCarloResult]) -> Dict:
        """Analyze Monte Carlo simulation results"""
        if not results:
            return {}
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'scenario': r.scenario,
            'total_return': r.total_return,
            'sharpe_ratio': r.sharpe_ratio,
            'max_drawdown': r.max_drawdown,
            'win_rate': r.win_rate,
            'profit_factor': r.profit_factor,
            'net_return': r.net_return,
            'volatility': r.market_conditions.get('volatility', 0),
            'mean_return': r.market_conditions.get('mean_return', 0),
            'skewness': r.market_conditions.get('skewness', 0),
            'kurtosis': r.market_conditions.get('kurtosis', 0)
        } for r in results])
        
        # Scenario analysis
        scenario_analysis = {}
        for scenario in df['scenario'].unique():
            scenario_data = df[df['scenario'] == scenario]
            scenario_analysis[scenario] = {
                'count': len(scenario_data),
                'mean_net_return': scenario_data['net_return'].mean(),
                'std_net_return': scenario_data['net_return'].std(),
                'positive_return_rate': (scenario_data['net_return'] > 0).mean(),
                'mean_sharpe': scenario_data['sharpe_ratio'].mean(),
                'mean_max_dd': scenario_data['max_drawdown'].mean(),
                'worst_case': scenario_data['net_return'].min(),
                'best_case': scenario_data['net_return'].max(),
                'var_95': np.percentile(scenario_data['net_return'], 5),
                'cvar_95': scenario_data[scenario_data['net_return'] <= np.percentile(scenario_data['net_return'], 5)]['net_return'].mean()
            }
        
        # Overall statistics
        summary = {
            'total_simulations': len(results),
            'scenarios': df['scenario'].value_counts().to_dict(),
            'overall_mean_return': df['net_return'].mean(),
            'overall_std_return': df['net_return'].std(),
            'overall_positive_rate': (df['net_return'] > 0).mean(),
            'overall_mean_sharpe': df['sharpe_ratio'].mean(),
            'overall_mean_max_dd': df['max_drawdown'].mean(),
            'overall_var_95': np.percentile(df['net_return'], 5),
            'overall_cvar_95': df[df['net_return'] <= np.percentile(df['net_return'], 5)]['net_return'].mean(),
            'scenario_analysis': scenario_analysis,
            'market_condition_correlation': {
                'volatility_vs_return': df['volatility'].corr(df['net_return']),
                'mean_return_vs_strategy_return': df['mean_return'].corr(df['net_return']),
                'skewness_vs_return': df['skewness'].corr(df['net_return']),
                'kurtosis_vs_return': df['kurtosis'].corr(df['net_return'])
            }
        }
        
        return summary
    
    def plot_monte_carlo_results(self, results: List[MonteCarloResult], save_path: str = None):
        """Plot Monte Carlo simulation results"""
        if not results:
            print("No results to plot")
            return
        
        df = pd.DataFrame([{
            'scenario': r.scenario,
            'total_return': r.total_return,
            'sharpe_ratio': r.sharpe_ratio,
            'max_drawdown': r.max_drawdown,
            'win_rate': r.win_rate,
            'profit_factor': r.profit_factor,
            'net_return': r.net_return,
            'volatility': r.market_conditions.get('volatility', 0),
            'mean_return': r.market_conditions.get('mean_return', 0)
        } for r in results])
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Monte Carlo Simulation Results', fontsize=16)
        
        # Net Return Distribution by Scenario
        for scenario in df['scenario'].unique():
            scenario_data = df[df['scenario'] == scenario]
            axes[0, 0].hist(scenario_data['net_return'], bins=30, alpha=0.6, label=scenario)
        axes[0, 0].set_title('Net Return Distribution by Scenario')
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
        
        # Market Volatility vs Strategy Return
        axes[1, 0].scatter(df['volatility'], df['net_return'], alpha=0.6)
        axes[1, 0].set_title('Market Volatility vs Strategy Return')
        axes[1, 0].set_xlabel('Market Volatility')
        axes[1, 0].set_ylabel('Strategy Net Return ($)')
        
        # Scenario Comparison
        scenario_means = df.groupby('scenario')['net_return'].mean()
        axes[1, 1].bar(scenario_means.index, scenario_means.values)
        axes[1, 1].set_title('Average Net Return by Scenario')
        axes[1, 1].set_xlabel('Market Scenario')
        axes[1, 1].set_ylabel('Average Net Return ($)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Win Rate vs Net Return
        axes[1, 2].scatter(df['win_rate'], df['net_return'], alpha=0.6)
        axes[1, 2].set_title('Win Rate vs Net Return')
        axes[1, 2].set_xlabel('Win Rate')
        axes[1, 2].set_ylabel('Net Return ($)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Monte Carlo results plot saved to {save_path}")
        
        plt.show() 