from typing import Dict, Optional
from typing import Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

'''
this gets the stats of a specific set of returns so we can plug this into backtesting
'''

class PerformanceStats:

    @staticmethod
    def compute(initial_capital: float, pnl_series: pd.Series) -> Dict[str, Any]:
        returns = pnl_series.diff().dropna() / initial_capital

        # Use actual data range for annualization
        n_years = 1.0
        actual_start_str = ''
        actual_end_str = ''
        if len(pnl_series) > 1:
            dt_index = pnl_series.index
            # Only attempt conversion if index is not a DatetimeIndex and not a dtype object
            if not isinstance(dt_index, pd.DatetimeIndex) and not isinstance(dt_index, type(pd.Series(dtype='float').dtype)):
                try:
                    dt_index = pd.to_datetime(dt_index)
                except Exception:
                    dt_index = None
            if isinstance(dt_index, pd.DatetimeIndex):
                actual_start = dt_index[0]
                actual_end = dt_index[-1]
                if isinstance(actual_start, pd.Timestamp) and isinstance(actual_end, pd.Timestamp):
                    n_days = (actual_end - actual_start).days
                    n_years = n_days / 365.25 if n_days > 0 else 1.0
                    actual_start_str = str(actual_start)
                    actual_end_str = str(actual_end)

        sharpe = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0
        # robust drawdown calculation
        running_max = pnl_series.cummax()
        valid = running_max > 1e-6
        drawdown = pd.Series(0.0, index=pnl_series.index, dtype=float)
        drawdown[valid] = (pnl_series[valid] - running_max[valid]) / running_max[valid]
        drawdown = drawdown.replace([np.inf, -np.inf], np.nan).dropna()
        max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
        total_return = float(pnl_series.iloc[-1] - pnl_series.iloc[0]) if not pnl_series.empty else 0.0

        # updated metrics using n_years
        annualized_return = float((total_return / initial_capital) / n_years) if n_years > 0 else 0.0
        volatility = float(returns.std() * np.sqrt(252 / n_years)) if returns.std() > 0 and n_years > 0 else 0.0
        win_rate = float(len(returns[returns > 0]) / len(returns)) if len(returns) > 0 else 0.0
        profit_factor = float(abs(returns[returns > 0].sum() / returns[returns < 0].sum())) if len(returns[returns < 0]) > 0 and returns[returns < 0].sum() != 0 else 0.0
        calmar_ratio = float(annualized_return / abs(max_dd)) if max_dd != 0 else 0.0

        return {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "volatility": float(volatility),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "calmar_ratio": float(calmar_ratio),
            "return_std": float(returns.std()),
            "avg_daily_return": float(returns.mean()),
            "actual_start": actual_start_str,
            "actual_end": actual_end_str,
            "n_years": float(n_years)
        }
    
    @staticmethod
    def plot(pnl_series: pd.Series, title: str = "Strategy Performance", save_path: Optional[str] = None, 
             metadata: Optional[Dict] = None, trading_costs: Optional[pd.Series] = None, save_image: bool = False):
        """
        Plot comprehensive performance analysis
        
        Args:
            pnl_series: PnL series
            title: Plot title
            save_path: Path to save image (optional)
            metadata: Strategy metadata
            trading_costs: Trading costs series
            save_image: Whether to save the image (default: False)
        """
        if pnl_series.empty:
            print("[PerformanceStats] cannot plot: pnl series is empty.")
            return

        # generate dynamic title and filename
        if metadata and 'tickers' in metadata:
            tickers = metadata['tickers']
            if isinstance(tickers, list) and len(tickers) >= 2:
                ticker_pair = f"{tickers[0]}-{tickers[1]}"
                dynamic_title = f"cointegration strategy: {ticker_pair}"
                
                # create filename with tickers and timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if save_path is None:
                    save_path = f"backtest_comprehensive_{ticker_pair}_{timestamp}.png"
            else:
                dynamic_title = title
                if save_path is None:
                    save_path = f"backtest_comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        else:
            dynamic_title = title
            if save_path is None:
                save_path = f"backtest_comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        # add date range to title if available
        if metadata and 'start_date' in metadata and 'end_date' in metadata:
            dynamic_title += f" ({metadata['start_date']} to {metadata['end_date']})"

        # calculate performance metrics
        initial_capital = metadata.get('initial_capital', 100000) if metadata else 100000
        stats = PerformanceStats.compute(initial_capital, pnl_series)
        # update dynamic_title to show true data range
        if 'actual_start' in stats and 'actual_end' in stats:
            dynamic_title += f" (actual: {stats['actual_start'][:10]} to {stats['actual_end'][:10]})"
        
        # robust drawdown for plotting
        running_max = pnl_series.cummax()
        valid = running_max > 1e-6
        idx = pd.Index(pnl_series.index)
        drawdown = pd.Series(0.0, index=idx, dtype=float)
        drawdown[valid] = (pnl_series[valid] - running_max[valid]) / running_max[valid]
        drawdown = drawdown.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        # calculate net pnl (gross - costs)
        net_pnl = pnl_series.copy()
        if trading_costs is not None and not trading_costs.empty:
            net_pnl = pnl_series - trading_costs

        # create comprehensive plot
        fig = plt.figure(figsize=(16, 12))
        
        # create grid layout
        gs = fig.add_gridspec(4, 3, height_ratios=[2, 1, 1, 1], width_ratios=[2, 1, 1])
        
        # main performance plot
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(pnl_series.index, pnl_series.values.astype(float), label="gross pnl", linewidth=2, color='blue')
        if trading_costs is not None and not trading_costs.empty:
            ax1.plot(net_pnl.index, net_pnl.values.astype(float), label="net pnl", linewidth=2, color='green', linestyle='--')
        ax1.set_ylabel("pnl ($)")
        ax1.set_title(dynamic_title, fontsize=16, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # drawdown plot
        ax2 = fig.add_subplot(gs[1, :])
        ax2.fill_between(drawdown.index, drawdown.values.astype(float) * 100, color='red', alpha=0.3, label="drawdown")
        ax2.set_ylabel("drawdown (%)")
        ax2.set_xlabel("date")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # performance metrics table
        ax3 = fig.add_subplot(gs[2:, 0])
        ax3.axis('off')
        
        # create metrics text
        metrics_text = "performance metrics:\n\n"
        metrics_text += f"total return: ${stats['total_return']:,.2f}\n"
        metrics_text += f"annualized return: {stats['annualized_return']:.2%}\n"
        metrics_text += f"sharpe ratio: {stats['sharpe_ratio']:.2f}\n"
        metrics_text += f"calmar ratio: {stats['calmar_ratio']:.2f}\n"
        metrics_text += f"max drawdown: {stats['max_drawdown']:.2%}\n"
        metrics_text += f"volatility: {stats['volatility']:.2%}\n"
        metrics_text += f"win rate: {stats['win_rate']:.1%}\n"
        metrics_text += f"profit factor: {stats['profit_factor']:.2f}\n"
        
        if trading_costs is not None and not trading_costs.empty:
            total_costs = trading_costs.iloc[-1]
            net_return = stats['total_return'] - total_costs
            metrics_text += f"\ntrading costs: ${total_costs:,.2f}\n"
            metrics_text += f"net return: ${net_return:,.2f}\n"
            metrics_text += f"cost ratio: {total_costs/stats['total_return']:.2%}\n"
        
        ax3.text(0.05, 0.95, metrics_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        
        # strategy parameters
        ax4 = fig.add_subplot(gs[2:, 1])
        ax4.axis('off')
        
        params_text = "strategy parameters:\n\n"
        if metadata:
            if 'lookback_bars' in metadata:
                params_text += f"lookback bars: {metadata['lookback_bars']}\n"
            if 'entry_z' in metadata:
                params_text += f"entry z-score: {metadata['entry_z']}\n"
            if 'exit_z' in metadata:
                params_text += f"exit z-score: {metadata['exit_z']}\n"
            if 'max_position_size' in metadata:
                params_text += f"max position: {metadata['max_position_size']:.1%}\n"
        
        params_text += f"\nexecution costs:\n"
        slippage_bps = metadata.get('slippage_bps', 5) if metadata else 5
        commission = metadata.get('commission_per_trade', 1.00) if metadata else 1.00
        bid_ask_spread = metadata.get('bid_ask_spread_bps', 5) if metadata else 5
        params_text += f"slippage: {slippage_bps} bps\n"
        params_text += f"commission: ${commission:.2f}\n"
        params_text += f"bid-ask spread: {bid_ask_spread} bps\n"
        
        ax4.text(0.05, 0.95, params_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
        
        # trading statistics
        ax5 = fig.add_subplot(gs[2:, 2])
        ax5.axis('off')
        
        # calculate trading statistics
        returns = pnl_series.diff().dropna()
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        stats_text = "trading statistics:\n\n"
        stats_text += f"total trades: {len(returns)}\n"
        stats_text += f"winning trades: {len(positive_returns)}\n"
        stats_text += f"losing trades: {len(negative_returns)}\n"
        if len(positive_returns) > 0:
            stats_text += f"avg win: ${positive_returns.mean():.2f}\n"
        if len(negative_returns) > 0:
            stats_text += f"avg loss: ${negative_returns.mean():.2f}\n"
        stats_text += f"largest win: ${returns.max():.2f}\n"
        stats_text += f"largest loss: ${returns.min():.2f}\n"
        
        # calculate holding periods if possible
        if len(returns) > 0:
            avg_holding_period = len(pnl_series) / len(returns[returns != 0]) if len(returns[returns != 0]) > 0 else 0
            stats_text += f"avg holding period: {avg_holding_period:.1f} bars\n"
        
        ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen"))
        
        plt.tight_layout()
        if save_image:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[PerformanceStats] comprehensive plot saved to {save_path}")
        else:
            print(f"[PerformanceStats] comprehensive plot not saved.")
        plt.show()
        
        # print summary to console
        print(f"\n=== comprehensive backtest summary ===")
        print(f"strategy: {dynamic_title}")
        start_date = metadata.get('start_date', 'N/A') if metadata else 'N/A'
        end_date = metadata.get('end_date', 'N/A') if metadata else 'N/A'
        print(f"period: {start_date} to {end_date}")
        print(f"initial capital: ${initial_capital:,.2f}")
        print(f"total return: ${stats['total_return']:,.2f}")
        print(f"annualized return: {stats['annualized_return']:.2%}")
        print(f"sharpe ratio: {stats['sharpe_ratio']:.2f}")
        print(f"max drawdown: {stats['max_drawdown']:.2%}")
        print(f"win rate: {stats['win_rate']:.1%}")
        if trading_costs is not None and not trading_costs.empty:
            print(f"total trading costs: ${trading_costs.iloc[-1]:,.2f}")
            print(f"net return: ${stats['total_return'] - trading_costs.iloc[-1]:,.2f}")
        print(f"total trades: {len(returns)}")
        print("=" * 50)