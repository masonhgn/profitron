# strategy specification: cointegration-based pairs trading

## document information

- strategy name: cointegration-based mean reversion pairs trading
- version: 1.0
- created: 2024-12-19
- last updated: 2024-12-19
- author: profitron v2 team
- status: production ready
- classification: proprietary

---

## executive summary

this strategy implements a cointegration-based mean reversion approach for pairs trading, specifically targeting ethereum etf pairs (etha/ethv). the strategy identifies statistically cointegrated asset pairs and generates trading signals based on the z-score of the cointegration spread, with comprehensive risk management and position sizing.

## latest backtest results

![latest backtest results](backtest_comprehensive_ETHA-ETHV_20250713_013231.png)

### key metrics (latest backtest)
- total return: $6,597.96
- annualized return: 92.69%
- sharpe ratio: 1.59
- calmar ratio: 0.22
- max drawdown: -42.93%
- win rate: 8.2%
- profit factor: 3.22
- trading costs: $576.00
- net return: $6,021.96
- number of trades: 2,554

---

## mathematical formulation

1. cointegration model
2. hedge ratio estimation
3. spread calculation
4. z-score computation
5. signal generation

---

## implementation details

- data requirements: etha, ethv, 1-hour bars, open/high/low/close/volume
- parameter configuration: lookback_bars=15, entry_z=1.8, exit_z=0.5, max_position_size=0.5, min_correlation=0.7, cointegration_pvalue_threshold=0.05, max_spread_volatility=0.1
- position sizing: multi-factor approach based on z-score, volatility, and risk
- risk management: position limits, volatility controls, correlation requirements

---

## backtesting methodology

- execution modeling: slippage, commission, bid-ask spread, minimum trade size
- performance metrics: total return, annualized return, sharpe ratio, calmar ratio, max drawdown, win rate, profit factor, trading costs, number of trades
- validation framework: parameter sensitivity, time period analysis, walk-forward, monte carlo

---

## operational procedures

- pre-trading validation: data quality, statistical validation, parameter validation
- daily operations: market open checks, during trading monitoring, market close reconciliation
- risk monitoring: real-time monitoring of position size, spread volatility, correlation stability, z-score distribution 