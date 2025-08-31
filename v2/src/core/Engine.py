# src/core/Engine.py

import yaml
import pandas as pd
import importlib
from ..backtest.Backtester import Backtester, BacktestConfig
from ..signals.SignalMonitor import SignalMonitor
from ..portfolio.MetaPortfolio import MetaPortfolio
from ..brokers.BrokerContext import BrokerContext
from strategies.TradingStrategy import TradingStrategy
from ..data_collection.DataManager import DataManager
from ..utils.utilities import load_environment, load_yaml_config, resolve_config_values
import time
from ..execution.Executor import Executor
from ..utils.telegram_alerts import TelegramAlerts
from pathlib import Path

class Engine:
    def __init__(self, config_path: str = "configs/engine/Engine.yaml"):
        # load and resolve configuration
        load_environment()
        
        # Check if this is a main config (points to strategy and engine configs)
        if self._is_main_config(config_path):
            self.config = self._load_main_config(config_path)
        elif self._is_strategy_config(config_path):
            # Strategy config only - load with default engine config
            self.config = self._load_strategy_with_default_engine(config_path)
        else:
            # Single config file (legacy or engine config)
            raw_config = load_yaml_config(config_path)
            self.config = resolve_config_values(raw_config)
        
        # initialize components
        self.mode = self.config.get("mode", "paper")  # live or paper or backtest
        self.meta = MetaPortfolio()
        self.data_manager = DataManager(config=self.config)
        self._init_brokers()
        self.strategy = self._init_strategy()
        self.monitor = self._init_monitor()
        
        # Initialize Telegram alerts
        self.alerts = TelegramAlerts()
        if self.alerts.enabled:
            self.alerts.test_connection()
        
        if self.mode == "backtest":
            self.backtester = self._init_backtester()
        else:
            self.executor = Executor(self.monitor, self.config)
    
    def _is_main_config(self, config_path: str) -> bool:
        """Check if this is a main config file that points to other configs"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return "strategy_config" in config and "engine_config" in config
        except:
            return False
    
    def _is_strategy_config(self, config_path: str) -> bool:
        """Check if this is a strategy config file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return "strategy" in config
        except:
            return False
    
    def _load_main_config(self, main_config_path: str) -> dict:
        """Load and merge strategy and engine configs from main config"""
        with open(main_config_path, 'r') as f:
            main_config = yaml.safe_load(f)
        
        # Load strategy config
        config_dir = Path(main_config_path).parent
        strategy_config_path = config_dir / main_config["strategy_config"]
        engine_config_path = config_dir / main_config["engine_config"]
        
        strategy_config = load_yaml_config(strategy_config_path)
        engine_config = load_yaml_config(engine_config_path)
        
        # Merge configs (engine config takes precedence)
        merged_config = {**strategy_config, **engine_config}
        
        # Override mode if specified in main config
        if "mode" in main_config:
            merged_config["mode"] = main_config["mode"]
        
        # Resolve environment variables
        return resolve_config_values(merged_config)
    
    def _load_strategy_with_default_engine(self, strategy_config_path: str) -> dict:
        """Load strategy config and merge with default engine config"""
        # Load strategy config
        strategy_config = load_yaml_config(strategy_config_path)
        
        # Load default engine config
        engine_config_path = Path("configs/engine/Engine.yaml")
        engine_config = load_yaml_config(engine_config_path)
        
        # Merge configs (engine config takes precedence)
        merged_config = {**strategy_config, **engine_config}
        
        # Resolve environment variables
        return resolve_config_values(merged_config)
    
    def _init_brokers(self):
        """initialize broker connections"""
        # Get broker configs for current mode (paper or live)
        mode = self.config.get("mode", "paper")
        brokers_cfg = self.config.get("brokers", {}).get(mode, {})
        
        # Get the brokerage specified in strategy config, default to alpaca
        strategy_cfg = self.config.get("strategy", {})
        specified_brokerage = strategy_cfg.get("brokerage", "alpaca")
        
        # Only initialize the specified brokerage
        if specified_brokerage in brokers_cfg:
            broker_cfg = brokers_cfg[specified_brokerage]
            broker_type = broker_cfg.get("type", "alpaca")
            
            if broker_type == "ibkr":
                from src.brokers.IBKRBrokerContext import IBKRBrokerContext
                ctx = IBKRBrokerContext(specified_brokerage, broker_cfg)
            else:
                ctx = BrokerContext(specified_brokerage, broker_cfg)
            
            self.meta.register_broker(specified_brokerage, ctx)
            print(f"[Engine] Initialized {specified_brokerage} broker for strategy")
        else:
            print(f"[Engine] Warning: Specified brokerage '{specified_brokerage}' not found in config")
    
    def _init_strategy(self) -> TradingStrategy:
        """Dynamically load strategy from config using explicit module path"""
        strat_cfg = self.config["strategy"]
        strategy_name = strat_cfg["name"]
        strategy_params = strat_cfg["params"]
        
        # Get module path from config
        module_path = strat_cfg.get("module")
        if not module_path:
            raise ValueError(f"Strategy config must include 'module' path for {strategy_name}")
        
        try:
            # Dynamically import the module
            module = importlib.import_module(module_path)
            
            # Get the strategy class from the module
            strategy_class = getattr(module, strategy_name)
            
            # Verify it's a subclass of TradingStrategy
            if not issubclass(strategy_class, TradingStrategy):
                raise ValueError(f"{strategy_name} is not a valid TradingStrategy")
            
            # Convert asset strings to proper dictionary format if needed
            params = strategy_params.copy()
            if "asset_1" in params and isinstance(params["asset_1"], str):
                params["asset_1"] = {"symbol": params["asset_1"], "type": "equity"}
            if "asset_2" in params and isinstance(params["asset_2"], str):
                params["asset_2"] = {"symbol": params["asset_2"], "type": "equity"}
            
            return strategy_class(**params)
            
        except ImportError as e:
            raise ValueError(f"Could not import strategy module '{module_path}': {e}")
        except AttributeError as e:
            raise ValueError(f"Strategy class '{strategy_name}' not found in module '{module_path}': {e}")
        except Exception as e:
            raise ValueError(f"Error initializing strategy '{strategy_name}': {e}")
    
    def _init_monitor(self) -> SignalMonitor:
        """initialize signal monitor"""
        return SignalMonitor(
            strategy=self.strategy,
            data_manager=self.data_manager,
            config={
                "frequency": self.strategy.get_frequency(),
                "fields": self.strategy.get_fields(),
                "lookback_window": self.strategy.get_lookback_window(),
                "poll_interval": self.strategy.get_poll_interval(),
            }
        )
    
    def _init_backtester(self) -> Backtester:
        """initialize backtester"""
        bt_cfg = self.config["backtest"]
        bt_config = BacktestConfig(**bt_cfg["params"])
        
        assets = self.strategy.get_assets()  # [{'symbol': 'ETH', 'type': 'crypto'}, ...]
        frequency = self.strategy.get_frequency()
        fields = self.strategy.get_fields()
        
        data = self.data_manager.get_price_data(
            assets=assets,
            start_date=bt_config.start_date,
            end_date=bt_config.end_date,
            frequency=frequency,
            fields=fields
        )
        
        return Backtester(
            strategies=[self.strategy],
            price_data=data,
            config=bt_config
        )
    
    def run(self):
        """run the trading engine"""
        print(f"[Engine] running in {self.mode.upper()} mode..")
        if self.mode == "backtest":
            self._run_backtest()
        else:
            self._run_live_or_paper()
    
    def _run_backtest(self):
        """run backtest mode"""
        results = self.backtester.run()
        print("[Engine] backtest complete!!! summary:")
        
        # print key metrics
        pnl = results.get("pnl", pd.Series())
        if isinstance(pnl, pd.Series) and not pnl.empty:
            print(f"final pnl: ${pnl.iloc[-1]:.2f}")
        
        stats = results.get("stats", {})
        if isinstance(stats, dict) and stats:
            print(f"sharpe ratio: {stats.get('sharpe_ratio', 0):.2f}")
            print(f"max drawdown: {stats.get('max_drawdown', 0):.2%}")
    
    def _run_live_or_paper(self):
        """run live or paper trading mode"""
        poll_interval = self.strategy.get_poll_interval()
        
        print(f"[Engine] Starting {self.mode.upper()} trading mode")
        print(f"[Engine] Polling interval: {poll_interval} seconds ({poll_interval/3600:.1f} hours)")
        print(f"[Engine] Strategy: {self.strategy.__class__.__name__}")
        print(f"[Engine] Assets: {[asset['symbol'] for asset in self.strategy.get_assets()]}")
        print(f"[Engine] Entering main loop...")
        
        # Send startup alert
        if self.alerts.enabled:
            self.alerts.send_system_status(
                f"Trading system started in {self.mode.upper()} mode",
                f"Strategy: {self.strategy.__class__.__name__}\n"
                f"Assets: {[asset['symbol'] for asset in self.strategy.get_assets()]}\n"
                f"Polling: {poll_interval/3600:.1f} hours"
            )
        
        while True:
            try:
                self.executor.execute()
                if self.config.get("print_summary", True):
                    # Optionally print summary if you have a portfolio tracker
                    pass
                print(f"[Engine] Sleeping for {poll_interval} seconds...")
                time.sleep(poll_interval)
            except KeyboardInterrupt:
                print(f"[Engine] Received interrupt signal, shutting down...")
                if self.alerts.enabled:
                    self.alerts.send_system_status("Trading system shutdown", "Received interrupt signal")
                break
            except Exception as e:
                error_msg = f"Error in main loop: {e}"
                print(f"[Engine] {error_msg}")
                if self.alerts.enabled:
                    self.alerts.send_error_alert(error_msg, "Main trading loop")
                print(f"[Engine] Continuing after error...")
                time.sleep(poll_interval)
    
    def _get_price_lookup(self, signals):
        """get current prices for signals"""
        prices = {}
        for sig in signals:
            broker = self.meta.brokers.get(self.meta._select_broker(sig).name)
            if broker and broker.api:
                try:
                    ticker = broker.api.fetch_ticker(sig.symbol.replace("-", "/"))
                    prices[sig.symbol] = ticker["last"]
                except:
                    prices[sig.symbol] = 0.0
        return prices
