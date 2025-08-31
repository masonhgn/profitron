# main.py

import argparse
import sys
import traceback

from .core.Engine import Engine

def main():
    parser = argparse.ArgumentParser(description="trading engine")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/engine/Engine.yaml",
        help="path to configuration file (engine config, strategy config, or main config)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["paper", "live"],
        help="override trading mode (paper or live)"
    )
    
    args = parser.parse_args()
    
    try:
        engine = Engine(config_path=args.config)
        
        # Override mode if specified
        if args.mode:
            engine.config["mode"] = args.mode
        
        engine.run()
        
    except KeyboardInterrupt:
        print("\n[main] interrupted by user. shutting down gracefully.")
        sys.exit(0)
        
    except Exception as e:
        print(f"[main] error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
