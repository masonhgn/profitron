
#src/utils.py

import yaml
import os
from typing import Dict, Any
from dotenv import load_dotenv


def load_yaml_config(path: str) -> Dict[str, Any]:
    """
    loads a yaml config to be passed into a dataclass constructor
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_environment():
    """Load environment variables from .env file"""
    load_dotenv()


def get_env_var(key: str, default: str = None) -> str:
    """
    Get environment variable safely
    
    Args:
        key: Environment variable name
        default: Default value if not found
        
    Returns:
        Environment variable value or default
    """
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(f"Environment variable {key} not found and no default provided")
    return value


def resolve_config_values(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve environment variable references in config
    
    Args:
        config: Configuration dictionary that may contain env var references
        
    Returns:
        Config with environment variables resolved
    """
    def _resolve_value(value):
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            # Extract environment variable name
            env_var = value[2:-1]  # Remove ${ and }
            return get_env_var(env_var)
        elif isinstance(value, dict):
            return {k: _resolve_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_resolve_value(v) for v in value]
        else:
            return value
    
    return _resolve_value(config)