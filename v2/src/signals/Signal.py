# src/signals/Signal.py

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class Signal:
    symbol: str
    action: str  # "buy", "sell", "hold"
    asset_type: str = "crypto"
    target_pct: Optional[float] = None  # used for proportional sizing
    quantity: Optional[float] = None    # for absolute units
    order_type: str = "market"
    meta: Dict[str, Any] = field(default_factory=dict)
