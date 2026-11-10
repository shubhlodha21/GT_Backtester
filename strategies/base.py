"""Strategy contract.

A strategy declares its tunable parameters in `params` (the UI builds inputs
from it) and implements `generate()`, which returns:

  signal      : +1 = go long, -1 = go short, 0 = no new instruction, on the bar
                where the decision is made (at that bar's close). The engine
                fills on the NEXT bar's open, so there is no look-ahead.
  indicators  : dict of name -> Series to draw on the price chart.

Strategies that need fills INSIDE a bar (stop losses, breakout levels) instead
implement `simulate()`, which returns the trades as core.engine.Fill objects;
the engine turns them into the same trade sheet / MTM / metrics.

To add a strategy: create a file in this folder with a Strategy subclass and
add it to REGISTRY in strategies/__init__.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Param:
    label: str
    default: float | str
    min: float = 0
    max: float = 0
    step: float = 1
    kind: str = "int"  # "int", "float" or "time" (default "HH:MM"; min/max/step unused)
    help: str | None = None


ALL_DIRECTIONS = ["Long & Short", "Long only", "Short only"]


class Strategy:
    name: str = "Base"
    params: dict[str, Param] = {}
    # Trade directions this strategy supports; the UI offers only these.
    # e.g. a long-only strategy sets  directions = ["Long only"]
    directions: list[str] = ALL_DIRECTIONS
    # Whether the strategy can be run in intraday (square-off daily) mode.
    supports_intraday: bool = True
    # Timeframes the strategy must run on (None = user picks). e.g. ["5m"]
    timeframes: list[str] | None = None
    # Default for the "Capital / trade" input when this strategy is selected.
    default_capital_per_trade: float = 100_000.0

    @property
    def custom_engine(self) -> bool:
        """True when the strategy simulates its own fills (overrides simulate())."""
        return type(self).simulate is not Strategy.simulate

    def validate(self, p: dict) -> str | None:
        """Return an error message if the parameter combination is invalid."""
        return None

    def generate(self, df: pd.DataFrame, **p) -> tuple[pd.Series, dict[str, pd.Series]]:
        raise NotImplementedError

    def simulate(self, df: pd.DataFrame, **p) -> tuple[pd.DataFrame, list, dict[str, pd.Series]]:
        """Own simulation: returns (bars actually used, list of core.engine.Fill, indicators)."""
        raise NotImplementedError
