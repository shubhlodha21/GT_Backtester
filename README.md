# GT_Backtester

Modular backtester for the DataBento 1-second parquet files in `D:\DataBento\data_split_adjusted`.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Optional, to pre-build the 1-minute cache for every ticker (~5 s each):

```bash
python build_cache.py
```

## Layout

| Path | What it does |
|---|---|
| `app.py` | Streamlit UI: parameters in the sidebar, summary / equity / MTM / trade sheet / chart / export tabs |
| `core/config.py` | Data path, session hours, timeframe list |
| `core/corporate_actions.csv` | Record of the spin-off adjustment factors. Already applied directly to the raw files (originals in `D:\DataBento\backup_before_spinoff_adjust`); `ADJUST_CORPORATE_ACTIONS` in `core/config.py` is therefore `False` |
| `core/data.py` | Builds `cache/1m/<TICKER>.parquet` from raw 1 s data, resamples to 1m, 5m, 15m, 1h, 4h and 1D (bars anchored at 09:30), RTH-only filter |
| `core/engine.py` | Vectorised engine: next-bar-open fills, costs, intraday square-off, bar MTM, trade sheet, daily MTM |
| `core/fmt.py` | Date format used everywhere (UI, charts, CSV, Excel): `DD/MM/YYYY`, `DD/MM/YYYY HH:MM` with time |
| `core/metrics.py` | Net P&L, CAGR, Sharpe, max DD, win %, avg / largest win & loss % … |
| `strategies/` | One file per strategy; register it in `strategies/__init__.py` |

## Adding a strategy

```python
# strategies/my_strat.py
from .base import Param, Strategy

class MyStrat(Strategy):
    name = "My Strat"
    params = {"length": Param("Length", 20, 2, 500)}
    directions = ["Long & Short", "Long only", "Short only"]  # UI offers only these
    supports_intraday = True                                   # hide the intraday toggle if False

    def generate(self, df, length):
        signal = ...          # +1 go long / -1 go short / 0 nothing, per bar
        return signal, {"My line": some_series}
```

Then add `MyStrat` to `REGISTRY` in `strategies/__init__.py`. The UI creates inputs for its parameters automatically.

### Strategies that fill inside a bar

Stops and breakout levels fill at a price inside the candle, which the signal engine (next-bar open) can't express.
Such a strategy implements `simulate(df, **params)` instead of `generate()` and returns
`(bars used, [core.engine.Fill, ...], indicators)`; the engine turns the fills into the same trade sheet, MTM and
metrics. Optional class attributes: `timeframes = ["5m"]` (locks the timeframe), `directions = ["Long only"]`,
`supports_intraday = False`, `default_capital_per_trade`. Params can be `kind="time"` (`"HH:MM"`).

Example: `strategies/breakout_v4.py` — the V4 5m breakout (first open entry, fixed stop loss, re-entry on a new
high since the first entry), ported from `Breakout_Live\V4\sma_v4\engine.py` and checked trade-for-trade against it.

## Execution model

- Signals are computed on bar close; orders fill at the **next bar's open** (no look-ahead).
- Costs = commission per share + slippage in bps of the fill price, on every fill.
- Sizing: fixed quantity, or fixed capital per trade (`floor(capital / entry price)` shares).
- Intraday mode: entries only between "Entries from" and "Square-off at"; open positions are closed at the square-off bar, nothing is held overnight.
- Prices are adjusted for splits and for spin-offs, both in the source data (spin-offs listed in `core/corporate_actions.csv`: every price before the ex-date was multiplied by `factor`). They are **not** adjusted for dividends, so Buy & Hold figures are price-only.
- Timestamps are US/Eastern. Backtests use RTH only (09:30–16:00); pre-market and after-hours bars are ignored.
