"""Index universes: named ticker lists the UI can filter by.

Each list is a CSV in core/universes/ with a `Symbol` column (extra columns
such as Security / GICS Sector are allowed and ignored). To refresh a list,
replace its CSV and update AS_OF. To add an index, drop in a CSV and add an
entry to UNIVERSES.

These are CURRENT memberships. Backtesting today's members over past years
adds survivorship bias: companies that joined recently were not in the index
back then.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

ALL = "All tickers"
_DIR = Path(__file__).resolve().parent / "universes"

# display name -> CSV file
UNIVERSES: dict[str, str | None] = {
    ALL: None,
    "Nasdaq-100": "nasdaq100.csv",
    "S&P 500": "sp500.csv",
}

# When each list was last taken (DD/MM/YYYY), shown in the UI.
AS_OF = {
    "Nasdaq-100": "22/10/2026",   # Wikipedia, List of NASDAQ-100 companies
    "S&P 500": "29/10/2026",      # D:\DataBento\sp500_tickers.csv
}


@lru_cache(maxsize=None)
def members(name: str) -> tuple[str, ...]:
    """All index members (whether or not we have data for them)."""
    file = UNIVERSES.get(name)
    if file is None:
        return ()
    syms = pd.read_csv(_DIR / file)["Symbol"].astype(str).str.strip()
    return tuple(dict.fromkeys(s for s in syms if s))


def filter_available(name: str, available: list[str]) -> tuple[list[str], list[str]]:
    """(members we have data for, members we don't). ALL returns everything available."""
    if UNIVERSES.get(name) is None:
        return list(available), []
    have = set(available)
    m = members(name)
    return sorted(t for t in m if t in have), sorted(t for t in m if t not in have)
