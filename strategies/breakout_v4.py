"""High-Breakout (Breakout V4) - ported from D:\\Trading_Strategies\\Gautam Sir\\Breakout_Live\\V4\\sma_v4\\engine.py.

    First entry - buy at the OPEN of the first 5m candle of the test (once per ticker)
    Exit        - a 5m LOW at/below the fixed stop (entry - Stop Loss %) -> sold at the stop
                  (at the candle's open if it opens below the stop). Not stopped -> closed at
                  the last close ("End of Data"). No end-of-day exit: held overnight.
    Re-entry    - after a stop: the first later 5m candle (start time inside Entry From - Entry To)
                  whose HIGH goes above the highest 5m high since the FIRST entry (one running high
                  that never resets) -> buy AT that level (at the open if it opens above it).
                  On that breakout candle the stop counts only if the candle CLOSES at/below it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.engine import Fill

from .base import Param, Strategy

INITIAL = "Initial open"
BREAKOUT = "Breakout re-entry"

# NYSE half days: the regular session ends 13:00 ET; later prints are after-hours and are dropped.
EARLY_CLOSE_DAYS = pd.DatetimeIndex(pd.to_datetime([
    "2018-07-03", "2018-11-23", "2018-12-24", "2019-07-03", "2019-11-29", "2019-12-24",
    "2020-11-27", "2020-12-24", "2021-11-26", "2022-11-25", "2023-07-03", "2023-11-24",
    "2024-07-03", "2024-11-29", "2024-12-24", "2025-07-03", "2025-11-28", "2025-12-24",
    "2026-11-27", "2026-12-24"]))
SESSION_CLOSE = pd.Timedelta(hours=16)
EARLY_CLOSE = pd.Timedelta(hours=13)
BAR = pd.Timedelta(minutes=5)


def _minutes(hhmm) -> int:
    h, m = str(hhmm)[:5].split(":")
    return int(h) * 60 + int(m)


def drop_after_early_close(df: pd.DataFrame) -> pd.DataFrame:
    day = df.index.normalize()
    late = day.isin(EARLY_CLOSE_DAYS) & (df.index - day >= EARLY_CLOSE)
    return df[~late]


def bar_close_times(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Close time of each 5m candle, capped at the session close (16:00, or 13:00 on half days)."""
    end = index + BAR
    day = index.normalize()
    cap = day + pd.to_timedelta(np.where(day.isin(EARLY_CLOSE_DAYS), EARLY_CLOSE.to_timedelta64(),
                                         SESSION_CLOSE.to_timedelta64()))
    return end.where(end <= cap, cap)


class BreakoutV4(Strategy):
    """5m breakout: buy the first open, fixed stop loss, re-enter on a new high since the first entry. Long only, held overnight."""

    name = "High-Breakout"
    params = {
        "entry_from": Param("Entry From", "09:30", kind="time",
                            help="Start time of the first 5m candle allowed to give a breakout re-entry."),
        "entry_to": Param("Entry To", "15:55", kind="time",
                          help="Start time of the last 5m candle allowed to give a breakout re-entry."),
        "stop_loss_pct": Param("Stop Loss %", 0.25, 0.0, 20.0, 0.05, kind="float",
                               help="Fixed stop below the entry price, checked on every 5m low. 0 = no stop "
                                    "(the first entry is held to the end)."),
    }
    directions = ["Long only"]
    supports_intraday = False
    timeframes = ["5m"]
    # Uses the app-wide default ($100,000 per trade, same as Initial capital).

    def validate(self, p):
        if _minutes(p["entry_from"]) > _minutes(p["entry_to"]):
            return "Entry From must be at or before Entry To"
        return None

    def simulate(self, df: pd.DataFrame, stop_loss_pct: float, entry_from, entry_to):
        m5 = drop_after_early_close(df)
        o, h, l, c = (m5[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        times, n = m5.index, len(m5)
        tracked = pd.Series(np.nan, index=m5.index)
        if n < 2:
            return m5, [], {"Tracked high": tracked}
        close_t = bar_close_times(times)
        tod = times.hour * 60 + times.minute
        can_enter = np.asarray((tod >= _minutes(entry_from)) & (tod <= _minutes(entry_to)))
        run_max = np.maximum.accumulate(h)        # highest high since the first entry (bar 0)
        tracked[:] = run_max
        sl = float(stop_loss_pct)

        fills, e, kind, level = [], 0, INITIAL, None
        while True:
            intrabar = kind == BREAKOUT and o[e] <= level
            entry = level if intrabar else o[e]
            stop = entry * (1 - sl / 100) if sl > 0 else None
            stop_hit = None
            if stop is not None:
                lows = l[e:].copy()
                if intrabar:                      # breakout candle: only a close at/below the stop counts
                    lows[0] = c[e]
                hit = np.flatnonzero(lows <= stop)
                if len(hit):
                    stop_hit = e + int(hit[0])
            if stop_hit is not None:
                x = stop_hit
                gap = bool(o[x] < stop and not (intrabar and x == e))
                px, xt = (o[x] if gap else stop), times[x]
                why = f"Stop loss {sl:g}%" + (" (gap, at open)" if gap else "")
            else:
                x = n - 1
                px, xt, why = c[x], close_t[x], "End of Data"
            fills.append(Fill(e, x, float(entry), float(px), why, exit_time=xt, extra={
                "Entry Type": kind,
                "Breakout Level": round(float(level), 4) if kind == BREAKOUT else np.nan,
                "Stop Price": round(stop, 4) if stop is not None else np.nan,
            }))
            if stop_hit is None:
                break                              # held to the end of the data
            level = float(run_max[x])              # only ever goes up
            nxt = x + 1
            hit = np.flatnonzero((h[nxt:] > level) & can_enter[nxt:])
            if not len(hit):
                break
            e, kind = nxt + int(hit[0]), BREAKOUT
        return m5, fills, {"Tracked high": tracked}
