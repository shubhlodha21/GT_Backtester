"""Vectorised backtest engine.

Execution model
  * The strategy decides at the CLOSE of bar i; the order fills at the OPEN of bar i+1.
  * Costs = commission per share + slippage (bps of fill price), charged on every fill.
  * Positions are marked to market at every bar close (bar-level MTM), then
    rolled up to a daily MTM sheet.
  * Intraday mode: entries only inside [entry_start, squareoff); any open
    position is closed at the first bar at/after squareoff (or the day's last
    bar if none), so nothing is carried overnight. A cross must happen on the
    same day to open a trade.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time

import numpy as np
import pandas as pd


@dataclass
class Settings:
    initial_capital: float = 100_000.0
    direction: str = "Long & Short"  # "Long & Short" | "Long only" | "Short only"
    sizing: str = "Fixed capital"    # "Fixed capital" | "Fixed qty"
    qty: int = 100
    capital_per_trade: float = 100_000.0
    commission_per_share: float = 0.005
    slippage_bps: float = 1.0
    intraday: bool = False
    entry_start: time = time(9, 30)
    squareoff: time = time(15, 55)


@dataclass
class Result:
    ticker: str
    timeframe: str
    bars: pd.DataFrame          # OHLCV + indicators + position + bar MTM
    trades: pd.DataFrame        # trade sheet
    daily: pd.DataFrame         # daily MTM sheet
    indicators: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.ticker} · {self.timeframe}"


def run_backtest(df: pd.DataFrame, signal: pd.Series, indicators: dict, s: Settings,
                 ticker: str, timeframe: str) -> Result:
    idx = df.index
    o = df["open"].to_numpy(float)
    c = df["close"].to_numpy(float)
    day = idx.normalize()
    tod = idx.time
    intraday = s.intraday and timeframe != "1D"

    # 1) Desired direction after each bar's close, from crossover events.
    ev = signal.astype(float).replace(0, np.nan)
    if intraday:
        in_window = (tod >= s.entry_start) & (tod < s.squareoff)
        ev[~in_window] = np.nan
        state = ev.groupby(day).ffill().fillna(0)
    else:
        state = ev.ffill().fillna(0)
    if s.direction == "Long only":
        state = state.clip(lower=0)
    elif s.direction == "Short only":
        state = state.clip(upper=0)

    # 2) Executed direction: decision at close of i -> held from open of i+1.
    if intraday:
        pos = state.groupby(day).shift(1).fillna(0)
        last_of_day = ~pd.Series(day, index=idx).duplicated(keep="last").to_numpy()
        pos[(tod >= s.squareoff) | last_of_day] = 0
    else:
        pos = state.shift(1).fillna(0)
    pos = pos.to_numpy()

    # 3) Segment the position into trades and size each one at its entry price.
    seg = np.cumsum(np.r_[True, pos[1:] != pos[:-1]])
    first_of_seg = np.r_[True, seg[1:] != seg[:-1]]
    if s.sizing == "Fixed qty":
        qty_seg = np.full(len(pos), float(s.qty))
    else:
        entry_px = pd.Series(np.where(first_of_seg, o, np.nan)).ffill().to_numpy()
        qty_seg = np.floor(s.capital_per_trade / entry_px)
    h = pos * qty_seg  # signed shares held during each bar

    # 4) Bar-level mark-to-market.
    h_prev = np.r_[0.0, h[:-1]]
    c_prev = np.r_[c[0], c[:-1]]
    gross = h_prev * (o - c_prev) + h * (c - o)
    traded = np.abs(h - h_prev)
    cost = traded * (o * s.slippage_bps / 1e4 + s.commission_per_share)
    net = gross - cost
    equity = s.initial_capital + np.cumsum(net)
    peak = np.maximum.accumulate(equity)

    bars = df.copy()
    for name, ser in indicators.items():
        bars[name] = ser
    bars["signal"] = signal
    bars["position"] = h
    bars["gross_pnl"] = gross
    bars["costs"] = cost
    bars["mtm_pnl"] = net
    bars["equity"] = equity
    bars["drawdown"] = equity - peak

    trades = _trade_sheet(bars, pos, state.to_numpy(), seg, s, ticker, timeframe)
    daily = _daily_mtm(bars, trades, s)
    return Result(ticker, timeframe, bars, trades, daily, list(indicators))


def _trade_sheet(bars, pos, state, seg, s: Settings, ticker, timeframe) -> pd.DataFrame:
    cols = ["Trade #", "Ticker", "TF", "Side", "Qty", "Entry Time", "Entry Price",
            "Exit Time", "Exit Price", "Exit Reason", "Bars Held", "Holding Time",
            "Gross P&L", "Costs", "Net P&L", "Return %", "MAE", "MFE", "Cum P&L"]
    active = pos != 0
    if not active.any():
        return pd.DataFrame(columns=cols)

    idx = bars.index
    o, hi, lo, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    qty_abs = np.abs(bars["position"].to_numpy())
    rows = []
    n = len(bars)
    seg_ids = np.unique(seg[active])
    # first/last bar of each segment
    starts = pd.Series(np.arange(n)).groupby(seg).min()
    ends = pd.Series(np.arange(n)).groupby(seg).max()
    for k, sid in enumerate(seg_ids, 1):
        a, b = int(starts[sid]), int(ends[sid])
        side = 1 if pos[a] > 0 else -1
        q = qty_abs[a]
        entry_px = o[a]
        if b + 1 < n:
            exit_t, exit_px = idx[b + 1], o[b + 1]
            reason = "Signal" if state[b] != pos[b] else "Square-off"
            fill_count = 2
        else:
            exit_t, exit_px, reason, fill_count = idx[b], c[b], "Open (MTM)", 1
        gross = side * q * (exit_px - entry_px)
        slip = s.slippage_bps / 1e4
        costs = q * (entry_px * slip + s.commission_per_share)
        if fill_count == 2:
            costs += q * (exit_px * slip + s.commission_per_share)
        # adverse / favourable excursion per share, in price points
        if side > 0:
            mae, mfe = lo[a:b + 1].min() - entry_px, hi[a:b + 1].max() - entry_px
        else:
            mae, mfe = entry_px - hi[a:b + 1].max(), entry_px - lo[a:b + 1].min()
        rows.append({
            "Trade #": k, "Ticker": ticker, "TF": timeframe,
            "Side": "LONG" if side > 0 else "SHORT", "Qty": int(q),
            "Entry Time": idx[a], "Entry Price": round(entry_px, 4),
            "Exit Time": exit_t, "Exit Price": round(exit_px, 4), "Exit Reason": reason,
            "Bars Held": b - a + 1, "Holding Time": exit_t - idx[a],
            "Gross P&L": gross, "Costs": costs, "Net P&L": gross - costs,
            "Return %": 100 * side * (exit_px - entry_px) / entry_px,
            "MAE": mae, "MFE": mfe,
        })
    t = pd.DataFrame(rows)
    t["Cum P&L"] = t["Net P&L"].cumsum()
    return t[cols]


def _daily_mtm(bars: pd.DataFrame, trades: pd.DataFrame, s: Settings) -> pd.DataFrame:
    day = bars.index.normalize()
    g = bars.groupby(day)
    d = pd.DataFrame({
        "Gross P&L": g["gross_pnl"].sum(),
        "Costs": g["costs"].sum(),
        "MTM P&L": g["mtm_pnl"].sum(),
        "EOD Position": g["position"].last(),
        "EOD Close": g["close"].last(),
    })
    closed = trades[trades["Exit Reason"] != "Open (MTM)"] if len(trades) else trades
    if len(closed):
        by_exit = closed.groupby(pd.to_datetime(closed["Exit Time"]).dt.normalize())
        d["Trades Closed"] = by_exit.size().reindex(d.index, fill_value=0)
        d["Realized P&L"] = by_exit["Net P&L"].sum().reindex(d.index, fill_value=0.0)
    else:
        d["Trades Closed"] = 0
        d["Realized P&L"] = 0.0
    d["Cum P&L"] = d["MTM P&L"].cumsum()
    d["Unrealized (open)"] = d["Cum P&L"] - d["Realized P&L"].cumsum()
    d["Equity"] = s.initial_capital + d["Cum P&L"]
    peak = d["Equity"].cummax()
    d["Drawdown"] = d["Equity"] - peak
    d["Drawdown %"] = 100 * d["Drawdown"] / peak
    d.index.name = "Date"
    order = ["Trades Closed", "Realized P&L", "Gross P&L", "Costs", "MTM P&L",
             "Unrealized (open)", "Cum P&L", "Equity", "Drawdown", "Drawdown %",
             "EOD Position", "EOD Close"]
    return d[order]


@dataclass
class Fill:
    """One trade produced by a strategy that runs its own simulation (fills inside a bar).

    e / x   : bar positions of the entry and exit bars (x >= e)
    entry_price / exit_price : actual fill prices (may be inside the bar's range)
    exit_time : timestamp shown as the exit time (defaults to bar x's start)
    extra   : additional trade-sheet columns (e.g. Entry Type, Stop Price)
    """
    e: int
    x: int
    entry_price: float
    exit_price: float
    exit_reason: str
    side: int = 1
    exit_time: pd.Timestamp | None = None
    extra: dict = field(default_factory=dict)


def run_fills(df: pd.DataFrame, fills: list[Fill], indicators: dict, s: Settings,
              ticker: str, timeframe: str) -> Result:
    """Build the same Result as run_backtest, from trades a strategy simulated itself.

    Bar-level MTM with intrabar fills: on the entry bar P&L runs from the fill price to the
    close, on later bars close-to-close, and on the exit bar from the previous close to the
    exit price. Every exit (stop, signal or End of Data) is a real fill and pays costs.
    """
    n = len(df)
    idx = df.index
    o, hi, lo, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    gross, cost, hold = np.zeros(n), np.zeros(n), np.zeros(n)
    signal = np.zeros(n, dtype="int8")
    slip = s.slippage_bps / 1e4
    rows = []
    for k, f in enumerate(fills, 1):
        e, x, pe, px, side = f.e, f.x, f.entry_price, f.exit_price, f.side
        if s.sizing == "Fixed qty":
            q = float(s.qty)
        else:
            q = float(np.floor(s.capital_per_trade / pe)) if pe > 0 else 0.0
        sq = side * q
        if x == e:
            gross[e] += sq * (px - pe)
        else:
            gross[e] += sq * (c[e] - pe)
            gross[e + 1:x] += sq * (c[e + 1:x] - c[e:x - 1])
            gross[x] += sq * (px - c[x - 1])
        hold[e:x] += sq
        c_in = q * (pe * slip + s.commission_per_share)
        c_out = q * (px * slip + s.commission_per_share)
        cost[e] += c_in
        cost[x] += c_out
        signal[e] = side
        if side > 0:
            mae, mfe = lo[e:x + 1].min() - pe, hi[e:x + 1].max() - pe
        else:
            mae, mfe = pe - hi[e:x + 1].max(), pe - lo[e:x + 1].min()
        exit_t = f.exit_time if f.exit_time is not None else idx[x]
        g = sq * (px - pe)
        rows.append({
            "Trade #": k, "Ticker": ticker, "TF": timeframe,
            "Side": "LONG" if side > 0 else "SHORT", "Qty": int(q),
            "Entry Time": idx[e], "Entry Price": round(pe, 4),
            "Exit Time": exit_t, "Exit Price": round(px, 4), "Exit Reason": f.exit_reason,
            "Bars Held": x - e + 1, "Holding Time": exit_t - idx[e],
            "Gross P&L": g, "Costs": c_in + c_out, "Net P&L": g - c_in - c_out,
            "Return %": 100 * side * (px - pe) / pe,
            "MAE": mae, "MFE": mfe, **f.extra,
        })

    net = gross - cost
    equity = s.initial_capital + np.cumsum(net)
    bars = df.copy()
    for name, ser in indicators.items():
        bars[name] = ser
    bars["signal"] = signal
    bars["position"] = hold
    bars["gross_pnl"] = gross
    bars["costs"] = cost
    bars["mtm_pnl"] = net
    bars["equity"] = equity
    bars["drawdown"] = equity - np.maximum.accumulate(equity)

    cols = ["Trade #", "Ticker", "TF", "Side", "Qty", "Entry Time", "Entry Price",
            "Exit Time", "Exit Price", "Exit Reason", "Bars Held", "Holding Time",
            "Gross P&L", "Costs", "Net P&L", "Return %", "MAE", "MFE"]
    extra_cols = list(dict.fromkeys(k for f in fills for k in f.extra))
    if rows:
        trades = pd.DataFrame(rows)
        trades["Cum P&L"] = trades["Net P&L"].cumsum()
    else:
        trades = pd.DataFrame(columns=cols + extra_cols + ["Cum P&L"])
    trades = trades[cols + extra_cols + ["Cum P&L"]]
    daily = _daily_mtm(bars, trades, s)
    return Result(ticker, timeframe, bars, trades, daily, list(indicators))


def combine_daily(results: list[Result], initial_capital: float) -> pd.DataFrame:
    """Portfolio daily MTM: sum of each run's daily P&L (each run trades its own sizing)."""
    pnl = pd.concat({r.key: r.daily["MTM P&L"] for r in results}, axis=1).fillna(0.0)
    out = pnl.copy()
    out["Total MTM P&L"] = pnl.sum(axis=1)
    out["Cum P&L"] = out["Total MTM P&L"].cumsum()
    out["Equity"] = initial_capital + out["Cum P&L"]
    peak = out["Equity"].cummax()
    out["Drawdown"] = out["Equity"] - peak
    out["Drawdown %"] = 100 * out["Drawdown"] / peak
    out.index.name = "Date"
    return out
