"""Streamlit UI for the backtester.  Run with:  streamlit run app.py"""
from __future__ import annotations

import io
from datetime import date, time, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core.config import DEFAULT_TIMEFRAMES, TIMEFRAMES
from core.data import data_range, list_tickers, load_bars
from core.fmt import DATE_JS, DATETIME_JS, DATETIME_XL, DATE_XL, fmt_date, for_csv, for_excel, is_date_only, style_time_axis
from core.universes import UNIVERSES, filter_available
from core.engine import Result, Settings, combine_daily, run_backtest, run_fills
from core.metrics import annual_vol_pct, buy_hold_pct, summarize
from strategies import REGISTRY

st.set_page_config(page_title="GT_Backtester", page_icon="📈", layout="wide")

PORTFOLIO = "Portfolio (all runs)"
BENCHMARK = "SPY"
UP, DOWN = "#26a69a", "#ef5350"


@st.cache_data(show_spinner=False, max_entries=64)
def cached_bars(ticker, tf, start, end):
    return load_bars(ticker, tf, start, end)


@st.cache_data(show_spinner=False, max_entries=1024)
def cached_range(ticker):
    return data_range(ticker)


def _date_col(values):
    return st.column_config.DatetimeColumn(format=DATE_JS if is_date_only(values) else DATETIME_JS)


def table(df: pd.DataFrame, **kw):
    """Display a frame with DD/MM/YYYY dates, 2-dp numbers and readable durations."""
    df = df.copy()
    cfg = {}
    if pd.api.types.is_datetime64_any_dtype(df.index):
        cfg["_index"] = _date_col(df.index)
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            cfg[col] = _date_col(df[col])
        elif pd.api.types.is_timedelta64_dtype(df[col]):
            df[col] = df[col].astype(str).str.replace("0 days ", "", regex=False)
        elif pd.api.types.is_float_dtype(df[col]):
            cfg[col] = st.column_config.NumberColumn(format="%.2f")
    st.dataframe(df, column_config=cfg, **kw)


# ───────────────────────────── Sidebar ─────────────────────────────
# Plain widgets (no st.form) so the strategy's own options appear as soon as it
# is selected. Nothing heavy runs until "Run backtest" is pressed.
# Not cached: re-listed on every run so tickers added to / removed from the data
# folder show up (or disappear) without restarting the app.
tickers_all = list_tickers()
run = False
with st.sidebar:
    st.title("📈 GT_Backtester")

    st.subheader("Universe")
    universe = st.selectbox("Index", list(UNIVERSES), key="universe",
                            help="Limit the ticker list to an index's current members.")
    options, _ = filter_available(universe, tickers_all)
    # Key per universe, so switching index resets the picker to that index's tickers.
    # Picking an index pre-selects all of its tickers (remove any you don't want);
    # "All tickers" starts with just AAPL.
    if UNIVERSES[universe] is None:
        default = ["AAPL"] if "AAPL" in options else options[:1]
    else:
        default = options
    tickers = st.multiselect("Tickers", options, key=f"tickers:{universe}", default=default)
    if len(tickers) > 20:
        st.caption(f"{len(tickers)} tickers selected. Large runs on 1m/5m bars take a while.")
    tickers = [t for t in tickers if t in tickers_all]  # drop any whose data was removed
    # A strategy may fix its timeframe (e.g. High-Breakout always runs on 5m). The strategy
    # picker sits further down, so read its current value from session state.
    _chosen = st.session_state.get("strategy")
    fixed_tfs = REGISTRY[_chosen].timeframes if _chosen in REGISTRY else None
    if fixed_tfs:
        tfs = list(fixed_tfs)
        st.caption(f"Timeframe: **{', '.join(tfs)}** (set by {_chosen})")
    else:
        tfs = st.multiselect("Timeframes", list(TIMEFRAMES), default=DEFAULT_TIMEFRAMES)

    # Dates default to the full span of data for the selected tickers. The widget
    # key depends on the selection, so changing tickers resets the dates to the
    # new full range.
    if tickers:
        ranges = {t: cached_range(t) for t in tickers}
        data_start = min(lo for lo, _ in ranges.values())
        data_end = max(hi for _, hi in ranges.values())
    else:
        ranges, data_start, data_end = {}, date(2018, 5, 1), date.today()
    sel_key = "|".join(sorted(tickers))
    c1, c2 = st.columns(2)
    start = c1.date_input("From", data_start, min_value=data_start, max_value=data_end,
                          format=DATE_JS, key=f"from:{sel_key}")
    end = c2.date_input("To", data_end, min_value=data_start, max_value=data_end,
                        format=DATE_JS, key=f"to:{sel_key}")
    if tickers:
        st.caption(f"📅 Data available: **{fmt_date(data_start)} → {fmt_date(data_end)}** "
                   "(defaults to the full range)")
        if len({r for r in ranges.values()}) > 1:
            with st.expander("Data range per ticker"):
                st.dataframe(pd.DataFrame(
                    [{"Ticker": t, "From": fmt_date(lo), "To": fmt_date(hi)} for t, (lo, hi) in ranges.items()]
                ).set_index("Ticker"), use_container_width=True)
    st.caption("Session: RTH only (09:30–16:00 ET)")

    st.subheader("Strategy")
    strat_name = st.selectbox("Strategy", list(REGISTRY), index=None, placeholder="Select a strategy…",
                              key="strategy")

    if strat_name is None:
        st.info("Select a strategy to see its parameters and execution options.")
    else:
        strat = REGISTRY[strat_name]()
        if strat.__doc__:
            st.caption(strat.__doc__.strip())
        params = {}
        cols = st.columns(2)
        for i, (k, p) in enumerate(strat.params.items()):
            key = f"param:{strat_name}:{k}"
            if p.kind == "int":
                params[k] = cols[i % 2].number_input(p.label, int(p.min), int(p.max), int(p.default),
                                                     int(p.step), key=key, help=p.help)
            elif p.kind == "time":
                hh, mm = map(int, str(p.default).split(":"))
                params[k] = cols[i % 2].time_input(p.label, time(hh, mm), step=300, key=key,
                                                   help=p.help).strftime("%H:%M")
            else:
                params[k] = cols[i % 2].number_input(p.label, float(p.min), float(p.max), float(p.default),
                                                     float(p.step), key=key, help=p.help)

        st.subheader("Execution")
        capital = st.number_input("Initial capital ($)", 1_000.0, 1e9, 100_000.0, 10_000.0)

        # Only the directions this strategy supports.
        if len(strat.directions) > 1:
            direction = st.selectbox("Direction", strat.directions, key=f"dir:{strat_name}")
        else:
            direction = strat.directions[0]
            st.caption(f"Direction: **{direction}** (set by this strategy)")

        sizing = st.radio("Position sizing", ["Fixed capital", "Fixed qty"], horizontal=True)
        cap_per_trade, qty = 100_000.0, 100
        if sizing == "Fixed capital":
            cap_per_trade = st.number_input("Capital / trade ($)", 100.0, 1e9,
                                            float(strat.default_capital_per_trade), 10_000.0,
                                            key=f"cap:{strat_name}")
        else:
            qty = st.number_input("Qty / trade", 1, 10_000_000, 100)

        c1, c2 = st.columns(2)
        comm = c1.number_input("Commission $/share", 0.0, 1.0, 0.0, 0.001, format="%.4f")
        slip = c2.number_input("Slippage (bps)", 0.0, 100.0, 0.0, 0.5)

        intraday, entry_start, squareoff = False, time(9, 30), time(15, 55)
        if strat.supports_intraday:
            intraday = st.toggle("Intraday (square-off daily)", value=False,
                                 help="No overnight positions. Ignored for 1D.")
            if intraday:
                c1, c2 = st.columns(2)
                entry_start = c1.time_input("Entries from", time(9, 30), step=300)
                squareoff = c2.time_input("Square-off at", time(15, 55), step=300)

        run = st.button("▶ Run backtest", type="primary", use_container_width=True)

# ───────────────────────────── Run ─────────────────────────────
if run:
    err = strat.validate(params)
    if not tickers or not tfs:
        st.error("Pick at least one ticker and one timeframe.")
    elif start > end:
        st.error("'From' date is after 'To' date.")
    elif err:
        st.error(err)
    else:
        settings = Settings(capital, direction, sizing, int(qty), cap_per_trade, comm, slip,
                            intraday, entry_start, squareoff)
        results: list[Result] = []
        jobs = [(t, tf) for t in tickers for tf in tfs]
        bar = st.progress(0.0, "Starting…")
        for i, (t, tf) in enumerate(jobs):
            bar.progress(i / len(jobs), f"{t} · {tf}  (first use of a ticker builds its 1-minute cache, ~5 s)")
            df = cached_bars(t, tf, start, end)
            if df.empty or len(df) < 3:
                st.warning(f"No data for {t} {tf} in range.")
                continue
            if strat.custom_engine:
                used, fills, ind = strat.simulate(df, **params)
                results.append(run_fills(used, fills, ind, settings, t, tf))
            else:
                sig, ind = strat.generate(df, **params)
                results.append(run_backtest(df, sig, ind, settings, t, tf))
        bar.empty()
        st.session_state["results"] = results
        st.session_state["settings"] = settings
        st.session_state["label"] = f"{strat_name} {params} · RTH · {fmt_date(start)} → {fmt_date(end)}"

results: list[Result] = st.session_state.get("results", [])
if not results:
    st.info("Choose tickers and timeframes, select a strategy in the sidebar, then press **Run backtest**.")
    st.stop()

settings: Settings = st.session_state["settings"]
st.caption(st.session_state["label"])

# ───────────────────────────── Summary ─────────────────────────────
def benchmarks(ticker: str | None, d0: date, d1: date) -> dict:
    """Buy & hold of the ticker and of SPY over the same dates as the backtest (RTH daily bars)."""
    out = {}
    if ticker is not None:
        out["Buy & Hold %"] = buy_hold_pct(cached_bars(ticker, "1D", d0, d1))
    try:
        spy = cached_bars(BENCHMARK, "1D", d0, d1)
    except FileNotFoundError:
        spy = None
    out["SPY Buy & Hold %"] = buy_hold_pct(spy)
    out["SPY Volatility %"] = annual_vol_pct(spy["close"]) if spy is not None else np.nan
    return out


rows = []
for r in results:
    m = summarize(r.daily["Equity"], r.trades, settings.initial_capital)
    d0, d1 = r.daily.index[0].date(), r.daily.index[-1].date()
    rows.append({"Run": r.key, **m, **benchmarks(r.ticker, d0, d1)})
portfolio = combine_daily(results, settings.initial_capital * len(results)) if len(results) > 1 else None
if portfolio is not None:
    all_trades = pd.concat([r.trades for r in results], ignore_index=True)
    m = summarize(portfolio["Equity"], all_trades, settings.initial_capital * len(results))
    bench = benchmarks(None, portfolio.index[0].date(), portfolio.index[-1].date())
    # Portfolio buy & hold = equal-weight average of each run's buy & hold (same capital per run).
    bench["Buy & Hold %"] = float(np.nanmean([row["Buy & Hold %"] for row in rows]))
    rows.append({"Run": PORTFOLIO, **m, **bench})
summary = pd.DataFrame(rows).set_index("Run")
# Keep related numbers side by side: returns with benchmarks, volatility with SPY volatility.
_order = ["Net P&L", "Return %", "Buy & Hold %", "SPY Buy & Hold %", "CAGR %", "Max DD", "Max DD %",
          "Sharpe", "Annualized Volatility %", "SPY Volatility %"]
summary = summary[[c for c in _order if c in summary] + [c for c in summary if c not in _order]]

best = summary.drop(index=PORTFOLIO, errors="ignore")
k1, k2, k3, k4, k5 = st.columns(5)
top = summary.loc[PORTFOLIO] if portfolio is not None else summary.iloc[0]
k1.metric("Net P&L", f"${top['Net P&L']:,.0f}", f"{top['Return %']:.1f}%")
k2.metric("Max drawdown", f"${top['Max DD']:,.0f}", f"{top['Max DD %']:.1f}%", delta_color="off")
k3.metric("Sharpe", f"{top['Sharpe']:.2f}")
k4.metric("Trades", f"{int(top.get('Trades', 0)):,}", f"{top.get('Win %', 0):.1f}% win", delta_color="off")
k5.metric("Best run", best["Net P&L"].idxmax(), f"${best['Net P&L'].max():,.0f}")

st.subheader("Summary by run")
table(summary, use_container_width=True)

run_names = ([PORTFOLIO] if portfolio is not None else []) + [r.key for r in results]
by_key = {r.key: r for r in results}

tab_eq, tab_mtm, tab_trades, tab_chart, tab_dl = st.tabs(
    ["Equity & drawdown", "MTM sheet", "Trade sheet", "Price chart", "Export"])

# ───────────────────────────── Equity ─────────────────────────────
with tab_eq:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
    for r in results:
        fig.add_trace(go.Scatter(x=r.daily.index, y=r.daily["Cum P&L"], name=r.key, mode="lines"), 1, 1)
    if portfolio is not None:
        fig.add_trace(go.Scatter(x=portfolio.index, y=portfolio["Cum P&L"], name=PORTFOLIO,
                                 line=dict(width=3, color="#111")), 1, 1)
        dd = portfolio["Drawdown"]
    else:
        dd = results[0].daily["Drawdown"]
    fig.add_trace(go.Scatter(x=dd.index, y=dd, fill="tozeroy", name="Drawdown",
                             line=dict(color=DOWN), showlegend=False), 2, 1)
    fig.update_yaxes(title_text="Cumulative P&L ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown ($)", row=2, col=1)
    style_time_axis(fig, intraday=False)
    fig.update_layout(height=560, hovermode="x unified", margin=dict(t=10, b=10),
                      legend=dict(orientation="h", y=1.04))
    st.plotly_chart(fig, use_container_width=True)

# ───────────────────────────── MTM ─────────────────────────────
with tab_mtm:
    pick = st.selectbox("Run", run_names, key="mtm_run")
    daily = portfolio if pick == PORTFOLIO else by_key[pick].daily
    pnl_col = "Total MTM P&L" if pick == PORTFOLIO else "MTM P&L"

    st.markdown("**Monthly MTM P&L ($)**")
    monthly = daily[pnl_col].groupby([daily.index.year, daily.index.month]).sum().unstack()
    monthly.columns = [pd.Timestamp(2000, m, 1).strftime("%b") for m in monthly.columns]
    monthly["Year"] = monthly.sum(axis=1)
    monthly.index.name = "Year"
    st.dataframe(monthly.style.format("{:,.0f}", na_rep="")
                 .background_gradient(cmap="RdYlGn", axis=None, subset=monthly.columns[:-1]),
                 use_container_width=True)

    st.markdown("**Daily MTM sheet**")
    table(daily.sort_index(ascending=False), use_container_width=True, height=420)
    st.caption("MTM P&L = change in marked-to-market value that day (realized + change in open P&L − costs). "
               "Realized P&L = net P&L of trades closed that day.")

# ───────────────────────────── Trades ─────────────────────────────
with tab_trades:
    pick_t = st.selectbox("Run", run_names, key="trade_run")
    trades = (pd.concat([r.trades for r in results], ignore_index=True)
              if pick_t == PORTFOLIO else by_key[pick_t].trades)
    c1, c2, c3 = st.columns(3)
    side_f = c1.multiselect("Side", ["LONG", "SHORT"], default=["LONG", "SHORT"])
    res_f = c2.radio("Result", ["All", "Winners", "Losers"], horizontal=True)
    trades_v = trades[trades["Side"].isin(side_f)]
    if res_f == "Winners":
        trades_v = trades_v[trades_v["Net P&L"] > 0]
    elif res_f == "Losers":
        trades_v = trades_v[trades_v["Net P&L"] <= 0]
    c3.metric("Shown", f"{len(trades_v):,} trades", f"${trades_v['Net P&L'].sum():,.0f}")
    table(trades_v, use_container_width=True, height=480, hide_index=True)

# ───────────────────────────── Chart ─────────────────────────────
with tab_chart:
    c1, c2 = st.columns([2, 3])
    pick_c = c1.selectbox("Run", [r.key for r in results], key="chart_run")
    r = by_key[pick_c]
    days = pd.Index(r.bars.index.normalize().unique())
    default_from = days[max(0, len(days) - (250 if r.timeframe == "1D" else 5))]
    win = c2.slider("Window", min_value=days[0].date(), max_value=days[-1].date(),
                    value=(default_from.date(), days[-1].date()), format=DATE_JS)
    b = r.bars.loc[pd.Timestamp(win[0]): pd.Timestamp(win[1]) + timedelta(days=1)]
    if len(b) > 20_000:
        st.warning(f"{len(b):,} bars in window — showing the last 20,000. Narrow the window for detail.")
        b = b.iloc[-20_000:]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=b.index, open=b["open"], high=b["high"], low=b["low"], close=b["close"],
                                 name=r.ticker, increasing_line_color=UP, decreasing_line_color=DOWN), 1, 1)
    for name, color in zip(r.indicators, ["#ff9800", "#2962ff", "#9c27b0", "#795548"]):
        fig.add_trace(go.Scatter(x=b.index, y=b[name], name=name, line=dict(width=1.3, color=color)), 1, 1)
    t = r.trades
    if len(t):
        t = t[(t["Entry Time"] <= b.index[-1]) & (t["Exit Time"] >= b.index[0])]
        for side, sym, col in (("LONG", "triangle-up", UP), ("SHORT", "triangle-down", DOWN)):
            ts = t[t["Side"] == side]
            fig.add_trace(go.Scatter(x=ts["Entry Time"], y=ts["Entry Price"], mode="markers", name=f"{side} entry",
                                     marker=dict(symbol=sym, size=11, color=col, line=dict(width=1, color="#000")),
                                     customdata=ts[["Trade #", "Qty"]],
                                     hovertemplate="#%{customdata[0]} qty %{customdata[1]}<br>%{y:.2f}"), 1, 1)
        fig.add_trace(go.Scatter(x=t["Exit Time"], y=t["Exit Price"], mode="markers", name="Exit",
                                 marker=dict(symbol="x", size=9, color="#444"),
                                 customdata=t[["Trade #", "Net P&L", "Exit Reason"]],
                                 hovertemplate="#%{customdata[0]} %{customdata[2]}<br>P&L %{customdata[1]:,.2f}"), 1, 1)
    fig.add_trace(go.Scatter(x=b.index, y=b["equity"] - settings.initial_capital, name="Cum P&L",
                             line=dict(color="#555")), 2, 1)
    breaks = []
    if r.timeframe != "1D":
        breaks.append(dict(bounds=["sat", "mon"]))
        breaks.append(dict(bounds=[16, 9.5], pattern="hour"))
    fig.update_xaxes(rangebreaks=breaks, rangeslider_visible=False)
    style_time_axis(fig, intraday=r.timeframe != "1D")
    fig.update_layout(height=650, hovermode="x unified", margin=dict(t=10, b=10),
                      legend=dict(orientation="h", y=1.04))
    st.plotly_chart(fig, use_container_width=True)

# ───────────────────────────── Export ─────────────────────────────
with tab_dl:
    all_trades = pd.concat([r.trades for r in results], ignore_index=True)
    c1, c2, c3 = st.columns(3)
    c1.download_button("Trade sheet (CSV)", for_csv(all_trades),
                       "tradesheet.csv", "text/csv", use_container_width=True)
    mtm_long = pd.concat([r.daily.assign(Run=r.key) for r in results]).reset_index()
    c2.download_button("Daily MTM, all runs (CSV)", for_csv(mtm_long),
                       "mtm_daily.csv", "text/csv", use_container_width=True)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter", datetime_format=DATETIME_XL, date_format=DATE_XL) as xw:
        summary.to_excel(xw, sheet_name="Summary")
        for_excel(all_trades).to_excel(xw, sheet_name="Trades", index=False)
        if portfolio is not None:
            for_excel(portfolio).to_excel(xw, sheet_name="MTM Portfolio")
        for r in results:
            sheet = f"MTM {r.ticker} {r.timeframe}"[:31]
            for_excel(r.daily).to_excel(xw, sheet_name=sheet)
    c3.download_button("Full workbook (Excel)", buf.getvalue(), "backtest.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)
    st.caption("Workbook sheets: Summary, Trades, MTM Portfolio (if >1 run), and one daily MTM sheet per ticker × timeframe.")
