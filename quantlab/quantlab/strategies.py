"""Strategy library. Each function takes MarketData and returns target weights.

Row t of the returned frame is the portfolio decided at the close of day t using data up to
and including day t. The backtester applies it to day t+1 returns.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd

from .backtest import BacktestResult, backtest
from .data import MarketData, TRADING_DAYS, calendar_features
from .ml import ml_timing


def _ann_vol(px: pd.DataFrame | pd.Series, window: int):
    return px.pct_change().rolling(window).std() * np.sqrt(TRADING_DAYS)


def trend_following(md, fast=50, slow=200, target_vol=0.10, vol_window=20, max_gross=1.5, assets=("EQ", "BOND")):
    """Long each asset while fast MA > slow MA, sized by inverse volatility (managed-futures style)."""
    px = md.prices[list(assets)]
    on = (px.rolling(fast).mean() > px.rolling(slow).mean()).astype(float)
    w = on * (target_vol / len(assets)) / _ann_vol(px, vol_window)
    gross = w.sum(axis=1).replace(0.0, np.nan)
    scale = np.minimum(1.0, max_gross / gross).fillna(1.0)
    return w.mul(scale, axis=0).fillna(0.0)


def mean_reversion(md, asset="EQ", window=20, entry=1.5, exit_z=0.0):
    """Buy an oversold index (z-score below -entry), sell when it is back to its mean."""
    px = md.prices[asset]
    z = (px - px.rolling(window).mean()) / px.rolling(window).std()
    state = np.where(z <= -entry, 1.0, np.where(z >= exit_z, 0.0, np.nan))
    pos = pd.Series(state, index=px.index).ffill().fillna(0.0)
    return pd.DataFrame({asset: pos})


def pairs_stat_arb(md, a="A", b="B", z_window=60, entry=2.0, exit_=0.5, gross=1.0):
    """Statistical arbitrage: trade the log-price spread of two related assets back to its mean.

    Dollar-neutral (hedge ratio 1), the classic distance-method setup. With real stocks you
    would first pick pairs by testing cointegration on a *prior* period, then estimate the
    hedge ratio from returns; estimating it from price levels on a rolling window injects
    noise that swamps the spread (this lab's first version did exactly that).

    True arbitrage (ETF vs NAV, futures basis) needs latency and infrastructure you cannot
    backtest on daily bars; this relative-value rule is the part retail research can test.
    """
    spread = np.log(md.prices[a]) - np.log(md.prices[b])
    z = (spread - spread.rolling(z_window).mean()) / spread.rolling(z_window).std()
    state = np.where(z <= -entry, 1.0, np.where(z >= entry, -1.0, np.where(z.abs() <= exit_, 0.0, np.nan)))
    pos = pd.Series(state, index=z.index).ffill().fillna(0.0)
    w = pd.DataFrame(0.0, index=z.index, columns=[a, b])
    w[a] = 0.5 * gross * pos
    w[b] = -0.5 * gross * pos
    return w


def turn_of_month(md, first_n=3, include_last=True):
    """Hold equities over the last business day and first `first_n` of each month, else cash."""
    nxt = calendar_features(md.prices.index, ahead=True)  # features of the day we will hold through
    hold = nxt["bday_of_month"] <= first_n
    if include_last:
        hold = hold | (nxt["bdays_to_end"] == 1)
    return pd.DataFrame({"EQ": hold.astype(float)})


def month_end_rebalance(md, last_n=2, threshold=0.01, size=0.5):
    """Fade the month's equity-vs-bond move on the last days of the month.

    Hypothesis: balanced funds and pensions rebalance back to target at month end, so a month
    where equities beat bonds creates selling pressure on equities (and buying of bonds).
    """
    rets = md.prices[["EQ", "BOND"]].pct_change().fillna(0.0)
    mtd = (1.0 + rets).groupby(rets.index.to_period("M")).cumprod()  # month-to-date, causal
    gap = mtd["EQ"] / mtd["BOND"] - 1.0
    nxt = calendar_features(rets.index, ahead=True)
    active = nxt["bdays_to_end"] <= last_n
    sig = -np.sign(gap) * (gap.abs() > threshold)
    w = pd.DataFrame(0.0, index=rets.index, columns=["EQ", "BOND"])
    w["EQ"] = size * sig * active
    w["BOND"] = -w["EQ"]
    return w


def market_timing(md, ma=200, vol_cap=0.22, vol_window=20):
    """Hold equities only when price is above its long MA and volatility is calm; else bonds."""
    eq = md.prices["EQ"]
    on = (eq > eq.rolling(ma).mean()) & (_ann_vol(eq, vol_window) < vol_cap)
    w = pd.DataFrame(0.0, index=eq.index, columns=["EQ", "BOND"])
    w["EQ"] = on.astype(float)
    w["BOND"] = (~on).astype(float)
    return w


STRATEGIES = {
    "trend": trend_following,
    "mean_rev": mean_reversion,
    "pairs": pairs_stat_arb,
    "turn_of_month": turn_of_month,
    "month_end_flow": month_end_rebalance,
    "mkt_timing": market_timing,
    "ml_logit": partial(ml_timing, model="logit"),
    "ml_gbm": partial(ml_timing, model="gbm"),
}

COSTS_BPS = {"pairs": 10.0}  # one-way cost per unit turnover; ETFs ~5, single names ~10
BORROW_BPS = {}  # annual fee on short exposure (default 50)


def run_all(md: MarketData, include_benchmarks: bool = True) -> dict[str, BacktestResult]:
    rets = md.returns
    out: dict[str, BacktestResult] = {}
    for name, fn in STRATEGIES.items():
        out[name] = backtest(
            rets, fn(md), md.cash, COSTS_BPS.get(name, 5.0), BORROW_BPS.get(name, 50.0), name
        )
    if include_benchmarks:
        idx = rets.index
        out["buy_hold_EQ"] = backtest(rets, pd.DataFrame({"EQ": 1.0}, index=idx), md.cash, name="buy_hold_EQ")
        out["static_60_40"] = backtest(
            rets, pd.DataFrame({"EQ": 0.6, "BOND": 0.4}, index=idx), md.cash, name="static_60_40"
        )
    return out
