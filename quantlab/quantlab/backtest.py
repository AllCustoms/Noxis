"""Vectorised backtester.

Timing convention (the single rule that prevents look-ahead bias):
    weights[t] are decided with information up to the CLOSE of day t
    and earn the return of day t+1.
Costs are charged on turnover; short positions pay a borrow fee; idle capital earns cash.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import TRADING_DAYS


def perf_metrics(r: pd.Series, rf: pd.Series | float = 0.0) -> dict:
    r = r.dropna()
    rf_s = rf.reindex(r.index).fillna(0.0) if isinstance(rf, pd.Series) else rf
    ex = r - rf_s
    years = len(r) / TRADING_DAYS
    eq = (1.0 + r).cumprod()
    sd = r.std()
    downside = np.sqrt((np.minimum(ex, 0.0) ** 2).mean())
    dd = eq / eq.cummax() - 1.0
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0
    return {
        "cagr": float(cagr),
        "vol": float(sd * np.sqrt(TRADING_DAYS)),
        "sharpe": float(ex.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan"),
        "sortino": float(ex.mean() / downside * np.sqrt(TRADING_DAYS)) if downside > 0 else float("nan"),
        "max_dd": float(dd.min()),
        "calmar": float(cagr / abs(dd.min())) if dd.min() < 0 else float("nan"),
    }


@dataclass
class BacktestResult:
    name: str
    net: pd.Series
    gross: pd.Series
    weights: pd.DataFrame  # weights actually held each day (already lagged)
    turnover: pd.Series
    costs: pd.Series

    @property
    def equity(self) -> pd.Series:
        return (1.0 + self.net).cumprod()

    def metrics(self, cash: pd.Series | float = 0.0) -> dict:
        m = perf_metrics(self.net, cash)
        m["ann_turnover"] = float(self.turnover.mean() * TRADING_DAYS)
        m["avg_gross"] = float(self.weights.abs().sum(axis=1).mean())
        return m


def backtest(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    cash: pd.Series,
    cost_bps: float = 5.0,
    borrow_bps_annual: float = 50.0,
    name: str = "strategy",
) -> BacktestResult:
    r = returns[list(weights.columns)]
    w = weights.reindex(r.index).fillna(0.0)
    w_held = w.shift(1).fillna(0.0)  # decided at close t-1, earns return t

    turnover = w_held.diff().abs().sum(axis=1)
    turnover.iloc[0] = w_held.iloc[0].abs().sum()

    cash_ret = cash.reindex(r.index).fillna(0.0)
    idle = 1.0 - w_held.sum(axis=1)  # negative if levered: borrowing costs the cash rate
    gross = (w_held * r).sum(axis=1) + idle * cash_ret

    short_exposure = (-w_held.clip(upper=0.0)).sum(axis=1)
    costs = turnover * cost_bps / 1e4 + short_exposure * borrow_bps_annual / 1e4 / TRADING_DAYS
    return BacktestResult(name, gross - costs, gross, w_held, turnover, costs)


def combine(returns: pd.DataFrame, method: str = "inv_vol", window: int = 126) -> pd.Series:
    """Blend strategy return streams. inv_vol uses only past volatility (lagged one day)."""
    n = returns.shape[1]
    if method == "equal":
        return returns.mean(axis=1)
    vol = returns.rolling(window, min_periods=30).std().shift(1)
    inv = 1.0 / vol
    w = inv.div(inv.sum(axis=1), axis=0).fillna(1.0 / n)
    return (w * returns).sum(axis=1)
