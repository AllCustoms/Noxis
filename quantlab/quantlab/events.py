"""Event study for index-deletion / ETF-reconstitution price pressure.

Input: matrix of abnormal returns, one row per event, one column per day relative to the
effective date (day 0). Two simple tradable rules are scored net of costs:
  pre : short from the announcement close (day -10) to the close before the effective date
  post: buy the close before the effective date and hold the rebound through day +5
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _summ(pnl: pd.Series) -> dict:
    n = len(pnl)
    sd = pnl.std(ddof=1)
    return {
        "mean_pct": float(pnl.mean() * 100),
        "t_stat": float(pnl.mean() / (sd / np.sqrt(n))) if sd > 0 else float("nan"),
        "hit_rate": float((pnl > 0).mean()),
        "n_events": int(n),
    }


def deletion_event_study(ar: pd.DataFrame, cost_bps: float = 25.0, borrow_bps_annual: float = 300.0) -> dict:
    """cost_bps is the round-trip trading cost for a single stock (spread + impact + fees)."""
    cost = cost_bps / 1e4
    pre_days = len(range(-9, 0))
    pre = -ar.loc[:, -9:-1].sum(axis=1) - cost - borrow_bps_annual / 1e4 * pre_days / 252
    post = ar.loc[:, 0:5].sum(axis=1) - cost
    car = ar.mean().cumsum() * 100
    return {"pre_short": _summ(pre), "post_rebound": _summ(post), "mean_car_pct": car}
