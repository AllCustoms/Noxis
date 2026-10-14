"""Walk-forward machine learning layer.

The model never sees the future:
  * features at day t use data up to t only (economic data is shifted by a publication lag)
  * the label is the sign of the next `horizon`-day equity return
  * at each refit date r the model trains only on rows whose label was fully known by r
    (rows s <= r - horizon: a "purge", so overlapping labels cannot leak)
  * it then predicts the following `refit_every` days out-of-sample
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import MarketData, TRADING_DAYS, calendar_features


def build_features(md: MarketData, macro_lag: int = 21) -> pd.DataFrame:
    eq, bd = md.prices["EQ"], md.prices["BOND"]
    r = eq.pct_change()
    f = pd.DataFrame(index=eq.index)
    for k in (1, 5, 21, 63):
        f[f"eq_ret_{k}"] = eq.pct_change(k)
    f["bond_ret_21"] = bd.pct_change(21)
    f["vol_21"] = r.rolling(21).std() * np.sqrt(TRADING_DAYS)
    f["vol_ratio"] = r.rolling(5).std() / r.rolling(63).std()
    f["ma_ratio"] = eq.rolling(50).mean() / eq.rolling(200).mean() - 1.0
    f["drawdown"] = eq / eq.rolling(252, min_periods=20).max() - 1.0
    lagged = md.macro.shift(macro_lag)  # publication lag guard
    f["macro"] = lagged
    f["macro_chg_21"] = lagged.diff(21)

    rets = md.prices[["EQ", "BOND"]].pct_change().fillna(0.0)
    mtd = (1.0 + rets).groupby(eq.index.to_period("M")).cumprod()
    f["mtd_gap"] = mtd["EQ"] / mtd["BOND"] - 1.0
    nxt = calendar_features(eq.index, ahead=True)  # calendar of tomorrow: known in advance
    f["next_bday_of_month"] = nxt["bday_of_month"]
    f["next_bdays_to_end"] = nxt["bdays_to_end"]
    return f


def _make_model(kind: str, seed: int):
    if kind == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500))
    if kind == "gbm":
        return HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0, random_state=seed
        )
    raise ValueError(kind)


def walk_forward_probs(
    X: pd.DataFrame, y: pd.Series, horizon: int, min_train: int, refit_every: int, kind: str, seed: int = 0
) -> pd.Series:
    n = len(X)
    p = pd.Series(np.nan, index=X.index)
    ok = X.notna().all(axis=1).values
    lab = y.notna().values
    for r in range(min_train, n, refit_every):
        tr = np.arange(0, r - horizon + 1)  # label of row s uses price at s+horizon <= r
        tr = tr[ok[tr] & lab[tr]]
        if len(tr) < 250 or y.iloc[tr].nunique() < 2:
            continue
        model = _make_model(kind, seed)
        model.fit(X.iloc[tr], y.iloc[tr])
        te = np.arange(r, min(r + refit_every, n))
        te = te[ok[te]]
        if len(te):
            p.iloc[te] = model.predict_proba(X.iloc[te])[:, 1]
    return p


def ml_timing(
    md: MarketData,
    model: str = "logit",
    horizon: int = 5,
    min_train: int = 756,
    refit_every: int = 63,
    macro_lag: int = 21,
    smooth: int = 5,
    lo: float = 0.45,
    hi: float = 0.60,
    seed: int = 0,
) -> pd.DataFrame:
    """Equity exposure from the walk-forward probability that equities rise over `horizon` days.

    Weight = 0 when P(up) <= lo, 1 when P(up) >= hi, linear in between, then smoothed to
    keep turnover (and therefore costs) down. Remainder sits in cash.
    """
    X = build_features(md, macro_lag)
    eq = md.prices["EQ"]
    fwd = eq.shift(-horizon) / eq - 1.0
    y = (fwd > 0).astype(float).where(fwd.notna())
    p = walk_forward_probs(X, y, horizon, min_train, refit_every, model, seed)
    w = ((p - lo) / (hi - lo)).clip(0.0, 1.0)
    w = w.rolling(smooth, min_periods=1).mean().fillna(0.0)
    out = pd.DataFrame(0.0, index=eq.index, columns=["EQ"])
    out["EQ"] = w
    return out
