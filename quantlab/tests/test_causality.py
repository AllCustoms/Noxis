"""Guards against the classic backtesting bug: using information from the future.

Run with:  python tests/test_causality.py     (or pytest, if installed)
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from quantlab.backtest import backtest  # noqa: E402
from quantlab.broker import PaperBroker, RiskLimits  # noqa: E402
from quantlab.data import calendar_features, synthetic_market  # noqa: E402
from quantlab.strategies import STRATEGIES  # noqa: E402


def test_calendar_features():
    idx = pd.bdate_range("2024-01-29", "2024-02-06")
    cf = calendar_features(idx)
    by_date = {d.strftime("%Y-%m-%d"): (r.bday_of_month, r.bdays_to_end) for d, r in cf.iterrows()}
    assert by_date["2024-01-31"][1] == 1  # last business day of January
    assert by_date["2024-01-30"][1] == 2
    assert by_date["2024-02-01"][0] == 1  # first business day of February
    assert by_date["2024-02-05"][0] == 3
    ahead = calendar_features(idx, ahead=True)
    assert ahead.loc["2024-01-30", "bdays_to_end"] == 1  # tomorrow (31 Jan) is month end
    assert ahead.loc["2024-02-02", "bday_of_month"] == 3  # Friday -> next business day is Monday the 5th, the 3rd of the month


def test_weight_at_t_earns_return_t_plus_1():
    md = synthetic_market(years=3, seed=3)
    rets = md.returns
    w = pd.DataFrame(0.0, index=rets.index, columns=["EQ"])
    w.iloc[100, 0] = 1.0
    res = backtest(rets, w, md.cash, cost_bps=0.0, borrow_bps_annual=0.0)
    diff = res.net - md.cash
    day = rets.index[101]
    assert abs(diff.loc[day] - (rets["EQ"].iloc[101] - md.cash.iloc[101])) < 1e-12
    assert diff.drop(day).abs().max() < 1e-12


def test_strategies_never_use_future_data():
    md = synthetic_market(years=6, seed=3)
    cut = 1100  # beyond the ML warm-up of 756 days, so ML weights are non-trivial too
    short = md.truncate(cut)
    for name, fn in STRATEGIES.items():
        full_w = fn(md).iloc[:cut]
        cut_w = fn(short)
        assert np.allclose(full_w.values, cut_w.values, atol=1e-9), f"{name} peeks at the future"


def test_paper_broker_limits_and_kill_switch():
    b = PaperBroker(10_000, RiskLimits(max_drawdown_halt=0.10))
    px = {"X": 100.0}
    b.rebalance({"X": 5.0}, px, "d1")  # absurd leverage request
    assert sum(abs(w) for w in b.weights(px).values()) <= 1.0 + 1e-9
    b.peak = b.equity(px)
    crash = {"X": 80.0}  # 20% fall breaches the 10% halt
    b.rebalance({"X": 1.0}, crash, "d2")
    assert b.halted and not b.positions  # liquidated, refuses to re-enter
    b.rebalance({"X": 1.0}, crash, "d3")
    assert not b.positions


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
