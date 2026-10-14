"""Daily paper-trading runner.

    python -m quantlab.live --strategy trend --state paper_state.json
    python -m quantlab.live --csv prices.csv --equity SPY --bond AGG --strategy trend

Run it once per day after the close (schedule it yourself with cron or Task Scheduler). It
loads data, computes today's target weights, and rebalances the PAPER broker.
"""
from __future__ import annotations

import argparse

import pandas as pd

from .broker import PaperBroker, RiskLimits
from .data import MarketData, load_csv_prices, synthetic_market
from .strategies import STRATEGIES


def run_once(md: MarketData, strategy: str, broker: PaperBroker, names: dict[str, str] | None = None) -> dict:
    w = STRATEGIES[strategy](md).iloc[-1]
    last = md.prices.iloc[-1]
    names = names or {}
    target = {names.get(a, a): float(v) for a, v in w.items()}
    prices = {names.get(a, a): float(last[a]) for a in last.index}
    date = str(md.prices.index[-1].date())
    trades = broker.rebalance(target, prices, date)
    return {
        "date": date,
        "target": target,
        "trades": trades,
        "equity": broker.equity(prices),
        "halted": broker.halted,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="trend", choices=sorted(STRATEGIES))
    ap.add_argument("--csv", help="wide CSV of adjusted closes (date + one column per ticker)")
    ap.add_argument("--equity", default="EQ")
    ap.add_argument("--bond", default="BOND")
    ap.add_argument("--state", default="paper_state.json")
    ap.add_argument("--capital", type=float, default=10_000.0)
    a = ap.parse_args()

    if a.csv:
        md = MarketData.from_prices(load_csv_prices(a.csv), a.equity, a.bond)
        names = {"EQ": a.equity, "BOND": a.bond}
    else:
        print("No --csv given: using SYNTHETIC demo data (prices are not real).")
        md, names = synthetic_market(), {}
    broker = PaperBroker(a.capital, RiskLimits(), state_path=a.state)
    out = run_once(md, a.strategy, broker, names)
    print(pd.Series(out["target"]).round(3).to_string())
    print(f"{out['date']}: {len(out['trades'])} paper trades, equity {out['equity']:,.2f}, halted={out['halted']}")


if __name__ == "__main__":
    main()
