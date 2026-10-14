"""End-to-end demo.

Runs every strategy on two synthetic markets:
  RIGGED : edges deliberately planted (does the pipeline find them?)
  NULL   : no edges at all           (does the pipeline wrongly "find" any?)
The null world is the important one: a research process that produces profit from noise is
worse than useless.

    python run_demo.py
"""
import pathlib

import pandas as pd

from quantlab.backtest import combine, perf_metrics
from quantlab.data import synthetic_market
from quantlab.events import deletion_event_study
from quantlab.montecarlo import forward_projection, sharpe_test
from quantlab.strategies import STRATEGIES, run_all

OUT = pathlib.Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
N_TESTED = len(STRATEGIES) + 1  # strategies plus the event-study rules: multiple-testing factor
pd.set_option("display.width", 200)


def evaluate(world: str, md) -> pd.DataFrame:
    res = run_all(md)
    rows = {}
    for name, r in res.items():
        m = r.metrics(md.cash)
        if name in STRATEGIES:
            t = sharpe_test(r.net, md.cash)
            m.update(p_value=t["p_value"], p_adj=min(1.0, t["p_value"] * N_TESTED), sh_lo=t["ci_lo"], sh_hi=t["ci_hi"])
        rows[name] = m
    df = pd.DataFrame(rows).T

    blend = combine(pd.DataFrame({k: r.net for k, r in res.items() if k in STRATEGIES}))
    bm = perf_metrics(blend, md.cash)
    df.loc["blend_all_inv_vol"] = pd.Series({**bm, "ann_turnover": float("nan"), "avg_gross": float("nan")})

    print(f"\n=== {world} WORLD ({len(md.prices)} trading days) ===")
    cols = ["cagr", "vol", "sharpe", "max_dd", "ann_turnover", "p_value", "p_adj", "sh_lo", "sh_hi"]
    print(df[cols].to_string(float_format=lambda v: f"{v:7.3f}", na_rep="      -"))

    ev = deletion_event_study(md.deletions)
    print("\nIndex-deletion event study (net of costs), % per event:")
    for k in ("pre_short", "post_rebound"):
        e = ev[k]
        print(f"  {k:13s} mean {e['mean_pct']:6.2f}%  t={e['t_stat']:5.2f}  hit {e['hit_rate']:.0%}  n={e['n_events']}")

    proj = forward_projection(blend)
    print("\nBlend, 1-year Monte Carlo (block bootstrap of its own history):")
    print("  " + "  ".join(f"{k}={v:.3f}" for k, v in proj.items()))

    df.to_csv(OUT / f"{world.lower()}_metrics.csv")
    pd.DataFrame({k: r.net for k, r in res.items()}).assign(blend_all_inv_vol=blend).to_csv(
        OUT / f"{world.lower()}_returns.csv"
    )
    return df


if __name__ == "__main__":
    evaluate("RIGGED", synthetic_market(effects=True, seed=7))
    evaluate("NULL", synthetic_market(effects=False, seed=11))
    print(f"\nMultiple-testing note: p_adj = p_value x {N_TESTED} (Bonferroni). Trust p_adj, not p_value.")
    print(f"CSV outputs written to {OUT}")
    from quantlab.plots import plot_sharpe_cis

    print(f"Chart written to {plot_sharpe_cis(OUT)}")
