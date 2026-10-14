"""Chart: Sharpe ratio (net of costs) with bootstrap interval for every strategy, in the rigged
and null markets. Emphasis form: one accent hue marks strategies that survive the
multiple-testing adjustment, neutral gray for the rest, with filled vs hollow markers as a
second channel so identity never rests on colour alone. The numbers behind it are in
outputs/*_metrics.csv (the table view).
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE, ACCENT = "#e1e0d9", "#c3c2b7", "#2a78d6"

LABELS = {
    "trend": "Trend following",
    "mean_rev": "Mean reversion (index)",
    "pairs": "Pairs / stat-arb",
    "turn_of_month": "Turn of month",
    "month_end_flow": "Month-end rebalancing flow",
    "mkt_timing": "Market timing (MA + vol)",
    "ml_logit": "ML: logistic regression",
    "ml_gbm": "ML: gradient boosting",
}


def plot_sharpe_cis(outdir: str | pathlib.Path) -> pathlib.Path:
    out = pathlib.Path(outdir)
    worlds = [("rigged", "Rigged market: edges planted"), ("null", "Null market: no edges exist")]
    plt.rcParams["font.family"] = "sans-serif"
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharex=True, sharey=True, facecolor=SURFACE)
    fig.subplots_adjust(left=0.215, right=0.975, top=0.76, bottom=0.2, wspace=0.06)

    for ax, (key, title) in zip(axes, worlds):
        df = pd.read_csv(out / f"{key}_metrics.csv", index_col=0).loc[list(LABELS)]
        ax.set_facecolor(SURFACE)
        for i, (name, r) in enumerate(df.iterrows()):
            y = len(df) - 1 - i
            hit = r["p_adj"] < 0.05
            col = ACCENT if hit else MUTED
            ax.plot([r["sh_lo"], r["sh_hi"]], [y, y], color=col, lw=2, solid_capstyle="round", zorder=2)
            ax.plot(
                r["sharpe"], y, "o", ms=9, zorder=3, markerfacecolor=col if hit else SURFACE,
                markeredgecolor=SURFACE if hit else MUTED, markeredgewidth=2 if hit else 1.6,
            )
            if hit:
                ax.annotate(f"{r['sharpe']:.2f}", (r["sh_hi"], y), xytext=(7, 0), textcoords="offset points",
                            va="center", fontsize=10, color=INK)
        ax.axvline(0, color=BASE, lw=1.2, zorder=1)
        ax.set_title(title, loc="left", fontsize=11.5, color=INK, pad=10, fontweight="semibold")
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels([LABELS[n] for n in df.index][::-1], fontsize=10.5, color=INK2)
        ax.tick_params(axis="x", colors=MUTED, labelsize=10, length=0)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel("Sharpe ratio after costs (bar = 90% bootstrap interval)", fontsize=10, color=INK2, labelpad=8)
        ax.set_ylim(-0.7, len(df) - 0.3)

    fig.suptitle("The pipeline finds the edges that were planted, and none where there were none",
                 x=0.02, y=0.965, ha="left", fontsize=14, color=INK, fontweight="semibold")
    fig.text(0.02, 0.885, "15 years of synthetic daily data per market. Real-market results will differ; "
             "this checks the machinery, not the strategies.", fontsize=10.5, color=INK2, ha="left")
    handles = [
        Line2D([], [], marker="o", ls="", ms=9, markerfacecolor=ACCENT, markeredgecolor=SURFACE,
               markeredgewidth=2, label="Significant after multiple-testing adjustment (p_adj < 0.05)"),
        Line2D([], [], marker="o", ls="", ms=9, markerfacecolor=SURFACE, markeredgecolor=MUTED,
               markeredgewidth=1.6, label="Not significant"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.005), ncol=2, frameon=False,
               fontsize=10, labelcolor=INK2, handletextpad=0.3, columnspacing=2.0)
    path = out / "sharpe_by_strategy.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path
