"""Monte Carlo tools built on the circular block bootstrap.

Blocks (default 21 days) keep volatility clustering and short-term autocorrelation that a
plain shuffle would destroy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TRADING_DAYS


def block_bootstrap_paths(x: np.ndarray, n_paths: int, horizon: int, block: int = 21, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    N = len(x)
    n_blocks = -(-horizon // block)
    starts = rng.integers(0, N, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % N
    return x[idx.reshape(n_paths, -1)[:, :horizon]]


def _sharpe(paths: np.ndarray) -> np.ndarray:
    return paths.mean(axis=1) / paths.std(axis=1, ddof=1) * np.sqrt(TRADING_DAYS)


def sharpe_test(net: pd.Series, cash: pd.Series, n: int = 2000, block: int = 21, seed: int = 0) -> dict:
    """Is the observed Sharpe distinguishable from luck?

    p_value: share of bootstrap worlds with zero true edge (excess returns demeaned) whose
             Sharpe is at least as high as the one observed.
    ci_lo/ci_hi: 5th and 95th percentile of the Sharpe when resampling the actual returns.
    """
    ex = (net - cash.reindex(net.index).fillna(0.0)).dropna().values
    obs = ex.mean() / ex.std(ddof=1) * np.sqrt(TRADING_DAYS)
    null = _sharpe(block_bootstrap_paths(ex - ex.mean(), n, len(ex), block, seed))
    boot = _sharpe(block_bootstrap_paths(ex, n, len(ex), block, seed + 1))
    return {
        "sharpe": float(obs),
        "p_value": float((np.sum(null >= obs) + 1) / (n + 1)),
        "ci_lo": float(np.percentile(boot, 5)),
        "ci_hi": float(np.percentile(boot, 95)),
    }


def forward_projection(net: pd.Series, horizon: int = TRADING_DAYS, n: int = 5000, block: int = 21, seed: int = 2) -> dict:
    """Resample history into `n` possible futures of length `horizon` and summarise the spread."""
    paths = block_bootstrap_paths(net.values, n, horizon, block, seed)
    eq = np.cumprod(1.0 + paths, axis=1)
    term = eq[:, -1] - 1.0
    maxdd = (eq / np.maximum.accumulate(eq, axis=1) - 1.0).min(axis=1)
    return {
        "return_p05": float(np.percentile(term, 5)),
        "return_median": float(np.median(term)),
        "return_p95": float(np.percentile(term, 95)),
        "prob_loss": float((term < 0).mean()),
        "prob_drawdown_over_20pct": float((maxdd < -0.20).mean()),
        "median_max_drawdown": float(np.median(maxdd)),
    }
