"""Data layer.

* ``synthetic_market`` builds a fake-but-structured market so the whole pipeline can be
  developed and *validated* offline.  Every planted effect can be switched off
  (``effects=False``) to give a "null world" in which no strategy should have an edge.
  If the pipeline finds an edge in the null world, the pipeline is broken.
* Loaders for real data (CSV, yfinance, FRED) are included but need network access.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- calendar
def calendar_features(idx: pd.DatetimeIndex, ahead: bool = False) -> pd.DataFrame:
    """Calendar position of each date, using business-day rules only (no market data).

    bday_of_month: 1 for the first business day of the month.
    bdays_to_end:  1 for the last business day of the month, 2 for the one before, ...
    ahead=True returns the features of the *next* business day, which is what a strategy
    deciding at today's close needs to know about the day it is about to hold through.
    Holidays are ignored (weekday calendar), which is exact for the synthetic data and a
    small approximation for real exchange calendars.
    """
    d = idx.values.astype("datetime64[D]")
    if ahead:
        d = np.busday_offset(d, 1, roll="forward")
    month = d.astype("datetime64[M]")
    m_start = month.astype("datetime64[D]")
    next_m = (month + 1).astype("datetime64[D]")
    return pd.DataFrame(
        {
            "bday_of_month": np.busday_count(m_start, d + 1),
            "bdays_to_end": np.busday_count(d, next_m),
        },
        index=idx,
    )


# --------------------------------------------------------------------------- container
@dataclass
class MarketData:
    prices: pd.DataFrame  # must contain EQ and BOND; A and B optional (pairs)
    cash: pd.Series  # daily cash return
    macro: pd.Series  # economic indicator on the price calendar, publication lag NOT applied
    deletions: pd.DataFrame | None = None  # event-study matrix (events x relative day)
    regime: pd.Series | None = None

    @property
    def returns(self) -> pd.DataFrame:
        return self.prices.pct_change().fillna(0.0)

    def truncate(self, n: int) -> "MarketData":
        """First n rows only. Used to prove that strategies never peek at the future."""
        return MarketData(
            self.prices.iloc[:n],
            self.cash.iloc[:n],
            self.macro.iloc[:n],
            self.deletions,
            None if self.regime is None else self.regime.iloc[:n],
        )

    @classmethod
    def from_prices(
        cls,
        prices: pd.DataFrame,
        equity: str,
        bond: str,
        cash_rate: float = 0.03,
        macro: pd.Series | None = None,
    ) -> "MarketData":
        """Wrap real adjusted-close prices. Equity/bond columns are renamed EQ/BOND."""
        px = prices.rename(columns={equity: "EQ", bond: "BOND"}).sort_index().ffill().dropna()
        cash = pd.Series(cash_rate / TRADING_DAYS, index=px.index)
        mac = (
            macro.reindex(px.index).ffill().fillna(0.0)
            if macro is not None
            else pd.Series(0.0, index=px.index)
        )
        return cls(px, cash, mac)


# --------------------------------------------------------------------------- synthetic
def synthetic_deletion_events(
    n_events: int, rng: np.random.Generator, effects: bool = True, idio_vol: float = 0.012
) -> pd.DataFrame:
    """Abnormal daily returns around an index-deletion effective date (relative day 0).

    Planted (effects=True): price pressure of -0.25%/day on days -9..-1 as index trackers
    are expected to sell, then a +0.30%/day rebound on days 0..+5.
    """
    cols = list(range(-10, 11))
    ar = rng.normal(0.0, idio_vol, size=(n_events, len(cols)))
    if effects:
        for j, d in enumerate(cols):
            if -9 <= d <= -1:
                ar[:, j] -= 0.0025
            elif 0 <= d <= 5:
                ar[:, j] += 0.0030
    return pd.DataFrame(ar, columns=cols, index=pd.RangeIndex(n_events, name="event"))


def synthetic_market(years: int = 15, seed: int = 7, effects: bool = True) -> MarketData:
    """Fake market with fat tails, volatility regimes and (optionally) planted effects.

    Planted when effects=True:
      * persistent bull / bear / sideways drift regimes        -> trend following can work
      * mean-reverting spread between assets A and B           -> pairs / stat-arb can work
      * extra return on the last and first 3 business days     -> turn-of-month can work
      * month-end reversal of equity-vs-bond month-to-date gap -> rebalancing-flow can work
      * weak lagged link from a macro indicator to equities    -> ML may find it
      * price pressure around index deletions                  -> event study can find it
    With effects=False none of these exist (same vol clustering and fat tails, no edge).
    The planted effects are deliberately exaggerated so the pipeline's detection power can be
    checked in 15 years of data; real-world versions are smaller and noisier.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=int(years * TRADING_DAYS))
    n = len(idx)
    cal = calendar_features(idx)

    # volatility / drift regimes: 0 = bull, 1 = bear, 2 = sideways
    mu = np.array([0.00065, -0.00090, 0.00010]) if effects else np.full(3, 0.00030)
    vol = np.array([0.0075, 0.0180, 0.0100])
    # long-lived regimes (bull ~290 days, bear ~50, sideways ~100); exaggerated vs real markets
    P = np.array([[0.9965, 0.0020, 0.0015], [0.012, 0.980, 0.008], [0.006, 0.004, 0.990]])
    regime = np.zeros(n, dtype=int)
    for t in range(1, n):
        regime[t] = rng.choice(3, p=P[regime[t - 1]])

    z_eq = rng.standard_t(5, n) / np.sqrt(5 / 3)  # unit-variance fat tails
    shock = vol[regime] * z_eq
    eq = mu[regime] + shock

    macro = np.zeros(n)
    eps = rng.normal(size=n)
    for t in range(1, n):
        macro[t] = 0.995 * macro[t - 1] + 0.1 * eps[t]  # roughly unit variance

    bond = 0.00015 + 0.0042 * rng.standard_t(5, n) / np.sqrt(5 / 3) - 0.10 * shock

    if effects:
        eq[1:] += 0.0003 * macro[:-1]
        tom = (cal["bday_of_month"].values <= 3) | (cal["bdays_to_end"].values == 1)
        eq[tom] += 0.0012

    # month-end reversal driven by the month-to-date equity-vs-bond gap (known the day before)
    mtd_e = mtd_b = 1.0
    for t in range(n):
        if cal["bday_of_month"].iloc[t] == 1:
            mtd_e = mtd_b = 1.0
        if effects and cal["bdays_to_end"].iloc[t] <= 2:
            gap = float(np.clip(mtd_e / mtd_b - 1.0, -0.06, 0.06))
            eq[t] += -0.08 * gap
            bond[t] += 0.05 * gap
        mtd_e *= 1.0 + eq[t]
        mtd_b *= 1.0 + bond[t]

    # pair of assets sharing a common factor with an OU spread between them
    kappa = 0.08 if effects else 0.0
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = spread[t - 1] * (1.0 - kappa) + 0.006 * rng.normal()
    common = np.cumsum(eq + 0.004 * rng.normal(size=n))
    pa = 100.0 * np.exp(common + 0.5 * spread)
    pb = 80.0 * np.exp(common - 0.5 * spread)

    prices = pd.DataFrame(
        {"EQ": 100.0 * np.cumprod(1.0 + eq), "BOND": 100.0 * np.cumprod(1.0 + bond), "A": pa, "B": pb},
        index=idx,
    )
    return MarketData(
        prices=prices,
        cash=pd.Series(0.0001, index=idx),
        macro=pd.Series(macro, index=idx),
        deletions=synthetic_deletion_events(60, rng, effects),
        regime=pd.Series(regime, index=idx),
    )


# --------------------------------------------------------------------------- real data
def load_csv_prices(path: str) -> pd.DataFrame:
    """Wide CSV: first column = date, one adjusted-close column per asset."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return df.dropna(how="all").ffill()


def load_yfinance(tickers: list[str], start: str = "2005-01-01") -> pd.DataFrame:
    """Adjusted closes via yfinance (pip install yfinance). Needs internet access."""
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install yfinance") from e
    df = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    return df.dropna(how="all").ffill()


def load_fred(series_id: str, publication_lag_days: int = 30) -> pd.Series:
    """Economic series from FRED, shifted by a crude publication lag.

    Warning: this is the *latest revised* history. Point-in-time (as first published) data
    is available from ALFRED; revisions are a classic source of fake backtest performance.
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    raw = pd.read_csv(url, index_col=0, parse_dates=True).iloc[:, 0]
    s = pd.to_numeric(raw, errors="coerce").dropna()
    s.index = s.index + pd.Timedelta(days=publication_lag_days)
    return s
