# quantlab

A small research lab for testing systematic investing ideas, built so that a strategy has to
*earn* trust before it gets near money. It is not financial advice and it does not place real
trades.

```
data  ->  strategies  ->  backtest (costs, no look-ahead)  ->  Monte Carlo significance  ->  paper broker
```

## Run it

```bash
pip install numpy pandas scikit-learn matplotlib      # yfinance optional, for real data
python tests/test_causality.py                        # 4 safety tests
python run_demo.py                                    # all strategies, rigged + null market, chart, CSVs
python -m quantlab.live --strategy trend              # one paper-trading step (synthetic data unless --csv)
```

## What is in it

| File | Job |
|---|---|
| `quantlab/data.py` | Synthetic market with switchable planted edges; loaders for CSV, yfinance, FRED |
| `quantlab/backtest.py` | Vectorised backtester: weights decided at close *t* earn day *t+1*; costs, borrow fees, cash yield |
| `quantlab/strategies.py` | Trend following, index mean reversion, pairs stat-arb, turn-of-month, month-end rebalancing flow, market timing, two ML timers |
| `quantlab/ml.py` | Walk-forward ML: expanding window, purged labels, publication-lagged economic data |
| `quantlab/montecarlo.py` | Block bootstrap: Sharpe p-values and intervals, 1-year forward projections |
| `quantlab/events.py` | Event study for index-deletion price pressure |
| `quantlab/broker.py` | Paper broker with position caps, no leverage, turnover cap, drawdown kill switch |
| `quantlab/live.py` | Daily runner: data -> target weights -> paper rebalance, state saved to JSON |

## How the lab keeps itself honest

1. **Null market.** Same fat tails and volatility clustering, zero edge. If a strategy "works"
   there, the pipeline is broken. In the demo, nothing passes.
2. **Rigged market.** Edges planted on purpose. The pipeline should find them, and it found
   pairs, month-end flow and the logistic ML timer. It also exposed a bug in my first pairs
   implementation (hedge ratio estimated from price levels was pure noise).
3. **Causality test.** Every strategy's weights on truncated data must equal its weights on the
   full data over the same dates. A strategy that peeks at the future fails this.
4. **Multiple-testing adjustment.** Nine ideas were tried, so p-values are multiplied by nine
   (Bonferroni). Trying many ideas and keeping the winner is how fake edges get published.
5. **Costs and borrow fees** are charged on every trade.

## What firms use, and how much of it to believe

I could not verify the claims "ETF month-end rebalancing worked for 20+ years" or "the deletion
trade worked since the first ETF". Treat them as hypotheses; this lab is the tool for testing
them on real data. What is publicly documented, and what to watch for:

| Idea | Who is known to use it | Evidence and caveats | Here |
|---|---|---|---|
| Trend following | Managed-futures funds (CTAs) | Long academic record across many markets (e.g. Hurst, Ooi, Pedersen 2017), but long flat or losing stretches | `trend_following` |
| Statistical arbitrage / pairs | Quant hedge funds, prop desks | Gatev, Goetzmann, Rouwenhorst (2006) found profits that later shrank as the trade got crowded | `pairs_stat_arb` |
| True arbitrage (ETF vs NAV, futures basis) | Market makers, HFT | Latency and infrastructure bound; not testable on daily bars | not modelled |
| Index inclusion/deletion pressure | Event-driven funds | Published work (e.g. Greenwood and Sammon 2022) reports the classic effect has shrunk a lot; costs and borrow matter | `events.py` |
| Month-end rebalancing flows | Discussed by macro and flow desks | Plausible mechanism (balanced funds and pensions rebalance), evidence mixed, size unclear | `month_end_rebalance` |
| Turn-of-month | Academic anomaly (Ariel 1987; Lakonishok and Smidt 1988) | Small per day, and costs eat much of it; the demo shows this | `turn_of_month` |
| Market timing | Tactical allocation funds | Moving-average timing (e.g. Faber 2007) mostly reduces drawdowns rather than adding return | `market_timing` |
| ML forecasting | Quant funds, with heavy infrastructure | Simple models usually match complex ones out of sample; leakage is the main failure | `ml.py` |

What the big quant firms actually run is proprietary. The sources above are public research.

## Known limits (read before trusting any number)

- **The demo uses synthetic data.** It validates the machinery, not the strategies. Run on real
  prices before concluding anything (`load_yfinance`, `load_csv_prices`, `MarketData.from_prices`;
  these loaders are written but were not exercised in the environment this was built in).
- **Survivorship and revisions.** Real ETF lists drop dead funds, and FRED returns revised
  economic history. Use point-in-time data (ALFRED) for anything macro.
- **Single stocks need borrow data and realistic costs**, which this lab approximates.
- **Weight drift** between rebalances is ignored by the backtester (small for daily rebalancing).
- **Do not tune on the same data you judge on.** Hold out the most recent years untouched.

## Before any real money

1. Replicate results on real data with an untouched hold-out period.
2. Paper trade for months with `quantlab.live` (schedule it yourself with cron or Task Scheduler).
3. Only then consider a broker adapter. `LiveBroker` deliberately raises `NotImplementedError`,
   and the paper broker's limits (no leverage, 15% drawdown halt) should be carried over.
4. Trading your own account is one thing; managing other people's money is regulated.
