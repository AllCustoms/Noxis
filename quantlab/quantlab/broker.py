"""Execution layer: a paper broker with hard risk limits.

Nothing here can send a real order. ``LiveBroker`` is intentionally a stub that refuses to
run; wiring a real broker API should be a deliberate, separate decision made only after
months of paper trading (see README).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RiskLimits:
    max_weight_per_asset: float = 1.0  # fraction of equity in any one asset
    max_gross: float = 1.0  # sum of |weights|; 1.0 = no leverage
    allow_short: bool = False
    max_turnover_per_rebalance: float = 0.5  # fraction of equity traded in one go
    max_drawdown_halt: float = 0.15  # halt and go to cash if equity falls 15% from peak
    min_trade_value: float = 25.0  # ignore dust trades
    cash_buffer: float = 0.005  # keep 0.5% spare for fees and slippage


class PaperBroker:
    def __init__(
        self,
        cash: float = 10_000.0,
        limits: RiskLimits | None = None,
        commission_bps: float = 2.0,
        slippage_bps: float = 3.0,
        state_path: str | None = None,
    ):
        self.limits = limits or RiskLimits()
        self.commission_bps, self.slippage_bps = commission_bps, slippage_bps
        self.state_path = state_path
        self.cash = cash
        self.positions: dict[str, float] = {}
        self.peak = cash
        self.halted = False
        self.trades: list[dict] = []
        if state_path and Path(state_path).exists():
            self._load()

    # ---- state
    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(q * prices[a] for a, q in self.positions.items())

    def weights(self, prices: dict[str, float]) -> dict[str, float]:
        eq = self.equity(prices)
        return {a: q * prices[a] / eq for a, q in self.positions.items() if abs(q) > 1e-12}

    def reset_halt(self) -> None:
        """Manual human decision to resume trading after a drawdown halt."""
        self.halted = False

    def _save(self) -> None:
        if self.state_path:
            Path(self.state_path).write_text(
                json.dumps(
                    {"cash": self.cash, "positions": self.positions, "peak": self.peak,
                     "halted": self.halted, "trades": self.trades},
                    indent=2,
                )
            )

    def _load(self) -> None:
        s = json.loads(Path(self.state_path).read_text())
        self.cash, self.positions, self.peak = s["cash"], s["positions"], s["peak"]
        self.halted, self.trades = s["halted"], s["trades"]

    # ---- risk
    def _apply_limits(self, target: dict[str, float]) -> dict[str, float]:
        L = self.limits
        t = {}
        for a, w in target.items():
            if not L.allow_short:
                w = max(w, 0.0)
            t[a] = max(-L.max_weight_per_asset, min(L.max_weight_per_asset, w))
        gross = sum(abs(w) for w in t.values())
        cap = L.max_gross * (1.0 - L.cash_buffer)
        if gross > cap:
            t = {a: w * cap / gross for a, w in t.items()}
        return t

    # ---- trading
    def rebalance(self, target: dict[str, float], prices: dict[str, float], date: str) -> list[dict]:
        L = self.limits
        eq = self.equity(prices)
        self.peak = max(self.peak, eq)
        if not self.halted and eq < self.peak * (1.0 - L.max_drawdown_halt):
            self.halted = True  # kill switch
        if self.halted:
            target = {}  # liquidate and stay in cash until a human resets
        target = self._apply_limits(target)

        deltas = {}
        for a in set(target) | set(self.positions):
            cur = self.positions.get(a, 0.0) * prices[a]
            d = target.get(a, 0.0) * eq - cur
            if abs(d) >= L.min_trade_value:
                deltas[a] = d
        turnover = sum(abs(d) for d in deltas.values()) / eq
        if not self.halted and turnover > L.max_turnover_per_rebalance:
            k = L.max_turnover_per_rebalance / turnover
            deltas = {a: d * k for a, d in deltas.items()}

        done = []
        for a, d in sorted(deltas.items(), key=lambda kv: kv[1]):  # sells before buys
            side = 1.0 if d > 0 else -1.0
            fill = prices[a] * (1.0 + side * self.slippage_bps / 1e4)
            qty = d / prices[a]
            fee = abs(qty) * fill * self.commission_bps / 1e4
            self.cash -= qty * fill + fee
            self.positions[a] = self.positions.get(a, 0.0) + qty
            if abs(self.positions[a]) < 1e-12:
                del self.positions[a]
            rec = {"date": date, "asset": a, "qty": qty, "fill": fill, "fee": fee, "halted": self.halted}
            done.append(rec)
        self.trades.extend(done)
        self._save()
        return done


class LiveBroker:
    """Deliberately not implemented."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Live trading is intentionally not wired up. Run paper trading for months first, "
            "then add a broker adapter yourself with human approval of every strategy change."
        )
