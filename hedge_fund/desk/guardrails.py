"""Guardrails — limits enforced in code that no model can talk its way past.

Ported from NoFxAiOS/nofx ("the model proposes, the runtime disposes"),
translated from crypto perps to a stock fund:

- Re-entry cooldown: a ticker traded recently is not traded again until the
  cooldown passes. NoFx found re-entering a just-closed symbol a consistent
  loss source and settled on 4h; a weekly fundamentals fund defaults to the
  same.
- Daily order cap: at most N orders per New York trading day — a runaway loop
  cannot churn the account.
- Drawdown breaker: equity falling more than X% from its peak stops the
  autopilot (and can flatten the book).
- Safe mode: N consecutive failed cycles pause the autopilot until a person
  looks — NoFx blocks new entries while the model is failing.

Pure bookkeeping: the Desk asks, this answers. State survives restarts in
~/.hedge-fund/guardrails/<fund>.json.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from hedge_fund.brokers.models import Order

_NY = ZoneInfo("America/New_York")


@dataclass
class GuardrailLimits:
    reentry_cooldown_minutes: float = 240.0
    max_orders_per_day: int = 40
    max_drawdown_pct: float = 0.15
    flatten_on_drawdown: bool = False
    safe_mode_after_failures: int = 3


@dataclass
class GuardrailState:
    last_trade: dict[str, str] = field(default_factory=dict)   # ticker -> ISO time
    orders_today: int = 0
    day: str = ""
    peak_equity: float = 0.0
    consecutive_failures: int = 0
    tripped: str | None = None                                 # why trading is halted


class Guardrails:
    def __init__(self, path: Path, limits: GuardrailLimits | None = None) -> None:
        self.path = path
        self.limits = limits or GuardrailLimits()
        self.state = GuardrailState()
        if path.exists():
            try:
                self.state = GuardrailState(**json.loads(path.read_text()))
            except (TypeError, ValueError):
                pass  # an unreadable state file starts fresh rather than halting forever

    # -- orders -----------------------------------------------------------

    def screen(self, orders: list[Order], now: datetime | None = None) -> tuple[list[Order], list[tuple[Order, str]]]:
        """Split *orders* into (allowed, blocked-with-reason)."""
        now = now or datetime.now(_NY)
        self._roll_day(now)
        allowed: list[Order] = []
        blocked: list[tuple[Order, str]] = []
        if self.state.tripped:
            return [], [(o, f"halted: {self.state.tripped}") for o in orders]
        cooldown = timedelta(minutes=self.limits.reentry_cooldown_minutes)
        budget = self.limits.max_orders_per_day - self.state.orders_today
        for order in orders:
            last = self.state.last_trade.get(order.ticker)
            if last and now - datetime.fromisoformat(last) < cooldown:
                left = cooldown - (now - datetime.fromisoformat(last))
                blocked.append((order, f"re-entry cooldown: traded {order.ticker} recently, "
                                       f"{int(left.total_seconds() // 60)} min left"))
            elif budget <= 0:
                blocked.append((order, f"daily order cap ({self.limits.max_orders_per_day}) reached"))
            else:
                allowed.append(order)
                budget -= 1
        return allowed, blocked

    def record_fill(self, ticker: str, now: datetime | None = None) -> None:
        now = now or datetime.now(_NY)
        self._roll_day(now)
        self.state.last_trade[ticker] = now.isoformat()
        self.state.orders_today += 1
        self._save()

    # -- account health ---------------------------------------------------

    def check_equity(self, equity: float) -> str | None:
        """Update the peak; trip the breaker (and return why) on a deep drawdown."""
        self.state.peak_equity = max(self.state.peak_equity, equity)
        peak = self.state.peak_equity
        if peak > 0 and not self.state.tripped:
            dd = (peak - equity) / peak
            if dd > self.limits.max_drawdown_pct:
                self.state.tripped = (f"drawdown breaker: equity ${equity:,.0f} is {dd:.1%} below its "
                                      f"${peak:,.0f} peak (limit {self.limits.max_drawdown_pct:.0%})")
                self._save()
                return self.state.tripped
        self._save()
        return None

    def cycle_result(self, ok: bool) -> str | None:
        """Count consecutive failures; enter safe mode (and return why) at the limit."""
        self.state.consecutive_failures = 0 if ok else self.state.consecutive_failures + 1
        if self.state.consecutive_failures >= self.limits.safe_mode_after_failures and not self.state.tripped:
            self.state.tripped = f"safe mode: {self.state.consecutive_failures} cycles failed in a row"
            self._save()
            return self.state.tripped
        self._save()
        return None

    def reset(self) -> None:
        """A person has looked: clear the halt and re-anchor the peak."""
        self.state.tripped = None
        self.state.consecutive_failures = 0
        self.state.peak_equity = 0.0
        self._save()

    def view(self) -> dict:
        return {"limits": asdict(self.limits), "state": asdict(self.state)}

    # -- private ----------------------------------------------------------

    def _roll_day(self, now: datetime) -> None:
        day = now.astimezone(_NY).date().isoformat()
        if day != self.state.day:
            self.state.day = day
            self.state.orders_today = 0

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(self.state), indent=1))
