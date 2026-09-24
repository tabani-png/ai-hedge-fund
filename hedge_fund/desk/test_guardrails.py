"""Guardrails unit tests — cooldown, daily cap, persistence."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from hedge_fund.brokers.models import Order
from hedge_fund.desk.guardrails import GuardrailLimits, Guardrails

NY = ZoneInfo("America/New_York")
T0 = datetime(2026, 9, 24, 10, 0, tzinfo=NY)


def _o(t):
    return Order(ticker=t, side="buy", quantity=1, price=1.0)


def test_cooldown_expires(tmp_path):
    g = Guardrails(tmp_path / "g.json", GuardrailLimits(reentry_cooldown_minutes=60))
    g.record_fill("AAPL", T0)
    assert g.screen([_o("AAPL")], T0 + timedelta(minutes=30))[0] == []
    assert len(g.screen([_o("AAPL")], T0 + timedelta(minutes=61))[0]) == 1


def test_daily_cap_resets_next_day(tmp_path):
    g = Guardrails(tmp_path / "g.json", GuardrailLimits(max_orders_per_day=2, reentry_cooldown_minutes=0))
    for t in ("A", "B"):
        g.record_fill(t, T0)
    allowed, blocked = g.screen([_o("C")], T0)
    assert allowed == [] and "daily order cap" in blocked[0][1]
    assert len(g.screen([_o("C")], T0 + timedelta(days=1))[0]) == 1


def test_state_persists_and_halt_blocks_everything(tmp_path):
    g = Guardrails(tmp_path / "g.json", GuardrailLimits(max_drawdown_pct=0.1))
    g.check_equity(100.0)
    assert g.check_equity(85.0)
    again = Guardrails(tmp_path / "g.json")
    assert again.state.tripped
    assert again.screen([_o("A")], T0)[1][0][1].startswith("halted")
