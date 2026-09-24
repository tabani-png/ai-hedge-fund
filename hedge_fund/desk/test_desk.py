"""Desk tests — preview, approve, execute, autopilot, kill switch — with
fake analysts, a fake data client, and a real PaperBroker on tmp_path."""

import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
import yaml

from hedge_fund.brokers.paper import PaperBroker
from hedge_fund.data.models import Price
from hedge_fund.desk.desk import Desk, us_market_open
from hedge_fund.fund.spec import Fund
from hedge_fund.models import Signal

CLOSES = {"AAPL": 200.0, "MSFT": 400.0}


class FakeData:
    def get_prices(self, ticker, start, end, **kw):
        c = CLOSES.get(ticker)
        return [] if c is None else [Price(open=c, close=c, high=c, low=c, volume=1, time=f"{end}T00:00:00Z")]


class Bull:
    name = "bull"

    def predict(self, ticker, date, data_client):
        return Signal(model_name="bull", ticker=ticker, date=date, value=1.0, reasoning="up only")


@pytest.fixture
def desk(tmp_path, monkeypatch):
    mandates = tmp_path / "mandates"
    mandates.mkdir()
    (mandates / "f.yaml").write_text(yaml.safe_dump({
        "name": "f", "capital": 10_000,
        "strategies": [{"name": "solo", "models": [{"name": "bull"}]}],
        "risk": {"max_position_pct": 0.5, "max_gross_exposure": 1.0},
    }))
    d = Desk(mandates_dir=mandates, paper_dir=tmp_path / "paper", guard_dir=tmp_path / "guard",
             data_client_factory=FakeData,
             fund_factory=lambda spec: Fund(spec, models={"solo": [Bull()]}))
    monkeypatch.setattr("hedge_fund.desk.desk.us_market_open", lambda now=None: True)
    return d


def _wait(desk, job_id, *statuses):
    for _ in range(200):
        if desk.jobs[job_id]["status"] in statuses:
            return desk.jobs[job_id]
        time.sleep(0.01)
    raise AssertionError(desk.jobs[job_id])


def test_preview_then_approve_trades_the_paper_book(desk, tmp_path):
    job_id = desk.start_cycle("f", ["AAPL", "MSFT"], "paper", auto_execute=False)
    job = _wait(desk, job_id, "awaiting_approval")
    assert {o.ticker for o in job["record"].orders} == {"AAPL", "MSFT"}
    # nothing sent yet
    assert PaperBroker(tmp_path / "paper" / "f.json", cash=0).positions() == {}

    desk.approve(job_id)
    job = _wait(desk, job_id, "done")
    book = PaperBroker(tmp_path / "paper" / "f.json", cash=0)
    assert book.positions()["AAPL"].shares == 25   # 50% of 10k / 200
    assert book.positions()["MSFT"].shares == 12   # 50% of 10k / 400, floored
    assert len(job["results"]) == 2 and all(r["error"] is None for r in job["results"])
    assert list((tmp_path / "mandates").glob("f-run-*.json"))  # receipt saved

    # second cycle: already at target -> no trades
    again = desk.start_cycle("f", ["AAPL", "MSFT"], "paper", auto_execute=True)
    assert _wait(desk, again, "done")["record"].orders == []


def test_reject_sends_nothing(desk, tmp_path):
    job_id = desk.start_cycle("f", ["AAPL"], "paper", auto_execute=False)
    _wait(desk, job_id, "awaiting_approval")
    desk.reject(job_id)
    assert desk.jobs[job_id]["status"] == "rejected"
    with pytest.raises(ValueError):
        desk.approve(job_id)


def test_stale_preview_refused(desk):
    job_id = desk.start_cycle("f", ["AAPL"], "paper", auto_execute=False)
    _wait(desk, job_id, "awaiting_approval")
    desk.manual_order("f", "paper", "MSFT", "buy", 1)
    with pytest.raises(ValueError, match="changed"):
        desk.approve(job_id)


def test_autopilot_runs_and_kill_flattens(desk, tmp_path):
    desk.start_autopilot("f", ["AAPL"], "paper", interval_minutes=60)
    for _ in range(300):
        last = desk.autopilot.get("last_job")
        if last and desk.jobs[last]["status"] == "done":
            break
        time.sleep(0.01)
    assert PaperBroker(tmp_path / "paper" / "f.json", cash=0).positions()["AAPL"].shares == 25

    desk.kill("f", "paper", flatten=True)
    assert desk.autopilot["enabled"] is False
    assert PaperBroker(tmp_path / "paper" / "f.json", cash=0).positions() == {}


def test_unknown_fund_and_broker(desk):
    with pytest.raises(KeyError):
        desk.start_cycle("nope", ["AAPL"], "paper", auto_execute=False)
    with pytest.raises(ValueError):
        desk.start_cycle("f", ["AAPL"], "robinhood", auto_execute=False)


def test_market_hours():
    ny = ZoneInfo("America/New_York")
    assert us_market_open(datetime(2026, 9, 24, 10, 0, tzinfo=ny))
    assert not us_market_open(datetime(2026, 9, 24, 9, 0, tzinfo=ny))
    assert not us_market_open(datetime(2026, 9, 26, 12, 0, tzinfo=ny))  # Saturday


def test_guardrail_cooldown_holds_reentry(desk):
    job_id = desk.start_cycle("f", ["AAPL"], "paper", auto_execute=True)
    _wait(desk, job_id, "done")
    desk.manual_order("f", "paper", "AAPL", "sell", 5)  # a human trims; the agents want back in
    again = desk.start_cycle("f", ["AAPL"], "paper", auto_execute=True)
    job = _wait(desk, again, "done")
    assert job["results"][0]["blocked"].startswith("re-entry cooldown")


def test_drawdown_breaker_stops_autopilot(desk):
    desk.guardrails("f").check_equity(10_000)
    assert desk.guardrails("f").check_equity(8_000).startswith("drawdown breaker")
    desk.start_autopilot("f", ["AAPL"], "paper", interval_minutes=60)
    for _ in range(100):
        if desk.autopilot.get("last_skip"):
            break
        time.sleep(0.01)
    assert "halted" in desk.autopilot["last_skip"]
    assert desk.autopilot.get("last_job") is None
    desk.stop_autopilot()


def test_safe_mode_after_failures(desk):
    g = desk.guardrails("f")
    assert g.cycle_result(ok=False) is None
    assert g.cycle_result(ok=False) is None
    assert g.cycle_result(ok=False).startswith("safe mode")
    g.reset()
    assert desk.guardrails("f").state.tripped is None


def test_leaderboard_ranks_paper_books(desk, tmp_path):
    (tmp_path / "mandates" / "g.yaml").write_text(yaml.safe_dump({
        "name": "g", "capital": 10_000,
        "strategies": [{"name": "solo", "models": [{"name": "bull"}]}],
        "risk": {"max_position_pct": 0.5, "max_gross_exposure": 1.0},
    }))
    desk.manual_order("f", "paper", "AAPL", "buy", 1)
    PaperBroker(tmp_path / "paper" / "g.json", cash=12_000.0)
    board = desk.leaderboard()
    assert [r["fund"] for r in board] == ["g", "f"]
    assert board[0]["return"] == pytest.approx(0.2)
