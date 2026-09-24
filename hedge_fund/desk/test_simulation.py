"""End-to-end: the whole fund stack through the synthetic market."""

import pytest

from hedge_fund.desk.simulation import simulate


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return simulate(weeks=16, seed=3, crash_week=10, workdir=tmp_path_factory.mktemp("sim"), progress=None)


def test_every_cycle_holds_its_invariants(run):
    assert run["violations"] == []


def test_every_analyst_speaks(run):
    import json
    from pathlib import Path

    receipts = sorted(Path(run["workdir"], "mandates").glob("multi-repo-fund-run-*.json"))
    assert len(receipts) >= 10
    seen = {s["model_name"] for r in receipts for sr in json.loads(r.read_text())["strategies"] for s in sr["signals"]
            if not s["metadata"].get("abstained")}
    assert seen == {"graham", "buffett", "munger", "tradingagents", "pead", "prediction_markets"}


def test_odds_overlay_turns_risk_off_into_the_crash(run):
    views = run["odds_view"]
    assert views[-1] < views[0]


def test_leaderboard_and_kill_drill(run):
    assert {r["fund"] for r in run["leaderboard"]} == {"multi-repo-fund", "example-fund", "tradingagents-only"}
    assert run["desk"].account("paper", "tradingagents-only")["positions"] == []


def test_breaker_trips_and_halts(tmp_path):
    r = simulate(weeks=10, seed=3, crash_week=5, workdir=tmp_path, progress=None, max_drawdown=0.03)
    trips = [e for e in r["events"] if e["detail"].startswith("drawdown breaker")]
    assert trips, r["events"]
    fund = trips[0]["fund"]
    assert r["desk"].guardrails(fund).state.tripped
    assert r["violations"] == []
