"""TradingAgents and prediction-market alpha models — with fakes for the
external systems, and the TradingAgents runner exercised as a real subprocess."""

import sys
import textwrap
from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from hedge_fund.predictions.calibration import Calibration
from hedge_fund.predictions.odds import PmxtOdds
from hedge_fund.signals import ALPHA_MODEL_REGISTRY
from hedge_fund.signals.prediction_markets import PredictionMarketModel
from hedge_fund.signals.trading_agents import SubprocessGraph, TradingAgentsModel

TODAY = date.today().isoformat()


# ---------------------------------------------------------------- TradingAgents

class FakeGraph:
    def __init__(self, rating, fail=False):
        self.rating, self.fail = rating, fail

    def propagate(self, ticker, date):
        if self.fail:
            raise RuntimeError("LLM down")
        return {"final_trade_decision": f"Rating: {self.rating} on {ticker}",
                "investment_debate_state": {"judge_decision": "bulls won"}}, self.rating


@pytest.mark.parametrize("rating,value", [("Buy", 1.0), ("Overweight", 0.5), ("Hold", 0.0),
                                          ("Underweight", -0.5), ("Sell", -1.0)])
def test_rating_maps_to_conviction(rating, value):
    s = TradingAgentsModel(graph_factory=lambda: FakeGraph(rating)).predict("NVDA", TODAY, None)
    assert s.value == value and not s.metadata.get("abstained")
    assert f"Rating: {rating}" in s.reasoning


@pytest.mark.parametrize("graph", [FakeGraph("REVIEW"), FakeGraph("Buy", fail=True)])
def test_unusable_decision_abstains(graph):
    s = TradingAgentsModel(graph_factory=lambda: graph).predict("NVDA", TODAY, None)
    assert s.value == 0.0 and s.metadata["abstained"]


def test_missing_install_abstains():
    def boom():
        raise ImportError("no tradingagents")
    s = TradingAgentsModel(graph_factory=boom).predict("NVDA", TODAY, None)
    assert s.metadata["abstained"] and "TRADINGAGENTS_PYTHON" in s.reasoning


def test_subprocess_bridge_round_trip(tmp_path, monkeypatch):
    """The runner script executed by a real interpreter, against a stub
    tradingagents package on its path."""
    pkg = tmp_path / "stub" / "tradingagents"
    (pkg / "graph").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "graph" / "__init__.py").write_text("")
    (pkg / "default_config.py").write_text("DEFAULT_CONFIG = {}\n")
    (pkg / "graph" / "trading_graph.py").write_text(textwrap.dedent("""
        class TradingAgentsGraph:
            def __init__(self, config): pass
            def propagate(self, ticker, date):
                print("noisy graph output")
                return {"final_trade_decision": "Rating: Underweight"}, "Underweight"
    """))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "stub"))
    state, rating = SubprocessGraph(sys.executable).propagate("AAPL", TODAY)
    assert rating == "Underweight" and "Underweight" in state["final_trade_decision"]


def test_registered():
    assert ALPHA_MODEL_REGISTRY["tradingagents"] is TradingAgentsModel
    assert ALPHA_MODEL_REGISTRY["prediction_markets"] is PredictionMarketModel


# ---------------------------------------------------------------- calibration

def test_identity_until_fitted():
    assert Calibration()(0.37) == pytest.approx(0.37)


def test_fit_corrects_longshot_bias():
    # 5c contracts that win 2% of the time; 95c contracts that win 97%.
    rows = [{"price": 5, "won": int(i < 2)} for i in range(100)] + \
           [{"price": 95, "won": int(i < 97)} for i in range(100)]
    cal = Calibration.from_positions(pd.DataFrame(rows), source="test")
    assert cal(0.05) == pytest.approx(0.02)
    assert cal(0.95) == pytest.approx(0.97)
    assert 0.02 < cal(0.5) < 0.97  # interpolated, monotone


def test_calibration_round_trips(tmp_path):
    cal = Calibration({10: 0.07, 90: 0.93}, source="x")
    loaded = Calibration.load(cal.save(tmp_path / "c.json"))
    assert loaded(0.1) == pytest.approx(0.07) and loaded.source == "x"


# ---------------------------------------------------------------- odds + model

def _market(title, yes, liquidity):
    return SimpleNamespace(title=title, url="u", volume_24h=1.0, liquidity=liquidity,
                           outcomes=[SimpleNamespace(label="Yes", price=yes), SimpleNamespace(label="No", price=1 - yes)])


class FakePmxt:
    def __init__(self, books):
        self.books = books

    def fetch_markets(self, query=None):
        return self.books.get(query, [])


def test_pmxt_picks_most_liquid_yes():
    odds = PmxtOdds(client_factory=lambda: FakePmxt({"US recession": [
        _market("thin", 0.9, 10), _market("deep", 0.3, 5000)]}))
    q = odds.quote("US recession")
    assert (q.title, q.price) == ("deep", 0.3)


def test_macro_odds_conviction():
    odds = PmxtOdds(client_factory=lambda: FakePmxt({
        "US recession": [_market("Recession in 2026?", 0.25, 100)],
        "Fed rate cut": [_market("Fed cuts in December?", 0.80, 100)],
    }))
    m = PredictionMarketModel(odds=odds, calibration=Calibration())
    s = m.predict("AAPL", TODAY, None)
    # (-1 * (0.5 - 1) + 1 * (1.6 - 1)) / 2 = (0.5 + 0.6) / 2
    assert s.value == pytest.approx(0.55)
    assert "Recession in 2026?" in s.reasoning and not s.metadata
    assert m.predict("MSFT", TODAY, None).value == s.value  # one read per cycle


def test_macro_odds_abstains_in_the_past_and_without_markets():
    m = PredictionMarketModel(odds=PmxtOdds(client_factory=lambda: FakePmxt({})), calibration=Calibration())
    past = (date.today() - timedelta(days=1)).isoformat()
    assert m.predict("AAPL", past, None).metadata["abstained"]
    assert m.predict("AAPL", TODAY, None).metadata["abstained"]
