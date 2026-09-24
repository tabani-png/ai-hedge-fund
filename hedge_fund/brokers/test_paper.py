"""PaperBroker tests — the book survives a restart."""

import pytest

from hedge_fund.brokers.models import Order, Position
from hedge_fund.brokers.paper import PaperBroker
from hedge_fund.brokers.sim import SimBroker


def test_book_persists_across_instances(tmp_path):
    path = tmp_path / "fund.json"
    broker = PaperBroker(path, cash=10_000.0)
    broker.place_order(Order(ticker="AAPL", side="buy", quantity=10, price=100.0))

    reopened = PaperBroker(path, cash=999.0)  # starting cash ignored once the file exists
    assert reopened.cash() == pytest.approx(9_000.0)
    assert reopened.positions()["AAPL"].shares == 10
    assert reopened.fills[-1]["ticker"] == "AAPL"


def test_reset_flattens(tmp_path):
    broker = PaperBroker(tmp_path / "fund.json", cash=10_000.0)
    broker.place_order(Order(ticker="AAPL", side="buy", quantity=10, price=100.0))
    broker.reset(5_000.0)
    reopened = PaperBroker(tmp_path / "fund.json", cash=1.0)
    assert reopened.positions() == {}
    assert reopened.cash() == 5_000.0
    assert reopened.fills == []


def test_sim_from_book_copies_state():
    sim = SimBroker.from_book(500.0, {"AAPL": Position(ticker="AAPL", shares=3)})
    sim.place_order(Order(ticker="AAPL", side="sell", quantity=3, price=10.0))
    assert sim.positions() == {}
    assert sim.cash() == pytest.approx(530.0)
