"""AlpacaBroker tests — a fake HTTP session stands in for Alpaca."""

import json

import pytest

from hedge_fund.brokers.alpaca import PAPER_URL, AlpacaBroker, AlpacaError
from hedge_fund.brokers.models import Order


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.content = json.dumps(payload).encode() if payload is not None else b""
        self.text = self.content.decode()

    def json(self):
        return self._payload


class FakeAlpaca:
    """Just enough of Alpaca: positions, account, market orders that fill
    immediately (or with a scripted status)."""

    def __init__(self, positions=None, order_status="filled", fill_price=101.0):
        self.headers = {}
        self.positions = dict(positions or {})
        self.order_status = order_status
        self.fill_price = fill_price
        self.submitted = []
        self.deleted = []

    def request(self, method, url, timeout=None, params=None, json=None):
        path = url.removeprefix(PAPER_URL)
        if method == "GET" and path == "/v2/positions":
            return FakeResponse(200, [{"symbol": t, "qty": str(q)} for t, q in self.positions.items()])
        if method == "GET" and path == "/v2/account":
            return FakeResponse(200, {"cash": "5000.5", "equity": "9000"})
        if method == "POST" and path == "/v2/orders":
            self.submitted.append(json)
            qty = int(json["qty"])
            if self.order_status == "filled":
                sign = 1 if json["side"] == "buy" else -1
                self.positions[json["symbol"]] = self.positions.get(json["symbol"], 0) + sign * qty
                return FakeResponse(200, {"id": f"o{len(self.submitted)}", "status": "filled",
                                          "filled_qty": str(qty), "filled_avg_price": str(self.fill_price)})
            return FakeResponse(200, {"id": "o1", "status": self.order_status, "filled_qty": "0"})
        if method == "GET" and path.startswith("/v2/orders/"):
            return FakeResponse(200, {"id": "o1", "status": self.order_status, "filled_qty": "0"})
        if method == "DELETE":
            self.deleted.append(path)
            return FakeResponse(204, None)
        return FakeResponse(404, {"message": "nope"})


def _broker(fake, **kw):
    return AlpacaBroker("k", "s", session=fake, poll_interval=0, **kw)


def test_requires_keys(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    with pytest.raises(AlpacaError):
        AlpacaBroker()


def test_live_needs_env_opt_in(monkeypatch):
    monkeypatch.delenv("ALPACA_LIVE", raising=False)
    with pytest.raises(AlpacaError):
        AlpacaBroker("k", "s", live=True, session=FakeAlpaca())
    monkeypatch.setenv("ALPACA_LIVE", "1")
    assert AlpacaBroker("k", "s", live=True, session=FakeAlpaca()).base_url.startswith("https://api.")


def test_positions_and_cash():
    b = _broker(FakeAlpaca(positions={"AAPL": 10, "TSLA": -3, "DUST": 0.4}))
    assert {t: p.shares for t, p in b.positions().items()} == {"AAPL": 10, "TSLA": -3}
    assert b.cash() == pytest.approx(5000.5)


def test_market_order_fills():
    fake = FakeAlpaca()
    fill = _broker(fake).place_order(Order(ticker="AAPL", side="buy", quantity=5, price=100.0))
    assert (fill.quantity, fill.price) == (5, 101.0)
    assert fake.submitted[0]["type"] == "market"
    assert fake.submitted[0]["client_order_id"].startswith("aihf-")


def test_flip_through_zero_is_split():
    fake = FakeAlpaca(positions={"AAPL": 10})
    fill = _broker(fake).place_order(Order(ticker="AAPL", side="sell", quantity=15, price=100.0))
    assert [o["qty"] for o in fake.submitted] == ["10", "5"]
    assert fill.quantity == 15
    assert fake.positions["AAPL"] == -5


def test_rejected_order_raises():
    with pytest.raises(AlpacaError, match="rejected"):
        _broker(FakeAlpaca(order_status="rejected")).place_order(
            Order(ticker="AAPL", side="buy", quantity=1, price=1.0))


def test_unfilled_order_times_out_and_cancels():
    fake = FakeAlpaca(order_status="new")
    with pytest.raises(AlpacaError, match="not filled"):
        _broker(fake, fill_timeout=0).place_order(Order(ticker="AAPL", side="buy", quantity=1, price=1.0))
    assert fake.deleted == ["/v2/orders/o1"]


def test_http_error_surfaces_message():
    class Failing(FakeAlpaca):
        def request(self, *a, **k):
            return FakeResponse(403, {"message": "forbidden"})

    with pytest.raises(AlpacaError, match="forbidden"):
        _broker(Failing()).cash()
