"""Dashboard server tests — real HTTP against a Desk with fake analysts."""

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from hedge_fund.desk.test_desk import desk  # noqa: F401  (fixture)
from hedge_fund.web.server import make_handler


@pytest.fixture
def base(desk):  # noqa: F811
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(desk))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _get(url):
    with urllib.request.urlopen(url) as r:
        return r.status, r.read()


def _post(url, body, header=True):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **({"X-AIHF": "1"} if header else {})})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_page_and_state(base):
    status, html = _get(base + "/")
    assert status == 200 and b"AIHF" in html
    status, raw = _get(base + "/api/state?fund=f&broker=paper")
    state = json.loads(raw)
    assert [f["name"] for f in state["funds"]] == ["f"]
    assert state["brokers"]["paper"]["ready"] is True
    assert state["account"]["equity"] == pytest.approx(10_000)


def test_post_needs_header(base):
    assert _post(base + "/api/cycle", {"fund": "f", "broker": "paper", "tickers": "AAPL"}, header=False)[0] == 403


def test_cycle_approve_over_http(base):
    status, out = _post(base + "/api/cycle", {"fund": "f", "broker": "paper", "tickers": "AAPL"})
    assert status == 200
    for _ in range(200):
        job = json.loads(_get(f"{base}/api/job?id={out['job']}")[1])
        if job["status"] == "awaiting_approval":
            break
        time.sleep(0.01)
    assert job["record"]["orders"][0]["ticker"] == "AAPL"
    assert _post(base + "/api/approve", {"id": out["job"]})[0] == 200


def test_bad_input_is_400(base):
    assert _post(base + "/api/order", {"fund": "f", "broker": "paper", "ticker": "AAPL", "side": "hold", "quantity": 1})[0] == 400
    assert _post(base + "/api/keys", {"name": "PATH", "value": "x"})[0] == 400
