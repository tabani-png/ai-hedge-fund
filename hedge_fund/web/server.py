"""The trading dashboard — a local web UI over the Desk.

    aihf web                 # http://127.0.0.1:8765
    aihf web --port 9000 --open

Standard library only (http.server): one JSON API plus one static page, no
build step, no new dependencies. Binds to localhost by default — this page
can place trades, so it is not something to expose to a network.

Every POST must carry the `X-AIHF: 1` header. Browsers will not attach a
custom header to a cross-site request without a CORS preflight this server
never approves, so another website cannot drive the desk from your browser.
"""

from __future__ import annotations

import json
import os
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from hedge_fund.brokers import alpaca_configured
from hedge_fund.desk import BROKER_KINDS, Desk
from hedge_fund.llm import PROVIDER_ENV_VARS, load_api_models
from hedge_fund.llm.client import DEFAULT_MODEL
from hedge_fund.tui.keys import masked, save_credential

STATIC = Path(__file__).resolve().parent / "static"

# Keys the dashboard may write — nothing else reaches .env through it.
SETTABLE_KEYS = ["FINANCIAL_DATASETS_API_KEY", "ALPACA_API_KEY", "ALPACA_SECRET_KEY",
                 *sorted(set(PROVIDER_ENV_VARS.values())),
                 "PMXT_API_KEY", "PMXT_WALLET_ADDRESS", "TRADINGAGENTS_PYTHON"]


def broker_status() -> dict[str, dict]:
    alpaca = alpaca_configured()
    return {
        "paper": {"ready": True, "label": "Local paper", "note": "no keys needed; book saved on disk"},
        "alpaca": {"ready": alpaca, "label": "Alpaca paper",
                   "note": "real market fills, fake money" if alpaca else "add Alpaca keys in Settings"},
        "alpaca-live": {"ready": alpaca and os.environ.get("ALPACA_LIVE") == "1", "label": "Alpaca LIVE",
                        "note": "REAL MONEY — start with ALPACA_LIVE=1 to enable"},
    }


def state(desk: Desk, fund: str | None, broker: str | None) -> dict:
    specs = desk.fund_specs()
    funds = [{
        "name": s.name, "capital": s.capital, "rebalance": s.rebalance, "benchmark": s.benchmark,
        "strategies": [{"name": st.title, "models": [m.name for m in st.models]} for st in s.strategies],
    } for s in specs.values()]
    out = {
        "funds": funds,
        "brokers": broker_status(),
        "keys": {k: (masked(os.environ[k]) if os.environ.get(k) else None) for k in SETTABLE_KEYS},
        "model": os.environ.get("HEDGE_FUND_LLM_MODEL") or DEFAULT_MODEL,
        "models": [{"label": label, "id": mid, "provider": prov} for label, mid, prov in load_api_models()],
        "autopilot": desk.autopilot,
        "jobs": desk.job_list(),
        "log": list(desk.log)[:100],
        "account": None,
        "account_error": None,
        "leaderboard": desk.leaderboard(),
        "guardrails": desk.guardrails(fund).view() if fund in specs else None,
    }
    if fund in specs and broker in BROKER_KINDS and out["brokers"][broker]["ready"]:
        try:
            out["account"] = desk.account(broker, fund)
        except Exception as exc:
            out["account_error"] = str(exc)
    return out


def handle_post(desk: Desk, path: str, body: dict) -> dict:
    if path == "/api/cycle":
        job_id = desk.start_cycle(body["fund"], _tickers(body), body["broker"],
                                  auto_execute=bool(body.get("auto_execute")))
        return {"job": job_id}
    if path == "/api/approve":
        return desk.approve(body["id"])
    if path == "/api/reject":
        return desk.reject(body["id"])
    if path == "/api/autopilot":
        if body.get("action") == "stop":
            desk.stop_autopilot()
            return desk.autopilot
        minutes = body.get("interval_minutes")
        return desk.start_autopilot(
            body["fund"], _tickers(body), body["broker"],
            interval_minutes=float(minutes) if minutes not in (None, "") else None,
            auto_execute=bool(body.get("auto_execute", True)),
            market_hours_only=bool(body.get("market_hours_only", True)),
        )
    if path == "/api/kill":
        desk.kill(body.get("fund"), body.get("broker"), flatten=bool(body.get("flatten")))
        return {"ok": True}
    if path == "/api/order":
        qty = int(body["quantity"])
        if qty <= 0:
            raise ValueError("quantity must be positive")
        if body["side"] not in ("buy", "sell"):
            raise ValueError("side must be buy or sell")
        return desk.manual_order(body["fund"], body["broker"], body["ticker"], body["side"], qty)
    if path == "/api/keys":
        name, value = body["name"], str(body.get("value", "")).strip()
        if name not in SETTABLE_KEYS:
            raise ValueError(f"{name} is not a key this dashboard manages")
        if not value:
            raise ValueError("empty key")
        save_credential(name, value)
        return {"ok": True}
    if path == "/api/model":
        os.environ["HEDGE_FUND_LLM_MODEL"] = body["model"]
        desk._funds.clear()  # rebuild agents with the new model on the next cycle
        return {"ok": True}
    if path == "/api/guardrails":
        if body.get("action") == "reset":
            desk.guardrails(body["fund"]).reset()
            desk._note("info", f"{body['fund']}: guardrails reset — trading allowed again")
        for key in ("reentry_cooldown_minutes", "max_orders_per_day", "max_drawdown_pct",
                    "flatten_on_drawdown", "safe_mode_after_failures"):
            if key in body.get("limits", {}):
                current = getattr(desk.limits, key)
                setattr(desk.limits, key, type(current)(body["limits"][key]))
        return desk.guardrails(body["fund"]).view()
    if path == "/api/reset-paper":
        desk.reset_paper(body["fund"])
        return {"ok": True}
    raise KeyError(path)


def _tickers(body: dict) -> list[str]:
    raw = body.get("tickers", "")
    if isinstance(raw, list):
        return raw
    return raw.replace(",", " ").split()


def make_handler(desk: Desk):
    class Handler(BaseHTTPRequestHandler):
        server_version = "aihf"

        def log_message(self, fmt, *args):  # keep the terminal quiet
            pass

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path in ("/", "/index.html"):
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if url.path == "/api/state":
                return self._json(200, state(desk, q.get("fund"), q.get("broker")))
            if url.path == "/api/job":
                try:
                    return self._json(200, desk._job_view(desk._job(q.get("id", ""))))
                except KeyError as exc:
                    return self._json(404, {"error": str(exc)})
            return self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.headers.get("X-AIHF") != "1":
                return self._json(403, {"error": "missing X-AIHF header"})
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                return self._json(200, handle_post(desk, urlparse(self.path).path, body))
            except KeyError as exc:
                return self._json(400, {"error": f"missing or unknown: {exc}"})
            except (ValueError, TypeError) as exc:
                return self._json(400, {"error": str(exc)})
            except Exception as exc:
                traceback.print_exc()
                return self._json(500, {"error": str(exc)})

        def _json(self, code: int, payload) -> None:
            self._send(code, json.dumps(payload, default=str).encode(), "application/json")

        def _send(self, code: int, data: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False,
          desk: Desk | None = None) -> None:
    desk = desk or Desk()
    httpd = ThreadingHTTPServer((host, port), make_handler(desk))
    url = f"http://{host}:{port}"
    print(f"AI Hedge Fund trading desk → {url}   (ctrl+c to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        desk.stop_autopilot(quiet=True)
        httpd.server_close()
