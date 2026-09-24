"""AlpacaBroker — real order execution through Alpaca's trading API.

Paper trading by default (https://paper-api.alpaca.markets — free, fake
money, real market fills). Live trading is opt-in twice: pass `live=True`
AND set ALPACA_LIVE=1 in the environment. One flag alone is refused, so a
stray argument or a stray export can never move real money on its own.

Keys come from ALPACA_API_KEY / ALPACA_SECRET_KEY (create them in the Alpaca
dashboard under "API Keys").

Contract (see protocol.py): place_order fills COMPLETELY or raises. Market
orders are submitted and polled until filled; anything else — rejected,
canceled, or still open when the timeout lands — cancels the remainder and
raises. Positions are always read back from Alpaca, so the fund's book is
the broker's book even after a raise.

Plain `requests`, no SDK: the four endpoints this needs don't justify a
dependency.
"""

from __future__ import annotations

import os
import time
import uuid

import requests

from hedge_fund.brokers.models import Fill, Order, Position

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"

_TERMINAL_BAD = {"canceled", "expired", "rejected", "suspended", "stopped", "done_for_day"}


class AlpacaError(RuntimeError):
    """An Alpaca request failed or an order did not fill completely."""


def alpaca_configured() -> bool:
    return bool(os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY"))


class AlpacaBroker:
    """Broker backed by an Alpaca account (paper unless explicitly live)."""

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        *,
        live: bool = False,
        fill_timeout: float = 30.0,
        poll_interval: float = 0.5,
        session: requests.Session | None = None,
    ) -> None:
        api_key = api_key or os.environ.get("ALPACA_API_KEY")
        secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            raise AlpacaError("set ALPACA_API_KEY and ALPACA_SECRET_KEY to trade through Alpaca")
        if live and os.environ.get("ALPACA_LIVE") != "1":
            raise AlpacaError("live trading needs ALPACA_LIVE=1 in the environment as well as live=True")
        self.live = live
        self.base_url = LIVE_URL if live else PAPER_URL
        self.fill_timeout = fill_timeout
        self.poll_interval = poll_interval
        self._session = session or requests.Session()
        self._session.headers.update({
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        })

    # ------------------------------------------------------------------
    # Broker protocol
    # ------------------------------------------------------------------

    def positions(self) -> dict[str, Position]:
        out: dict[str, Position] = {}
        for p in self._request("GET", "/v2/positions"):
            # The engine trades whole shares; fractional dust (from manual
            # trades in the Alpaca UI) truncates toward zero.
            shares = int(float(p["qty"]))
            if shares != 0:
                out[p["symbol"]] = Position(ticker=p["symbol"], shares=shares)
        return out

    def cash(self) -> float:
        return float(self.account()["cash"])

    def place_order(self, order: Order) -> Fill:
        """Fill *order* completely at market, or raise.

        An order that crosses zero (long -> short or back) is split into a
        close and an open: Alpaca rejects a single order that flips a
        position. The returned Fill carries the combined quantity at the
        volume-weighted price.
        """
        held = self.positions().get(order.ticker)
        current = held.shares if held else 0
        signed = order.quantity if order.side == "buy" else -order.quantity
        legs = [order.quantity]
        if current != 0 and (current > 0) != (current + signed > 0) and current + signed != 0:
            legs = [abs(current), order.quantity - abs(current)]

        filled_qty = 0
        notional = 0.0
        for qty in legs:
            q, px = self._submit_and_wait(order.ticker, order.side, qty)
            filled_qty += q
            notional += q * px
        return Fill(ticker=order.ticker, side=order.side, quantity=filled_qty, price=notional / filled_qty)

    # ------------------------------------------------------------------
    # Extras the dashboard uses
    # ------------------------------------------------------------------

    def account(self) -> dict:
        return self._request("GET", "/v2/account")

    def clock(self) -> dict:
        return self._request("GET", "/v2/clock")

    def market_open(self) -> bool:
        return bool(self.clock().get("is_open"))

    def position_details(self) -> list[dict]:
        return self._request("GET", "/v2/positions")

    def recent_orders(self, limit: int = 50) -> list[dict]:
        return self._request("GET", "/v2/orders", params={"status": "all", "limit": limit, "direction": "desc"})

    def cancel_all_orders(self) -> None:
        self._request("DELETE", "/v2/orders")

    def close_all_positions(self) -> None:
        self._request("DELETE", "/v2/positions", params={"cancel_orders": "true"})

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _submit_and_wait(self, ticker: str, side: str, qty: int) -> tuple[int, float]:
        submitted = self._request("POST", "/v2/orders", json={
            "symbol": ticker,
            "qty": str(qty),
            "side": side,
            "type": "market",
            "time_in_force": "day",
            "client_order_id": f"aihf-{uuid.uuid4().hex[:20]}",
        })
        order_id = submitted["id"]
        deadline = time.monotonic() + self.fill_timeout
        state = submitted
        while True:
            status = state.get("status")
            if status == "filled":
                return int(float(state["filled_qty"])), float(state["filled_avg_price"])
            if status in _TERMINAL_BAD:
                raise AlpacaError(f"{side} {qty} {ticker}: order {status}")
            if time.monotonic() >= deadline:
                try:
                    self._request("DELETE", f"/v2/orders/{order_id}")
                except AlpacaError:
                    pass  # may have filled in the meantime; the raise below still stands
                raise AlpacaError(
                    f"{side} {qty} {ticker}: not filled within {self.fill_timeout:.0f}s "
                    f"(status {status}, filled {state.get('filled_qty', 0)}) — remainder canceled"
                )
            time.sleep(self.poll_interval)
            state = self._request("GET", f"/v2/orders/{order_id}")

    def _request(self, method: str, path: str, **kwargs):
        try:
            resp = self._session.request(method, self.base_url + path, timeout=15, **kwargs)
        except requests.RequestException as exc:
            raise AlpacaError(f"{method} {path}: {exc}") from exc
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("message", resp.text)
            except ValueError:
                detail = resp.text
            raise AlpacaError(f"{method} {path}: HTTP {resp.status_code} — {detail}")
        if resp.status_code == 204 or not resp.content:
            return {}
        return resp.json()
