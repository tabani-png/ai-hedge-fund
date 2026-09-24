"""The trading desk — what turns a fund cycle into real trades.

run_cycle decides; the desk executes. Every cycle goes the same two steps:

1. PREVIEW  — run_cycle against a SimBroker holding a copy of the real
              broker's book. The analysts argue, risk clamps, and the
              proposed orders fall out — nothing has been sent.
2. EXECUTE  — send exactly those orders to the real broker, one at a time,
              recording each fill or error. The receipt is then rewritten
              with the broker's actual fills, positions, cash, and NAV.

In manual mode a person approves between the steps; on autopilot the desk
runs both on the mandate's clock (or a chosen interval), skipping ticks
while the market is closed. The kill switch stops autopilot, drops pending
approvals, and (optionally) cancels open orders and flattens the book.

Textual-free and HTTP-free: the web dashboard is a thin client over this,
and a CLI or chat control plane could be too.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from hedge_fund.brokers import AlpacaBroker, Broker, Fill, Order, PaperBroker, SimBroker
from hedge_fund.data import CachedDataClient, FDClient
from hedge_fund.desk.guardrails import GuardrailLimits, Guardrails
from hedge_fund.fund import Fund, FundSpec, load_spec, normalize_universe
from hedge_fund.paths import USER_DIR, ensure_mandates_dir
from hedge_fund.pipeline import CycleRecord, run_cycle
from hedge_fund.pipeline.run_cycle import _mark_prices

PAPER_DIR = USER_DIR / "paper"
GUARD_DIR = USER_DIR / "guardrails"
BROKER_KINDS = ("paper", "alpaca", "alpaca-live")
_NY = ZoneInfo("America/New_York")
_CADENCE_MINUTES = {"daily": 24 * 60, "weekly": 7 * 24 * 60, "monthly": 30 * 24 * 60}


def us_market_open(now: datetime | None = None) -> bool:
    """Regular session, Mon-Fri 9:30-16:00 New York. Holidays are not
    modeled here — Alpaca's own clock is used whenever Alpaca is the broker."""
    now = (now or datetime.now(_NY)).astimezone(_NY)
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return 9 * 60 + 30 <= minutes < 16 * 60


class Desk:
    """One process-wide desk. Thread-safe: the web server calls it from
    request threads while jobs and the autopilot run in their own."""

    def __init__(
        self,
        *,
        mandates_dir: Path | None = None,
        paper_dir: Path = PAPER_DIR,
        guard_dir: Path = GUARD_DIR,
        clock: Callable[[], datetime] | None = None,
        data_client_factory: Callable[[], object] | None = None,
        fund_factory: Callable[[FundSpec], Fund] = Fund,
        broker_factory: Callable[[str, FundSpec], Broker] | None = None,
    ) -> None:
        self.mandates_dir = mandates_dir or ensure_mandates_dir()
        self.paper_dir = paper_dir
        self._data_client_factory = data_client_factory or (lambda: CachedDataClient(FDClient()))
        self._fund_factory = fund_factory
        self._broker_factory = broker_factory or self._default_broker
        self._lock = threading.RLock()
        self._funds: dict[str, Fund] = {}  # built once per fund: model caches survive
        self.jobs: dict[str, dict] = {}
        self.log: deque[dict] = deque(maxlen=300)
        self.autopilot: dict = {"enabled": False}
        self._autopilot_stop = threading.Event()
        self._autopilot_thread: threading.Thread | None = None
        self._marks_cache: dict[tuple[str, str], tuple[float, float]] = {}
        self.guard_dir = guard_dir
        # Wall clock by default; the simulator drives a fake one so a year of
        # weekly cycles runs in seconds with cooldowns and dates still honest.
        self.clock = clock or (lambda: datetime.now(_NY))
        self.limits = GuardrailLimits()

    def today(self) -> str:
        return self.clock().date().isoformat()

    def guardrails(self, fund_name: str) -> Guardrails:
        return Guardrails(self.guard_dir / f"{fund_name}.json", self.limits)

    def leaderboard(self) -> list[dict]:
        """Every fund's local paper book ranked by return — NoFx's competition
        board, for mandates: run several side by side, see which desk wins."""
        rows = []
        for name, spec in self.fund_specs().items():
            if not (self.paper_dir / f"{name}.json").exists():
                continue
            try:
                acct = self.account("paper", name)
            except Exception:
                continue
            rows.append({
                "fund": name, "equity": acct["equity"], "start": spec.capital,
                "return": acct["equity"] / spec.capital - 1, "positions": len(acct["positions"]),
                "trades": len(acct["tape"]), "halted": self.guardrails(name).state.tripped,
            })
        return sorted(rows, key=lambda r: r["return"], reverse=True)

    # ------------------------------------------------------------------
    # Funds and brokers
    # ------------------------------------------------------------------

    def fund_specs(self) -> dict[str, FundSpec]:
        specs: dict[str, FundSpec] = {}
        for path in sorted(self.mandates_dir.glob("*.yaml")):
            try:
                spec = load_spec(path)
                specs[spec.name] = spec
            except Exception:
                continue  # a broken mandate should not take the desk down
        return specs

    def spec(self, fund_name: str) -> FundSpec:
        specs = self.fund_specs()
        if fund_name not in specs:
            raise KeyError(f"no mandate named {fund_name!r} in {self.mandates_dir}")
        return specs[fund_name]

    def broker(self, kind: str, spec: FundSpec) -> Broker:
        if kind not in BROKER_KINDS:
            raise ValueError(f"unknown broker {kind!r}; pick one of {', '.join(BROKER_KINDS)}")
        return self._broker_factory(kind, spec)

    def _default_broker(self, kind: str, spec: FundSpec) -> Broker:
        if kind == "paper":
            return PaperBroker(self.paper_dir / f"{spec.name}.json", cash=spec.capital)
        return AlpacaBroker(live=(kind == "alpaca-live"))

    def _fund(self, spec: FundSpec) -> Fund:
        with self._lock:
            cached = self._funds.get(spec.name)
            if cached is None or cached.spec != spec:
                cached = self._fund_factory(spec)
                self._funds[spec.name] = cached
            return cached

    def market_open(self, broker: Broker) -> bool:
        if isinstance(broker, AlpacaBroker):
            return broker.market_open()
        return us_market_open()

    # ------------------------------------------------------------------
    # Account view
    # ------------------------------------------------------------------

    def account(self, kind: str, fund_name: str) -> dict:
        spec = self.spec(fund_name)
        broker = self.broker(kind, spec)
        if isinstance(broker, AlpacaBroker):
            acct = broker.account()
            rows = [{
                "ticker": p["symbol"],
                "shares": float(p["qty"]),
                "price": float(p.get("current_price") or 0),
                "value": float(p.get("market_value") or 0),
                "cost": float(p.get("avg_entry_price") or 0),
                "pnl": float(p.get("unrealized_pl") or 0),
            } for p in broker.position_details()]
            tape = [{
                "time": o.get("filled_at") or o.get("submitted_at"),
                "ticker": o["symbol"], "side": o["side"],
                "quantity": float(o.get("filled_qty") or o.get("qty") or 0),
                "price": float(o["filled_avg_price"]) if o.get("filled_avg_price") else None,
                "status": o["status"],
            } for o in broker.recent_orders(30)]
            return {
                "broker": kind, "fund": fund_name,
                "cash": float(acct["cash"]), "equity": float(acct["equity"]),
                "buying_power": float(acct.get("buying_power") or 0),
                "start": spec.capital, "positions": rows, "tape": tape,
                "market_open": broker.market_open(),
            }

        held = broker.positions()
        marks = self._marks(sorted(held))
        rows = [{
            "ticker": t, "shares": p.shares, "price": marks.get(t),
            "value": p.shares * marks[t] if t in marks else None,
            "cost": None, "pnl": None,
        } for t, p in sorted(held.items())]
        cash = broker.cash()
        equity = cash + sum(r["value"] or 0 for r in rows)
        tape = [{**f, "status": "filled"} for f in reversed(getattr(broker, "fills", [])[-30:])]
        return {
            "broker": kind, "fund": fund_name, "cash": cash, "equity": equity,
            "buying_power": cash, "start": spec.capital, "positions": rows,
            "tape": tape, "market_open": us_market_open(),
        }

    def _marks(self, tickers: list[str]) -> dict[str, float]:
        """Last closes for display, cached five minutes per ticker."""
        now = time.time()
        out: dict[str, float] = {}
        missing = []
        for t in tickers:
            hit = self._marks_cache.get((t, self.today()))
            if hit and now - hit[1] < 300:
                out[t] = hit[0]
            else:
                missing.append(t)
        if missing:
            try:
                client = self._data_client_factory()
                marks, _ = _mark_prices(missing, self.today(), {}, client)
                for t, px in marks.items():
                    self._marks_cache[(t, self.today())] = (px, now)
                    out[t] = px
            except Exception as exc:
                self._note("warn", f"could not price {', '.join(missing)}: {exc}")
        return out

    # ------------------------------------------------------------------
    # Cycles: preview -> (approve) -> execute
    # ------------------------------------------------------------------

    def start_cycle(self, fund_name: str, tickers: list[str], kind: str,
                    *, auto_execute: bool, source: str = "manual") -> str:
        spec = self.spec(fund_name)
        universe = normalize_universe(tickers)
        self.broker(kind, spec)  # fail fast on bad broker config, before a thread
        job_id = uuid.uuid4().hex[:10]
        job = {
            "id": job_id, "fund": fund_name, "broker": kind, "universe": universe,
            "source": source, "auto_execute": auto_execute, "status": "running",
            "started": self.clock().isoformat(timespec="seconds"),
            "record": None, "results": [], "error": None,
        }
        with self._lock:
            self.jobs[job_id] = job
        self._note("info", f"{source}: {fund_name} cycle started on {kind} over {', '.join(universe)}")
        threading.Thread(target=self._run_job, args=(job_id,), daemon=True).start()
        return job_id

    def _run_job(self, job_id: str) -> None:
        job = self.jobs[job_id]
        try:
            spec = self.spec(job["fund"])
            broker = self.broker(job["broker"], spec)
            record = self.preview(spec, broker, job["universe"])
            job["record"] = record
            job["book"] = {t: p.shares for t, p in broker.positions().items()}
            self.guardrails(job["fund"]).cycle_result(ok=True)
            n = len(record.orders)
            if n == 0:
                job["status"] = "done"
                self._note("info", f"{job['fund']}: agents want no trades this cycle")
                self._save_receipt(record)
            elif job["auto_execute"]:
                self._execute_job(job, broker)
            else:
                job["status"] = "awaiting_approval"
                self._note("info", f"{job['fund']}: {n} orders proposed — waiting for approval")
        except Exception as exc:
            job["status"] = "failed"
            job["error"] = str(exc)
            job["trace"] = traceback.format_exc()
            self._note("error", f"{job['fund']} cycle failed: {exc}")
            tripped = self.guardrails(job["fund"]).cycle_result(ok=False)
            if tripped:
                self.stop_autopilot(quiet=True)
                self._note("error", f"{job['fund']}: {tripped} — autopilot stopped")

    def preview(self, spec: FundSpec, broker: Broker, universe: list[str]) -> CycleRecord:
        """run_cycle against a copy of *broker*'s book: real decisions, no orders sent."""
        dry = SimBroker.from_book(broker.cash(), broker.positions())
        client = self._data_client_factory()
        try:
            return run_cycle(self._fund(spec), self.today(), dry, client, universe)
        finally:
            close = getattr(client, "close", None) or getattr(getattr(client, "_client", None), "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def approve(self, job_id: str) -> dict:
        job = self._job(job_id)
        if job["status"] != "awaiting_approval":
            raise ValueError(f"job {job_id} is {job['status']}, not awaiting approval")
        spec = self.spec(job["fund"])
        broker = self.broker(job["broker"], spec)
        now = {t: p.shares for t, p in broker.positions().items()}
        if now != job.get("book"):
            job["status"] = "stale"
            raise ValueError("the book changed since this preview — run the cycle again")
        job["status"] = "executing"
        threading.Thread(target=self._execute_job, args=(job, broker), daemon=True).start()
        return self._job_view(job)

    def reject(self, job_id: str) -> dict:
        job = self._job(job_id)
        if job["status"] == "awaiting_approval":
            job["status"] = "rejected"
            self._note("info", f"{job['fund']}: proposal rejected, nothing sent")
        return self._job_view(job)

    def _execute_job(self, job: dict, broker: Broker) -> None:
        job["status"] = "executing"
        record: CycleRecord = job["record"]
        if isinstance(broker, AlpacaBroker) and not broker.market_open():
            job["status"] = "failed"
            job["error"] = "market is closed — Alpaca market orders would sit until the open"
            self._note("warn", f"{job['fund']}: {job['error']}")
            return
        guard = self.guardrails(job["fund"])
        allowed, blocked = guard.screen(record.orders, self.clock())
        results, fills = self.execute(allowed, broker, guard=guard)
        for order, why in blocked:
            results.append({"order": order.model_dump(), "fill": None, "error": None, "blocked": why})
            self._note("warn", f"guardrail held {order.side} {order.quantity} {order.ticker}: {why}")
        job["results"] = results
        positions = {t: p.shares for t, p in broker.positions().items()}
        cash = broker.cash()
        marks = dict(record.marks)
        nav = cash + sum(s * marks.get(t, 0.0) for t, s in positions.items())
        job["record"] = record.model_copy(update={
            "fills": fills, "positions": positions, "cash": cash, "nav": nav,
        })
        errors = [r for r in results if r["error"]]
        job["status"] = "partial" if errors else "done"
        self._note("error" if errors else "trade",
                   f"{job['fund']}: {len(fills)}/{len(record.orders)} orders filled"
                   + (f", {len(errors)} failed" if errors else "")
                   + (f", {len(blocked)} held by guardrails" if blocked else ""))
        self._save_receipt(job["record"])

    def execute(self, orders: list[Order], broker: Broker,
                guard: Guardrails | None = None) -> tuple[list[dict], list[Fill]]:
        """Send *orders* in order (sells first, as build_orders sorts them).
        One failure never stops the rest: each result says what happened."""
        results: list[dict] = []
        fills: list[Fill] = []
        for order in orders:
            try:
                fill = broker.place_order(order)
                fills.append(fill)
                if guard is not None:
                    guard.record_fill(fill.ticker, self.clock())
                results.append({"order": order.model_dump(), "fill": fill.model_dump(), "error": None})
                self._note("trade", f"{fill.side.upper()} {fill.quantity} {fill.ticker} @ ${fill.price:,.2f}")
            except Exception as exc:
                results.append({"order": order.model_dump(), "fill": None, "error": str(exc)})
                self._note("error", f"{order.side} {order.quantity} {order.ticker} failed: {exc}")
        return results, fills

    def manual_order(self, fund_name: str, kind: str, ticker: str, side: str, quantity: int) -> dict:
        spec = self.spec(fund_name)
        broker = self.broker(kind, spec)
        ticker = ticker.strip().upper()
        price = self._marks([ticker]).get(ticker)
        if price is None:
            if isinstance(broker, AlpacaBroker):
                price = 1.0  # reference only: Alpaca fills at its own quote
            else:
                raise ValueError(f"no recent price for {ticker}")
        if isinstance(broker, AlpacaBroker) and not broker.market_open():
            raise ValueError("market is closed")
        results, _ = self.execute([Order(ticker=ticker, side=side, quantity=quantity, price=price)], broker)
        return results[0]

    # ------------------------------------------------------------------
    # Autopilot and kill switch
    # ------------------------------------------------------------------

    def start_autopilot(self, fund_name: str, tickers: list[str], kind: str, *,
                        interval_minutes: float | None = None, auto_execute: bool = True,
                        market_hours_only: bool = True) -> dict:
        spec = self.spec(fund_name)
        universe = normalize_universe(tickers)
        self.broker(kind, spec)
        self.stop_autopilot(quiet=True)
        interval = float(interval_minutes or _CADENCE_MINUTES[spec.rebalance])
        with self._lock:
            self.autopilot = {
                "enabled": True, "fund": fund_name, "broker": kind, "universe": universe,
                "interval_minutes": interval, "auto_execute": auto_execute,
                "market_hours_only": market_hours_only, "next_run": time.time(),
                "last_job": None, "last_skip": None,
            }
            self._autopilot_stop = threading.Event()
            self._autopilot_thread = threading.Thread(target=self._autopilot_loop,
                                                      args=(self._autopilot_stop,), daemon=True)
            self._autopilot_thread.start()
        self._note("info", f"autopilot ON: {fund_name} on {kind} every {interval:g} min"
                   + ("" if auto_execute else " (proposals need approval)"))
        return self.autopilot

    def stop_autopilot(self, quiet: bool = False) -> None:
        with self._lock:
            was_on = self.autopilot.get("enabled")
            self._autopilot_stop.set()
            self.autopilot = {**self.autopilot, "enabled": False}
        if was_on and not quiet:
            self._note("info", "autopilot OFF")

    RISK_CHECK_SECONDS = 300  # drawdown watch between cycles, as NoFx monitors continuously

    def _autopilot_loop(self, stop: threading.Event) -> None:
        next_risk_check = time.time() + self.RISK_CHECK_SECONDS
        while not stop.is_set():
            ap = self.autopilot
            if time.time() >= next_risk_check:
                next_risk_check = time.time() + self.RISK_CHECK_SECONDS
                try:
                    self._risk_check(ap)
                except Exception as exc:
                    self._note("warn", f"risk check failed: {exc}")
            if time.time() >= ap["next_run"]:
                ap["next_run"] = time.time() + ap["interval_minutes"] * 60
                try:
                    self._autopilot_tick(ap)
                except Exception as exc:
                    self._note("error", f"autopilot tick failed: {exc}")
            stop.wait(1.0)

    def _autopilot_tick(self, ap: dict) -> None:
        last = self.jobs.get(ap.get("last_job") or "")
        if last and last["status"] in ("running", "executing", "awaiting_approval"):
            ap["last_skip"] = "previous cycle still open"
            return
        spec = self.spec(ap["fund"])
        if self._risk_check(ap):
            return
        if ap["market_hours_only"] and not self.market_open(self.broker(ap["broker"], spec)):
            ap["last_skip"] = f"market closed at {datetime.now(_NY):%a %H:%M} ET"
            ap["next_run"] = time.time() + min(ap["interval_minutes"], 15) * 60
            return
        ap["last_skip"] = None
        ap["last_job"] = self.start_cycle(ap["fund"], ap["universe"], ap["broker"],
                                          auto_execute=ap["auto_execute"], source="autopilot")

    def _risk_check(self, ap: dict) -> bool:
        """Mark the book against its peak; trip the breaker (kill, maybe
        flatten) on a deep drawdown. True means trading is halted."""
        guard = self.guardrails(ap["fund"])
        if guard.state.tripped:
            ap["last_skip"] = f"halted — {guard.state.tripped} (reset it in Guardrails)"
            return True
        try:
            equity = self.account(ap["broker"], ap["fund"])["equity"]
        except Exception as exc:
            self._note("warn", f"could not read equity for the drawdown check: {exc}")
            return False
        tripped = guard.check_equity(equity)
        if tripped:
            ap["last_skip"] = tripped
            self.kill(ap["fund"], ap["broker"], flatten=self.limits.flatten_on_drawdown)
            self._note("error", f"{ap['fund']}: {tripped}")
            return True
        return False

    def kill(self, fund_name: str | None, kind: str | None, *, flatten: bool) -> None:
        """Stop everything: autopilot off, pending proposals rejected, and on
        Alpaca open orders canceled. With *flatten*, close every position."""
        self.stop_autopilot()
        for job in list(self.jobs.values()):
            if job["status"] == "awaiting_approval":
                job["status"] = "rejected"
        msg = "KILL SWITCH: autopilot stopped, pending proposals dropped"
        if fund_name and kind:
            spec = self.spec(fund_name)
            broker = self.broker(kind, spec)
            if isinstance(broker, AlpacaBroker):
                broker.cancel_all_orders()
                msg += ", open orders canceled"
                if flatten:
                    broker.close_all_positions()
                    msg += ", all positions closing"
            elif flatten:
                held = broker.positions()
                marks = self._marks(sorted(held))
                orders = [Order(ticker=t, side="sell" if p.shares > 0 else "buy",
                                quantity=abs(p.shares), price=marks[t])
                          for t, p in held.items() if t in marks]
                self.execute(orders, broker)
                msg += f", {len(orders)} positions flattened"
        self._note("error", msg)

    def reset_paper(self, fund_name: str) -> None:
        spec = self.spec(fund_name)
        PaperBroker(self.paper_dir / f"{spec.name}.json", cash=spec.capital).reset(spec.capital)
        self._note("info", f"{fund_name}: paper book reset to ${spec.capital:,.0f}")

    # ------------------------------------------------------------------
    # Views and plumbing
    # ------------------------------------------------------------------

    def job_list(self, limit: int = 20) -> list[dict]:
        jobs = sorted(self.jobs.values(), key=lambda j: j["started"], reverse=True)[:limit]
        return [self._job_view(j, with_record=False) for j in jobs]

    def _job(self, job_id: str) -> dict:
        if job_id not in self.jobs:
            raise KeyError(f"no job {job_id}")
        return self.jobs[job_id]

    def _job_view(self, job: dict, with_record: bool = True) -> dict:
        view = {k: v for k, v in job.items() if k not in ("record", "trace", "book")}
        record: CycleRecord | None = job.get("record")
        if record is not None:
            view["n_orders"] = len(record.orders)
            view["nav"] = record.nav
            if with_record:
                view["record"] = json.loads(record.model_dump_json())
        return view

    def _save_receipt(self, record: CycleRecord) -> None:
        """Same filename shape the TUI writes, so its fund history shows these runs."""
        try:
            self.mandates_dir.mkdir(parents=True, exist_ok=True)
            stamp = self.clock().strftime("%Y-%m-%d-%H%M%S")
            (self.mandates_dir / f"{record.fund}-run-{stamp}.json").write_text(record.model_dump_json(indent=2))
        except OSError as exc:
            self._note("warn", f"could not save receipt: {exc}")

    def _note(self, level: str, message: str) -> None:
        self.log.appendleft({"time": datetime.now().isoformat(timespec="seconds"),
                             "level": level, "message": message})
