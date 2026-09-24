"""Full-stack simulation: the whole fund, every repo's piece, a year in seconds.

    aihf simulate                   # 52 weeks, three funds, a crash in week 40
    aihf simulate --weeks 26 --seed 3 --serve    # then open the desk on the result

Everything the fund does runs through the REAL code: the Desk's autopilot
tick, run_cycle, blend + master risk, the NoFx-style guardrails (cooldowns,
order caps, drawdown breaker), the persistent PaperBroker, receipts, the
leaderboard, and every alpha model's own logic — the persona agents build
real fundamentals snapshots and parse real LLM-shaped JSON, PEAD reads real
earnings records, the prediction-market model reads odds through PmxtOdds
and debiases them with a Calibration, TradingAgents is wrapped exactly as in
production.

Only the outside world is simulated, deterministically from --seed:
- a market: per-ticker "quality" that drives both fundamentals and drift,
  quarterly earnings with BEAT/MISS jumps and post-announcement drift, a
  macro recession-risk path that drags the whole market, and an optional
  crash week;
- the LLM behind the persona agents (reads ROE / leverage from the prompt);
- TradingAgents' desk decision (trend + quality, as its analysts would see);
- the Polymarket/Kalshi order book (odds that track the recession path).

So this is a test of the machine, not a claim about alpha: the world is
built so an informed fund CAN win, and the question is whether every part
of the plumbing gets the signal from the analysts to the book intact and
safely. Real performance needs real keys (see README → Trading desk).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import yaml

from hedge_fund.data.models import CompanyFacts, EarningsData, EarningsRecord, FinancialMetrics, Price
from hedge_fund.desk.desk import Desk
from hedge_fund.desk.guardrails import GuardrailLimits
from hedge_fund.fund.spec import Fund, FundSpec
from hedge_fund.llm import PromptCache
from hedge_fund.predictions.calibration import Calibration
from hedge_fund.predictions.odds import PmxtOdds
from hedge_fund.signals import ALPHA_MODEL_REGISTRY, LLMAgent
from hedge_fund.signals.prediction_markets import PredictionMarketModel
from hedge_fund.signals.trading_agents import TradingAgentsModel

_NY = ZoneInfo("America/New_York")
UNIVERSE = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "XOM", "INTC", "F"]
SECTORS = {"AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "GOOGL": "Communication",
           "AMZN": "Consumer", "META": "Communication", "JPM": "Financials", "XOM": "Energy",
           "INTC": "Technology", "F": "Consumer"}


# ---------------------------------------------------------------------------
# The simulated world
# ---------------------------------------------------------------------------

class SimMarket:
    """Deterministic synthetic market implementing the DataClient calls the
    fund makes (prices, TTM metrics, facts, earnings history)."""

    def __init__(self, start: datetime, weeks: int, seed: int, crash_week: int | None) -> None:
        rng = np.random.default_rng(seed)
        self.start = start.date()
        first = self.start - timedelta(days=5 * 365)
        last = self.start + timedelta(weeks=weeks + 1)
        self.days = [d for d in (first + timedelta(n) for n in range((last - first).days + 1)) if d.weekday() < 5]
        n = len(self.days)
        self.quality = {t: q for t, q in zip(UNIVERSE, np.linspace(0.9, -0.9, len(UNIVERSE)))}
        rng.shuffle(order := list(UNIVERSE))
        self.quality = {t: self.quality[UNIVERSE[i]] for i, t in enumerate(order)}

        # Macro: recession probability wanders; high risk drags every stock.
        rec = np.empty(n)
        rec[0] = 0.3
        for i in range(1, n):
            rec[i] = min(0.95, max(0.03, rec[i - 1] + rng.normal(0, 0.01) + 0.002 * (0.3 - rec[i - 1])))
        crash_day = None
        idx = None
        if crash_week is not None:
            crash_day = self.start + timedelta(weeks=crash_week)
            idx = next(i for i, d in enumerate(self.days) if d >= crash_day)
            # The crowd sees it coming: recession odds climb the week before.
            rec[idx - 5:] = np.minimum(0.95, rec[idx - 5:] + np.linspace(0, 0.4, n - idx + 5).clip(max=0.4))
        self.recession = dict(zip(self.days, rec))
        market = 0.0003 - 0.0012 * (rec - 0.3) + rng.normal(0, 0.009, n)
        if idx is not None:
            market[idx:idx + 5] -= 0.06  # a -26% week for the index

        # Earnings: 25 days after each quarter end; BEAT odds rise with quality.
        self.earnings: dict[str, list[EarningsRecord]] = {}
        jumps: dict[str, dict] = {t: {} for t in UNIVERSE}
        for t in UNIVERSE:
            recs = []
            for q_end in _quarter_ends(first, last):
                filed = q_end + timedelta(days=25)
                if filed > last:
                    continue
                beat = rng.random() < 0.5 + 0.3 * self.quality[t]
                recs.append(EarningsRecord(
                    ticker=t, report_period=q_end.isoformat(), source_type="8-K", filing_date=filed.isoformat(),
                    quarterly=EarningsData(eps_surprise="BEAT" if beat else "MISS")))
                jumps[t][filed] = 1 if beat else -1
            self.earnings[t] = sorted(recs, key=lambda r: r.filing_date, reverse=True)

        self.closes: dict[str, dict] = {}
        for t in UNIVERSE:
            q = self.quality[t]
            beta = 1.0 + 0.3 * rng.standard_normal()
            idio = rng.normal(0, 0.014, n)
            drift_left = 0
            direction = 0
            logp = math.log(50 + 250 * rng.random())
            series = {}
            for i, d in enumerate(self.days):
                r = beta * market[i] + idio[i] + 0.0004 * q
                if d in jumps[t]:
                    direction = jumps[t][d]
                    r += 0.03 * direction
                    drift_left = 10
                elif drift_left:
                    r += 0.002 * direction  # post-earnings drift
                    drift_left -= 1
                logp += r
                series[d] = math.exp(logp)
            self.closes[t] = series
        self.closes["SPY"] = {}
        lvl = math.log(500)
        for i, d in enumerate(self.days):
            lvl += market[i]
            self.closes["SPY"][d] = math.exp(lvl)
        self.crash_day = crash_day

    # -- DataClient ------------------------------------------------------

    def get_prices(self, ticker, start_date, end_date, **kwargs):
        s, e = _d(start_date), _d(end_date)
        series = self.closes.get(ticker, {})
        return [Price(open=c, close=c, high=c, low=c, volume=1_000_000, time=f"{d.isoformat()}T00:00:00Z")
                for d, c in series.items() if s <= d <= e]

    def get_financial_metrics(self, ticker, end_date, period="ttm", limit=10):
        if ticker not in self.quality:
            return []
        q = self.quality[ticker]
        e = _d(end_date)
        rows = []
        for q_end in sorted(_quarter_ends(self.days[0], e), reverse=True):
            filed = q_end + timedelta(days=30)
            if filed > e:
                continue
            k = len(rows)
            wobble = 0.01 * math.sin(k + sum(map(ord, ticker)) % 7)
            px = self._close_on(ticker, filed)
            eps = max(0.5, 5 + 4 * q)
            rows.append(FinancialMetrics(
                ticker=ticker, report_period=q_end.isoformat(), period="ttm", filing_date=filed.isoformat(),
                market_cap=px * 1e9, price_to_earnings_ratio=px / eps, return_on_equity=0.13 + 0.12 * q + wobble,
                gross_margin=0.45 + 0.15 * q, operating_margin=0.2 + 0.1 * q, net_margin=0.12 + 0.08 * q + wobble,
                debt_to_equity=max(0.05, 1.0 - 0.8 * q), current_ratio=1.5 + 0.5 * q,
                revenue_growth=0.06 + 0.08 * q, earnings_per_share=eps, book_value_per_share=20 * (1.02 ** -k),
                free_cash_flow_per_share=eps * 0.8))
            if len(rows) >= limit:
                break
        return rows

    def get_company_facts(self, ticker):
        return CompanyFacts(ticker=ticker, name=ticker, sector=SECTORS.get(ticker))

    def get_market_cap(self, ticker, end_date):
        return self._close_on(ticker, _d(end_date)) * 1e9

    def get_earnings_history(self, ticker, limit=8, **kwargs):
        return self.earnings.get(ticker, [])[: max(limit, 20)]

    def _close_on(self, ticker, day):
        series = self.closes[ticker]
        while day not in series and day > self.days[0]:
            day -= timedelta(days=1)
        return series.get(day, next(iter(series.values())))


class ScriptedLLM:
    """Stands in for the persona agents' LLM: reads the fundamentals snapshot
    it is shown and answers in the persona JSON contract."""

    model = "simulated-analyst"

    def complete(self, system: str, user: str) -> str:
        roe = _num(user, r"ROE avg:\s*(-?[\d.]+)")
        de = _num(user, r"Debt/equity \(latest\):\s*(-?[\d.]+)")
        if roe is None:
            return json.dumps({"signal": "neutral", "confidence": 10, "reasoning": "Could not read the numbers."})
        noise = (int(hashlib.sha256((system + user).encode()).hexdigest()[:6], 16) / 0xFFFFFF - 0.5) * 0.3
        score = math.tanh(8 * (roe - 0.13)) - 0.25 * max(0.0, (de or 0) - 1.0) + noise
        signal = "bullish" if score > 0.15 else "bearish" if score < -0.15 else "neutral"
        return json.dumps({
            "signal": signal, "confidence": int(min(95, 30 + 60 * abs(score))),
            "reasoning": f"ROE averages {roe:.0%} with debt/equity {de or 0:.2f} — "
                         f"{'a durable, well-financed franchise' if score > 0 else 'returns do not cover the risk'}.",
        })


class SimTradingAgentsDesk:
    """TradingAgents' final call, as its analysts would reach it here: the
    fundamentals analyst sees quality, the market analyst sees the trend."""

    def __init__(self, market: SimMarket) -> None:
        self.market = market

    def propagate(self, ticker, date):
        closes = [p.close for p in self.market.get_prices(ticker, (_d(date) - timedelta(days=90)).isoformat(), date)]
        trend = closes[-1] / closes[0] - 1 if len(closes) > 1 else 0.0
        score = 0.6 * self.market.quality.get(ticker, 0) + 2.0 * trend
        rating = ("Buy" if score > 0.5 else "Overweight" if score > 0.15 else "Hold" if score > -0.15
                  else "Underweight" if score > -0.5 else "Sell")
        return {"final_trade_decision": f"Rating: {rating}. 90-day trend {trend:+.1%}; "
                                        f"fundamentals analyst quality read {self.market.quality.get(ticker, 0):+.2f}. "
                                        "Bull and bear researchers debated; risk team signed off."}, rating


class SimPmxt:
    """The prediction-market order book, as pmxt would return it."""

    def __init__(self, market: SimMarket, clock) -> None:
        self.market, self.clock = market, clock

    QUERIES = ("US recession", "Fed rate cut")

    def _yes(self, query, day):
        while day not in self.market.recession and day > self.market.days[0]:
            day -= timedelta(days=1)
        rec = float(self.market.recession[day])
        return round({"US recession": rec, "Fed rate cut": min(0.95, 0.35 + 0.3 * (rec - 0.3))}[query], 2)

    def fetch_markets(self, query=None):
        if query not in self.QUERIES:
            return []
        yes = self._yes(query, self.clock().date())
        return [SimpleNamespace(title=f"{query} by year end?", url="sim://", volume_24h=1e6, liquidity=1e6,
                                outcomes=[SimpleNamespace(label="Yes", price=yes, outcome_id=query),
                                          SimpleNamespace(label="No", price=round(1 - yes, 2), outcome_id=query + ":no")])]

    def fetch_ohlcv(self, outcome_id, resolution="1d", start=None, end=None, limit=None):
        """Daily candles stamped at the 20:00 UTC close, only up to `end`."""
        candles = []
        for day in self.market.days:
            stamp = datetime(day.year, day.month, day.day, 20, tzinfo=ZoneInfo("UTC"))
            if (start and stamp < start) or (end and stamp > end):
                continue
            p = self._yes(outcome_id, day)
            candles.append(SimpleNamespace(timestamp=stamp, open=p, high=p, low=p, close=p, volume=1e5))
        return candles


# ---------------------------------------------------------------------------
# Funds under test
# ---------------------------------------------------------------------------

def _funds() -> dict[str, dict]:
    here = Path(__file__).resolve().parent.parent / "fund"
    return {
        "multi-repo-fund": yaml.safe_load((here / "multi-repo.yaml").read_text()),
        "example-fund": yaml.safe_load((here / "example.yaml").read_text()),
        "tradingagents-only": {
            "name": "tradingagents-only", "capital": 100000, "rebalance": "weekly",
            "strategies": [{"name": "multi-agent-desk", "models": [{"name": "tradingagents"}]}],
            "risk": {"max_position_pct": 0.2, "max_gross_exposure": 1.0},
        },
    }


def _fund_factory(market: SimMarket, clock, cache_dir: Path):
    def build(spec: FundSpec) -> Fund:
        models = {}
        for strategy in spec.strategies:
            staff = []
            for m in strategy.models:
                cls = ALPHA_MODEL_REGISTRY[m.name]
                if issubclass(cls, LLMAgent):
                    staff.append(cls(llm=ScriptedLLM(), cache=PromptCache(cache_dir)))
                elif cls is TradingAgentsModel:
                    staff.append(TradingAgentsModel(graph_factory=lambda: SimTradingAgentsDesk(market)))
                elif cls is PredictionMarketModel:
                    staff.append(PredictionMarketModel(
                        odds=PmxtOdds(client_factory=lambda: SimPmxt(market, clock)),
                        # a longshot-biased book: cheap YES wins less than it costs
                        calibration=Calibration({5: 0.03, 10: 0.07, 25: 0.22, 50: 0.5, 75: 0.78, 90: 0.93, 95: 0.97},
                                                source="simulated fit")))
                else:
                    staff.append(cls(**m.params))
            models[strategy.name] = staff
        return Fund(spec, models=models)
    return build


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def simulate(weeks: int = 52, seed: int = 7, crash_week: int | None = 40, workdir: Path | None = None,
             progress=print, max_drawdown: float = 0.20, backtest_weeks: int = 52) -> dict:
    if crash_week is not None and not 0 < crash_week < weeks:
        crash_week = None  # a crash past the end of the run never happens
    workdir = Path(workdir or tempfile.mkdtemp(prefix="aihf-sim-"))
    (workdir / "mandates").mkdir(parents=True, exist_ok=True)
    for name, data in _funds().items():
        (workdir / "mandates" / f"{name}.yaml").write_text(yaml.safe_dump(data, sort_keys=False))

    today = datetime.now(_NY).date()
    monday = today + timedelta(days=(7 - today.weekday()) % 7 or 7)  # sim starts next Monday
    now = {"t": datetime(monday.year, monday.month, monday.day, 10, 0, tzinfo=_NY)}
    clock = lambda: now["t"]  # noqa: E731
    market = SimMarket(now["t"], weeks, seed, crash_week)
    desk = Desk(mandates_dir=workdir / "mandates", paper_dir=workdir / "paper", guard_dir=workdir / "guard",
                data_client_factory=lambda: market, fund_factory=_fund_factory(market, clock, workdir / "llm-cache"),
                clock=clock)
    desk.limits = GuardrailLimits(reentry_cooldown_minutes=240, max_orders_per_day=40, max_drawdown_pct=max_drawdown,
                                  flatten_on_drawdown=True, safe_mode_after_failures=3)
    funds = list(_funds())
    violations: list[str] = []
    pilots = {f: {"enabled": True, "fund": f, "broker": "paper", "universe": UNIVERSE, "interval_minutes": 7 * 24 * 60,
                  "auto_execute": True, "market_hours_only": False, "next_run": 0, "last_job": None,
                  "last_skip": None} for f in funds}

    # 1) Backtest every fund over the year BEFORE the forward run — through the
    #    desk's backtest jobs, the same path the dashboard's Backtest button takes.
    bt_end = (now["t"] - timedelta(days=3)).date().isoformat()
    bt_start = (now["t"] - timedelta(weeks=backtest_weeks)).date().isoformat()
    backtests: dict[str, dict] = {}
    if backtest_weeks:
        for f in funds:
            job = _wait(desk, desk.start_backtest(f, UNIVERSE, bt_start, bt_end), timeout=300)
            if job["status"] != "done":
                violations.append(f"backtest {f}: {job['status']} — {job.get('error')}")
                continue
            result = job["result"]
            for rec in result.records:
                _check_record(rec, f"backtest {f} {rec.as_of}", violations)
            backtests[f] = {**job["metrics"], "dates": result.dates, "nav": result.nav,
                            "benchmark_nav": result.benchmark_nav,
                            "odds_views": [next((s.value for sr in r.strategies for s in sr.signals
                                                 if s.model_name == "prediction_markets" and not s.metadata.get("abstained")), None)
                                           for r in result.records]}
            if progress:
                m = job["metrics"]
                progress(f"  backtest {f:<20} {m['total_return_pct']:+.1%} vs SPY {m['benchmark_return_pct']:+.1%}  "
                         f"sharpe {m['sharpe_ratio']:.2f}  max dd {m['max_drawdown_pct']:.1%}  ({m['n_cycles']} cycles)")

    # 2) Then run them forward on autopilot, week by week.
    curve: dict[str, list[float]] = {f: [] for f in [*funds, "SPY"]}
    net: dict[str, list[float]] = {f: [] for f in funds}
    odds_view: list[float] = []
    dates: list[str] = []
    events: list[dict] = []
    spy0 = market._close_on("SPY", now["t"].date())
    t0 = time.time()
    for week in range(weeks):
        for f in funds:
            ap = pilots[f]
            before = ap.get("last_job")
            desk._autopilot_tick(ap)
            job_id = ap.get("last_job")
            if job_id and job_id != before:
                job = _wait(desk, job_id)
                if job["status"] not in ("done", "partial"):
                    violations.append(f"week {week} {f}: cycle {job['status']} — {job.get('error')}")
                held = [r for r in job.get("results", []) if r.get("blocked")]
                if held:
                    events.append({"week": week, "fund": f, "kind": "guardrail",
                                   "detail": f"{len(held)} orders held: {held[0]['blocked']}"})
                _check_book(desk, f, job, week, violations)
                if f == funds[0]:
                    odds_view.append(next((s.value for sr in job["record"].strategies for s in sr.signals
                                           if s.model_name == "prediction_markets"), 0.0))
            elif ap.get("last_skip"):
                events.append({"week": week, "fund": f, "kind": "skip", "detail": ap["last_skip"]})
        # Between cycles the autopilot's risk watch marks the book daily.
        for day in range(1, 5):
            now["t"] += timedelta(days=1)
            for f in funds:
                if desk._risk_check(pilots[f]) and pilots[f]["last_skip"].startswith("drawdown breaker"):
                    events.append({"week": week, "fund": f, "kind": "skip", "detail": pilots[f]["last_skip"]})
        now["t"] -= timedelta(days=4)
        for f in funds:
            acct = desk.account("paper", f)
            curve[f].append(acct["equity"])
            net[f].append(sum(p["value"] or 0 for p in acct["positions"]) / acct["equity"])
        curve["SPY"].append(100000 * market._close_on("SPY", now["t"].date()) / spy0)
        dates.append(now["t"].date().isoformat())
        if progress and (week % 4 == 3 or week == weeks - 1):
            progress(f"  week {week + 1:>2}/{weeks}  " + "  ".join(
                f"{f} ${curve[f][-1]:,.0f}" for f in [*funds, "SPY"]))
        now["t"] += timedelta(weeks=1)

    # Kill switch drill on a live book: flatten one fund, prove it is flat.
    drill = funds[-1]
    desk.kill(drill, "paper", flatten=True)
    if desk.account("paper", drill)["positions"]:
        violations.append(f"kill switch left positions open in {drill}")

    report = {
        "workdir": str(workdir), "weeks": weeks, "seed": seed, "start": dates[0], "end": dates[-1],
        "crash_week": crash_week, "universe": UNIVERSE, "seconds": round(time.time() - t0, 1),
        "funds": {f: _stats(curve[f]) for f in [*funds, "SPY"]},
        "backtests": backtests, "backtest_window": [bt_start, bt_end],
        "leaderboard": desk.leaderboard(), "events": events, "violations": violations,
        "dates": dates, "curves": curve, "net_exposure": net, "odds_view": odds_view,
    }
    (workdir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    report["desk"] = desk
    return report


def _check_book(desk: Desk, fund: str, job: dict, week: int, violations: list[str]) -> None:
    """Invariants every executed cycle must hold."""
    record = job["record"]
    acct = desk.account("paper", fund)
    nav = acct["cash"] + sum(p["value"] or 0 for p in acct["positions"])
    if abs(nav - acct["equity"]) > 0.01:
        violations.append(f"week {week} {fund}: NAV does not reconcile ({nav} vs {acct['equity']})")
    book = {p["ticker"]: p["shares"] for p in acct["positions"]}
    if book != record.positions:
        violations.append(f"week {week} {fund}: receipt positions differ from the broker's book")
    _check_record(record, f"week {week} {fund}", violations)
    limit = record.spec.risk.max_position_pct
    for p in acct["positions"]:
        if p["value"] is not None and abs(p["value"]) / record.equity_before > limit + 0.02:
            violations.append(f"week {week} {fund}: {p['ticker']} is {abs(p['value']) / record.equity_before:.1%} of equity")


def _check_record(record, where: str, violations: list[str]) -> None:
    """Invariants any cycle record must hold, live or backtested."""
    limit = record.spec.risk.max_position_pct
    for t, w in record.final_weights.items():
        if abs(w) > limit + 1e-9:
            violations.append(f"{where}: {t} target {w:.1%} breaches the {limit:.0%} cap")
    gross = sum(abs(w) for w in record.final_weights.values())
    if gross > record.spec.risk.max_gross_exposure + 1e-9:
        violations.append(f"{where}: gross {gross:.2f} breaches its cap")
    nav = record.cash + sum(s * record.marks[t] for t, s in record.positions.items())
    if abs(nav - record.nav) > 0.01:
        violations.append(f"{where}: NAV {record.nav} does not reconcile to {nav}")
    for sr in record.strategies:
        for sig in sr.signals:
            if sig.date != record.as_of:
                violations.append(f"{where}: {sig.model_name} formed its view for {sig.date}")


def _stats(curve: list[float]) -> dict:
    eq = np.array(curve)
    rets = eq[1:] / eq[:-1] - 1 if len(eq) > 1 else np.array([0.0])
    peak = np.maximum.accumulate(eq)
    sd = rets.std(ddof=1) if len(rets) > 1 else 0.0
    return {"final": float(eq[-1]), "return": float(eq[-1] / 100000 - 1),
            "sharpe": float(rets.mean() / sd * math.sqrt(52)) if sd > 0 else 0.0,
            "max_drawdown": float(((peak - eq) / peak).max()), "weeks": len(eq)}


def _wait(desk: Desk, job_id: str, timeout: float = 60.0) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        job = desk.jobs[job_id]
        if job["status"] not in ("running", "executing"):
            return job
        time.sleep(0.005)
    raise TimeoutError(f"job {job_id} still {desk.jobs[job_id]['status']}")


def _quarter_ends(first, last):
    out = []
    for year in range(first.year - 1, last.year + 1):
        for m, d in ((3, 31), (6, 30), (9, 30), (12, 31)):
            q = datetime(year, m, d).date()
            if first <= q <= last:
                out.append(q)
    return out


def _d(s):
    return datetime.fromisoformat(str(s)[:10]).date()


def _num(text: str, pattern: str) -> float | None:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def print_report(report: dict) -> None:
    print(f"\nSimulated seed {report['seed']}, {report['seconds']}s")
    if report.get("backtests"):
        a, b = report["backtest_window"]
        print(f"\nBACKTEST {a} → {b} (history before the forward run)")
        print(f"{'fund':<22}{'return':>9}{'vs SPY':>9}{'sharpe':>8}{'max dd':>8}{'cycles':>8}")
        for f, m in report["backtests"].items():
            print(f"{f:<22}{m['total_return_pct']:>+9.1%}{m['benchmark_return_pct']:>+9.1%}{m['sharpe_ratio']:>8.2f}"
                  f"{m['max_drawdown_pct']:>8.1%}{m['n_cycles']:>8}")
    print(f"\nFORWARD {report['start']} → {report['end']} on autopilot ({report['weeks']} weeks, "
          f"crash in week {report['crash_week']})")
    print(f"{'fund':<22}{'final':>12}{'return':>9}{'sharpe':>8}{'max dd':>8}")
    for f, s in report["funds"].items():
        print(f"{f:<22}{s['final']:>12,.0f}{s['return']:>+9.1%}{s['sharpe']:>8.2f}{s['max_drawdown']:>8.1%}")
    kinds: dict[str, int] = {}
    for e in report["events"]:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    print(f"\nguardrail holds: {kinds.get('guardrail', 0)}   skipped ticks: {kinds.get('skip', 0)}")
    for e in report["events"]:
        if e["kind"] == "skip" and ("breaker" in e["detail"] or "halted" in e["detail"]):
            print(f"  week {e['week']:>2} {e['fund']}: {e['detail']}")
            break
    print(f"\ninvariant violations: {len(report['violations'])}")
    for v in report["violations"][:20]:
        print(f"  ✗ {v}")
    print(f"\nreceipts, paper books and report.json in {report['workdir']}")
