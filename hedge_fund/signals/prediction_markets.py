"""Prediction-market macro overlay — what crowds betting real money expect.

Reads live odds on macro events (via pmxt: Polymarket, Kalshi, ...), corrects
each price for the longshot bias with a calibration curve fit the way
prediction-market-analysis measures it, and turns the result into one
risk-on / risk-off conviction applied to every ticker:

    conviction = sum(direction * weight * (2p - 1)) / sum(weight)

where p is the calibrated probability of the event and direction says
whether the event is good (+1) or bad (-1) for equities. A 70% recession
market pulls every name toward short; an 80% rate-cut market pushes toward
long. Blended at a modest weight it acts as a regime tilt on the stock
pickers, not a stock picker itself.

Point-in-time: live odds are only known today. For any past as-of date the
model abstains — a backtest never sees tomorrow's crowd.
"""

from __future__ import annotations

from datetime import date as _date
from typing import Any

from hedge_fund.data.protocol import DataClient
from hedge_fund.models import Signal
from hedge_fund.predictions.calibration import Calibration
from hedge_fund.predictions.odds import PmxtOdds
from hedge_fund.signals.base import AlphaModel

DEFAULT_EVENTS = [
    {"query": "US recession", "direction": -1.0, "weight": 1.0},
    {"query": "Fed rate cut", "direction": 1.0, "weight": 1.0},
]


class PredictionMarketModel(AlphaModel):
    def __init__(
        self,
        *,
        events: list[dict] | None = None,
        venue: str = "polymarket",
        odds: Any = None,
        calibration: Calibration | None = None,
    ) -> None:
        self.events = events or DEFAULT_EVENTS
        self.odds = odds or PmxtOdds(venue)
        self.calibration = calibration or Calibration.load()
        self._by_date: dict[str, tuple[float, str, bool]] = {}

    @property
    def name(self) -> str:
        return "prediction_markets"

    def predict(self, ticker: str, date: str, data_client: DataClient) -> Signal:
        if date < _date.today().isoformat():
            return self._signal(ticker, date, 0.0, "Live odds only exist today — abstaining on a past date "
                                "(no lookahead).", abstained=True)
        if date not in self._by_date:  # one odds read per cycle, shared by every ticker
            self._by_date[date] = self._read()
        value, reasoning, abstained = self._by_date[date]
        return self._signal(ticker, date, value, reasoning, abstained)

    def _read(self) -> tuple[float, str, bool]:
        lines, num, den = [], 0.0, 0.0
        for ev in self.events:
            try:
                q = self.odds.quote(ev["query"])
            except Exception as exc:
                lines.append(f"- {ev['query']}: unavailable ({exc})")
                continue
            if q is None:
                lines.append(f"- {ev['query']}: no market found")
                continue
            p = self.calibration(q.price)
            w = float(ev.get("weight", 1.0))
            d = float(ev.get("direction", 1.0))
            num += d * w * (2 * p - 1)
            den += w
            lines.append(f"- {q.title} [{q.venue}]: {q.outcome} {q.price:.0%} market → {p:.0%} calibrated "
                         f"({'risk-on' if d > 0 else 'risk-off'} if yes)")
        if den == 0:
            return 0.0, "No prediction-market odds available:\n" + "\n".join(lines), True
        value = max(-1.0, min(1.0, num / den))
        cal = f"calibration: {self.calibration.source}"
        return value, f"Macro odds → {'risk-on' if value > 0 else 'risk-off'} {value:+.2f} ({cal})\n" + "\n".join(lines), False

    def _signal(self, ticker: str, date: str, value: float, reasoning: str, abstained: bool) -> Signal:
        return Signal(model_name=self.name, ticker=ticker, date=date, value=value, reasoning=reasoning,
                      metadata={"abstained": True} if abstained else {})
