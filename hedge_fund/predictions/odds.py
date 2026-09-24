"""Live prediction-market odds through pmxt (the ccxt of prediction markets).

    pip install pmxt
    export PMXT_API_KEY=pmxt_live_...        # pmxt.dev/dashboard
    export PMXT_WALLET_ADDRESS=0x...         # hosted reads want an address too

Read-only: the fund uses the odds as a signal and never trades them here.

Backtests use `quote_at(query, date)`: the market's daily price candle on or
before that date (pmxt fetch_ohlcv), so a past cycle sees only the odds the
crowd had posted by then. Market *search* is today's catalog — a market that
closed before today can't be found — which is a survivorship caveat worth
knowing: history only reaches as far back as markets still listed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Any, Callable


@dataclass
class OddsQuote:
    venue: str
    title: str
    outcome: str
    price: float        # 0-1, the market-implied probability of `outcome`
    volume_24h: float
    liquidity: float
    url: str


class PmxtOdds:
    """Finds the most liquid market matching a query and quotes its YES side."""

    def __init__(self, venue: str = "polymarket", *, client_factory: Callable[[], Any] | None = None) -> None:
        self.venue = venue
        self._client_factory = client_factory or self._default_client
        self._client: Any = None

    def _default_client(self) -> Any:
        import pmxt

        cls = {"polymarket": pmxt.Polymarket, "kalshi": pmxt.Kalshi}[self.venue]
        kwargs = {}
        if os.environ.get("PMXT_API_KEY"):
            kwargs["pmxt_api_key"] = os.environ["PMXT_API_KEY"]
        if os.environ.get("PMXT_WALLET_ADDRESS"):
            kwargs["wallet_address"] = os.environ["PMXT_WALLET_ADDRESS"]
        return cls(**kwargs)

    def quote(self, query: str) -> OddsQuote | None:
        """The YES price right now."""
        best = self._best(query)
        if best is None:
            return None
        m, yes = best
        return OddsQuote(self.venue, m.title, str(yes.label), float(yes.price),
                         float(m.volume_24h or 0), float(m.liquidity or 0), m.url or "")

    def quote_at(self, query: str, as_of: str) -> OddsQuote | None:
        """The YES price as of the close of *as_of* (YYYY-MM-DD), from daily
        candles — never a later print. None if the market has no candle yet."""
        best = self._best(query)
        if best is None:
            return None
        m, yes = best
        cutoff = datetime.combine(datetime.fromisoformat(as_of).date(), time.max, tzinfo=timezone.utc)
        candles = self._client.fetch_ohlcv(yes.outcome_id, resolution="1d",
                                           start=cutoff - timedelta(days=14), end=cutoff) or []
        past = [c for c in candles if _ts(c.timestamp) <= cutoff]
        if not past:
            return None
        last = max(past, key=lambda c: _ts(c.timestamp))
        return OddsQuote(self.venue, m.title, str(yes.label), float(last.close),
                         float(getattr(last, "volume", 0) or 0), float(m.liquidity or 0), m.url or "")

    def _best(self, query: str):
        """(market, yes_outcome) for the most liquid market matching *query*."""
        if self._client is None:
            self._client = self._client_factory()
        best = None
        for m in self._client.fetch_markets(query=query) or []:
            yes = next((o for o in m.outcomes if str(o.label).lower() in ("yes", "y")), None)
            if yes is None and len(m.outcomes) == 2:
                yes = m.outcomes[0]
            if yes is None or yes.price is None:
                continue
            if best is None or (m.liquidity or 0) > (best[0].liquidity or 0):
                best = (m, yes)
        return best


def _ts(value) -> datetime:
    """pmxt candle timestamps arrive as datetimes or epoch seconds/millis."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    v = float(value)
    return datetime.fromtimestamp(v / 1000 if v > 1e11 else v, tz=timezone.utc)
