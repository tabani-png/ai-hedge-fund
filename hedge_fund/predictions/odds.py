"""Live prediction-market odds through pmxt (the ccxt of prediction markets).

    pip install pmxt
    export PMXT_API_KEY=pmxt_live_...        # pmxt.dev/dashboard
    export PMXT_WALLET_ADDRESS=0x...         # hosted reads want an address too

Read-only: the fund uses the odds as a signal and never trades them here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
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
        if self._client is None:
            self._client = self._client_factory()
        markets = self._client.fetch_markets(query=query) or []
        best = None
        for m in markets:
            yes = next((o for o in m.outcomes if str(o.label).lower() in ("yes", "y")), None)
            if yes is None and len(m.outcomes) == 2:
                yes = m.outcomes[0]
            if yes is None or yes.price is None:
                continue
            if best is None or (m.liquidity or 0) > (best[0].liquidity or 0):
                best = (m, yes)
        if best is None:
            return None
        m, yes = best
        return OddsQuote(self.venue, m.title, str(yes.label), float(yes.price),
                         float(m.volume_24h or 0), float(m.liquidity or 0), m.url or "")
