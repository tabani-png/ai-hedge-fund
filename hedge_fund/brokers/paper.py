"""PaperBroker — a SimBroker whose book survives between runs.

The ledger the roadmap calls "the carried book": positions and cash are
written to a JSON file after every fill, and read back on construction, so a
fund that trades Monday still holds those shares on Tuesday. No keys, no
network — the zero-setup way to run the fund forward in time.

Fills are SimBroker fills (exactly at the order's reference price). Every fill
is appended to the file's `fills` list with a timestamp, so the dashboard can
show a trade tape without reading receipts.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from hedge_fund.brokers.models import Fill, Order
from hedge_fund.brokers.sim import SimBroker

_TAPE_LIMIT = 500  # fills kept in the file; older ones live in receipts


class PaperBroker(SimBroker):
    """SimBroker persisted to *path*. Starts with *cash* only if the file is new."""

    def __init__(self, path: str | Path, cash: float) -> None:
        self.path = Path(path)
        self.fills: list[dict] = []
        super().__init__(cash=cash)
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self._cash = float(data["cash"])
            self._shares = {t: int(s) for t, s in data.get("positions", {}).items() if int(s) != 0}
            self.fills = list(data.get("fills", []))
        else:
            self._save()

    def place_order(self, order: Order) -> Fill:
        fill = super().place_order(order)
        self.fills.append({**fill.model_dump(), "time": datetime.now(timezone.utc).isoformat()})
        self.fills = self.fills[-_TAPE_LIMIT:]
        self._save()
        return fill

    def reset(self, cash: float) -> None:
        """Flatten the book to *cash* and clear the tape."""
        self._cash = cash
        self._shares = {}
        self.fills = []
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "cash": self._cash,
            "positions": self._shares,
            "fills": self.fills,
        }, indent=2))
        tmp.replace(self.path)  # atomic: a crash never leaves half a book
