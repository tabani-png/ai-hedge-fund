"""Calibration — what a prediction-market price is really worth.

A contract trading at 5c does not win 5% of the time: prediction markets
carry a longshot bias, and Jon-Becker/prediction-market-analysis measures it
directly ("win rate by price": every resolved trade, bucketed by the price
paid, against how often that side actually won). This module ports that
exact measurement and turns it into a correction:

    calibrated_probability = win_rate_at(price)

Fit it from the dataset that repo publishes (Kalshi/Polymarket parquet):

    aihf calibrate ~/prediction-market-analysis/data/kalshi

The fitted table is saved to ~/.hedge-fund/calibration.json and used by the
prediction-market alpha model. With no fit on disk the curve is the identity —
prices are taken at face value rather than corrected by numbers nobody measured.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hedge_fund.paths import USER_DIR

CALIBRATION_PATH = USER_DIR / "calibration.json"
_MIN_TRADES = 50  # a price bucket thinner than this is too noisy to trust

# The query from prediction-market-analysis src/analysis/kalshi/win_rate_by_price.py:
# both sides of every trade on a finalized market, bucketed by the price paid.
KALSHI_WIN_RATE_SQL = """
WITH resolved_markets AS (
    SELECT ticker, result FROM '{markets}/*.parquet'
    WHERE status = 'finalized' AND result IN ('yes', 'no')
),
all_positions AS (
    SELECT CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
           CASE WHEN t.taker_side = m.result THEN 1 ELSE 0 END AS won
    FROM '{trades}/*.parquet' t INNER JOIN resolved_markets m ON t.ticker = m.ticker
    UNION ALL
    SELECT CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END AS price,
           CASE WHEN t.taker_side != m.result THEN 1 ELSE 0 END AS won
    FROM '{trades}/*.parquet' t INNER JOIN resolved_markets m ON t.ticker = m.ticker
)
SELECT price, COUNT(*) AS total_trades, SUM(won) AS wins
FROM all_positions GROUP BY price ORDER BY price
"""


class Calibration:
    """Monotone map from market price (0-1) to realized win probability."""

    def __init__(self, table: dict[int, float] | None = None, source: str = "identity") -> None:
        # table: price in cents (1-99) -> realized win rate (0-1)
        self.table = dict(sorted((table or {}).items()))
        self.source = source

    @property
    def fitted(self) -> bool:
        return bool(self.table)

    def __call__(self, price: float) -> float:
        price = min(max(float(price), 0.0), 1.0)
        if not self.table:
            return price
        xs = np.array([0, *self.table.keys(), 100], dtype=float) / 100.0
        ys = np.array([0.0, *self.table.values(), 1.0], dtype=float)
        ys = np.maximum.accumulate(ys)  # a higher price never means a lower chance
        return float(np.interp(price, xs, ys))

    # -- fitting -------------------------------------------------------

    @classmethod
    def from_win_rates(cls, df: pd.DataFrame, source: str) -> "Calibration":
        """*df* has columns price (cents), total_trades, wins — the shape
        prediction-market-analysis's win-rate query returns."""
        table = {
            int(r.price): float(r.wins) / float(r.total_trades)
            for r in df.itertuples()
            if 1 <= int(r.price) <= 99 and r.total_trades >= _MIN_TRADES
        }
        if not table:
            raise ValueError("no price bucket had enough resolved trades to fit")
        return cls(table, source=source)

    @classmethod
    def from_positions(cls, df: pd.DataFrame, source: str) -> "Calibration":
        """*df* has one row per position: price (cents) and won (0/1)."""
        grouped = df.groupby("price")["won"].agg(total_trades="count", wins="sum").reset_index()
        return cls.from_win_rates(grouped, source)

    @classmethod
    def fit_kalshi_dataset(cls, data_dir: str | Path) -> "Calibration":
        """Run the win-rate query over the dataset's kalshi/{trades,markets}
        parquet (needs `pip install duckdb`, as that repo does)."""
        import duckdb

        data_dir = Path(data_dir).expanduser()
        sql = KALSHI_WIN_RATE_SQL.format(trades=data_dir / "trades", markets=data_dir / "markets")
        return cls.from_win_rates(duckdb.connect().execute(sql).df(), source=f"kalshi dataset {data_dir}")

    # -- persistence ---------------------------------------------------

    def save(self, path: Path = CALIBRATION_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"source": self.source, "table": self.table}, indent=1))
        return path

    @classmethod
    def load(cls, path: Path = CALIBRATION_PATH) -> "Calibration":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        return cls({int(k): float(v) for k, v in data["table"].items()}, source=data.get("source", str(path)))
