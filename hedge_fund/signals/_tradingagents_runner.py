"""Run one TradingAgents decision and print it as one JSON line.

Executed by TRADINGAGENTS_PYTHON (a separate venv), never imported by the
hedge fund — so it may only use the standard library and tradingagents.

    python _tradingagents_runner.py NVDA 2026-09-01
    -> {"rating": "Overweight", "state": {"final_trade_decision": "...", ...}}
"""

import json
import sys

KEYS = ("final_trade_decision", "investment_plan", "trader_investment_plan",
        "fundamentals_report", "sentiment_report", "news_report", "market_report")


def main() -> None:
    sys.path = [p for p in sys.path if not p.rstrip("/").endswith("hedge_fund/signals")]
    ticker, date = sys.argv[1], sys.argv[2]
    real_stdout = sys.stdout
    sys.stdout = sys.stderr  # the graph may print; only our JSON goes to stdout
    try:
        from tradingagents.default_config import DEFAULT_CONFIG
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        state, rating = TradingAgentsGraph(config=DEFAULT_CONFIG.copy()).propagate(ticker, date)
        out = {"rating": str(rating), "state": {k: str(state.get(k) or "") for k in KEYS}}
    except Exception as exc:
        out = {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        sys.stdout = real_stdout
    print(json.dumps(out))


if __name__ == "__main__":
    main()
