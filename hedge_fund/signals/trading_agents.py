"""TradingAgents as an analyst — a whole trading firm in one seat.

TauricResearch/TradingAgents runs a full desk per ticker: fundamentals,
sentiment, news, and technical analysts; a bull and a bear researcher who
debate; a trader; a risk team; and a portfolio manager who signs off with a
five-tier rating. Here that entire firm becomes ONE alpha model on the fund:
its final rating is a conviction, its final report the thesis. It blends
with Buffett, PEAD, and the prediction-market overlay like any other view.

    Buy +1.0 · Overweight +0.5 · Hold 0 · Underweight -0.5 · Sell -1.0

TradingAgents is an optional install, and its langchain pins conflict with
this package's, so it normally lives in its OWN virtualenv:

    python -m venv ~/.hedge-fund/ta-venv
    ~/.hedge-fund/ta-venv/bin/pip install git+https://github.com/TauricResearch/TradingAgents
    export TRADINGAGENTS_PYTHON=~/.hedge-fund/ta-venv/bin/python

With TRADINGAGENTS_PYTHON set, each ticker runs in that interpreter as a
subprocess (hedge_fund/signals/_tradingagents_runner.py, stdlib-only, JSON
over stdout). Without it, an in-process `import tradingagents` is tried. If it is missing, misconfigured, or returns an unreadable decision
("REVIEW"), this model ABSTAINS — the fund keeps running on its other
analysts rather than trading on noise. Its own LLM and data vendor come from
its TRADINGAGENTS_* environment variables (see its README).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable

from hedge_fund.data.protocol import DataClient
from hedge_fund.models import Signal
from hedge_fund.signals.base import AlphaModel

RATING_CONVICTION = {
    "buy": 1.0,
    "overweight": 0.5,
    "hold": 0.0,
    "underweight": -0.5,
    "sell": -1.0,
}

# The report sections worth quoting as the thesis, most decisive first.
_REPORT_KEYS = (
    "final_trade_decision",
    "investment_plan",
    "trader_investment_plan",
    "fundamentals_report",
    "sentiment_report",
    "news_report",
    "market_report",
)
_THESIS_CHARS = 4000


_RUNNER = Path(__file__).resolve().parent / "_tradingagents_runner.py"
_TIMEOUT_S = 1800  # a full desk debate is many LLM calls


class SubprocessGraph:
    """`propagate` in another interpreter — the venv TradingAgents lives in."""

    def __init__(self, python: str) -> None:
        self.python = os.path.expanduser(python)
        if not Path(self.python).exists():
            raise FileNotFoundError(f"TRADINGAGENTS_PYTHON={python} does not exist")

    def propagate(self, ticker: str, date: str) -> tuple[dict, str]:
        proc = subprocess.run([self.python, str(_RUNNER), ticker, date], env=_desk_env(),
                              capture_output=True, text=True, timeout=_TIMEOUT_S)
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
        if not lines:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
            raise RuntimeError(" / ".join(tail) or f"exit {proc.returncode}")
        out = json.loads(lines[-1])
        if "error" in out:
            raise RuntimeError(out["error"])
        return out["state"], out["rating"]


# Our provider names -> TradingAgents' llm_provider values.
_TA_PROVIDERS = {"Anthropic": "anthropic", "OpenAI": "openai", "Google": "google",
                 "xAI": "xai", "DeepSeek": "deepseek"}


def _desk_env() -> dict[str, str]:
    """Unless TRADINGAGENTS_LLM_PROVIDER is set explicitly, the TradingAgents
    desk reasons with the same model as the rest of the fund — one key, one
    model picker."""
    env = dict(os.environ)
    if "TRADINGAGENTS_LLM_PROVIDER" not in env:
        from hedge_fund.llm import provider_for
        from hedge_fund.llm.client import DEFAULT_MODEL

        model = env.get("HEDGE_FUND_LLM_MODEL") or DEFAULT_MODEL
        provider = _TA_PROVIDERS.get(provider_for(model) or "")
        if provider:
            env["TRADINGAGENTS_LLM_PROVIDER"] = provider
            env.setdefault("TRADINGAGENTS_DEEP_THINK_LLM", model)
            env.setdefault("TRADINGAGENTS_QUICK_THINK_LLM", model)
    return env


def _default_graph() -> Any:
    python = os.environ.get("TRADINGAGENTS_PYTHON")
    if python:
        return SubprocessGraph(python)
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    return TradingAgentsGraph(config=DEFAULT_CONFIG.copy())


class TradingAgentsModel(AlphaModel):
    """Wraps `TradingAgentsGraph.propagate(ticker, date)` as a Signal."""

    def __init__(self, *, graph_factory: Callable[[], Any] | None = None) -> None:
        self._graph_factory = graph_factory or _default_graph
        self._graph: Any = None
        self._setup_error: str | None = None

    @property
    def name(self) -> str:
        return "tradingagents"

    def predict(self, ticker: str, date: str, data_client: DataClient) -> Signal:
        graph = self._ensure_graph()
        if graph is None:
            return self._abstain(ticker, date, self._setup_error or "TradingAgents unavailable")
        try:
            state, rating = graph.propagate(ticker, date)
        except Exception as exc:  # one bad ticker never sinks the cycle
            return self._abstain(ticker, date, f"TradingAgents run failed: {exc}")

        conviction = RATING_CONVICTION.get(str(rating).strip().lower())
        if conviction is None:
            return self._abstain(ticker, date, f"TradingAgents returned no tradeable rating ({rating!r})")
        return Signal(
            model_name=self.name,
            ticker=ticker,
            date=date,
            value=conviction,
            reasoning=f"TradingAgents desk rating: {str(rating).capitalize()}\n\n{_thesis(state)}",
            metadata={"rating": str(rating).capitalize()},
        )

    def _ensure_graph(self) -> Any:
        if self._graph is None and self._setup_error is None:
            try:
                self._graph = self._graph_factory()
            except ImportError:
                self._setup_error = ("TradingAgents is not installed — set TRADINGAGENTS_PYTHON to the "
                                     "interpreter of a venv that has it (see this module's docstring)")
            except Exception as exc:
                self._setup_error = f"TradingAgents setup failed: {exc}"
        return self._graph

    def _abstain(self, ticker: str, date: str, why: str) -> Signal:
        return Signal(model_name=self.name, ticker=ticker, date=date, value=0.0,
                      reasoning=why, metadata={"abstained": True})


def _thesis(state: Any) -> str:
    if not isinstance(state, dict):
        return str(state or "")[:_THESIS_CHARS]
    parts = []
    for key in _REPORT_KEYS:
        text = state.get(key)
        if isinstance(text, dict):  # debate states nest their text
            text = text.get("judge_decision") or text.get("history")
        if text:
            parts.append(f"## {key.replace('_', ' ').title()}\n{str(text).strip()}")
    return "\n\n".join(parts)[:_THESIS_CHARS]
