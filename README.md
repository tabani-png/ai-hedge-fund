# AI Hedge Fund

This is a proof of concept for an AI-powered hedge fund. The goal of this project is to explore the use of AI to make trading decisions. This project is for **educational** purposes only and is not intended for real trading or investment.

> **🚧 The project is evolving.** We're rebuilding it into a persistent, always-on AI hedge fund — a *fund* as a first-class entity you can backtest, paper-trade, and (opt-in) run live, with the investor agents reimagined as pluggable, backtestable "alpha models." Read the **[Vision →](VISION.md)** and the **[Roadmap →](ROADMAP.md)**.

Trades only happen when you ask: every run defaults to a simulated broker. The trading desk (below) can send the agents' orders to a persistent local paper book or to an Alpaca account.

[![Twitter Follow](https://img.shields.io/twitter/follow/virattt?style=social)](https://twitter.com/virattt)

## Disclaimer

This project is for **educational and research purposes only**.

- Not intended for real trading or investment
- No investment advice or guarantees provided
- Creator assumes no liability for financial losses
- Consult a financial advisor for investment decisions
- Past performance does not indicate future results

By using this software, you agree to use it solely for learning purposes.

## How to Install

```bash
pipx install aihf
```

(or `uv tool install aihf`, or `pip install aihf` into an environment of your choice)

Then run it from anywhere:

```bash
aihf
```

### API keys

The app asks for keys the first time it needs them and saves them to `~/.hedge-fund/.env` — nothing to configure up front. It needs:

- A [Financial Datasets](https://financialdatasets.ai) API key, for prices, fundamentals, and earnings.
- One model API key for the investor agents. Supported providers: Anthropic, OpenAI, DeepSeek, Google, xAI, Kimi, TypeSafe (Jev).

Keys exported in your shell always win over the saved file.

## How to Run

### Interactive app

```bash
aihf
```

With no arguments, this launches the interactive terminal app. Build a fund — pick stocks, strategies, rebalance cadence — or backtest a saved fund and watch its equity curve draw against its benchmark. Funds you build are saved as mandate files in `~/.hedge-fund/mandates/`.

### Non-interactive

Run one fund cycle from a mandate file. The full cycle record prints to stdout as JSON; a short human summary goes to stderr:

```bash
aihf ~/.hedge-fund/mandates/example.yaml --tickers AAPL,MSFT
```

Run the same mandate with Jev after configuring `TYPESAFE_API_KEY`:

```bash
aihf ~/.hedge-fund/mandates/example.yaml --tickers AAPL,MSFT --model jev-1.13.0
```

Backtest the mandate over history at its rebalance cadence:

```bash
aihf ~/.hedge-fund/mandates/example.yaml --tickers AAPL,MSFT --backtest
```

A mandate is the desk — strategies, staff, risk, capital, cadence — and never names tickers; `--tickers` says what to point it at for this run.

### Trading desk (web UI)

```bash
aihf web --open        # http://127.0.0.1:8765
```

A local dashboard that lets the agents trade:

- **Preview trades**: the agents research your tickers and propose orders. Nothing is sent until you press **Approve & execute**.
- **Run & trade**: research and execute in one step.
- **Autopilot**: runs the fund every N minutes (default: the mandate's rebalance cadence), only while the market is open, and executes automatically or waits for your approval.
- **KILL**: stops autopilot, drops pending proposals, cancels open orders, and can optionally flatten every position.
- Live equity, P&L, positions, trade tape, activity log, a manual order ticket, each agent's conviction and written thesis, and an in-app key manager.
- **Backtest**: replays the fund over any date range at its rebalance cadence (every agent, blend, and risk limit, point-in-time) and compares it with the mandate's benchmark. You get return, excess return, Sharpe, max drawdown, an equity curve against the benchmark, and a table of every rebalance. Click a rebalance to see its orders and each agent's thesis. Results are saved alongside the TUI's backtests.

Brokers:

| Broker | What it is | Setup |
|--------|-----------|-------|
| Local paper | A persistent simulated book in `~/.hedge-fund/paper/` | none |
| Alpaca paper | Real market fills with fake money | `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` (free at alpaca.markets) |
| Alpaca live | **Real money** | the same keys, plus `ALPACA_LIVE=1` when you launch |

### The combined fund: every repo in one book

`~/.hedge-fund/mandates/multi-repo.yaml` (created on first run) runs four pods together:

| Pod | Comes from | What it adds |
|-----|-----------|--------------|
| deep-value | this repo | Graham, Buffett, and Munger persona agents |
| multi-agent-desk | [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) | a whole firm per ticker: analysts, a bull/bear debate, a trader, a risk team, and a 5-tier rating (Buy +1 … Sell -1) |
| earnings-drift | this repo | the PEAD quant model |
| macro-odds | [pmxt](https://github.com/pmxt-dev/pmxt) + [prediction-market-analysis](https://github.com/Jon-Becker/prediction-market-analysis) | live Polymarket/Kalshi odds on recession and rate cuts, corrected for longshot bias, applied as a risk-on/risk-off tilt. Backtests read each market's daily candle on or before the date, so there's no lookahead |

The desk enforces [NoFx](https://github.com/NoFxAiOS/nofx)-style guardrails on top of the mandate's risk limits: re-entry cooldowns, a daily order cap, a drawdown circuit breaker (checked every 5 minutes, and it can flatten the book), and a safe mode after repeated failed cycles. The dashboard shows a leaderboard that ranks funds by return.

Setup for the external pieces (each one is optional; if it's missing, that analyst abstains and the fund keeps running):

```bash
# TradingAgents lives in its own venv (its langchain pins conflict with this package's)
python -m venv ~/.hedge-fund/ta-venv
~/.hedge-fund/ta-venv/bin/pip install git+https://github.com/TauricResearch/TradingAgents
export TRADINGAGENTS_PYTHON=~/.hedge-fund/ta-venv/bin/python   # it uses the fund's LLM by default

# prediction-market odds
pip install pmxt
export PMXT_API_KEY=...            # pmxt.dev/dashboard
export PMXT_WALLET_ADDRESS=0x...

# optional: fit the longshot-bias correction from prediction-market-analysis's dataset
pip install duckdb
aihf calibrate ~/prediction-market-analysis/data/kalshi
```

### Simulate before you trade

```bash
aihf simulate               # backtest 3 funds over a year, then run them forward 52 weeks with a crash in week 40
aihf simulate --serve       # then open the trading desk on the simulated books
aihf simulate --max-drawdown 0.05   # watch the breaker trip and flatten
```

The simulator runs the real stack week by week: every analyst, the desk's autopilot, guardrails, paper books, receipts, and the leaderboard. It checks invariants on every cycle (NAV reconciles, risk caps hold, the receipts match the broker). Only the outside world is synthetic: market data, LLM replies, TradingAgents' verdict, and the odds. No API keys needed. It tests the machinery, not whether the fund makes money.

The CLI can use the same brokers: `aihf mandate.yaml --tickers AAPL,MSFT --broker paper` (or `--broker alpaca`).

## Development

```bash
git clone https://github.com/virattt/ai-hedge-fund.git
cd ai-hedge-fund
poetry install
poetry run aihf
poetry run pytest hedge_fund
```

## How to Contribute

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

**Important**: Please keep your pull requests small and focused. This will make it easier to review and merge.

## Feature Requests

If you have a feature request, please open an [issue](https://github.com/virattt/ai-hedge-fund/issues) and make sure it is tagged with `enhancement`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
