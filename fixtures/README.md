# Fixtures

Sample user-side data for the Valura AI assignment.

You will not find market data, prices, sector classifications, or benchmarks here — the `current_price_usd` values in user fixtures are snapshots for test purposes only. Get live data from MCP servers, the `yfinance` package, or any source you choose. **Do not hardcode live market data into your code.**

The data is global — US, UK, EU, Japan, Singapore. Tickers use proper exchange suffixes (`AAPL`, `ASML.AS`, `HSBA.L`, `7203.T`) so they resolve against any market data provider.

---

## Layout

| Directory | Purpose |
|---|---|
| `users/` | 10 user profiles — 5 with the canonical required schema, 5 scaffold originals |
| `conversations/` | 3 multi-turn test cases for follow-up resolution and topic switching |
| `test_queries/` | Labeled query sets — gold standard for classifier and safety-guard testing |

---

## Users — canonical schema

These five files use the schema your Pydantic model should validate against:
`user_id`, `name`, `kyc_status`, `risk_profile`, `currency`, `portfolio[]` (with `ticker`, `quantity`, `avg_cost_usd`, `current_price_usd`).

| File | user_id | Edge case |
|---|---|---|
| `user_001_aggressive.json` | `usr_aggr_001` | Aggressive trader — 8 tech-heavy positions, all large gains |
| `user_002_concentrated.json` | `usr_conc_002` | ~93% of portfolio value in a single stock (NVDA) |
| `user_003_global.json` | `usr_glob_003` | Singapore-based — USD, EUR, GBP, JPY holdings; `avg_cost_usd` and `current_price_usd` are USD-converted |
| `user_004_empty.json` | `usr_004` | KYC complete, zero positions — agent must not crash |
| `user_005_dividend.json` | `usr_divd_005` | Conservative dividend retiree — 7 income-focused holdings |

FX rates used for `user_003_global.json` conversions: EUR/USD 1.083, GBP/USD 1.274, JPY/USD 0.00667.

## Users — scaffold originals (legacy schema)

These files were provided as part of the assignment scaffold and use a slightly different shape (`positions`, `kyc.status`, `base_currency`). They are referenced by the existing test skeletons via `user_id`.

| File | user_id | Note |
|---|---|---|
| `user_001_active_trader_us.json` | `usr_001` | Referenced by `test_portfolio_health_skeleton.py` |
| `user_003_concentrated.json` | `usr_003` | Referenced by `test_portfolio_health_skeleton.py` |
| `user_006_multi_currency.json` | `usr_006` | Multi-currency scaffold original |
| `user_008_retiree.json` | `usr_008` | Retiree scaffold original |

---

## Conversations

Each file contains a `test_cases[]` array. Every test case provides:
- `prior_user_turns[]` — the conversation history (user turns only) leading up to the current turn
- `current_user_turn` — the query your classifier should classify
- `expected.agent` and `expected.entities` — the gold-standard routing

| File | What it tests |
|---|---|
| `follow_up_session.json` | Pronoun and entity carryover ("how much do I own?" after "tell me about NVDA") |
| `multi_intent_session.json` | Topic switches — context must NOT carry inappropriately |
| `ambiguous_session.json` | Typos, vague references, missing parameters |

---

## Test queries

| File | Count | Format |
|---|---|---|
| `intent_classification.json` | 60 | `{query, expected_agent, expected_entities}` — 8 agents, ~7-8 examples each, includes follow-up queries |
| `safety_pairs.json` | 45 | `{query, should_block, category}` — 27 harmful + 10 educational + 8 benign |

### Agent taxonomy (intent_classification.json)

| Agent | Description |
|---|---|
| `portfolio_health` | Structured assessment of the user's current portfolio |
| `market_research` | Factual or recent info about an instrument, sector, index, or event |
| `investment_strategy` | Action-oriented advice — buy/sell/hold/hedge/rebalance |
| `financial_calculator` | Deterministic numerical computation — DCA, compound interest, FX, tax |
| `risk_assessment` | Risk metrics, exposure analysis, stress tests, what-if scenarios |
| `portfolio_recommendations` | Recommend specific products or funds matching the user's profile |
| `predictive_analysis` | Forward-looking — forecasts, price targets, trend extrapolation |
| `support` | Platform issues, account questions, billing, onboarding |

### Safety categories (safety_pairs.json)

| Category | should_block |
|---|---|
| `insider_trading` | true |
| `market_manipulation` | true |
| `money_laundering` | true |
| `guaranteed_returns` | true |
| `reckless_advice` | true |
| `educational` | **false** — asking how a harmful practice works is allowed |
| `benign` | **false** — normal finance questions |

---

## Matching rules (for grading)

Your classifier output is matched against the gold files using the following rules:

**Agent (`expected_agent`):** exact string match against the taxonomy above.

**Entities (`expected_entities`):** subset match with normalization. Your output must contain every value listed; extra values are allowed.

| Field | Normalization rule |
|---|---|
| `tickers` (array) | **Case-insensitive; exchange suffix optional.** `AAPL` matches `aapl`, `Aapl`, and `AAPL.US`. Strip suffix before comparing: `ASML.AS` → `ASML`. |
| `topics` / `sectors` (arrays) | Case-folded; exact substring match per element. `"ETF"` matches `"etf"` and `"ETFs"`. |
| `amount` (number) | Within **±5%** of the gold value. |
| `rate` (number) | Within **±5%** of the gold value (e.g., 0.07 ± 0.0035). |
| `period_years` (number) | **Exact integer.** |
| `currency` (string) | ISO 4217, exact match (`USD`, `GBP`, `JPY`). |
| `index` (string) | Exact match against canonical names: `S&P 500`, `FTSE 100`, `NIKKEI 225`, `MSCI World`, `Nasdaq`. |
| `action` (string) | Exact match against vocabulary: `buy`, `sell`, `hold`, `hedge`, `rebalance`. |
| `goal` (string) | Exact match against vocabulary: `retirement`, `education`, `house`, `FIRE`, `emergency_fund`. |
| `frequency` (string) | Exact match: `daily`, `weekly`, `monthly`, `yearly`. |
| `horizon` (string) | Exact match against vocabulary: `6_months`, `1_year`, `5_years`, `10_years`. |
| `time_period` (string) | Exact match: `today`, `this_week`, `this_month`, `this_year`. |

These rules are open. Implement them in your `tests/` entity matcher. The grader uses the same rules.

---

## Open vs hidden test sets

These fixtures are **open**. We will run a **separate, larger labeled set** during evaluation. Optimising only against the public set will hurt your score — the hidden set covers the same vocabulary and rules but with novel queries.
