# Executive Summary

- We are designing an **offline, fully reproducible NautilusTrader research/backtesting codebase** for Binance (Spot + USD‑M futures) using **only public `binance.vision` files** (no API keys, no live feeds, no authenticated providers).
- The codebase will be implemented in **four phases** (this document specifies Phases 2–4), with strict boundaries: **ETL/catalog build → feature/signal research → strategy prototyping → experiment runner & reporting**.
- Historical data will be parsed from daily/monthly CSVs (e.g., `aggTrades`, `bookTicker`, `klines_1m`, `metrics`, `fundingRate`, `bookDepth`) into **canonical NautilusTrader data objects** (`TradeTick`, `QuoteTick`, `Bar`) plus **custom `Data` classes** for non-standard streams (e.g., OI metrics, bookDepth percent-depth).
- The architecture explicitly prevents the major failure modes seen in “faulty” ingestion code (e.g., **precision loss from rounding**, **timestamp misalignment**, **mixing clocks**, **incorrect routing of custom data**, **non-deterministic ordering**).
- The primary research target is a liquidation-cascade family of signals (e.g., **LPI / forced-flow pressure**, **liquidity depletion**, **exhaustion detection**) consistent with the provided LPI proposal and methodology guidelines; the system is designed so that alternative signals can be plugged in without redesign.
- Backtesting will be **event-driven** in NautilusTrader, with deterministic ingestion, deterministic event ordering, and explicit execution realism hooks (latency/queue/fees) implemented as controlled “model toggles” in later phases.
- Phase outputs are materialized as **versioned artifacts**: a Parquet data catalog + manifests (Phase 2), feature/label datasets + strategy modules (Phase 3), and walk-forward experiment outputs + reports (Phase 4).

---

# Goals and Non-Goals

## Goals

1. **Offline-only pipeline**: consume *only* `binance.vision` market data already downloaded to disk.
2. **Use NautilusTrader Python interfaces only** for all phases (ETL, data models, strategy, backtest runner).
3. **Reproducibility**:
   - deterministic parsing,
   - deterministic event ordering,
   - deterministic experiment configs,
   - stable artifacts (catalog, manifests, reports).
4. **Multi-phase structure** that supports:
   - single-instrument prototyping (default),
   - later expansion to multi-instrument universes (futures + spot).
5. **Microstructure-correct canonicalization**:
   - preserve numerical precision,
   - correct timestamp semantics,
   - correct aggressor-side inference,
   - avoid look-ahead traps for bars.

## Non-Goals (Explicit Exclusions)

- **No API keys**. No authenticated endpoints. No CCXT. No REST/WS fetching.
- **No Binance live connectors** (no live adapters, no live TradingNode connectivity).
- **No external paid data** (Databento, Tardis, etc.) even if Nautilus supports them.
- **No UI/GUI** (no dashboards, no web apps).
- **No “automatic exchangeInfo discovery”** unless the required metadata is also available offline from `binance.vision` (assume it is not; instrument specs will be configured).
- **No new strategy features beyond the research target** (liquidation pressure / volatility-regime alpha) unless explicitly added in later phases.

---

# Data Source and File Layout (binance.vision)

This codebase supports **two universes** with **two temporal granularities** (daily + monthly) and multiple endpoints per universe. The folder roots and patterns mirror the provided inventory.

## Supported on-disk layout patterns

### Futures (USD‑M / UM) — canonical root example
```
data/raw/futures/
  daily/
    aggTrades/      BTCUSDT-aggTrades-YYYY-MM-DD.csv
    bookTicker/     BTCUSDT-bookTicker-YYYY-MM-DD.csv
    bookDepth/      BTCUSDT-bookDepth-YYYY-MM-DD.csv        # nonstandard format (percent-depth)
    metrics/        BTCUSDT-metrics-YYYY-MM-DD.csv          # open interest snapshots (e.g., 5-min)
    klines_1m/       BTCUSDT-1m-YYYY-MM-DD.csv              # 1m OHLCV
  monthly/
    fundingRate/    BTCUSDT-fundingRate-YYYY-MM.csv
    aggTrades/      BTCUSDT-aggTrades-YYYY-MM.csv
    bookTicker/     BTCUSDT-bookTicker-YYYY-MM.csv
    klines/         BTCUSDT-1m-YYYY-MM.csv                  # if present
    ... (indexPriceKlines, markPriceKlines, premiumIndexKlines as available)
```

### Spot — canonical root example
```
spot_data/
  daily/
    aggTrades/      SYMBOL-aggTrades-YYYY-MM-DD.csv
    klines/         SYMBOL-1m-YYYY-MM-DD.csv
    trades/         SYMBOL-trades-YYYY-MM-DD.csv
  monthly/
    aggTrades/      SYMBOL-aggTrades-YYYY-MM.csv
    klines/         SYMBOL-1m-YYYY-MM.csv
    trades/         SYMBOL-trades-YYYY-MM.csv
```

> **Design default:** Phase 2 will ingest **daily futures** first (because the LPI proposal is futures-centric and the sample structure is complete), with spot ingestion implemented as the same framework + different parsers.

## Dataset → columns → canonical internal schema (mapping table)

### Futures mappings

| Dataset (binance.vision) | Example file pattern | Raw columns (observed) | Canonical internal schema |
|---|---|---|---|
| `aggTrades` | `daily/aggTrades/BTCUSDT-aggTrades-YYYY-MM-DD.csv` | `agg_trade_id, price, quantity, first_trade_id, last_trade_id, transact_time, is_buyer_maker` (names may vary case) | `nautilus_trader.model.data.TradeTick` with: `instrument_id`, `price`, `size`, `aggressor_side`, `trade_id`, `ts_event`, `ts_init` |
| `trades` | `daily/trades/BTCUSDT-trades-YYYY-MM-DD.csv` | `id, price, qty, quote_qty, time, is_buyer_maker` | `TradeTick` (same as above) |
| `bookTicker` | `daily/bookTicker/BTCUSDT-bookTicker-YYYY-MM-DD.csv` | `update_id, best_bid_price, best_bid_qty, best_ask_price, best_ask_qty, transaction_time, event_time` | `nautilus_trader.model.data.QuoteTick` with: `instrument_id`, `bid_price`, `ask_price`, `bid_size`, `ask_size`, `ts_event`, `ts_init` |
| `klines` (futures traded price) | `daily/klines/BTCUSDT-1m-YYYY-MM-DD.csv` | standard 1m kline schema | `nautilus_trader.model.data.Bar` (`BarType` = `...-1-MINUTE-LAST-EXTERNAL`) |
| `indexPriceKlines` | `daily/indexPriceKlines/...` | 1m schema (price, but volume=0) | `Bar` with a distinct `BarType` (e.g., `...-1-MINUTE-LAST-EXTERNAL`) but **separate namespace** (see Canonical Data Model) |
| `markPriceKlines` | `daily/markPriceKlines/...` | 1m schema | `Bar` (separate `BarType`) |
| `premiumIndexKlines` | `daily/premiumIndexKlines/...` | 1m schema (premium values) | `Bar` (separate `BarType` or custom `Data`, depending on how Nautilus expects non-price bars) **TO BE CONFIRMED FROM NAUTILUS CONTEXT** |
| `metrics` (OI snapshots) | `daily/metrics/BTCUSDT-metrics-YYYY-MM-DD.csv` | `create_time, symbol, sum_open_interest, sum_open_interest_value, ... ratios ...` | Custom `Data` (e.g., `BinanceOiMetrics`) persisted in catalog |
| `fundingRate` | `monthly/fundingRate/BTCUSDT-fundingRate-YYYY-MM.csv` | `calc_time, funding_interval_hours, last_funding_rate` | Custom `Data` (e.g., `BinanceFundingRate`) persisted in catalog |
| `bookDepth` (percent-depth) | `daily/bookDepth/BTCUSDT-bookDepth-YYYY-MM-DD.csv` | `timestamp, percentage, depth, notional` | Custom `Data` (e.g., `BinanceBookDepthPct`) persisted in catalog |

### Spot mappings

| Dataset (binance.vision) | Example file pattern | Raw columns (observed) | Canonical internal schema |
|---|---|---|---|
| `aggTrades` | `daily/aggTrades/BTCUSDT-aggTrades-YYYY-MM-DD.csv` | positional (no header) common: `aggTradeId, price, quantity, firstTradeId, lastTradeId, timestamp, isBuyerMaker, isBestMatch` | `TradeTick` |
| `trades` | `daily/trades/BTCUSDT-trades-YYYY-MM-DD.csv` | `tradeId, price, qty, quoteQty, time, isBuyerMaker, isBestMatch` | `TradeTick` |
| `klines` | `daily/klines/BTCUSDT-1m-YYYY-MM-DD.csv` | standard kline columns | `Bar` |

---

# Canonical Data Model

This section defines the **internal “truth”** used across phases regardless of source file quirks.

## Time and timezone

- **Canonical timestamp unit:** nanoseconds since Unix epoch (`ts_event`, `ts_init` are integers in ns).
- **Canonical timezone:** UTC.
- **Source conversions:**
  - Binance timestamps in these CSVs are typically **milliseconds** (e.g., `1711756800000`). Convert via:  
    \[
    t_{ns} = 10^6 \cdot t_{ms}
    \]
  - Some fields (e.g., `metrics.create_time`) may be ISO strings (`2024-03-30 00:05:00`). Convert to UTC nanoseconds via a strict parser.

## Instrument identifiers

- Canonical instrument key: `InstrumentId.from_str(f"{SYMBOL}.{VENUE}")` (as used in Nautilus examples).
- **Venue naming convention (default):**
  - `BINANCE` for futures UM and/or spot if you want a single venue name.
  - If splitting venues is desired (spot vs futures), use distinct venues (e.g., `BINANCE_SPOT`, `BINANCE_FUTURES`) **only if you explicitly want cross-venue basis research**. Default is to keep one venue per backtest run to avoid accidental cross-market netting assumptions.

> **Spec default:** For LPI research on USD‑M perps, treat data as **one venue** and one instrument (e.g., `BTCUSDT.BINANCE`). If later we add spot/futures basis, we will separate venues.

## Price/size precision rules (critical)

### Non-negotiable: no lossy rounding during ingestion
- The faulty ingestion approach rounds prices to `:.2f` and sizes to `:.3f`. This is **not acceptable** because:
  - it destroys microstructure signals,
  - it can create artificial zero returns,
  - it can corrupt tick-size logic and execution simulation.

**Canonical rule:**
- Parse numeric strings from CSV **as strings**, and pass them through instrument constructors (`instrument.make_price`, `instrument.make_qty`) or `Price.from_str` / `Quantity.from_str` *without reformatting*.

### Instrument specification source
Binance `binance.vision` files do not reliably carry tick/lot metadata. Therefore:

- Instrument specs (tick size, step size, min qty, price precision, size precision, fees) will be provided via a **versioned config file** (Phase 2 artifact), not inferred ad hoc.
- If the backtest requires an instrument object (e.g., `CryptoPerpetual`), the instrument must be constructed from this config.

**TO BE CONFIRMED FROM NAUTILUS CONTEXT:** exact instrument class choice:
- The provided code examples reference `CryptoPerpetual`; if this is available and appropriate, we use it.
- Otherwise, define an adapter boundary: `InstrumentFactory.create_perp(instrument_spec) -> Instrument`.

## Event types and required vs optional streams

### Required (for liquidation-pressure research)
- `TradeTick` from futures `aggTrades` **or** `trades` (prefer `aggTrades` for compressed noise reduction).
- `QuoteTick` from futures `bookTicker` (best bid/ask).
- `OI metrics` custom `Data` from futures `metrics` (slow regime prior).
- Optional but recommended: `bookDepth` percent-depth custom `Data` if you intend to use depth-within-X% features.

### Optional (context/regime only)
- `Bar` from futures `klines_1m` (trade price bars).
- `fundingRate` custom `Data` from monthly funding files.
- `markPriceKlines`, `indexPriceKlines`, `premiumIndexKlines` as additional context.

---

# System Architecture (Phases 2–4)

## Component architecture (ETL → features → strategy → backtest)

```text
+----------------------------------------------------------------------------------+
|                                  PHASED ARCHITECTURE                             |
+----------------------------------------------------------------------------------+
|                                                                                  |
|  Phase 2: Offline ETL + Catalog Build                                             |
|   [binance.vision CSVs] -> [Parsers] -> [Canonical Events] -> [ParquetDataCatalog]|
|            |                    |                 |                 |            |
|            |                    |                 |                 +--> manifest |
|            |                    |                 +--> validation logs            |
|            |                    +--> schema/version checks                        |
|            +--> file inventory snapshot                                            |
|                                                                                  |
|  Phase 3: Research Layer (Features/Signals + Strategy Prototypes)                 |
|   [Catalog Loader] -> [Feature Pipeline] -> [Signal Objects] -> [Strategy Module] |
|                     (LPI_fast, EXH, LEV)                         |               |
|                                                                  v               |
|                                                        [Nautilus BacktestEngine] |
|                                                                                  |
|  Phase 4: Experiment Runner (Walk-forward + Reporting)                           |
|   [Experiment Configs] -> [Backtest Runs] -> [Metrics + Reports + Artifacts]      |
|                                                                                  |
+----------------------------------------------------------------------------------+
```

## Dataflow diagram (files → parsing → normalization → storage → simulation)

```text
+----------------------------------------------------------------------------------+
|                                   DATAFLOW                                       |
+----------------------------------------------------------------------------------+
|                                                                                  |
|  (A) Raw Files on Disk                                                           |
|      data/raw/futures/daily/{aggTrades,bookTicker,metrics,bookDepth,klines_1m}    |
|      data/raw/futures/monthly/fundingRate                                        |
|                                                                                  |
|          |                                                                       |
|          v                                                                       |
|  (B) Parsing + Normalization (streaming, chunked)                                |
|      - detect header vs headerless                                                |
|      - strict column mapping                                                      |
|      - parse timestamps -> ns                                                     |
|      - preserve numeric precision (no rounding)                                   |
|      - dedup by unique ids / (ts, id) pairs                                       |
|                                                                                  |
|          |                                                                       |
|          v                                                                       |
|  (C) Canonical Events                                                            |
|      TradeTick / QuoteTick / Bar / Custom Data                                   |
|                                                                                  |
|          |                                                                       |
|          v                                                                       |
|  (D) Storage                                                                     |
|      ParquetDataCatalog (versioned) + manifest + ingestion report                |
|                                                                                  |
|          |                                                                       |
|          v                                                                       |
|  (E) Loading to Backtest                                                         |
|      - deterministic ordering across types                                        |
|      - optional filtering by date range                                           |
|      - optional downsampling / bucketing                                          |
|                                                                                  |
|          |                                                                       |
|          v                                                                       |
|  (F) Simulation                                                                  |
|      BacktestEngine -> Strategy -> Orders/Fills -> Reports                        |
|                                                                                  |
+----------------------------------------------------------------------------------+
```

## Phase ownership (and explicit non-ownership)

### Phase 2 owns
- File discovery and dataset inventory (daily/monthly).
- Parsing/validation/dedup.
- Writing ParquetDataCatalog.
- Producing **manifest** (what was ingested, counts, min/max timestamps, checksum-like signatures).

Phase 2 does **not** own:
- strategy logic,
- alpha definitions,
- execution modeling beyond getting data into canonical form.

### Phase 3 owns
- Feature and signal definitions (LPI family, volatility estimates, exhaustion detector).
- Strategy prototypes and ablations (maker-only vs mixed, gating logic).
- Backtest harness configuration (single-run deterministic experiments).

Phase 3 does **not** own:
- multi-run orchestration at scale,
- experiment registry,
- full reporting suite.

### Phase 4 owns
- Walk-forward experiment orchestration and configuration management.
- Batch runs, result persistence, and report generation.
- Deterministic experiment IDs and artifact layout.

Phase 4 does **not** own:
- changing canonical schemas,
- redesigning Phase 2 parsers,
- redefining signal math.

---

# Mathematical Formulation

This formalizes the research target in a way that is implementable with the offline datasets available.

## Notation

Let \( t \) denote event time (nanoseconds). Let \( \Delta \) denote a bucket interval in event-time (e.g., 1s) or volume-time (e.g., \(1{,}000{,}000\) USDT notional).

- Trades: \( \mathcal{T}_b = \{(p_k, q_k, d_k, \tau_k)\} \) in bucket \( b \), where:
  - \( p_k \) = trade price,
  - \( q_k \) = trade size (base units),
  - \( d_k \in \{-1, +1\} \) indicates aggressor direction (sell/buy),
  - \( \tau_k \) = trade timestamp.
- Quotes (top-of-book): \( (b_t, a_t, B_t, A_t) \) = best bid/ask prices and sizes at time \(t\).
- Open interest metric: \( OI_t \) (slow updates, e.g., every 5 minutes from `metrics`).

## Returns and realized volatility

Define midprice:
\[
m_t = \frac{a_t + b_t}{2}
\]

Define log return sampled on a chosen clock (trade-time or fixed-time):
\[
r_{t_i} = \ln\left(\frac{m_{t_i}}{m_{t_{i-1}}}\right)
\]

Rolling realized variance (RV) over a window \(W\):
\[
RV_t(W) = \sum_{i: t-W < t_i \le t} r_{t_i}^2
\]
and realized volatility:
\[
\sigma_t(W) = \sqrt{RV_t(W)}
\]

## Liquidation Pressure Index family (offline-implementable)

Given the limitations of `binance.vision` offline data:
- We have **trade prints** and **top-of-book** (`bookTicker`).
- We do **not** have full L2 diffs unless separately downloaded; we treat `bookDepth` percent-depth as an optional proxy.

### (1) Impact sell / buy notional in a bucket

For bucket \(b\), define impact threshold \(\eta\) (in fractional terms, e.g., 5 bps = 0.0005).

Let \(b_{pre(k)}\) be the last-known best bid immediately before trade \(k\), and \(a_{pre(k)}\) best ask.

Impact sell notional:
\[
IS_b = \sum_{k \in \mathcal{T}_b} (p_k q_k)\,\mathbf{1}[d_k = -1]\,\mathbf{1}[p_k < b_{pre(k)}(1-\eta)]
\]

Impact buy notional:
\[
IB_b = \sum_{k \in \mathcal{T}_b} (p_k q_k)\,\mathbf{1}[d_k = +1]\,\mathbf{1}[p_k > a_{pre(k)}(1+\eta)]
\]

### (2) Liquidity depletion proxy

Top-of-book spread:
\[
s_t = a_t - b_t
\]

If only `bookTicker` is available, define a minimal liquidity proxy:
\[
LD_b = Z\left(\frac{s_{t_b}}{m_{t_b}}\right)
\]
where \(t_b\) is the bucket close timestamp (or last quote timestamp inside bucket).

If optional `bookDepth` percent-depth (notional within ±X%) is available, define:
\[
Depth^\$_t(X) = \text{notional depth within } X\% \text{ of touch}
\]
and:
\[
LD_b = \frac{s_{t_b}}{Depth^\$_{t_b}(X) + \epsilon}
\]

### (3) Fast pressure score

Define robust z-scores \(Z(\cdot)\) computed on rolling history (window \(H\) in buckets):
\[
P_b = Z(IS_b - IB_b)
\]
\[
L_b = Z(LD_b)
\]

Fast liquidation pressure index:
\[
LPI^{fast}_b = P_b + \lambda_{LD}\,L_b
\]

### (4) Slow leverage regime prior (not tick-synced)

Given slow OI/funding updates, define:
\[
LEV_t = Z(OI_t) + \lambda_F Z(Funding_t)
\]
or (if funding is not loaded initially):
\[
LEV_t = Z(OI_t)
\]

This is used **only as a regime filter**, not as a denominator in a fast signal (avoids the “incompatible clocks” failure mode).

### (5) Exhaustion detection

Define pressure change:
\[
\Delta LPI^{fast}_b = LPI^{fast}_b - LPI^{fast}_{b-1}
\]

Exhaustion score:
\[
EXH_b = Z(-\Delta LPI^{fast}_b) + \lambda_{liq} Z(\Delta Depth^\$_{t_b}(X))
\]
If depth is unavailable, use a spread-reversion proxy:
\[
EXH_b = Z(-\Delta LPI^{fast}_b) + \lambda_{spr} Z(-(s_{t_b} - s_{t_{b-1}}))
\]

## Walk-forward / rolling window backtest definition (Phase 4)

Let the total timeline be partitioned into contiguous segments.

- Training window length: \(T_{train}\)
- Test window length: \(T_{test}\)
- Step size: \(T_{step}\)

For fold \(j\):
- Train interval: \([t_j, t_j + T_{train})\)
- Test interval: \([t_j + T_{train}, t_j + T_{train} + T_{test})\)

Purging/embargo (to avoid leakage in clustered volatility):
- Purge interval: \(T_{purge}\) removed around the boundary.

## Position sizing and risk constraints (research-grade defaults)

Define target notional exposure:
\[
N_b = \min\left(N_{max}, \frac{E \cdot \kappa \cdot \mu}{\sigma_b^2 + \lambda_{jump}}\right)
\]
where:
- \(E\) is equity,
- \(\mu\) is assumed edge per trade (research assumption; must be validated),
- \(\kappa \in (0,1]\) is risk scaler,
- \(\lambda_{jump}\) is jump-risk floor.

Hard constraints:
- Max gross leverage \( \le L_{max} \)
- Kill-switch if spread exceeds threshold or depth proxy collapses.

---

# Algorithmic Specifications

## ETL ingestion (Phase 2): streaming, deduplication, validation

### Objectives
- Convert heterogeneous CSV formats into canonical NautilusTrader event objects.
- Avoid memory blowups via chunked reads.
- Ensure deterministic output.

### File discovery algorithm
1. Given `RAW_ROOT` and enabled datasets, enumerate candidate files by glob patterns:
   - daily: `.../daily/<dataset>/*.csv`
   - monthly: `.../monthly/<dataset>/*.csv`
2. Extract symbol and date from filename using regex:
   - daily: `YYYY-MM-DD`
   - monthly: `YYYY-MM`
3. Build an inventory table: `(dataset, symbol, period_type, date_key, path, size_bytes, mtime)`.

### Parsing algorithm (per dataset)
Each parser must:
- accept a file path and an `InstrumentId`,
- produce an iterator / list of canonical events,
- emit validation stats (rows read, rows emitted, parse errors, timestamp range).

**Key rules (apply to all parsers):**
- Determine header vs headerless by inspecting first row / column names.
- Map columns by either:
  - known header names, or
  - positional schema (documented per dataset).
- Convert timestamps:
  - ms → ns via `ns = ms * 1_000_000`.
  - ISO string → UTC ns via a strict parser.
- Preserve numeric precision:
  - never format floats into fixed decimals,
  - prefer raw strings if available.
- Deduplicate:
  - For `aggTrades`: primary key = `agg_trade_id`.
  - For `trades`: primary key = `id`.
  - For `bookTicker`: primary key = `event_time` + `update_id` (or `(event_time, bid, ask)` fallback).
  - For bars: primary key = `open_time`.
- Enforce monotonicity checks (soft-fail with logging):
  - timestamps should be non-decreasing within file after dedup.
  - if violations occur, sort deterministically by `(ts_event, secondary_id)`.

### Catalog write strategy
- Use `ParquetDataCatalog` (as shown in Nautilus examples) as the persistent store.
- Write in batches:
  - chunk input rows,
  - convert to objects,
  - `catalog.write_data(batch)`.

**Deterministic ordering within batch:**
- Always sort events by:
  \[
  (ts\_event,\; tie\_breaker)
  \]
  tie-breaker is dataset-specific (trade_id, update_id, open_time).

### Validation outputs (Phase 2 artifacts)
- `manifest.json` (or `.yaml`) capturing:
  - git commit hash (if available), code version,
  - raw file inventory snapshot,
  - per-file counts and timestamp ranges,
  - total counts by dataset,
  - parse error counts.
- `validation_report.md` summarizing issues.

## Feature/stat computation pipeline (Phase 3)

### Inputs
- `TradeTick` stream (futures).
- `QuoteTick` stream (futures).
- Optional `Bar` stream.
- Custom `Data` streams: OI metrics, funding, bookDepth.

### Pipeline design
- Maintain a **Last-Known-State (LKS)** buffer for top-of-book:
  - update on each `QuoteTick`,
  - for each `TradeTick`, associate the last quote timestamp \( \le \tau_k \).
- Aggregate trades into buckets (choose one clock per experiment):
  1. Fixed-time buckets (e.g., 1s) based on `ts_event`, **or**
  2. Volume buckets based on accumulated notional.

**Default:** volume buckets sized by notional \(V_{bucket}^\$\) to align with “volume as the clock”.

### Rolling normalization
- Maintain robust rolling statistics for z-scores:
  - mean/std on rolling window of bucket values, or
  - median/MAD (preferred for heavy tails).

**Implementation note:** Use an online rolling mechanism (bounded memory) to avoid scanning large histories.

## Strategy decision loop (Phase 3)

We design two variants, sharing common signals, consistent with the LPI proposal’s execution realism.

### Shared state (per instrument)
- latest quote: best bid/ask, spread, mid
- rolling volatility estimate \(\sigma_b\)
- signal values: `LPI_fast`, `EXH`, `LEV`
- execution gates: spread threshold, data freshness thresholds

### Variant A: Cascade Momentum (continuation)
- Entry condition:
  - \(|LPI^{fast}_b| \ge \theta_{LPI}\)
  - and \(EXH_b \le \theta_{EXH,low}\)
- Direction:
  - if \(LPI^{fast}_b > 0\): trade **with sell pressure** (short)
  - if \(LPI^{fast}_b < 0\): trade **with buy pressure** (long)
- Exit condition:
  - time stop (short horizon),
  - or \(EXH_b\) crosses above exhaustion threshold,
  - or spread gate violated.

### Variant B: Post-Exhaustion Mean Reversion
- Entry condition:
  - \(|LPI^{fast}_b| \ge \theta_{LPI}\)
  - and \(EXH_b \ge \theta_{EXH,high}\)
  - and (optional) leverage prior \(LEV\) indicates crowdedness.
- Direction:
  - trade **against** the prior pressure.
- Execution:
  - maker-first logic (post-only), price improvement, tranche entries.
- Exit:
  - normalization of LPI,
  - adverse selection guard (markout negative),
  - hard stop if pressure re-accelerates.

> **Important:** This design explicitly fixes the “catching the falling knife” issue by requiring exhaustion for mean reversion.

## Backtest orchestration (Phase 4): rolling windows, caching strategy

### Execution plan
- Use `BacktestEngine` (as shown in examples) as the core simulator.
- Prefer a **catalog-driven data loading** approach to avoid loading all events into RAM at once.

**TO BE CONFIRMED FROM NAUTILUS CONTEXT:** best practice for reading subsets from `ParquetDataCatalog` by time range for `TradeTick` / `QuoteTick` / custom `Data`.  
- If the catalog offers time-range queries, use them.
- If not, implement a thin “CatalogSliceLoader” that:
  - loads partitions by date/file boundaries (known from manifest),
  - streams them into the engine in deterministic order.

### Walk-forward loop
For each fold:
1. Initialize engine with venue + instrument spec.
2. Load required data for `[train_start, test_end]` (signals may require warmup).
3. Run backtest over test interval (strategy runs continuously but metrics only scored on test).
4. Persist fold artifacts: fills, positions, equity curve, signal traces.

---

# Performance, Disk Footprint, and Determinism Plan

## Complexity and I/O considerations

### Ingestion (Phase 2)
- Let \(N\) be total CSV rows across datasets.
- Parsing is \(O(N)\).
- Object creation is expensive; minimize Python overhead by:
  - chunked ingestion,
  - avoiding per-row string formatting,
  - minimizing exception-based control flow.

### Storage
- ParquetDataCatalog size is driven primarily by:
  - `TradeTick` volume (largest),
  - `QuoteTick` volume (also large),
  - bars and metrics are small by comparison.

**Disk planning default (qualitative):**
- Expect multi-GB for multi-month tick history per major symbol.

## Caching rules and allowed materialization

Allowed on disk:
- ParquetDataCatalog (primary).
- Manifest and validation reports.
- Optional “research extracts”:
  - downsampled quote stream,
  - precomputed bucketed features for a fixed bucket size.

Not allowed (by default):
- duplicating full raw CSV contents in another format (raw is already on disk).
- silently changing canonical schemas without bumping version/manifest.

## Determinism plan

Determinism requires four guarantees:

1. **Stable parsing**
   - fixed column mappings per dataset,
   - explicit timezone handling (UTC),
   - no locale-dependent parsing.

2. **Stable ordering**
   - within each dataset: sort by `(ts_event, primary_id)`.
   - across datasets: global merge ordering by `ts_event` then a deterministic stream priority:
     - quotes before trades at identical timestamps (so trades see last-known quote),
     - bars after ticks at same `open_time` (to avoid look-ahead use of OHLC).

3. **Stable numeric handling**
   - no float formatting,
   - controlled conversion boundaries (`Price/Quantity` from strings),
   - avoid pandas float rounding.

4. **Stable experiment identifiers**
   - experiment config hashed into an ID (Phase 4),
   - all outputs stored under that ID.

---

# Interfaces & Contracts Between Phases

This section defines the **exact artifacts** that must exist so later phases can be implemented without redesign.

## Phase 2 outputs (ETL + catalog)

### Artifact A: ParquetDataCatalog directory
- Path: `data/catalog/<catalog_name>/`
- Contains:
  - instrument definitions (if stored),
  - `TradeTick`, `QuoteTick`, `Bar`,
  - custom `Data` objects (OI metrics, funding, depth proxies).

### Artifact B: `manifest.json`
Must include:
- `raw_root` absolute path (string)
- `ingestion_version` (string)
- list of ingested datasets and file counts
- per dataset:
  - total events written,
  - min/max `ts_event`,
  - dedup stats,
  - parse error stats.

### Artifact C: `instrument_specs.yaml`
- Defines instrument parameters needed to build Nautilus instruments:
  - symbol, venue, instrument class (perp/spot),
  - tick size, lot size,
  - price/size precision,
  - fees (maker/taker),
  - min notional / min quantity.

**TO BE CONFIRMED FROM NAUTILUS CONTEXT:** exact required fields for the chosen instrument class constructor.  
Contract: Phase 3/4 must not parse raw CSV to infer specs; they must read this file.

## Phase 3 outputs (signals + strategies)

### Artifact D: Feature/signal module API
- `FeatureEngine` that consumes Nautilus events and produces a `SignalSnapshot` object at bucket boundaries.

Contract (conceptual):
- Input: `TradeTick`, `QuoteTick`, optional custom `Data`.
- Output: `SignalSnapshot(ts_event, lpi_fast, exh, lev, spread, depth_proxy, rv, validity_flags, …)`.

Where Nautilus signatures are uncertain, define adapter boundary:

- **Adapter interface:** `SignalEngineProtocol`
  - `on_trade_tick(tick)`
  - `on_quote_tick(tick)`
  - `on_custom_data(data)`
  - `maybe_emit_signal() -> Optional[SignalSnapshot]`

This prevents redesign if we later need to adjust how Nautilus dispatches events.

### Artifact E: Strategy modules
- Nautilus `Strategy` subclasses:
  - Variant A: cascade momentum,
  - Variant B: post-exhaustion mean reversion.

They must depend only on:
- canonical event types + custom data classes from Phase 2,
- the FeatureEngine/SignalEngine module.

## Phase 4 outputs (runner + metrics)

### Artifact F: Experiment config format
- `experiment.yaml` describing:
  - instruments to run,
  - date ranges,
  - bucket size / tick interval,
  - signal parameters (\(\eta, \theta_{LPI}, \theta_{EXH}\), etc.),
  - execution model toggles (maker-only, latency penalty, fee tier).

### Artifact G: Result bundle per experiment
Directory layout:
```
runs/<experiment_id>/
  config_resolved.yaml
  manifest_ref.json
  fills.csv
  positions.csv
  account_report.txt (or csv)
  signal_trace.parquet (optional)
  summary.json
```

**TO BE CONFIRMED FROM NAUTILUS CONTEXT:** exact report generation functions available (examples show `engine.trader.generate_*_report()` returning pandas DataFrames).

---

# Validation & Testing Strategy

## Unit tests (Phase 2)

1. **Timestamp conversion tests**
   - ms → ns exactness.
   - ISO → ns UTC correctness.

2. **Schema mapping tests**
   - headerless vs headered detection produces correct column mapping.

3. **Precision preservation tests**
   - ensure ingestion does not apply formatting like `:.2f` / `:.3f`.
   - roundtrip: raw string → `Price/Quantity` → back to string retains value within instrument precision rules.

4. **Deduplication tests**
   - duplicates in `agg_trade_id` removed deterministically.
   - stable ordering after dedup.

## Integration tests (Phase 2 → Phase 3)

1. **Catalog write/read sanity**
   - after catalog build, load a small date range and assert event counts match manifest.
2. **Cross-stream alignment sanity**
   - for a selected time range, assert trades see a “recent enough” quote (LKS staleness threshold).

## Strategy tests (Phase 3)

1. **No-lookahead invariants**
   - signals computed at bucket close must not use future quotes/trades.
2. **State machine invariants**
   - no overlapping entry orders if strategy says “one-at-a-time”.
   - order cancel/replace logic deterministic.

## Acceptance criteria per phase

### Phase 2 acceptance
- Catalog builds successfully for at least one symbol and one month.
- Manifest produced with nonzero counts for `TradeTick` and `QuoteTick`.
- No lossy rounding present in stored prices/sizes (spot-check sample).

### Phase 3 acceptance
- Signal engine produces stable `SignalSnapshot` stream.
- At least one strategy variant places trades in a small test interval (even if unprofitable).
- Ablation toggles (e.g., no depth proxy, no OI prior) can be turned on/off without code changes.

### Phase 4 acceptance
- Walk-forward runner executes multiple folds deterministically.
- Produces standardized result bundles with summaries and fill reports.

---

# Risks and Mitigations

## Risk: Precision loss from ingestion (rounding / float formatting)
- **Mitigation:** enforce “no formatting” rule; parse numeric fields as strings; add unit tests.

## Risk: Timestamp semantics mismatch (transact_time vs event_time)
- `bookTicker` has both `transaction_time` and `event_time`.
- **Mitigation:** define canonical choice:
  - `ts_event` uses **event time** if present,
  - fallback to transaction time if event time missing,
  - always record source field in manifest stats.

## Risk: Incompatible clocks (fast tick data vs slow OI updates)
- **Mitigation:** treat OI/funding as slow regime priors only; do not normalize 100ms signals by slow deltas.

## Risk: Memory blowups when loading a year of ticks into RAM
- **Mitigation:** catalog + slicing loader; chunked ingestion; phase 4 walk-forward loads only needed intervals.

## Risk: Non-deterministic event ordering when merging multiple streams
- **Mitigation:** global ordering spec and stable tiebreakers; write ordering tests.

## Risk: Schema drift across binance.vision dumps
- Some files headerless, some headered, some have different column names.
- **Mitigation:** robust header detection + positional schema fallback + per-file validation logs.

## Risk: Strategy “paper alpha” due to execution realism gap
- **Mitigation:** phase 3 includes markout tracking hooks and execution gates; phase 4 adds systematic latency/fee sensitivity experiments.

---

# Deliverables Checklist (Phase 2–4)

## Phase 2 (ETL + Catalog)
- [ ] Configurable file discovery for futures + spot roots.
- [ ] Dataset parsers for: futures `aggTrades`, `bookTicker`, `klines_1m`, `metrics`, `fundingRate`, optional `bookDepth`.
- [ ] Canonical event construction using NautilusTrader types (`TradeTick`, `QuoteTick`, `Bar`) + custom `Data` via `@customdataclass`.
- [ ] ParquetDataCatalog writer with batch/chunk support.
- [ ] `manifest.json` + validation report.

## Phase 3 (Signals + Strategy Prototypes)
- [ ] Signal engine implementing `LPI_fast`, `EXH`, `LEV` with robust normalization.
- [ ] Strategy variant A (cascade momentum).
- [ ] Strategy variant B (post-exhaustion mean reversion).
- [ ] Backtest harness for a single run with deterministic configuration.
- [ ] Ablation toggles (no depth proxy, no OI prior, maker-only vs mixed).

## Phase 4 (Runner + Reporting)
- [ ] Experiment config format and resolver.
- [ ] Walk-forward orchestrator (train/test splits + purge).
- [ ] Result bundles with standardized filenames and summaries.
- [ ] Deterministic experiment IDs and reproducibility metadata capture.

---

# Blocking Questions (Must Answer Before Phase 2)

1. **Universe scope for initial implementation:** Should Phase 2/3 target **only Binance USD‑M futures (UM)** first (recommended), or must **spot + futures** both be supported in the initial ingestion deliverable? Answer: UM futures only (Binance USD‑M). No spot in the initial ingestion deliverable.
2. **Instrument set and time range:** Confirm the initial instrument(s) and date range to treat as the “acceptance baseline” (e.g., `BTCUSDT` from `2023-05-16` to `2024-03-31` as implied by the file listings). Answer: Acceptance baseline is BTCUSDT (Binance UM perpetual), 2023-05-16 through 2024-03-31 (UTC).
3. **Instrument specification source:** Will you provide a minimal `instrument_specs.yaml` (tick size, size increment, precisions, fees), or should the system assume a fixed spec for BTCUSDT perps as a temporary default (explicitly flagged as temporary)? Answer: You should provide instrument_specs.yaml (instead of hard-coding), using the system’s pre-assumed fixed spec for BTCUSDT UM perp as the baseline. The YAML will be explicitly marked “assumed / acceptance baseline”.
4. **Strategy objective**: Should the default Phase 3 strategy be the Phoenix/LPI liquidation-pressure strategy implied by the provided prototype (FSI/VPIN/LDR/OI + Kelly sizing), or do you want a different baseline (e.g., EMA cross)? Answer: Default strategy = EMA cross baseline. Additionally implement Phoenix/LPI (FSI/VPIN/LDR/OI + sizing) as a comparative strategy.
5. **Backtest date range and instruments**: What are the exact start/end timestamps (UTC) and the exact symbols (e.g., BTCUSDT only, or multiple)? Answer: Backtest BTCUSDT only, UTC start 2023-05-16T00:00:00Z, UTC end 2024-03-31T23:59:59.999Z (or end-exclusive 2024-04-01T00:00:00Z, choose one and standardize).
