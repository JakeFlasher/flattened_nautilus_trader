## 1) Time-sliced loading from a Phase 2 `ParquetDataCatalog` (no raw CSV re-parse)

### 1A) **Recommended for Phase 4 backtests**: let `BacktestNode` load slices via `BacktestDataConfig`

NautilusTrader’s high-level backtest API is designed to load *time-filtered* data from a `ParquetDataCatalog` using `BacktestDataConfig` objects, which are then consumed by `BacktestNode` / `BacktestRunConfig`. ([nautilustrader.io](https://nautilustrader.io/docs/latest/tutorials/loading_external_data/))

**Authoritative config fields you can rely on (start/end slicing):**

`BacktestDataConfig(..., start_time=..., end_time=...)` accepts ISO-8601 strings (UTC) or UNIX nanoseconds ints. It also supports `instrument_id` or `instrument_ids` and optional `filter_expr`, `client_id`, `metadata`, etc. ([nautilustrader.io](https://nautilustrader.io/docs/nightly/api_reference/config/))

### Authoritative snippet (Phase 4 runner style)

```python
from nautilus_trader.backtest.node import (
    BacktestNode,
    BacktestDataConfig,
    BacktestEngineConfig,
    BacktestRunConfig,
    BacktestVenueConfig,
)
from nautilus_trader.config import ImportableStrategyConfig
from nautilus_trader.model import QuoteTick, TradeTick, BarType, Venue

# Your custom data classes (Phase 2 wrote them into the same catalog)
from phoenix_etl.custom_data import BinanceOiMetrics, BinanceFundingRate, BinanceBookDepthPct

CATALOG_PATH = "/abs/path/to/data/catalog/phoenix_um_btcusdt"  # Phase 2 output

instrument_id = "BTCUSDT.BINANCE"
start = "2023-05-16T00:00:00Z"
end   = "2023-05-17T00:00:00Z"  # recommend end-exclusive at day boundary for clarity

data_configs = [
    BacktestDataConfig(
        catalog_path=CATALOG_PATH,
        data_cls=QuoteTick,
        instrument_id=instrument_id,
        start_time=start,
        end_time=end,
    ),
    BacktestDataConfig(
        catalog_path=CATALOG_PATH,
        data_cls=TradeTick,
        instrument_id=instrument_id,
        start_time=start,
        end_time=end,
    ),

    # Custom Data streams (same time slicing contract)
    BacktestDataConfig(
        catalog_path=CATALOG_PATH,
        data_cls=BinanceOiMetrics,
        instrument_id=instrument_id,
        start_time=start,
        end_time=end,
    ),
    BacktestDataConfig(
        catalog_path=CATALOG_PATH,
        data_cls=BinanceFundingRate,
        instrument_id=instrument_id,
        start_time=start,
        end_time=end,
    ),
    BacktestDataConfig(
        catalog_path=CATALOG_PATH,
        data_cls=BinanceBookDepthPct,
        instrument_id=instrument_id,
        start_time=start,
        end_time=end,
    ),
]

# Strategy wiring via ImportableStrategyConfig (Phase 4 will swap this per experiment)
strategies = [
    ImportableStrategyConfig(
        strategy_path="phoenix_research.strategies.ema_cross_baseline:EmaCrossBaseline",
        config_path="phoenix_research.strategies.ema_cross_baseline:EmaCrossBaselineConfig",
        config={
            "instrument_id": instrument_id,
            "bar_type": BarType.from_str(f"{instrument_id}-1-MINUTE-LAST-EXTERNAL"),
            "trade_size": "0.001",
            "fast_ema_period": 10,
            "slow_ema_period": 20,
        },
    ),
]

venue_configs = [
    BacktestVenueConfig(
        name="BINANCE",
        oms_type="NETTING",
        account_type="MARGIN",
        base_currency="USDT",
        starting_balances=["10000 USDT"],
    )
]

run_config = BacktestRunConfig(
    venues=venue_configs,
    data=data_configs,
    engine=BacktestEngineConfig(strategies=strategies),
    start=start,
    end=end,
)

node = BacktestNode(configs=[run_config])
results = node.run()

engine = node.get_engine(run_config.id)
```

**Why this is the Phase 4-friendly route**

- It is the documented way to load slices from a catalog into a backtest (so you don’t need to invent a custom “slice loader”). ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/data?utm_source=openai))
- It avoids low-level pitfalls around adding catalog-returned objects directly to `BacktestEngine` (see 1B note below). ([nautilustrader.io](https://nautilustrader.io/docs/latest/api_reference/backtest/?utm_source=openai))

**Important constraint (affects Phase 4 design):**  
Nautilus notes that *custom data* is only supported in **non-streaming** mode (`streaming=False` / “load all data at once”). If you were counting on chunked streaming for memory, you must either (i) ensure the slice fits in RAM or (ii) exclude custom data from the backtest engine and treat it as external research-only inputs. ([nautilustrader.io](https://nautilustrader.io/docs/latest/api_reference/backtest/?utm_source=openai))

---

### 1B) **Research / inspection**: query the `ParquetDataCatalog` directly (time-sliced)

If Phase 4 also needs “offline research extracts” (for debugging, assertions, or building signal traces), you can query the catalog directly.

There are **two** documented query styles:

#### (i) Generic `query(...)` (works for core + custom data)

`ParquetDataCatalog.query(...)` supports `start` and `end` as TimestampLike (`int|str|float`), plus `identifiers` (instrument IDs), and optional `where`. ([nautilustrader.io](https://nautilustrader.io/docs/latest/api_reference/persistence/))

```python
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model import QuoteTick, TradeTick
from phoenix_etl.custom_data import BinanceOiMetrics

catalog = ParquetDataCatalog("/abs/path/to/data/catalog/phoenix_um_btcusdt")

quotes = catalog.query(
    data_cls=QuoteTick,
    identifiers=["BTCUSDT.BINANCE"],
    start="2023-05-16T00:00:00Z",
    end="2023-05-17T00:00:00Z",
)

trades = catalog.query(
    data_cls=TradeTick,
    identifiers=["BTCUSDT.BINANCE"],
    start="2023-05-16T00:00:00Z",
    end="2023-05-17T00:00:00Z",
)

oi = catalog.query(
    data_cls=BinanceOiMetrics,
    identifiers=["BTCUSDT.BINANCE"],
    start="2023-05-16T00:00:00Z",
    end="2023-05-17T00:00:00Z",
)
```

#### (ii) Convenience accessors (`quote_ticks`, `trade_ticks`, …)

The docs show direct calls like:

- `catalog.quote_ticks(instrument_ids=[...], start=..., end=...)` ([nautilustrader.io](https://nautilustrader.io/docs/latest/tutorials/loading_external_data/))
- `catalog.trade_ticks(instrument_ids=[...], ...)` exists per API reference ([nautilustrader.io](https://nautilustrader.io/docs/latest/api_reference/persistence/))

Use this when you want explicit typed lists for core market data.

---

### 1C) Practical Phase 4 decision: **don’t write a custom slice loader unless you must**

Given the documented backtest route uses `BacktestDataConfig` → `BacktestNode` (and already handles query construction + sorting + integration), Phase 4 should treat direct catalog queries as a *debug/research tool*, not as the backtest ingestion path. ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/data?utm_source=openai))

---

## 2) Instruments: how Phase 4 should supply `BTCUSDT.BINANCE` (Phase 2 didn’t store instruments)

### Required reality: **a backtest needs an `Instrument` in the cache**

The low-level backtest tutorial shows instruments must be added before data, and the backtest API documents `BacktestEngine.add_instrument(...)`. ([nautilustrader.io](https://nautilustrader.io/docs/latest/getting_started/backtest_low_level/))  
For the high-level API, the docs describe an “instrument loading” step from the catalog during backtest startup. ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/data?utm_source=openai))

So if Phase 2 does **not** write instrument definitions, Phase 4 must fix that.

---

### Recommended Phase 4 approach (answer): **(a) + (c)**

#### **(a) Canonical source of truth: `instrument_specs.yaml` (offline, deterministic)**  
You already planned this in Phase 1: specs provided as an explicit artifact, not inferred from CSVs.

#### **(c) Operationally easiest for BacktestNode**: write instrument definitions into the same catalog **once** (idempotent)

This is explicitly supported: `ParquetDataCatalog.write_data([instrument])` is shown in Nautilus’ external-data tutorial (they write an instrument, then write ticks). ([nautilustrader.io](https://nautilustrader.io/docs/latest/tutorials/loading_external_data/))  
The catalog API also exposes `catalog.instruments(...)` so you can check if it’s already present. ([nautilustrader.io](https://nautilustrader.io/docs/latest/api_reference/persistence/))

> Why not (b) `TestInstrumentProvider`?  
> It’s great for *test stubs* (spot examples), but it’s not guaranteed to match a **Binance USD‑M perpetual** instrument (tick size, margin settings, fee fields, and your required `InstrumentId` string). The docs only demonstrate spot helpers like `ethusdt_binance()` in the low-level tutorial. ([nautilustrader.io](https://nautilustrader.io/docs/latest/getting_started/backtest_low_level/))

---

### Concrete “Phase 4 preflight” contract (copy/paste into your spec)

**Phase 4 must run the following preflight step before creating `BacktestNode`:**

1. Load `instrument_specs.yaml`.
2. Build a `CryptoPerpetual` instrument for `InstrumentId.from_str("BTCUSDT.BINANCE")`.
3. Check if the instrument exists in the target `ParquetDataCatalog`.
4. If missing, `catalog.write_data([instrument])` to persist it.

`CryptoPerpetual` constructor signature (authoritative) is in the instruments API reference. ([nautilustrader.io](https://nautilustrader.io/docs/latest/api_reference/model/instruments/))  
Instrument ID parsing format is documented by `InstrumentId.from_str(...)` (examples include `BTCUSDT.BINANCE`). ([nautilustrader.io](https://nautilustrader.io/docs/latest/api_reference/model/identifiers?utm_source=openai))

---

### Minimal constructor template for Binance UM perp (`CryptoPerpetual`)

```python
from decimal import Decimal
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.currencies import BTC, USDT
from nautilus_trader.persistence.catalog import ParquetDataCatalog

# TODO: confirm Symbol import in your pinned Nautilus version (often nautilus_trader.model.identifiers.Symbol)
from nautilus_trader.model.identifiers import Symbol  # adjust if needed

def ensure_btcusdt_perp_in_catalog(catalog_path: str, spec: dict) -> None:
    catalog = ParquetDataCatalog(catalog_path)

    inst_id = InstrumentId.from_str("BTCUSDT.BINANCE")

    # Check by ID; API returns list[Instrument]
    existing = catalog.instruments(instrument_ids=[inst_id.value])
    if existing:
        return

    instrument = CryptoPerpetual(
        instrument_id=inst_id,
        raw_symbol=Symbol(spec["raw_symbol"]),                 # "BTCUSDT"
        base_currency=BTC,
        quote_currency=USDT,
        settlement_currency=USDT,
        is_inverse=bool(spec.get("is_inverse", False)),        # USDT-margined linear perp => False
        price_precision=int(spec["price_precision"]),
        size_precision=int(spec["size_precision"]),
        price_increment=Price.from_str(spec["price_increment"]),   # e.g. "0.10"
        size_increment=Quantity.from_str(spec["size_increment"]),  # e.g. "0.001"
        ts_event=0,
        ts_init=0,
        maker_fee=Decimal(spec["maker_fee"]),                  # e.g. "0.0002"
        taker_fee=Decimal(spec["taker_fee"]),                  # e.g. "0.0004"
    )

    # Persist instrument definition into the SAME catalog used for data loading
    catalog.write_data([instrument])
```

Notes:
- The required fields in your YAML should map exactly to the `CryptoPerpetual(...)` constructor fields (precision, increments, fees, inverse flag). ([nautilustrader.io](https://nautilustrader.io/docs/latest/api_reference/model/instruments/))
- You must keep `InstrumentId` consistent with Phase 2 ingestion (you’re using `BTCUSDT.BINANCE` everywhere). ([nautilustrader.io](https://nautilustrader.io/docs/latest/api_reference/model/identifiers?utm_source=openai))
- If you later add spot + futures, you should consider distinct symbols (e.g., `BTCUSDT-PERP.BINANCE`) to avoid cross-venue ambiguity; Nautilus docs warn about symbology collisions. ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/instruments/?utm_source=openai))

---

## 3) Phase 4 `summary.json`: minimum required keys + where to get equity/returns

### 3A) Minimum `summary.json` keys (proposed contract)

Phase 4 should write a `summary.json` that is:

- **self-contained** for comparing runs
- **stable** across strategy variants
- **computable from Nautilus outputs** without relying on logs

Here is a **minimum key set** that will cover your Phase 1 expectations *and* support walk-forward reporting:

```json
{
  "schema_version": "1",
  "experiment_id": "string",
  "run_config_id": "string",
  "strategy": {
    "name": "string",
    "variant": "string|null",
    "config_hash": "string"
  },
  "universe": {
    "venue": "BINANCE",
    "instrument_ids": ["BTCUSDT.BINANCE"],
    "start_utc": "2023-05-16T00:00:00Z",
    "end_utc": "2023-05-17T00:00:00Z"
  },
  "counts": {
    "orders": 0,
    "fills": 0,
    "positions": 0
  },
  "performance": {
    "starting_equity": 0.0,
    "ending_equity": 0.0,
    "total_pnl": 0.0,
    "total_return": 0.0,
    "max_drawdown": null,
    "sharpe": null,
    "deflated_sharpe": null
  },
  "artifacts": {
    "fills_csv": "fills.csv",
    "positions_csv": "positions.csv",
    "orders_csv": "orders.csv",
    "account_report_csv": "account_report.csv"
  }
}
```

Optional-but-useful additions (recommended once stable):
- `fees_paid`, `maker_vs_taker_breakdown`
- `win_rate`, `profit_factor`, `avg_trade_pnl`
- `exposure`: max gross exposure, time-in-market
- `errors`: any ingestion/backtest exceptions

---

### 3B) Where to get equity/returns series (authoritative, Nautilus-native)

You have two “official” data sources post-run:

#### (i) **Reports** (pandas DataFrames), via Trader helper methods  
Nautilus documents generating reports via `engine.trader.generate_*_report()`, including orders, fills, positions, and account reporting. ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/reports?utm_source=openai))

Use:
- `engine.trader.generate_orders_report()`
- `engine.trader.generate_fills_report()` (or `generate_order_fills_report()` depending on your pinned version)
- `engine.trader.generate_positions_report()`
- `engine.trader.generate_account_report(Venue("BINANCE"))` (shown in the quickstart analysis step) ([nautilustrader.io](https://nautilustrader.io/docs/latest/getting_started/quickstart?utm_source=openai))

**Equity/returns series recommendation (Phase 4):**
- Treat the **account report** as the canonical equity curve (base currency = USDT in your venue config).
- Compute returns from its equity column (simple returns or log returns), then compute Sharpe and DSR.

#### (ii) **Portfolio analyzer stats** (ready-made metrics)  
Nautilus also documents pulling aggregate performance statistics from:

- `engine.portfolio.analyzer.get_performance_stats_pnls()`
- `engine.portfolio.analyzer.get_performance_stats_returns()`
- `engine.portfolio.analyzer.get_performance_stats_general()` ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/reports?utm_source=openai))

**Phase 4 rule of thumb:**
- Use portfolio analyzer stats for **max_drawdown, sharpe, etc.** if present.
- Use account-report-derived series for **Deflated Sharpe Ratio** (DSR) since DSR is not a standard Nautilus metric and depends on your multiple-testing accounting.

---

### 3C) Concrete Phase 4 extraction template (robust to column-name drift)

```python
import math
import pandas as pd
from nautilus_trader.model import Venue

def extract_summary(engine, run_config, *, experiment_id: str) -> dict:
    venue = Venue("BINANCE")

    orders = engine.trader.generate_orders_report()
    fills  = engine.trader.generate_fills_report()
    pos    = engine.trader.generate_positions_report()
    acct   = engine.trader.generate_account_report(venue)

    # Counts
    counts = {
        "orders": int(len(orders)),
        "fills": int(len(fills)),
        "positions": int(len(pos)),
    }

    # Equity series (best-effort column mapping)
    equity_col_candidates = ["equity", "equity_total", "net_liquidation_value", "balance_total"]
    equity_col = next((c for c in equity_col_candidates if c in acct.columns), None)

    starting_equity = ending_equity = total_return = total_pnl = None
    if equity_col is not None and len(acct) >= 2:
        equity = acct[equity_col].astype(float)
        starting_equity = float(equity.iloc[0])
        ending_equity = float(equity.iloc[-1])
        total_pnl = ending_equity - starting_equity
        total_return = (ending_equity / starting_equity - 1.0) if starting_equity else None

    # Portfolio analyzer stats (if available)
    sharpe = max_dd = None
    try:
        stats_returns = engine.portfolio.analyzer.get_performance_stats_returns()
        stats_general = engine.portfolio.analyzer.get_performance_stats_general()
        # Key names vary; treat as best-effort.
        sharpe = float(stats_returns.get("sharpe", stats_returns.get("sharpe_ratio"))) if stats_returns else None
        max_dd = float(stats_general.get("max_drawdown", stats_general.get("max_drawdown_pct"))) if stats_general else None
    except Exception:
        pass

    summary = {
        "schema_version": "1",
        "experiment_id": experiment_id,
        "run_config_id": run_config.id,
        "strategy": {
            "name": "TBD",          # set from your experiment config
            "variant": None,        # e.g., "ema_cross" | "phoenix_lpi"
            "config_hash": "TBD",   # hash resolved strategy config
        },
        "universe": {
            "venue": "BINANCE",
            "instrument_ids": ["BTCUSDT.BINANCE"],
            "start_utc": str(run_config.start),
            "end_utc": str(run_config.end),
        },
        "counts": counts,
        "performance": {
            "starting_equity": starting_equity,
            "ending_equity": ending_equity,
            "total_pnl": total_pnl,
            "total_return": total_return,
            "max_drawdown": max_dd,
            "sharpe": sharpe,
            "deflated_sharpe": None,  # computed in your Phase 4 metrics layer if desired
        },
        "artifacts": {
            "orders_csv": "orders.csv",
            "fills_csv": "fills.csv",
            "positions_csv": "positions.csv",
            "account_report_csv": "account_report.csv",
        },
    }
    return summary
```

This template is aligned with:
- documented report generation patterns ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/reports?utm_source=openai))
- documented portfolio analyzer access ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/reports?utm_source=openai))

---

## Net-net “blocked questions” answers (ready to paste)

1) **Time-sliced loading:** Use `BacktestDataConfig(... start_time=..., end_time=...)` with `BacktestNode`. For ad hoc research, use `ParquetDataCatalog.query(data_cls=..., identifiers=[...], start=..., end=...)`. ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/data?utm_source=openai))

2) **Instruments:** Implement (a)+(c): build `CryptoPerpetual` from `instrument_specs.yaml` and write it into the same `ParquetDataCatalog` (idempotent preflight) so `BacktestNode` can load it. Do **not** depend on `TestInstrumentProvider` for Binance USD‑M perpetual correctness. ([nautilustrader.io](https://nautilustrader.io/docs/latest/api_reference/model/instruments/))

3) **`summary.json`:** Define a stable minimum schema (IDs, time window, counts, equity/PnL/return, Sharpe, max drawdown, artifact filenames). Extract equity/returns from `engine.trader.generate_account_report(...)` and/or portfolio analyzer stats; extract counts from orders/fills/positions reports. ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/reports?utm_source=openai))

---
Learn more:
1. [Loading external data | NautilusTrader Documentation](htt://nautilustrader.io/docs/latest/tutorials/loading_external_data/)
2. [Config | NautilusTrader Documentation](htt://nautilustrader.io/docs/nightly/api_reference/config/)
3. [Data | NautilusTrader Documentation](htt://nautilustrader.io/docs/latest/concepts/data?utm_source=openai)
4. [Backtest | NautilusTrader Documentation](htt://nautilustrader.io/docs/latest/api_reference/backtest/?utm_source=openai)
5. [Persistence | NautilusTrader Documentation](htt://nautilustrader.io/docs/latest/api_reference/persistence/)
6. [Backtest (low-level API) | NautilusTrader Documentation](htt://nautilustrader.io/docs/latest/getting_started/backtest_low_level/)
7. [Instruments | NautilusTrader Documentation](htt://nautilustrader.io/docs/latest/api_reference/model/instruments/)
8. [Identifiers | NautilusTrader Documentation](htt://nautilustrader.io/docs/latest/api_reference/model/identifiers?utm_source=openai)
9. [Instruments | NautilusTrader Documentation](htt://nautilustrader.io/docs/latest/concepts/instruments/?utm_source=openai)
10. [Reports | NautilusTrader Documentation](htt://nautilustrader.io/docs/latest/concepts/reports?utm_source=openai)
11. [Quickstart | NautilusTrader Documentation](htt://nautilustrader.io/docs/latest/getting_started/quickstart?utm_source=openai)