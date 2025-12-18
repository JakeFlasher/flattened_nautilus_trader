# Chapter 4 — Implementation Appendix (Pseudo‑Code & Diagrams)  
## (Glue Code: local Binance Futures CSV → ParquetDataCatalog → BacktestNode → Strategy → Risk → Execution)

**Role:** Lead Systems Architect & Data Engineer  
**Objective:** Provide the reproducible, high‑performance “glue” connecting:

- **Raw data** (offline CSV) under `data/raw/futures/*`  
  - `data/raw/futures/daily/aggTrades/…`  
  - `data/raw/futures/daily/bookTicker/…`  
  - `data/raw/futures/daily/klines_1m/…`  
- **Math/strategy** (Chapters 1–3: YZ volatility, VPIN toxicity, liquidity gating, risk sizing)  
- **Target runtime**: `ParquetDataCatalog` replayed by a `BacktestNode`/`DataEngine`.

---

## 4.1 The ETL Pipeline (Raw CSV → Parquet)

### 4.1.1 Raw input contracts (strict, offline)

We assume **offline file access only**, and we hard‑bind to the paths you provided:

- **AggTrades (trades / flow):**  
  `data/raw/futures/daily/aggTrades/BTCUSDT-aggTrades-YYYY-MM-DD.csv`
- **BookTicker (top-of-book quotes):**  
  `data/raw/futures/daily/bookTicker/BTCUSDT-bookTicker-YYYY-MM-DD.csv`
- **Klines 1m (OHLCV bars):**  
  `data/raw/futures/daily/klines_1m/BTCUSDT-1m-YYYY-MM-DD.csv`

> **Engineering invariant:** ingestion is deterministic.  
> We always process files in **lexicographic order** (which is chronological for these filenames).

---

### 4.1.2 Timestamp precision contract (ms → Unix nanoseconds)

Your CSV timestamps are in **milliseconds**, e.g. `1711756800000`. Nautilus uses **nanoseconds** (Unix epoch ns) everywhere.

**Conversion rule:**

\[
\texttt{ts\_event\_ns} = \texttt{timestamp\_ms} \times 1{,}000{,}000
\]

```python
from __future__ import annotations

NS_PER_MS: int = 1_000_000

def ms_to_unix_nanos(ts_ms: int) -> int:
    """
    Convert Unix milliseconds to Unix nanoseconds.

    Nautilus contract: timestamps are integer nanoseconds in UTC.
    """
    return int(ts_ms) * NS_PER_MS
```

---

### 4.1.3 Aggressor mapping: `is_buyer_maker` → `AggressorSide`

Binance futures semantics (consistent with Chapters 1–2):

- `is_buyer_maker == True`  → buyer was maker → seller was taker → **aggressor = SELL**
- `is_buyer_maker == False` → buyer was taker → **aggressor = BUY**

```python
from __future__ import annotations

from nautilus_trader.model.enums import AggressorSide

def aggressor_side_from_is_buyer_maker(is_buyer_maker: bool) -> AggressorSide:
    """
    Binance convention:
      - True  => seller-initiated trade => aggressor SELL
      - False => buyer-initiated trade  => aggressor BUY
    """
    return AggressorSide.SELL if is_buyer_maker else AggressorSide.BUY
```

This mapping is **critical** because:

- VPIN bucket accounting uses buy/sell volumes.
- Your Chapter 1 sign convention \(s_i\) depends on it.

---

### 4.1.4 Streaming CSV parsing strategy (high-performance, tolerant)

Binance “vision-style” CSVs are usually comma-separated, but you also showed whitespace/tabular outputs. We implement:

- **Dialect detection** (comma vs whitespace)
- **Header skipping**
- **Line-level streaming** (no pandas in ETL hot path)
- **Malformed line skipping** (fail-safe: skip bad lines rather than crash ETL)

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional
import csv


def _looks_like_header(fields: list[str]) -> bool:
    # Header usually contains alpha chars like "open_time" / "agg_trade_id"
    return any(any(ch.isalpha() for ch in tok) for tok in fields)


def iter_token_rows(path: Path) -> Iterator[list[str]]:
    """
    Stream tokenized rows from either comma CSV or whitespace format.

    - Skips empty lines
    - Skips header rows
    - Does not allocate entire file in memory
    """
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        # Peek a small sample to decide delimiter
        sample = []
        for _ in range(5):
            ln = f.readline()
            if not ln:
                break
            ln = ln.strip()
            if ln:
                sample.append(ln)

        if not sample:
            return

        is_comma = "," in sample[0]

        # Rewind
        f.seek(0)

        if is_comma:
            reader = csv.reader(f, delimiter=",")
            for row in reader:
                row = [x.strip() for x in row if x is not None]
                if not row or all(not x for x in row):
                    continue
                if _looks_like_header(row):
                    continue
                yield row
        else:
            # Whitespace tokenization
            first = True
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                row = ln.split()
                if first and _looks_like_header(row):
                    first = False
                    continue
                first = False
                yield row
```

---

### 4.1.5 BinanceFuturesCSVWrangler (AggTrades → TradeTick, BookTicker → QuoteTick, Klines → Bar)

Below is the core wrangler pseudo-code. It yields **typed Nautilus objects** with correct routing keys:

- `TradeTick.instrument_id`
- `QuoteTick.instrument_id`
- `Bar.bar_type`

…and strict timestamps (`ts_event`, `ts_init`) in **nanoseconds**.

> **Nautilus API references (from your docs / codebase):**  
> - `nautilus_trader.model.data` types (`TradeTick`, `QuoteTick`, `Bar`)  
> - `nautilus_trader.model.objects` fixed‑point types (`Price`, `Quantity`)  
> - `nautilus_trader.model.enums.AggressorSide` for aggressor mapping  
> - `nautilus_trader.persistence.catalog.parquet.ParquetDataCatalog` for storage  
> - `nautilus_trader.persistence.wranglers.*DataWrangler` for Arrow record batches

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from nautilus_trader.model import InstrumentId, BarType
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.enums import AggressorSide

from nautilus_trader.model.data import QuoteTick, TradeTick, Bar

NS_PER_MS: int = 1_000_000


def _parse_bool(token: str) -> bool:
    t = token.strip().lower()
    if t in ("true", "t", "1"):
        return True
    if t in ("false", "f", "0"):
        return False
    raise ValueError(f"Invalid boolean token: {token!r}")


def ms_to_unix_nanos(ts_ms: int) -> int:
    return int(ts_ms) * NS_PER_MS


def aggressor_side_from_is_buyer_maker(is_buyer_maker: bool) -> AggressorSide:
    return AggressorSide.SELL if is_buyer_maker else AggressorSide.BUY


# -----------------------------
# Row schemas (typed parsing)
# -----------------------------

@dataclass(frozen=True)
class AggTradesRow:
    agg_trade_id: int
    price: str
    quantity: str
    first_trade_id: int
    last_trade_id: int
    transact_time_ms: int
    is_buyer_maker: bool


def parse_aggtrades_row(tokens: list[str]) -> Optional[AggTradesRow]:
    """
    Expected futures aggTrades columns (per your preview):
      agg_trade_id price quantity first_trade_id last_trade_id transact_time is_buyer_maker
    """
    if len(tokens) < 7:
        return None
    try:
        return AggTradesRow(
            agg_trade_id=int(tokens[0]),
            price=tokens[1],
            quantity=tokens[2],
            first_trade_id=int(tokens[3]),
            last_trade_id=int(tokens[4]),
            transact_time_ms=int(tokens[5]),
            is_buyer_maker=_parse_bool(tokens[6]),
        )
    except ValueError:
        return None


@dataclass(frozen=True)
class BookTickerRow:
    update_id: int
    best_bid_price: str
    best_bid_qty: str
    best_ask_price: str
    best_ask_qty: str
    transaction_time_ms: int
    event_time_ms: int


def parse_bookticker_row(tokens: list[str]) -> Optional[BookTickerRow]:
    """
    Expected futures bookTicker columns (per your preview):
      update_id best_bid_price best_bid_qty best_ask_price best_ask_qty transaction_time event_time
    """
    if len(tokens) < 7:
        return None
    try:
        return BookTickerRow(
            update_id=int(tokens[0]),
            best_bid_price=tokens[1],
            best_bid_qty=tokens[2],
            best_ask_price=tokens[3],
            best_ask_qty=tokens[4],
            transaction_time_ms=int(tokens[5]),
            event_time_ms=int(tokens[6]),
        )
    except ValueError:
        return None


@dataclass(frozen=True)
class Kline1mRow:
    open_time_ms: int
    open_: str
    high: str
    low: str
    close: str
    volume: str
    close_time_ms: int


def parse_kline_1m_row(tokens: list[str]) -> Optional[Kline1mRow]:
    """
    Expected futures klines_1m columns (per your preview):
      open_time open high low close volume close_time quote_volume count taker_buy_volume taker_buy_quote_volume ignore
    We only require the first 7.
    """
    if len(tokens) < 7:
        return None
    try:
        return Kline1mRow(
            open_time_ms=int(tokens[0]),
            open_=tokens[1],
            high=tokens[2],
            low=tokens[3],
            close=tokens[4],
            volume=tokens[5],
            close_time_ms=int(tokens[6]),
        )
    except ValueError:
        return None


# -----------------------------
# Wrangler (streaming)
# -----------------------------

class BinanceFuturesCSVWrangler:
    """
    Streaming wrangler: raw Binance futures CSV -> Nautilus model objects.

    Design goals:
    - no pandas, minimal allocations
    - strict ts_event/ts_init in ns
    - deterministic file ordering
    """

    def __init__(self, instrument_id: InstrumentId, bar_type_1m: BarType) -> None:
        self.instrument_id = instrument_id
        self.bar_type_1m = bar_type_1m

    def iter_trade_ticks(self, path: Path) -> Iterator[TradeTick]:
        for tokens in iter_token_rows(path):
            row = parse_aggtrades_row(tokens)
            if row is None:
                continue

            ts_event = ms_to_unix_nanos(row.transact_time_ms)
            ts_init = ts_event  # offline/backtest ingestion: init == event

            yield TradeTick(
                instrument_id=self.instrument_id,
                price=Price.from_str(row.price),
                size=Quantity.from_str(row.quantity),
                aggressor_side=aggressor_side_from_is_buyer_maker(row.is_buyer_maker),
                trade_id=str(row.agg_trade_id),
                ts_event=ts_event,
                ts_init=ts_init,
                # Optional debug metadata:
                # info={"first_trade_id": row.first_trade_id, "last_trade_id": row.last_trade_id},
            )

    def iter_quote_ticks(self, path: Path) -> Iterator[QuoteTick]:
        for tokens in iter_token_rows(path):
            row = parse_bookticker_row(tokens)
            if row is None:
                continue

            ts_event = ms_to_unix_nanos(row.event_time_ms)
            ts_init = ts_event

            yield QuoteTick(
                instrument_id=self.instrument_id,
                bid_price=Price.from_str(row.best_bid_price),
                ask_price=Price.from_str(row.best_ask_price),
                bid_size=Quantity.from_str(row.best_bid_qty),
                ask_size=Quantity.from_str(row.best_ask_qty),
                ts_event=ts_event,
                ts_init=ts_init,
                # Optional debug metadata:
                # info={"update_id": row.update_id, "tx_time_ms": row.transaction_time_ms},
            )

    def iter_bars_1m(self, path: Path) -> Iterator[Bar]:
        for tokens in iter_token_rows(path):
            row = parse_kline_1m_row(tokens)
            if row is None:
                continue

            # Contract: bar timestamped on CLOSE
            ts_event = ms_to_unix_nanos(row.close_time_ms)
            ts_init = ts_event

            yield Bar(
                bar_type=self.bar_type_1m,
                open=Price.from_str(row.open_),
                high=Price.from_str(row.high),
                low=Price.from_str(row.low),
                close=Price.from_str(row.close),
                volume=Quantity.from_str(row.volume),
                ts_event=ts_event,
                ts_init=ts_init,
            )
```

---

### 4.1.6 Writing to ParquetDataCatalog (batch, Arrow-first)

NautilusTrader’s Parquet pipeline expects Arrow record batches, produced by built-in wranglers:

- `TradeTickDataWrangler`
- `QuoteTickDataWrangler`
- `BarDataWrangler`

We write in **bounded batches** to keep memory flat.

```python
from __future__ import annotations

from pathlib import Path
from typing import Iterable, TypeVar, Iterator, Callable

from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from nautilus_trader.persistence.wranglers import (
    TradeTickDataWrangler,
    QuoteTickDataWrangler,
    BarDataWrangler,
)

from nautilus_trader.model.data import TradeTick, QuoteTick, Bar

T = TypeVar("T")


def batched(it: Iterator[T], batch_size: int) -> Iterator[list[T]]:
    batch: list[T] = []
    for x in it:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_parquet_catalog_from_csv(
    *,
    raw_root: Path,
    catalog_root: Path,
    wrangler: "BinanceFuturesCSVWrangler",
    batch_size_ticks: int = 250_000,
    batch_size_bars: int = 100_000,
) -> None:
    """
    Offline ETL entrypoint:
      CSV (daily files) -> ParquetDataCatalog

    Paths are strict:
      - raw_root must contain data/raw/futures/...
      - catalog_root is where ParquetDataCatalog writes (e.g., data/catalog/)
    """
    catalog_root.mkdir(parents=True, exist_ok=True)
    catalog = ParquetDataCatalog(path=catalog_root)

    trade_w = TradeTickDataWrangler()
    quote_w = QuoteTickDataWrangler()
    bar_w = BarDataWrangler()

    # --- Trades (aggTrades) ---
    trades_dir = raw_root / "futures" / "daily" / "aggTrades"
    for fp in sorted(trades_dir.glob("BTCUSDT-aggTrades-*.csv")):
        for chunk in batched(wrangler.iter_trade_ticks(fp), batch_size_ticks):
            rb = trade_w.to_record_batch(chunk)
            catalog.write(rb)

    # --- Quotes (bookTicker) ---
    quotes_dir = raw_root / "futures" / "daily" / "bookTicker"
    for fp in sorted(quotes_dir.glob("BTCUSDT-bookTicker-*.csv")):
        for chunk in batched(wrangler.iter_quote_ticks(fp), batch_size_ticks):
            rb = quote_w.to_record_batch(chunk)
            catalog.write(rb)

    # --- Bars (klines_1m) ---
    bars_dir = raw_root / "futures" / "daily" / "klines_1m"
    for fp in sorted(bars_dir.glob("BTCUSDT-1m-*.csv")):
        for chunk in batched(wrangler.iter_bars_1m(fp), batch_size_bars):
            rb = bar_w.to_record_batch(chunk)
            catalog.write(rb)
```

**Reproducibility guarantees:**

- Deterministic file ordering (`sorted(glob(...))`)
- Deterministic timestamp conversion (ms→ns)
- Deterministic schema mapping (pure functions)
- Parquet is append-only during ingestion (no in-place mutation)

---

## 4.2 System Configuration (The “Wiring”)

This section shows the concrete wiring from **catalog → backtest node → strategy** using the May 2023 dataset.

### 4.2.1 Derive start/end time from filenames (May 2023)

Your raw inventory includes (at least):

- `BTCUSDT-aggTrades-2023-05-16.csv` … `2023-05-31.csv`
- `BTCUSDT-bookTicker-2023-05-16.csv` … `2023-05-31.csv`
- `BTCUSDT-1m-2023-05-16.csv` … `2023-05-31.csv`

We set:

- **Start:** `2023-05-16T00:00:00Z`
- **End:** `2023-06-01T00:00:00Z` (exclusive upper bound, safest for replay)

```python
from __future__ import annotations

import re
import datetime as dt
from pathlib import Path
from typing import Iterable


_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

def extract_dates_from_filenames(files: Iterable[Path]) -> list[dt.date]:
    out: list[dt.date] = []
    for fp in files:
        m = _DATE_RE.search(fp.name)
        if not m:
            continue
        out.append(dt.date.fromisoformat(m.group(1)))
    return sorted(set(out))


def compute_backtest_window_from_raw(raw_root: Path) -> tuple[dt.datetime, dt.datetime]:
    agg_dir = raw_root / "futures" / "daily" / "aggTrades"
    files = sorted(agg_dir.glob("BTCUSDT-aggTrades-*.csv"))
    dates = extract_dates_from_filenames(files)
    if not dates:
        raise ValueError(f"No aggTrades files found under {agg_dir}")

    start_date = dates[0]
    end_date_exclusive = dates[-1] + dt.timedelta(days=1)

    start = dt.datetime.combine(start_date, dt.time(0, 0, 0), tzinfo=dt.timezone.utc)
    end = dt.datetime.combine(end_date_exclusive, dt.time(0, 0, 0), tzinfo=dt.timezone.utc)
    return (start, end)
```

---

### 4.2.2 Instrument + venue setup (Binance Futures BTCUSDT perpetual)

You requested:

- **Instrument:** BTCUSDT on Binance
- **Precision:** infer from preview (e.g. price `69850.53` → 2 dp, qty `0.00100` → 3 dp)

Because this is **futures/perpetual**, we keep the chapter’s canonical ID:

```python
from nautilus_trader.model import InstrumentId, BarType

INSTRUMENT_ID = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")
BAR_TYPE_1M = BarType.from_str("BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL")
```

**Precision constants (from your preview):**

- `price_precision = 2`  
- `size_precision  = 3`  
- `price_increment = 0.01`  
- `size_increment  = 0.001`

> If your raw futures files show trailing zeros like `69903.60000000`, that is still consistent with **2 decimal** tick size; the formatter is just printing extra zeros.

#### Instrument object (pseudo-code)

The exact constructor varies by Nautilus version, but the intent is invariant: define a `CryptoPerpetual` (or equivalent) with the above precisions and currencies.

```python
from __future__ import annotations

from nautilus_trader.model import InstrumentId, Symbol, Venue
from nautilus_trader.model.currencies import BTC, USDT
from nautilus_trader.model.instruments.crypto_perpetual import CryptoPerpetual
from nautilus_trader.model.objects import Price, Quantity

def make_btcusdt_perp_binance_instrument(ts_event: int, ts_init: int) -> CryptoPerpetual:
    """
    Minimal backtest instrument definition.

    Notes:
    - price_precision=2 because prices look like 69850.53
    - size_precision=3 because quantities look like 0.001
    """
    venue = Venue("BINANCE")
    symbol = Symbol("BTCUSDT-PERP")
    instrument_id = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")

    return CryptoPerpetual(
        instrument_id=instrument_id,
        symbol=symbol,
        venue=venue,
        base_currency=BTC,
        quote_currency=USDT,
        settlement_currency=USDT,
        price_precision=2,
        size_precision=3,
        price_increment=Price.from_str("0.01"),
        size_increment=Quantity.from_str("0.001"),
        ts_event=ts_event,
        ts_init=ts_init,
    )
```

---

### 4.2.3 Build the catalog at `data/catalog/`

This is the reproducible offline step you run once.

```python
from __future__ import annotations

from pathlib import Path

from nautilus_trader.model import InstrumentId, BarType

def build_catalog_main() -> None:
    raw_root = Path("data/raw")
    catalog_root = Path("data/catalog")

    instrument_id = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")
    bar_type_1m = BarType.from_str("BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL")

    csv_wrangler = BinanceFuturesCSVWrangler(
        instrument_id=instrument_id,
        bar_type_1m=bar_type_1m,
    )

    build_parquet_catalog_from_csv(
        raw_root=raw_root,
        catalog_root=catalog_root,
        wrangler=csv_wrangler,
        batch_size_ticks=250_000,
        batch_size_bars=100_000,
    )

if __name__ == "__main__":
    build_catalog_main()
```

---

### 4.2.4 BacktestNode configuration (May 2023 run_config)

Below is a “wiring script” that:

1. Computes `start/end` from filenames (May 2023).
2. Instantiates the backtest node.
3. Registers venue + instrument.
4. Adds the Chapter 2 strategy + execution algorithm.
5. Runs the node.

> **Important:** Nautilus has both “low-level” (`BacktestEngine`) and “high-level” (`BacktestNode`) APIs.  
> You asked for `BacktestNode` + `ParquetDataCatalog`, so we wire those.

```python
from __future__ import annotations

import datetime as dt
from pathlib import Path

from nautilus_trader.model import InstrumentId, BarType
from nautilus_trader.model.objects import Money

# High-level backtest API (module exists in your docs: nautilus_trader.backtest.node)
from nautilus_trader.backtest.node import BacktestNode
from nautilus_trader.config import (
    BacktestNodeConfig,
    BacktestRunConfig,
    BacktestEngineConfig,
    CacheConfig,
    LoggingConfig,
)

from nautilus_trader.model.enums import OmsType, AccountType
from nautilus_trader.model.currencies import USDT

# Your Chapter 2 components (you provide these modules)
from strategies.alpha_strategy import AlphaStrategy, AlphaStrategyConfig
from execution.adaptive_twap import AdaptiveTWAPExecAlgorithm


def run_backtest_main() -> None:
    raw_root = Path("data/raw")
    catalog_root = Path("data/catalog")

    # Derive run window from May 2023 files
    start_dt, end_dt = compute_backtest_window_from_raw(raw_root)
    # Example: 2023-05-16 00:00:00+00:00 -> 2023-06-01 00:00:00+00:00

    instrument_id = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")
    bar_type_1m = BarType.from_str("BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL")

    # Backtest run configuration (explicit dates)
    run_config = BacktestRunConfig(
        start=start_dt,
        end=end_dt,
    )

    # Engine-level configuration
    engine_config = BacktestEngineConfig(
        logging=LoggingConfig(log_level="INFO"),
        cache=CacheConfig(
            tick_capacity=2_000_000,  # adjust based on memory
            bar_capacity=200_000,
        ),
    )

    node_config = BacktestNodeConfig(
        run=run_config,
        engine=engine_config,
        catalog_path=str(catalog_root),
    )

    node = BacktestNode(config=node_config)

    # --------------------------
    # Venue + instrument wiring
    # --------------------------
    # Create a minimal instrument definition at backtest start time
    ts0 = int(start_dt.timestamp() * 1e9)
    instrument = make_btcusdt_perp_binance_instrument(ts_event=ts0, ts_init=ts0)

    node.add_instrument(instrument)

    node.add_venue(
        venue=instrument_id.venue,  # "BINANCE"
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USDT,
        starting_balances=[Money.from_str("100000 USDT")],
        # fee_model=...,  # optional
    )

    # --------------------------
    # Strategy + execution algo
    # --------------------------
    strat_config = AlphaStrategyConfig(
        instrument_id=instrument_id,
        bar_type_1m=bar_type_1m,
        vpin_bucket_volume="25.0",   # BTC units (example; calibrate)
        vpin_window_buckets=50,
        yz_window_days=14,
        yz_min_days=5,
        max_spread_bps=3.0,
        min_top_size="1.0",
        vpin_threshold=0.65,
        # Chapter 3 sizing params would also live here
    )
    strategy = AlphaStrategy(config=strat_config)

    exec_algo = AdaptiveTWAPExecAlgorithm()
    node.add_exec_algorithm(exec_algo)

    node.add_strategy(strategy)

    # Build + run
    node.build()
    result = node.run()
    node.dispose()

    # result / engine access varies; keep it explicit for tearsheet below
    engine = result.engine  # pseudo: adapt to your BacktestNode result object

    # Tearsheet output
    from nautilus_trader.analysis.tearsheet import create_tearsheet
    create_tearsheet(engine=engine, output_path="tearsheet_may_2023.html")


if __name__ == "__main__":
    run_backtest_main()
```

**Notes on correctness and safety:**

- If any component relies on instrument specs (precision/increments), define them **before** replay.
- Keep **one node per process** (Nautilus global singleton constraints; see architecture docs).

---

## 4.3 Execution Flow Diagram (ASCII, end-to-end lifecycle)

This diagram is intentionally “high fidelity”: it includes ingestion, node init, data replay ordering, and the full execution chain.

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ 1) INGEST (Offline ETL)                                                      │
└──────────────────────────────────────────────────────────────────────────────┘

   data/raw/futures/daily/aggTrades/ BTCUSDT-aggTrades-2023-05-16.csv  ...
   data/raw/futures/daily/bookTicker/ BTCUSDT-bookTicker-2023-05-16.csv ...
   data/raw/futures/daily/klines_1m/   BTCUSDT-1m-2023-05-16.csv        ...
               │
               │ (stream tokens; ms->ns; schema map)
               ▼
┌───────────────────────────────┐
│ BinanceFuturesCSVWrangler      │
│ - AggTrades  -> TradeTick      │
│   * is_buyer_maker -> Aggressor│
│ - BookTicker -> QuoteTick      │
│ - Klines 1m  -> Bar            │
│ - ts_event = ms * 1,000,000    │
└───────────────┬───────────────┘
                │ (batch into Arrow RecordBatches)
                ▼
┌───────────────────────────────┐
│ ParquetDataCatalog (Arrow)     │
│ path = data/catalog/           │
│ - write(TradeTick batches)     │
│ - write(QuoteTick batches)     │
│ - write(Bar batches)           │
└────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────────────────┐
│ 2) INITIALIZATION (BacktestNode / NautilusKernel wiring)                      │
└──────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────┐
│ BacktestNode                  │
│ - loads BacktestNodeConfig    │
│   * run.start / run.end       │
│   * engine config             │
│   * catalog_path              │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ NautilusKernel                │
│ - initializes MessageBus      │
│ - initializes Cache           │
│ - initializes DataEngine      │
│ - initializes RiskEngine      │
│ - initializes ExecutionEngine │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Load instrument + venue        │
│ - CryptoPerpetual BTCUSDT-PERP │
│ - precision: px=2, qty=3       │
│ - starting balances            │
└────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────────────────┐
│ 3) EVENT LOOP (Replay -> Strategy -> Risk -> Orders)                          │
└──────────────────────────────────────────────────────────────────────────────┘

 ParquetDataCatalog
   │
   │  (DataEngine replays & merges streams by ts_event)
   ▼
┌───────────────────────────────┐
│ DataEngine                     │
│ - orders events by ts_event ns │
│ - routes by InstrumentId /     │
│   BarType                      │
└───────────────┬───────────────┘
                │ subscriptions
                ▼
┌───────────────────────────────┐
│ AlphaStrategy (Actor/Strategy) │
│ on_trade_tick: VPIN update     │
│ on_quote_tick: liquidity state │
│ on_bar: RS/YZ update           │
│                                 │
│ -> produces Z_t, sigma_t         │
│ -> computes Q_target             │
│ -> reconcile with Q_actual       │
│ -> emits trading commands        │
└───────────────┬───────────────┘
                │ SubmitOrder (exec_algorithm_id="TWAP_ADAPTIVE")
                ▼
┌───────────────────────────────┐
│ ExecAlgorithm (AdaptiveTWAP)   │
│ - consumes VolatilitySignal    │
│ - adjusts pacing / pauses      │
│ - spawns child orders          │
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ RiskEngine                      │
│ - validates qty/price/notional   │
│ - enforces caps + fail-fast      │
│ - may DENY invalid orders        │
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ ExecutionEngine                 │
│ - accepts validated commands     │
│ - updates order lifecycle        │
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ Backtest Execution Client       │
│ - simulated fills / matching     │
│ - emits OrderFilled events       │
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ Portfolio + Cache update         │
│ - positions, PnL, exposures      │
│ - analyzer collects returns      │
└────────────────────────────────┘
```

---

## 4.4 Performance Metrics & Tearsheet

### 4.4.1 PortfolioAnalyzer metrics (formulas)

NautilusTrader’s `PortfolioAnalyzer` computes returns series and standard performance statistics. Here are the canonical mathematical definitions you requested (assuming a return series \(r_t\) sampled at a fixed frequency, and optionally an annualization factor \(A\), e.g. \(A=252\) for trading days or \(A=365\) for crypto daily, or \(A=525{,}600\) for minutes).

#### Sharpe Ratio

Let \(r_t\) be periodic returns and \(r_f\) be the per-period risk-free return (often 0 in crypto backtests unless you model it).

\[
\mathrm{Sharpe}
=
\sqrt{A}\;
\frac{\mathbb{E}[r_t - r_f]}{\sqrt{\mathrm{Var}(r_t - r_f)}}
\]

Implementation detail (discrete sample estimate):

- Numerator: \(\bar{r} = \frac{1}{T}\sum_{t=1}^T (r_t-r_f)\)
- Denominator: sample std dev \(s = \sqrt{\frac{1}{T-1}\sum ( (r_t-r_f) - \bar{r})^2}\)

#### Sortino Ratio

Define **downside deviation** relative to a target (typically \(0\)):

\[
\mathrm{DD}
=
\sqrt{\mathbb{E}\left[\min(r_t - r_{\text{target}}, 0)^2\right]}
\]

Then:

\[
\mathrm{Sortino}
=
\sqrt{A}\;
\frac{\mathbb{E}[r_t - r_{\text{target}}]}{\mathrm{DD}}
\]

#### Max Drawdown (MDD)

Let equity curve \(E_t\) (or cumulative return curve) define running peak:

\[
P_t = \max_{s \le t} E_s
\]

Drawdown at time \(t\):

\[
D_t = \frac{E_t}{P_t} - 1
\]

Max drawdown:

\[
\mathrm{MDD} = \min_t D_t
\]

(So it is a negative number; some reports present \(|\mathrm{MDD}|\).)

---

### 4.4.2 Generate HTML tearsheet (Plotly) from backtest engine

Nautilus provides Plotly tearsheets via `nautilus_trader.analysis.tearsheet.create_tearsheet`.

Minimal snippet (post-run):

```python
from __future__ import annotations

from nautilus_trader.analysis.tearsheet import create_tearsheet

def write_tearsheet(engine, output_path: str = "tearsheet.html") -> None:
    """
    Write an interactive HTML tearsheet for a completed backtest.

    `engine` should be the BacktestEngine instance used by your BacktestNode.
    """
    create_tearsheet(
        engine=engine,
        output_path=output_path,
    )
```

Optional customization (charts/theme) is also supported via `TearsheetConfig` (see visualization docs), but the above is the canonical “one-liner” used in most workflows.

---

## Practical checklist (so Chapter 4 runs end‑to‑end)

1. **Run ETL once**  
   - Input: `data/raw/futures/daily/...`  
   - Output: `data/catalog/` (Parquet)
2. **Define instrument + venue**  
   - `InstrumentId`: `BTCUSDT-PERP.BINANCE`  
   - `price_precision=2`, `size_precision=3`  
3. **BacktestNode run window**  
   - `2023-05-16T00:00:00Z` → `2023-06-01T00:00:00Z`
4. **Attach strategy & exec algo**  
   - `AlphaStrategy` (Chapter 2 logic)  
   - `AdaptiveTWAPExecAlgorithm` (signal-aware execution)
5. **Run** → **generate tearsheet**

If you paste the exact NautilusTrader version you are using (and whether you are on v1 Cython package or the newer v2 `python/` PyO3 package), I can tighten the BacktestNode configuration section to match your exact class names/fields **without** any “pseudo” assumptions.