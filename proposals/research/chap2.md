# Chapter 2 — NautilusTrader Architecture (CSV → DataEngine → Alpha → Execution)

This chapter specifies the **exact event-driven translation** of Chapter 1’s alpha components into NautilusTrader:

- **Factor 1:** \( \sigma^2_{\mathrm{YZ}} \) (Yang–Zhang volatility) computed from **1-minute bars**.  
- **Factor 2:** **VPIN** (flow toxicity) computed from **tick-level `aggTrades`** (treated as trade ticks).  
- **State:** \( \mathcal{L} \) (liquidity state) computed from **`bookTicker`** (top-of-book quotes).

**Strict dataset constraints (local CSV only):**

- Trades: `data/raw/futures/daily/aggTrades/BTCUSDT-aggTrades-*.csv`  
- Quotes: `data/raw/futures/daily/bookTicker/BTCUSDT-bookTicker-*.csv`  
- Bars: `data/raw/futures/daily/klines_1m/BTCUSDT-1m-*.csv`

The architecture is built around Nautilus’ deterministic, event-driven runtime:

```
CSV files → (wrangle) → ParquetDataCatalog → BacktestNode/DataEngine → Strategy callbacks
                                                   │
                                                   └→ MessageBus custom signals → ExecAlgorithm
```

---

## 2.1 Custom data loading & wrangling (Binance CSV → Nautilus objects)

### 2.1.1 Canonical instrument and bar identifiers

We will standardize on:

```python
from nautilus_trader.model import InstrumentId, BarType

INSTRUMENT_ID = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")
BAR_TYPE_1M = BarType.from_str("BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL")
```

> **Why this matters:**  
> The `InstrumentId` and `BarType` form the **routing keys** for the DataEngine and Cache. Any mismatch breaks ingestion determinism and downstream indicator/strategy wiring.

---

### 2.1.2 Schema mapping tables

#### A) `bookTicker` (quotes) → `QuoteTick`

Raw file example (futures):  
`data/raw/futures/daily/bookTicker/BTCUSDT-bookTicker-YYYY-MM-DD.csv`

| Raw CSV column          | Type     | Nautilus field                          | Notes |
|-------------------------|----------|------------------------------------------|------|
| `best_bid_price`        | decimal  | `QuoteTick.bid_price`                    | Convert to `Price`. |
| `best_bid_qty`          | decimal  | `QuoteTick.bid_size`                     | Convert to `Quantity`. |
| `best_ask_price`        | decimal  | `QuoteTick.ask_price`                    | Convert to `Price`. |
| `best_ask_qty`          | decimal  | `QuoteTick.ask_size`                     | Convert to `Quantity`. |
| `event_time`            | int (ms) | `QuoteTick.ts_event` (ns)                | **Primary event clock** for quotes. |
| `transaction_time`      | int (ms) | (optional) `QuoteTick.info["tx_time"]`   | Keep for debugging/reconciliation. |
| `update_id`             | int      | (optional) `QuoteTick.info["update_id"]` | Useful for ordering audits. |

**Timestamp contract:**  
`ts_event = event_time * 1_000_000` (ms → ns)  
`ts_init = ts_event` in backtests (or clock time in live ingestion).

---

#### B) `aggTrades` (trades) → `TradeTick` (or TradeTick-like)

Raw file example (futures):  
`data/raw/futures/daily/aggTrades/BTCUSDT-aggTrades-YYYY-MM-DD.csv`

| Raw CSV column       | Type     | Nautilus field                                 | Notes |
|----------------------|----------|-----------------------------------------------|------|
| `price`              | decimal  | `TradeTick.price`                              | Convert to `Price`. |
| `quantity`           | decimal  | `TradeTick.size` / `TradeTick.quantity`        | Convert to `Quantity`. |
| `transact_time`      | int (ms) | `TradeTick.ts_event` (ns)                      | **Primary event clock** for trades. |
| `agg_trade_id`       | int      | `TradeTick.trade_id` (or `info["agg_id"]`)     | Treat as unique tick identifier. |
| `is_buyer_maker`     | bool     | `TradeTick.aggressor_side`                     | See aggressor mapping below. |
| `first_trade_id`     | int      | `info["first_trade_id"]`                       | Optional. |
| `last_trade_id`      | int      | `info["last_trade_id"]`                        | Optional. |

**Aggressor mapping (Chapter 1, §1.4.1):**

Let `BM = is_buyer_maker`.

- `BM == True` → buyer is maker → seller is taker → **aggressor is SELL**  
- `BM == False` → buyer is taker → **aggressor is BUY**

This directly maps into the trade sign \(s_i\):

\[
s_i =
\begin{cases}
+1 & \text{if } \mathrm{BM}_i=0 \ (\text{BUY initiated}) \\
-1 & \text{if } \mathrm{BM}_i=1 \ (\text{SELL initiated})
\end{cases}
\]

---

#### C) `klines_1m` (bars) → `Bar`

Raw file example (futures):  
`data/raw/futures/daily/klines_1m/BTCUSDT-1m-YYYY-MM-DD.csv`

| Raw CSV column     | Type     | Nautilus field          | Notes |
|--------------------|----------|-------------------------|------|
| `open`             | decimal  | `Bar.open`              | Price. |
| `high`             | decimal  | `Bar.high`              | Price. |
| `low`              | decimal  | `Bar.low`               | Price. |
| `close`            | decimal  | `Bar.close`             | Price. |
| `volume`           | decimal  | `Bar.volume`            | Quantity (base units). |
| `close_time`       | int (ms) | `Bar.ts_event` (ns)     | We timestamp bars on **close**. |
| `open_time`        | int (ms) | (optional) `info[...]`  | Used only if you need open-time alignment. |
| `quote_volume`     | decimal  | (optional) `info[...]`  | Optional. |
| `count`            | int      | (optional) `info[...]`  | Optional. |
| `taker_buy_volume` | decimal  | (optional) `info[...]`  | Optional. |

**Bar timestamping rule:**  
Use `close_time` for `ts_event` so `on_bar` is causally “after the minute completes”.

---

### 2.1.3 Wrangling implementation (robust CSV parsing, daily globbing, batch writing)

You have two viable approaches:

1. **Convert CSV → ParquetDataCatalog** (recommended for backtests and repeated research).  
2. **Stream CSV directly into a BacktestEngine** (good for quick experiments; harder to reuse).

This spec implements **(1)**.

#### 2.1.3.1 CSV reader utilities (delimiter + header detection)

Your sample futures CSVs show both headered whitespace-delimited outputs (and occasionally malformed lines). We must parse defensively.

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import csv


def _detect_dialect(path: Path) -> csv.Dialect:
    """
    Detect whether file is comma-separated or whitespace-separated.
    We avoid pandas here because tick files are large and we want streaming.
    """
    sample = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:5]
    sample = [ln for ln in sample if ln.strip()]
    if not sample:
        raise ValueError(f"Empty file: {path}")

    # Heuristic: if commas appear in first non-empty line, treat as CSV
    if "," in sample[0]:
        class _Comma(csv.Dialect):
            delimiter = ","
            quotechar = '"'
            doublequote = True
            skipinitialspace = False
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL
        return _Comma()

    # Otherwise treat as "split on whitespace" (not a true csv dialect)
    # We'll handle whitespace separately.
    return None  # sentinel


def _iter_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield ln


def _looks_like_header(fields: list[str]) -> bool:
    # Header likely contains non-numeric tokens like 'open_time' or 'agg_trade_id'
    for tok in fields:
        if any(c.isalpha() for c in tok):
            return True
    return False


def iter_rows(path: Path) -> Iterator[list[str]]:
    """
    Yields tokenized rows as list[str], skipping header and malformed lines.
    """
    dialect = _detect_dialect(path)

    if dialect is not None:
        reader = csv.reader(path.open("r", encoding="utf-8", errors="ignore"), dialect=dialect)
        for row in reader:
            row = [x.strip() for x in row]
            if not row or all(not x for x in row):
                continue
            if _looks_like_header(row):
                continue
            yield row
        return

    # Whitespace tokenization
    first = True
    for ln in _iter_lines(path):
        row = ln.split()
        if first and _looks_like_header(row):
            first = False
            continue
        first = False

        # Skip lines that are obviously broken (e.g., single gigantic integer)
        if len(row) < 4:
            continue

        yield row
```

---

#### 2.1.3.2 Wranglers: CSV rows → Nautilus data objects

Below is a **code-spec** style wrangler layer. The key design decision is that these wranglers emit *typed* Nautilus objects with a strict timestamp contract.

> **Note on exact constructors:**  
> Depending on your NautilusTrader version, `QuoteTick`, `TradeTick`, and `Bar` constructors may differ slightly. The architecture is invariant: you must produce those objects with correct `instrument_id`, correct numeric types, and correct `ts_event`.

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from nautilus_trader.model import InstrumentId, BarType
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.enums import AggressorSide

from nautilus_trader.model.data import QuoteTick, TradeTick, Bar  # adjust imports if needed


NS_PER_MS = 1_000_000


@dataclass(frozen=True)
class BinanceFuturesBookTickerRow:
    update_id: int
    bid_px: str
    bid_qty: str
    ask_px: str
    ask_qty: str
    tx_time_ms: int
    event_time_ms: int


def parse_bookticker_row(row: list[str]) -> Optional[BinanceFuturesBookTickerRow]:
    """
    Expected futures bookTicker columns (per your dataset preview):
      update_id best_bid_price best_bid_qty best_ask_price best_ask_qty transaction_time event_time
    """
    if len(row) < 7:
        return None
    try:
        return BinanceFuturesBookTickerRow(
            update_id=int(row[0]),
            bid_px=row[1],
            bid_qty=row[2],
            ask_px=row[3],
            ask_qty=row[4],
            tx_time_ms=int(row[5]),
            event_time_ms=int(row[6]),
        )
    except ValueError:
        return None


def iter_quote_ticks_from_bookticker_csv(
    path: Path,
    instrument_id: InstrumentId,
) -> Iterator[QuoteTick]:
    for row in iter_rows(path):
        parsed = parse_bookticker_row(row)
        if parsed is None:
            continue

        ts_event = parsed.event_time_ms * NS_PER_MS
        ts_init = ts_event  # backtest: init = event time

        yield QuoteTick(
            instrument_id=instrument_id,
            bid_price=Price.from_str(parsed.bid_px),
            ask_price=Price.from_str(parsed.ask_px),
            bid_size=Quantity.from_str(parsed.bid_qty),
            ask_size=Quantity.from_str(parsed.ask_qty),
            ts_event=ts_event,
            ts_init=ts_init,
            # Optional: keep raw ordering metadata for audits
            # info={"update_id": parsed.update_id, "tx_time_ms": parsed.tx_time_ms},
        )
```

AggTrades → TradeTick:

```python
from dataclasses import dataclass
from typing import Optional, Iterator
from pathlib import Path

from nautilus_trader.model import InstrumentId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.data import TradeTick


@dataclass(frozen=True)
class BinanceFuturesAggTradeRow:
    agg_trade_id: int
    price: str
    qty: str
    first_trade_id: int
    last_trade_id: int
    transact_time_ms: int
    is_buyer_maker: bool


def _parse_bool(tok: str) -> bool:
    tok = tok.strip().lower()
    if tok in ("true", "1", "t"):
        return True
    if tok in ("false", "0", "f"):
        return False
    raise ValueError(f"Invalid bool token: {tok}")


def parse_aggtrade_row(row: list[str]) -> Optional[BinanceFuturesAggTradeRow]:
    """
    Expected futures aggTrades columns (per your dataset preview):
      agg_trade_id price quantity first_trade_id last_trade_id transact_time is_buyer_maker
    """
    if len(row) < 7:
        return None
    try:
        return BinanceFuturesAggTradeRow(
            agg_trade_id=int(row[0]),
            price=row[1],
            qty=row[2],
            first_trade_id=int(row[3]),
            last_trade_id=int(row[4]),
            transact_time_ms=int(row[5]),
            is_buyer_maker=_parse_bool(row[6]),
        )
    except ValueError:
        return None


def aggressor_side_from_is_buyer_maker(is_buyer_maker: bool) -> AggressorSide:
    # Chapter 1 sign convention:
    # is_buyer_maker == True => seller-initiated => aggressor SELL
    return AggressorSide.SELL if is_buyer_maker else AggressorSide.BUY


def iter_trade_ticks_from_aggtrades_csv(
    path: Path,
    instrument_id: InstrumentId,
) -> Iterator[TradeTick]:
    for row in iter_rows(path):
        parsed = parse_aggtrade_row(row)
        if parsed is None:
            continue

        ts_event = parsed.transact_time_ms * NS_PER_MS
        ts_init = ts_event

        yield TradeTick(
            instrument_id=instrument_id,
            price=Price.from_str(parsed.price),
            size=Quantity.from_str(parsed.qty),
            aggressor_side=aggressor_side_from_is_buyer_maker(parsed.is_buyer_maker),
            trade_id=str(parsed.agg_trade_id),  # or int, depends on model
            ts_event=ts_event,
            ts_init=ts_init,
            # Optional:
            # info={
            #     "first_trade_id": parsed.first_trade_id,
            #     "last_trade_id": parsed.last_trade_id,
            #     "is_buyer_maker": parsed.is_buyer_maker,
            # },
        )
```

Klines 1m → Bar:

```python
from dataclasses import dataclass
from typing import Optional, Iterator
from pathlib import Path

from nautilus_trader.model import BarType
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.data import Bar


@dataclass(frozen=True)
class BinanceFuturesKline1mRow:
    open_time_ms: int
    open_: str
    high: str
    low: str
    close: str
    volume: str
    close_time_ms: int


def parse_kline_1m_row(row: list[str]) -> Optional[BinanceFuturesKline1mRow]:
    """
    Expected futures klines columns (per your dataset preview):
      open_time open high low close volume close_time quote_volume count taker_buy_volume taker_buy_quote_volume ignore
    """
    if len(row) < 7:
        return None
    try:
        return BinanceFuturesKline1mRow(
            open_time_ms=int(row[0]),
            open_=row[1],
            high=row[2],
            low=row[3],
            close=row[4],
            volume=row[5],
            close_time_ms=int(row[6]),
        )
    except ValueError:
        return None


def iter_bars_from_klines_1m_csv(
    path: Path,
    bar_type: BarType,
) -> Iterator[Bar]:
    for row in iter_rows(path):
        parsed = parse_kline_1m_row(row)
        if parsed is None:
            continue

        ts_event = parsed.close_time_ms * NS_PER_MS
        ts_init = ts_event

        yield Bar(
            bar_type=bar_type,
            open=Price.from_str(parsed.open_),
            high=Price.from_str(parsed.high),
            low=Price.from_str(parsed.low),
            close=Price.from_str(parsed.close),
            volume=Quantity.from_str(parsed.volume),
            ts_event=ts_event,
            ts_init=ts_init,
        )
```

---

#### 2.1.3.3 Writing into a Parquet Data Catalog (batch pipeline)

**Design goal:** Don’t load full-day tick files into memory. Use **batch writes**.

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from nautilus_trader.persistence.wranglers import (
    QuoteTickDataWrangler,
    TradeTickDataWrangler,
    BarDataWrangler,
)

from nautilus_trader.model import InstrumentId, BarType


@dataclass(frozen=True)
class IngestSpec:
    instrument_id: InstrumentId
    bar_type_1m: BarType
    raw_root: Path
    catalog_root: Path
    batch_size: int = 250_000  # trades/quotes can be huge; tune for RAM


def _glob_sorted(pattern: str) -> list[Path]:
    paths = [Path(p) for p in sorted(Path().glob(pattern))]
    return paths


def ingest_binance_futures_csv_to_catalog(spec: IngestSpec) -> None:
    """
    Convert local Binance CSV files into the Nautilus ParquetDataCatalog.

    This function is intentionally "offline" (run once) to build a reusable catalog.
    """
    catalog = ParquetDataCatalog(path=spec.catalog_root)

    # Built-in wranglers take Data objects and serialize them for Parquet storage
    q_wrangler = QuoteTickDataWrangler()
    t_wrangler = TradeTickDataWrangler()
    b_wrangler = BarDataWrangler()

    # --- Quotes ---
    quote_files = sorted((spec.raw_root / "futures" / "daily" / "bookTicker").glob("BTCUSDT-bookTicker-*.csv"))
    for fp in quote_files:
        buf = []
        for qt in iter_quote_ticks_from_bookticker_csv(fp, spec.instrument_id):
            buf.append(qt)
            if len(buf) >= spec.batch_size:
                catalog.write(q_wrangler.to_record_batch(buf))
                buf.clear()
        if buf:
            catalog.write(q_wrangler.to_record_batch(buf))

    # --- Trades (aggTrades treated as TradeTicks) ---
    trade_files = sorted((spec.raw_root / "futures" / "daily" / "aggTrades").glob("BTCUSDT-aggTrades-*.csv"))
    for fp in trade_files:
        buf = []
        for tt in iter_trade_ticks_from_aggtrades_csv(fp, spec.instrument_id):
            buf.append(tt)
            if len(buf) >= spec.batch_size:
                catalog.write(t_wrangler.to_record_batch(buf))
                buf.clear()
        if buf:
            catalog.write(t_wrangler.to_record_batch(buf))

    # --- Bars (1m) ---
    bar_files = sorted((spec.raw_root / "futures" / "daily" / "klines_1m").glob("BTCUSDT-1m-*.csv"))
    for fp in bar_files:
        buf = []
        for bar in iter_bars_from_klines_1m_csv(fp, spec.bar_type_1m):
            buf.append(bar)
            if len(buf) >= 100_000:  # bars are smaller
                catalog.write(b_wrangler.to_record_batch(buf))
                buf.clear()
        if buf:
            catalog.write(b_wrangler.to_record_batch(buf))
```

**Key engineering invariants:**

- `ts_event` is always nanoseconds UTC.  
- `ts_event` monotonicity should hold within each dataset stream.  
- The **catalog is append-only** in ingestion; de-duplication should be handled before ingestion if needed.

---

### 2.1.4 AggTrades vs TradeTick: practical interpretation

Binance `aggTrades` aggregates multiple fills into one record. For our VPIN pipeline:

- We treat each `aggTrades` row as a **single signed volume impulse** \((s_i, q_i)\).
- If one record’s quantity spans multiple VPIN volume buckets, **we must split it** across buckets (implemented in §2.3.2).

This is consistent with VPIN being defined over **volume-synchronized buckets** rather than strict tick counts.

---

## 2.2 The strategy actor (`AlphaStrategy`)

### 2.2.1 Configuration object

We define a typed config so the same strategy runs in backtest and live (even though this chapter uses local CSV).

```python
from __future__ import annotations

from decimal import Decimal
from typing import Optional

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model import InstrumentId, BarType


class AlphaStrategyConfig(StrategyConfig):
    instrument_id: InstrumentId
    bar_type_1m: BarType

    # VPIN parameters
    vpin_bucket_volume: Decimal          # V_b in base units (e.g., BTC)
    vpin_window_buckets: int = 50        # rolling n

    # Yang–Zhang parameters
    yz_window_days: int = 14             # m sessions
    yz_min_days: int = 5                 # warmup gate

    # Liquidity state parameters
    max_spread_bps: float = 3.0
    min_top_size: Decimal = Decimal("1.0")  # contracts/BTC units depending on instrument sizing

    # Risk / gating thresholds
    vpin_threshold: float = 0.65
```

---

### 2.2.2 Strategy skeleton and subscriptions (`on_start`)

```python
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal
import math
from typing import Deque, Optional

from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model import InstrumentId, BarType
from nautilus_trader.model.data import Bar, QuoteTick, TradeTick
from nautilus_trader.model.enums import AggressorSide


@dataclass
class LiquidityState:
    mid: float = math.nan
    spread: float = math.nan
    spread_bps: float = math.nan
    bid_size: float = 0.0
    ask_size: float = 0.0

    # derived regime flags
    is_low_liquidity: bool = False


@dataclass
class VPINState:
    # bucket parameters
    bucket_target: Decimal
    bucket_remaining: Decimal
    buy_vol: Decimal = Decimal("0")
    sell_vol: Decimal = Decimal("0")

    # rolling window
    window: Deque[float] = None
    sum_imbalance: float = 0.0  # sum of I_k in last n buckets


@dataclass
class YZState:
    # current session (UTC day) accumulators
    current_day: Optional[str] = None  # YYYY-MM-DD
    day_open: Optional[float] = None
    day_close: Optional[float] = None
    rs_sum: float = 0.0

    # previous day close for boundary return r_o
    prev_day_close: Optional[float] = None

    # rolling windows (length m days)
    ro: Deque[float] = None   # r^(o)_d
    rc: Deque[float] = None   # r^(c)_d
    rs: Deque[float] = None   # RS_d

    # last computed outputs
    yz_var: float = math.nan


class AlphaStrategy(Strategy):
    def __init__(self, config: AlphaStrategyConfig) -> None:
        super().__init__(config)

        self.instrument_id: InstrumentId = config.instrument_id
        self.bar_type_1m: BarType = config.bar_type_1m

        # Runtime state
        self.liq = LiquidityState()

        self.vpin = VPINState(
            bucket_target=config.vpin_bucket_volume,
            bucket_remaining=config.vpin_bucket_volume,
            window=deque(maxlen=config.vpin_window_buckets),
        )

        self.yz = YZState(
            ro=deque(maxlen=config.yz_window_days),
            rc=deque(maxlen=config.yz_window_days),
            rs=deque(maxlen=config.yz_window_days),
        )

        # last computed VPIN
        self.vpin_value: float = math.nan

    def on_start(self) -> None:
        # Ensure instrument exists (in live/backtest it should come from provider/catalog)
        instrument = self.cache.instrument(self.instrument_id)
        if instrument is None:
            self.log.error(f"Instrument not found in cache: {self.instrument_id}")
            self.stop()
            return

        # Subscribe to all three required streams
        self.subscribe_bars(self.bar_type_1m)
        self.subscribe_quote_ticks(self.instrument_id)
        self.subscribe_trade_ticks(self.instrument_id)

        self.log.info(
            "AlphaStrategy started | "
            f"instrument_id={self.instrument_id} | "
            f"bar_type={self.bar_type_1m}"
        )
```

**State management principle:**  
Every callback updates *only* O(1) incremental state. No pandas in the hot loop.

---

## 2.3 Event-driven logic (the “hot loop”)

### 2.3.1 `on_bar`: Yang–Zhang update (and why it should not block)

**Math → code mapping (Chapter 1):**

Per-bar Rogers–Satchell contribution:

- \( u_k = \log(H_k/O_k) \)
- \( d_k = \log(L_k/O_k) \)
- \( c_k = \log(C_k/O_k) \)
- \( \widehat{\sigma}^2_{\mathrm{RS},k} = u_k(u_k - c_k) + d_k(d_k - c_k) \)

We compute this in **O(1)** per bar, then only compute full YZ when a UTC day boundary is detected.

```python
import pandas as pd  # only for timestamp formatting if needed (optional)


def _utc_day_from_ns(ts_event_ns: int) -> str:
    # Fast path: avoid pandas in hot loop if possible.
    # In production, implement integer-based day partitioning.
    import datetime
    dt = datetime.datetime.utcfromtimestamp(ts_event_ns / 1e9)
    return dt.strftime("%Y-%m-%d")


def _safe_log(x: float) -> float:
    return math.log(x) if x > 0.0 else float("nan")


class AlphaStrategy(Strategy):
    # ... (init/on_start above) ...

    def on_bar(self, bar: Bar) -> None:
        # Bar is 1-minute. We timestamped on close, so this event is "end of minute".

        o = float(bar.open)
        h = float(bar.high)
        l = float(bar.low)
        c = float(bar.close)

        # Chapter 1, Definition 3 + 4 (RS)
        u = _safe_log(h / o)
        d = _safe_log(l / o)
        cc = _safe_log(c / o)
        rs_k = u * (u - cc) + d * (d - cc)

        day = _utc_day_from_ns(bar.ts_event)

        if self.yz.current_day is None:
            # Start first session
            self.yz.current_day = day
            self.yz.day_open = o
            self.yz.day_close = c
            self.yz.rs_sum = rs_k
            return

        if day == self.yz.current_day:
            # Same UTC session: accumulate
            self.yz.day_close = c
            self.yz.rs_sum += rs_k
            return

        # --- Day rollover: finalize previous session d-1, start new session d ---
        self._finalize_yz_session_and_roll(day_open=o, day_close=c, new_day=day, rs_first=rs_k)

    def _finalize_yz_session_and_roll(self, day_open: float, day_close: float, new_day: str, rs_first: float) -> None:
        # Finalize previous day stats
        prev_open = self.yz.day_open
        prev_close = self.yz.day_close
        prev_rs = self.yz.rs_sum

        if prev_open is None or prev_close is None:
            # defensive (should never happen)
            self.yz.current_day = new_day
            self.yz.day_open = day_open
            self.yz.day_close = day_close
            self.yz.rs_sum = rs_first
            return

        # Chapter 1, Def 6: boundary return r^(o)_d = log(O_d / C_{d-1})
        # Here "O_d" is new session open, "C_{d-1}" is prev_close.
        r_o = math.log(day_open / prev_close) if (day_open > 0 and prev_close > 0) else float("nan")

        # Chapter 1, Def 6: within-session return r^(c)_{d-1} = log(C_{d-1} / O_{d-1})
        r_c = math.log(prev_close / prev_open) if (prev_close > 0 and prev_open > 0) else float("nan")

        # RS_{d-1} already aggregated as sum of per-minute RS contributions
        rs_d = prev_rs

        # Push into rolling windows
        self.yz.ro.append(r_o)
        self.yz.rc.append(r_c)
        self.yz.rs.append(rs_d)

        self.yz.prev_day_close = prev_close

        # Compute YZ only if we have enough sessions
        if len(self.yz.ro) >= self.config.yz_min_days:
            self.yz.yz_var = self._compute_yang_zhang_var()

            # Publish signal if you want regime updates only at day boundaries
            self._maybe_publish_vol_signal(ts_event_ns=None)  # implement below

        # Start new session accumulators
        self.yz.current_day = new_day
        self.yz.day_open = day_open
        self.yz.day_close = day_close
        self.yz.rs_sum = rs_first

    def _sample_var(self, xs: list[float]) -> float:
        # Simple (but stable enough for small m). For large m use Welford.
        xs = [x for x in xs if math.isfinite(x)]
        n = len(xs)
        if n <= 1:
            return float("nan")
        mu = sum(xs) / n
        return sum((x - mu) ** 2 for x in xs) / (n - 1)

    def _compute_yang_zhang_var(self) -> float:
        """
        Chapter 1, Definition 7:
          σ^2_YZ = σ^2_o + k σ^2_c + (1-k) σ^2_RS
        where:
          σ^2_o = var(r_o), σ^2_c = var(r_c), σ^2_RS = mean(RS_d)
        """
        ro = list(self.yz.ro)
        rc = list(self.yz.rc)
        rs = list(self.yz.rs)

        m = len(ro)
        if m < 2:
            return float("nan")

        var_o = self._sample_var(ro)
        var_c = self._sample_var(rc)
        rs_mean = sum(rs) / m

        # Standard YZ weight (Yang & Zhang 2000):
        # k = 0.34 / (1.34 + (m+1)/(m-1))
        k = 0.34 / (1.34 + (m + 1) / (m - 1))

        return var_o + k * var_c + (1.0 - k) * rs_mean
```

#### Is this “heavy”?

Not if you implement it this way:

- Per bar: just a few logs and multiplications.  
- Per day: a small rolling window variance computation (m ~ 14–30).  
- No matrix algebra, no pandas, no recalculation across history.

If you *do* want to recompute intraday (e.g., every minute), use:

- rolling window deques for RS contributions (minute-level),
- and incremental variance (Welford) for \(r_o, r_c\).

---

### 2.3.2 `on_trade_tick`: VPIN bucket logic (volume-time)

**Math → code mapping (Chapter 1):**

Volume buckets of size \(V_b\). For bucket \(j\):

- \(V^+_j\): buy-initiated volume  
- \(V^-_j\): sell-initiated volume  
- \(I_j = |V^+_j - V^-_j|\)

VPIN at time \(j\):

\[
\mathrm{VPIN}_j = \frac{1}{nV_b}\sum_{k=j-n+1}^j I_k
\]

We implement it in **O(1)** amortized per trade tick using:

- current bucket accumulators,
- rolling deque of imbalances,
- running sum of imbalances.

**Critical detail:** an `aggTrades` tick can be larger than the remaining bucket capacity. We must **split** the tick across buckets.

```python
class AlphaStrategy(Strategy):
    # ... prior code ...

    def on_trade_tick(self, tick: TradeTick) -> None:
        """
        Hot loop path: update VPIN.

        We assume:
          - tick.size is base-volume (BTC units for BTCUSDT perpetual).
          - tick.aggressor_side inferred from is_buyer_maker at ingestion time.
        """
        qty = Decimal(str(tick.size))
        if qty <= 0:
            return

        is_buy = (tick.aggressor_side == AggressorSide.BUY)

        # Consume quantity possibly across multiple buckets
        while qty > 0:
            take = min(qty, self.vpin.bucket_remaining)

            if is_buy:
                self.vpin.buy_vol += take
            else:
                self.vpin.sell_vol += take

            self.vpin.bucket_remaining -= take
            qty -= take

            if self.vpin.bucket_remaining == 0:
                # Finalize bucket
                self._close_vpin_bucket()

                # Reset bucket
                self.vpin.bucket_remaining = self.vpin.bucket_target
                self.vpin.buy_vol = Decimal("0")
                self.vpin.sell_vol = Decimal("0")

        # Optional: if VPIN breaches, publish signal immediately (trade-time risk gate)
        if math.isfinite(self.vpin_value) and self.vpin_value > self.config.vpin_threshold:
            self._publish_vpin_toxicity_signal(ts_event_ns=tick.ts_event)

    def _close_vpin_bucket(self) -> None:
        """
        Implements Chapter 1 Definition 10 + 11:
          I_j = |V^+ - V^-|
          VPIN = (sum I_j) / (n * V_b)
        """
        v_plus = float(self.vpin.buy_vol)
        v_minus = float(self.vpin.sell_vol)
        imbalance = abs(v_plus - v_minus)

        # Update rolling window (O(1))
        if len(self.vpin.window) == self.vpin.window.maxlen:
            oldest = self.vpin.window[0]
            self.vpin.sum_imbalance -= oldest

        self.vpin.window.append(imbalance)
        self.vpin.sum_imbalance += imbalance

        n = len(self.vpin.window)
        denom = n * float(self.vpin.bucket_target)
        self.vpin_value = (self.vpin.sum_imbalance / denom) if denom > 0 else float("nan")
```

---

### 2.3.3 `on_quote_tick`: liquidity state \( \mathcal{L} \) and synchronization with trades

We derive a minimal \( \mathcal{L} \) from **top-of-book**:

- Midprice \(m_t = \frac{b_t + a_t}{2}\)
- Spread \(s_t = a_t - b_t\)
- Spread in bps: \( \text{spread\_bps} = 10^4 \cdot s_t / m_t \)
- Top sizes: \(q^{bid}_t, q^{ask}_t\)

Then classify low liquidity if:

- `spread_bps > max_spread_bps` **OR**
- `min(bid_size, ask_size) < min_top_size`

```python
class AlphaStrategy(Strategy):
    # ... prior code ...

    def on_quote_tick(self, quote: QuoteTick) -> None:
        bid = float(quote.bid_price)
        ask = float(quote.ask_price)
        if bid <= 0 or ask <= 0 or ask < bid:
            return

        mid = 0.5 * (bid + ask)
        spread = ask - bid
        spread_bps = 10_000.0 * (spread / mid)

        bid_sz = float(quote.bid_size)
        ask_sz = float(quote.ask_size)

        self.liq.mid = mid
        self.liq.spread = spread
        self.liq.spread_bps = spread_bps
        self.liq.bid_size = bid_sz
        self.liq.ask_size = ask_sz

        min_top = float(self.config.min_top_size)
        self.liq.is_low_liquidity = (spread_bps > self.config.max_spread_bps) or (min(bid_sz, ask_sz) < min_top)

        # Optional: publish a liquidity regime change signal
        # (only if state flips to reduce MessageBus noise)
```

#### Synchronizing QuoteTick and TradeTick

**Do not attempt to “join” streams by wall-clock order.**  
Instead:

- Treat `ts_event` as the global ordering key.
- In backtests, the DataEngine will deliver data in **timestamp order** (subject to ties).
- Maintain **last-known quote state** and use it for the next trade tick. This is the correct microstructure stance: trades occur *against* the prevailing/just-updated book.

**Tie-handling rule:** If `QuoteTick.ts_event == TradeTick.ts_event`, the ordering may vary. Your strategy must be robust:

- Use the most recent cached quote.
- If quote missing, fall back to price-only logic.

---

## 2.4 Signal generation & execution routing

### 2.4.1 Custom signal: `VolatilitySignal(Data)`

We publish a *single* canonical event that downstream components consume.

```python
from __future__ import annotations

from enum import Enum
from typing import Optional

from nautilus_trader.core.data import Data
from nautilus_trader.model.custom import customdataclass
from nautilus_trader.model import InstrumentId


class Regime(str, Enum):
    NORMAL = "NORMAL"
    HIGH_VARIANCE = "HIGH_VARIANCE"
    HIGH_TOXICITY = "HIGH_TOXICITY"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"


@customdataclass
class VolatilitySignal(Data):
    instrument_id: InstrumentId

    yz_var: float
    vpin: float
    spread_bps: float

    regime: Regime

    # Required by Data contract
    ts_event: int
    ts_init: int
```

Publishing from the strategy:

```python
class AlphaStrategy(Strategy):
    # ...

    def _classify_regime(self) -> Regime:
        if self.liq.is_low_liquidity:
            return Regime.LOW_LIQUIDITY
        if math.isfinite(self.vpin_value) and self.vpin_value > self.config.vpin_threshold:
            return Regime.HIGH_TOXICITY
        # Example variance gate (you will calibrate this)
        if math.isfinite(self.yz.yz_var) and self.yz.yz_var > 1e-4:
            return Regime.HIGH_VARIANCE
        return Regime.NORMAL

    def _publish_vpin_toxicity_signal(self, ts_event_ns: int) -> None:
        self._publish_signal(ts_event_ns)

    def _maybe_publish_vol_signal(self, ts_event_ns: Optional[int]) -> None:
        # For day-boundary YZ updates, you may publish at the final bar's ts_event
        if ts_event_ns is None:
            return

        self._publish_signal(ts_event_ns)

    def _publish_signal(self, ts_event_ns: int) -> None:
        regime = self._classify_regime()

        sig = VolatilitySignal(
            instrument_id=self.instrument_id,
            yz_var=float(self.yz.yz_var) if math.isfinite(self.yz.yz_var) else float("nan"),
            vpin=float(self.vpin_value) if math.isfinite(self.vpin_value) else float("nan"),
            spread_bps=float(self.liq.spread_bps) if math.isfinite(self.liq.spread_bps) else float("nan"),
            regime=regime,
            ts_event=ts_event_ns,
            ts_init=ts_event_ns,
        )
        self.publish_data(VolatilitySignal, sig)
```

---

### 2.4.2 Execution algorithm subscription and adaptive TWAP routing

We implement an execution algorithm that:

- Subscribes to `VolatilitySignal`.
- Maintains per-instrument execution parameters:
  - `interval_secs` (smaller = faster execution)
  - `paused` flag (stop spawning while liquidity is unacceptable)

#### Adaptive TWAP algorithm (spec)

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from nautilus_trader.execution.algorithm import ExecAlgorithm
from nautilus_trader.model.orders.base import Order
from nautilus_trader.model import InstrumentId

import pandas as pd


@dataclass
class AdaptiveParams:
    interval_secs: float = 2.5
    horizon_secs: float = 20.0
    paused: bool = False


class AdaptiveTWAPExecAlgorithm(ExecAlgorithm):
    """
    Execution algorithm that adapts to VolatilitySignal regimes.

    - HIGH_VARIANCE  => shorten interval (execute faster)
    - LOW_LIQUIDITY  => pause spawning (do not place child orders)
    """
    def __init__(self) -> None:
        super().__init__()
        self.params_by_instrument: Dict[InstrumentId, AdaptiveParams] = {}

    def on_start(self) -> None:
        self.subscribe_data(VolatilitySignal)

    def on_data(self, data) -> None:
        if not isinstance(data, VolatilitySignal):
            return

        p = self.params_by_instrument.get(data.instrument_id, AdaptiveParams())

        match data.regime:
            case Regime.HIGH_VARIANCE:
                # Execute faster: smaller interval, same horizon (or shorten horizon too)
                p.interval_secs = 1.0
                p.paused = False
            case Regime.LOW_LIQUIDITY:
                # Pause: do not spawn new child orders
                p.paused = True
            case Regime.HIGH_TOXICITY:
                # Conservative: slightly slower, or switch to passive, or reduce size
                p.interval_secs = 5.0
                p.paused = False
            case _:
                p.interval_secs = 2.5
                p.paused = False

        self.params_by_instrument[data.instrument_id] = p

    def on_order(self, order: Order) -> None:
        """
        Primary order arrives here when strategy submits with exec_algorithm_id="TWAP_ADAPTIVE".
        """
        instrument_id = order.instrument_id
        p = self.params_by_instrument.get(instrument_id, AdaptiveParams())

        if p.paused:
            # Do not execute; you can either:
            #  (a) hold the order and retry later via timer, or
            #  (b) cancel/deny internally (strategy policy choice).
            self.log.warning(f"AdaptiveTWAP paused for {instrument_id}, holding order {order.client_order_id}")
            self.clock.set_timer(
                name=f"retry_{order.client_order_id}",
                interval=pd.Timedelta(seconds=1),
                callback=lambda _: self.on_order(order),
            )
            return

        # Standard TWAP: spawn smaller child orders at interval
        # NOTE: pseudo-API; exact spawn_* signatures depend on Nautilus version.
        interval = pd.Timedelta(seconds=p.interval_secs)
        horizon = pd.Timedelta(seconds=p.horizon_secs)

        # Minimal example: split into N slices
        n_slices = max(1, int(p.horizon_secs / p.interval_secs))
        child_qty = order.quantity / n_slices

        for k in range(n_slices):
            self.clock.set_time_alert(
                name=f"spawn_{order.client_order_id}_{k}",
                alert_time=self.clock.utc_now() + k * interval,
                callback=lambda _, q=child_qty: self.spawn_market(order, quantity=q),
            )
```

#### Strategy routing to the exec algorithm

In the strategy, you route orders by specifying the algorithm ID:

```python
from nautilus_trader.model import ExecAlgorithmId
from nautilus_trader.model.enums import OrderSide

class AlphaStrategy(Strategy):
    # ...

    def _submit_target_order(self, qty) -> None:
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.BUY,
            quantity=qty,
            exec_algorithm_id=ExecAlgorithmId("TWAP_ADAPTIVE"),
        )
        self.submit_order(order)
```

---

## ASCII sequence diagram: TradeTick end-to-end (CSV → VPIN → order)

Below is the **single-tick** path you must preserve.

```
┌──────────────────────────────┐
│ CSV file                      │
│ data/raw/futures/daily/...     │
│ BTCUSDT-aggTrades-YYYY-MM-DD   │
└───────────────┬───────────────┘
                │ iter_rows() + parse_aggtrade_row()
                ▼
┌──────────────────────────────┐
│ Wrangler                      │
│ aggTrades → TradeTick         │
│ - ts_event = transact_time    │
│ - aggressor_side from BM      │
└───────────────┬───────────────┘
                │ write → ParquetDataCatalog
                ▼
┌──────────────────────────────┐
│ DataEngine (backtest replay)  │
│ - merges streams by ts_event  │
│ - routes by InstrumentId      │
└───────────────┬───────────────┘
                │ subscription: subscribe_trade_ticks(INSTRUMENT_ID)
                ▼
┌──────────────────────────────┐
│ AlphaStrategy.on_trade_tick   │
│ - update VPIN bucket          │
│ - if bucket closes:           │
│     I_j = |V+ - V-|           │
│     VPIN = Σ I / (n*Vb)       │
│ - if VPIN > threshold:        │
│     publish VolatilitySignal  │
└───────────────┬───────────────┘
                │ MessageBus (Actor publish_data)
                ▼
┌──────────────────────────────┐
│ AdaptiveTWAPExecAlgorithm     │
│ - on_data(VolatilitySignal)   │
│ - adjust interval / paused    │
└───────────────┬───────────────┘
                │ strategy submits order w/ exec_algorithm_id
                ▼
┌──────────────────────────────┐
│ Execution flow                │
│ Strategy → ExecAlgorithm      │
│ → RiskEngine → ExecutionEngine│
│ → ExecutionClient             │
└──────────────────────────────┘
```

---

## Summary: the math-to-code placement (explicit)

- **RS / YZ volatility (Chapter 1 §1.3):**  
  Implemented in `AlphaStrategy.on_bar` + `_compute_yang_zhang_var`.  
  - RS per minute: `rs_k = u*(u-c) + d*(d-c)`  
  - YZ per day/window: `var_o + k*var_c + (1-k)*rs_mean`

- **VPIN (Chapter 1 §1.4):**  
  Implemented in `AlphaStrategy.on_trade_tick` + `_close_vpin_bucket`.  
  - Bucket fill (volume clock): `bucket_remaining` decrement  
  - Imbalance \(I_j\): `abs(buy_vol - sell_vol)`  
  - VPIN: `sum_imbalance / (n * V_b)`

- **Liquidity state \( \mathcal{L} \):**  
  Implemented in `AlphaStrategy.on_quote_tick`.  
  - Spread bps: `10_000 * (ask-bid)/mid`  
  - Low-liquidity gate: spread and/or top size thresholds

- **Signal routing:**  
  `VolatilitySignal` is published from `AlphaStrategy` and consumed by `AdaptiveTWAPExecAlgorithm` via Actor data subscriptions.

---

If you want, I can extend Chapter 2 with a **full backtest node wiring** (catalog creation → node config → adding the strategy + exec algorithm), including the exact directory layout under a Parquet catalog and the minimal instrument definition required for `BTCUSDT-PERP.BINANCE` to be valid in backtest mode.