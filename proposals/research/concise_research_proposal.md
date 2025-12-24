```
## 1) Input processing (read + outline)

### Short internal outline of the provided proposal (Ch. 1–4 + review)
- **Chapter 1 (Alpha / microstructure):** Latent price + microstructure noise framing; volatility + flow/toxicity style factors; signal fusion idea.
- **Chapter 2 (Architecture):** CSV → typed Nautilus objects → Parquet catalog → event-driven strategy callbacks (`on_bar`, `on_quote_tick`, `on_trade_tick`).
- **Chapter 3 (Risk):** Position sizing + liquidity gates + fail-safes; target vs actual reconciliation.
- **Chapter 4 (Glue / backtest):** Deterministic ETL, catalog build, backtest node wiring, tearsheet metrics.
- **Review section:** Flags missing end-to-end alpha definition, timestamp semantics/cost realism, and perps-specific costs (esp. funding).

### Re-stating your core constraints (before research)
- **Strategy type:** *Medium-frequency* crypto strategy (minutes-to-hours, not sub-second HFT).
- **Core signal:** **Liquidity Pressure Index (LPI)** built from **volume + spread + order book imbalance**.
- **Implementation target:** **`nautilus_trader`** event-driven backtests (local files / ParquetDataCatalog).

---

## 2) Web research results (what exists + what is defensible)

### 2.1 “Liquidity Pressure Index” / close equivalents in the wild
There isn’t a single canonical, universally accepted “Liquidity Pressure Index” in crypto academia; instead, practitioners and data vendors use closely related constructs:

- **Order Book Pressure** (vendor definition):  
  *Order_Book_Pressure = bid depth − ask depth* (directional; positive = buy-side pressure). ([docs.amberdata.io](https://docs.amberdata.io/data-dictionary/analytics/derivatives/order-book-pressure))

- **Normalized Order Book Imbalance (NOBI)** (academic, crypto-exchange liquidity study):  
  Defines imbalance as depth difference (ask − bid) and normalizes by total depth to map into **[-1, 1]**, with discussion that **deeper levels are more informative than just L1** and that **5‑minute snapshots** reduce top-of-book noise. ([mdpi.com](https://www.mdpi.com/1911-8074/18/3/124?utm_source=openai))

- **Weighted “liquidity pressure” / decay-by-distance** (trading platform docs):  
  A common practitioner approach is **depth weighted by distance** from best prices using an exponential decay. This is conceptually useful if you have multi-level depth. ([help.atas.net](https://help.atas.net/en/support/solutions/articles/72000648682-dom-pressure?utm_source=openai))

- **Microprice family (spread + imbalance → “fair price”)**:  
  Stoikov’s micro-price is explicitly an **adjustment to mid-price using bid–ask spread and imbalance**, with evidence it can forecast short-term prices better than mid/weighted mid. ([papers.ssrn.com](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694))  
  This is not “LPI” by name, but it is exactly the *spread + imbalance* core you want (volume is typically used as a regime/strength filter).

### 2.2 Evidence that imbalance/spread/volume contain predictive information (but horizons matter)
- **Order flow imbalance (OFI) explains short-horizon price changes** in a robust, linear way, with slope inversely related to market depth (classic microstructure result). ([arxiv.org](https://arxiv.org/abs/1011.6402?utm_source=openai))  
- **Queue/quote imbalance predicts the next mid-price move direction** (statistically significant in equities; mechanism generalizes). ([arxiv.org](https://arxiv.org/abs/1512.03492?utm_source=openai))  
- **Medium-frequency tradability (non-crypto evidence):** intraday order imbalance can predict returns from **1 to 90 minutes** and trading strategies can be profitable in that context (Chinese equities). ([sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S0927538X15300056?utm_source=openai))  
- **Crypto-specific caution on “how far ahead”:** at least for *Bitcoin order book gaps*, the gap/return “causality” can be mostly **instantaneous** (i.e., predictive power decays fast). ([arxiv.org](https://arxiv.org/abs/2106.02187?utm_source=openai))  
- **Crypto risk forecasting usage:** order imbalance has been used as a predictor in **Bitcoin crash-risk nowcasting** (showing information content, even if not directly “alpha”). ([pmc.ncbi.nlm.nih.gov](https://pmc.ncbi.nlm.nih.gov/articles/PMC10040314/?utm_source=openai))

### 2.3 Existing implementations you can learn from
- **NautilusTrader** supports `QuoteTick`, `TradeTick`, `Bar`, and multiple order book granularities (L1/L2/L3), with the backtest pipeline designed to work on these data types. ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/data?utm_source=openai))  
- Nautilus provides **order book backtest tutorials** (including Binance order book workflows) and strategy patterns (`on_quote_tick`, `on_bar`, etc.). ([nautilustrader.io](https://nautilustrader.io/docs/nightly/tutorials/backtest_binance_orderbook))  
- **Open-source crypto imbalance backtesting code** exists (BitMEX/Binance Futures imbalance strategies): `nkaz001/algotrading-example` (moved to `hftbacktest`). ([github.com](https://github.com/nkaz001/algotrading-example?utm_source=openai))  

---

## 3) Revised proposal (tight, medium-frequency LPI strategy)

### 3.1 Objective
Build and evaluate a **medium-frequency BTCUSDT perpetual** strategy whose decisions are driven by a **Liquidity Pressure Index** that combines:
- **Order book imbalance** (direction)
- **Bid–ask spread** (execution cost / liquidity regime)
- **Volume** (signal strength / confirmation)

### 3.2 Hypothesis (testable)
1. **Directional pressure:** persistent buy-side (sell-side) imbalance increases the probability of **positive (negative)** short-horizon mid-price changes. ([arxiv.org](https://arxiv.org/abs/1011.6402?utm_source=openai))  
2. **Tradeability filter:** the signal is more tradeable when **spread is tight** (lower friction), and less tradeable / higher slippage risk when spread widens. ([mdpi.com](https://www.mdpi.com/1911-8074/18/3/124?utm_source=openai))  
3. **Volume confirmation:** imbalance signals are more reliable during **above-normal volume** (higher participation), and more spoofable/noisy during thin volume. (This is a standard microstructure intuition; volume itself is often a conditioning variable rather than “the alpha”.)

### 3.3 Data requirements (matches your local folder structure)
Minimum viable dataset (futures):
- **Quotes:** `data/raw/futures/daily/bookTicker/BTCUSDT-bookTicker-YYYY-MM-DD.csv`
- **Bars:** `data/raw/futures/daily/klines_1m/BTCUSDT-1m-YYYY-MM-DD.csv`

Optional (Phase 2, deeper imbalance):
- **Depth:** `data/raw/futures/daily/bookDepth/BTCUSDT-bookDepth-YYYY-MM-DD.csv`  
  (Only if it truly contains *bid-vs-ask depth* at symmetric offsets; your preview looks like “depth at ±% levels”, which can be used to construct a depth-based imbalance.)

### 3.4 LPI definition (explicit, implementable)
Let at decision time \(t\) (here: each 1‑minute bar close):

**(A) Spread (basis points)**  
\[
\text{spread\_bps}_t = 10^4 \cdot \frac{a_t - b_t}{(a_t+b_t)/2}
\]

**(B) L1 imbalance (top-of-book)**  
\[
\text{imb}_t=\frac{q^b_t - q^a_t}{q^b_t + q^a_t + \epsilon}\in[-1,1]
\]

**(C) Volume surprise (per 1m bar)**  
Let \(v_t\) be the bar volume; define \(x_t=\log(1+v_t)\). Maintain rolling mean/std over \(N\) minutes:
\[
z^{(v)}_t = \frac{x_t-\mu_x}{\sigma_x+\epsilon}
\]

**(D) Spread surprise**  
Rolling z-score on spread:
\[
z^{(s)}_t = \frac{\text{spread\_bps}_t-\mu_s}{\sigma_s+\epsilon}
\]

**(E) Liquidity Pressure Index (directional)**
A simple, medium-frequency-friendly form:
\[
\text{LPI}_t = \text{imb}_t \cdot \max(0, z^{(v)}_t)\; \Big/\; \left(1+\max(0, z^{(s)}_t)\right)
\]
- Imbalance sets **direction**.
- Volume only **amplifies** when unusually high.
- Wide spreads **penalize** trading.

> Why this form is defensible: it’s consistent with (i) imbalance predicting short-horizon moves ([arxiv.org](https://arxiv.org/abs/1512.03492?utm_source=openai)) and (ii) spreads being a primary liquidity/cost dimension. ([ecb.europa.eu](https://www.ecb.europa.eu/press/economic-bulletin/html/eb201802.en.html?utm_source=openai))

### 3.5 Trading rule (medium frequency)
- Decision clock: **1-minute** (bar close).  
- **Entry:**  
  - If `spread_bps_t > max_spread_bps`: do **not** enter; optionally reduce to flat.
  - If `LPI_t > +T_entry`: target **LONG**.
  - If `LPI_t < -T_entry`: target **SHORT**.
- **Exit:**  
  - If `|LPI_t| < T_exit`: flatten.
  - If spread widens beyond gate: flatten.

### 3.6 Execution & costs (must be explicit)
- Medium-frequency execution can be **market orders** (simple) or **top-of-book limit orders** (more complex).  
- In backtests, you must include:
  - bid/ask spread crossing (quotes),
  - fees,
  - and (for perps) a funding policy (either model funding explicitly or enforce “flat at funding timestamps”).  
Nautilus supports realistic backtesting across different data types and fill models; configure accordingly. ([nautilustrader.io](https://nautilustrader.io/docs/latest/api_reference/backtest/))

---

## 4) Profitability (evidence-based, no guarantees)

### What the research supports
- **Predictive content exists:**  
  Order-flow/queue imbalance and related signals have repeatedly shown statistical predictability for short-horizon price changes. ([arxiv.org](https://arxiv.org/abs/1011.6402?utm_source=openai))

- **Medium-frequency tradability is plausible but not automatic:**  
  There is at least non-crypto evidence that intraday imbalances can predict returns up to ~90 minutes and yield profitable strategies in that market context. ([sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S0927538X15300056?utm_source=openai))

### What pushes against profitability (especially in crypto perps)
- **Horizon decay:** crypto order-book relationships (e.g., gaps→returns) can be mostly instantaneous, which means a “medium-frequency” implementation may lose the edge if it reacts too slowly. ([arxiv.org](https://arxiv.org/abs/2106.02187?utm_source=openai))  
- **Costs dominate easily:** imbalance signals often move *within* the spread or are too small versus fees/spread unless you have a cost-aware execution design; this is also consistent with microprice intuition (predicts short-term price direction but lives inside bid/ask). ([papers.ssrn.com](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694))  
- **Data realism matters:** top-of-book is noisy; deeper depth and 5-minute snapshots can be more stable in some crypto liquidity analyses. ([mdpi.com](https://www.mdpi.com/1911-8074/18/3/124?utm_source=openai))  

### High-risk self-check (explicit)
- I am **not** assuming this is profitable by default.  
- Any suggested windows/thresholds below are **starting points** (common practitioner defaults), not “proven optimal.”  
- Profitability must be demonstrated **after** fees + spread + slippage + funding policy in your exact dataset and venue model.

---

## 5) Implementation guide (NautilusTrader, minimal + idiomatic)

### 5.1 Data ingestion (CSV → `ParquetDataCatalog`)
Nautilus supports writing core market data via `ParquetDataCatalog.write_data(...)`. ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/data?utm_source=openai))

```python
# ingest_lpi_data.py
from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Iterator, Optional

from nautilus_trader.model import BarType, InstrumentId
from nautilus_trader.model.data import Bar, QuoteTick
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog

NS_PER_MS = 1_000_000


def ms_to_ns(ts_ms: int) -> int:
    return int(ts_ms) * NS_PER_MS


def iter_csv_rows(path: Path) -> Iterator[list[str]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # Skip header-like rows
            if any(any(ch.isalpha() for ch in tok) for tok in row):
                continue
            yield [tok.strip() for tok in row]


@dataclass(frozen=True)
class BookTickerRow:
    bid_px: str
    bid_qty: str
    ask_px: str
    ask_qty: str
    event_time_ms: int


def parse_bookticker(row: list[str]) -> Optional[BookTickerRow]:
    # Futures bookTicker preview: update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,transaction_time,event_time
    if len(row) < 7:
        return None
    try:
        return BookTickerRow(
            bid_px=row[1],
            bid_qty=row[2],
            ask_px=row[3],
            ask_qty=row[4],
            event_time_ms=int(row[6]),
        )
    except ValueError:
        return None


def iter_quote_ticks(path: Path, instrument_id: InstrumentId) -> Iterator[QuoteTick]:
    for row in iter_csv_rows(path):
        r = parse_bookticker(row)
        if r is None:
            continue
        ts_event = ms_to_ns(r.event_time_ms)
        yield QuoteTick(
            instrument_id=instrument_id,
            bid_price=Price.from_str(r.bid_px),
            ask_price=Price.from_str(r.ask_px),
            bid_size=Quantity.from_str(r.bid_qty),
            ask_size=Quantity.from_str(r.ask_qty),
            ts_event=ts_event,
            ts_init=ts_event,
        )


@dataclass(frozen=True)
class Kline1mRow:
    open_: str
    high: str
    low: str
    close: str
    volume: str
    close_time_ms: int


def parse_kline_1m(row: list[str]) -> Optional[Kline1mRow]:
    # Futures kline preview: open_time,open,high,low,close,volume,close_time,...
    if len(row) < 7:
        return None
    try:
        return Kline1mRow(
            open_=row[1],
            high=row[2],
            low=row[3],
            close=row[4],
            volume=row[5],
            close_time_ms=int(row[6]),
        )
    except ValueError:
        return None


def iter_bars_1m(path: Path, bar_type: BarType) -> Iterator[Bar]:
    for row in iter_csv_rows(path):
        r = parse_kline_1m(row)
        if r is None:
            continue
        ts_event = ms_to_ns(r.close_time_ms)  # timestamp bars on CLOSE
        yield Bar(
            bar_type=bar_type,
            open=Price.from_str(r.open_),
            high=Price.from_str(r.high),
            low=Price.from_str(r.low),
            close=Price.from_str(r.close),
            volume=Quantity.from_str(r.volume),
            ts_event=ts_event,
            ts_init=ts_event,
        )


def ingest_to_catalog(
    raw_root: Path,
    catalog_root: Path,
    instrument_id: InstrumentId,
    bar_type_1m: BarType,
    batch_size: int = 250_000,
) -> None:
    catalog = ParquetDataCatalog(str(catalog_root))

    # Quotes
    q_dir = raw_root / "futures" / "daily" / "bookTicker"
    for fp in sorted(q_dir.glob("BTCUSDT-bookTicker-*.csv")):
        buf: list[QuoteTick] = []
        for qt in iter_quote_ticks(fp, instrument_id):
            buf.append(qt)
            if len(buf) >= batch_size:
                catalog.write_data(buf, skip_disjoint_check=True)
                buf.clear()
        if buf:
            catalog.write_data(buf, skip_disjoint_check=True)

    # Bars
    b_dir = raw_root / "futures" / "daily" / "klines_1m"
    for fp in sorted(b_dir.glob("BTCUSDT-1m-*.csv")):
        buf_b: list[Bar] = []
        for bar in iter_bars_1m(fp, bar_type_1m):
            buf_b.append(bar)
            if len(buf_b) >= 100_000:
                catalog.write_data(buf_b, skip_disjoint_check=True)
                buf_b.clear()
        if buf_b:
            catalog.write_data(buf_b, skip_disjoint_check=True)


if __name__ == "__main__":
    RAW_ROOT = Path("data/raw")
    CATALOG_ROOT = Path("data/catalog")
    INSTRUMENT_ID = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")
    BAR_TYPE_1M = BarType.from_str("BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL")

    ingest_to_catalog(
        raw_root=RAW_ROOT,
        catalog_root=CATALOG_ROOT,
        instrument_id=INSTRUMENT_ID,
        bar_type_1m=BAR_TYPE_1M,
    )
```

- Writes **only what LPI needs**: `QuoteTick` (spread + imbalance) and 1m `Bar` (volume).
- Uses `write_data(...)` as the canonical catalog API. ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/data?utm_source=openai))
- Timestamps are **ms → ns** and bars are timestamped on **close**.
- Uses chunked writes to avoid loading full days into memory.
- `skip_disjoint_check=True` is used defensively if your files overlap. (Remove if your data is strictly disjoint.)

---

### 5.2 Strategy implementation (`LiquidityPressureStrategy`)
```python
# lpi_strategy.py
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from decimal import Decimal

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model import BarType, InstrumentId
from nautilus_trader.model.data import Bar, QuoteTick
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy


class LiquidityPressureConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal

    lookback_bars: int = 60
    entry_threshold: float = 1.0
    exit_threshold: float = 0.2
    max_spread_bps: float = 5.0  # hard liquidity gate


@dataclass
class RollingZ:
    window: int
    xs: deque[float]

    def __init__(self, window: int) -> None:
        self.window = int(window)
        self.xs = deque(maxlen=self.window)

    def update(self, x: float) -> None:
        if math.isfinite(x):
            self.xs.append(float(x))

    def mean_std(self) -> tuple[float, float]:
        n = len(self.xs)
        if n < 2:
            return (math.nan, math.nan)
        mu = sum(self.xs) / n
        var = sum((v - mu) ** 2 for v in self.xs) / (n - 1)
        return (mu, math.sqrt(max(0.0, var)))

    def z(self, x: float, eps: float = 1e-12) -> float:
        mu, sd = self.mean_std()
        if not math.isfinite(mu) or not math.isfinite(sd):
            return 0.0
        return (x - mu) / (sd + eps)


class LiquidityPressureStrategy(Strategy):
    def __init__(self, config: LiquidityPressureConfig) -> None:
        super().__init__(config)

        self.instrument: Instrument | None = None
        self._last_quote: QuoteTick | None = None

        self._vol_z = RollingZ(window=config.lookback_bars)
        self._spr_z = RollingZ(window=config.lookback_bars)

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Instrument not found: {self.config.instrument_id}")
            self.stop()
            return

        self.subscribe_quote_ticks(self.config.instrument_id)
        self.subscribe_bars(self.config.bar_type)

    def on_quote_tick(self, tick: QuoteTick) -> None:
        self._last_quote = tick  # keep last-known quote

    def on_bar(self, bar: Bar) -> None:
        if self.instrument is None:
            return
        if self._last_quote is None:
            return

        # ----- compute spread + imbalance from last quote -----
        bid = float(self._last_quote.bid_price)
        ask = float(self._last_quote.ask_price)
        if bid <= 0 or ask <= 0 or ask < bid:
            return

        mid = 0.5 * (bid + ask)
        spread_bps = 10_000.0 * ((ask - bid) / mid)

        bid_sz = float(self._last_quote.bid_size)
        ask_sz = float(self._last_quote.ask_size)
        imb = (bid_sz - ask_sz) / (bid_sz + ask_sz + 1e-12)  # [-1, 1] approx

        # Hard liquidity gate
        if spread_bps > float(self.config.max_spread_bps):
            self._flatten_if_needed()
            return

        # ----- volume feature from 1m bar -----
        v = float(bar.volume)
        x_vol = math.log1p(max(0.0, v))
        self._vol_z.update(x_vol)
        self._spr_z.update(spread_bps)

        z_vol = self._vol_z.z(x_vol)
        z_spr = self._spr_z.z(spread_bps)

        vol_boost = max(0.0, z_vol)
        spr_penalty = 1.0 + max(0.0, z_spr)

        lpi = imb * vol_boost / spr_penalty

        # ----- trade logic -----
        if lpi > float(self.config.entry_threshold):
            self._target_long()
        elif lpi < -float(self.config.entry_threshold):
            self._target_short()
        elif abs(lpi) < float(self.config.exit_threshold):
            self._flatten_if_needed()

    def _order_qty(self) -> Quantity:
        assert self.instrument is not None
        return self.instrument.make_qty(self.config.trade_size)

    def _target_long(self) -> None:
        if self.portfolio.is_flat(self.config.instrument_id):
            self._buy()
        elif self.portfolio.is_net_short(self.config.instrument_id):
            self.close_all_positions(self.config.instrument_id)
            self._buy()

    def _target_short(self) -> None:
        if self.portfolio.is_flat(self.config.instrument_id):
            self._sell()
        elif self.portfolio.is_net_long(self.config.instrument_id):
            self.close_all_positions(self.config.instrument_id)
            self._sell()

    def _flatten_if_needed(self) -> None:
        if not self.portfolio.is_flat(self.config.instrument_id):
            self.close_all_positions(self.config.instrument_id)

    def _buy(self) -> None:
        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self._order_qty(),
            time_in_force=TimeInForce.FOK,
        )
        self.submit_order(order)

    def _sell(self) -> None:
        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.SELL,
            quantity=self._order_qty(),
            time_in_force=TimeInForce.FOK,
        )
        self.submit_order(order)

    def on_stop(self) -> None:
        self.cancel_all_orders(self.config.instrument_id)
        self.close_all_positions(self.config.instrument_id)
        self.unsubscribe_bars(self.config.bar_type)
        self.unsubscribe_quote_ticks(self.config.instrument_id)
```

- Medium-frequency decisioning: **acts only on `on_bar`** (1-minute cadence).
- LPI uses **exactly**: volume (bar), spread (quote), imbalance (quote).
- Hard spread gate prevents trading in poor liquidity regimes.
- Position logic is *flat/long/short* only (no pyramiding).
- Uses market orders for simplicity; backtest realism depends on venue fill/fee config. ([nautilustrader.io](https://nautilustrader.io/docs/latest/api_reference/backtest/))

---

### 5.3 Backtest wiring (catalog → `BacktestNode`)
Below is the canonical pattern: `BacktestDataConfig` loads data from the catalog, then the node runs. ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/data?utm_source=openai))

```python
# run_lpi_backtest.py
from __future__ import annotations

from decimal import Decimal

from nautilus_trader.backtest.node import BacktestDataConfig, BacktestEngineConfig, BacktestNode, BacktestRunConfig, BacktestVenueConfig
from nautilus_trader.config import ImportableStrategyConfig, LoggingConfig
from nautilus_trader.model import BarType, InstrumentId, QuoteTick, Bar

# Your strategy module path
from lpi_strategy import LiquidityPressureConfig, LiquidityPressureStrategy


INSTRUMENT_ID = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")
BAR_TYPE_1M = BarType.from_str("BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL")
CATALOG_PATH = "data/catalog"

data = [
    BacktestDataConfig(
        catalog_path=CATALOG_PATH,
        data_cls=QuoteTick,
        instrument_id=str(INSTRUMENT_ID),
        start_time="2023-05-16",
        end_time="2023-06-01",
    ),
    BacktestDataConfig(
        catalog_path=CATALOG_PATH,
        data_cls=Bar,
        instrument_id=str(INSTRUMENT_ID),
        start_time="2023-05-16",
        end_time="2023-06-01",
        bar_type=str(BAR_TYPE_1M),
    ),
]

run = BacktestRunConfig(
    start="2023-05-16T00:00:00Z",
    end="2023-06-01T00:00:00Z",
    venues=[
        BacktestVenueConfig(
            name="BINANCE",
            oms_type="NETTING",
            account_type="MARGIN",
            base_currency="USDT",
        )
    ],
    data=data,
    strategies=[
        ImportableStrategyConfig(
            strategy_cls="lpi_strategy:LiquidityPressureStrategy",
            config=LiquidityPressureConfig(
                instrument_id=INSTRUMENT_ID,
                bar_type=BAR_TYPE_1M,
                trade_size=Decimal("0.01"),
                lookback_bars=60,
                entry_threshold=1.0,
                exit_threshold=0.2,
                max_spread_bps=5.0,
            ),
        )
    ],
)

engine = BacktestEngineConfig(
    logging=LoggingConfig(log_level="INFO"),
)

node = BacktestNode(config=BacktestNode.config_type(run=run, engine=engine))
node.run()
node.dispose()
```

- Uses the catalog as the single source of truth (`BacktestDataConfig`).
- Loads **QuoteTick** and **Bar** streams for the same instrument.
- Runs a single strategy instance with explicit parameters.
- You should still add: fee model + fill model + funding policy (outside this minimal code).
- Date range is set to your May 2023 sample window (adjust as needed).

---

## 6) What You would validate next (minimal research checklist)
- **Signal decay test:** IC / hit-rate of LPI for horizons {1m, 5m, 15m}. Expect decay; quantify. ([arxiv.org](https://arxiv.org/abs/2106.02187?utm_source=openai))  
- **Cost dominance test:** fees + spread + slippage; does net PnL survive? (Do not trust gross-only). ([nautilustrader.io](https://nautilustrader.io/docs/latest/api_reference/backtest/))  
- **Depth robustness:** if you can construct deeper imbalance (Depth10 / bookDepth), compare L1 vs deeper (academic evidence suggests deeper can be more robust). ([mdpi.com](https://www.mdpi.com/1911-8074/18/3/124?utm_source=openai))  
 

---
Learn more:
1. [Order Book Pressure - Amberdata Docs](https://docs.amberdata.io/data-dictionary/analytics/derivatives/order-book-pressure)
2. [Order Book Liquidity on Crypto Exchanges | MDPI](https://www.mdpi.com/1911-8074/18/3/124?utm_source=openai)
3. [DOM Pressure : ATAS](https://help.atas.net/en/support/solutions/articles/72000648682-dom-pressure?utm_source=openai)
4. [The Micro-Price: A High Frequency Estimator of Future Prices by Sasha Stoikov :: SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694)
5. [The Price Impact of Order Book Events](https://arxiv.org/abs/1011.6402?utm_source=openai)
6. [Queue Imbalance as a One-Tick-Ahead Price Predictor in a Limit Order Book](https://arxiv.org/abs/1512.03492?utm_source=openai)
7. [Do order imbalances predict Chinese stock returns? New evidence from intraday data - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0927538X15300056?utm_source=openai)
8. [Time-dependent relations between gaps and returns in a Bitcoin order book](https://arxiv.org/abs/2106.02187?utm_source=openai)
9. [Nowcasting bitcoin’s crash risk with order imbalance - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10040314/?utm_source=openai)
10. [Data | NautilusTrader Documentation](https://nautilustrader.io/docs/latest/concepts/data?utm_source=openai)
11. [Backtest: Binance OrderBook data | NautilusTrader Documentation](https://nautilustrader.io/docs/nightly/tutorials/backtest_binance_orderbook)
12. [GitHub - nkaz001/algotrading-example: algorithmic trading backtest and optimization examples using order book imbalances. (bitcoin, cryptocurrency, bitmex, binance futures, market making)](https://github.com/nkaz001/algotrading-example?utm_source=openai)
13. [Economic Bulletin Issue 2, 2018](https://www.ecb.europa.eu/press/economic-bulletin/html/eb201802.en.html?utm_source=openai)
14. [Backtest | NautilusTrader Documentation](https://nautilustrader.io/docs/latest/api_reference/backtest/)
```

```
## 0) Input processing (read + outline)

### Short internal outline of the provided proposal (the latest version you provided: `concise_research_proposal.md`)
1. **Goal & constraints:** medium-frequency BTCUSDT perpetual strategy; signal is a Liquidity Pressure Index (LPI) from **spread + L1 imbalance + volume**; implemented in `nautilus_trader`.
2. **Literature scan (lightweight):** LPI naming isn’t standardized; closest equivalents are order-book pressure/imbalance and microprice-type signals.
3. **Explicit LPI formula:**  
   - `spread_bps` from `bookTicker`  
   - `imb` from `bookTicker` sizes  
   - `z_vol` from `klines_1m` volume  
   - `z_spr` from rolling spread  
   - `LPI = imb * max(0, z_vol) / (1 + max(0, z_spr))`
4. **Trading rule:** trade only on **1-minute bar closes**, with a **hard spread gate**, entry/exit thresholds, and flat/long/short state machine.
5. **Implementation:** CSV → `ParquetDataCatalog`, then backtest with `BacktestNode`, `QuoteTick` + `Bar` streams.

### Core constraints (restated *before* research)
- **Signal class:** Liquidity Pressure Index built from **(i) top-of-book imbalance**, **(ii) bid–ask spread**, **(iii) 1m volume**.
- **Frequency:** **medium-frequency** (minutes-to-hours). The proposal’s concrete choice is **1-minute decisioning** (bar-close).
- **Platform:** **NautilusTrader** end-to-end:
  - ingest local Binance-style CSVs → `ParquetDataCatalog`
  - backtest via `BacktestNode`
  - strategy uses `on_quote_tick` + `on_bar` (no custom engine).

---

## 1) Web research (exhaustive-enough, cross-domain)

### 1.1 “Liquidity Pressure Index” in crypto: naming vs closest canonical constructs
There is **no single canonical** “Liquidity Pressure Index” term in crypto microstructure literature. What *does* exist (and is widely used) are close equivalents:

- **Order book pressure** as “bid depth − ask depth” (vendor definition; concept matches your “pressure” framing). ([docs.amberdata.io](https://docs.amberdata.io/data-dictionary/analytics/derivatives/order-book-pressure?utm_source=openai))  
- **Normalized order book imbalance (NOBI)** and depth-based imbalance comparisons in crypto liquidity studies; key empirical point: **deeper levels are often more informative than L1**.   
- **Microprice** family: explicit “spread + imbalance → short-horizon fair price” estimator (very close to your intended LPI core, but typically used at higher frequency). Stoikov (SSRN) and the companion repo are standard references.   

### 1.2 Evidence that imbalance/spread contain predictive information (but horizon decays)
Strong and commonly cited results (mostly equities, but mechanism is generic to LOBs):

- **OFI / imbalance at the best bid/ask** explains short-horizon price changes; the relation is close to linear and more robust than pure volume–return relations.   
- **Queue imbalance predicts the *direction* of the next mid-price move** (one-tick-ahead) with significant relationship in a logistic-regression framework.   

Crypto-specific “warning label” that matters for your *medium-frequency* design:

- For Bitcoin order book **gaps vs returns**, causal relation may be **largely instantaneous**; this is exactly the kind of result that can kill a 1-minute implementation if the edge decays before you act.   

### 1.3 Existing implementations (GitHub) relevant to “pressure/imbalance” strategies
Useful public references you can learn from (not proof of profitability, but proof of *implementability* and common patterns):

- Stoikov’s **microprice** reference implementation repo.   
- A repo that **calibrates microprice to BitMEX quote data** and uses it for market making / quoting logic (again, HFT-ish, but conceptually aligned).   
- `nkaz001/algotrading-example` (moved to `hftbacktest`): order book imbalance backtests for Bitcoin on BitMEX/Binance Futures.   

### 1.4 NautilusTrader “how to do it” references
For your exact requested pipeline (catalog → backtest node, and config objects):

- `ParquetDataCatalog.write_data(...)` and `BacktestDataConfig` usage are explicitly documented. ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/data/))  
- The “Loading external data” tutorial shows the canonical backtest wiring: `BacktestVenueConfig`, `BacktestDataConfig`, `ImportableStrategyConfig`, `BacktestRunConfig`, then `BacktestNode(configs=[...]).run()`.   

---

## 2) Revised proposal (tight + implementable)

### Title
**Medium-Frequency Liquidity Pressure Index (LPI) Strategy for BTCUSDT Perpetuals (Binance-style data) — NautilusTrader Implementation**

### 2.1 Objective
Build and test a **medium-frequency** strategy for BTCUSDT perpetuals where decisions are driven by an **LPI** combining:
- **Direction:** top-of-book imbalance (bid size vs ask size)
- **Cost/regime:** bid–ask spread (bps)
- **Strength:** 1-minute volume surprise

### 2.2 Hypothesis (testable, falsifiable)
1. **Directional pressure:** persistent L1 imbalance shifts the probability of the next short-horizon mid move. This is consistent with OFI/imbalance literature on LOBs.   
2. **Tradeability filter:** when spread widens, the same directional signal becomes less tradable (higher friction/adverse selection), so we penalize/disable trading under wide spreads (explicit gate).  
3. **Volume confirmation:** imbalance is more credible during unusually high volume; otherwise it is more likely to be noise/spoof-like (hence `max(0, z_vol)` amplification, not raw volume-as-alpha).

### 2.3 Data requirements (matches your local folder structure)
Minimum viable (futures/perps):
- **Top-of-book quotes:**  
  `data/raw/futures/daily/bookTicker/BTCUSDT-bookTicker-YYYY-MM-DD.csv`
- **1m bars:**  
  `data/raw/futures/daily/klines_1m/BTCUSDT-1m-YYYY-MM-DD.csv`

Optional (Phase 2, deeper pressure):
- **Depth / book depth summaries:**  
  `data/raw/futures/daily/bookDepth/...` (only if it contains *separable bid-vs-ask depth at symmetric offsets*; otherwise it’s not a clean imbalance input)

### 2.4 LPI definition (explicit)
At decision time \(t\) (use **bar close** timestamps):

**(A) Spread in bps**
\[
\text{spread\_bps}_t = 10^4 \cdot \frac{a_t - b_t}{(a_t+b_t)/2}
\]

**(B) L1 imbalance**
\[
\text{imb}_t = \frac{q^b_t - q^a_t}{q^b_t + q^a_t + \epsilon} \in [-1,1]
\]

**(C) Volume surprise**
Let \(v_t\) be 1m bar volume; define \(x_t=\log(1+v_t)\). Rolling z-score:
\[
z^{(v)}_t = \frac{x_t - \mu_x}{\sigma_x + \epsilon}
\]

**(D) Spread surprise**
Rolling z-score on `spread_bps`:
\[
z^{(s)}_t = \frac{\text{spread\_bps}_t - \mu_s}{\sigma_s + \epsilon}
\]

**(E) Liquidity Pressure Index**
\[
\text{LPI}_t
=
\text{imb}_t \cdot \max(0, z^{(v)}_t)
\;\Big/\;
\left(1 + \max(0, z^{(s)}_t)\right)
\]

**Rationale / provenance:**
- “Pressure = bid−ask liquidity” is a common industry definition (vendor metric). ([docs.amberdata.io](https://docs.amberdata.io/data-dictionary/analytics/derivatives/order-book-pressure?utm_source=openai))  
- “Imbalance predicts short-horizon direction” is well-established in LOB research.   
- “Microprice = spread + imbalance → future-price estimator” is a close conceptual cousin.   

### 2.5 Trading rule (medium-frequency, 1-minute cadence)
- **Decision clock:** each 1m bar close.
- **Hard liquidity gate:** if `spread_bps > max_spread_bps`, then **do not enter**, and optionally flatten.
- **Entry:**  
  - if `LPI_t > +T_entry` → target **LONG**  
  - if `LPI_t < -T_entry` → target **SHORT**
- **Exit:**  
  - if `|LPI_t| < T_exit` → **flatten**
  - if spread gate triggers → **flatten**
- **Position state machine:** {flat, long, short}; no pyramiding.

### 2.6 Profitability (evidence-based, no guarantees)
What the literature and public work supports:
- **Imbalance has statistical predictive content** at short horizons (especially very short horizons).   
- Microprice-style signals are explicitly designed to use imbalance+spread to estimate near-future “fair value”.   

What pushes against profitability **at 1-minute frequency**:
- Some Bitcoin order-book relationships (e.g., gap→return) can be **mostly instantaneous**, implying rapid decay; a 1-minute decision loop can be “too slow.”   
- Public repos demonstrate implementation patterns, not guaranteed net profitability (costs/latency dominate).   

**High-risk self-check (explicit):**
- I am **not** assuming “LPI is alpha” by default.  
- The proposal is framed as: *“there is commonly observed short-horizon imbalance predictability; test whether it remains tradable at minute horizons after costs.”*  
- Any parameters below are **starting points** (engineering defaults), not “optimal.”

### 2.7 Validation plan (minimal but research-grade)
- **Signal decay:** measure IC / hit-rate for horizons {1m, 5m, 15m}. If the edge is only sub-minute, minute trading likely fails.   
- **Cost dominance:** include spread crossing + fees (+ slippage model if available).  
- **Robustness:** walk-forward on thresholds; report sensitivity surfaces (T_entry, lookback N, max_spread_bps).  
- **Liquidity regime stratification:** evaluate separately in low vs high spread regimes.

---

## 3) Implementation guide (NautilusTrader, minimal + idiomatic)

Below are **three complete files**:
1. `ingest_lpi_catalog.py` — CSV → `ParquetDataCatalog`
2. `lpi_strategy.py` — Strategy (LPI)
3. `run_lpi_backtest.py` — Backtest wiring (`BacktestNode`)

These follow Nautilus’ documented catalog/backtest APIs. ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/data/))  

---

### 3.1 `ingest_lpi_catalog.py` (CSV → ParquetDataCatalog)

```python
# ingest_lpi_catalog.py
from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Iterator, Optional

from nautilus_trader.model import Bar, BarType, QuoteTick
from nautilus_trader.model.identifiers import InstrumentId, Symbol
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Currency, Price, Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog

NS_PER_MS: int = 1_000_000


def ms_to_ns(ts_ms: int) -> int:
    return int(ts_ms) * NS_PER_MS


def _looks_like_header(tokens: list[str]) -> bool:
    return any(any(ch.isalpha() for ch in tok) for tok in tokens)


def iter_token_rows(path: Path) -> Iterator[list[str]]:
    """
    Streams tokenized rows from either comma-separated CSV or whitespace-delimited files.
    Skips header-like rows automatically.
    """
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        # Peek to detect delimiter
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
        f.seek(0)

        if is_comma:
            reader = csv.reader(f, delimiter=",")
            for row in reader:
                if not row:
                    continue
                tokens = [tok.strip() for tok in row if tok is not None]
                if not tokens or _looks_like_header(tokens):
                    continue
                yield tokens
        else:
            first = True
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                tokens = ln.split()
                if first and _looks_like_header(tokens):
                    first = False
                    continue
                first = False
                yield tokens


def make_btcusdt_perp_instrument(*, ts_event: int, ts_init: int) -> CryptoPerpetual:
    """
    Minimal stub instrument definition for BTCUSDT perpetual on BINANCE.
    This must match your dataset's InstrumentId routing key.

    Notes:
    - Use high precision to safely ingest Binance-style formatted decimals.
    - For research-grade work, replace fees/margins with your exact venue/tier assumptions.
    """
    instrument_id = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")

    return CryptoPerpetual(
        instrument_id=instrument_id,
        raw_symbol=Symbol("BTCUSDT-PERP"),
        base_currency=Currency.from_str("BTC"),
        quote_currency=Currency.from_str("USDT"),
        settlement_currency=Currency.from_str("USDT"),
        is_inverse=False,
        price_precision=8,
        size_precision=8,
        price_increment=Price.from_str("0.10000000"),
        size_increment=Quantity.from_str("0.00100000"),
        max_quantity=None,
        min_quantity=None,
        max_notional=None,
        min_notional=None,
        max_price=None,
        min_price=None,
        margin_init=Decimal("0.05"),
        margin_maint=Decimal("0.03"),
        maker_fee=Decimal("0"),
        taker_fee=Decimal("0"),
        ts_event=ts_event,
        ts_init=ts_init,
        info={},
    )


@dataclass(frozen=True)
class BookTickerRow:
    bid_px: Decimal
    bid_qty: Decimal
    ask_px: Decimal
    ask_qty: Decimal
    event_time_ms: int


def parse_bookticker(tokens: list[str]) -> Optional[BookTickerRow]:
    """
    Expected Binance futures bookTicker columns:
      update_id, best_bid_price, best_bid_qty, best_ask_price, best_ask_qty, transaction_time, event_time
    """
    if len(tokens) < 7:
        return None
    try:
        return BookTickerRow(
            bid_px=Decimal(tokens[1]),
            bid_qty=Decimal(tokens[2]),
            ask_px=Decimal(tokens[3]),
            ask_qty=Decimal(tokens[4]),
            event_time_ms=int(tokens[6]),
        )
    except Exception:
        return None


@dataclass(frozen=True)
class Kline1mRow:
    open_: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    close_time_ms: int


def parse_kline_1m(tokens: list[str]) -> Optional[Kline1mRow]:
    """
    Expected Binance futures kline columns:
      open_time, open, high, low, close, volume, close_time, ...
    """
    if len(tokens) < 7:
        return None
    try:
        return Kline1mRow(
            open_=Decimal(tokens[1]),
            high=Decimal(tokens[2]),
            low=Decimal(tokens[3]),
            close=Decimal(tokens[4]),
            volume=Decimal(tokens[5]),
            close_time_ms=int(tokens[6]),
        )
    except Exception:
        return None


def ingest_to_catalog(
    *,
    raw_root: Path,
    catalog_root: Path,
    start_ts_ns: int,
) -> None:
    """
    Writes:
      - Instrument definition
      - QuoteTicks from bookTicker
      - Bars (1m) from klines_1m
    into a ParquetDataCatalog.
    """
    catalog_root.mkdir(parents=True, exist_ok=True)
    catalog = ParquetDataCatalog(str(catalog_root))

    instrument = make_btcusdt_perp_instrument(ts_event=start_ts_ns, ts_init=start_ts_ns)
    bar_type_1m = BarType.from_str(f"{instrument.id.value}-1-MINUTE-LAST-EXTERNAL")

    # Write instrument first (required for BacktestNode to resolve instruments cleanly)
    catalog.write_data([instrument])

    # --- Quotes ---
    q_dir = raw_root / "futures" / "daily" / "bookTicker"
    quote_batch: list[QuoteTick] = []
    for fp in sorted(q_dir.glob("BTCUSDT-bookTicker-*.csv")):
        for tokens in iter_token_rows(fp):
            row = parse_bookticker(tokens)
            if row is None:
                continue
            ts_event = ms_to_ns(row.event_time_ms)
            qt = QuoteTick(
                instrument_id=instrument.id,
                bid_price=instrument.make_price(row.bid_px),
                ask_price=instrument.make_price(row.ask_px),
                bid_size=instrument.make_qty(row.bid_qty),
                ask_size=instrument.make_qty(row.ask_qty),
                ts_event=ts_event,
                ts_init=ts_event,
            )
            quote_batch.append(qt)
            if len(quote_batch) >= 250_000:
                catalog.write_data(quote_batch, skip_disjoint_check=True)
                quote_batch.clear()
    if quote_batch:
        catalog.write_data(quote_batch, skip_disjoint_check=True)

    # --- Bars (1m) ---
    b_dir = raw_root / "futures" / "daily" / "klines_1m"
    bar_batch: list[Bar] = []
    for fp in sorted(b_dir.glob("BTCUSDT-1m-*.csv")):
        for tokens in iter_token_rows(fp):
            row = parse_kline_1m(tokens)
            if row is None:
                continue
            ts_event = ms_to_ns(row.close_time_ms)  # timestamp bars on close
            bar = Bar(
                bar_type=bar_type_1m,
                open=instrument.make_price(row.open_),
                high=instrument.make_price(row.high),
                low=instrument.make_price(row.low),
                close=instrument.make_price(row.close),
                volume=instrument.make_qty(row.volume),
                ts_event=ts_event,
                ts_init=ts_event,
            )
            bar_batch.append(bar)
            if len(bar_batch) >= 100_000:
                catalog.write_data(bar_batch, skip_disjoint_check=True)
                bar_batch.clear()
    if bar_batch:
        catalog.write_data(bar_batch, skip_disjoint_check=True)


if __name__ == "__main__":
    RAW_ROOT = Path("data/raw")
    CATALOG_ROOT = Path("data/catalog_lpi")

    # Example: May 2023 window start (ns). Use your true start date if different.
    START_TS_NS = 1684195200 * 1_000_000_000  # 2023-05-16T00:00:00Z

    ingest_to_catalog(
        raw_root=RAW_ROOT,
        catalog_root=CATALOG_ROOT,
        start_ts_ns=START_TS_NS,
    )
```

**Logic (≤5 bullets):**
- Streams Binance-style CSV rows and converts timestamps **ms → ns** (Nautilus time contract). ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/data/))  
- Writes a **stub `CryptoPerpetual` instrument** so the catalog/backtest can resolve `InstrumentId` deterministically. (Constructor pattern matches Nautilus usage in public examples.) ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/data/))  
- Writes only what LPI needs: `QuoteTick` (spread/imbalance) and `Bar` (volume).  
- Uses batched `catalog.write_data(...)` for memory control. ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/data/))  
- Bars are timestamped on **close** (causal bar-close decisioning).

---

### 3.2 `lpi_strategy.py` (LiquidityPressureStrategy)

```python
# lpi_strategy.py
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from decimal import Decimal

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model import Bar, BarType, QuoteTick
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy


@dataclass
class RollingZ:
    window: int
    xs: deque[float]

    def __init__(self, window: int) -> None:
        self.window = int(window)
        self.xs = deque(maxlen=self.window)

    def update(self, x: float) -> None:
        if math.isfinite(x):
            self.xs.append(float(x))

    def mean_std(self) -> tuple[float, float]:
        n = len(self.xs)
        if n < 2:
            return (math.nan, math.nan)
        mu = sum(self.xs) / n
        var = sum((v - mu) ** 2 for v in self.xs) / (n - 1)
        return (mu, math.sqrt(max(0.0, var)))

    def z(self, x: float, eps: float = 1e-12) -> float:
        mu, sd = self.mean_std()
        if not math.isfinite(mu) or not math.isfinite(sd):
            return 0.0
        return (x - mu) / (sd + eps)


class LiquidityPressureConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType

    trade_size: Decimal  # base quantity (BTC units)

    lookback_bars: int = 60
    entry_threshold: float = 1.0
    exit_threshold: float = 0.2
    max_spread_bps: float = 5.0  # hard liquidity gate


class LiquidityPressureStrategy(Strategy):
    """
    Medium-frequency LPI strategy:
      - Acts on 1-minute bar closes.
      - Uses last-known QuoteTick for spread + imbalance.
      - Uses bar volume for volume surprise.
    """

    def __init__(self, config: LiquidityPressureConfig) -> None:
        super().__init__(config)

        self.instrument: Instrument | None = None
        self._last_quote: QuoteTick | None = None

        self._vol_z = RollingZ(window=config.lookback_bars)
        self._spr_z = RollingZ(window=config.lookback_bars)

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Instrument not found: {self.config.instrument_id}")
            self.stop()
            return

        self.subscribe_quote_ticks(self.config.instrument_id)
        self.subscribe_bars(self.config.bar_type)

    def on_quote_tick(self, tick: QuoteTick) -> None:
        self._last_quote = tick

    def on_bar(self, bar: Bar) -> None:
        if self.instrument is None or self._last_quote is None:
            return

        # --- spread + imbalance from last quote ---
        bid = float(self._last_quote.bid_price)
        ask = float(self._last_quote.ask_price)
        if bid <= 0.0 or ask <= 0.0 or ask < bid:
            return

        mid = 0.5 * (bid + ask)
        spread_bps = 10_000.0 * ((ask - bid) / mid)

        bid_sz = float(self._last_quote.bid_size)
        ask_sz = float(self._last_quote.ask_size)
        imb = (bid_sz - ask_sz) / (bid_sz + ask_sz + 1e-12)  # ~[-1, 1]

        # Hard liquidity gate
        if spread_bps > float(self.config.max_spread_bps):
            self._flatten_if_needed()
            return

        # --- volume surprise from 1m bar ---
        v = float(bar.volume)
        x_vol = math.log1p(max(0.0, v))

        self._vol_z.update(x_vol)
        self._spr_z.update(spread_bps)

        z_vol = self._vol_z.z(x_vol)
        z_spr = self._spr_z.z(spread_bps)

        vol_boost = max(0.0, z_vol)
        spr_penalty = 1.0 + max(0.0, z_spr)

        lpi = imb * vol_boost / spr_penalty

        # --- trading rule ---
        if lpi > float(self.config.entry_threshold):
            self._target_long()
        elif lpi < -float(self.config.entry_threshold):
            self._target_short()
        elif abs(lpi) < float(self.config.exit_threshold):
            self._flatten_if_needed()

    def _order_qty(self) -> Quantity:
        assert self.instrument is not None
        return self.instrument.make_qty(self.config.trade_size)

    def _target_long(self) -> None:
        if self.portfolio.is_flat(self.config.instrument_id):
            self._buy()
        elif self.portfolio.is_net_short(self.config.instrument_id):
            self.close_all_positions(self.config.instrument_id)
            self._buy()

    def _target_short(self) -> None:
        if self.portfolio.is_flat(self.config.instrument_id):
            self._sell()
        elif self.portfolio.is_net_long(self.config.instrument_id):
            self.close_all_positions(self.config.instrument_id)
            self._sell()

    def _flatten_if_needed(self) -> None:
        if not self.portfolio.is_flat(self.config.instrument_id):
            self.close_all_positions(self.config.instrument_id)

    def _buy(self) -> None:
        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self._order_qty(),
            time_in_force=TimeInForce.FOK,
        )
        self.submit_order(order)

    def _sell(self) -> None:
        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.SELL,
            quantity=self._order_qty(),
            time_in_force=TimeInForce.FOK,
        )
        self.submit_order(order)

    def on_stop(self) -> None:
        self.cancel_all_orders(self.config.instrument_id)
        self.close_all_positions(self.config.instrument_id)
        self.unsubscribe_bars(self.config.bar_type)
        self.unsubscribe_quote_ticks(self.config.instrument_id)
```

**Logic (≤5 bullets):**
- Decisioning happens **only** on `on_bar` (1-minute cadence).
- Uses **last-known** `QuoteTick` for spread/imbalance (no expensive joins).
- Uses rolling z-scores for `log1p(volume)` and `spread_bps`.
- Hard spread gate forces flat under poor liquidity.
- Flat/long/short state machine (no pyramiding).

---

### 3.3 `run_lpi_backtest.py` (BacktestNode wiring)

```python
# run_lpi_backtest.py
from __future__ import annotations

from decimal import Decimal

from nautilus_trader.backtest.node import (
    BacktestDataConfig,
    BacktestEngineConfig,
    BacktestNode,
    BacktestRunConfig,
    BacktestVenueConfig,
)
from nautilus_trader.config import ImportableStrategyConfig, LoggingConfig
from nautilus_trader.model import Bar, BarType, QuoteTick
from nautilus_trader.model.identifiers import InstrumentId


def main() -> None:
    CATALOG_PATH = "data/catalog_lpi"

    instrument_id = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")
    bar_type_1m = BarType.from_str("BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL")

    start = "2023-05-16T00:00:00Z"
    end = "2023-06-01T00:00:00Z"

    venue = BacktestVenueConfig(
        name="BINANCE",
        oms_type="NETTING",
        account_type="MARGIN",
        base_currency="USDT",
        starting_balances=["100000 USDT"],
    )

    data = [
        BacktestDataConfig(
            catalog_path=CATALOG_PATH,
            data_cls=QuoteTick,
            instrument_id=instrument_id,
            start_time=start,
            end_time=end,
        ),
        BacktestDataConfig(
            catalog_path=CATALOG_PATH,
            data_cls=Bar,
            instrument_id=instrument_id,
            bar_spec="1-MINUTE-LAST",
            start_time=start,
            end_time=end,
        ),
    ]

    strategies = [
        ImportableStrategyConfig(
            strategy_path="lpi_strategy:LiquidityPressureStrategy",
            config_path="lpi_strategy:LiquidityPressureConfig",
            config={
                "instrument_id": instrument_id,
                "bar_type": bar_type_1m,
                "trade_size": Decimal("0.01"),
                "lookback_bars": 60,
                "entry_threshold": 1.0,
                "exit_threshold": 0.2,
                "max_spread_bps": 5.0,
            },
        )
    ]

    engine = BacktestEngineConfig(
        strategies=strategies,
        logging=LoggingConfig(log_level="INFO"),
    )

    run = BacktestRunConfig(
        engine=engine,
        venues=[venue],
        data=data,
        start=start,
        end=end,
    )

    node = BacktestNode(configs=[run])
    results = node.run()
    node.dispose()

    # Print a minimal summary
    print(results[0])


if __name__ == "__main__":
    main()
```

**Logic (≤5 bullets):**
- Loads `QuoteTick` and `Bar` streams from `ParquetDataCatalog`. ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/data/))  
- Uses `bar_spec="1-MINUTE-LAST"` to load the bar series; the stored bars still route by `BarType` inside the engine. ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/data/))  
- Strategy is wired via `ImportableStrategyConfig` (idiomatic high-level backtest config).   
- Date window matches your May 2023 sample (edit as needed).  
- Keeps venue config minimal (you should later add fees/slippage assumptions for realism).

---

## 4) Final “profitability” stance (no hype)
Based on the historical anomalies and microstructure evidence:
- **Directional imbalance signals** do have documented predictive content at short horizons (queue imbalance / OFI-style results).   
- Microprice-style signals explicitly leverage **imbalance + spread** to estimate near-future prices, reinforcing that your feature family is “microstructure-plausible.”   

However, for BTC specifically, there is credible evidence that some order-book→return relations can be **mostly instantaneous**, which can eliminate tradability at a 1-minute cadence.   

So the correct research posture is:
- **Test** whether LPI retains signal at {1m, 5m, 15m} and survives costs.
- Do **not** assume profitability from statistical predictability alone.

If you paste your exact `nautilus_trader` version (e.g., `pip show nautilus_trader`) I can pin the code to that specific version’s API surface; the above is aligned to the current Nautilus high-level docs and catalog workflow. ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/data/))

---
Learn more:
1. [Order Book Pressure - Amberdata Docs](https://docs.amberdata.io/data-dictionary/analytics/derivatives/order-book-pressure?utm_source=openai)
2. [Data | NautilusTrader Documentation](https://nautilustrader.io/docs/latest/concepts/data/)
```

```
## 0) Input processing (as requested)

### Short internal outline of the provided proposal + codebase
1. **Data & contracts**
   - Binance Vision-style CSVs for *futures* (and spot in examples) with daily/monthly partitions.
   - Canonical routing keys: `InstrumentId`, `BarType`, and strict timestamp conversion to **Unix ns**.

2. **ETL**
   - Scan/parse CSVs → build an inventory → filter by universe/dataset/symbol/date → parse into Nautilus `Data` objects → write a `ParquetDataCatalog`.

3. **Liquidity Pressure Index (LPI) signal**
   - Maintain last quote (mid/spread/age).
   - Accumulate signed *impulsive* trade notional into buckets (time or notional).
   - Optional: incorporate *depth at ±X%* via pre-aggregated `bookDepth` data.
   - Standardize with rolling z-scores; compute `lpi_fast`, `delta_lpi`, and an “exhaustion” proxy.

4. **Strategy logic**
   - Gate on quote freshness + spread + (optional) depth freshness.
   - When `|LPI|` is large: choose **momentum** vs **mean-reversion** variant using exhaustion.
   - Place market/marketable orders for momentum; post-only/passive limits for mean-reversion; TTL cancellation; time-based exits.

5. **Backtesting harness (NautilusTrader)**
   - `BacktestNode` + catalog replay, deterministic ordering, reporting (orders/fills/positions/account).

### Core constraints (re-stated)
- **Signal family:** Liquidity Pressure Index using **volume/flow**, **spread**, and **order-book imbalance/depth**.
- **Frequency:** **Medium-frequency** (think seconds to minutes, not microsecond HFT).
- **Engine:** Must be implemented and backtested using **`nautilus_trader`** (Parquet catalog + `BacktestNode` preferred for scale). ([nautilustrader.io](https://nautilustrader.io/docs/nightly/concepts/backtesting/))

---

## 1) Web-researched grounding: what is “Liquidity Pressure” in microstructure?

### 1.1 There is no single “canonical” LPI in academic crypto literature
“Liquidity pressure” is used informally across practitioner writeups, but academically the components that *do* have strong foundations are:

- **Order flow imbalance (OFI)** at/near top-of-book
- **Order book depth / resiliency**
- **Bid–ask spread (transaction cost proxy)**

A clean way to defend an “LPI” is: *define it explicitly as a standardized fusion of OFI-like pressure and liquidity tightness (spread vs depth)*, rather than claiming it’s a standard named index.

### 1.2 Strong, directly relevant microstructure results (general markets)
- **Price changes over short intervals are mainly driven by OFI**, and the impact slope is **inversely related to depth**. This is exactly the intuition your LPI is trying to encode (pressure × liquidity). ([arxiv.org](https://arxiv.org/abs/1011.6402?utm_source=openai))  
- Liquidity imbalance can correlate with both **sign and magnitude** of subsequent price moves, and **the effect depends strongly on the time scale** (e.g., tens of seconds vs minutes). That supports a *medium-frequency* stance: choose a horizon where imbalance persists long enough to overcome costs. ([arxiv.org](https://arxiv.org/abs/1504.02956?utm_source=openai))

### 1.3 Crypto-specific evidence: useful but mixed
- There is research showing that **LOB-derived variables can forecast crypto return direction** and can be used in a trading setup (the paper reports improvements in prediction and “cumulative profit” in their tested environment). ([arxiv.org](https://arxiv.org/abs/2312.16190?utm_source=openai))  
- But there is also evidence that adding “informed trading” indicators (including order imbalance / toxicity / order frequency) **does not necessarily improve cryptocurrency return prediction accuracy** in some ML settings—i.e., the edge may be weak, unstable, or cost-dominated. ([sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S027553192200071X?utm_source=openai))  
- Order-book dynamics in **BTC/USDT** show stylized properties and time-scale dependence in the **volume at best bid/ask**, reinforcing that (a) microstructure features are real, but (b) behavior can change with sampling frequency. ([sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S037843712400743X?utm_source=openai))

### 1.4 Profitability caveat that matters for “imbalance strategies”
For pure order-book-imbalance strategies, **latency rank can dominate profitability** in competitive environments (shown via simulation). This is a warning sign: if you push LPI too “HFT”, you are likely competing in the worst possible arena. ([arxiv.org](https://arxiv.org/abs/2006.08682?utm_source=openai))  

**Implication for your spec:** pick **seconds-to-minutes** buckets, add **staleness/spread gates**, and design execution to reduce taker churn.

### 1.5 Why your spread+depth fusion is defensible
Central-bank style liquidity measurement explicitly notes that more informative liquidity indicators combine **spread** with **order book depth** (cost + capacity). ([ecb.europa.eu](https://www.ecb.europa.eu/press/economic-bulletin/html/eb201802.en.html?utm_source=openai))  

That’s the right conceptual scaffold for an LPI.

---

## 2) Revised proposal (tight, concise, implementable)

### 2.1 Thesis (what we are testing)
**Hypothesis:**  
In BTCUSDT perpetual futures, short-horizon returns over a *medium-frequency decision interval* (Δ ≈ 5s–60s) are conditionally predictable from:
- **Directional pressure** (aggressive flow / executed imbalance),
- **Liquidity tightness** (spread vs depth / spread vs mid),
- **Liquidity exhaustion** signals (pressure decelerating while spread widens or depth thins).

This aligns with established OFI → price change links, and the known role of depth and spreads in price impact. ([arxiv.org](https://arxiv.org/abs/1011.6402?utm_source=openai))

### 2.2 Data constraints (explicit)
**Minimum viable (L1 + trades):**
- `QuoteTick` (best bid/ask + sizes) for spread, mid, quote age. ([nautilustrader.io](https://nautilustrader.io/docs/latest/api_reference/model/data/?utm_source=openai))  
- `TradeTick` (executed trades) for signed flow / volume pressure.

**Optional (recommended if available):**
- Pre-aggregated depth bands (e.g., your `bookDepth` at ±5%) as a proxy for depth/resiliency.
- Funding / OI as slow regime priors (not required for the LPI definition itself).

### 2.3 Liquidity Pressure Index (LPI): exact definition (what the strategy computes)

Fix a decision bucketization:
- **Time buckets:** Δt = `bucket_interval_ms` (recommended for medium-frequency).
- Or **notional buckets:** bucket closes when cumulative notional ≥ `bucket_notional_threshold`.

Maintain last quote state:
- `mid = (bid + ask)/2`
- `spread = ask - bid`
- `rel_spread = spread / mid`
- `quote_age_ms = (bucket_end - last_quote_ts)/1e6`

#### Pressure component (executed imbalance, “impulse-filtered”)
For each trade with price `p`, size `q`, aggressor side `BUY/SELL`, define notional `n = p*q`.

Impulse filter (one practical choice):
- Count a **buy-impulse** when `p > ask*(1+η)`
- Count a **sell-impulse** when `p < bid*(1-η)`
where `η = eta_bps / 10_000`.

Bucket totals:
- `IB = Σ notional of buy-impulses`
- `IS = Σ notional of sell-impulses`
- `pressure_raw = IS - IB`

#### Liquidity-demand component (spread vs depth)
Two variants:
- If depth-band notional `D` (e.g., sum notional within ±5%) is available and fresh:  
  `ld_raw = spread / (D + eps)`
- Else fallback:  
  `ld_raw = spread / mid`

#### Standardize and fuse
Compute rolling z-scores (robust median/MAD is a good default):
- `pressure_z = zscore(pressure_raw)`
- `ld_z = zscore(ld_raw)`

**LPI:**
- `lpi_fast = pressure_z + λ_ld * ld_z`

This fusion is directly consistent with “impact slope inversely related to depth” intuition: the same pressure should matter more when liquidity is thin. ([arxiv.org](https://arxiv.org/abs/1011.6402?utm_source=openai))

#### Exhaustion proxy (regime switch trigger)
Define:
- `delta_lpi = lpi_fast - prev_lpi`
- Exhaustion rises when **pressure weakens** (`-delta_lpi` large) **and** liquidity deteriorates (spread widens or depth falls).

A simple standardized exhaustion score:
- `exh = zscore(-delta_lpi) + λ_exh * zscore(Δdepth)`  
  (or use `-Δspread` if depth bands are not available)

### 2.4 Trading policy (only two modes; no extra “alpha” features)

**Gates (hard filters):**
- Quote exists and not stale (`quote_age_ms ≤ quote_max_age_ms`).
- Spread not extreme (`rel_spread_bps ≤ max_rel_spread_bps`).
- If using depth: depth is fresh and above `min_depth_notional`.

**Entry condition:**
- Enter only when `|lpi_fast| ≥ theta_lpi`.

**Variant selection:**
- If `exh ≤ theta_exh_low` → **momentum**
- If `exh ≥ theta_exh_high` → **mean reversion**
- Otherwise: no trade (avoid ambiguous regime)

**Direction:**
Interpretation: `pressure_raw > 0` means sell pressure dominates.
- Momentum:
  - if `lpi_fast > 0` → SELL
  - if `lpi_fast < 0` → BUY
- Mean reversion:
  - if `lpi_fast > 0` → BUY
  - if `lpi_fast < 0` → SELL

**Exits:**
- Momentum: time-stop after `max_hold_buckets_momentum`, or if exhaustion flips to mean-reversion regime.
- Mean reversion: exit when `|lpi_fast| ≤ lpi_exit_abs`, or time-stop `max_hold_buckets_mean_rev`.

### 2.5 Implementation realism requirements (what must be explicit in backtests)
- **Event ordering:** Nautilus processes data streams in chronological order using `ts_init`; your ETL must ensure monotonicity / sorting. ([nautilustrader.io](https://nautilustrader.io/docs/nightly/tutorials/backtest_binance_orderbook))  
- **Venue book type must match your data granularity.** If you set `L2/L3` book types, Nautilus will ignore non-order-book data for execution processing. For LPI using `QuoteTick/TradeTick`, keep venue book type at **L1** (default) to avoid “orders never fill” pathologies. ([nautilustrader.io](https://nautilustrader.io/docs/nightly/concepts/backtesting/))  
- **Costs matter:** backtest must include slippage/latency/fees assumptions where possible (FillModel / FeeModel exist in Nautilus). ([nautilustrader.io](https://nautilustrader.io/docs/latest/api_reference/backtest/))  

---

## 3) Profitability (evidence-based, no promises)

### What the literature supports
- **There is real short-horizon structure** in order-book markets: OFI/liquidity imbalance correlates with near-future price moves, and the effect depends on horizon. ([arxiv.org](https://arxiv.org/abs/1011.6402?utm_source=openai))  
- **LOB-based prediction can translate to trading setups** in crypto-related research (at least in controlled experiments / simulations). ([arxiv.org](https://arxiv.org/abs/2312.16190?utm_source=openai))  

### What can easily kill profitability (especially in crypto perps)
- **Latency arms race** can dominate returns for imbalance-style strategies if you trade too fast. This pushes you away from “HFT OBI” and toward **medium-frequency** + **gating**. ([arxiv.org](https://arxiv.org/abs/2006.08682?utm_source=openai))  
- Some empirical work finds “informed trading” indicators do **not** reliably improve crypto return prediction in certain setups—so you must treat LPI as a hypothesis to validate, not a given edge. ([sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S027553192200071X?utm_source=openai))  
- Fees and spread-crossing costs are first-order in perps; Nautilus explicitly models fee and fill components, so you should run sensitivity sweeps rather than one “best-case” backtest. ([nautilustrader.io](https://nautilustrader.io/docs/latest/api_reference/backtest/))  

### What is *plausible* (careful wording)
Based on historically observed short-horizon anomalies around order-flow imbalance and liquidity depletion in order-book markets, an LPI can plausibly function as:
- a **trade filter** (avoid trading when liquidity is fragile), and/or
- a **conditional directional signal** at horizons where imbalance persists long enough to beat costs. ([arxiv.org](https://arxiv.org/abs/1011.6402?utm_source=openai))  

But you should not expect stable profits without:
- explicit cost modeling,
- stability tests across regimes,
- and avoiding “too-fast” triggers.

---

## 4) Implementation guide (NautilusTrader-specific, using your repo as-is)

Your provided `12.24.2025.py` already implements:
- CSV → `ParquetDataCatalog` ETL (deterministic),
- custom `Binance*` data classes,
- `PhoenixLpiStrategy`,
- a `run_phase4` backtest harness.

The Nautilus pattern is consistent with the official tutorial: write data to a `ParquetDataCatalog`, then run a `BacktestNode` with `BacktestRunConfig` + `ImportableStrategyConfig`. ([nautilustrader.io](https://nautilustrader.io/docs/nightly/tutorials/backtest_binance_orderbook))

### 4.1 Required files (minimal)
Create **two YAML configs** + **instrument specs**.

#### A) `instrument_specs.yaml` (minimal, 1 instrument)
```yaml
instruments:
  - instrument_id: "BTCUSDT.BINANCE"
    raw_symbol: "BTCUSDT"
    base_currency: "BTC"
    quote_currency: "USDT"
    settlement_currency: "USDT"
    is_inverse: false
    price_precision: 2
    size_precision: 3
    price_increment: "0.01"
    size_increment: "0.001"
    maker_fee: "0.0002"
    taker_fee: "0.0004"
```

- Maker/taker fees are first-class instrument properties in Nautilus instruments. ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/instruments/?utm_source=openai))  
- Precision/increments must match your exchange contract; otherwise you will get unrealistic fills or order rejections.

#### B) `etl_config.yaml` (for `python 12.24.2025.py ingest --config ...`)
```yaml
raw_roots:
  - "./data/raw"          # points at data/raw/futures/...
catalog_path: "./data/catalog/phoenix_btcusdt"
manifest_path: "./data/catalog/phoenix_btcusdt/manifest.json"
validation_report_path: "./data/catalog/phoenix_btcusdt/validation_report.md"

universe: "futures"
venue: "BINANCE"

symbols: ["BTCUSDT"]
datasets:
  - "aggTrades"
  - "bookTicker"
  - "bookDepth"
  - "metrics"
  - "fundingRate"

start_date_key: "2023-05-16"
end_date_key: "2023-05-31"

batch_size: 50000
strict: false
sort_within_batch: true

instrument_specs_path: "./instrument_specs.yaml"
ingest_workers: 4
ingest_queue_max_batches: 8
```

- Your folder layout matches what Nautilus expects: futures daily files by dataset.  
- This ETL step exists so backtests can stream from Parquet instead of loading CSVs into RAM. That’s the recommended high-level approach. ([nautilustrader.io](https://nautilustrader.io/docs/nightly/concepts/backtesting/))  

#### C) `suite_config.yaml` (for `python 12.24.2025.py run --config ...`)
```yaml
schema_version: "1"

catalog:
  path: "./data/catalog/phoenix_btcusdt"
  manifest_path: "./data/catalog/phoenix_btcusdt/manifest.json"
  instrument_specs_path: "./instrument_specs.yaml"

output:
  dir: "./runs"
  suite_name: "lpi_btcusdt_may2023"
  overwrite: true

universe:
  venue: "BINANCE"
  instrument_ids: ["BTCUSDT.BINANCE"]
  oms_type: "NETTING"
  account_type: "MARGIN"
  base_currency: "USDT"
  starting_balances: ["10000 USDT"]

data:
  include:
    - "QuoteTick"
    - "TradeTick"
    - "BinanceBookDepthPct"
    - "BinanceOiMetrics"
    - "BinanceFundingRate"

time:
  start_utc: "2023-05-16T00:00:00Z"
  end_utc: "2023-06-01T00:00:00Z"
  warmup: "0s"

mode:
  kind: "single"

strategy:
  name: "phoenix_lpi"
  config:
    trade_size: "0.01"
    bucket_mode: "time"
    bucket_interval_ms: 10000
    eta_bps: 5.0

    use_depth_pct: true
    depth_pct_abs: 5.0
    min_depth_notional: 0.0

    use_zscore: true
    z_kind: "median_mad"
    z_window_buckets: 600
    z_min_count: 30

    theta_lpi: 2.0
    theta_exh_low: -0.5
    theta_exh_high: 0.5
    lpi_exit_abs: 0.5

    max_rel_spread_bps: 20.0
    quote_max_age_ms: 1000
    depth_max_age_ms: 5000

    momentum_use_market: true
    mean_rev_post_only: true
    mean_rev_improve_ticks: 0

    entry_ttl_buckets: 5
    max_hold_buckets_momentum: 10
    max_hold_buckets_mean_rev: 60

    close_positions_on_stop: true
    reduce_only_on_stop: true

determinism_seed: 0
```

- Use **`bucket_mode: time`** to keep the strategy in the “medium-frequency” lane.
- Your end time should be exclusive (`2023-06-01T00:00:00Z`) to avoid off-by-one-day ingestion assumptions (this matches common replay practice).

### 4.2 Run commands
```bash
# 1) Build catalog
python 12.24.2025.py ingest --config etl_config.yaml

# 2) Run backtest suite
python 12.24.2025.py run --config suite_config.yaml
```

### 4.3 What to inspect in outputs (minimum viable validation)
- `validation_report.md`: confirm catalog types load and that event counts are non-zero.
- `orders.csv`, `fills.csv`, `positions.csv`, `account_report.csv`: confirm fills occur and the strategy is not “stuck” behind gates.
- Confirm your venue config uses **L1 execution** semantics (do *not* set L2/L3 book types unless you actually replay order book deltas). ([nautilustrader.io](https://nautilustrader.io/docs/nightly/concepts/backtesting/))  

---

## 5) High-risk self-check (profitability section)
- I did **not** assume the LPI is profitable; I only stated that **imbalance/liquidity measures are empirically linked to short-horizon price changes** and that turning this into net returns is non-trivial. ([arxiv.org](https://arxiv.org/abs/1011.6402?utm_source=openai))  
- Any parameter suggestions (e.g., 10s buckets, MAD z-score) are **implementation conveniences**, not claimed optima; they require tuning and stability tests.
- Key unstated alpha risks explicitly called out:
  - latency competition, ([arxiv.org](https://arxiv.org/abs/2006.08682?utm_source=openai))
  - cost domination (fees/slippage), ([nautilustrader.io](https://nautilustrader.io/docs/latest/api_reference/backtest/))
  - instability of “informed trading” indicators in crypto. ([sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S027553192200071X?utm_source=openai))  

If you want, paste **your intended holding horizon** (e.g., “target 30–120 seconds”) and I’ll tighten the LPI bucketization + thresholds so the proposal is dimensionally consistent with that single Δ.

---
Learn more:
1. [Backtesting | NautilusTrader Documentation](https://nautilustrader.io/docs/nightly/concepts/backtesting/)
2. [The Price Impact of Order Book Events](https://arxiv.org/abs/1011.6402?utm_source=openai)
3. [Liquidity crises on different time scales](https://arxiv.org/abs/1504.02956?utm_source=openai)
4. [Hawkes-based cryptocurrency forecasting via Limit Order Book data](https://arxiv.org/abs/2312.16190?utm_source=openai)
5. [Can investors’ informed trading predict cryptocurrency returns? Evidence from machine learning - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S027553192200071X?utm_source=openai)
6. [Empirical properties of volume dynamics in the limit order book - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S037843712400743X?utm_source=openai)
7. [The Importance of Low Latency to Order Book Imbalance Trading Strategies](https://arxiv.org/abs/2006.08682?utm_source=openai)
8. [Economic Bulletin Issue 2, 2018](https://www.ecb.europa.eu/press/economic-bulletin/html/eb201802.en.html?utm_source=openai)
9. [Data | NautilusTrader Documentation](https://nautilustrader.io/docs/latest/api_reference/model/data/?utm_source=openai)
10. [Backtest: Binance OrderBook data | NautilusTrader Documentation](https://nautilustrader.io/docs/nightly/tutorials/backtest_binance_orderbook)
11. [Backtest | NautilusTrader Documentation](https://nautilustrader.io/docs/latest/api_reference/backtest/)
12. [Instruments | NautilusTrader Documentation](https://nautilustrader.io/docs/latest/concepts/instruments/?utm_source=openai)
```