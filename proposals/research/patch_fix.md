This PART 1 patch fixes a major backtest-correctness bug (1‑minute bars were timestamped on **open** instead of **close**), tightens instrument-spec parsing to be silent and strict, and makes multi-stream replay tie-order more deterministic by using `append_data` priorities when the Nautilus version supports it.

- What changed: 1m klines now use **close_time** (or a safe close-time fallback) for `ts_event/ts_init`; instrument specs parsing is silent + stricter; backtest data configs now pass `append_data` to enforce deterministic tie priority when available.
- Where: `FuturesKlines1mParser`, `load_instrument_specs`, and `build_backtest_data_configs` in `existing_code.py`.
- Risks: If your kline CSVs truly lack `close_time` and do not represent 1‑minute intervals, the fallback close-time computation could be wrong; if your Nautilus version doesn’t support `append_data`, priorities will be ignored (still works).
- Next steps: Reply CONTINUE for PART 2.
- Open questions: (1) Confirm your NautilusTrader version string (pip) so we can lock exact BacktestDataConfig fields beyond `append_data` if needed. (2) For same-timestamp ties, do you want QuoteTick→TradeTick→Bar as the canonical causality order?

PATCHSET: Deterministic bars + instrument specs hygiene + backtest tie priorities (PART 1)  
TARGET: existing_code.py  
PART: 1/<?>

PATCH 01: Fix 1m bar timestamp semantics (close-time, no look-ahead)  
INTENT: Ensure 1-minute bars are timestamped on close (`close_time`) for causal backtests (chap2 §2.1.2 / review “bar close convention”).  
OP:
- ACTION: ReplaceSymbol
- LOCATOR: FuturesKlines1mParser
NEW_CODE:
```python
class FuturesKlines1mParser(BaseParser):
    dataset = "klines_1m"

    def parse_file(
        self,
        path: Path,
        instrument_id: Any,
        *,
        strict: bool,
        sort_within_batch: bool,
        batch_size: int,
    ) -> Iterable[ParseResult]:
        # chap2 §2.1.2: bar ts_event should be the bar close time (ns), not open time.
        N = load_nautilus_imports()
        Bar = N.Bar
        BarType = N.BarType
        Price = N.Price
        Quantity = N.Quantity

        bar_type = BarType.from_str(f"{instrument_id}-1-MINUTE-LAST-EXTERNAL")

        NS_PER_MS = 1_000_000
        # Binance kline convention: close_time = open_time + interval_ms - 1, for 1m => +59999ms.
        DEFAULT_CLOSE_OFFSET_NS = (60_000 - 1) * NS_PER_MS

        rows_read = 0
        errors = 0
        dedup = 0
        ts_min = None
        ts_max = None
        last_close_ns: int | None = None
        events: list[Any] = []

        if detect_header(path):
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows_read += 1
                    try:
                        r = {str(k).strip().lower(): ("" if v is None else str(v).strip()) for k, v in row.items()}

                        open_time = r.get("open_time") or r.get("opentime") or r.get("t")
                        close_time = r.get("close_time") or r.get("closetime") or r.get("close_time_ms") or r.get("ct")

                        o = r.get("open") or r.get("o")
                        h = r.get("high") or r.get("h")
                        l = r.get("low") or r.get("l")
                        c = r.get("close") or r.get("c")
                        v = r.get("volume") or r.get("v")

                        if open_time is None or o is None or h is None or l is None or c is None or v is None:
                            raise ValueError("missing required columns")

                        open_ns, e1 = safe_parse_ts(open_time, strict)
                        errors += e1
                        if open_ns is None:
                            continue

                        close_ns: int | None = None
                        if close_time is not None and str(close_time).strip() != "":
                            close_ns, e2 = safe_parse_ts(close_time, strict)
                            errors += e2

                        # If close_time is absent or unparsable, compute a conservative close based on open+1m-1ms.
                        if close_ns is None:
                            close_ns = int(open_ns) + DEFAULT_CLOSE_OFFSET_NS

                        if last_close_ns is not None and close_ns == last_close_ns:
                            dedup += 1
                            continue
                        last_close_ns = close_ns

                        bar = Bar(
                            bar_type=bar_type,
                            open=Price.from_str(o),
                            high=Price.from_str(h),
                            low=Price.from_str(l),
                            close=Price.from_str(c),
                            volume=Quantity.from_str(v),
                            ts_event=close_ns,
                            ts_init=close_ns,
                        )
                        events.append(bar)
                        ts_min = close_ns if ts_min is None else min(ts_min, close_ns)
                        ts_max = close_ns if ts_max is None else max(ts_max, close_ns)

                        if len(events) >= batch_size:
                            batch = finalize_batch(
                                events,
                                sort_key=lambda x: (x.ts_event,),
                                sort_within_batch=sort_within_batch,
                            )
                            yield ParseResult(
                                events=batch,
                                rows_read=rows_read,
                                events_emitted=len(batch),
                                parse_errors=errors,
                                dedup_dropped=dedup,
                                ts_min=ts_min,
                                ts_max=ts_max,
                            )
                            events = []
                    except Exception:
                        if strict:
                            raise
                        errors += 1
                        continue
        else:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    rows_read += 1
                    try:
                        if len(row) < 6:
                            raise ValueError("row too short")
                        open_time, o, h, l, c, v = row[0], row[1], row[2], row[3], row[4], row[5]
                        close_time = row[6] if len(row) > 6 else ""

                        open_ns, e1 = safe_parse_ts(open_time, strict)
                        errors += e1
                        if open_ns is None:
                            continue

                        close_ns: int | None = None
                        if str(close_time).strip():
                            close_ns, e2 = safe_parse_ts(close_time, strict)
                            errors += e2
                        if close_ns is None:
                            close_ns = int(open_ns) + DEFAULT_CLOSE_OFFSET_NS

                        if last_close_ns is not None and close_ns == last_close_ns:
                            dedup += 1
                            continue
                        last_close_ns = close_ns

                        bar = Bar(
                            bar_type=bar_type,
                            open=Price.from_str(o.strip()),
                            high=Price.from_str(h.strip()),
                            low=Price.from_str(l.strip()),
                            close=Price.from_str(c.strip()),
                            volume=Quantity.from_str(v.strip()),
                            ts_event=close_ns,
                            ts_init=close_ns,
                        )
                        events.append(bar)
                        ts_min = close_ns if ts_min is None else min(ts_min, close_ns)
                        ts_max = close_ns if ts_max is None else max(ts_max, close_ns)

                        if len(events) >= batch_size:
                            batch = finalize_batch(
                                events,
                                sort_key=lambda x: (x.ts_event,),
                                sort_within_batch=sort_within_batch,
                            )
                            yield ParseResult(
                                events=batch,
                                rows_read=rows_read,
                                events_emitted=len(batch),
                                parse_errors=errors,
                                dedup_dropped=dedup,
                                ts_min=ts_min,
                                ts_max=ts_max,
                            )
                            events = []
                    except Exception:
                        if strict:
                            raise
                        errors += 1
                        continue

        if events:
            batch = finalize_batch(
                events,
                sort_key=lambda x: (x.ts_event,),
                sort_within_batch=sort_within_batch,
            )
            yield ParseResult(
                events=batch,
                rows_read=rows_read,
                events_emitted=len(batch),
                parse_errors=errors,
                dedup_dropped=dedup,
                ts_min=ts_min,
                ts_max=ts_max,
            )
```

PATCH 02: Silent, strict instrument_specs ingestion  
INTENT: Remove noisy prints and fail fast on malformed instrument specs (required for deterministic, debuggable runs).  
OP:
- ACTION: ReplaceSymbol
- LOCATOR: load_instrument_specs
NEW_CODE:
```python
def load_instrument_specs(path: str) -> dict[str, InstrumentSpec]:
    p = Path(path).expanduser().resolve()
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if raw is None:
        raise InstrumentSpecError("instrument_specs.yaml is empty")
    if not isinstance(raw, dict):
        raise InstrumentSpecError("instrument_specs.yaml must be a YAML mapping")

    items = raw.get("instruments")
    if not isinstance(items, list) or not items:
        raise InstrumentSpecError("instrument_specs.yaml must contain a non-empty 'instruments' list")

    def _req(item: dict[str, Any], key: str) -> Any:
        v = item.get(key)
        if v is None:
            raise InstrumentSpecError(f"instrument spec missing required key {key!r}")
        if isinstance(v, str) and not v.strip():
            raise InstrumentSpecError(f"instrument spec key {key!r} is empty")
        return v

    out: dict[str, InstrumentSpec] = {}
    for item in items:
        if not isinstance(item, dict):
            raise InstrumentSpecError("each instrument spec must be a mapping")

        inst_id = str(_req(item, "instrument_id")).strip()
        raw_symbol = str(item.get("raw_symbol") or inst_id.split(".")[0]).strip()

        quote_cur = str(item.get("quote_currency") or "USDT").strip()
        spec = InstrumentSpec(
            instrument_id=inst_id,
            raw_symbol=raw_symbol,
            base_currency=str(item.get("base_currency") or "BTC").strip(),
            quote_currency=quote_cur,
            settlement_currency=str(item.get("settlement_currency") or quote_cur).strip(),
            is_inverse=bool(item.get("is_inverse", False)),
            price_precision=int(_req(item, "price_precision")),
            size_precision=int(_req(item, "size_precision")),
            price_increment=str(_req(item, "price_increment")).strip(),
            size_increment=str(_req(item, "size_increment")).strip(),
            maker_fee=str(_req(item, "maker_fee")).strip(),
            taker_fee=str(_req(item, "taker_fee")).strip(),
        )
        out[spec.instrument_id] = spec

    return out
```

PATCH 03: Deterministic backtest stream tie order via `append_data`  
INTENT: Make replay ordering deterministic for same-`ts_init` ties by assigning stream priority when supported (review.md “timestamp semantics”).  
OP:
- ACTION: ReplaceSymbol
- LOCATOR: build_backtest_data_configs
NEW_CODE:
```python
def build_backtest_data_configs(N: NautilusPhase4Imports, plan: DataLoadPlan) -> list[Any]:
    """
    Backtest replay ordering note (review.md):
      - Backtests process events in ts_init order; same-ts_init ties are resolved by stream priority.
      - Nautilus uses 'append_data' (when supported) to affect that priority.
    We pass append_data where available via _call_with_supported_kwargs (version-safe).
    """
    out: list[Any] = []
    include_set = {str(s).strip() for s in plan.include if str(s).strip()}
    custom = _load_custom_data_classes()
    ds, de = plan.data_range.to_iso()

    def _mk(data_cls: Any, instrument_id: Any, *, append_data: bool | None = None, bar_type: Any | None = None) -> Any:
        kwargs: dict[str, Any] = {
            "catalog_path": plan.catalog_path,
            "data_cls": data_cls,
            "instrument_id": instrument_id,
            "start_time": ds,
            "end_time": de,
        }
        if append_data is not None:
            kwargs["append_data"] = bool(append_data)
        if bar_type is not None:
            kwargs["bar_type"] = bar_type
        return _call_with_supported_kwargs(N.BacktestDataConfig, kwargs)

    for inst_id in plan.instrument_ids:
        # Highest priority: quotes (so fills/logic can see L1 before processing trades/bars at same ts_init).
        if "QuoteTick" in include_set:
            out.append(_mk(N.QuoteTick, inst_id, append_data=False))

        if "TradeTick" in include_set:
            out.append(_mk(N.TradeTick, inst_id, append_data=True))

        if "BinanceOiMetrics" in include_set:
            out.append(_mk(custom["BinanceOiMetrics"], inst_id, append_data=True))
        if "BinanceFundingRate" in include_set:
            out.append(_mk(custom["BinanceFundingRate"], inst_id, append_data=True))
        if "BinanceBookDepthPct" in include_set:
            out.append(_mk(custom["BinanceBookDepthPct"], inst_id, append_data=True))

        # Lowest priority: bars (bar close is an aggregate; should not pre-empt ticks at same ts_init).
        if "Bar" in include_set:
            bt = None
            try:
                bt = N.BarType.from_str(f"{inst_id}-1-MINUTE-LAST-EXTERNAL")
            except Exception:
                bt = None
            out.append(_mk(N.Bar, inst_id, append_data=True, bar_type=bt))

    return out
```

How to apply
1) Open `existing_code.py` and apply the three ReplaceSymbol patches above.
2) Re-run your ETL ingest (bars will be re-timestamped on close; mixed catalogs will be inconsistent).
3) Run a small backtest window and confirm `Bar.ts_event` matches `close_time` (or open+59999ms fallback).
4) If your Nautilus version ignores `append_data`, note it in logs/errors and tell me the version (see Open questions).
5) Reply `CONTINUE` for PART 2.

This PART 2 patch adds a **deterministic, end-to-end online alpha pipeline** (bounded \(Z_t\), VPIN gate, liquidity gates, and Chapter‑3‑style risk sizing mapped to a max position) and wires it into the existing `phoenix_lpi` strategy entrypoint via `alpha_mode`, without breaking existing configs.

- **What changed:** Added `AlphaZScoreStrategy` (online \(Z_t\) pipeline + guarded execution) and a builder switch so `strategy.name: phoenix_lpi` can run either legacy LPI (`alpha_mode: lpi`, default) or the proposal’s missing \(Z_t\) pipeline (`alpha_mode: zscore`).
- **Where:** Inserted new strategy/config near the strategy section; extended shim registration; replaced `_build_phoenix_lpi` to route based on `alpha_mode`.
- **Risks:** If your Nautilus build lacks `TimeInForce.IOC` for limit orders (or uses different semantics), IOC-crossing execution may not fill as expected; if your suite `data.include` omits `Bar`, the alpha mode will remain inert (no `on_bar` triggers).
- **Next steps:** Reply CONTINUE for PART 3.
- **Open questions:** (1) What exact `nautilus_trader` version are you on (pip version string)? (2) In your backtests, does `LimitOrder` with `IOC` fill off `QuoteTick` bid/ask as expected, or does it require `TradeTick`? (3) Do you want alpha decisions strictly on 1‑minute bar close (current), or also on a faster timer/quote cadence?

---

PATCHSET: End-to-end online Z_t pipeline (alpha mode) + deterministic IOC-cross execution (PART 2)  
TARGET: existing_code.py  
PART: 2/?>  

PATCH 04: Add AlphaZScoreStrategy (Z_t pipeline + risk sizing + guarded IOC execution)  
INTENT: Implement the missing online alpha pipeline \(Z_t\) (review.md fatal) with deterministic, cost-aware execution using only local CSV-derived streams (bars+trades+quotes).  
OP:
- ACTION: InsertBeforeAnchor  
- LOCATOR: `class NautilusPhase4Imports:`  
NEW_CODE:
```python
class AlphaZScoreStrategyConfig(StrategyConfig, frozen=True):
    """
    Implements review.md fix: define an explicit online Z_t pipeline.

    NOTE: We keep the external suite interface stable by routing this strategy via
    strategy.name="phoenix_lpi" + strategy.config.alpha_mode="zscore" (see _build_phoenix_lpi).
    """
    instrument_id: Any
    trade_size: Decimal  # interpreted as MAX ABS position size in base units (CRO-safe cap)
    alpha_mode: str = "zscore"

    # Volatility feature: Rogers–Satchell per 1m bar, smoothed over a rolling window.
    # review.md: horizon consistency -> we use 1m-bar variance for sizing, not daily YZ.
    vol_window_bars: PositiveInt = 120

    # Robust standardization for features (RollingZScore already implemented in this file).
    z_window_buckets: PositiveInt = 600
    z_kind: str = "median_mad"
    z_min_count: PositiveInt = 30

    # Flow feature: trade-flow imbalance per bar (tOFI-like), normalized to [-1, 1].
    flow_eps: float = 1e-12

    # VPIN toxicity gate (trade-only, volume time). Set vpin_bucket_volume<=0 to disable.
    vpin_bucket_volume: Decimal = Decimal("0")
    vpin_window_buckets: PositiveInt = 50
    vpin_threshold: float = 0.65

    # Liquidity gates from QuoteTick (chap2/chap3): spread/age/top size.
    max_rel_spread_bps: float = 20.0
    quote_max_age_ms: int = 1000
    min_top_size: Decimal = Decimal("0")

    # Signal fusion (review.md minimal viable): Z_t = clip(w_vol*z_vol + w_flow*z_flow, [-3,3]).
    w_vol: float = 1.0
    w_flow: float = 1.0

    # Chapter 3-style sizing (implemented in a bounded way without relying on portfolio equity APIs):
    #   mu = kappa * tanh(Z/z_sat)
    #   f  = clip(mu/(gamma*sigma^2), [-f_max, f_max])
    #   Q_target = trade_size * (f/f_max)  (so trade_size is the hard max position)
    risk_gamma: float = 1.0
    alpha_kappa: float = 0.0   # default 0 => no trading unless explicitly enabled
    alpha_z_sat: float = 1.5
    f_max: float = 1.0

    # Execution controls
    rebalance_epsilon_qty: Decimal = Decimal("0")  # 0 => use instrument.size_increment if available
    slippage_ticks: int = 0
    close_positions_on_stop: bool = True


class AlphaZScoreStrategy(Strategy):
    """
    Event-driven implementation:
      - on_quote_tick: update liquidity state
      - on_trade_tick: update VPIN + per-minute signed flow accumulators
      - on_bar (1m, close): compute RS variance + Z_t, then reconcile to Q_target using IOC-cross orders

    Determinism notes (review.md):
      - Backtests process events by ts_init; in our ETL ts_init==ts_event (ns).
      - PART 1 sets stream priority so QuoteTick > TradeTick > Bar at same ts_init.
    """
    def __init__(self, config: AlphaZScoreStrategyConfig) -> None:
        super().__init__(config=config)
        self.instrument: Instrument | None = None
        self._bar_type = BarType.from_str(f"{config.instrument_id}-1-MINUTE-LAST-EXTERNAL")

        self._var = RollingMeanStd(window=int(config.vol_window_bars))
        self._z_vol = RollingZScore(
            window=int(config.z_window_buckets),
            kind=str(config.z_kind),
            min_count=int(config.z_min_count),
        )
        self._z_flow = RollingZScore(
            window=int(config.z_window_buckets),
            kind=str(config.z_kind),
            min_count=int(config.z_min_count),
        )

        self._flow_buy = Decimal("0")
        self._flow_sell = Decimal("0")

        self._vpin_target = Decimal(str(config.vpin_bucket_volume))
        self._vpin_remaining = self._vpin_target if self._vpin_target > 0 else Decimal("0")
        self._vpin_buy = Decimal("0")
        self._vpin_sell = Decimal("0")
        self._vpin_window = deque(maxlen=int(config.vpin_window_buckets))
        self._vpin_sum_imb = 0.0
        self._vpin_value = float("nan")

        self._last_quote_ts: int | None = None
        self._last_bid: Decimal | None = None
        self._last_ask: Decimal | None = None
        self._last_spread_bps: float | None = None
        self._last_min_top: Decimal | None = None

        self.alpha_z: float = 0.0
        self.sigma2: float = float("nan")

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Instrument not found in cache: {self.config.instrument_id}")
            self.stop()
            return
        self.subscribe_bars(self._bar_type)
        self.subscribe_quote_ticks(self.config.instrument_id)
        self.subscribe_trade_ticks(self.config.instrument_id)

    def on_quote_tick(self, tick: QuoteTick) -> None:
        bid = price_to_decimal(getattr(tick, "bid_price"))
        ask = price_to_decimal(getattr(tick, "ask_price"))
        if bid <= 0 or ask <= 0 or ask < bid:
            return
        mid = (bid + ask) / Decimal("2")
        spr = ask - bid
        rel_bps = float((spr / mid) * Decimal("10000")) if mid > 0 else float("inf")

        bid_sz = qty_to_decimal(getattr(tick, "bid_size"))
        ask_sz = qty_to_decimal(getattr(tick, "ask_size"))

        self._last_quote_ts = int(getattr(tick, "ts_event"))
        self._last_bid = bid
        self._last_ask = ask
        self._last_spread_bps = rel_bps
        self._last_min_top = bid_sz if bid_sz < ask_sz else ask_sz

    def on_trade_tick(self, tick: TradeTick) -> None:
        q = qty_to_decimal(getattr(tick, "size"))
        if q <= 0:
            return
        is_buy = False
        try:
            is_buy = getattr(tick, "aggressor_side") == AggressorSide.BUYER  # type: ignore[attr-defined]
        except Exception:
            is_buy = "BUY" in str(getattr(tick, "aggressor_side", "")).upper()

        if is_buy:
            self._flow_buy += q
        else:
            self._flow_sell += q

        if self._vpin_target <= 0:
            return

        qty_left = q
        while qty_left > 0:
            take = qty_left if qty_left <= self._vpin_remaining else self._vpin_remaining
            if is_buy:
                self._vpin_buy += take
            else:
                self._vpin_sell += take
            self._vpin_remaining -= take
            qty_left -= take

            if self._vpin_remaining == 0:
                imb = abs(float(self._vpin_buy - self._vpin_sell))
                if len(self._vpin_window) == self._vpin_window.maxlen:
                    self._vpin_sum_imb -= self._vpin_window[0]
                self._vpin_window.append(imb)
                self._vpin_sum_imb += imb
                denom = float(len(self._vpin_window)) * float(self._vpin_target)
                self._vpin_value = (self._vpin_sum_imb / denom) if denom > 0 else float("nan")
                self._vpin_remaining = self._vpin_target
                self._vpin_buy = Decimal("0")
                self._vpin_sell = Decimal("0")

    def on_bar(self, bar: Bar) -> None:
        # chap2 §2.1.2 + PART 1: bar ts_event is CLOSE time (ns).
        o = price_to_float(getattr(bar, "open"))
        h = price_to_float(getattr(bar, "high"))
        l = price_to_float(getattr(bar, "low"))
        c = price_to_float(getattr(bar, "close"))
        if o <= 0 or h <= 0 or l <= 0 or c <= 0:
            return

        u = math.log(h / o)
        d = math.log(l / o)
        cc = math.log(c / o)
        rs = (u * (u - cc)) + (d * (d - cc))
        if not math.isfinite(rs) or rs <= 0:
            return

        self._var.push(float(rs))
        self.sigma2 = max(float(self._var.mean), 1e-12)

        z_vol = self._z_vol.zscore(math.log(rs + 1e-12), update=True)

        buy = float(self._flow_buy)
        sell = float(self._flow_sell)
        denom = buy + sell + float(self.config.flow_eps)
        flow_norm = (buy - sell) / denom if denom > 0 else 0.0
        self._flow_buy = Decimal("0")
        self._flow_sell = Decimal("0")
        z_flow = self._z_flow.zscore(flow_norm, update=True)

        z = (float(self.config.w_vol) * z_vol) + (float(self.config.w_flow) * z_flow)
        self.alpha_z = max(-3.0, min(3.0, z))

        q_target = self._target_qty(self.alpha_z, self.sigma2)

        ts_now = int(getattr(bar, "ts_event"))
        if not self._gates_ok(ts_now):
            q_target = Decimal("0")

        q_actual = self._net_pos_qty()
        dQ = q_target - q_actual

        eps = Decimal(str(self.config.rebalance_epsilon_qty))
        if eps <= 0 and self.instrument is not None:
            try:
                eps = qty_to_decimal(getattr(self.instrument, "size_increment", Decimal("0")))
            except Exception:
                eps = Decimal("0")

        if dQ.copy_abs() <= eps:
            return

        self._submit_ioc_cross(dQ)

    def _gates_ok(self, ts_now: int) -> bool:
        if self._last_quote_ts is None or self._last_bid is None or self._last_ask is None:
            return False
        age_ms = (ts_now - int(self._last_quote_ts)) / 1_000_000.0
        if age_ms > float(self.config.quote_max_age_ms):
            return False
        if self._last_spread_bps is None or self._last_spread_bps > float(self.config.max_rel_spread_bps):
            return False
        if Decimal(str(self.config.min_top_size)) > 0:
            if self._last_min_top is None or self._last_min_top < Decimal(str(self.config.min_top_size)):
                return False
        if self._vpin_target > 0 and math.isfinite(self._vpin_value) and self._vpin_value > float(self.config.vpin_threshold):
            return False
        return True

    def _target_qty(self, z: float, sigma2: float) -> Decimal:
        if float(self.config.alpha_kappa) <= 0 or float(self.config.risk_gamma) <= 0 or float(self.config.f_max) <= 0:
            return Decimal("0")
        mu = float(self.config.alpha_kappa) * math.tanh(z / float(self.config.alpha_z_sat))
        f = mu / (float(self.config.risk_gamma) * float(sigma2))
        f_max = float(self.config.f_max)
        f = max(-f_max, min(f_max, f))
        return Decimal(str(self.config.trade_size)) * Decimal(str(f / f_max))

    def _net_pos_qty(self) -> Decimal:
        try:
            np = self.portfolio.net_position(self.config.instrument_id)  # type: ignore[attr-defined]
        except Exception:
            return Decimal("0")
        q = getattr(np, "quantity", np)
        try:
            return qty_to_decimal(q)
        except Exception:
            try:
                return Decimal(str(q))
            except Exception:
                return Decimal("0")

    def _submit_ioc_cross(self, dQ: Decimal) -> None:
        if self.instrument is None:
            return
        side = OrderSide.BUY if dQ > 0 else OrderSide.SELL
        qty_abs = dQ.copy_abs()
        try:
            qty = self.instrument.make_qty(qty_abs)
        except Exception:
            return

        slip = max(0, int(getattr(self.config, "slippage_ticks", 0)))
        if self._last_bid is not None and self._last_ask is not None:
            tick = price_to_decimal(getattr(self.instrument, "price_increment", Decimal("0")))
            if side == OrderSide.BUY:
                px = self._last_ask + (tick * slip)
            else:
                px = self._last_bid - (tick * slip)
            try:
                price = self.instrument.make_price(px)
                order: LimitOrder = self.order_factory.limit(
                    instrument_id=self.config.instrument_id,
                    order_side=side,
                    quantity=qty,
                    price=price,
                    post_only=False,
                    time_in_force=TimeInForce.IOC,
                )
                self.submit_order(order)
                return
            except Exception:
                pass

        order2: MarketOrder = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=side,
            quantity=qty,
            time_in_force=TimeInForce.IOC,
        )
        self.submit_order(order2)

    def on_stop(self) -> None:
        self.cancel_all_orders(self.config.instrument_id)
        if bool(self.config.close_positions_on_stop):
            self.close_all_positions(self.config.instrument_id, reduce_only=True)

    def on_reset(self) -> None:
        self._var = RollingMeanStd(window=int(self.config.vol_window_bars))
        self._z_vol = RollingZScore(window=int(self.config.z_window_buckets), kind=str(self.config.z_kind), min_count=int(self.config.z_min_count))
        self._z_flow = RollingZScore(window=int(self.config.z_window_buckets), kind=str(self.config.z_kind), min_count=int(self.config.z_min_count))
        self._flow_buy = Decimal("0")
        self._flow_sell = Decimal("0")
        self._vpin_remaining = self._vpin_target if self._vpin_target > 0 else Decimal("0")
        self._vpin_buy = Decimal("0")
        self._vpin_sell = Decimal("0")
        self._vpin_window = deque(maxlen=int(self.config.vpin_window_buckets))
        self._vpin_sum_imb = 0.0
        self._vpin_value = float("nan")
        self.alpha_z = 0.0
        self.sigma2 = float("nan")
        self._last_quote_ts = None
        self._last_bid = None
        self._last_ask = None
        self._last_spread_bps = None
        self._last_min_top = None
```

PATCH 05: Register AlphaZScoreStrategy shims  
INTENT: Ensure ImportableStrategyConfig can import the new strategy/config via stable shim module paths.  
OP:
- ACTION: InsertAfterAnchor  
- LOCATOR: `_ensure_module("phoenix_research.strategies.phoenix_lpi").PhoenixLpiStrategyConfig = PhoenixLpiStrategyConfig`  
NEW_CODE:
```python
    # Alpha Z-score strategy (proposal end-to-end Z_t pipeline)
    _ensure_module("phoenix_research.strategies").AlphaZScoreStrategy = AlphaZScoreStrategy
    _ensure_module("phoenix_research.strategies").AlphaZScoreStrategyConfig = AlphaZScoreStrategyConfig
    _ensure_module("phoenix_research.strategies.alpha_zscore").AlphaZScoreStrategy = AlphaZScoreStrategy
    _ensure_module("phoenix_research.strategies.alpha_zscore").AlphaZScoreStrategyConfig = AlphaZScoreStrategyConfig
```

PATCH 06: Route phoenix_lpi builder via alpha_mode (legacy LPI vs Z_t alpha strategy)  
INTENT: Preserve the external `strategy.name: phoenix_lpi` interface while enabling the missing Z_t pipeline via `strategy.config.alpha_mode: zscore`.  
OP:
- ACTION: ReplaceSymbol  
- LOCATOR: `_build_phoenix_lpi`  
NEW_CODE:
```python
def _build_phoenix_lpi(N: NautilusPhase4Imports, inst: Any, cfg: dict[str, Any]) -> Any:
    trade_size = _require_decimal(cfg, "trade_size")
    mode = str(cfg.get("alpha_mode") or "lpi").strip().lower()

    if mode in ("lpi", "phoenix_lpi"):
        config_obj: dict[str, Any] = {
            "instrument_id": inst,
            "trade_size": trade_size,
        }
        allowed = {
            "bucket_mode",
            "bucket_interval_ms",
            "bucket_notional_threshold",
            "eta_bps",
            "use_depth_pct",
            "depth_pct_abs",
            "use_zscore",
            "z_window_buckets",
            "z_kind",
            "z_min_count",
            "lambda_ld",
            "lambda_exh_liq",
            "lambda_exh_spr",
            "theta_lpi",
            "theta_exh_low",
            "theta_exh_high",
            "lpi_exit_abs",
            "max_rel_spread_bps",
            "min_depth_notional",
            "quote_max_age_ms",
            "depth_max_age_ms",
            "momentum_use_market",
            "mean_rev_post_only",
            "mean_rev_improve_ticks",
            "entry_ttl_buckets",
            "max_hold_buckets_momentum",
            "max_hold_buckets_mean_rev",
            "close_positions_on_stop",
            "reduce_only_on_stop",
        }
        for k, v in cfg.items():
            if k in ("instrument_id", "trade_size"):
                continue
            if k not in allowed:
                raise StrategyFactoryError(f"Unsupported PhoenixLpiStrategyConfig key: {k!r}")
            config_obj[k] = v
        kwargs = {
            "strategy_path": "phoenix_research.strategies.phoenix_lpi:PhoenixLpiStrategy",
            "config_path": "phoenix_research.strategies.phoenix_lpi:PhoenixLpiStrategyConfig",
            "config": config_obj,
        }
        return _call_with_supported_kwargs(N.ImportableStrategyConfig, kwargs)

    if mode in ("zscore", "alpha", "alpha_zscore"):
        # review.md fatal fix: explicit Z_t pipeline strategy (bars+trades+quotes), routed via phoenix_lpi entrypoint.
        config_obj2: dict[str, Any] = {
            "instrument_id": inst,
            "trade_size": trade_size,
            "alpha_mode": "zscore",
        }
        allowed2 = {
            "alpha_mode",
            "vol_window_bars",
            "z_window_buckets",
            "z_kind",
            "z_min_count",
            "flow_eps",
            "vpin_bucket_volume",
            "vpin_window_buckets",
            "vpin_threshold",
            "max_rel_spread_bps",
            "quote_max_age_ms",
            "min_top_size",
            "w_vol",
            "w_flow",
            "risk_gamma",
            "alpha_kappa",
            "alpha_z_sat",
            "f_max",
            "rebalance_epsilon_qty",
            "slippage_ticks",
            "close_positions_on_stop",
        }
        for k, v in cfg.items():
            if k in ("instrument_id", "trade_size"):
                continue
            if k not in allowed2:
                raise StrategyFactoryError(f"Unsupported AlphaZScoreStrategyConfig key: {k!r}")
            config_obj2[k] = v
        kwargs2 = {
            "strategy_path": "phoenix_research.strategies.alpha_zscore:AlphaZScoreStrategy",
            "config_path": "phoenix_research.strategies.alpha_zscore:AlphaZScoreStrategyConfig",
            "config": config_obj2,
        }
        return _call_with_supported_kwargs(N.ImportableStrategyConfig, kwargs2)

    raise StrategyFactoryError(f"Unsupported alpha_mode for phoenix_lpi: {mode!r}")
```

---

### How to apply
1) Apply PART 1 patches (already accepted) to `existing_code.py`.  
2) Apply PART 2 patches above to the same file.  
3) In your Phase4 suite YAML, keep `strategy.name: phoenix_lpi` and set `strategy.config.alpha_mode: zscore`.  
4) Ensure `data.include` contains at least: `QuoteTick`, `TradeTick`, `Bar`.  
5) Re-run `ingest` (so bars are close-timestamped per PART 1) and then run a short backtest window.  
6) If fills don’t occur, confirm QuoteTick/TradeTick data exists in that window and report your Nautilus version + whether IOC limit fills off quotes.
7) 
