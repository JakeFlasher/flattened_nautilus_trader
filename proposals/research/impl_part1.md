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
