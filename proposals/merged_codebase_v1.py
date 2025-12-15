import argparse
import csv
import hashlib
import importlib
import inspect
import json
import math
import os
import platform
import random
import re
import statistics
import sys
import types
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

import pandas as pd
import yaml


def _call_with_supported_kwargs(fn: Any, kwargs: dict[str, Any]) -> Any:
    try:
        sig = inspect.signature(fn)
    except Exception:
        return fn(**kwargs)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(**kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in params}
    return fn(**filtered)


def _ensure_module(name: str) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is not None:
        return mod
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    if "." in name:
        parent_name, child = name.rsplit(".", 1)
        parent = _ensure_module(parent_name)
        setattr(parent, child, mod)
    return mod


def _install_shims() -> None:
    m = _ensure_module("phoenix_etl")
    m.__dict__["__version__"] = "0.1.0"
    _ensure_module("phoenix_etl.nautilus_compat").load_nautilus_imports = load_nautilus_imports
    _ensure_module("phoenix_etl.custom_data").BinanceFundingRate = BinanceFundingRate
    _ensure_module("phoenix_etl.custom_data").BinanceOiMetrics = BinanceOiMetrics
    _ensure_module("phoenix_etl.custom_data").BinanceBookDepthPct = BinanceBookDepthPct

    mr = _ensure_module("phoenix_research")
    mr.__dict__["__version__"] = "0.1.0"
    _ensure_module("phoenix_research.nautilus_utils").price_to_float = price_to_float
    _ensure_module("phoenix_research.nautilus_utils").qty_to_float = qty_to_float
    _ensure_module("phoenix_research.nautilus_utils").price_to_decimal = price_to_decimal
    _ensure_module("phoenix_research.nautilus_utils").qty_to_decimal = qty_to_decimal
    _ensure_module("phoenix_research.nautilus_utils").safe_div = safe_div

    _ensure_module("phoenix_research.stats").RollingMeanStd = RollingMeanStd
    _ensure_module("phoenix_research.stats").RollingMedianMad = RollingMedianMad
    _ensure_module("phoenix_research.stats").RollingSum = RollingSum
    _ensure_module("phoenix_research.stats").RollingZScore = RollingZScore
    _ensure_module("phoenix_research.stats.rolling").RollingMeanStd = RollingMeanStd
    _ensure_module("phoenix_research.stats.rolling").RollingMedianMad = RollingMedianMad
    _ensure_module("phoenix_research.stats.rolling").RollingSum = RollingSum
    _ensure_module("phoenix_research.stats.rolling").RollingZScore = RollingZScore

    _ensure_module("phoenix_research.signals").LpiBucketMode = LpiBucketMode
    _ensure_module("phoenix_research.signals").LpiSignalConfig = LpiSignalConfig
    _ensure_module("phoenix_research.signals").LpiSignalEngine = LpiSignalEngine
    _ensure_module("phoenix_research.signals").LpiSnapshot = LpiSnapshot
    _ensure_module("phoenix_research.signals.lpi").LpiBucketMode = LpiBucketMode
    _ensure_module("phoenix_research.signals.lpi").LpiSignalConfig = LpiSignalConfig
    _ensure_module("phoenix_research.signals.lpi").LpiSignalEngine = LpiSignalEngine
    _ensure_module("phoenix_research.signals.lpi").LpiSnapshot = LpiSnapshot

    _ensure_module("phoenix_research.strategies").EmaCrossBaseline = EmaCrossBaseline
    _ensure_module("phoenix_research.strategies").EmaCrossBaselineConfig = EmaCrossBaselineConfig
    _ensure_module("phoenix_research.strategies").PhoenixLpiStrategy = PhoenixLpiStrategy
    _ensure_module("phoenix_research.strategies").PhoenixLpiStrategyConfig = PhoenixLpiStrategyConfig

    _ensure_module("phoenix_research.strategies.ema_cross_baseline").EmaCrossBaseline = EmaCrossBaseline
    _ensure_module("phoenix_research.strategies.ema_cross_baseline").EmaCrossBaselineConfig = EmaCrossBaselineConfig
    _ensure_module("phoenix_research.strategies.phoenix_lpi").PhoenixLpiStrategy = PhoenixLpiStrategy
    _ensure_module("phoenix_research.strategies.phoenix_lpi").PhoenixLpiStrategyConfig = PhoenixLpiStrategyConfig


@dataclass(frozen=True)
class NautilusImports:
    InstrumentId: Any
    TradeId: Any
    Price: Any
    Quantity: Any
    AggressorSide: Any
    TradeTick: Any
    QuoteTick: Any
    Bar: Any
    BarType: Any
    Data: Any
    customdataclass: Any
    ParquetDataCatalog: Any


_NAUTILUS_IMPORTS_CACHE: NautilusImports | None = None


def load_nautilus_imports() -> NautilusImports:
    global _NAUTILUS_IMPORTS_CACHE
    if _NAUTILUS_IMPORTS_CACHE is not None:
        return _NAUTILUS_IMPORTS_CACHE

    try:
        from nautilus_trader.model.identifiers import InstrumentId
    except Exception:
        from nautilus_trader.model import InstrumentId

    try:
        from nautilus_trader.model.identifiers import TradeId
    except Exception:
        from nautilus_trader.model import TradeId

    from nautilus_trader.model.objects import Price, Quantity
    from nautilus_trader.model.enums import AggressorSide
    from nautilus_trader.model.data import Bar, BarType, QuoteTick, TradeTick

    try:
        from nautilus_trader.core.data import Data
    except Exception:
        from nautilus_trader.core import Data

    from nautilus_trader.model.custom import customdataclass

    try:
        from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
    except Exception:
        from nautilus_trader.persistence.catalog import ParquetDataCatalog

    _NAUTILUS_IMPORTS_CACHE = NautilusImports(
        InstrumentId=InstrumentId,
        TradeId=TradeId,
        Price=Price,
        Quantity=Quantity,
        AggressorSide=AggressorSide,
        TradeTick=TradeTick,
        QuoteTick=QuoteTick,
        Bar=Bar,
        BarType=BarType,
        Data=Data,
        customdataclass=customdataclass,
        ParquetDataCatalog=ParquetDataCatalog,
    )
    return _NAUTILUS_IMPORTS_CACHE


N2 = load_nautilus_imports()
DataBase = N2.Data
customdataclass = N2.customdataclass
InstrumentIdCls = N2.InstrumentId


@customdataclass
class BinanceFundingRate(DataBase):
    instrument_id: InstrumentIdCls
    rate_str: str
    interval_hours: int


@customdataclass
class BinanceOiMetrics(DataBase):
    instrument_id: InstrumentIdCls
    symbol: str
    sum_open_interest_str: str
    sum_open_interest_value_str: str
    count_toptrader_long_short_ratio_str: str | None = None
    sum_toptrader_long_short_ratio_str: str | None = None
    count_long_short_ratio_str: str | None = None
    sum_taker_long_short_vol_ratio_str: str | None = None


@customdataclass
class BinanceBookDepthPct(DataBase):
    instrument_id: InstrumentIdCls
    percentage_str: str
    depth_str: str
    notional_str: str


NANOS_PER_MILLI = 1_000_000
NANOS_PER_SECOND = 1_000_000_000


class TimestampParseError(ValueError):
    pass


def ms_to_ns(ms: str | int) -> int:
    if isinstance(ms, int):
        return ms * NANOS_PER_MILLI
    s = str(ms).strip()
    if not s:
        raise TimestampParseError("empty ms timestamp")
    if s.isdigit() or (s[0] == "-" and s[1:].isdigit()):
        return int(s) * NANOS_PER_MILLI
    raise TimestampParseError(f"non-integer ms timestamp: {ms!r}")


def iso_to_ns(s: str) -> int:
    raw = str(s).strip()
    if not raw:
        raise TimestampParseError("empty ISO timestamp")
    raw = raw.replace("Z", "+00:00")
    if " " in raw and "T" not in raw:
        raw = raw.replace(" ", "T", 1)
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError as e:
        raise TimestampParseError(f"invalid ISO timestamp: {s!r}") from e
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return int(dt.timestamp() * NANOS_PER_SECOND)


def parse_ts_to_ns(value: str | int) -> int:
    if isinstance(value, int):
        return ms_to_ns(value)
    s = str(value).strip()
    if not s:
        raise TimestampParseError("empty timestamp")
    if s.isdigit() or (s[0] == "-" and s[1:].isdigit()):
        return ms_to_ns(s)
    return iso_to_ns(s)


DATE_DAILY_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


@dataclass(frozen=True)
class ParsedPath:
    path: Path
    universe: str
    period: str
    dataset: str
    symbol: str
    date_key: str


def _extract_symbol_and_date_from_filename(filename: str) -> tuple[str, str | None, str | None]:
    parts = filename.split("-")
    if not parts:
        return "", None, None
    symbol = parts[0]

    m_daily = DATE_DAILY_RE.search(filename)
    if m_daily:
        y, mo, d = m_daily.groups()
        return symbol, f"{y}-{mo}-{d}", None

    m_month = re.search(r"(\d{4})-(\d{2})\.csv$", filename)
    if m_month:
        y, mo = m_month.groups()
        return symbol, None, f"{y}-{mo}"

    return symbol, None, None


def iter_candidate_csv_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        yield from root.rglob("*.csv")


def classify_binance_vision_path(path: Path) -> ParsedPath | None:
    parts = [p.lower() for p in path.parts]
    universe = None
    period = None

    if "spot_data" in parts:
        universe = "spot"
    if "future_data" in parts or "futures" in parts or ("data" in parts and "raw" in parts and "futures" in parts):
        universe = "futures" if universe != "spot" else "futures"

    if universe is None:
        p = str(path).lower()
        if "/spot_data/" in p:
            universe = "spot"
        elif "/future_data/" in p or "/data/raw/futures/" in p:
            universe = "futures"
        else:
            return None

    if "daily" in parts or "daily_data" in parts:
        period = "daily"
    elif "monthly" in parts or "monthly_data" in parts:
        period = "monthly"
    else:
        return None

    dataset = path.parent.name
    symbol, daily_key, monthly_key = _extract_symbol_and_date_from_filename(path.name)
    if not symbol:
        return None
    date_key = daily_key if period == "daily" else monthly_key
    if date_key is None:
        return None

    return ParsedPath(
        path=path,
        universe=universe,
        period=period,
        dataset=dataset,
        symbol=symbol,
        date_key=date_key,
    )


@dataclass(frozen=True)
class InventoryRow:
    universe: str
    period: str
    dataset: str
    symbol: str
    date_key: str
    path: str
    size_bytes: int
    mtime_ns: int


def build_inventory(roots: Iterable[Path]) -> list[InventoryRow]:
    rows: list[InventoryRow] = []
    for p in iter_candidate_csv_files(roots):
        parsed = classify_binance_vision_path(p)
        if parsed is None:
            continue
        st = p.stat()
        rows.append(
            InventoryRow(
                universe=parsed.universe,
                period=parsed.period,
                dataset=parsed.dataset,
                symbol=parsed.symbol,
                date_key=parsed.date_key,
                path=str(p),
                size_bytes=st.st_size,
                mtime_ns=getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000)),
            ),
        )
    rows.sort(key=lambda r: (r.universe, r.period, r.dataset, r.symbol, r.date_key, r.path))
    return rows


def _normalize_filter_key_for_row(row_period: str, key: str | None) -> str | None:
    if key is None:
        return None
    s = str(key).strip()
    if not s:
        return None
    if row_period == "monthly":
        if len(s) >= 7:
            return s[:7]
    return s


def filter_inventory(
    rows: list[InventoryRow],
    universe: str,
    datasets: set[str],
    symbols: set[str] | None,
    start_date_key: str | None,
    end_date_key: str | None,
) -> list[InventoryRow]:
    out: list[InventoryRow] = []
    for r in rows:
        if r.universe != universe:
            continue
        if r.dataset not in datasets:
            continue
        if symbols is not None and r.symbol not in symbols:
            continue

        start_key = _normalize_filter_key_for_row(r.period, start_date_key)
        end_key = _normalize_filter_key_for_row(r.period, end_date_key)

        if start_key is not None and r.date_key < start_key:
            continue
        if end_key is not None and r.date_key > end_key:
            continue

        out.append(r)

    out.sort(key=lambda r: (r.period, r.dataset, r.symbol, r.date_key, r.path))
    return out


def default_roots_from_args(raw_roots: list[str]) -> list[Path]:
    roots: list[Path] = []
    for s in raw_roots:
        p = Path(os.path.expanduser(s)).resolve()
        roots.append(p)
    return roots


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _stable_signature_for_file(path: str, size_bytes: int, mtime_ns: int) -> str:
    h = hashlib.sha256()
    h.update(path.encode("utf-8"))
    h.update(b"|")
    h.update(str(size_bytes).encode("utf-8"))
    h.update(b"|")
    h.update(str(mtime_ns).encode("utf-8"))
    return h.hexdigest()


@dataclass
class FileIngestStats:
    path: str
    universe: str
    period: str
    dataset: str
    symbol: str
    date_key: str
    size_bytes: int
    mtime_ns: int
    signature: str
    rows_read: int = 0
    events_written: int = 0
    parse_errors: int = 0
    dedup_dropped: int = 0
    ts_min: int | None = None
    ts_max: int | None = None


@dataclass
class DatasetStats:
    dataset: str
    events_written: int = 0
    parse_errors: int = 0
    dedup_dropped: int = 0
    ts_min: int | None = None
    ts_max: int | None = None


@dataclass
class Manifest:
    ingestion_version: str = "0.1.0"
    created_utc: str = field(default_factory=_now_utc_iso)
    raw_roots: list[str] = field(default_factory=list)
    catalog_path: str = ""
    venue: str = "BINANCE"
    universe: str = "futures"
    symbols: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    start_date_key: str | None = None
    end_date_key: str | None = None
    environment: dict[str, Any] = field(default_factory=dict)
    by_file: list[FileIngestStats] = field(default_factory=list)
    by_dataset: dict[str, DatasetStats] = field(default_factory=dict)

    def finalize(self) -> None:
        self.environment = {
            "python": sys.version,
            "platform": platform.platform(),
        }
        agg: dict[str, DatasetStats] = {}
        for f in self.by_file:
            ds = agg.get(f.dataset)
            if ds is None:
                ds = DatasetStats(dataset=f.dataset)
                agg[f.dataset] = ds
            ds.events_written += f.events_written
            ds.parse_errors += f.parse_errors
            ds.dedup_dropped += f.dedup_dropped
            if f.ts_min is not None:
                ds.ts_min = f.ts_min if ds.ts_min is None else min(ds.ts_min, f.ts_min)
            if f.ts_max is not None:
                ds.ts_max = f.ts_max if ds.ts_max is None else max(ds.ts_max, f.ts_max)
        self.by_dataset = agg

    def to_json(self) -> str:
        self.finalize()
        obj = asdict(self)
        return json.dumps(obj, indent=2, sort_keys=True)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")


def build_file_stats_base(
    *,
    path: str,
    universe: str,
    period: str,
    dataset: str,
    symbol: str,
    date_key: str,
    size_bytes: int,
    mtime_ns: int,
) -> FileIngestStats:
    return FileIngestStats(
        path=path,
        universe=universe,
        period=period,
        dataset=dataset,
        symbol=symbol,
        date_key=date_key,
        size_bytes=size_bytes,
        mtime_ns=mtime_ns,
        signature=_stable_signature_for_file(path, size_bytes, mtime_ns),
    )


@dataclass(frozen=True)
class EtlConfig:
    raw_roots: list[str]
    catalog_path: str
    manifest_path: str
    validation_report_path: str
    universe: Literal["futures", "spot"] = "futures"
    venue: str = "BINANCE"
    symbols: list[str] | None = None
    datasets: list[str] = field(default_factory=list)
    start_date_key: str | None = None
    end_date_key: str | None = None
    batch_size: int = 50_000
    strict: bool = False
    sort_within_batch: bool = True

    @staticmethod
    def default_datasets_for_universe(universe: str) -> list[str]:
        if universe == "futures":
            return [
                "aggTrades",
                "bookTicker",
                "klines_1m",
                "metrics",
                "bookDepth",
                "fundingRate",
            ]
        return ["aggTrades", "trades", "klines"]


def load_etl_config(path: Path) -> EtlConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config must be a YAML mapping")
    universe = str(data.get("universe", "futures"))
    datasets = data.get("datasets")
    if datasets is None:
        datasets = EtlConfig.default_datasets_for_universe(universe)
    return EtlConfig(
        raw_roots=list(data["raw_roots"]),
        catalog_path=str(data["catalog_path"]),
        manifest_path=str(data["manifest_path"]),
        validation_report_path=str(data.get("validation_report_path", "validation_report.md")),
        universe=universe,
        venue=str(data.get("venue", "BINANCE")),
        symbols=data.get("symbols"),
        datasets=list(datasets),
        start_date_key=data.get("start_date_key"),
        end_date_key=data.get("end_date_key"),
        batch_size=int(data.get("batch_size", 50_000)),
        strict=bool(data.get("strict", False)),
        sort_within_batch=bool(data.get("sort_within_batch", True)),
    )


@dataclass(frozen=True)
class ParseResult:
    events: list[Any]
    rows_read: int
    events_emitted: int
    parse_errors: int
    dedup_dropped: int
    ts_min: int | None
    ts_max: int | None


class CsvParseError(ValueError):
    pass


def _looks_like_header(row0: list[str]) -> bool:
    if not row0:
        return False
    for c in row0:
        s = c.strip()
        if not s:
            continue
        if any(ch.isalpha() for ch in s) or "_" in s:
            return True
    return False


def iter_csv_rows(path: Path) -> Iterator[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            yield row


def detect_header(path: Path) -> bool:
    it = iter_csv_rows(path)
    try:
        row0 = next(it)
    except StopIteration:
        return False
    return _looks_like_header(row0)


def parse_bool(val: str) -> bool:
    s = str(val).strip().lower()
    if s in ("true", "t", "1", "yes", "y"):
        return True
    if s in ("false", "f", "0", "no", "n"):
        return False
    raise CsvParseError(f"invalid boolean: {val!r}")


def safe_parse_ts(val: str | int, strict: bool) -> tuple[int | None, int]:
    try:
        return parse_ts_to_ns(val), 0
    except (TimestampParseError, ValueError):
        if strict:
            raise
        return None, 1


def finalize_batch(
    events: list[Any],
    *,
    sort_key: Any,
    sort_within_batch: bool,
) -> list[Any]:
    if not events:
        return events
    if sort_within_batch:
        events.sort(key=sort_key)
    return events


class BaseParser:
    dataset: str

    def parse_file(
        self,
        path: Path,
        instrument_id: Any,
        *,
        strict: bool,
        sort_within_batch: bool,
        batch_size: int,
    ) -> Iterable[ParseResult]:
        raise NotImplementedError


class FuturesAggTradesParser(BaseParser):
    dataset = "aggTrades"

    def parse_file(
        self,
        path: Path,
        instrument_id: Any,
        *,
        strict: bool,
        sort_within_batch: bool,
        batch_size: int,
    ) -> Iterable[ParseResult]:
        N = load_nautilus_imports()
        TradeTick = N.TradeTick
        TradeId = N.TradeId
        Price = N.Price
        Quantity = N.Quantity
        AggressorSide = N.AggressorSide

        rows_read = 0
        errors = 0
        dedup = 0
        ts_min = None
        ts_max = None
        last_trade_id: str | None = None
        events: list[Any] = []

        if detect_header(path):
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows_read += 1
                    try:
                        r = {str(k).strip().lower(): ("" if v is None else str(v).strip()) for k, v in row.items()}
                        trade_id = r.get("agg_trade_id") or r.get("aggtradeid") or r.get("a") or r.get("id")
                        price_s = r.get("price") or r.get("p")
                        qty_s = r.get("quantity") or r.get("qty") or r.get("q")
                        ts_ms = r.get("transact_time") or r.get("transacttime") or r.get("timestamp") or r.get("t")
                        is_buyer_maker_s = r.get("is_buyer_maker") or r.get("isbuyermaker") or r.get("m")

                        if trade_id is None or price_s is None or qty_s is None or ts_ms is None or is_buyer_maker_s is None:
                            raise ValueError("missing required columns")

                        ts_event, e = safe_parse_ts(ts_ms, strict)
                        errors += e
                        if ts_event is None:
                            continue

                        if last_trade_id is not None and str(trade_id) == last_trade_id:
                            dedup += 1
                            continue
                        last_trade_id = str(trade_id)

                        is_buyer_maker = parse_bool(is_buyer_maker_s)
                        side = AggressorSide.SELLER if is_buyer_maker else AggressorSide.BUYER

                        tick = TradeTick(
                            instrument_id=instrument_id,
                            price=Price.from_str(price_s),
                            size=Quantity.from_str(qty_s),
                            aggressor_side=side,
                            trade_id=TradeId(str(trade_id)),
                            ts_event=ts_event,
                            ts_init=ts_event,
                        )
                        events.append(tick)
                        ts_min = ts_event if ts_min is None else min(ts_min, ts_event)
                        ts_max = ts_event if ts_max is None else max(ts_max, ts_event)

                        if len(events) >= batch_size:
                            batch = finalize_batch(
                                events,
                                sort_key=lambda x: (x.ts_event, str(x.trade_id)),
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
                        if len(row) < 7:
                            raise ValueError("row too short")
                        trade_id, price_s, qty_s = row[0], row[1], row[2]
                        ts_ms = row[5]
                        is_buyer_maker_s = row[6]

                        ts_event, e = safe_parse_ts(ts_ms, strict)
                        errors += e
                        if ts_event is None:
                            continue

                        if last_trade_id is not None and str(trade_id) == last_trade_id:
                            dedup += 1
                            continue
                        last_trade_id = str(trade_id)

                        is_buyer_maker = parse_bool(is_buyer_maker_s)
                        side = AggressorSide.SELLER if is_buyer_maker else AggressorSide.BUYER

                        tick = TradeTick(
                            instrument_id=instrument_id,
                            price=Price.from_str(str(price_s).strip()),
                            size=Quantity.from_str(str(qty_s).strip()),
                            aggressor_side=side,
                            trade_id=TradeId(str(trade_id).strip()),
                            ts_event=ts_event,
                            ts_init=ts_event,
                        )
                        events.append(tick)
                        ts_min = ts_event if ts_min is None else min(ts_min, ts_event)
                        ts_max = ts_event if ts_max is None else max(ts_max, ts_event)

                        if len(events) >= batch_size:
                            batch = finalize_batch(
                                events,
                                sort_key=lambda x: (x.ts_event, str(x.trade_id)),
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
                sort_key=lambda x: (x.ts_event, str(x.trade_id)),
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


class FuturesBookTickerParser(BaseParser):
    dataset = "bookTicker"

    def parse_file(
        self,
        path: Path,
        instrument_id: Any,
        *,
        strict: bool,
        sort_within_batch: bool,
        batch_size: int,
    ) -> Iterable[ParseResult]:
        N = load_nautilus_imports()
        QuoteTick = N.QuoteTick
        Price = N.Price
        Quantity = N.Quantity

        rows_read = 0
        errors = 0
        dedup = 0
        ts_min = None
        ts_max = None
        last_key: tuple[int, str] | None = None
        events: list[Any] = []

        if detect_header(path):
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows_read += 1
                    try:
                        r = {str(k).strip().lower(): ("" if v is None else str(v).strip()) for k, v in row.items()}
                        update_id = r.get("update_id") or r.get("updateid") or r.get("u")
                        bid_p = r.get("best_bid_price") or r.get("bid_price") or r.get("bidprice") or r.get("b")
                        bid_q = r.get("best_bid_qty") or r.get("bid_qty") or r.get("bidqty") or r.get("bq")
                        ask_p = r.get("best_ask_price") or r.get("ask_price") or r.get("askprice") or r.get("a")
                        ask_q = r.get("best_ask_qty") or r.get("ask_qty") or r.get("askqty") or r.get("aq")
                        event_time = r.get("event_time") or r.get("eventtime") or r.get("e")
                        transact_time = r.get("transaction_time") or r.get("transactiontime") or r.get("t")

                        ts_field = event_time if (event_time is not None and event_time != "") else transact_time
                        if update_id is None or bid_p is None or bid_q is None or ask_p is None or ask_q is None or ts_field is None:
                            raise ValueError("missing required columns")

                        ts_event, e = safe_parse_ts(ts_field, strict)
                        errors += e
                        if ts_event is None:
                            continue

                        key = (ts_event, str(update_id))
                        if last_key is not None and key == last_key:
                            dedup += 1
                            continue
                        last_key = key

                        tick = QuoteTick(
                            instrument_id=instrument_id,
                            bid_price=Price.from_str(bid_p),
                            ask_price=Price.from_str(ask_p),
                            bid_size=Quantity.from_str(bid_q),
                            ask_size=Quantity.from_str(ask_q),
                            ts_event=ts_event,
                            ts_init=ts_event,
                        )
                        events.append(tick)
                        ts_min = ts_event if ts_min is None else min(ts_min, ts_event)
                        ts_max = ts_event if ts_max is None else max(ts_max, ts_event)

                        if len(events) >= batch_size:
                            batch = finalize_batch(
                                events,
                                sort_key=lambda x: (x.ts_event, str(x.bid_price), str(x.ask_price)),
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
                        update_id = row[0]
                        bid_p, bid_q, ask_p, ask_q = row[1], row[2], row[3], row[4]
                        transact_time = row[5]
                        event_time = row[6] if len(row) > 6 else ""
                        ts_field = event_time.strip() or transact_time.strip()

                        ts_event, e = safe_parse_ts(ts_field, strict)
                        errors += e
                        if ts_event is None:
                            continue

                        key = (ts_event, str(update_id))
                        if last_key is not None and key == last_key:
                            dedup += 1
                            continue
                        last_key = key

                        tick = QuoteTick(
                            instrument_id=instrument_id,
                            bid_price=Price.from_str(str(bid_p).strip()),
                            ask_price=Price.from_str(str(ask_p).strip()),
                            bid_size=Quantity.from_str(str(bid_q).strip()),
                            ask_size=Quantity.from_str(str(ask_q).strip()),
                            ts_event=ts_event,
                            ts_init=ts_event,
                        )
                        events.append(tick)
                        ts_min = ts_event if ts_min is None else min(ts_min, ts_event)
                        ts_max = ts_event if ts_max is None else max(ts_max, ts_event)

                        if len(events) >= batch_size:
                            batch = finalize_batch(
                                events,
                                sort_key=lambda x: (x.ts_event, str(x.bid_price), str(x.ask_price)),
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
                sort_key=lambda x: (x.ts_event, str(x.bid_price), str(x.ask_price)),
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
        N = load_nautilus_imports()
        Bar = N.Bar
        BarType = N.BarType
        Price = N.Price
        Quantity = N.Quantity

        bar_type = BarType.from_str(f"{instrument_id}-1-MINUTE-LAST-EXTERNAL")

        rows_read = 0
        errors = 0
        dedup = 0
        ts_min = None
        ts_max = None
        last_open_time: int | None = None
        events: list[Any] = []

        if detect_header(path):
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows_read += 1
                    try:
                        r = {str(k).strip().lower(): ("" if v is None else str(v).strip()) for k, v in row.items()}
                        open_time = r.get("open_time") or r.get("opentime") or r.get("t")
                        o = r.get("open") or r.get("o")
                        h = r.get("high") or r.get("h")
                        l = r.get("low") or r.get("l")
                        c = r.get("close") or r.get("c")
                        v = r.get("volume") or r.get("v")

                        if open_time is None or o is None or h is None or l is None or c is None or v is None:
                            raise ValueError("missing required columns")

                        ts_event, e = safe_parse_ts(open_time, strict)
                        errors += e
                        if ts_event is None:
                            continue

                        if last_open_time is not None and ts_event == last_open_time:
                            dedup += 1
                            continue
                        last_open_time = ts_event

                        bar = Bar(
                            bar_type=bar_type,
                            open=Price.from_str(o),
                            high=Price.from_str(h),
                            low=Price.from_str(l),
                            close=Price.from_str(c),
                            volume=Quantity.from_str(v),
                            ts_event=ts_event,
                            ts_init=ts_event,
                        )
                        events.append(bar)
                        ts_min = ts_event if ts_min is None else min(ts_min, ts_event)
                        ts_max = ts_event if ts_max is None else max(ts_max, ts_event)

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

                        ts_event, e = safe_parse_ts(open_time, strict)
                        errors += e
                        if ts_event is None:
                            continue

                        if last_open_time is not None and ts_event == last_open_time:
                            dedup += 1
                            continue
                        last_open_time = ts_event

                        bar = Bar(
                            bar_type=bar_type,
                            open=Price.from_str(o.strip()),
                            high=Price.from_str(h.strip()),
                            low=Price.from_str(l.strip()),
                            close=Price.from_str(c.strip()),
                            volume=Quantity.from_str(v.strip()),
                            ts_event=ts_event,
                            ts_init=ts_event,
                        )
                        events.append(bar)
                        ts_min = ts_event if ts_min is None else min(ts_min, ts_event)
                        ts_max = ts_event if ts_max is None else max(ts_max, ts_event)

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


class FuturesMetricsParser(BaseParser):
    dataset = "metrics"

    def parse_file(
        self,
        path: Path,
        instrument_id: Any,
        *,
        strict: bool,
        sort_within_batch: bool,
        batch_size: int,
    ) -> Iterable[ParseResult]:
        rows_read = 0
        errors = 0
        dedup = 0
        ts_min = None
        ts_max = None
        last_ts: int | None = None
        events: list[Any] = []

        if detect_header(path):
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows_read += 1
                    try:
                        r = {str(k).strip().lower(): ("" if v is None else str(v).strip()) for k, v in row.items()}
                        create_time = r.get("create_time") or r.get("createtime")
                        symbol = r.get("symbol") or ""
                        oi_qty = r.get("sum_open_interest") or r.get("sumopeninterest")
                        oi_val = r.get("sum_open_interest_value") or r.get("sumopeninterestvalue")

                        if create_time is None or oi_qty is None or oi_val is None:
                            raise ValueError("missing required columns")

                        ts_event, e = safe_parse_ts(create_time, strict)
                        errors += e
                        if ts_event is None:
                            continue

                        if last_ts is not None and ts_event == last_ts:
                            dedup += 1
                            continue
                        last_ts = ts_event

                        data = BinanceOiMetrics(
                            instrument_id=instrument_id,
                            symbol=symbol,
                            sum_open_interest_str=str(oi_qty),
                            sum_open_interest_value_str=str(oi_val),
                            count_toptrader_long_short_ratio_str=r.get("count_toptrader_long_short_ratio"),
                            sum_toptrader_long_short_ratio_str=r.get("sum_toptrader_long_short_ratio"),
                            count_long_short_ratio_str=r.get("count_long_short_ratio"),
                            sum_taker_long_short_vol_ratio_str=r.get("sum_taker_long_short_vol_ratio"),
                            ts_event=ts_event,
                            ts_init=ts_event,
                        )
                        events.append(data)
                        ts_min = ts_event if ts_min is None else min(ts_min, ts_event)
                        ts_max = ts_event if ts_max is None else max(ts_max, ts_event)

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
                        if len(row) < 4:
                            raise ValueError("row too short")
                        create_time, symbol, oi_qty, oi_val = row[0], row[1], row[2], row[3]

                        ts_event, e = safe_parse_ts(create_time, strict)
                        errors += e
                        if ts_event is None:
                            continue

                        if last_ts is not None and ts_event == last_ts:
                            dedup += 1
                            continue
                        last_ts = ts_event

                        data = BinanceOiMetrics(
                            instrument_id=instrument_id,
                            symbol=str(symbol),
                            sum_open_interest_str=str(oi_qty),
                            sum_open_interest_value_str=str(oi_val),
                            ts_event=ts_event,
                            ts_init=ts_event,
                        )
                        events.append(data)
                        ts_min = ts_event if ts_min is None else min(ts_min, ts_event)
                        ts_max = ts_event if ts_max is None else max(ts_max, ts_event)

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


class FuturesBookDepthPctParser(BaseParser):
    dataset = "bookDepth"

    def parse_file(
        self,
        path: Path,
        instrument_id: Any,
        *,
        strict: bool,
        sort_within_batch: bool,
        batch_size: int,
    ) -> Iterable[ParseResult]:
        rows_read = 0
        errors = 0
        dedup = 0
        ts_min = None
        ts_max = None
        last_key: tuple[int, str] | None = None
        events: list[Any] = []

        if detect_header(path):
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows_read += 1
                    try:
                        r = {str(k).strip().lower(): ("" if v is None else str(v).strip()) for k, v in row.items()}
                        ts_raw = r.get("timestamp") or r.get("time") or r.get("t")
                        pct = r.get("percentage") or r.get("pct")
                        depth = r.get("depth") or r.get("quantity") or r.get("qty")
                        notional = r.get("notional") or r.get("quote") or r.get("value")

                        if ts_raw is None or pct is None or depth is None or notional is None:
                            raise ValueError("missing required columns")

                        ts_event, e = safe_parse_ts(ts_raw, strict)
                        errors += e
                        if ts_event is None:
                            continue

                        key = (ts_event, str(pct))
                        if last_key is not None and key == last_key:
                            dedup += 1
                            continue
                        last_key = key

                        data = BinanceBookDepthPct(
                            instrument_id=instrument_id,
                            percentage_str=str(pct),
                            depth_str=str(depth),
                            notional_str=str(notional),
                            ts_event=ts_event,
                            ts_init=ts_event,
                        )
                        events.append(data)
                        ts_min = ts_event if ts_min is None else min(ts_min, ts_event)
                        ts_max = ts_event if ts_max is None else max(ts_max, ts_event)

                        if len(events) >= batch_size:
                            batch = finalize_batch(
                                events,
                                sort_key=lambda x: (x.ts_event, x.percentage_str),
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
                        if len(row) < 4:
                            raise ValueError("row too short")
                        ts_raw, pct, depth, notional = row[0], row[1], row[2], row[3]
                        ts_event, e = safe_parse_ts(ts_raw, strict)
                        errors += e
                        if ts_event is None:
                            continue

                        key = (ts_event, str(pct))
                        if last_key is not None and key == last_key:
                            dedup += 1
                            continue
                        last_key = key

                        data = BinanceBookDepthPct(
                            instrument_id=instrument_id,
                            percentage_str=str(pct).strip(),
                            depth_str=str(depth).strip(),
                            notional_str=str(notional).strip(),
                            ts_event=ts_event,
                            ts_init=ts_event,
                        )
                        events.append(data)
                        ts_min = ts_event if ts_min is None else min(ts_min, ts_event)
                        ts_max = ts_event if ts_max is None else max(ts_max, ts_event)

                        if len(events) >= batch_size:
                            batch = finalize_batch(
                                events,
                                sort_key=lambda x: (x.ts_event, x.percentage_str),
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
                sort_key=lambda x: (x.ts_event, x.percentage_str),
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


class FuturesFundingRateParser(BaseParser):
    dataset = "fundingRate"

    def parse_file(
        self,
        path: Path,
        instrument_id: Any,
        *,
        strict: bool,
        sort_within_batch: bool,
        batch_size: int,
    ) -> Iterable[ParseResult]:
        rows_read = 0
        errors = 0
        dedup = 0
        ts_min = None
        ts_max = None
        last_ts: int | None = None
        events: list[Any] = []

        if detect_header(path):
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows_read += 1
                    try:
                        r = {str(k).strip().lower(): ("" if v is None else str(v).strip()) for k, v in row.items()}
                        calc_time = r.get("calc_time") or r.get("calctime") or r.get("time")
                        interval = r.get("funding_interval_hours") or r.get("fundingintervalhours") or r.get("interval")
                        rate = r.get("last_funding_rate") or r.get("lastfundingrate") or r.get("rate")

                        if calc_time is None or interval is None or rate is None:
                            raise ValueError("missing required columns")

                        ts_event, e = safe_parse_ts(calc_time, strict)
                        errors += e
                        if ts_event is None:
                            continue

                        if last_ts is not None and ts_event == last_ts:
                            dedup += 1
                            continue
                        last_ts = ts_event

                        data = BinanceFundingRate(
                            instrument_id=instrument_id,
                            rate_str=str(rate),
                            interval_hours=int(str(interval)),
                            ts_event=ts_event,
                            ts_init=ts_event,
                        )
                        events.append(data)
                        ts_min = ts_event if ts_min is None else min(ts_min, ts_event)
                        ts_max = ts_event if ts_max is None else max(ts_max, ts_event)

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
                        if len(row) < 3:
                            raise ValueError("row too short")
                        calc_time, interval, rate = row[0], row[1], row[2]
                        ts_event, e = safe_parse_ts(calc_time, strict)
                        errors += e
                        if ts_event is None:
                            continue

                        if last_ts is not None and ts_event == last_ts:
                            dedup += 1
                            continue
                        last_ts = ts_event

                        data = BinanceFundingRate(
                            instrument_id=instrument_id,
                            rate_str=str(rate).strip(),
                            interval_hours=int(str(interval).strip()),
                            ts_event=ts_event,
                            ts_init=ts_event,
                        )
                        events.append(data)
                        ts_min = ts_event if ts_min is None else min(ts_min, ts_event)
                        ts_max = ts_event if ts_max is None else max(ts_max, ts_event)

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


@dataclass(frozen=True)
class ParserRegistry:
    universe: str

    def get(self, dataset: str) -> Any:
        if self.universe == "futures":
            if dataset == "aggTrades":
                return FuturesAggTradesParser()
            if dataset == "bookTicker":
                return FuturesBookTickerParser()
            if dataset in ("klines_1m", "klines"):
                return FuturesKlines1mParser()
            if dataset == "metrics":
                return FuturesMetricsParser()
            if dataset == "bookDepth":
                return FuturesBookDepthPctParser()
            if dataset == "fundingRate":
                return FuturesFundingRateParser()
            raise KeyError(f"no parser for dataset={dataset!r} universe={self.universe!r}")
        raise KeyError(f"unsupported universe: {self.universe!r}")


@dataclass(frozen=True)
class EtlOutputs:
    catalog_path: Path
    manifest_path: Path
    validation_report_path: Path


def instrument_id_for(symbol: str, venue: str) -> object:
    N = load_nautilus_imports()
    return N.InstrumentId.from_str(f"{symbol}.{venue}")


def run_etl(cfg: EtlConfig) -> EtlOutputs:
    N = load_nautilus_imports()
    roots = default_roots_from_args(cfg.raw_roots)
    inv = build_inventory(roots)
    datasets = set(cfg.datasets)
    symbols = set(cfg.symbols) if cfg.symbols else None

    rows = filter_inventory(
        inv,
        universe=cfg.universe,
        datasets=datasets,
        symbols=symbols,
        start_date_key=cfg.start_date_key,
        end_date_key=cfg.end_date_key,
    )

    catalog_path = Path(cfg.catalog_path).expanduser().resolve()
    catalog_path.mkdir(parents=True, exist_ok=True)
    catalog = N.ParquetDataCatalog(str(catalog_path))

    manifest = Manifest(
        raw_roots=[str(p) for p in roots],
        catalog_path=str(catalog_path),
        venue=cfg.venue,
        universe=cfg.universe,
        symbols=sorted({r.symbol for r in rows}),
        datasets=sorted(datasets),
        start_date_key=cfg.start_date_key,
        end_date_key=cfg.end_date_key,
    )

    registry = ParserRegistry(universe=cfg.universe)
    rows.sort(key=lambda r: (r.period, r.dataset, r.symbol, r.date_key, r.path))

    for r in rows:
        parser = registry.get(r.dataset)
        inst_id = instrument_id_for(r.symbol, cfg.venue)

        fstats = build_file_stats_base(
            path=r.path,
            universe=r.universe,
            period=r.period,
            dataset=r.dataset,
            symbol=r.symbol,
            date_key=r.date_key,
            size_bytes=r.size_bytes,
            mtime_ns=r.mtime_ns,
        )

        for result in parser.parse_file(
            Path(r.path),
            inst_id,
            strict=cfg.strict,
            sort_within_batch=cfg.sort_within_batch,
            batch_size=cfg.batch_size,
        ):
            if result.events:
                catalog.write_data(result.events)

            fstats.rows_read = max(fstats.rows_read, result.rows_read)
            fstats.events_written += result.events_emitted
            fstats.parse_errors = result.parse_errors
            fstats.dedup_dropped = result.dedup_dropped
            if result.ts_min is not None:
                fstats.ts_min = result.ts_min if fstats.ts_min is None else min(fstats.ts_min, result.ts_min)
            if result.ts_max is not None:
                fstats.ts_max = result.ts_max if fstats.ts_max is None else max(fstats.ts_max, result.ts_max)

        manifest.by_file.append(fstats)

    manifest_path = Path(cfg.manifest_path).expanduser().resolve()
    manifest.write(manifest_path)

    validation_path = Path(cfg.validation_report_path).expanduser().resolve()
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(_build_validation_report_text(manifest, catalog_path), encoding="utf-8")

    return EtlOutputs(
        catalog_path=catalog_path,
        manifest_path=manifest_path,
        validation_report_path=validation_path,
    )


def _build_validation_report_text(manifest: Manifest, catalog_path: Path) -> str:
    N = load_nautilus_imports()
    lines: list[str] = []
    lines.append("Validation Report")
    lines.append("")
    lines.append(f"catalog_path: {catalog_path}")
    lines.append(f"ingestion_version: {manifest.ingestion_version}")
    lines.append(f"created_utc: {manifest.created_utc}")
    lines.append(f"universe: {manifest.universe}")
    lines.append(f"venue: {manifest.venue}")
    lines.append("")

    try:
        catalog = N.ParquetDataCatalog(str(catalog_path))
        data_types = catalog.list_data_types()
        lines.append("Catalog data types")
        lines.append(str(data_types))
        lines.append("")
    except Exception as e:
        lines.append("Catalog data types")
        lines.append(f"FAILED: {e!r}")
        lines.append("")

    manifest.finalize()
    lines.append("Totals by dataset (manifest)")
    for ds_name in sorted(manifest.by_dataset.keys()):
        ds = manifest.by_dataset[ds_name]
        lines.append(
            f"{ds.dataset} events_written={ds.events_written} parse_errors={ds.parse_errors} dedup_dropped={ds.dedup_dropped} ts_min={ds.ts_min} ts_max={ds.ts_max}",
        )
    lines.append("")
    lines.append("Per-file summary (first 50 files)")
    for f in manifest.by_file[:50]:
        lines.append(
            f"{f.dataset} {f.symbol} {f.date_key} rows_read={f.rows_read} events_written={f.events_written} parse_errors={f.parse_errors} dedup_dropped={f.dedup_dropped} ts_min={f.ts_min} ts_max={f.ts_max} path={f.path}",
        )
    lines.append("")
    return "\n".join(lines)


def _call0(obj: Any, name: str) -> Any:
    fn = getattr(obj, name, None)
    if fn is None:
        raise AttributeError(name)
    return fn()


def price_to_float(x: Any) -> float:
    if x is None:
        raise TypeError("price_to_float(None)")
    try:
        return float(_call0(x, "as_double"))
    except Exception:
        pass
    try:
        return float(_call0(x, "as_decimal"))
    except Exception:
        pass
    return float(x)


def qty_to_float(x: Any) -> float:
    if x is None:
        raise TypeError("qty_to_float(None)")
    try:
        return float(_call0(x, "as_double"))
    except Exception:
        pass
    try:
        return float(_call0(x, "as_decimal"))
    except Exception:
        pass
    return float(x)


def price_to_decimal(x: Any) -> Decimal:
    if x is None:
        raise TypeError("price_to_decimal(None)")
    try:
        v = _call0(x, "as_decimal")
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v))
    except Exception:
        return Decimal(str(price_to_float(x)))


def qty_to_decimal(x: Any) -> Decimal:
    if x is None:
        raise TypeError("qty_to_decimal(None)")
    try:
        v = _call0(x, "as_decimal")
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v))
    except Exception:
        return Decimal(str(qty_to_float(x)))


def safe_div(n: float, d: float, *, default: Any = 0.0) -> Any:
    if d == 0.0:
        return default
    return n / d


@dataclass
class RollingMeanStd:
    window: int
    _xs: deque[float] = field(init=False, default_factory=deque)
    _sum: float = 0.0
    _sumsq: float = 0.0

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError("window must be positive")

    def __len__(self) -> int:
        return len(self._xs)

    def push(self, x: float) -> None:
        x = float(x)
        self._xs.append(x)
        self._sum += x
        self._sumsq += x * x
        if len(self._xs) > self.window:
            old = self._xs.popleft()
            self._sum -= old
            self._sumsq -= old * old

    @property
    def mean(self) -> float:
        n = len(self._xs)
        if n == 0:
            return 0.0
        return self._sum / n

    @property
    def var(self) -> float:
        n = len(self._xs)
        if n == 0:
            return 0.0
        mu = self._sum / n
        v = (self._sumsq / n) - (mu * mu)
        return 0.0 if v < 0.0 else v

    @property
    def std(self) -> float:
        return math.sqrt(self.var)


@dataclass
class RollingMedianMad:
    window: int
    scale_to_sigma: bool = True
    _xs: deque[float] = field(init=False, default_factory=deque)

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError("window must be positive")

    def __len__(self) -> int:
        return len(self._xs)

    def push(self, x: float) -> None:
        self._xs.append(float(x))
        if len(self._xs) > self.window:
            self._xs.popleft()

    @property
    def median(self) -> float:
        if not self._xs:
            return 0.0
        return float(statistics.median(self._xs))

    @property
    def mad(self) -> float:
        if not self._xs:
            return 0.0
        m = self.median
        devs = [abs(x - m) for x in self._xs]
        mad = float(statistics.median(devs)) if devs else 0.0
        if self.scale_to_sigma:
            mad *= 1.4826
        return mad


@dataclass
class RollingSum:
    window: int
    _xs: deque[float] = field(init=False, default_factory=deque)
    _sum: float = 0.0

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError("window must be positive")

    def __len__(self) -> int:
        return len(self._xs)

    def push(self, x: float) -> None:
        x = float(x)
        self._xs.append(x)
        self._sum += x
        if len(self._xs) > self.window:
            old = self._xs.popleft()
            self._sum -= old

    @property
    def sum(self) -> float:
        return self._sum


ZScoreKind = Literal["mean_std", "median_mad"]


@dataclass
class RollingZScore:
    window: int
    kind: ZScoreKind = "mean_std"
    min_count: int = 10
    eps: float = 1e-12
    _meanstd: RollingMeanStd | None = field(init=False, default=None)
    _medmad: RollingMedianMad | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError("window must be positive")
        if self.min_count < 0:
            raise ValueError("min_count must be non-negative")
        if self.kind == "mean_std":
            self._meanstd = RollingMeanStd(self.window)
        elif self.kind == "median_mad":
            self._medmad = RollingMedianMad(self.window)
        else:
            raise ValueError(f"unsupported kind: {self.kind!r}")

    def __len__(self) -> int:
        if self._meanstd is not None:
            return len(self._meanstd)
        assert self._medmad is not None
        return len(self._medmad)

    def _loc_scale(self) -> tuple[float, float]:
        if self._meanstd is not None:
            return self._meanstd.mean, self._meanstd.std
        assert self._medmad is not None
        return self._medmad.median, self._medmad.mad

    def zscore(self, x: float, *, update: bool = True) -> float:
        n = len(self)
        if n < self.min_count:
            z = 0.0
        else:
            loc, scale = self._loc_scale()
            if scale <= self.eps:
                z = 0.0
            else:
                z = (float(x) - loc) / scale
        if update:
            self.push(float(x))
        return z

    def push(self, x: float) -> None:
        if self._meanstd is not None:
            self._meanstd.push(float(x))
        else:
            assert self._medmad is not None
            self._medmad.push(float(x))

    @property
    def location(self) -> float:
        loc, _ = self._loc_scale()
        return loc

    @property
    def scale(self) -> float:
        _, scale = self._loc_scale()
        return scale


LpiBucketMode = Literal["time", "notional"]


@dataclass(frozen=True)
class LpiSignalConfig:
    bucket_mode: LpiBucketMode = "notional"
    bucket_interval_ms: int = 1000
    bucket_notional_threshold: float = 5_000_000.0
    eta_bps: float = 5.0
    lambda_ld: float = 1.0
    lambda_exh_liq: float = 1.0
    lambda_exh_spr: float = 1.0
    use_depth_pct: bool = False
    depth_pct_abs: float = 5.0
    depth_max_age_ms: int = 5_000
    quote_max_age_ms: int = 1_000
    use_zscore: bool = True
    z_window_buckets: int = 600
    z_kind: Literal["mean_std", "median_mad"] = "median_mad"
    z_min_count: int = 30
    vol_window_buckets: int = 300
    vol_floor: float = 1e-9
    eps: float = 1e-12

    def eta_frac(self) -> float:
        return float(self.eta_bps) / 10_000.0

    def bucket_interval_ns(self) -> int:
        return int(self.bucket_interval_ms) * 1_000_000

    def quote_max_age_ns(self) -> int:
        return int(self.quote_max_age_ms) * 1_000_000

    def depth_max_age_ns(self) -> int:
        return int(self.depth_max_age_ms) * 1_000_000


@dataclass(frozen=True)
class LpiSnapshot:
    ts_event: int
    bucket_start_ns: int
    bucket_end_ns: int
    mid: float | None
    spread: float | None
    rel_spread: float | None
    quote_age_ms: float | None
    depth_notional: float | None
    depth_age_ms: float | None
    is_notional: float
    ib_notional: float
    total_notional: float
    pressure_raw: float
    ld_raw: float
    pressure_z: float | None
    ld_z: float | None
    lpi_fast: float
    delta_lpi: float | None
    exh: float | None
    lev: float | None
    rv: float | None
    sigma: float | None


class _QuoteLKS:
    def __init__(self) -> None:
        self.bid: float | None = None
        self.ask: float | None = None
        self.mid: float | None = None
        self.spread: float | None = None
        self.ts: int | None = None

    def update(self, bid: float, ask: float, ts_event: int) -> None:
        self.bid = bid
        self.ask = ask
        self.mid = 0.5 * (bid + ask)
        self.spread = ask - bid
        self.ts = ts_event


class _DepthPctLKS:
    def __init__(self) -> None:
        self.ts: int | None = None
        self.notional_by_pct: dict[float, float] = {}

    @staticmethod
    def _parse_pct(s: str) -> float | None:
        raw = (s or "").strip()
        if not raw:
            return None
        try:
            return float(raw)
        except Exception:
            return None

    @staticmethod
    def _parse_notional(s: str) -> float | None:
        raw = (s or "").strip()
        if not raw:
            return None
        try:
            return float(raw)
        except Exception:
            return None

    def update(self, pct_str: str, notional_str: str, ts_event: int) -> None:
        pct = self._parse_pct(pct_str)
        notional = self._parse_notional(notional_str)
        if pct is None or notional is None:
            return
        if self.ts is None or ts_event > self.ts:
            self.ts = ts_event
            self.notional_by_pct = {}
        if self.ts == ts_event:
            self.notional_by_pct[pct] = notional

    def depth_total_at_abs_pct(self, pct_abs: float) -> float | None:
        if self.ts is None:
            return None
        p = float(pct_abs)
        bid = self.notional_by_pct.get(-p)
        ask = self.notional_by_pct.get(+p)
        if bid is None and ask is None:
            return None
        return float((bid or 0.0) + (ask or 0.0))


class _SlowLeveragePrior:
    def __init__(self, *, z_window: int, z_kind: str, z_min_count: int) -> None:
        self._oi_z = RollingZScore(window=z_window, kind=z_kind, min_count=z_min_count)
        self._funding_z = RollingZScore(window=z_window, kind=z_kind, min_count=z_min_count)
        self._last_oi: float | None = None
        self._last_funding: float | None = None
        self._last_lev: float | None = None

    @staticmethod
    def _parse_float(s: str) -> float | None:
        raw = (s or "").strip()
        if not raw:
            return None
        try:
            return float(raw)
        except Exception:
            return None

    def on_oi_metrics(self, oi_value_str: str) -> None:
        x = self._parse_float(oi_value_str)
        if x is None:
            return
        z = self._oi_z.zscore(x, update=True)
        self._last_oi = x
        self._last_lev = z if self._last_funding is None else None
        if self._last_funding is not None:
            fz = self._funding_z.zscore(self._last_funding, update=False)
            self._last_lev = z + fz

    def on_funding_rate(self, funding_str: str) -> None:
        x = self._parse_float(funding_str)
        if x is None:
            return
        z = self._funding_z.zscore(x, update=True)
        self._last_funding = x
        if self._last_oi is None:
            self._last_lev = z
        else:
            oiz = self._oi_z.zscore(self._last_oi, update=False)
            self._last_lev = oiz + z

    def last(self) -> float | None:
        return self._last_lev


class LpiSignalEngine:
    def __init__(self, cfg: LpiSignalConfig) -> None:
        self.cfg = cfg
        self._q = _QuoteLKS()
        self._depth = _DepthPctLKS() if cfg.use_depth_pct else None
        self._lev = _SlowLeveragePrior(
            z_window=max(100, cfg.z_window_buckets),
            z_kind=cfg.z_kind,
            z_min_count=max(10, cfg.z_min_count),
        )
        self._z_pressure = RollingZScore(
            window=cfg.z_window_buckets,
            kind=cfg.z_kind,
            min_count=cfg.z_min_count,
            eps=cfg.eps,
        )
        self._z_ld = RollingZScore(
            window=cfg.z_window_buckets,
            kind=cfg.z_kind,
            min_count=cfg.z_min_count,
            eps=cfg.eps,
        )
        self._z_neg_delta_lpi = RollingZScore(
            window=cfg.z_window_buckets,
            kind=cfg.z_kind,
            min_count=cfg.z_min_count,
            eps=cfg.eps,
        )
        self._z_delta_depth = RollingZScore(
            window=cfg.z_window_buckets,
            kind=cfg.z_kind,
            min_count=cfg.z_min_count,
            eps=cfg.eps,
        )
        self._z_neg_delta_spread = RollingZScore(
            window=cfg.z_window_buckets,
            kind=cfg.z_kind,
            min_count=cfg.z_min_count,
            eps=cfg.eps,
        )
        self._rv = RollingSum(window=cfg.vol_window_buckets)
        self._last_mid_for_rv: float | None = None
        self._bucket_start_ns: int | None = None
        self._bucket_end_ns: int | None = None
        self._is_notional: float = 0.0
        self._ib_notional: float = 0.0
        self._total_notional: float = 0.0
        self._prev_lpi: float | None = None
        self._prev_spread: float | None = None
        self._prev_depth_notional: float | None = None

    def on_quote_tick(self, tick: object) -> list[LpiSnapshot]:
        ts = int(getattr(tick, "ts_event"))
        out = self._advance_time(ts)
        bid = price_to_float(getattr(tick, "bid_price"))
        ask = price_to_float(getattr(tick, "ask_price"))
        self._q.update(bid, ask, ts)
        return out

    def on_trade_tick(self, tick: object) -> list[LpiSnapshot]:
        ts = int(getattr(tick, "ts_event"))
        out = self._advance_time(ts)
        if self._q.bid is None or self._q.ask is None or self._q.ts is None:
            return out
        if self._q.ts > ts:
            return out

        N = load_nautilus_imports()

        price = price_to_float(getattr(tick, "price"))
        size = qty_to_float(getattr(tick, "size"))
        notional = price * size
        self._total_notional += abs(notional)

        side = getattr(tick, "aggressor_side", None)
        is_sell = False
        try:
            is_sell = side == N.AggressorSide.SELLER
        except Exception:
            s = str(side).upper()
            is_sell = "SELL" in s

        eta = self.cfg.eta_frac()
        if is_sell:
            thresh = float(self._q.bid) * (1.0 - eta)
            if price < thresh:
                self._is_notional += notional
        else:
            thresh = float(self._q.ask) * (1.0 + eta)
            if price > thresh:
                self._ib_notional += notional

        if self.cfg.bucket_mode == "notional":
            if self._bucket_start_ns is None:
                self._init_bucket_for_ts(ts)
            if self._total_notional >= self.cfg.bucket_notional_threshold:
                snap = self._finalize_bucket(ts_event=ts, bucket_end_ns=ts)
                self._reset_bucket(next_bucket_start_ns=ts)
                if snap is not None:
                    out.append(snap)

        return out

    def on_custom_data(self, data: object) -> list[LpiSnapshot]:
        ts = int(getattr(data, "ts_event"))
        out = self._advance_time(ts)
        if isinstance(data, BinanceBookDepthPct):
            if self._depth is not None:
                self._depth.update(data.percentage_str, data.notional_str, ts)
            return out
        if isinstance(data, BinanceOiMetrics):
            self._lev.on_oi_metrics(data.sum_open_interest_value_str)
            return out
        if isinstance(data, BinanceFundingRate):
            self._lev.on_funding_rate(data.rate_str)
            return out
        return out

    def _init_bucket_for_ts(self, ts: int) -> None:
        if self.cfg.bucket_mode == "time":
            interval = self.cfg.bucket_interval_ns()
            start = (ts // interval) * interval
            end = start + interval
            self._bucket_start_ns = start
            self._bucket_end_ns = end
        else:
            self._bucket_start_ns = ts
            self._bucket_end_ns = None

    def _reset_bucket(self, *, next_bucket_start_ns: int) -> None:
        self._is_notional = 0.0
        self._ib_notional = 0.0
        self._total_notional = 0.0
        if self.cfg.bucket_mode == "time":
            assert self._bucket_end_ns is not None
            self._bucket_start_ns = self._bucket_end_ns
            self._bucket_end_ns = self._bucket_start_ns + self.cfg.bucket_interval_ns()
        else:
            self._bucket_start_ns = next_bucket_start_ns
            self._bucket_end_ns = None

    def _advance_time(self, ts: int) -> list[LpiSnapshot]:
        if self._bucket_start_ns is None:
            self._init_bucket_for_ts(ts)
        if self.cfg.bucket_mode != "time":
            return []
        assert self._bucket_end_ns is not None
        out: list[LpiSnapshot] = []
        while ts >= self._bucket_end_ns:
            snap = self._finalize_bucket(ts_event=self._bucket_end_ns, bucket_end_ns=self._bucket_end_ns)
            self._reset_bucket(next_bucket_start_ns=self._bucket_end_ns)
            if snap is not None:
                out.append(snap)
        return out

    def _finalize_bucket(self, *, ts_event: int, bucket_end_ns: int) -> LpiSnapshot | None:
        if self._bucket_start_ns is None:
            return None
        bucket_start = int(self._bucket_start_ns)
        bucket_end = int(bucket_end_ns)

        mid = self._q.mid
        spread = self._q.spread
        rel_spread = None
        quote_age_ms = None
        if self._q.ts is not None and mid is not None and spread is not None:
            quote_age_ns = bucket_end - self._q.ts
            quote_age_ms = quote_age_ns / 1_000_000.0
            rel_spread = safe_div(spread, mid, default=None)

        depth_notional = None
        depth_age_ms = None
        if self._depth is not None and self._depth.ts is not None:
            age_ns = bucket_end - self._depth.ts
            depth_age_ms = age_ns / 1_000_000.0
            if age_ns <= self.cfg.depth_max_age_ns():
                depth_notional = self._depth.depth_total_at_abs_pct(self.cfg.depth_pct_abs)

        pressure_raw = self._is_notional - self._ib_notional

        ld_raw = 0.0
        if spread is not None and mid is not None and mid > 0.0:
            if self.cfg.use_depth_pct and depth_notional is not None and depth_notional > 0.0:
                ld_raw = spread / (depth_notional + self.cfg.eps)
            else:
                ld_raw = spread / mid

        if self.cfg.use_zscore:
            pressure_z = self._z_pressure.zscore(pressure_raw, update=True)
            ld_z = self._z_ld.zscore(ld_raw, update=True)
            lpi_fast = pressure_z + (self.cfg.lambda_ld * ld_z)
        else:
            pressure_z = None
            ld_z = None
            lpi_fast = pressure_raw + (self.cfg.lambda_ld * ld_raw)

        delta_lpi = None
        exh = None
        if self._prev_lpi is not None:
            delta_lpi = lpi_fast - self._prev_lpi
            if self.cfg.use_zscore:
                z_neg_dlpi = self._z_neg_delta_lpi.zscore(-delta_lpi, update=True)
                if self.cfg.use_depth_pct and depth_notional is not None and self._prev_depth_notional is not None:
                    ddepth = depth_notional - self._prev_depth_notional
                    z_ddepth = self._z_delta_depth.zscore(ddepth, update=True)
                    exh = z_neg_dlpi + (self.cfg.lambda_exh_liq * z_ddepth)
                else:
                    if spread is not None and self._prev_spread is not None:
                        dspr = spread - self._prev_spread
                        z_neg_dspr = self._z_neg_delta_spread.zscore(-dspr, update=True)
                        exh = z_neg_dlpi + (self.cfg.lambda_exh_spr * z_neg_dspr)
            else:
                exh = -delta_lpi
                if self.cfg.use_depth_pct and depth_notional is not None and self._prev_depth_notional is not None:
                    exh += self.cfg.lambda_exh_liq * (depth_notional - self._prev_depth_notional)
                elif spread is not None and self._prev_spread is not None:
                    exh += self.cfg.lambda_exh_spr * (-(spread - self._prev_spread))

        self._prev_lpi = lpi_fast
        self._prev_spread = spread
        self._prev_depth_notional = depth_notional

        rv = None
        sigma = None
        if mid is not None and mid > 0.0:
            if self._last_mid_for_rv is not None and self._last_mid_for_rv > 0.0:
                r = math.log(mid / self._last_mid_for_rv)
                self._rv.push(r * r)
            self._last_mid_for_rv = mid
            rv = self._rv.sum
            sigma = math.sqrt(max(rv, 0.0))

        lev = self._lev.last()

        return LpiSnapshot(
            ts_event=int(ts_event),
            bucket_start_ns=bucket_start,
            bucket_end_ns=bucket_end,
            mid=mid,
            spread=spread,
            rel_spread=rel_spread,
            quote_age_ms=quote_age_ms,
            depth_notional=depth_notional,
            depth_age_ms=depth_age_ms,
            is_notional=float(self._is_notional),
            ib_notional=float(self._ib_notional),
            total_notional=float(self._total_notional),
            pressure_raw=float(pressure_raw),
            ld_raw=float(ld_raw),
            pressure_z=pressure_z,
            ld_z=ld_z,
            lpi_fast=float(lpi_fast),
            delta_lpi=(float(delta_lpi) if delta_lpi is not None else None),
            exh=(float(exh) if exh is not None else None),
            lev=lev,
            rv=rv,
            sigma=sigma,
        )


from nautilus_trader.config import PositiveInt, StrategyConfig
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.indicators import ExponentialMovingAverage
from nautilus_trader.model.data import Bar, BarType, QuoteTick, TradeTick
from nautilus_trader.model.data import DataType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.orders import LimitOrder, MarketOrder
from nautilus_trader.trading.strategy import Strategy


class EmaCrossBaselineConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    fast_ema_period: PositiveInt = 10
    slow_ema_period: PositiveInt = 20
    request_historical_bars: bool = False
    close_positions_on_stop: bool = True
    order_time_in_force: TimeInForce = TimeInForce.GTC


class EmaCrossBaseline(Strategy):
    def __init__(self, config: EmaCrossBaselineConfig) -> None:
        PyCondition.is_true(
            config.fast_ema_period < config.slow_ema_period,
            "{config.fast_ema_period=} must be less than {config.slow_ema_period=}",
        )
        super().__init__(config=config)
        self.instrument: Instrument | None = None
        self.fast_ema = ExponentialMovingAverage(config.fast_ema_period)
        self.slow_ema = ExponentialMovingAverage(config.slow_ema_period)

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument for {self.config.instrument_id}")
            self.stop()
            return
        self.register_indicator_for_bars(self.config.bar_type, self.fast_ema)
        self.register_indicator_for_bars(self.config.bar_type, self.slow_ema)
        if self.config.request_historical_bars:
            import pandas as _pd

            self.request_bars(self.config.bar_type, start=self.clock.utc_now() - _pd.Timedelta(days=1))
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        try:
            if not self.indicators_initialized():
                return
        except Exception:
            if getattr(self.fast_ema, "count", 0) < int(self.config.slow_ema_period):
                return
        if self.fast_ema.value >= self.slow_ema.value:
            if self.portfolio.is_flat(self.config.instrument_id):
                self._buy()
            elif self.portfolio.is_net_short(self.config.instrument_id):
                self.close_all_positions(self.config.instrument_id)
                self._buy()
        else:
            if self.portfolio.is_flat(self.config.instrument_id):
                self._sell()
            elif self.portfolio.is_net_long(self.config.instrument_id):
                self.close_all_positions(self.config.instrument_id)
                self._sell()

    def _buy(self) -> None:
        assert self.instrument is not None
        order: MarketOrder = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self.instrument.make_qty(self.config.trade_size),
            time_in_force=self.config.order_time_in_force,
        )
        self.submit_order(order)

    def _sell(self) -> None:
        assert self.instrument is not None
        order: MarketOrder = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.SELL,
            quantity=self.instrument.make_qty(self.config.trade_size),
            time_in_force=self.config.order_time_in_force,
        )
        self.submit_order(order)

    def on_stop(self) -> None:
        self.cancel_all_orders(self.config.instrument_id)
        if self.config.close_positions_on_stop:
            self.close_all_positions(self.config.instrument_id)
        self.unsubscribe_bars(self.config.bar_type)

    def on_reset(self) -> None:
        self.fast_ema.reset()
        self.slow_ema.reset()


@dataclass(frozen=True)
class _PendingEntry:
    client_order_id: Any
    ts_submit: int
    side: OrderSide
    variant: str


class PhoenixLpiStrategyConfig(StrategyConfig, frozen=True):
    instrument_id: Any
    trade_size: Decimal
    bucket_mode: str = "notional"
    bucket_interval_ms: int = 1000
    bucket_notional_threshold: float = 5_000_000.0
    eta_bps: float = 5.0
    use_depth_pct: bool = False
    depth_pct_abs: float = 5.0
    use_zscore: bool = True
    z_window_buckets: PositiveInt = 600
    z_kind: str = "median_mad"
    z_min_count: PositiveInt = 30
    lambda_ld: float = 1.0
    lambda_exh_liq: float = 1.0
    lambda_exh_spr: float = 1.0
    theta_lpi: float = 2.0
    theta_exh_low: float = -0.5
    theta_exh_high: float = 0.5
    lpi_exit_abs: float = 0.5
    max_rel_spread_bps: float = 20.0
    min_depth_notional: float = 0.0
    quote_max_age_ms: int = 1000
    depth_max_age_ms: int = 5000
    momentum_use_market: bool = True
    mean_rev_post_only: bool = True
    mean_rev_improve_ticks: int = 0
    entry_ttl_buckets: PositiveInt = 5
    max_hold_buckets_momentum: PositiveInt = 10
    max_hold_buckets_mean_rev: PositiveInt = 60
    close_positions_on_stop: bool = True
    reduce_only_on_stop: bool = True


class PhoenixLpiStrategy(Strategy):
    def __init__(self, config: PhoenixLpiStrategyConfig) -> None:
        super().__init__(config=config)
        self.instrument: Instrument | None = None
        self.signal_engine = LpiSignalEngine(
            LpiSignalConfig(
                bucket_mode=("time" if config.bucket_mode == "time" else "notional"),
                bucket_interval_ms=int(config.bucket_interval_ms),
                bucket_notional_threshold=float(config.bucket_notional_threshold),
                eta_bps=float(config.eta_bps),
                lambda_ld=float(config.lambda_ld),
                lambda_exh_liq=float(config.lambda_exh_liq),
                lambda_exh_spr=float(config.lambda_exh_spr),
                use_depth_pct=bool(config.use_depth_pct),
                depth_pct_abs=float(config.depth_pct_abs),
                depth_max_age_ms=int(config.depth_max_age_ms),
                quote_max_age_ms=int(config.quote_max_age_ms),
                use_zscore=bool(config.use_zscore),
                z_window_buckets=int(config.z_window_buckets),
                z_kind=str(config.z_kind),
                z_min_count=int(config.z_min_count),
            ),
        )
        self._last_snapshot: LpiSnapshot | None = None
        self._pending_entry: _PendingEntry | None = None
        self._position_variant: str | None = None
        self._position_entry_bucket_end: int | None = None

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument for {self.config.instrument_id}")
            self.stop()
            return
        self.subscribe_quote_ticks(self.config.instrument_id)
        self.subscribe_trade_ticks(self.config.instrument_id)
        self.subscribe_data(DataType(BinanceOiMetrics), instrument_id=self.config.instrument_id)
        self.subscribe_data(DataType(BinanceFundingRate), instrument_id=self.config.instrument_id)
        self.subscribe_data(DataType(BinanceBookDepthPct), instrument_id=self.config.instrument_id)

    def on_quote_tick(self, tick: QuoteTick) -> None:
        snaps = self.signal_engine.on_quote_tick(tick)
        for s in snaps:
            self._on_bucket(s)

    def on_trade_tick(self, tick: TradeTick) -> None:
        snaps = self.signal_engine.on_trade_tick(tick)
        for s in snaps:
            self._on_bucket(s)

    def on_data(self, data) -> None:
        snaps = self.signal_engine.on_custom_data(data)
        for s in snaps:
            self._on_bucket(s)

    def _on_bucket(self, snap: LpiSnapshot) -> None:
        self._last_snapshot = snap
        self._maybe_cancel_pending_entry(snap)
        self._maybe_exit_position(snap)
        if self._pending_entry is not None:
            return
        if self.portfolio.is_flat(self.config.instrument_id):
            self._maybe_enter(snap)

    def _gates_ok(self, snap: LpiSnapshot) -> bool:
        if snap.mid is None or snap.spread is None or snap.rel_spread is None or snap.quote_age_ms is None:
            return False
        if snap.quote_age_ms > float(self.config.quote_max_age_ms):
            return False
        rel_bps = (snap.rel_spread * 10_000.0) if snap.rel_spread is not None else None
        if rel_bps is None:
            return False
        if rel_bps > float(self.config.max_rel_spread_bps):
            return False
        if self.config.use_depth_pct:
            if snap.depth_notional is None or snap.depth_age_ms is None:
                return False
            if snap.depth_age_ms > float(self.config.depth_max_age_ms):
                return False
            if snap.depth_notional < float(self.config.min_depth_notional):
                return False
        return True

    def _maybe_enter(self, snap: LpiSnapshot) -> None:
        if not self._gates_ok(snap):
            return
        lpi = float(snap.lpi_fast)
        exh = float(snap.exh) if snap.exh is not None else 0.0
        if abs(lpi) < float(self.config.theta_lpi):
            return

        variant: str | None = None
        side: OrderSide | None = None

        if exh <= float(self.config.theta_exh_low):
            variant = "momentum"
            if lpi > 0.0:
                side = OrderSide.SELL
            elif lpi < 0.0:
                side = OrderSide.BUY
        elif exh >= float(self.config.theta_exh_high):
            variant = "mean_reversion"
            if lpi > 0.0:
                side = OrderSide.BUY
            elif lpi < 0.0:
                side = OrderSide.SELL

        if variant is None or side is None:
            return
        if variant == "momentum":
            self._enter_momentum(side=side, snap=snap)
        else:
            self._enter_mean_reversion(side=side, snap=snap)

    def _enter_momentum(self, *, side: OrderSide, snap: LpiSnapshot) -> None:
        assert self.instrument is not None
        qty = self.instrument.make_qty(self.config.trade_size)
        if self.config.momentum_use_market:
            order: MarketOrder = self.order_factory.market(
                instrument_id=self.config.instrument_id,
                order_side=side,
                quantity=qty,
                time_in_force=TimeInForce.IOC,
            )
            self.submit_order(order)
            self._pending_entry = _PendingEntry(
                client_order_id=order.client_order_id,
                ts_submit=int(snap.bucket_end_ns),
                side=side,
                variant="momentum",
            )
        else:
            last_q = self.cache.quote_tick(self.config.instrument_id)
            if last_q is None:
                return
            bid = price_to_decimal(last_q.bid_price)
            ask = price_to_decimal(last_q.ask_price)
            tick = self.instrument.price_increment
            if side == OrderSide.BUY:
                px = self.instrument.make_price(ask + tick)
            else:
                px = self.instrument.make_price(bid - tick)
            order = self.order_factory.limit(
                instrument_id=self.config.instrument_id,
                order_side=side,
                quantity=qty,
                price=px,
                post_only=False,
                time_in_force=TimeInForce.IOC,
            )
            self.submit_order(order)
            self._pending_entry = _PendingEntry(
                client_order_id=order.client_order_id,
                ts_submit=int(snap.bucket_end_ns),
                side=side,
                variant="momentum",
            )

    def _enter_mean_reversion(self, *, side: OrderSide, snap: LpiSnapshot) -> None:
        assert self.instrument is not None
        last_q = self.cache.quote_tick(self.config.instrument_id)
        if last_q is None:
            return
        bid = price_to_decimal(last_q.bid_price)
        ask = price_to_decimal(last_q.ask_price)
        tick = self.instrument.price_increment
        improve = max(int(self.config.mean_rev_improve_ticks), 0)
        if side == OrderSide.BUY:
            px = bid
            if improve > 0:
                px_try = bid + (tick * improve)
                if px_try < ask:
                    px = px_try
            price = self.instrument.make_price(px)
        else:
            px = ask
            if improve > 0:
                px_try = ask - (tick * improve)
                if px_try > bid:
                    px = px_try
            price = self.instrument.make_price(px)
        qty = self.instrument.make_qty(self.config.trade_size)
        order: LimitOrder = self.order_factory.limit(
            instrument_id=self.config.instrument_id,
            order_side=side,
            quantity=qty,
            price=price,
            post_only=bool(self.config.mean_rev_post_only),
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)
        self._pending_entry = _PendingEntry(
            client_order_id=order.client_order_id,
            ts_submit=int(snap.bucket_end_ns),
            side=side,
            variant="mean_reversion",
        )

    def _maybe_cancel_pending_entry(self, snap: LpiSnapshot) -> None:
        if self._pending_entry is None:
            return
        ttl = int(self.config.entry_ttl_buckets)
        interval_ns = self.signal_engine.cfg.bucket_interval_ns()
        if int(snap.bucket_end_ns) - int(self._pending_entry.ts_submit) >= ttl * int(interval_ns):
            order = self.cache.order(self._pending_entry.client_order_id)
            if order is not None and getattr(order, "is_open", False):
                self.cancel_order(order)
            self._pending_entry = None
            return
        if not self._gates_ok(snap):
            order = self.cache.order(self._pending_entry.client_order_id)
            if order is not None and getattr(order, "is_open", False):
                self.cancel_order(order)
            self._pending_entry = None
            return

    def _maybe_exit_position(self, snap: LpiSnapshot) -> None:
        if self.portfolio.is_flat(self.config.instrument_id):
            self._position_variant = None
            self._position_entry_bucket_end = None
            return
        variant = self._position_variant or "unknown"
        entry_end = self._position_entry_bucket_end
        hold_buckets = None
        if entry_end is not None:
            interval_ns = self.signal_engine.cfg.bucket_interval_ns() if self.signal_engine.cfg.bucket_mode == "time" else None
            if interval_ns:
                hold_buckets = max(0, (int(snap.bucket_end_ns) - int(entry_end)) // int(interval_ns))
        lpi_abs = abs(float(snap.lpi_fast))
        if variant == "momentum":
            if hold_buckets is not None and hold_buckets >= int(self.config.max_hold_buckets_momentum):
                self.close_all_positions(self.config.instrument_id, reduce_only=True)
                self.cancel_all_orders(self.config.instrument_id)
                return
            if snap.exh is not None and float(snap.exh) >= float(self.config.theta_exh_high):
                self.close_all_positions(self.config.instrument_id, reduce_only=True)
                self.cancel_all_orders(self.config.instrument_id)
                return
        elif variant == "mean_reversion":
            if hold_buckets is not None and hold_buckets >= int(self.config.max_hold_buckets_mean_rev):
                self.close_all_positions(self.config.instrument_id, reduce_only=True)
                self.cancel_all_orders(self.config.instrument_id)
                return
            if lpi_abs <= float(self.config.lpi_exit_abs):
                self.close_all_positions(self.config.instrument_id, reduce_only=True)
                self.cancel_all_orders(self.config.instrument_id)
                return
            if snap.exh is not None and float(snap.exh) <= float(self.config.theta_exh_low):
                self.close_all_positions(self.config.instrument_id, reduce_only=True)
                self.cancel_all_orders(self.config.instrument_id)
                return
        else:
            if lpi_abs <= float(self.config.lpi_exit_abs):
                self.close_all_positions(self.config.instrument_id, reduce_only=True)
                self.cancel_all_orders(self.config.instrument_id)
                return

    def on_order_filled(self, event) -> None:
        if self._pending_entry is not None and getattr(event, "client_order_id", None) == self._pending_entry.client_order_id:
            self._position_variant = self._pending_entry.variant
            self._position_entry_bucket_end = self._last_snapshot.bucket_end_ns if self._last_snapshot else int(getattr(event, "ts_event", 0))
            self._pending_entry = None

    def on_stop(self) -> None:
        self.cancel_all_orders(self.config.instrument_id)
        if self.config.close_positions_on_stop:
            self.close_all_positions(
                instrument_id=self.config.instrument_id,
                reduce_only=bool(self.config.reduce_only_on_stop),
            )
        self.unsubscribe_quote_ticks(self.config.instrument_id)
        self.unsubscribe_trade_ticks(self.config.instrument_id)
        self.unsubscribe_data(DataType(BinanceOiMetrics), instrument_id=self.config.instrument_id)
        self.unsubscribe_data(DataType(BinanceFundingRate), instrument_id=self.config.instrument_id)
        self.unsubscribe_data(DataType(BinanceBookDepthPct), instrument_id=self.config.instrument_id)

    def on_reset(self) -> None:
        self.signal_engine = LpiSignalEngine(self.signal_engine.cfg)
        self._last_snapshot = None
        self._pending_entry = None
        self._position_variant = None
        self._position_entry_bucket_end = None


@dataclass(frozen=True)
class NautilusPhase4Imports:
    BacktestNode: Any
    BacktestRunConfig: Any
    BacktestEngineConfig: Any
    BacktestVenueConfig: Any
    BacktestDataConfig: Any
    ImportableStrategyConfig: Any
    LoggingConfig: Any
    Venue: Any
    InstrumentId: Any
    Symbol: Any
    CryptoPerpetual: Any
    Price: Any
    Quantity: Any
    QuoteTick: Any
    TradeTick: Any
    Bar: Any
    BarType: Any
    ParquetDataCatalog: Any


_NAUTILUS_PHASE4_CACHE: NautilusPhase4Imports | None = None


def load_nautilus_phase4_imports() -> NautilusPhase4Imports:
    global _NAUTILUS_PHASE4_CACHE
    if _NAUTILUS_PHASE4_CACHE is not None:
        return _NAUTILUS_PHASE4_CACHE

    from nautilus_trader.backtest.node import BacktestNode

    try:
        from nautilus_trader.backtest.config import BacktestDataConfig, BacktestEngineConfig, BacktestRunConfig, BacktestVenueConfig
    except Exception:
        from nautilus_trader.config import BacktestDataConfig, BacktestEngineConfig, BacktestRunConfig, BacktestVenueConfig

    try:
        from nautilus_trader.config import ImportableStrategyConfig
    except Exception:
        from nautilus_trader.backtest.config import ImportableStrategyConfig

    try:
        from nautilus_trader.config import LoggingConfig
    except Exception:
        from nautilus_trader.common.config import LoggingConfig

    try:
        from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
    except Exception:
        from nautilus_trader.model import InstrumentId, Symbol, Venue

    from nautilus_trader.model.objects import Price, Quantity

    from nautilus_trader.model.instruments import CryptoPerpetual

    from nautilus_trader.model.data import Bar, BarType, QuoteTick, TradeTick

    try:
        from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
    except Exception:
        from nautilus_trader.persistence.catalog import ParquetDataCatalog

    _NAUTILUS_PHASE4_CACHE = NautilusPhase4Imports(
        BacktestNode=BacktestNode,
        BacktestRunConfig=BacktestRunConfig,
        BacktestEngineConfig=BacktestEngineConfig,
        BacktestVenueConfig=BacktestVenueConfig,
        BacktestDataConfig=BacktestDataConfig,
        ImportableStrategyConfig=ImportableStrategyConfig,
        LoggingConfig=LoggingConfig,
        Venue=Venue,
        InstrumentId=InstrumentId,
        Symbol=Symbol,
        CryptoPerpetual=CryptoPerpetual,
        Price=Price,
        Quantity=Quantity,
        QuoteTick=QuoteTick,
        TradeTick=TradeTick,
        Bar=Bar,
        BarType=BarType,
        ParquetDataCatalog=ParquetDataCatalog,
    )
    return _NAUTILUS_PHASE4_CACHE


UTC = timezone.utc


class TimeParseError(ValueError):
    pass


_ISO_Z_RE = re.compile(r"Z$", re.IGNORECASE)
_DURATION_RE = re.compile(r"^\s*(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>ms|s|m|h|d)\s*$", re.IGNORECASE)


def parse_utc_datetime(value: str) -> datetime:
    s = (value or "").strip()
    if not s:
        raise TimeParseError("empty datetime string")
    s = _ISO_Z_RE.sub("+00:00", s)
    if " " in s and "T" not in s:
        s = s.replace(" ", "T", 1)
    try:
        dt = datetime.fromisoformat(s)
    except ValueError as e:
        raise TimeParseError(f"invalid datetime: {value!r}") from e
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    return dt


def format_utc_datetime(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    dt = dt.astimezone(UTC)
    return dt.isoformat(timespec="microseconds").replace("+00:00", "Z")


def parse_duration(value: str) -> timedelta:
    s = (value or "").strip()
    if not s:
        raise TimeParseError("empty duration string")
    if s == "0":
        return timedelta(0)
    m = _DURATION_RE.match(s)
    if not m:
        raise TimeParseError(f"invalid duration: {value!r}")
    v = float(m.group("value"))
    unit = m.group("unit").lower()
    if unit == "ms":
        return timedelta(milliseconds=v)
    if unit == "s":
        return timedelta(seconds=v)
    if unit == "m":
        return timedelta(minutes=v)
    if unit == "h":
        return timedelta(hours=v)
    if unit == "d":
        return timedelta(days=v)
    raise TimeParseError(f"unsupported duration unit: {unit!r}")


@dataclass(frozen=True)
class TimeRange:
    start: datetime
    end: datetime

    def validate(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("TimeRange datetimes must be timezone-aware")
        if self.start >= self.end:
            raise ValueError(f"invalid TimeRange: start={self.start} >= end={self.end}")

    def to_iso(self) -> tuple[str, str]:
        return format_utc_datetime(self.start), format_utc_datetime(self.end)


@dataclass(frozen=True)
class DeterminismConfig:
    seed: int = 0
    set_hash_seed_hint: bool = True


def set_global_determinism(cfg: DeterminismConfig) -> dict[str, Any]:
    random.seed(cfg.seed)
    try:
        import numpy as np

        np.random.seed(cfg.seed)
    except Exception:
        pass
    env: dict[str, Any] = {
        "seed": cfg.seed,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }
    if cfg.set_hash_seed_hint and env["pythonhashseed"] is None:
        env["pythonhashseed_hint"] = "Set PYTHONHASHSEED=0 in the environment for stronger determinism."
    return env


SchemaVersion = Literal["1"]
RunMode = Literal["single", "walk_forward"]
StrategyName = Literal["ema_cross_baseline", "phoenix_lpi"]


@dataclass(frozen=True)
class CatalogConfig:
    path: str
    manifest_path: str | None = None
    instrument_specs_path: str | None = None


@dataclass(frozen=True)
class OutputConfig:
    dir: str = "./runs"
    suite_name: str | None = None
    overwrite: bool = False


@dataclass(frozen=True)
class UniverseConfig:
    venue: str = "BINANCE"
    instrument_ids: list[str] = field(default_factory=lambda: ["BTCUSDT.BINANCE"])
    oms_type: str = "NETTING"
    account_type: str = "MARGIN"
    base_currency: str = "USDT"
    starting_balances: list[str] = field(default_factory=lambda: ["10000 USDT"])


@dataclass(frozen=True)
class DataConfig:
    include: list[str] = field(
        default_factory=lambda: [
            "QuoteTick",
            "TradeTick",
            "Bar",
            "BinanceOiMetrics",
            "BinanceFundingRate",
            "BinanceBookDepthPct",
        ],
    )


@dataclass(frozen=True)
class TimeConfig:
    start_utc: str = "2023-05-16T00:00:00Z"
    end_utc: str = "2023-05-17T00:00:00Z"
    warmup: str = "0s"

    def range(self) -> TimeRange:
        tr = TimeRange(
            start=parse_utc_datetime(self.start_utc),
            end=parse_utc_datetime(self.end_utc),
        )
        tr.validate()
        return tr


@dataclass(frozen=True)
class WalkForwardConfig:
    train: str = "7d"
    test: str = "1d"
    step: str = "1d"
    purge: str = "0s"
    embargo: str = "0s"
    max_folds: int | None = None


@dataclass(frozen=True)
class ModeConfig:
    kind: RunMode = "single"
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)


@dataclass(frozen=True)
class StrategySpec:
    name: StrategyName = "ema_cross_baseline"
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Phase4SuiteConfig:
    schema_version: SchemaVersion = "1"
    catalog: CatalogConfig = field(default_factory=lambda: CatalogConfig(path="./data/catalog/phoenix_um_btcusdt"))
    output: OutputConfig = field(default_factory=OutputConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    data: DataConfig = field(default_factory=DataConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    mode: ModeConfig = field(default_factory=ModeConfig)
    strategy: StrategySpec = field(default_factory=StrategySpec)
    determinism_seed: int = 0

    def resolved_suite_id(self) -> str:
        obj = to_primitive_dict(self)
        obj.get("output", {}).pop("dir", None)
        payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


def to_primitive_dict(cfg: Any) -> dict[str, Any]:
    if hasattr(cfg, "__dict__") and not isinstance(cfg, dict):
        d: dict[str, Any] = {}
        for k, v in cfg.__dict__.items():
            d[k] = to_primitive_value(v)
        return d
    if isinstance(cfg, dict):
        return {str(k): to_primitive_value(v) for k, v in cfg.items()}
    raise TypeError(f"unsupported cfg type: {type(cfg)}")


def to_primitive_value(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, list):
        return [to_primitive_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): to_primitive_value(x) for k, x in v.items()}
    if hasattr(v, "__dict__"):
        return to_primitive_dict(v)
    return str(v)


def load_suite_config(path: str) -> Phase4SuiteConfig:
    p = Path(path).expanduser().resolve()
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Phase 4 config must be a YAML mapping")
    schema_version = str(raw.get("schema_version", "1"))
    if schema_version != "1":
        raise ValueError(f"Unsupported schema_version={schema_version!r}")

    catalog_raw = raw.get("catalog", {}) or {}
    output_raw = raw.get("output", {}) or {}
    universe_raw = raw.get("universe", {}) or {}
    data_raw = raw.get("data", {}) or {}
    time_raw = raw.get("time", {}) or {}
    mode_raw = raw.get("mode", {}) or {}
    strategy_raw = raw.get("strategy", {}) or {}

    cfg = Phase4SuiteConfig(
        schema_version="1",
        catalog=CatalogConfig(
            path=str(catalog_raw.get("path")),
            manifest_path=(str(catalog_raw.get("manifest_path")) if catalog_raw.get("manifest_path") else None),
            instrument_specs_path=(str(catalog_raw.get("instrument_specs_path")) if catalog_raw.get("instrument_specs_path") else None),
        ),
        output=OutputConfig(
            dir=str(output_raw.get("dir", "./runs")),
            suite_name=(str(output_raw.get("suite_name")) if output_raw.get("suite_name") else None),
            overwrite=bool(output_raw.get("overwrite", False)),
        ),
        universe=UniverseConfig(
            venue=str(universe_raw.get("venue", "BINANCE")),
            instrument_ids=list(universe_raw.get("instrument_ids", ["BTCUSDT.BINANCE"])),
            oms_type=str(universe_raw.get("oms_type", "NETTING")),
            account_type=str(universe_raw.get("account_type", "MARGIN")),
            base_currency=str(universe_raw.get("base_currency", "USDT")),
            starting_balances=list(universe_raw.get("starting_balances", ["10000 USDT"])),
        ),
        data=DataConfig(
            include=list(
                data_raw.get(
                    "include",
                    [
                        "QuoteTick",
                        "TradeTick",
                        "Bar",
                        "BinanceOiMetrics",
                        "BinanceFundingRate",
                        "BinanceBookDepthPct",
                    ],
                ),
            ),
        ),
        time=TimeConfig(
            start_utc=str(time_raw.get("start_utc", "2023-05-16T00:00:00Z")),
            end_utc=str(time_raw.get("end_utc", "2023-05-17T00:00:00Z")),
            warmup=str(time_raw.get("warmup", "0s")),
        ),
        mode=ModeConfig(
            kind=str(mode_raw.get("kind", "single")),
            walk_forward=WalkForwardConfig(
                train=str((mode_raw.get("walk_forward") or {}).get("train", "7d")),
                test=str((mode_raw.get("walk_forward") or {}).get("test", "1d")),
                step=str((mode_raw.get("walk_forward") or {}).get("step", "1d")),
                purge=str((mode_raw.get("walk_forward") or {}).get("purge", "0s")),
                embargo=str((mode_raw.get("walk_forward") or {}).get("embargo", "0s")),
                max_folds=(mode_raw.get("walk_forward") or {}).get("max_folds"),
            ),
        ),
        strategy=StrategySpec(
            name=str(strategy_raw.get("name", "ema_cross_baseline")),
            config=dict(strategy_raw.get("config", {}) or {}),
        ),
        determinism_seed=int(raw.get("determinism_seed", 0)),
    )

    _ = cfg.time.range()
    if cfg.mode.kind not in ("single", "walk_forward"):
        raise ValueError(f"Unsupported mode.kind={cfg.mode.kind!r}")
    if cfg.strategy.name not in ("ema_cross_baseline", "phoenix_lpi"):
        raise ValueError(f"Unsupported strategy.name={cfg.strategy.name!r}")
    return cfg


@dataclass(frozen=True)
class WalkForwardFold:
    fold_index: int
    train_range: TimeRange
    test_range: TimeRange
    purge: timedelta
    embargo: timedelta

    def to_id_str(self) -> str:
        ts, te = self.test_range.start, self.test_range.end
        return f"fold_{self.fold_index:04d}_test_{ts:%Y%m%dT%H%M%SZ}_{te:%Y%m%dT%H%M%SZ}"

    def to_iso(self) -> dict[str, str]:
        tr_s, tr_e = self.train_range.to_iso()
        te_s, te_e = self.test_range.to_iso()
        return {
            "train_start_utc": tr_s,
            "train_end_utc": tr_e,
            "test_start_utc": te_s,
            "test_end_utc": te_e,
        }


def iter_walk_forward_folds(
    *,
    overall: TimeRange,
    train: timedelta,
    test: timedelta,
    step: timedelta,
    purge: timedelta,
    embargo: timedelta,
    max_folds: int | None = None,
) -> Iterator[WalkForwardFold]:
    overall.validate()
    if train <= timedelta(0) or test <= timedelta(0) or step <= timedelta(0):
        raise ValueError("train/test/step must be positive durations")
    if purge < timedelta(0) or embargo < timedelta(0):
        raise ValueError("purge/embargo must be non-negative")

    t = overall.start
    i = 0
    while True:
        train_start = t
        train_end = train_start + train
        test_start = train_end + purge
        test_end = test_start + test
        if test_end > overall.end:
            break
        fold = WalkForwardFold(
            fold_index=i,
            train_range=TimeRange(train_start, train_end),
            test_range=TimeRange(test_start, test_end),
            purge=purge,
            embargo=embargo,
        )
        yield fold
        i += 1
        if max_folds is not None and i >= int(max_folds):
            break
        next_start = train_start + step
        embargo_floor = test_end + embargo
        if next_start < embargo_floor:
            next_start = embargo_floor
        t = next_start


def fold_summary_dict(fold: WalkForwardFold) -> dict[str, object]:
    return {
        "fold_index": fold.fold_index,
        **fold.to_iso(),
        "purge_seconds": float(fold.purge.total_seconds()),
        "embargo_seconds": float(fold.embargo.total_seconds()),
        "fold_id": fold.to_id_str(),
    }


class InstrumentSpecError(ValueError):
    pass


@dataclass(frozen=True)
class InstrumentSpec:
    instrument_id: str
    raw_symbol: str
    base_currency: str
    quote_currency: str
    settlement_currency: str
    is_inverse: bool
    price_precision: int
    size_precision: int
    price_increment: str
    size_increment: str
    maker_fee: str
    taker_fee: str


def _load_currency_obj(code: str) -> Any:
    code = (code or "").strip().upper()
    if not code:
        raise InstrumentSpecError("empty currency code")
    try:
        mod = __import__("nautilus_trader.model.currencies", fromlist=[code])
        cur = getattr(mod, code, None)
        if cur is None:
            raise InstrumentSpecError(f"Currency {code!r} not found in nautilus_trader.model.currencies")
        return cur
    except Exception as e:
        raise InstrumentSpecError(f"Failed to resolve currency code {code!r}: {e}") from e


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
    out: dict[str, InstrumentSpec] = {}
    for item in items:
        if not isinstance(item, dict):
            raise InstrumentSpecError("each instrument spec must be a mapping")
        inst_id = str(item.get("instrument_id") or "").strip()
        if not inst_id:
            raise InstrumentSpecError("instrument spec missing 'instrument_id'")
        spec = InstrumentSpec(
            instrument_id=inst_id,
            raw_symbol=str(item.get("raw_symbol") or inst_id.split(".")[0]).strip(),
            base_currency=str(item.get("base_currency") or "BTC").strip(),
            quote_currency=str(item.get("quote_currency") or "USDT").strip(),
            settlement_currency=str(item.get("settlement_currency") or str(item.get("quote_currency") or "USDT")).strip(),
            is_inverse=bool(item.get("is_inverse", False)),
            price_precision=int(item.get("price_precision")),
            size_precision=int(item.get("size_precision")),
            price_increment=str(item.get("price_increment")).strip(),
            size_increment=str(item.get("size_increment")).strip(),
            maker_fee=str(item.get("maker_fee")).strip(),
            taker_fee=str(item.get("taker_fee")).strip(),
        )
        out[spec.instrument_id] = spec
    return out


def _build_crypto_perpetual_from_spec(N: NautilusPhase4Imports, spec: InstrumentSpec) -> Any:
    inst_id = N.InstrumentId.from_str(spec.instrument_id)
    base = _load_currency_obj(spec.base_currency)
    quote = _load_currency_obj(spec.quote_currency)
    settlement = _load_currency_obj(spec.settlement_currency)
    raw_symbol = N.Symbol(spec.raw_symbol)

    kwargs = {
        "instrument_id": inst_id,
        "raw_symbol": raw_symbol,
        "base_currency": base,
        "quote_currency": quote,
        "settlement_currency": settlement,
        "is_inverse": bool(spec.is_inverse),
        "price_precision": int(spec.price_precision),
        "size_precision": int(spec.size_precision),
        "price_increment": N.Price.from_str(spec.price_increment),
        "size_increment": N.Quantity.from_str(spec.size_increment),
        "ts_event": 0,
        "ts_init": 0,
        "maker_fee": Decimal(spec.maker_fee),
        "taker_fee": Decimal(spec.taker_fee),
    }
    try:
        return _call_with_supported_kwargs(N.CryptoPerpetual, kwargs)
    except Exception:
        return N.CryptoPerpetual(**kwargs)


def ensure_instruments_in_catalog(
    N: NautilusPhase4Imports,
    *,
    catalog_path: str,
    instrument_specs_path: str,
    instrument_ids: list[str],
) -> None:
    specs = load_instrument_specs(instrument_specs_path)
    catalog = N.ParquetDataCatalog(str(Path(catalog_path).expanduser().resolve()))
    for inst_id_str in instrument_ids:
        inst_id_str = str(inst_id_str).strip()
        if not inst_id_str:
            continue
        existing = None
        try:
            existing = catalog.instruments(instrument_ids=[inst_id_str])
        except Exception:
            try:
                existing = catalog.instruments(instrument_ids=[N.InstrumentId.from_str(inst_id_str)])
            except Exception:
                existing = None
        if existing:
            continue
        spec = specs.get(inst_id_str)
        if spec is None:
            raise InstrumentSpecError(
                f"instrument_specs.yaml is missing a spec for {inst_id_str!r}. Provide it under instruments[].instrument_id.",
            )
        instrument = _build_crypto_perpetual_from_spec(N, spec)
        catalog.write_data([instrument])


class StrategyFactoryError(ValueError):
    pass


def _require_decimal(cfg: dict[str, Any], key: str) -> Decimal:
    if key not in cfg:
        raise StrategyFactoryError(f"Missing required strategy.config.{key}")
    v = cfg[key]
    if isinstance(v, Decimal):
        return v
    if isinstance(v, (int, float)):
        return Decimal(str(v))
    if isinstance(v, str):
        return Decimal(v.strip())
    raise StrategyFactoryError(f"Invalid decimal for {key!r}: {v!r}")


def build_importable_strategies(
    N: NautilusPhase4Imports,
    *,
    strategy_name: str,
    strategy_config: dict[str, Any],
    instrument_ids: list[str],
) -> list[Any]:
    if not instrument_ids:
        raise StrategyFactoryError("No instrument_ids provided")
    strategies: list[Any] = []
    for inst_str in instrument_ids:
        inst = N.InstrumentId.from_str(inst_str)
        if strategy_name == "ema_cross_baseline":
            strategies.append(_build_ema_cross_baseline(N, inst, inst_str, strategy_config))
        elif strategy_name == "phoenix_lpi":
            strategies.append(_build_phoenix_lpi(N, inst, strategy_config))
        else:
            raise StrategyFactoryError(f"Unsupported strategy_name={strategy_name!r}")
    return strategies


def _build_ema_cross_baseline(N: NautilusPhase4Imports, inst: Any, inst_str: str, cfg: dict[str, Any]) -> Any:
    trade_size = _require_decimal(cfg, "trade_size")
    bar_type = N.BarType.from_str(f"{inst_str}-1-MINUTE-LAST-EXTERNAL")
    fast_ema_period = int(cfg.get("fast_ema_period", 10))
    slow_ema_period = int(cfg.get("slow_ema_period", 20))
    request_historical_bars = bool(cfg.get("request_historical_bars", False))
    close_positions_on_stop = bool(cfg.get("close_positions_on_stop", True))
    order_time_in_force = cfg.get("order_time_in_force", None)
    tif = None
    if order_time_in_force is not None:
        try:
            from nautilus_trader.model.enums import TimeInForce as _TIF

            tif = _TIF[str(order_time_in_force)]
        except Exception:
            tif = None
    config_obj: dict[str, Any] = {
        "instrument_id": inst,
        "bar_type": bar_type,
        "trade_size": trade_size,
        "fast_ema_period": fast_ema_period,
        "slow_ema_period": slow_ema_period,
        "request_historical_bars": request_historical_bars,
        "close_positions_on_stop": close_positions_on_stop,
    }
    if tif is not None:
        config_obj["order_time_in_force"] = tif
    kwargs = {
        "strategy_path": "phoenix_research.strategies.ema_cross_baseline:EmaCrossBaseline",
        "config_path": "phoenix_research.strategies.ema_cross_baseline:EmaCrossBaselineConfig",
        "config": config_obj,
    }
    return _call_with_supported_kwargs(N.ImportableStrategyConfig, kwargs)


def _build_phoenix_lpi(N: NautilusPhase4Imports, inst: Any, cfg: dict[str, Any]) -> Any:
    trade_size = _require_decimal(cfg, "trade_size")
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


class DataFactoryError(ValueError):
    pass


@dataclass(frozen=True)
class DataLoadPlan:
    catalog_path: str
    instrument_ids: list[str]
    include: list[str]
    data_range: TimeRange
    run_range: TimeRange

    def to_debug_dict(self) -> dict[str, object]:
        ds, de = self.data_range.to_iso()
        rs, re_ = self.run_range.to_iso()
        return {
            "catalog_path": self.catalog_path,
            "instrument_ids": list(self.instrument_ids),
            "include": list(self.include),
            "data_start_utc": ds,
            "data_end_utc": de,
            "run_start_utc": rs,
            "run_end_utc": re_,
        }


def _load_custom_data_classes() -> dict[str, Any]:
    return {
        "BinanceOiMetrics": BinanceOiMetrics,
        "BinanceFundingRate": BinanceFundingRate,
        "BinanceBookDepthPct": BinanceBookDepthPct,
    }


def build_backtest_data_configs(N: NautilusPhase4Imports, plan: DataLoadPlan) -> list[Any]:
    out: list[Any] = []
    include_set = {s.strip() for s in plan.include if str(s).strip()}
    custom = _load_custom_data_classes()
    ds, de = plan.data_range.to_iso()

    for inst_id in plan.instrument_ids:
        if "QuoteTick" in include_set:
            out.append(
                _call_with_supported_kwargs(
                    N.BacktestDataConfig,
                    {
                        "catalog_path": plan.catalog_path,
                        "data_cls": N.QuoteTick,
                        "instrument_id": inst_id,
                        "start_time": ds,
                        "end_time": de,
                    },
                ),
            )
        if "TradeTick" in include_set:
            out.append(
                _call_with_supported_kwargs(
                    N.BacktestDataConfig,
                    {
                        "catalog_path": plan.catalog_path,
                        "data_cls": N.TradeTick,
                        "instrument_id": inst_id,
                        "start_time": ds,
                        "end_time": de,
                    },
                ),
            )
        if "Bar" in include_set:
            out.append(
                _call_with_supported_kwargs(
                    N.BacktestDataConfig,
                    {
                        "catalog_path": plan.catalog_path,
                        "data_cls": N.Bar,
                        "instrument_id": inst_id,
                        "start_time": ds,
                        "end_time": de,
                    },
                ),
            )
        if "BinanceOiMetrics" in include_set:
            out.append(
                _call_with_supported_kwargs(
                    N.BacktestDataConfig,
                    {
                        "catalog_path": plan.catalog_path,
                        "data_cls": custom["BinanceOiMetrics"],
                        "instrument_id": inst_id,
                        "start_time": ds,
                        "end_time": de,
                    },
                ),
            )
        if "BinanceFundingRate" in include_set:
            out.append(
                _call_with_supported_kwargs(
                    N.BacktestDataConfig,
                    {
                        "catalog_path": plan.catalog_path,
                        "data_cls": custom["BinanceFundingRate"],
                        "instrument_id": inst_id,
                        "start_time": ds,
                        "end_time": de,
                    },
                ),
            )
        if "BinanceBookDepthPct" in include_set:
            out.append(
                _call_with_supported_kwargs(
                    N.BacktestDataConfig,
                    {
                        "catalog_path": plan.catalog_path,
                        "data_cls": custom["BinanceBookDepthPct"],
                        "instrument_id": inst_id,
                        "start_time": ds,
                        "end_time": de,
                    },
                ),
            )
    return out


@dataclass(frozen=True)
class RunReports:
    orders: pd.DataFrame
    fills: pd.DataFrame
    positions: pd.DataFrame
    account: pd.DataFrame


@dataclass(frozen=True)
class RunPerformance:
    starting_equity: float | None
    ending_equity: float | None
    total_pnl: float | None
    total_return: float | None
    max_drawdown: float | None
    sharpe: float | None
    deflated_sharpe: float | None


@dataclass(frozen=True)
class RunCounts:
    orders: int
    fills: int
    positions: int


def _safe_report(trader: Any, method_names: list[str]) -> pd.DataFrame:
    for name in method_names:
        fn = getattr(trader, name, None)
        if fn is None:
            continue
        try:
            out = fn()
            if isinstance(out, pd.DataFrame):
                return out
            return pd.DataFrame(out)
        except Exception:
            continue
    return pd.DataFrame()


def _safe_account_report(trader: Any, venue: Any) -> pd.DataFrame:
    fn = getattr(trader, "generate_account_report", None)
    if fn is None:
        return pd.DataFrame()
    try:
        out = fn(venue)
        if isinstance(out, pd.DataFrame):
            return out
        return pd.DataFrame(out)
    except Exception:
        return pd.DataFrame()


def extract_reports(N: NautilusPhase4Imports, engine: Any, *, venue_name: str) -> RunReports:
    trader = engine.trader
    orders = _safe_report(trader, ["generate_orders_report"])
    fills = _safe_report(trader, ["generate_fills_report", "generate_order_fills_report"])
    positions = _safe_report(trader, ["generate_positions_report"])
    venue = N.Venue(str(venue_name))
    account = _safe_account_report(trader, venue)
    return RunReports(
        orders=orders,
        fills=fills,
        positions=positions,
        account=account,
    )


def compute_counts(reports: RunReports) -> RunCounts:
    return RunCounts(
        orders=int(len(reports.orders)),
        fills=int(len(reports.fills)),
        positions=int(len(reports.positions)),
    )


def _equity_from_account_report(acct: pd.DataFrame) -> tuple[float | None, float | None, float | None, float | None]:
    if acct is None or len(acct) == 0:
        return None, None, None, None
    equity_col_candidates = [
        "equity",
        "equity_total",
        "net_liquidation_value",
        "balance_total",
    ]
    equity_col = next((c for c in equity_col_candidates if c in acct.columns), None)
    if equity_col is None:
        return None, None, None, None
    s = pd.to_numeric(acct[equity_col], errors="coerce").dropna()
    if len(s) < 2:
        return None, None, None, None
    start = float(s.iloc[0])
    end = float(s.iloc[-1])
    pnl = end - start
    ret = (end / start - 1.0) if start != 0.0 else None
    return start, end, pnl, ret


def _max_drawdown_from_account(acct: pd.DataFrame) -> float | None:
    equity_col_candidates = [
        "equity",
        "equity_total",
        "net_liquidation_value",
        "balance_total",
    ]
    equity_col = next((c for c in equity_col_candidates if c in acct.columns), None)
    if equity_col is None:
        return None
    equity = pd.to_numeric(acct[equity_col], errors="coerce").dropna()
    if len(equity) < 2:
        return None
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def _sharpe_from_account(acct: pd.DataFrame) -> float | None:
    equity_col_candidates = [
        "equity",
        "equity_total",
        "net_liquidation_value",
        "balance_total",
    ]
    equity_col = next((c for c in equity_col_candidates if c in acct.columns), None)
    if equity_col is None:
        return None
    equity = pd.to_numeric(acct[equity_col], errors="coerce").dropna()
    if len(equity) < 3:
        return None
    rets = equity.pct_change().dropna()
    if len(rets) < 3:
        return None
    mu = float(rets.mean())
    sd = float(rets.std(ddof=0))
    if sd == 0.0:
        return None
    return float(math.sqrt(len(rets)) * (mu / sd))


def _first_float(obj: Any, keys: list[str]) -> float | None:
    if obj is None:
        return None
    for k in keys:
        if isinstance(obj, dict) and k in obj:
            try:
                return float(obj[k])
            except Exception:
                continue
    return None


def compute_performance(engine: Any, reports: RunReports) -> RunPerformance:
    starting_equity, ending_equity, total_pnl, total_return = _equity_from_account_report(reports.account)
    sharpe = None
    max_dd = None
    try:
        stats_returns = engine.portfolio.analyzer.get_performance_stats_returns()
        stats_general = engine.portfolio.analyzer.get_performance_stats_general()
        sharpe = _first_float(stats_returns, ["sharpe", "sharpe_ratio"])
        max_dd = _first_float(stats_general, ["max_drawdown", "max_drawdown_pct"])
    except Exception:
        pass
    if max_dd is None:
        max_dd = _max_drawdown_from_account(reports.account)
    if sharpe is None:
        sharpe = _sharpe_from_account(reports.account)
    return RunPerformance(
        starting_equity=starting_equity,
        ending_equity=ending_equity,
        total_pnl=total_pnl,
        total_return=total_return,
        max_drawdown=max_dd,
        sharpe=sharpe,
        deflated_sharpe=None,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_yaml(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(yaml.safe_dump(obj, sort_keys=True), encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_df_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    df2 = df.copy()
    for col in ("ts_event", "ts_init", "event_time", "timestamp"):
        if col in df2.columns:
            try:
                df2 = df2.sort_values(by=[col], kind="mergesort")
            except Exception:
                pass
            break
    df2.to_csv(path, index=False)


def write_run_bundle(
    *,
    out_dir: Path,
    resolved_config: dict[str, Any],
    manifest_ref: dict[str, Any],
    reports: dict[str, pd.DataFrame],
    summary: dict[str, Any],
) -> None:
    ensure_dir(out_dir)
    write_yaml(out_dir / "config_resolved.yaml", resolved_config)
    write_json(out_dir / "manifest_ref.json", manifest_ref)
    if "orders" in reports:
        write_df_csv(out_dir / "orders.csv", reports["orders"])
    if "fills" in reports:
        write_df_csv(out_dir / "fills.csv", reports["fills"])
    if "positions" in reports:
        write_df_csv(out_dir / "positions.csv", reports["positions"])
    if "account" in reports:
        write_df_csv(out_dir / "account_report.csv", reports["account"])
    write_json(out_dir / "summary.json", summary)


def write_suite_index(out_dir: Path, *, suite_summary: dict[str, Any]) -> None:
    ensure_dir(out_dir)
    write_json(out_dir / "aggregate_summary.json", suite_summary)


@dataclass(frozen=True)
class RunResult:
    run_dir: str
    summary: dict[str, Any]


def _hash_strategy_config(strategy: Any) -> str:
    obj = {
        "name": getattr(strategy, "name", None),
        "config": getattr(strategy, "config", None),
    }
    payload = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _data_range_with_warmup(cfg: Phase4SuiteConfig, run_start: Any, run_end: Any) -> TimeRange:
    warmup = parse_duration(cfg.time.warmup)
    data_start = run_start - warmup
    return TimeRange(data_start, run_end)


def run_phase4(cfg: Phase4SuiteConfig) -> list[RunResult]:
    _install_shims()
    N = load_nautilus_phase4_imports()
    det_env = set_global_determinism(DeterminismConfig(seed=int(cfg.determinism_seed)))
    suite_id = cfg.output.suite_name or cfg.resolved_suite_id()
    out_root = Path(cfg.output.dir).expanduser().resolve() / suite_id
    if out_root.exists() and not cfg.output.overwrite:
        raise FileExistsError(f"Output directory exists and overwrite=false: {out_root}")
    ensure_dir(out_root)
    write_yaml(out_root / "suite_config_resolved.yaml", to_primitive_dict(cfg))
    write_yaml(out_root / "determinism.yaml", det_env)
    if not cfg.catalog.instrument_specs_path:
        raise ValueError("catalog.instrument_specs_path is required")
    ensure_instruments_in_catalog(
        N,
        catalog_path=cfg.catalog.path,
        instrument_specs_path=cfg.catalog.instrument_specs_path,
        instrument_ids=list(cfg.universe.instrument_ids),
    )

    overall = cfg.time.range()
    manifest_ref = {
        "catalog_path": str(Path(cfg.catalog.path).expanduser().resolve()),
        "manifest_path": (str(Path(cfg.catalog.manifest_path).expanduser().resolve()) if cfg.catalog.manifest_path else None),
    }
    if cfg.catalog.manifest_path:
        try:
            mp = Path(cfg.catalog.manifest_path).expanduser().resolve()
            manifest_ref["manifest_json"] = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            manifest_ref["manifest_json"] = None

    results: list[RunResult] = []
    if cfg.mode.kind == "single":
        res = _run_one(
            N,
            cfg=cfg,
            out_dir=out_root / "windows" / "single",
            manifest_ref=manifest_ref,
            run_range=TimeRange(overall.start, overall.end),
            data_range=_data_range_with_warmup(cfg, overall.start, overall.end),
            window_id="single",
            fold_meta=None,
        )
        results.append(res)
    else:
        wf = cfg.mode.walk_forward
        train = parse_duration(wf.train)
        test = parse_duration(wf.test)
        step = parse_duration(wf.step)
        purge = parse_duration(wf.purge)
        embargo = parse_duration(wf.embargo)
        folds = list(
            iter_walk_forward_folds(
                overall=overall,
                train=train,
                test=test,
                step=step,
                purge=purge,
                embargo=embargo,
                max_folds=wf.max_folds,
            ),
        )
        for fold in folds:
            window_id = fold.to_id_str()
            out_dir = out_root / "windows" / window_id
            run_range = fold.test_range
            data_range = _data_range_with_warmup(cfg, fold.test_range.start, fold.test_range.end)
            res = _run_one(
                N,
                cfg=cfg,
                out_dir=out_dir,
                manifest_ref=manifest_ref,
                run_range=run_range,
                data_range=data_range,
                window_id=window_id,
                fold_meta=fold_summary_dict(fold),
            )
            results.append(res)
        suite_summary = {
            "schema_version": "1",
            "suite_id": suite_id,
            "counts": {"windows": len(results)},
            "windows": [r.summary for r in results],
        }
        write_suite_index(out_root, suite_summary=suite_summary)
    return results


def _run_one(
    N: NautilusPhase4Imports,
    *,
    cfg: Phase4SuiteConfig,
    out_dir: Path,
    manifest_ref: dict[str, Any],
    run_range: TimeRange,
    data_range: TimeRange,
    window_id: str,
    fold_meta: dict[str, Any] | None,
) -> RunResult:
    _install_shims()
    ensure_dir(out_dir)
    run_start_iso = format_utc_datetime(run_range.start)
    run_end_iso = format_utc_datetime(run_range.end)
    data_start_iso = format_utc_datetime(data_range.start)
    data_end_iso = format_utc_datetime(data_range.end)

    strategies = build_importable_strategies(
        N,
        strategy_name=cfg.strategy.name,
        strategy_config=cfg.strategy.config,
        instrument_ids=list(cfg.universe.instrument_ids),
    )

    venue_cfg = _call_with_supported_kwargs(
        N.BacktestVenueConfig,
        {
            "name": str(cfg.universe.venue),
            "oms_type": str(cfg.universe.oms_type),
            "account_type": str(cfg.universe.account_type),
            "base_currency": str(cfg.universe.base_currency),
            "starting_balances": list(cfg.universe.starting_balances),
        },
    )

    logging_cfg = _call_with_supported_kwargs(
        N.LoggingConfig,
        {
            "log_level": "ERROR",
            "log_colors": False,
            "use_pyo3": False,
        },
    )

    engine_cfg = _call_with_supported_kwargs(
        N.BacktestEngineConfig,
        {
            "strategies": strategies,
            "logging": logging_cfg,
        },
    )

    plan = DataLoadPlan(
        catalog_path=str(Path(cfg.catalog.path).expanduser().resolve()),
        instrument_ids=list(cfg.universe.instrument_ids),
        include=list(cfg.data.include),
        data_range=data_range,
        run_range=run_range,
    )
    data_cfgs = build_backtest_data_configs(N, plan)

    run_cfg_kwargs = {
        "engine": engine_cfg,
        "venues": [venue_cfg],
        "data": data_cfgs,
        "start": run_start_iso,
        "end": run_end_iso,
        "raise_exception": True,
    }
    run_cfg = _call_with_supported_kwargs(N.BacktestRunConfig, run_cfg_kwargs)
    node = _call_with_supported_kwargs(N.BacktestNode, {"configs": [run_cfg]})

    try:
        _ = node.run()
        run_id = getattr(run_cfg, "id", None)
        engine = node.get_engine(run_id) if run_id is not None else node.get_engine()

        reports = extract_reports(N, engine, venue_name=str(cfg.universe.venue))
        counts = compute_counts(reports)
        perf = compute_performance(engine, reports)

        summary = {
            "schema_version": "1",
            "experiment_id": cfg.output.suite_name or cfg.resolved_suite_id(),
            "run_config_id": getattr(run_cfg, "id", None),
            "window_id": window_id,
            "strategy": {
                "name": cfg.strategy.name,
                "variant": None,
                "config_hash": _hash_strategy_config(cfg.strategy),
            },
            "universe": {
                "venue": str(cfg.universe.venue),
                "instrument_ids": list(cfg.universe.instrument_ids),
                "start_utc": run_start_iso,
                "end_utc": run_end_iso,
                "data_start_utc": data_start_iso,
                "data_end_utc": data_end_iso,
            },
            "fold": fold_meta,
            "counts": {
                "orders": counts.orders,
                "fills": counts.fills,
                "positions": counts.positions,
            },
            "performance": {
                "starting_equity": perf.starting_equity,
                "ending_equity": perf.ending_equity,
                "total_pnl": perf.total_pnl,
                "total_return": perf.total_return,
                "max_drawdown": perf.max_drawdown,
                "sharpe": perf.sharpe,
                "deflated_sharpe": perf.deflated_sharpe,
            },
            "artifacts": {
                "orders_csv": "orders.csv",
                "fills_csv": "fills.csv",
                "positions_csv": "positions.csv",
                "account_report_csv": "account_report.csv",
            },
        }

        resolved_cfg = {
            "schema_version": "1",
            "window_id": window_id,
            "suite_config": to_primitive_dict(cfg),
            "run_range": {"start_utc": run_start_iso, "end_utc": run_end_iso},
            "data_range": {"start_utc": data_start_iso, "end_utc": data_end_iso},
            "data_plan": plan.to_debug_dict(),
        }

        write_run_bundle(
            out_dir=out_dir,
            resolved_config=resolved_cfg,
            manifest_ref=manifest_ref,
            reports={
                "orders": reports.orders,
                "fills": reports.fills,
                "positions": reports.positions,
                "account": reports.account,
            },
            summary=summary,
        )

        return RunResult(run_dir=str(out_dir), summary=summary)
    finally:
        try:
            node.dispose()
        except Exception:
            pass


def _cmd_scan(raw_roots: list[str], output: str | None) -> int:
    roots = default_roots_from_args(raw_roots)
    inv = build_inventory(roots)
    obj = [asdict(r) for r in inv]
    if output:
        out = Path(output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    else:
        print(json.dumps(obj[: min(len(obj), 50)], indent=2, sort_keys=True))
        if len(obj) > 50:
            print(f"... ({len(obj)} total rows)")
    return 0


def _cmd_classify(path: str) -> int:
    p = Path(path).expanduser().resolve()
    parsed = classify_binance_vision_path(p)
    if parsed is None:
        print("null")
        return 1
    print(json.dumps({**asdict(parsed), "path": str(parsed.path)}, indent=2, sort_keys=True))
    return 0


def _cmd_ingest(config_path: str) -> int:
    cfg = load_etl_config(Path(config_path).expanduser().resolve())
    out = run_etl(cfg)
    print(str(out.manifest_path))
    print(str(out.validation_report_path))
    print(str(out.catalog_path))
    return 0


def _cmd_run(config_path: str) -> int:
    cfg = load_suite_config(config_path)
    _ = run_phase4(cfg)
    return 0


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="phoenix-merged")
    sub = ap.add_subparsers(dest="cmd", required=True)

    scan = sub.add_parser("scan")
    scan.add_argument("--raw-root", action="append", required=True)
    scan.add_argument("--output", default=None)
    scan.set_defaults(func=lambda ns: _cmd_scan(ns.raw_root, ns.output))

    classify = sub.add_parser("classify")
    classify.add_argument("path")
    classify.set_defaults(func=lambda ns: _cmd_classify(ns.path))

    ingest = sub.add_parser("ingest")
    ingest.add_argument("--config", required=True)
    ingest.set_defaults(func=lambda ns: _cmd_ingest(ns.config))

    runp = sub.add_parser("run")
    runp.add_argument("--config", required=True)
    runp.set_defaults(func=lambda ns: _cmd_run(ns.config))

    ns = ap.parse_args(argv)
    raise SystemExit(ns.func(ns))


if __name__ == "__main__":
    main()
