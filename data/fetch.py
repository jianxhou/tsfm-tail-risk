"""Fetchers for the data universe.

Yahoo Finance chart API for market series, FRED CSV endpoint for rates — both via
stdlib HTTP (no extra dependency). Every fetch keeps the raw response bytes so
build.py can snapshot them under data/raw/ (Yahoo revises history; the snapshot
plus its sha256 in the manifest is our reproducibility anchor).

Return construction lives in build.py so every series goes through the identical
transform.
"""

from __future__ import annotations

import io
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

YAHOO_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    "?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit"
)
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={symbol}&cosd={start}&coed={end}"
TREASURY_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
    "&field_tdr_date_value={year}&page&_format=csv"
)
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"

DEFAULT_START = "2000-01-01"  # proposal §3.2: data 2000 → today


@dataclass
class FetchResult:
    df: pd.DataFrame          # columns: date (naive, exchange-local), close
    symbol: str
    source: str
    url: str
    timezone: str             # exchange/reporting timezone
    currency: str
    price_field: str          # 'adjclose' | 'close' | 'yield_pct'
    splits: int               # split events reported in-range
    dividends: int            # dividend events reported in-range
    raw: bytes                # verbatim response body (snapshot to data/raw/)
    raw_format: str           # 'json' | 'csv'
    fetched_at: str           # UTC ISO timestamp


def _http_get(url: str, retries: int = 4, backoff: float = 3.0) -> bytes:
    # timeout is generous: FRED's keyless CSV endpoint can take >30s to first byte
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=90) as resp:
                return resp.read()
        except Exception as err:  # noqa: BLE001 - retry any transport error
            last_err = err
            time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} attempts: {url}") from last_err


def fetch_yahoo(symbol: str, start: str = DEFAULT_START, end: str | None = None) -> FetchResult:
    """Fetch daily prices from the Yahoo chart API.

    Uses adjclose when the source provides it (splits + dividends folded in — required
    for single stocks and ETF proxies; identical to close for indices/FX/crypto/futures).
    Daily timestamps are session instants in UTC seconds; the trading date is recovered
    in the exchange's own timezone so a Tokyo session never bleeds across UTC midnight.
    """
    p1 = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    end = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    p2 = int(datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()) + 86400

    url = YAHOO_URL.format(symbol=urllib.parse.quote(symbol), p1=p1, p2=p2)
    raw = _http_get(url)
    payload = json.loads(raw.decode("utf-8"))

    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(f"{symbol}: Yahoo error {chart['error']}")
    result = chart["result"][0]
    meta = result["meta"]
    tz_name = meta.get("exchangeTimezoneName", "UTC")

    ts = result.get("timestamp")
    if not ts:
        raise RuntimeError(f"{symbol}: empty timestamp array")

    close = result["indicators"]["quote"][0]["close"]
    adj = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
    if adj is not None and len(adj) == len(close):
        prices, price_field = adj, "adjclose"
    else:
        prices, price_field = close, "close"

    idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(tz_name)
    df = pd.DataFrame({"date": idx.date, "close": prices})
    df = df.dropna(subset=["close"]).drop_duplicates(subset="date", keep="last")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    events = result.get("events", {})
    return FetchResult(
        df=df,
        symbol=symbol,
        source="yahoo",
        url=url,
        timezone=tz_name,
        currency=meta.get("currency", ""),
        price_field=price_field,
        splits=len(events.get("splits", {})),
        dividends=len(events.get("dividends", {})),
        raw=raw,
        raw_format="json",
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def fetch_fred(symbol: str, start: str = DEFAULT_START, end: str | None = None) -> FetchResult:
    """Fetch a daily FRED series (constant-maturity yields, percent).

    Missing observations (holidays) come back as '.' and are dropped, so a
    first-difference downstream spans the gap — standard for daily yield changes.
    """
    end = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    url = FRED_URL.format(symbol=urllib.parse.quote(symbol), start=start, end=end)
    raw = _http_get(url)

    df = pd.read_csv(io.BytesIO(raw))
    if df.shape[1] != 2:
        raise RuntimeError(f"{symbol}: unexpected FRED CSV shape {df.shape}")
    df.columns = ["date", "close"]
    df["close"] = pd.to_numeric(df["close"], errors="coerce")  # '.' -> NaN
    df = df.dropna(subset=["close"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"{symbol}: FRED returned no observations")

    return FetchResult(
        df=df,
        symbol=symbol,
        source="fred",
        url=url,
        timezone="America/New_York",  # H.15 release; informational, dates are release dates
        currency="USD",
        price_field="yield_pct",
        splits=0,
        dividends=0,
        raw=raw,
        raw_format="csv",
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


# per-year cache so the four maturities share one download pass per process
_treasury_years: dict[int, tuple[bytes, pd.DataFrame]] = {}


def _treasury_year(year: int) -> tuple[bytes, pd.DataFrame]:
    if year not in _treasury_years:
        raw = _http_get(TREASURY_URL.format(year=year))
        df = pd.read_csv(io.BytesIO(raw))
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
        _treasury_years[year] = (raw, df)
        time.sleep(0.3)
    return _treasury_years[year]


def fetch_treasury(symbol: str, start: str = DEFAULT_START, end: str | None = None) -> FetchResult:
    """Daily Treasury par yield curve from treasury.gov (the primary source behind
    FRED's DGS series). `symbol` is the CSV column name, e.g. '2 Yr', '30 Yr'.

    Fetched per calendar year and concatenated; the raw snapshot is the verbatim
    year responses joined in order (identical bytes across the four maturities).
    Missing cells (e.g. the 30y issuance gap 2002-02 → 2006-02) are dropped, so
    downstream first differences span the gap.
    """
    end = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    y0 = int(start[:4])
    y1 = int(end[:4])
    raws, frames = [], []
    for year in range(y0, y1 + 1):
        raw, df = _treasury_year(year)
        raws.append(raw)
        frames.append(df)

    allyears = pd.concat(frames, ignore_index=True)
    if symbol not in allyears.columns:
        raise RuntimeError(f"treasury.gov: column '{symbol}' not in {list(allyears.columns)}")
    out = allyears[["Date", symbol]].rename(columns={"Date": "date", symbol: "close"})
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["close"])
    out = out[(out["date"] >= start) & (out["date"] <= end)]
    out = out.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)
    if out.empty:
        raise RuntimeError(f"treasury.gov: no observations for '{symbol}'")

    return FetchResult(
        df=out,
        symbol=symbol,
        source="treasury",
        url=TREASURY_URL.format(year=f"{y0}..{y1}"),
        timezone="America/New_York",
        currency="USD",
        price_field="yield_pct",
        splits=0,
        dividends=0,
        raw=b"".join(raws),
        raw_format="csv",
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


FETCHERS = {"yahoo": fetch_yahoo, "fred": fetch_fred, "treasury": fetch_treasury}
