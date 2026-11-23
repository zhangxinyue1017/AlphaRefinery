'''Minimal Alpaca historical stock-bar adapter.'''

from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta, timezone
from typing import Iterable

import pandas as pd


class AlpacaDataError(RuntimeError):
    '''Raised when credentials, request settings, or an Alpaca request fail.'''


def _date_bound(value: str | date | datetime, *, end_inclusive: bool) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, date):
        result = datetime.combine(value, time.min)
    else:
        result = datetime.combine(date.fromisoformat(str(value)), time.min)
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    else:
        result = result.astimezone(timezone.utc)
    return result + timedelta(days=1) if end_inclusive else result


def fetch_daily_stock_bars(
    *,
    symbols: Iterable[str],
    start: str | date | datetime,
    end: str | date | datetime,
    feed: str = "sip",
    adjustment: str = "split",
    client=None,
) -> pd.DataFrame:
    '''Fetch daily US stock bars through the official ``alpaca-py`` SDK.

    The SDK follows ``next_page_token`` by default and merges all pages. The
    requested feed is passed through unchanged; this function never retries a
    denied SIP request with IEX.
    '''

    clean_symbols = tuple(dict.fromkeys(str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()))
    if not clean_symbols:
        raise ValueError("at least one symbol is required")

    try:
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
    except ImportError as exc:
        raise AlpacaDataError("alpaca-py is required; install the us-data project extra") from exc

    try:
        data_feed = DataFeed(str(feed).strip().lower())
        adjustment_mode = Adjustment(str(adjustment).strip().lower())
    except ValueError as exc:
        raise ValueError(f"unsupported Alpaca feed or adjustment: feed={feed!r}, adjustment={adjustment!r}") from exc

    if client is None:
        api_key = os.getenv("ALPACA_API_KEY", "").strip()
        secret_key = os.getenv("ALPACA_SECRET_KEY", "").strip()
        if not api_key or not secret_key:
            raise AlpacaDataError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")
        client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)

    request = StockBarsRequest(
        symbol_or_symbols=list(clean_symbols),
        timeframe=TimeFrame.Day,
        start=_date_bound(start, end_inclusive=False),
        end=_date_bound(end, end_inclusive=True),
        feed=data_feed,
        adjustment=adjustment_mode,
    )
    try:
        response = client.get_stock_bars(request)
    except Exception as exc:
        raise AlpacaDataError(
            f"Alpaca daily-bar request failed for feed={feed!r}; no feed fallback was attempted: {exc}"
        ) from exc

    frame = getattr(response, "df", None)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise AlpacaDataError("Alpaca returned no daily bars for the requested symbols and dates")
    return frame.copy()


__all__ = ["AlpacaDataError", "fetch_daily_stock_bars"]
