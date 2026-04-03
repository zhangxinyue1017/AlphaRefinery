from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from ._bootstrap import ensure_project_roots
from .contract import EXPECTED_INDEX_NAMES, validate_data

ensure_project_roots()

from gp_factor_qlib.data.data_process import (  # noqa: E402
    extract_baostock_filter_flags,
    filter_panel,
)


SeriesDict = dict[str, pd.Series]
EPS = 1e-12


def load_panel(
    panel_path: str | Path,
    *,
    columns: Iterable[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    instruments: Iterable[str] | None = None,
) -> pd.DataFrame:
    panel = pd.read_parquet(panel_path)
    if not isinstance(panel.index, pd.MultiIndex):
        raise TypeError("panel must use MultiIndex(datetime, instrument)")
    if tuple(panel.index.names) != EXPECTED_INDEX_NAMES:
        panel.index = panel.index.set_names(list(EXPECTED_INDEX_NAMES))

    panel = panel.sort_index()

    if columns is not None:
        keep = [col for col in columns if col in panel.columns]
        panel = panel[keep]

    if start is not None or end is not None:
        dt = panel.index.get_level_values("datetime")
        mask = pd.Series(True, index=panel.index)
        if start is not None:
            mask &= dt >= pd.Timestamp(start)
        if end is not None:
            mask &= dt <= pd.Timestamp(end)
        panel = panel.loc[mask.to_numpy()]

    if instruments is not None:
        wanted = set(str(x) for x in instruments)
        inst = panel.index.get_level_values("instrument")
        panel = panel.loc[inst.isin(wanted)]

    return panel


def load_benchmark_series(benchmark_path: str | Path) -> dict[str, pd.Series]:
    df = pd.read_csv(benchmark_path)
    if "date" not in df.columns:
        raise ValueError(f"{benchmark_path} missing date column")
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["date"])
    df = df.sort_values("datetime")
    out: dict[str, pd.Series] = {}
    for src, dst in {
        "open": "benchmark_open",
        "close": "benchmark_close",
        "high": "benchmark_high",
        "low": "benchmark_low",
        "volume": "benchmark_volume",
        "amount": "benchmark_amount",
    }.items():
        if src in df.columns:
            out[dst] = pd.Series(pd.to_numeric(df[src], errors="coerce").to_numpy(), index=df["datetime"], name=dst)
    if "benchmark_close" in out:
        out["market_return"] = out["benchmark_close"].pct_change()
    return out


def apply_main_filters(
    panel: pd.DataFrame,
    *,
    stock_only: bool = False,
    instrument_regex: str | None = None,
    zero_volume_price_na: bool = True,
    drop_limit_move: bool = False,
    exclude_st: bool = False,
    exclude_suspended: bool = False,
    min_listed_days: int | None = None,
    min_turnover_quantile: float | None = None,
    min_liquidity_quantile: float | None = None,
    liquidity_col: str = "amount",
    liquidity_fallback_col: str = "volume",
    liquidity_rolling_days: int = 20,
) -> tuple[pd.DataFrame, dict[str, object]]:
    flags = extract_baostock_filter_flags(panel)
    clean, report = filter_panel(
        panel,
        stock_only=stock_only,
        instrument_regex=instrument_regex,
        zero_volume_price_na=zero_volume_price_na,
        drop_limit_move=drop_limit_move,
        exclude_st=exclude_st,
        st_flags=flags.get("st_flags"),
        exclude_suspended=exclude_suspended,
        suspended_flags=flags.get("suspended_flags"),
        min_listed_days=min_listed_days,
        min_turnover_quantile=min_turnover_quantile,
        min_liquidity_quantile=min_liquidity_quantile,
        liquidity_col=liquidity_col,
        liquidity_fallback_col=liquidity_fallback_col,
        liquidity_rolling_days=liquidity_rolling_days,
    )
    return clean, report


def _should_apply_filters(
    *,
    apply_filters: bool,
    stock_only: bool,
    instrument_regex: str | None,
    drop_limit_move: bool,
    exclude_st: bool,
    exclude_suspended: bool,
    min_listed_days: int | None,
    min_turnover_quantile: float | None,
    min_liquidity_quantile: float | None,
) -> bool:
    return bool(
        apply_filters
        or stock_only
        or instrument_regex
        or drop_limit_move
        or exclude_st
        or exclude_suspended
        or min_listed_days is not None
        or min_turnover_quantile is not None
        or min_liquidity_quantile is not None
    )


def _broadcast_daily_series_to_panel(series: pd.Series, target_index: pd.MultiIndex) -> pd.Series:
    dt = target_index.get_level_values("datetime")
    values = series.reindex(dt).to_numpy()
    out = pd.Series(values, index=target_index, name=series.name)
    out.index = out.index.set_names(list(EXPECTED_INDEX_NAMES))
    return out


def _infer_baostock_float_market_cap(
    *,
    close: pd.Series,
    volume: pd.Series,
    turnover: pd.Series | None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    if turnover is None:
        nan_s = pd.Series(np.nan, index=close.index, dtype=float)
        return nan_s.rename("turnover_rate"), nan_s.rename("float_shares"), nan_s.rename("float_market_cap")

    turnover_pct = pd.to_numeric(turnover, errors="coerce")
    turnover_rate = (turnover_pct / 100.0).where(turnover_pct > 0).rename("turnover_rate")
    float_shares_raw = (volume.abs() / turnover_rate.replace(0.0, np.nan)).where(turnover_rate > 0)
    float_shares = (
        float_shares_raw.groupby(level="instrument", group_keys=False)
        .transform(lambda s: s.rolling(20, min_periods=1).median().ffill().bfill())
        .rename("float_shares")
    )
    float_shares = pd.Series(float_shares.to_numpy(), index=close.index, name="float_shares")
    float_market_cap = (close * float_shares).where((close > 0) & (float_shares > 0)).rename("float_market_cap")
    return turnover_rate, float_shares, float_market_cap


def _infer_cap_fields(data: SeriesDict) -> None:
    close = data.get("close")
    volume = data.get("volume")
    amount = data.get("amount")
    if close is None or volume is None:
        return

    turnover = data.get("turnover")
    turnover_rate, float_shares, float_market_cap = _infer_baostock_float_market_cap(
        close=close,
        volume=volume,
        turnover=turnover,
    )
    if turnover is not None:
        data["turnover_rate"] = turnover_rate
    data["float_shares"] = float_shares
    data["float_market_cap"] = float_market_cap

    cap_proxy = pd.Series(np.nan, index=close.index, dtype=float, name="cap")
    if amount is not None:
        cap_proxy = (
            amount.replace(0.0, np.nan)
            .groupby(level="instrument", group_keys=False)
            .rolling(252, min_periods=21)
            .mean()
            .droplevel(0)
            .rename("cap")
        )

    cap = float_market_cap.combine_first(cap_proxy).rename("cap")
    size = np.log(cap.clip(lower=EPS)).rename("size")
    data["cap"] = cap
    data["size"] = size


def build_data_bundle(
    panel_path: str | Path,
    *,
    benchmark_path: str | Path | None = None,
    columns: Iterable[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    instruments: Iterable[str] | None = None,
    extra_series: Mapping[str, pd.Series] | None = None,
    apply_filters: bool = False,
    stock_only: bool = False,
    instrument_regex: str | None = None,
    zero_volume_price_na: bool = True,
    drop_limit_move: bool = False,
    exclude_st: bool = False,
    exclude_suspended: bool = False,
    min_listed_days: int | None = None,
    min_turnover_quantile: float | None = None,
    min_liquidity_quantile: float | None = None,
    liquidity_col: str = "amount",
    liquidity_fallback_col: str = "volume",
    liquidity_rolling_days: int = 20,
) -> tuple[SeriesDict, dict[str, object]]:
    panel = load_panel(
        panel_path,
        columns=columns,
        start=start,
        end=end,
        instruments=instruments,
    )
    filter_report: dict[str, object] = {}

    if _should_apply_filters(
        apply_filters=apply_filters,
        stock_only=stock_only,
        instrument_regex=instrument_regex,
        drop_limit_move=drop_limit_move,
        exclude_st=exclude_st,
        exclude_suspended=exclude_suspended,
        min_listed_days=min_listed_days,
        min_turnover_quantile=min_turnover_quantile,
        min_liquidity_quantile=min_liquidity_quantile,
    ):
        panel, filter_report = apply_main_filters(
            panel,
            stock_only=stock_only,
            instrument_regex=instrument_regex,
            zero_volume_price_na=zero_volume_price_na,
            drop_limit_move=drop_limit_move,
            exclude_st=exclude_st,
            exclude_suspended=exclude_suspended,
            min_listed_days=min_listed_days,
            min_turnover_quantile=min_turnover_quantile,
            min_liquidity_quantile=min_liquidity_quantile,
            liquidity_col=liquidity_col,
            liquidity_fallback_col=liquidity_fallback_col,
            liquidity_rolling_days=liquidity_rolling_days,
        )

    data: SeriesDict = {}
    for col in panel.columns:
        data[col] = pd.to_numeric(panel[col], errors="coerce").astype(float).rename(col)

    if "close" in data:
        data["returns"] = data["close"].groupby(level="instrument", group_keys=False).pct_change().rename("returns")

    _infer_cap_fields(data)

    if benchmark_path is not None:
        benchmark = load_benchmark_series(benchmark_path)
        for name, series in benchmark.items():
            data[name] = _broadcast_daily_series_to_panel(series, panel.index)

    if extra_series is not None:
        for name, series in extra_series.items():
            aligned = series.reindex(panel.index)
            aligned.index = aligned.index.set_names(list(EXPECTED_INDEX_NAMES))
            data[name] = aligned.rename(name)

    validate_data(data)
    meta = {
        "filter_report": filter_report,
        "panel_index": panel.index,
        "rows_after_filter": int(len(panel)),
        "instruments_after_filter": int(panel.index.get_level_values("instrument").nunique()),
    }
    return data, meta


def build_data(
    panel_path: str | Path,
    *,
    benchmark_path: str | Path | None = None,
    columns: Iterable[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    instruments: Iterable[str] | None = None,
    extra_series: Mapping[str, pd.Series] | None = None,
    apply_filters: bool = False,
    stock_only: bool = False,
    instrument_regex: str | None = None,
    zero_volume_price_na: bool = True,
    drop_limit_move: bool = False,
    exclude_st: bool = False,
    exclude_suspended: bool = False,
    min_listed_days: int | None = None,
    min_turnover_quantile: float | None = None,
    min_liquidity_quantile: float | None = None,
    liquidity_col: str = "amount",
    liquidity_fallback_col: str = "volume",
    liquidity_rolling_days: int = 20,
) -> SeriesDict:
    data, _ = build_data_bundle(
        panel_path,
        benchmark_path=benchmark_path,
        columns=columns,
        start=start,
        end=end,
        instruments=instruments,
        extra_series=extra_series,
        apply_filters=apply_filters,
        stock_only=stock_only,
        instrument_regex=instrument_regex,
        zero_volume_price_na=zero_volume_price_na,
        drop_limit_move=drop_limit_move,
        exclude_st=exclude_st,
        exclude_suspended=exclude_suspended,
        min_listed_days=min_listed_days,
        min_turnover_quantile=min_turnover_quantile,
        min_liquidity_quantile=min_liquidity_quantile,
        liquidity_col=liquidity_col,
        liquidity_fallback_col=liquidity_fallback_col,
        liquidity_rolling_days=liquidity_rolling_days,
    )
    return data


def to_worldquant_frame(
    data: Mapping[str, pd.Series],
    *,
    fields: Iterable[str] | None = None,
) -> pd.DataFrame:
    use_fields = list(fields) if fields is not None else sorted(data.keys())
    parts: dict[str, pd.DataFrame] = {}
    for field in use_fields:
        if field not in data:
            continue
        series = data[field]
        wide = series.unstack(level="instrument").sort_index()
        wide.index.name = "date"
        parts[field] = wide
    if not parts:
        return pd.DataFrame()
    frame = pd.concat(parts, axis=1).sort_index(axis=1)
    return frame


def wide_frame_to_series(frame: pd.DataFrame, *, name: str | None = None) -> pd.Series:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be DataFrame")
    try:
        series = frame.stack(future_stack=True)
    except TypeError:
        series = frame.stack(dropna=False)
    series.index = series.index.set_names(list(EXPECTED_INDEX_NAMES))
    if name is not None:
        series.name = name
    return series.sort_index()


def available_fields(data: Mapping[str, pd.Series]) -> list[str]:
    return sorted(data.keys())
