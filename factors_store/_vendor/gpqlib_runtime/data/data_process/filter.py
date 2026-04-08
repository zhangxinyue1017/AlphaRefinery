"""股票池与样本过滤。

用途
----
- 把“可交易样本过滤”从因子计算/回测逻辑里独立出来；
- 统一处理股票池、停牌、新股期、流动性不足等过滤；
- 保留一份简洁实现，优先覆盖你当前已有真实数据可做的事情。

当前默认做的事情
----------------
1. 校验并规范化 `MultiIndex(datetime, instrument)`
2. 仅保留 A 股股票代码（可关闭）
3. 基础脏值处理：
   - `inf/-inf -> NaN`
   - 价格 `<= 0 -> NaN`
   - `volume/amount < 0 -> NaN`
   - `volume <= 0` 时价格列置空
4. 可选过滤：
   - ST：需要外部 `st_flags`
   - 停牌：可传显式 `suspended_flags`，也可在无字段时用 `volume <= 0` 代理推断
   - 新股：可传 `listed_days`，也可按样本首个出现日自动推导“已上市交易天数”
   - 流动性：可传外部 `liquidity`，也可自动用滚动成交额/成交量代理
   - 近似涨跌停一字板过滤

说明
----
- ST 在你当前面板里没有字段，因此这里只保留接口，不做伪造推断。
- 停牌/流动性在无显式字段时，走的是“代理口径”，报告里会写清来源。
- 本文件既可被 import，也可直接命令行执行。

最小用法
--------
```python
from factors_store._vendor.gpqlib_runtime.data.data_process import filter_panel

clean_panel, report = filter_panel(
    panel,
    stock_only=True,
    zero_volume_price_na=True,
    exclude_suspended=True,
    min_listed_days=60,
    min_liquidity_quantile=0.05,
)
```

命令行用法
----------
```bash
python -m factors_store._vendor.gpqlib_runtime.data.data_process.filter \
  --input /path/to/panel.parquet \
  --output /path/to/panel_clean.parquet \
  --stock-only \
  --zero-volume-price-na \
  --exclude-suspended \
  --min-listed-days 60 \
  --min-liquidity-quantile 0.05
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VALID_DUPLICATE_POLICIES = ("error", "first", "last", "mean")
DEFAULT_STOCK_REGEX = (
    r"^(?:(?:000|001|002|003|300|301)\d{3}\.SZ|"
    r"(?:600|601|603|605|688)\d{3}\.SH)$"
)
PRICE_COLUMNS = ("open", "high", "low", "close", "vwap", "preClose", "pre_close")
VALUE_COLUMNS = ("volume", "amount")


def _parse_limit_thresholds(raw: str) -> tuple[float, ...]:
    vals: list[float] = []
    for token in [x.strip() for x in raw.split(",") if x.strip()]:
        value = float(token)
        if not (0.0 < value < 1.0):
            raise ValueError(f"invalid threshold: {token}, should be in (0,1)")
        vals.append(value)
    if not vals:
        raise ValueError("limit move thresholds cannot be empty")
    return tuple(sorted(set(vals)))


def _load_panel(path: Path, duplicate_policy: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif suffix in {".pkl", ".pickle"}:
        df = pd.read_pickle(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("unsupported input suffix, use .parquet/.pkl/.csv")

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"loaded object must be DataFrame, got: {type(df)}")

    if not isinstance(df.index, pd.MultiIndex):
        if {"datetime", "instrument"}.issubset(df.columns):
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df = df.set_index(["datetime", "instrument"])
        else:
            raise ValueError("cannot build MultiIndex(datetime, instrument) from input")
    return ensure_panel_index(df, duplicate_policy=duplicate_policy)


def _save_panel(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(path)
    elif suffix in {".pkl", ".pickle"}:
        df.to_pickle(path)
    elif suffix == ".csv":
        df.to_csv(path)
    else:
        raise ValueError("unsupported output suffix, use .parquet/.pkl/.csv")


def ensure_panel_index(
    panel: pd.DataFrame,
    duplicate_policy: str = "error",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """规范化面板索引到 MultiIndex(datetime, instrument)。"""
    if duplicate_policy not in VALID_DUPLICATE_POLICIES:
        raise ValueError(f"invalid duplicate_policy={duplicate_policy!r}")
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be pandas DataFrame")
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels != 2:
        raise ValueError("panel index must be MultiIndex(datetime, instrument)")

    dt = pd.to_datetime(panel.index.get_level_values(0), errors="coerce")
    inst = panel.index.get_level_values(1).astype(str)
    if dt.isna().any():
        raise ValueError("panel datetime index contains NaT")

    out = panel.copy()
    out.index = pd.MultiIndex.from_arrays([dt, inst], names=["datetime", "instrument"])
    dup_mask = out.index.duplicated(keep=False)
    dup_rows = int(dup_mask.sum())
    dup_keys = int(out.index[dup_mask].nunique()) if dup_rows else 0

    rows_dropped = 0
    rows_aggregated = 0
    if dup_rows:
        if duplicate_policy == "error":
            raise ValueError(
                f"duplicated (datetime, instrument) rows found: rows={dup_rows}, keys={dup_keys}"
            )
        if duplicate_policy == "first":
            rows_dropped = int(out.index.duplicated(keep="first").sum())
            out = out[~out.index.duplicated(keep="first")]
        elif duplicate_policy == "last":
            rows_dropped = int(out.index.duplicated(keep="last").sum())
            out = out[~out.index.duplicated(keep="last")]
        else:
            numeric_cols = list(out.select_dtypes(include=[np.number]).columns)
            agg_spec = {col: ("mean" if col in numeric_cols else "first") for col in out.columns}
            before = len(out)
            out = out.groupby(level=["datetime", "instrument"], sort=False).agg(agg_spec)
            out = out.reindex(columns=panel.columns)
            rows_aggregated = before - len(out)

    out = out.sort_index()
    return out, {
        "duplicate_policy": duplicate_policy,
        "duplicated_index_rows_before": dup_rows,
        "duplicated_index_keys_before": dup_keys,
        "rows_dropped_by_duplicate_policy": rows_dropped,
        "rows_aggregated_by_duplicate_policy": rows_aggregated,
    }


def _align_aux_series(values: pd.Series | None, index: pd.MultiIndex) -> pd.Series | None:
    if values is None:
        return None
    if not isinstance(values, pd.Series):
        raise TypeError("auxiliary filters must be pandas Series")
    if isinstance(values.index, pd.MultiIndex):
        aligned = values.reindex(index)
        aligned.index = index
        return aligned
    inst = index.get_level_values("instrument")
    mapped = values.reindex(inst)
    return pd.Series(mapped.to_numpy(), index=index, name=values.name)


def _resolve_price_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in PRICE_COLUMNS if col in df.columns]


def _resolve_preclose_column(df: pd.DataFrame) -> str | None:
    for col in ("preClose", "pre_close"):
        if col in df.columns:
            return col
    return None


def build_listed_days(
    panel: pd.DataFrame,
    *,
    valid_price_col: str = "close",
) -> pd.Series:
    """按样本首个有效出现日，构造每条记录对应的“已上市交易天数”。

    这是交易日口径，不是自然日口径：
    - 首日为 1
    - 次日为 2
    - ...
    """
    if not isinstance(panel.index, pd.MultiIndex):
        raise ValueError("panel must use MultiIndex(datetime, instrument)")
    df = panel.sort_index()
    if valid_price_col in df.columns:
        valid = pd.to_numeric(df[valid_price_col], errors="coerce").notna()
    else:
        valid = pd.Series(True, index=df.index)
    base = valid.groupby(level="instrument").cumsum()
    base = base.where(valid)
    return base.rename("listed_days").astype(float)


def infer_suspended_flags(
    panel: pd.DataFrame,
    *,
    volume_col: str = "volume",
    amount_col: str = "amount",
) -> pd.Series:
    """在无显式停牌字段时，用成交量/成交额构造停牌代理标记。

    当前代理规则：
    - `volume <= 0` 视为停牌代理
    - 若无 `volume`，则退化到 `amount <= 0`
    """
    if not isinstance(panel.index, pd.MultiIndex):
        raise ValueError("panel must use MultiIndex(datetime, instrument)")

    if volume_col in panel.columns:
        volume = pd.to_numeric(panel[volume_col], errors="coerce")
        return (volume.notna() & (volume <= 0)).rename("suspended_flag")
    if amount_col in panel.columns:
        amount = pd.to_numeric(panel[amount_col], errors="coerce")
        return (amount.notna() & (amount <= 0)).rename("suspended_flag")
    return pd.Series(False, index=panel.index, name="suspended_flag")


def build_liquidity_measure(
    panel: pd.DataFrame,
    *,
    liquidity_col: str = "amount",
    fallback_col: str = "volume",
    rolling_days: int = 20,
    min_periods: int = 1,
) -> pd.Series:
    """构造流动性代理，默认使用 20 日滚动平均成交额。

    优先级：
    1. `liquidity_col`，默认 `amount`
    2. `fallback_col`，默认 `volume`
    """
    if not isinstance(panel.index, pd.MultiIndex):
        raise ValueError("panel must use MultiIndex(datetime, instrument)")
    if rolling_days <= 0:
        raise ValueError("rolling_days must be positive")

    chosen_col: str | None = None
    if liquidity_col in panel.columns:
        chosen_col = liquidity_col
    elif fallback_col in panel.columns:
        chosen_col = fallback_col
    if chosen_col is None:
        raise KeyError(f"panel must contain either {liquidity_col!r} or {fallback_col!r}")

    values = pd.to_numeric(panel[chosen_col], errors="coerce").sort_index()
    rolled = (
        values.groupby(level="instrument")
        .rolling(window=rolling_days, min_periods=min_periods)
        .mean()
        .droplevel(0)
        .reindex(values.index)
    )
    return rolled.rename(f"liquidity_{chosen_col}_{rolling_days}d")


def sanitize_basic_fields(
    panel: pd.DataFrame,
    *,
    zero_volume_price_na: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """做基础脏值处理，不处理截面去极值/标准化。"""
    df = panel.copy()
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    inf_counts: dict[str, int] = {}
    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float, copy=False)
        count = int(np.isinf(values).sum())
        if count:
            inf_counts[col] = count
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    price_cols = _resolve_price_columns(df)
    nonpositive_price_counts: dict[str, int] = {}
    for col in price_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        mask = values.notna() & (values <= 0)
        count = int(mask.sum())
        if count:
            df.loc[mask, col] = np.nan
            nonpositive_price_counts[col] = count

    negative_value_counts: dict[str, int] = {}
    for col in VALUE_COLUMNS:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        mask = values.notna() & (values < 0)
        count = int(mask.sum())
        if count:
            df.loc[mask, col] = np.nan
            negative_value_counts[col] = count

    zero_volume_rows = 0
    if zero_volume_price_na and "volume" in df.columns and price_cols:
        volume = pd.to_numeric(df["volume"], errors="coerce")
        mask = volume.notna() & (volume <= 0)
        zero_volume_rows = int(mask.sum())
        if zero_volume_rows:
            df.loc[mask, price_cols] = np.nan

    return df, {
        "inf_to_nan_counts": inf_counts,
        "nonpositive_price_to_nan_counts": nonpositive_price_counts,
        "negative_volume_amount_to_nan_counts": negative_value_counts,
        "zero_volume_price_na_enabled": bool(zero_volume_price_na),
        "rows_zero_volume_or_less": zero_volume_rows,
    }


def _build_limit_move_mask(
    df: pd.DataFrame,
    *,
    thresholds: tuple[float, ...],
    eps: float,
) -> tuple[pd.Series, list[str]]:
    required = ["open", "high", "low", "close"]
    preclose_col = _resolve_preclose_column(df)
    missing = [col for col in required if col not in df.columns]
    if preclose_col is None:
        missing.append("preClose")
    if missing:
        return pd.Series(False, index=df.index), missing

    open_ = pd.to_numeric(df["open"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    pre_close = pd.to_numeric(df[preclose_col], errors="coerce")

    one_price = (
        np.isclose(open_, high, rtol=0.0, atol=1e-12)
        & np.isclose(high, low, rtol=0.0, atol=1e-12)
        & np.isclose(low, close, rtol=0.0, atol=1e-12)
    )
    abs_ret = ((close / pre_close) - 1.0).abs()
    near_limit = pd.Series(False, index=df.index)
    for threshold in thresholds:
        near_limit |= (abs_ret - threshold).abs() <= eps

    mask = one_price & (pre_close > 0) & near_limit
    if "volume" in df.columns:
        volume = pd.to_numeric(df["volume"], errors="coerce")
        mask &= volume > 0
    return mask.fillna(False), []


def filter_panel(
    panel: pd.DataFrame,
    *,
    stock_only: bool = True,
    instrument_regex: str | None = None,
    duplicate_policy: str = "error",
    zero_volume_price_na: bool = True,
    drop_limit_move: bool = False,
    limit_move_thresholds: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30),
    limit_move_eps: float = 0.002,
    exclude_st: bool = False,
    st_flags: pd.Series | None = None,
    exclude_suspended: bool = False,
    suspended_flags: pd.Series | None = None,
    min_listed_days: int | None = None,
    listed_days: pd.Series | None = None,
    min_turnover_quantile: float | None = None,
    turnover: pd.Series | None = None,
    min_liquidity_quantile: float | None = None,
    liquidity: pd.Series | None = None,
    liquidity_col: str = "amount",
    liquidity_fallback_col: str = "volume",
    liquidity_rolling_days: int = 20,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """样本过滤主入口。

    当前已落地:
    - 股票代码过滤
    - 基础脏值处理
    - 一字板近似过滤

    已预留但默认可空:
    - ST 过滤
    - 停牌过滤
    - 上市天数过滤
    - 流动性过滤
    """
    df, index_info = ensure_panel_index(panel, duplicate_policy=duplicate_policy)
    raw_df = df.copy()
    report: dict[str, Any] = {
        "rows_before": int(len(df)),
        "cols_before": int(df.shape[1]),
        "index_quality": index_info,
    }

    regex = instrument_regex or (DEFAULT_STOCK_REGEX if stock_only else None)
    if regex:
        inst = df.index.get_level_values("instrument").astype(str)
        keep = inst.str.match(regex, na=False)
        report["instrument_filter_regex"] = regex
        report["rows_dropped_by_instrument_filter"] = int((~keep).sum())
        df = df[keep]
    else:
        report["instrument_filter_regex"] = None
        report["rows_dropped_by_instrument_filter"] = 0

    df, sanitize_report = sanitize_basic_fields(df, zero_volume_price_na=zero_volume_price_na)
    report.update(sanitize_report)

    st_aligned = _align_aux_series(st_flags, df.index)
    report["st_filter_enabled"] = bool(exclude_st)
    report["st_filter_available"] = st_aligned is not None
    report["rows_dropped_by_st_filter"] = 0
    if exclude_st and st_aligned is not None:
        mask = st_aligned.fillna(False).astype(bool)
        report["rows_dropped_by_st_filter"] = int(mask.sum())
        df = df[~mask]

    inferred_suspended = None
    suspended_source = "none"
    if exclude_suspended and suspended_flags is None:
        inferred_suspended = infer_suspended_flags(raw_df)
        suspended_source = "inferred_from_volume_or_amount"
    elif suspended_flags is not None:
        suspended_source = "provided"

    suspended_aligned = _align_aux_series(
        suspended_flags if suspended_flags is not None else inferred_suspended,
        df.index,
    )
    report["suspended_filter_enabled"] = bool(exclude_suspended)
    report["suspended_filter_available"] = suspended_aligned is not None
    report["suspended_filter_source"] = suspended_source
    report["rows_dropped_by_suspended_filter"] = 0
    if exclude_suspended and suspended_aligned is not None:
        mask = suspended_aligned.fillna(False).astype(bool)
        report["rows_dropped_by_suspended_filter"] = int(mask.sum())
        df = df[~mask]

    inferred_listed_days = None
    listed_days_source = "none"
    if min_listed_days is not None and listed_days is None:
        inferred_listed_days = build_listed_days(raw_df)
        listed_days_source = "inferred_from_first_valid_trade_date"
    elif listed_days is not None:
        listed_days_source = "provided"

    listed_days_aligned = _align_aux_series(
        listed_days if listed_days is not None else inferred_listed_days,
        df.index,
    )
    report["min_listed_days"] = min_listed_days
    report["listed_days_filter_available"] = listed_days_aligned is not None
    report["listed_days_filter_source"] = listed_days_source
    report["rows_dropped_by_listed_days_filter"] = 0
    report["rows_missing_listed_days"] = 0
    if min_listed_days is not None and min_listed_days > 0 and listed_days_aligned is not None:
        report["rows_missing_listed_days"] = int(listed_days_aligned.isna().sum())
        mask = listed_days_aligned.notna() & (listed_days_aligned < min_listed_days)
        report["rows_dropped_by_listed_days_filter"] = int(mask.sum())
        df = df[~mask]

    effective_liquidity_quantile = (
        min_liquidity_quantile if min_liquidity_quantile is not None else min_turnover_quantile
    )
    inferred_liquidity = None
    liquidity_source = "none"
    if effective_liquidity_quantile is not None and liquidity is None and turnover is None:
        inferred_liquidity = build_liquidity_measure(
            raw_df,
            liquidity_col=liquidity_col,
            fallback_col=liquidity_fallback_col,
            rolling_days=liquidity_rolling_days,
        )
        liquidity_source = f"inferred_from_{liquidity_col}_rolling_mean"
    elif liquidity is not None:
        liquidity_source = "provided"
    elif turnover is not None:
        liquidity_source = "provided_turnover_alias"

    liquidity_input = liquidity if liquidity is not None else turnover if turnover is not None else inferred_liquidity
    liquidity_aligned = _align_aux_series(liquidity_input, df.index)
    report["min_turnover_quantile"] = min_turnover_quantile
    report["min_liquidity_quantile"] = effective_liquidity_quantile
    report["liquidity_filter_available"] = liquidity_aligned is not None
    report["liquidity_filter_source"] = liquidity_source
    report["rows_dropped_by_liquidity_filter"] = 0
    report["rows_missing_liquidity"] = 0
    if effective_liquidity_quantile is not None:
        if not (0.0 < effective_liquidity_quantile < 1.0):
            raise ValueError("min_liquidity_quantile must be in (0, 1)")
        if liquidity_aligned is not None:
            report["rows_missing_liquidity"] = int(liquidity_aligned.isna().sum())
            rank_pct = liquidity_aligned.groupby(level="datetime").rank(pct=True, method="average")
            mask = rank_pct.notna() & (rank_pct < effective_liquidity_quantile)
            report["rows_dropped_by_liquidity_filter"] = int(mask.sum())
            df = df[~mask]
    report["rows_dropped_by_turnover_filter"] = report["rows_dropped_by_liquidity_filter"]
    report["rows_missing_turnover"] = report["rows_missing_liquidity"]

    if drop_limit_move:
        limit_mask, missing_cols = _build_limit_move_mask(
            df,
            thresholds=limit_move_thresholds,
            eps=limit_move_eps,
        )
        report["drop_limit_move_enabled"] = True
        report["drop_limit_move_rows"] = int(limit_mask.sum())
        report["drop_limit_move_missing_cols"] = missing_cols
        df = df[~limit_mask]
    else:
        report["drop_limit_move_enabled"] = False
        report["drop_limit_move_rows"] = 0
        report["drop_limit_move_missing_cols"] = []

    report["rows_after"] = int(len(df))
    report["cols_after"] = int(df.shape[1])
    report["instruments_after"] = int(df.index.get_level_values("instrument").nunique())
    report["first_date_after"] = str(df.index.get_level_values("datetime").min()) if len(df) else None
    report["last_date_after"] = str(df.index.get_level_values("datetime").max()) if len(df) else None
    return df.sort_index(), report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="股票池/样本过滤工具")
    parser.add_argument("--input", required=True, type=Path, help="输入面板路径（.parquet/.pkl/.csv）")
    parser.add_argument("--output", required=True, type=Path, help="输出面板路径（.parquet/.pkl/.csv）")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="报告输出路径（默认: output + .filter_report.json）",
    )

    parser.add_argument("--stock-only", action="store_true", help="仅保留严格 A 股个股代码")
    parser.add_argument("--instrument-regex", type=str, default=None, help="自定义 instrument 过滤正则")
    parser.add_argument(
        "--duplicate-policy",
        choices=list(VALID_DUPLICATE_POLICIES),
        default="error",
        help="重复 (datetime,instrument) 处理策略：error/first/last/mean",
    )
    parser.add_argument(
        "--zero-volume-price-na",
        action="store_true",
        help="当 volume<=0 时，将价格列置 NaN",
    )
    parser.add_argument(
        "--drop-limit-move",
        action="store_true",
        help="过滤近似涨跌停一字板（需 open/high/low/close/preClose）",
    )
    parser.add_argument(
        "--limit-move-thresholds",
        type=str,
        default="0.05,0.10,0.20,0.30",
        help="涨跌停阈值列表（逗号分隔），默认: 0.05,0.10,0.20,0.30",
    )
    parser.add_argument(
        "--limit-move-eps",
        type=float,
        default=0.002,
        help="涨跌停阈值容差，默认 0.002",
    )
    parser.add_argument("--exclude-suspended", action="store_true", help="过滤停牌样本；无显式字段时自动推断")
    parser.add_argument("--min-listed-days", type=int, default=None, help="过滤上市未满 N 个交易日的样本")
    parser.add_argument(
        "--min-liquidity-quantile",
        type=float,
        default=None,
        help="按日截面过滤最低流动性分位数，例如 0.05 表示去掉最低 5%",
    )
    parser.add_argument(
        "--liquidity-col",
        type=str,
        default="amount",
        help="自动推断流动性时优先使用的列，默认 amount",
    )
    parser.add_argument(
        "--liquidity-fallback-col",
        type=str,
        default="volume",
        help="自动推断流动性时的后备列，默认 volume",
    )
    parser.add_argument(
        "--liquidity-rolling-days",
        type=int,
        default=20,
        help="自动推断流动性时使用的滚动窗口天数，默认 20",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    limit_move_thresholds = _parse_limit_thresholds(args.limit_move_thresholds)

    panel, load_index_info = _load_panel(args.input, duplicate_policy=args.duplicate_policy)
    clean, report = filter_panel(
        panel,
        stock_only=args.stock_only,
        instrument_regex=args.instrument_regex,
        duplicate_policy=args.duplicate_policy,
        zero_volume_price_na=args.zero_volume_price_na,
        drop_limit_move=args.drop_limit_move,
        limit_move_thresholds=limit_move_thresholds,
        limit_move_eps=args.limit_move_eps,
        exclude_suspended=args.exclude_suspended,
        min_listed_days=args.min_listed_days,
        min_liquidity_quantile=args.min_liquidity_quantile,
        liquidity_col=args.liquidity_col,
        liquidity_fallback_col=args.liquidity_fallback_col,
        liquidity_rolling_days=args.liquidity_rolling_days,
    )
    report["load_index_quality"] = load_index_info

    _save_panel(clean, args.output)

    report_path = args.report_path or Path(str(args.output) + ".filter_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] clean panel: {clean.shape} -> {args.output}")
    print(f"[done] report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
