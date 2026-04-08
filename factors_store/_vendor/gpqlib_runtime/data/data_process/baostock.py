"""BaoStock 日线 CSV -> 标准 panel 适配。

用途
----
- 扫描 ``/root/dmd/BaoStock/daily/{SH,SZ}/{code}/daily.csv``；
- 统一字段命名和股票代码格式；
- 合并成 ``MultiIndex(datetime, instrument)`` 的 ``DataFrame``；
- 提取可直接传给 ``filter_panel`` 的 ST / 停牌标记；
- 对大规模全量数据支持流式写入 parquet，避免一次性占满内存。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .filter import VALID_DUPLICATE_POLICIES, ensure_panel_index

RAW_TO_STANDARD_COLUMNS = {
    "date": "datetime",
    "preclose": "pre_close",
    "turn": "turnover",
    "tradestatus": "trade_status",
    "pctChg": "pct_chg",
    "isST": "is_st",
}
NUMERIC_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "volume",
    "amount",
    "adjustflag",
    "turnover",
    "trade_status",
    "pct_chg",
    "is_st",
)
DEFAULT_COLUMN_ORDER = [
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "volume",
    "amount",
    "vwap",
    "turnover",
    "pct_chg",
    "is_st",
    "trade_status",
    "adjustflag",
]
_BAOSTOCK_CODE_RE = re.compile(r"^(?P<market>sh|sz)\.(?P<code>\d{6})$", re.IGNORECASE)


def normalize_baostock_instrument(code: str, *, default_market: str | None = None) -> str:
    """把 BaoStock 代码统一成 ``000001.SZ`` / ``600000.SH``。"""
    text = str(code).strip()
    if not text:
        raise ValueError("empty BaoStock code")

    matched = _BAOSTOCK_CODE_RE.fullmatch(text)
    if matched is not None:
        return f"{matched.group('code')}.{matched.group('market').upper()}"

    text_upper = text.upper()
    if "." in text_upper:
        left, right = text_upper.split(".", 1)
        if len(left) == 6 and left.isdigit() and right in {"SH", "SZ"}:
            return f"{left}.{right}"

    if len(text_upper) == 6 and text_upper.isdigit() and default_market is not None:
        market = str(default_market).upper()
        if market not in {"SH", "SZ"}:
            raise ValueError(f"unsupported default_market={default_market!r}")
        return f"{text_upper}.{market}"

    raise ValueError(f"unsupported BaoStock code format: {code!r}")


def normalize_baostock_panel_columns(panel: pd.DataFrame) -> pd.DataFrame:
    """统一 BaoStock panel 列名与列顺序。"""
    df = panel.rename(columns=RAW_TO_STANDARD_COLUMNS).copy()
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    if "vwap" not in df.columns:
        if {"amount", "volume"}.issubset(df.columns):
            volume = pd.to_numeric(df["volume"], errors="coerce")
            amount = pd.to_numeric(df["amount"], errors="coerce")
            df["vwap"] = (amount / volume).where(volume > 0, np.nan)
        else:
            df["vwap"] = np.nan

    extra_columns = [
        col for col in df.columns if col not in {"datetime", "instrument", "code"} and col not in DEFAULT_COLUMN_ORDER
    ]
    for col in DEFAULT_COLUMN_ORDER:
        if col not in df.columns:
            df[col] = np.nan
    ordered_columns = DEFAULT_COLUMN_ORDER + extra_columns
    return df[ordered_columns]


def _save_panel(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(output_path)
        return
    if suffix in {".pkl", ".pickle"}:
        df.to_pickle(output_path)
        return
    if suffix == ".csv":
        df.to_csv(output_path)
        return
    raise ValueError("unsupported output suffix, use .parquet/.pkl/.csv")


def _build_empty_report(root: Path, files_total: int) -> dict[str, Any]:
    return {
        "input_root": str(root),
        "files_total": int(files_total),
        "files_loaded": 0,
        "files_skipped": 0,
        "skipped_files_preview": [],
        "rows_after_merge": 0,
        "instruments_after_merge": 0,
        "dates_after_merge": 0,
        "first_date": None,
        "last_date": None,
        "markets_after_merge": {},
        "rows_by_market": {},
        "columns": list(DEFAULT_COLUMN_ORDER),
        "index_quality": {
            "duplicate_policy": "error",
            "duplicated_index_rows_before": 0,
            "duplicated_index_keys_before": 0,
            "rows_dropped_by_duplicate_policy": 0,
            "rows_aggregated_by_duplicate_policy": 0,
        },
    }


def _update_report_with_frame(
    report: dict[str, Any],
    df: pd.DataFrame,
    csv_path: Path,
) -> None:
    dates = df.index.get_level_values("datetime")
    instruments = df.index.get_level_values("instrument")
    market = csv_path.parts[-3].upper()

    report["files_loaded"] += 1
    report["rows_after_merge"] += int(len(df))
    report["markets_after_merge"][market] += 1
    report["rows_by_market"][market] += int(len(df))

    first_date = dates.min() if len(df) else None
    last_date = dates.max() if len(df) else None
    if first_date is not None:
        if report["first_date"] is None or pd.Timestamp(first_date) < pd.Timestamp(report["first_date"]):
            report["first_date"] = str(first_date)
    if last_date is not None:
        if report["last_date"] is None or pd.Timestamp(last_date) > pd.Timestamp(report["last_date"]):
            report["last_date"] = str(last_date)

    return


def _finalize_report(
    report: dict[str, Any],
    *,
    instruments_seen: set[str],
    dates_seen: set[pd.Timestamp],
) -> dict[str, Any]:
    report["instruments_after_merge"] = int(len(instruments_seen))
    report["dates_after_merge"] = int(len(dates_seen))
    report["markets_after_merge"] = {
        key: int(value) for key, value in sorted(report["markets_after_merge"].items())
    }
    report["rows_by_market"] = {
        key: int(value) for key, value in sorted(report["rows_by_market"].items())
    }
    return report


def _read_single_baostock_csv(
    csv_path: Path,
    *,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
    duplicate_policy: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    parts = csv_path.parts
    if len(parts) < 3:
        raise ValueError(f"unexpected BaoStock csv path: {csv_path}")

    market = parts[-3].upper()
    code = parts[-2]
    default_inst = normalize_baostock_instrument(code, default_market=market)

    df = pd.read_csv(csv_path)
    if df.empty:
        return pd.DataFrame(), {
            "duplicate_policy": duplicate_policy,
            "duplicated_index_rows_before": 0,
            "duplicated_index_keys_before": 0,
            "rows_dropped_by_duplicate_policy": 0,
            "rows_aggregated_by_duplicate_policy": 0,
        }

    if "date" not in df.columns and "datetime" not in df.columns:
        raise ValueError(f"{csv_path} missing date/datetime column")

    if "code" in df.columns:
        raw_code = next((str(x) for x in df["code"] if pd.notna(x) and str(x).strip()), None)
    else:
        raw_code = None
    instrument = normalize_baostock_instrument(raw_code or default_inst, default_market=market)

    df = df.rename(columns=RAW_TO_STANDARD_COLUMNS).copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df[df["datetime"].notna()].copy()
    if start_date is not None:
        df = df[df["datetime"] >= start_date]
    if end_date is not None:
        df = df[df["datetime"] <= end_date]
    if df.empty:
        return pd.DataFrame(), {
            "duplicate_policy": duplicate_policy,
            "duplicated_index_rows_before": 0,
            "duplicated_index_keys_before": 0,
            "rows_dropped_by_duplicate_policy": 0,
            "rows_aggregated_by_duplicate_policy": 0,
        }

    dt_values = pd.to_datetime(df["datetime"], errors="coerce")
    inst_values = pd.Index([instrument] * len(df), dtype="object")
    df["instrument"] = instrument
    df = normalize_baostock_panel_columns(df)
    df.index = pd.MultiIndex.from_arrays(
        [dt_values, inst_values],
        names=["datetime", "instrument"],
    )
    df, index_report = ensure_panel_index(df, duplicate_policy=duplicate_policy)
    return df, index_report


def load_baostock_csv_panel(
    input_root: str | Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    duplicate_policy: str = "error",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """把 BaoStock 单票 CSV 合并成标准 panel。"""
    if duplicate_policy not in VALID_DUPLICATE_POLICIES:
        raise ValueError(f"invalid duplicate_policy={duplicate_policy!r}")

    root = Path(input_root)
    if not root.exists():
        raise FileNotFoundError(f"input_root does not exist: {root}")

    start_ts = pd.Timestamp(start_date) if start_date is not None else None
    end_ts = pd.Timestamp(end_date) if end_date is not None else None

    csv_paths = sorted(root.glob("*/*/daily.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"no BaoStock daily csv found under: {root}")

    frames: list[pd.DataFrame] = []
    skipped_files: list[dict[str, str]] = []
    instruments_seen: set[str] = set()
    dates_seen: set[pd.Timestamp] = set()
    report = _build_empty_report(root, len(csv_paths))
    report["index_quality"]["duplicate_policy"] = duplicate_policy
    report["markets_after_merge"] = defaultdict(int)
    report["rows_by_market"] = defaultdict(int)

    duplicate_rows = 0
    duplicate_keys = 0
    duplicate_dropped = 0
    duplicate_aggregated = 0

    for csv_path in csv_paths:
        try:
            df, index_report = _read_single_baostock_csv(
                csv_path,
                start_date=start_ts,
                end_date=end_ts,
                duplicate_policy=duplicate_policy,
            )
        except Exception as exc:
            skipped_files.append({"path": str(csv_path), "reason": str(exc)})
            continue
        if df.empty:
            continue
        frames.append(df)
        instruments_seen.update(df.index.get_level_values("instrument").astype(str).unique())
        dates_seen.update(pd.to_datetime(df.index.get_level_values("datetime")).unique())
        _update_report_with_frame(report, df, csv_path)
        duplicate_rows += int(index_report["duplicated_index_rows_before"])
        duplicate_keys += int(index_report["duplicated_index_keys_before"])
        duplicate_dropped += int(index_report["rows_dropped_by_duplicate_policy"])
        duplicate_aggregated += int(index_report["rows_aggregated_by_duplicate_policy"])

    if not frames:
        raise ValueError("all BaoStock csv files were empty or failed to parse")

    panel = pd.concat(frames, axis=0, copy=False).sort_index()
    report["files_skipped"] = len(skipped_files)
    report["skipped_files_preview"] = skipped_files[:20]
    report["index_quality"] = {
        "duplicate_policy": duplicate_policy,
        "duplicated_index_rows_before": duplicate_rows,
        "duplicated_index_keys_before": duplicate_keys,
        "rows_dropped_by_duplicate_policy": duplicate_dropped,
        "rows_aggregated_by_duplicate_policy": duplicate_aggregated,
    }
    report = _finalize_report(report, instruments_seen=instruments_seen, dates_seen=dates_seen)
    return panel, report


def write_baostock_csv_panel(
    input_root: str | Path,
    output_path: str | Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    duplicate_policy: str = "error",
    compression: str = "snappy",
) -> dict[str, Any]:
    """流式合并 BaoStock 单票 CSV 为单个 parquet。"""
    if duplicate_policy not in VALID_DUPLICATE_POLICIES:
        raise ValueError(f"invalid duplicate_policy={duplicate_policy!r}")

    root = Path(input_root)
    if not root.exists():
        raise FileNotFoundError(f"input_root does not exist: {root}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output.with_name(output.name + ".tmp")
    if temp_output.exists():
        temp_output.unlink()

    start_ts = pd.Timestamp(start_date) if start_date is not None else None
    end_ts = pd.Timestamp(end_date) if end_date is not None else None
    csv_paths = sorted(root.glob("*/*/daily.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"no BaoStock daily csv found under: {root}")

    report = _build_empty_report(root, len(csv_paths))
    report["index_quality"]["duplicate_policy"] = duplicate_policy
    report["markets_after_merge"] = defaultdict(int)
    report["rows_by_market"] = defaultdict(int)
    skipped_files: list[dict[str, str]] = []
    instruments_seen: set[str] = set()
    dates_seen: set[pd.Timestamp] = set()

    duplicate_rows = 0
    duplicate_keys = 0
    duplicate_dropped = 0
    duplicate_aggregated = 0

    writer: pq.ParquetWriter | None = None
    try:
        for idx, csv_path in enumerate(csv_paths, start=1):
            try:
                df, index_report = _read_single_baostock_csv(
                    csv_path,
                    start_date=start_ts,
                    end_date=end_ts,
                    duplicate_policy=duplicate_policy,
                )
            except Exception as exc:
                skipped_files.append({"path": str(csv_path), "reason": str(exc)})
                continue

            if df.empty:
                continue

            duplicate_rows += int(index_report["duplicated_index_rows_before"])
            duplicate_keys += int(index_report["duplicated_index_keys_before"])
            duplicate_dropped += int(index_report["rows_dropped_by_duplicate_policy"])
            duplicate_aggregated += int(index_report["rows_aggregated_by_duplicate_policy"])

            instruments_seen.update(df.index.get_level_values("instrument").astype(str).unique())
            dates_seen.update(pd.to_datetime(df.index.get_level_values("datetime")).unique())
            _update_report_with_frame(report, df, csv_path)

            table = pa.Table.from_pandas(df, preserve_index=True)
            if writer is None:
                writer = pq.ParquetWriter(
                    where=temp_output,
                    schema=table.schema,
                    compression=compression,
                )
            writer.write_table(table)

            if idx % 500 == 0:
                print(
                    f"[baostock] processed files={idx}/{len(csv_paths)} "
                    f"rows={report['rows_after_merge']} instruments={len(instruments_seen)}",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise ValueError("all BaoStock csv files were empty or failed to parse")

    output.unlink(missing_ok=True)
    temp_output.replace(output)

    report["files_skipped"] = len(skipped_files)
    report["skipped_files_preview"] = skipped_files[:20]
    report["index_quality"] = {
        "duplicate_policy": duplicate_policy,
        "duplicated_index_rows_before": duplicate_rows,
        "duplicated_index_keys_before": duplicate_keys,
        "rows_dropped_by_duplicate_policy": duplicate_dropped,
        "rows_aggregated_by_duplicate_policy": duplicate_aggregated,
    }
    return _finalize_report(report, instruments_seen=instruments_seen, dates_seen=dates_seen)


def extract_baostock_filter_flags(panel: pd.DataFrame) -> dict[str, pd.Series]:
    """从 BaoStock 面板提取 ``filter_panel`` 可直接使用的标记。"""
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be pandas DataFrame")
    if not isinstance(panel.index, pd.MultiIndex) or tuple(panel.index.names) != ("datetime", "instrument"):
        raise ValueError("panel must use MultiIndex(datetime, instrument)")

    if "is_st" in panel.columns:
        is_st = pd.to_numeric(panel["is_st"], errors="coerce").fillna(0.0).astype(bool)
    elif "isST" in panel.columns:
        is_st = pd.to_numeric(panel["isST"], errors="coerce").fillna(0.0).astype(bool)
    else:
        is_st = pd.Series(False, index=panel.index, name="st_flags")

    if "trade_status" in panel.columns:
        trade_status = pd.to_numeric(panel["trade_status"], errors="coerce")
        suspended = trade_status.notna() & (trade_status != 1)
    elif "tradestatus" in panel.columns:
        trade_status = pd.to_numeric(panel["tradestatus"], errors="coerce")
        suspended = trade_status.notna() & (trade_status != 1)
    elif "volume" in panel.columns:
        volume = pd.to_numeric(panel["volume"], errors="coerce")
        suspended = volume.notna() & (volume <= 0)
    else:
        suspended = pd.Series(False, index=panel.index)

    return {
        "st_flags": is_st.rename("st_flags"),
        "suspended_flags": suspended.fillna(False).astype(bool).rename("suspended_flags"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert BaoStock daily csv files into a panel DataFrame.")
    parser.add_argument("--input-root", type=str, required=True, help="BaoStock daily root")
    parser.add_argument("--output", type=str, required=True, help="Output path: .parquet/.pkl/.csv")
    parser.add_argument("--start-date", type=str, default=None, help="Optional inclusive start date")
    parser.add_argument("--end-date", type=str, default=None, help="Optional inclusive end date")
    parser.add_argument(
        "--duplicate-policy",
        type=str,
        default="error",
        choices=VALID_DUPLICATE_POLICIES,
        help="Duplicate (datetime, instrument) handling policy",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Optional JSON report path, defaults to <output>.baostock_report.json",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.suffix.lower() in {".parquet", ".pq"}:
        report = write_baostock_csv_panel(
            args.input_root,
            output_path,
            start_date=args.start_date,
            end_date=args.end_date,
            duplicate_policy=args.duplicate_policy,
        )
        rows = int(report["rows_after_merge"])
        instruments = int(report["instruments_after_merge"])
    else:
        panel, report = load_baostock_csv_panel(
            args.input_root,
            start_date=args.start_date,
            end_date=args.end_date,
            duplicate_policy=args.duplicate_policy,
        )
        _save_panel(panel, output_path)
        rows = len(panel)
        instruments = panel.index.get_level_values("instrument").nunique()

    report_path = (
        Path(args.report_path)
        if args.report_path is not None
        else output_path.with_name(output_path.name + ".baostock_report.json")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] BaoStock panel: {output_path}")
    print(f"[done] rows={rows} instruments={instruments}")
    print(f"[done] report: {report_path}")


if __name__ == "__main__":
    main()
