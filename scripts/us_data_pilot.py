#!/usr/bin/env python3
'''Download an Alpaca US daily panel or evaluate one fixed public factor.'''

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))

import numpy as np
import pandas as pd

from factors_store.datahub.alpaca import fetch_daily_stock_bars
from factors_store.datahub.panel import file_sha256, normalize_alpaca_bars, save_panel, validate_panel


DEFAULT_SYMBOLS = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM", "JNJ", "WMT")
DEFAULT_START = "2023-01-01"
DEFAULT_END = "2025-12-31"
DEFAULT_PANEL = PROJECT_ROOT / "data" / "us" / f"alpaca_daily_{DEFAULT_START}_{DEFAULT_END}.parquet"
DEFAULT_FACTOR = "alpha101.alpha003"
FACTOR_EXPRESSION = "-1 * correlation(rank(open), rank(volume), 10)"
HISTORICAL_FACTOR_EXCLUSIONS = (
    "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06", "2023-01-09",
    "2023-01-10", "2023-01-11", "2023-01-12", "2023-01-13", "2024-06-26",
    "2024-07-12", "2024-07-15", "2024-07-16", "2024-07-17", "2024-07-18",
    "2024-07-19", "2024-07-22", "2024-12-02", "2024-12-03", "2024-12-23",
    "2024-12-24", "2024-12-27", "2024-12-30", "2024-12-31", "2025-01-02",
)
HISTORICAL_LABEL_EXCLUSIONS = (
    "2025-12-24", "2025-12-26", "2025-12-29", "2025-12-30", "2025-12-31",
)


def _load_credential_file(path: str | Path) -> None:
    '''Load only the two Alpaca variables without evaluating shell syntax.'''

    env_path = Path(path)
    if not env_path.exists():
        return
    allowed = {"ALPACA_API_KEY", "ALPACA_SECRET_KEY"}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if separator and key in allowed:
            os.environ.setdefault(key, value.strip().strip("'\""))


def _symbols(value: str) -> tuple[str, ...]:
    symbols = tuple(dict.fromkeys(item.strip().upper() for item in value.split(",") if item.strip()))
    if not symbols:
        raise argparse.ArgumentTypeError("at least one comma-separated symbol is required")
    return symbols


def _write_json(path: str | Path, payload: dict[str, object]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return output


def _finite_or_none(value: object) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def download(args: argparse.Namespace) -> int:
    _load_credential_file(args.env_file)
    raw_bars = fetch_daily_stock_bars(
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        feed=args.feed,
        adjustment=args.adjustment,
    )
    panel = normalize_alpaca_bars(raw_bars)
    coverage = validate_panel(panel, expected_symbols=args.symbols)
    if coverage["missing_symbols"]:
        raise RuntimeError(f"Alpaca response omitted requested symbols: {coverage['missing_symbols']}")

    output = Path(args.output)
    save_panel(panel, output)
    metadata_path = Path(args.metadata) if args.metadata else output.with_suffix(".metadata.json")
    metadata = {
        "provider": "alpaca",
        "frequency": "daily",
        "feed": args.feed,
        "adjustment": args.adjustment,
        "requested_start": args.start,
        "requested_end": args.end,
        "symbols": list(args.symbols),
        "row_count": coverage["row_count"],
        "observed_start": coverage["start"],
        "observed_end": coverage["end"],
        "missing_observations_by_symbol": coverage["missing_observations_by_symbol"],
        "panel_sha256": file_sha256(output),
    }
    _write_json(metadata_path, metadata)
    print(f"saved_panel={output}")
    print(f"saved_metadata={metadata_path}")
    print(f"rows={coverage['row_count']} symbols={coverage['symbol_count']} feed={args.feed}")
    return 0


def evaluate(args: argparse.Namespace) -> int:
    from factors_store.data import build_data
    from factors_store.eval import evaluate_factor, make_forward_return
    from factors_store.registry import create_default_registry

    panel_path = Path(args.panel)
    panel = pd.read_parquet(panel_path)
    coverage = validate_panel(panel)
    data = build_data(
        panel_path,
        apply_filters=False,
        infer_cap_fields=False,
    )
    registry = create_default_registry()
    factor = registry.compute(args.factor, data)
    label = make_forward_return(data["close"], horizon=args.horizon)
    result = evaluate_factor(
        factor,
        label=label,
        horizon=args.horizon,
        min_stocks=args.min_stocks,
    )
    rank_ic_series = result["rank_ic_series"]
    summary = {
        "evaluation_type": "predictive_cross_sectional_ic_not_trading_backtest",
        "factor": args.factor,
        "label": f"close[t+{args.horizon}] / close[t] - 1",
        "horizon_trading_days": int(args.horizon),
        "min_stocks_per_date": int(args.min_stocks),
        "panel_path": str(panel_path),
        "panel_sha256": file_sha256(panel_path),
        "panel_rows": coverage["row_count"],
        "panel_symbols": coverage["symbols"],
        "factor_non_null": int(factor.notna().sum()),
        "rank_ic_observations": int(rank_ic_series.notna().sum()),
        "rank_ic_mean": _finite_or_none(result["rank_ic_mean"]),
        "rank_ic_std": _finite_or_none(result["rank_ic_std"]),
        "rank_icir": _finite_or_none(result["rank_icir"]),
        "rank_icir_annualized": _finite_or_none(result["rank_icir_ann"]),
        "rank_ic_positive_rate": _finite_or_none(result["rank_ic_win_rate"]),
        "bao_stock_filters_applied": False,
        "synthetic_market_cap_inference": False,
    }
    output = Path(args.output) if args.output else panel_path.with_suffix(".evaluation.json")
    _write_json(output, summary)
    print(f"saved_evaluation={output}")
    print(
        f"factor={args.factor} rank_ic_mean={summary['rank_ic_mean']} "
        f"rank_icir={summary['rank_icir']} observations={summary['rank_ic_observations']}"
    )
    return 0


def _manual_rank_ic(factor: pd.Series, label: pd.Series, min_stocks: int) -> pd.Series:
    """Calculate daily Spearman IC without calling the framework evaluator."""

    joined = pd.concat({"factor": factor, "label": label}, axis=1).dropna()

    def one_day(frame: pd.DataFrame) -> float:
        if len(frame) < min_stocks:
            return float("nan")
        if frame["factor"].nunique() < 2 or frame["label"].nunique() < 2:
            return float("nan")
        factor_rank = frame["factor"].rank(method="average")
        label_rank = frame["label"].rank(method="average")
        return float(factor_rank.corr(label_rank, method="pearson"))

    return joined.groupby(level="datetime").apply(one_day).rename("rank_ic")


def _summarize_rank_ic(rank_ic: pd.Series) -> dict[str, object]:
    clean = rank_ic.dropna()
    mean = float(clean.mean())
    std = float(clean.std(ddof=1))
    icir = mean / std
    return {
        "observations": int(len(clean)),
        "mean": mean,
        "std_ddof_1": std,
        "icir": icir,
        "icir_annualized_sqrt_252": icir * math.sqrt(252.0),
        "positive_rate": float((clean > 0).mean()),
    }


def _independent_alpha003(data: dict[str, pd.Series]) -> pd.Series:
    """Pandas-only implementation used as an independent factor check."""

    open_rank = data["open"].unstack("instrument").rank(axis=1, method="min", pct=True)
    volume_rank = data["volume"].unstack("instrument").rank(axis=1, method="min", pct=True)
    factor = -open_rank.rolling(10, min_periods=10).corr(volume_rank)
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0.0).round(12)
    return factor.stack().rename(DEFAULT_FACTOR).sort_index()


def _excluded_dates(
    rank_ic: pd.Series,
    factor: pd.Series,
    label: pd.Series,
    all_dates: pd.Index,
    min_stocks: int,
) -> dict[str, list[str]]:
    """Classify every date lacking a valid daily Rank IC."""

    by_reason: dict[str, list[str]] = {
        "factor_constant_or_undefined": [],
        "forward_label_unavailable_or_insufficient": [],
        "other_insufficient_cross_section": [],
    }
    for date in all_dates:
        if date in rank_ic.index and pd.notna(rank_ic.loc[date]):
            continue
        daily = pd.concat(
            {
                "factor": factor.xs(date, level="datetime"),
                "label": label.xs(date, level="datetime"),
            },
            axis=1,
        ).dropna()
        date_text = pd.Timestamp(date).date().isoformat()
        if len(daily) < min_stocks:
            by_reason["forward_label_unavailable_or_insufficient"].append(date_text)
        elif daily["factor"].nunique() < 2:
            by_reason["factor_constant_or_undefined"].append(date_text)
        else:
            by_reason["other_insufficient_cross_section"].append(date_text)
    return {key: value for key, value in by_reason.items() if value}


def _git_text(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_fingerprints(paths: tuple[str, ...]) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for relative_path in paths:
        content = (PROJECT_ROOT / relative_path).read_bytes()
        fingerprints[relative_path] = hashlib.sha256(content).hexdigest()
    return fingerprints


def audit(args: argparse.Namespace) -> int:
    """Reproduce and independently verify the frozen US pilot baseline."""

    from factors_store.data import build_data
    from factors_store.eval import evaluate_factor, make_forward_return
    from factors_store.registry import create_default_registry

    panel_path = Path(args.panel)
    panel = pd.read_parquet(panel_path)
    coverage = validate_panel(panel)
    data = build_data(panel_path, apply_filters=False, infer_cap_fields=False)

    factor = create_default_registry().compute(DEFAULT_FACTOR, data).sort_index()
    framework_label = make_forward_return(data["close"], horizon=args.horizon).sort_index()
    framework_result = evaluate_factor(
        factor,
        label=framework_label,
        horizon=args.horizon,
        min_stocks=args.min_stocks,
    )
    framework_ic = framework_result["rank_ic_series"]

    independent_factor = _independent_alpha003(data)
    independent_label = (
        data["close"].unstack("instrument").shift(-args.horizon)
        / data["close"].unstack("instrument")
        - 1.0
    ).stack(future_stack=True).rename(f"fwd_ret_{args.horizon}").sort_index()
    independent_ic = _manual_rank_ic(independent_factor, independent_label, args.min_stocks)
    manual_framework_ic = _manual_rank_ic(factor, independent_label, args.min_stocks)

    all_dates = panel.index.get_level_values("datetime").unique().sort_values()
    corrected_exclusions = _excluded_dates(
        independent_ic, independent_factor, independent_label, all_dates, args.min_stocks
    )

    comparable_ic = pd.concat(
        {"framework": framework_ic, "independent": independent_ic}, axis=1
    ).dropna()
    false_legacy_dates = sorted(
        set(corrected_exclusions["factor_constant_or_undefined"])
        - set(HISTORICAL_FACTOR_EXCLUSIONS)
    )

    high_reference = panel[["open", "close", "vwap"]].max(axis=1)
    suspicious_high = panel.loc[panel["high"] / high_reference > 1.25]
    anomaly_rows = []
    for (date, instrument), row in suspicious_high.iterrows():
        anomaly_rows.append(
            {
                "date": pd.Timestamp(date).date().isoformat(),
                "instrument": str(instrument),
                **{field: float(row[field]) for field in ("open", "high", "low", "close", "volume", "vwap")},
            }
        )

    status_lines = [line for line in _git_text("status", "--porcelain").splitlines() if line]
    source_paths = (
        "factors_store/data.py",
        "factors_store/eval.py",
        "factors_store/factors/alpha101_like.py",
        "factors_store/operators.py",
        "factors_store/registry.py",
        "scripts/us_data_pilot.py",
    )
    payload = {
        "baseline": "US Pilot Baseline v1",
        "git": {
            "commit": _git_text("rev-parse", "HEAD"),
            "working_tree_dirty": bool(status_lines),
            "status": status_lines,
            "source_sha256": _source_fingerprints(source_paths),
        },
        "dataset": {
            "path": str(panel_path),
            "sha256": file_sha256(panel_path),
            "rows": coverage["row_count"],
            "trading_dates": int(len(all_dates)),
            "symbols": coverage["symbols"],
            "fields": list(panel.columns),
            "observed_start": coverage["start"],
            "observed_end": coverage["end"],
            "missing_values_by_field": {key: int(value) for key, value in panel.isna().sum().items()},
            "finite_values": bool(np.isfinite(panel.to_numpy(dtype=float)).all()),
            "positive_ohlcv": bool((panel[["open", "high", "low", "close", "volume"]] > 0).all().all()),
            "ohlc_bounds_valid": bool(
                (panel["high"] >= panel[["open", "close", "low"]].max(axis=1)).all()
                and (panel["low"] <= panel[["open", "close", "high"]].min(axis=1)).all()
            ),
            "vwap_within_daily_low_high": bool(
                ((panel["vwap"] >= panel["low"]) & (panel["vwap"] <= panel["high"])).all()
            ),
            "max_absolute_close_to_close_return": float(
                panel["close"].unstack("instrument").pct_change(fill_method=None).abs().max().max()
            ),
            "high_outliers_over_1_25x_open_close_vwap": anomaly_rows,
        },
        "settings": {
            "factor": DEFAULT_FACTOR,
            "factor_expression": FACTOR_EXPRESSION,
            "factor_information_cutoff": "end of date t; daily volume is required",
            "label": f"close[t+{args.horizon}] / close[t] - 1",
            "horizon_trading_days": int(args.horizon),
            "daily_metric": "cross-sectional Spearman Rank IC",
            "min_stocks_per_date": int(args.min_stocks),
            "icir": "mean(daily Rank IC) / sample std(daily Rank IC, ddof=1)",
            "loader_apply_filters": False,
            "loader_infer_cap_fields": False,
            "bao_stock_filters_applied": False,
            "synthetic_market_cap_inference": False,
        },
        "historical_722_result": {
            "metrics_reported_before_audit": {
                "observations": 722,
                "mean": 0.029748298448456938,
                "std_ddof_1": 0.34019805015270904,
                "icir": 0.08744405923285992,
                "icir_annualized_sqrt_252": 1.3881314061608931,
                "positive_rate": 0.5401662049861495,
            },
            "excluded_dates_by_reason": {
                "factor_constant_or_undefined": list(HISTORICAL_FACTOR_EXCLUSIONS),
                "forward_label_unavailable_or_insufficient": list(HISTORICAL_LABEL_EXCLUSIONS),
            },
            "numerically_false_valid_dates_count": int(len(false_legacy_dates)),
            "numerically_false_valid_dates": false_legacy_dates,
            "cause": (
                "pre-audit rolling correlation left tiny residuals for zero-variance windows "
                "and numerically split mathematical ties; Spearman ranking treated them as signal"
            ),
        },
        "validated_v1": {
            "framework_metrics": _summarize_rank_ic(framework_ic),
            "independent_metrics": _summarize_rank_ic(independent_ic),
            "excluded_dates_by_reason": corrected_exclusions,
            "framework_vs_independent_factor_max_abs_difference": float(
                (factor - independent_factor).abs().max()
            ),
            "framework_vs_independent_label_max_abs_difference": float(
                (framework_label - independent_label).abs().max()
            ),
            "framework_vs_independent_rank_ic_max_abs_difference": float(
                (comparable_ic["framework"] - comparable_ic["independent"]).abs().max()
            ),
            "framework_vs_manual_evaluator_rank_ic_max_abs_difference": float(
                (framework_ic - manual_framework_ic).abs().max()
            ),
            "matching_valid_rank_ic_dates": int(len(comparable_ic)),
        },
        "known_limitations": [
            "Ten-symbol engineering universe is small and survivorship-biased.",
            "Split adjustment is provider supplied and uses information about corporate actions.",
            "Factor uses full daily volume, so it is only available after the date-t session closes.",
            "NVDA 2024-06-10 SIP bar has a source high of 195.95 while open/close/VWAP are near 121.",
            "This is predictive IC evaluation, not an executable trading backtest.",
        ],
    }

    output = Path(args.output) if args.output else panel_path.with_suffix(".baseline_v1.json")
    _write_json(output, payload)
    print(f"saved_baseline={output}")
    print(
        f"historical_observations={payload['historical_722_result']['metrics_reported_before_audit']['observations']} "
        f"validated_observations={payload['validated_v1']['framework_metrics']['observations']} "
        f"validated_icir={payload['validated_v1']['framework_metrics']['icir']}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="download and validate Alpaca daily bars")
    download_parser.add_argument("--symbols", type=_symbols, default=DEFAULT_SYMBOLS)
    download_parser.add_argument("--start", default=DEFAULT_START)
    download_parser.add_argument("--end", default=DEFAULT_END)
    download_parser.add_argument("--feed", default="sip", help="Alpaca feed; SIP is never silently replaced by IEX")
    download_parser.add_argument("--adjustment", default="split")
    download_parser.add_argument("--env-file", default=str(PROJECT_ROOT / "api_keys.env"))
    download_parser.add_argument("--output", default=str(DEFAULT_PANEL))
    download_parser.add_argument("--metadata", default="")
    download_parser.set_defaults(handler=download)

    evaluate_parser = subparsers.add_parser("evaluate", help="run fixed-factor predictive Rank IC evaluation")
    evaluate_parser.add_argument("--panel", default=str(DEFAULT_PANEL))
    evaluate_parser.add_argument("--factor", default=DEFAULT_FACTOR)
    evaluate_parser.add_argument("--horizon", type=int, default=5)
    evaluate_parser.add_argument("--min-stocks", type=int, default=5)
    evaluate_parser.add_argument("--output", default="")
    evaluate_parser.set_defaults(handler=evaluate)

    audit_parser = subparsers.add_parser("audit", help="independently verify the frozen US pilot baseline")
    audit_parser.add_argument("--panel", default=str(DEFAULT_PANEL))
    audit_parser.add_argument("--horizon", type=int, default=5)
    audit_parser.add_argument("--min-stocks", type=int, default=5)
    audit_parser.add_argument("--output", default="")
    audit_parser.set_defaults(handler=audit)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
