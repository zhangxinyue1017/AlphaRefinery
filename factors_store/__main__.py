from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import pandas as pd

from . import (
    build_data_bundle,
    create_default_registry,
    evaluate_factor,
    prepare_backtest_inputs,
    run_factor_backtest,
    run_factor_backtest_report,
    summarize_backtest_result,
)
from .data_paths import DEFAULT_PANEL_PATH


_PARALLEL_DATA: dict[str, pd.Series] | None = None
_PARALLEL_REGISTRY = None
_PARALLEL_PREPARED = None


def _parallel_compute_factor(task: tuple[str, dict[str, object]]) -> tuple[str, pd.Series, dict[str, object]]:
    factor_name, cfg = task
    if _PARALLEL_DATA is None or _PARALLEL_REGISTRY is None:
        raise RuntimeError("parallel workers are not initialized")

    factor = _PARALLEL_REGISTRY.compute(factor_name, _PARALLEL_DATA)
    row: dict[str, object] = {
        "factor": factor_name,
        "nonnull": int(factor.notna().sum()),
    }
    if bool(cfg.get("do_eval")):
        summary = evaluate_factor(
            factor.rename("factor"),
            close=_PARALLEL_DATA["close"],
            horizon=int(cfg["horizon"]),
            min_stocks=int(cfg["min_stocks"]),
        )
        for key in ["mean_ic", "std_ic", "icir", "icir_ann", "ic_positive_rate"]:
            row[key] = summary.get(key)
    if bool(cfg.get("do_backtest")):
        if _PARALLEL_PREPARED is None:
            raise RuntimeError("parallel backtest inputs are not initialized")
        result = run_factor_backtest(
            factor.rename(factor_name),
            prepared=_PARALLEL_PREPARED,
            factor_name=factor_name,
            horizon=int(cfg["horizon"]),
            min_stocks=int(cfg["min_stocks"]),
            winsorize=bool(cfg["winsorize"]),
            zscore=bool(cfg["zscore"]),
            pure_mode=str(cfg["pure_mode"]),
            rolling_window=int(cfg["rolling_window"]),
            n_groups=int(cfg["n_groups"]),
            long_group=cfg["long_group"],
            short_group=cfg["short_group"],
            cost_bps=float(cfg["cost_bps"]),
            enable_alphalens=bool(cfg["enable_alphalens"]),
            alphalens_periods=tuple(cfg["alphalens_periods"]),
            alphalens_quantiles=int(cfg["alphalens_quantiles"]),
            alphalens_max_loss=float(cfg["alphalens_max_loss"]),
        )
        row.update(summarize_backtest_result(result))
    return factor_name, factor.rename(factor_name), row


def _parse_instruments(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    return values or None


def _parse_factor_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _load_factors_file(path: str | None) -> list[str]:
    if path is None:
        return []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one or more factors from factors_store.")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--factor", help="single factor name, e.g. alpha101.alpha001")
    selector.add_argument("--factor-prefix", help="batch select by prefix, e.g. alpha101.")
    selector.add_argument("--factors", help="comma-separated factors")
    selector.add_argument("--factors-file", help="text file with one factor name per line")
    selector.add_argument("--source", help="batch select by exact source tag, e.g. quants_playbook_pressure_v1")
    selector.add_argument("--source-prefix", help="batch select by source prefix, e.g. quants_playbook_")
    parser.add_argument(
        "--panel-path",
        default=str(DEFAULT_PANEL_PATH),
        help="path to panel parquet",
    )
    parser.add_argument(
        "--benchmark-path",
        default=None,
        help="optional benchmark csv; needed for many alpha191 factors",
    )
    parser.add_argument("--start", default=None, help="start date, e.g. 2024-01-01")
    parser.add_argument("--end", default=None, help="end date, e.g. 2024-12-31")
    parser.add_argument(
        "--instruments",
        default=None,
        help="comma-separated instruments, e.g. 000001.SZ,600000.SH",
    )
    parser.add_argument("--apply-filters", action="store_true", help="apply main-framework panel filters")
    parser.add_argument("--stock-only", action="store_true", help="keep stock-like instruments only")
    parser.add_argument("--instrument-regex", default=None, help="custom instrument regex for panel filtering")
    parser.add_argument(
        "--drop-limit-move",
        action="store_true",
        help="drop near one-price limit-move bars if pre_close fields are available",
    )
    parser.add_argument("--exclude-st", action="store_true", help="exclude ST samples using BaoStock flags")
    parser.add_argument(
        "--exclude-suspended",
        action="store_true",
        help="exclude suspended samples using BaoStock trade_status/volume flags",
    )
    parser.add_argument("--min-listed-days", type=int, default=None, help="exclude stocks listed fewer than N days")
    parser.add_argument(
        "--min-turnover-quantile",
        type=float,
        default=None,
        help="drop bottom turnover quantile within each date",
    )
    parser.add_argument(
        "--min-liquidity-quantile",
        type=float,
        default=None,
        help="drop bottom liquidity quantile within each date",
    )
    parser.add_argument(
        "--liquidity-col",
        default="amount",
        help="primary liquidity column for main-framework filtering",
    )
    parser.add_argument(
        "--liquidity-fallback-col",
        default="volume",
        help="fallback liquidity column for main-framework filtering",
    )
    parser.add_argument(
        "--liquidity-rolling-days",
        type=int,
        default=20,
        help="rolling days used when inferring liquidity in main-framework filtering",
    )
    parser.add_argument("--output", default=None, help="optional parquet/csv output path")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of parallel worker processes for batch mode; 1 means serial",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="also evaluate the factor using forward return from close",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="run full single-factor evaluation framework and write report outputs (single factor only)",
    )
    parser.add_argument("--horizon", type=int, default=5, help="forward return horizon for --eval")
    parser.add_argument("--min-stocks", type=int, default=10, help="minimum valid stocks per date for backtest/eval")
    parser.add_argument("--winsorize", action="store_true", help="apply daily MAD winsorization in backtest")
    parser.add_argument("--zscore", action="store_true", help="apply daily zscore in backtest")
    parser.add_argument(
        "--pure-mode",
        default="none",
        choices=["none", "factor_only", "partial"],
        help="pure IC mode used by full backtest",
    )
    parser.add_argument("--rolling-window", type=int, default=60, help="rolling window used in full backtest")
    parser.add_argument("--n-groups", type=int, default=5, help="group count used in full backtest")
    parser.add_argument("--long-group", type=int, default=None, help="optional long group id in full backtest")
    parser.add_argument("--short-group", type=int, default=None, help="optional short group id in full backtest")
    parser.add_argument("--cost-bps", type=float, default=10.0, help="turnover cost in bps for long-short backtest")
    parser.add_argument("--report-out-dir", default="./factor_eval_output", help="output dir for full backtest report")
    parser.add_argument("--report-prefix", default=None, help="optional output prefix for full backtest report")
    parser.add_argument("--enable-alphalens", action="store_true", help="enable alphalens in full backtest")
    parser.add_argument("--alphalens-periods", default="1,5,10", help="alphalens periods for full backtest")
    parser.add_argument("--alphalens-quantiles", type=int, default=5, help="alphalens quantiles")
    parser.add_argument("--alphalens-max-loss", type=float, default=0.35, help="alphalens max loss")
    parser.add_argument("--sort-by", default=None, help="summary sort column; batch mode only")
    parser.add_argument("--ascending", action="store_true", help="sort ascending in batch summary")
    parser.add_argument("--topk", type=int, default=None, help="keep top-k rows after sorting in batch summary")
    parser.add_argument("--min-nonnull", type=int, default=None, help="minimum non-null observations for batch summary")
    parser.add_argument(
        "--min-quick-mean-ic",
        type=float,
        default=None,
        help="minimum quick_rank_ic_mean for batch backtest (falls back to legacy quick_mean_ic alias)",
    )
    parser.add_argument(
        "--min-quick-icir",
        type=float,
        default=None,
        help="minimum quick_rank_icir for batch backtest (falls back to legacy quick_icir alias)",
    )
    parser.add_argument("--min-net-ann-return", type=float, default=None, help="minimum net_ann_return for batch backtest")
    parser.add_argument(
        "--export-factor-list",
        default=None,
        help="optional text file path to export filtered factor names, one per line",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=10,
        help="print top non-null rows for single factor, or top summary rows for batch",
    )
    return parser


def _resolve_factor_names(args: argparse.Namespace, registry) -> list[str]:
    if args.factor:
        return [args.factor]
    if args.factor_prefix:
        names = [name for name in registry.names() if name.startswith(args.factor_prefix)]
        if not names:
            raise ValueError(f"no factors matched prefix {args.factor_prefix!r}")
        return names
    if args.source:
        names = registry.find_names(source=args.source)
        if not names:
            raise ValueError(f"no factors matched source {args.source!r}")
        return names
    if args.source_prefix:
        names = registry.find_names(source_prefix=args.source_prefix)
        if not names:
            raise ValueError(f"no factors matched source prefix {args.source_prefix!r}")
        return names
    names = _parse_factor_list(args.factors) + _load_factors_file(args.factors_file)
    if not names:
        raise ValueError("no factor names resolved from selector arguments")
    missing = [name for name in names if name not in registry.names()]
    if missing:
        raise KeyError(f"unknown factor(s): {missing[:10]}")
    return names


def _save_frame(frame: pd.DataFrame, output: str) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_index = not isinstance(frame.index, pd.RangeIndex)
    if output_path.suffix.lower() == ".csv":
        frame.to_csv(output_path, index=write_index)
    else:
        frame.to_parquet(output_path)
    print(f"[saved] {output_path}")


def _print_filter_report(meta: dict[str, object]) -> None:
    report = meta.get("filter_report")
    if not isinstance(report, dict) or not report:
        return
    rows_before = report.get("rows_before")
    rows_after = report.get("rows_after")
    instruments_after = report.get("instruments_after")
    print(
        "[filter] "
        f"rows {rows_before} -> {rows_after}; "
        f"instruments_after={instruments_after}; "
        f"st_drop={report.get('rows_dropped_by_st_filter', 0)}; "
        f"suspended_drop={report.get('rows_dropped_by_suspended_filter', 0)}; "
        f"listed_drop={report.get('rows_dropped_by_listed_days_filter', 0)}; "
        f"liquidity_drop={report.get('rows_dropped_by_liquidity_filter', 0)}"
    )


def _compact_batch_summary(summary_df: pd.DataFrame, *, backtest: bool) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df
    if backtest:
        preferred = [
            "factor_name",
            "nonnull",
            "quick_rank_ic_mean",
            "quick_rank_icir",
            "quick_rank_ic_win_rate",
            "quick_rank_ic_monthly_win_rate",
            "quick_rank_ic_yearly_win_rate",
            "quick_ic_mean",
            "quick_icir",
            "quick_ic_win_rate",
            "net_ann_return",
            "net_sharpe",
            "long_only_net_ann_return",
            "long_only_net_sharpe",
            "long_only_net_excess_ann_return",
            "gross_ann_return",
            "mean_turnover",
            "mean_cost_bps",
            "long_only_mean_turnover",
            "long_only_mean_cost_bps",
            "alphalens_status",
        ]
    else:
        preferred = [
            "factor",
            "nonnull",
            "mean_ic",
            "std_ic",
            "icir",
            "icir_ann",
            "ic_positive_rate",
        ]
    keep = [col for col in preferred if col in summary_df.columns]
    if keep:
        return summary_df[keep]
    return summary_df


def _apply_batch_selection(summary_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df
    out = summary_df.copy()
    if args.min_nonnull is not None and "nonnull" in out.columns:
        out = out[out["nonnull"] >= int(args.min_nonnull)]
    rank_ic_col = "quick_rank_ic_mean" if "quick_rank_ic_mean" in out.columns else "quick_mean_ic" if "quick_mean_ic" in out.columns else None
    rank_icir_col = "quick_rank_icir" if "quick_rank_icir" in out.columns else "quick_icir" if "quick_icir" in out.columns else None
    if args.min_quick_mean_ic is not None and rank_ic_col is not None:
        out = out[out[rank_ic_col] >= float(args.min_quick_mean_ic)]
    if args.min_quick_icir is not None and rank_icir_col is not None:
        out = out[out[rank_icir_col] >= float(args.min_quick_icir)]
    if args.min_net_ann_return is not None and "net_ann_return" in out.columns:
        out = out[out["net_ann_return"] >= float(args.min_net_ann_return)]

    sort_by = args.sort_by
    if not sort_by:
        if args.backtest and "quick_rank_ic_mean" in out.columns:
            sort_by = "quick_rank_ic_mean"
        elif args.backtest and "quick_mean_ic" in out.columns:
            sort_by = "quick_mean_ic"
        elif args.eval and "mean_ic" in out.columns:
            sort_by = "mean_ic"
        elif "factor_name" in out.columns:
            sort_by = "factor_name"
        elif "factor" in out.columns:
            sort_by = "factor"
    if sort_by and sort_by in out.columns:
        out = out.sort_values(by=sort_by, ascending=bool(args.ascending), na_position="last")
    if args.topk is not None and int(args.topk) > 0:
        out = out.head(int(args.topk))
    return out


def _export_factor_list(summary_df: pd.DataFrame, path: str | None) -> None:
    if path is None:
        return
    name_col = "factor_name" if "factor_name" in summary_df.columns else "factor" if "factor" in summary_df.columns else None
    if name_col is None:
        raise ValueError("cannot export factor list because no factor name column exists")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(x).strip() for x in summary_df[name_col].dropna().tolist() if str(x).strip()]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"[saved] {output_path}")


def _run_single(args: argparse.Namespace, data: dict[str, pd.Series], registry) -> int:
    factor = registry.compute(args.factor, data)

    print(f"[factor] {args.factor}")
    print(f"[nonnull] {int(factor.notna().sum())}")
    preview = factor.dropna().head(int(args.head))
    if len(preview):
        print(preview.to_string())
    else:
        print("[preview] no non-null rows")

    if args.output:
        frame = factor.to_frame(name=args.factor)
        _save_frame(frame, args.output)

    if args.eval:
        summary = evaluate_factor(
            factor.rename("factor"),
            close=data["close"],
            horizon=int(args.horizon),
            min_stocks=int(args.min_stocks),
        )
        keys = ["mean_ic", "std_ic", "icir", "icir_ann", "ic_positive_rate"]
        print("[eval]")
        for key in keys:
            value = summary.get(key)
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")

    if args.backtest:
        periods = tuple(int(x.strip()) for x in str(args.alphalens_periods).split(",") if x.strip())
        result = run_factor_backtest_report(
            factor.rename(args.factor),
            data=data,
            factor_name=args.factor,
            out_dir=args.report_out_dir,
            out_prefix=args.report_prefix,
            horizon=int(args.horizon),
            min_stocks=int(args.min_stocks),
            winsorize=bool(args.winsorize),
            zscore=bool(args.zscore),
            pure_mode=str(args.pure_mode),
            rolling_window=int(args.rolling_window),
            n_groups=int(args.n_groups),
            long_group=args.long_group,
            short_group=args.short_group,
            cost_bps=float(args.cost_bps),
            enable_alphalens=bool(args.enable_alphalens),
            alphalens_periods=periods or (1, 5, 10),
            alphalens_quantiles=int(args.alphalens_quantiles),
            alphalens_max_loss=float(args.alphalens_max_loss),
        )
        metrics = result.get("metrics", {})
        quick_ic = metrics.get("quick_ic", {}) if isinstance(metrics, dict) else {}
        group_bt = metrics.get("group_backtest", {}) if isinstance(metrics, dict) else {}
        print("[backtest]")
        if isinstance(quick_ic, dict):
            for key in ["mean_ic", "icir", "icir_ann", "ic_positive_rate"]:
                value = quick_ic.get(key)
                if isinstance(value, (int, float)):
                    print(f"{key}: {value:.6f}")
        if isinstance(group_bt, dict):
            for key in ["ls_ann_return", "ls_sharpe", "ls_max_drawdown", "mean_turnover", "mean_cost_bps"]:
                value = group_bt.get(key)
                if isinstance(value, (int, float)):
                    print(f"{key}: {value:.6f}")
        print(f"[report] {result.get('report_dir')}")
    return 0


def _run_batch(args: argparse.Namespace, factor_names: list[str], data: dict[str, pd.Series], registry) -> int:
    keep_factor_values = not bool(args.backtest)
    values: dict[str, pd.Series] = {}
    rows: list[dict[str, object]] = []
    workers = max(int(args.workers), 1)
    periods = tuple(int(x.strip()) for x in str(args.alphalens_periods).split(",") if x.strip()) or (1, 5, 10)
    worker_cfg = {
        "do_eval": bool(args.eval),
        "do_backtest": bool(args.backtest),
        "horizon": int(args.horizon),
        "min_stocks": int(args.min_stocks),
        "winsorize": bool(args.winsorize),
        "zscore": bool(args.zscore),
        "pure_mode": str(args.pure_mode),
        "rolling_window": int(args.rolling_window),
        "n_groups": int(args.n_groups),
        "long_group": args.long_group,
        "short_group": args.short_group,
        "cost_bps": float(args.cost_bps),
        "enable_alphalens": bool(args.enable_alphalens),
        "alphalens_periods": periods,
        "alphalens_quantiles": int(args.alphalens_quantiles),
        "alphalens_max_loss": float(args.alphalens_max_loss),
    }
    prepared_backtest = prepare_backtest_inputs(data, horizon=int(args.horizon)) if args.backtest else None

    if workers == 1 or len(factor_names) <= 1:
        eval_keys = ["mean_ic", "std_ic", "icir", "icir_ann", "ic_positive_rate"]
        for factor_name in factor_names:
            factor = registry.compute(factor_name, data)
            if keep_factor_values:
                values[factor_name] = factor.rename(factor_name)
            row: dict[str, object] = {
                "factor": factor_name,
                "nonnull": int(factor.notna().sum()),
            }
            if args.eval:
                summary = evaluate_factor(
                    factor.rename("factor"),
                    close=data["close"],
                    horizon=int(args.horizon),
                    min_stocks=int(args.min_stocks),
                )
                for key in eval_keys:
                    row[key] = summary.get(key)
            if args.backtest:
                result = run_factor_backtest(
                    factor.rename(factor_name),
                    prepared=prepared_backtest,
                    factor_name=factor_name,
                    horizon=int(args.horizon),
                    min_stocks=int(args.min_stocks),
                    winsorize=bool(args.winsorize),
                    zscore=bool(args.zscore),
                    pure_mode=str(args.pure_mode),
                    rolling_window=int(args.rolling_window),
                    n_groups=int(args.n_groups),
                    long_group=args.long_group,
                    short_group=args.short_group,
                    cost_bps=float(args.cost_bps),
                    enable_alphalens=bool(args.enable_alphalens),
                    alphalens_periods=periods,
                    alphalens_quantiles=int(args.alphalens_quantiles),
                    alphalens_max_loss=float(args.alphalens_max_loss),
                )
                row.update(summarize_backtest_result(result))
            rows.append(row)
    else:
        if os.name != "posix":
            raise RuntimeError("parallel batch currently requires a POSIX system")

        global _PARALLEL_DATA, _PARALLEL_REGISTRY, _PARALLEL_PREPARED
        _PARALLEL_DATA = data
        _PARALLEL_REGISTRY = registry
        _PARALLEL_PREPARED = prepared_backtest
        tasks = [(factor_name, worker_cfg) for factor_name in factor_names]
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("fork")) as executor:
            for factor_name, factor, row in executor.map(_parallel_compute_factor, tasks):
                if keep_factor_values:
                    values[factor_name] = factor
                rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df = _apply_batch_selection(summary_df, args)
    summary_df = _compact_batch_summary(summary_df, backtest=bool(args.backtest))
    print(f"[batch] {len(factor_names)} factors")
    if len(summary_df):
        print(summary_df.head(int(args.head)).to_string(index=False))
    else:
        print("[batch] no rows")

    _export_factor_list(summary_df, args.export_factor_list)

    if args.output:
        if args.backtest:
            _save_frame(summary_df, args.output)
        else:
            result_df = pd.concat(values, axis=1).sort_index()
            _save_frame(result_df, args.output)
    return 0


def main() -> int:
    args = build_parser().parse_args()
    registry = create_default_registry()
    factor_names = _resolve_factor_names(args, registry)
    instruments = _parse_instruments(args.instruments)
    data, meta = build_data_bundle(
        panel_path=args.panel_path,
        benchmark_path=args.benchmark_path,
        start=args.start,
        end=args.end,
        instruments=instruments,
        apply_filters=bool(args.apply_filters),
        stock_only=bool(args.stock_only),
        instrument_regex=args.instrument_regex,
        drop_limit_move=bool(args.drop_limit_move),
        exclude_st=bool(args.exclude_st),
        exclude_suspended=bool(args.exclude_suspended),
        min_listed_days=args.min_listed_days,
        min_turnover_quantile=args.min_turnover_quantile,
        min_liquidity_quantile=args.min_liquidity_quantile,
        liquidity_col=args.liquidity_col,
        liquidity_fallback_col=args.liquidity_fallback_col,
        liquidity_rolling_days=int(args.liquidity_rolling_days),
    )
    _print_filter_report(meta)
    if len(factor_names) == 1:
        args.factor = factor_names[0]
        return _run_single(args, data, registry)
    return _run_batch(args, factor_names, data, registry)


if __name__ == "__main__":
    raise SystemExit(main())
