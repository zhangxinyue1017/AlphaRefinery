from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from .._bootstrap import ensure_project_roots
from ..data import build_data_bundle
from ..data_paths import DEFAULT_BENCHMARK_PATH, DEFAULT_PANEL_PATH
from ..eval import prepare_backtest_inputs, run_factor_backtest_report, summarize_backtest_result
from ..registry import create_default_registry

ensure_project_roots()
root_path = str(Path("/root").resolve())
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from gp_factor_qlib.autofactorset.eval_service import _resolved_input_cache_key, _to_db_jsonable
from gp_factor_qlib.autofactorset.library_db import LibraryFactorRecord, LibraryStore


DEFAULT_REBUILD_RUNS_DIR = Path("/root/workspace/zxy_workspace/AlphaRefinery/artifacts/runs/autofactorset_rebuild")


def _sanitize_slug(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("._-") or "candidate"


def _stable_cache_filename(factor_name: str) -> str:
    return f"{_sanitize_slug(factor_name)}.parquet"


def _default_target_db_path(source_db: Path, data_begin: str | None, data_end: str | None) -> Path:
    begin = str(data_begin or "begin").replace("-", "")
    end = str(data_end or "latest").replace("-", "")
    return source_db.with_name(f"{source_db.stem}_rebuild_{begin}_{end}{source_db.suffix}")


def _default_target_cache_dir(source_db: Path, data_begin: str | None, data_end: str | None) -> Path:
    begin = str(data_begin or "begin").replace("-", "")
    end = str(data_end or "latest").replace("-", "")
    return source_db.parent / f"feature_cache_rebuild_{begin}_{end}"


def _build_input_cache_key(
    *,
    data_dir: str,
    data_begin: str | None,
    data_end: str | None,
    sampling_mode: str,
    sampling_seed: int,
) -> str:
    ns = argparse.Namespace(
        data_source="baostock_parquet",
        data_dir=str(data_dir),
        data_begin=data_begin,
        data_end=data_end,
        n_days=0,
        n_sh_stocks=0,
        n_sz_stocks=0,
        sampling_mode=str(sampling_mode),
        sampling_seed=int(sampling_seed),
    )
    return _resolved_input_cache_key(ns)


def _resolve_required_columns(registry, factor_names: list[str]) -> list[str]:
    cols: set[str] = set()
    for name in factor_names:
        spec = registry.get(name)
        cols.update(str(x) for x in spec.required_fields)
    cols.update({"open", "close", "high", "low", "volume", "amount", "turnover"})
    return sorted(cols)


def _safe_float(v: Any) -> float | None:
    try:
        out = float(v)
    except Exception:
        return None
    if pd.isna(out):
        return None
    return out


def _load_factor_name_filter(path: Path | None) -> set[str]:
    if path is None:
        return set()
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return {line for line in lines if line}


def _select_source_records(
    *,
    source_store: LibraryStore,
    factor_name_pattern: str = "",
    factor_names_file: Path | None = None,
    limit: int | None = None,
) -> list[LibraryFactorRecord]:
    wanted_names = _load_factor_name_filter(factor_names_file)
    records = source_store.list_all()
    selected: list[LibraryFactorRecord] = []
    for rec in records:
        if factor_name_pattern and factor_name_pattern not in rec.factor_name:
            continue
        if wanted_names and rec.factor_name not in wanted_names:
            continue
        selected.append(rec)
    if limit is not None and int(limit) > 0:
        selected = selected[: int(limit)]
    return selected


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def rebuild_library_from_db(
    *,
    source_db_path: str | Path,
    target_db_path: str | Path | None = None,
    target_cache_dir: str | Path | None = None,
    panel_path: str | Path = DEFAULT_PANEL_PATH,
    benchmark_path: str | Path | None = DEFAULT_BENCHMARK_PATH,
    data_begin: str | None = None,
    data_end: str | None = None,
    factor_name_pattern: str = "",
    factor_names_file: str | Path | None = None,
    label_horizon: int = 1,
    limit: int | None = None,
    replace_existing: bool = False,
    sampling_mode: str = "latest_mcap_262",
    sampling_seed: int = 42,
    run_root: str | Path | None = None,
    recompute_metrics: bool = False,
) -> dict[str, Any]:
    source_db_path = Path(source_db_path).expanduser().resolve()
    if not source_db_path.exists():
        raise FileNotFoundError(f"source db not found: {source_db_path}")

    target_db_path = (
        Path(target_db_path).expanduser().resolve()
        if target_db_path
        else _default_target_db_path(source_db_path, data_begin, data_end).resolve()
    )
    target_cache_dir = (
        Path(target_cache_dir).expanduser().resolve()
        if target_cache_dir
        else _default_target_cache_dir(source_db_path, data_begin, data_end).resolve()
    )
    target_cache_dir.mkdir(parents=True, exist_ok=True)

    panel_path = Path(panel_path).expanduser().resolve()
    resolved_benchmark_path = Path(benchmark_path).expanduser().resolve() if benchmark_path else None
    if resolved_benchmark_path is not None and not resolved_benchmark_path.exists():
        resolved_benchmark_path = None

    source_store = LibraryStore(source_db_path)
    source_store.init_schema()
    target_store = LibraryStore(target_db_path)
    target_store.init_schema()

    selected_records = _select_source_records(
        source_store=source_store,
        factor_name_pattern=factor_name_pattern,
        factor_names_file=Path(factor_names_file).expanduser().resolve() if factor_names_file else None,
        limit=limit,
    )
    registry = create_default_registry()
    registry_names = set(registry.names())

    missing_registry_rows: list[dict[str, Any]] = []
    rebuildable_records: list[LibraryFactorRecord] = []
    for rec in selected_records:
        if rec.factor_name not in registry_names:
            missing_registry_rows.append(
                {
                    "factor_name": rec.factor_name,
                    "status": "missing_in_registry",
                    "source_row_id": rec.id,
                }
            )
            continue
        rebuildable_records.append(rec)

    if not rebuildable_records:
        return {
            "source_db_path": str(source_db_path),
            "target_db_path": str(target_db_path),
            "target_cache_dir": str(target_cache_dir),
            "selected": len(selected_records),
            "rebuildable": 0,
            "ok": 0,
            "errors": len(missing_registry_rows),
            "results": missing_registry_rows,
        }

    required_columns = _resolve_required_columns(
        registry,
        [rec.factor_name for rec in rebuildable_records],
    )
    data, _meta = build_data_bundle(
        panel_path,
        benchmark_path=resolved_benchmark_path,
        columns=required_columns,
        start=data_begin,
        end=data_end,
    )
    prepared = prepare_backtest_inputs(data, horizon=int(label_horizon)) if recompute_metrics else None
    input_cache_key = _build_input_cache_key(
        data_dir=str(panel_path),
        data_begin=data_begin,
        data_end=data_end,
        sampling_mode=sampling_mode,
        sampling_seed=int(sampling_seed),
    )

    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_root = Path(run_root).expanduser().resolve() if run_root else (
        DEFAULT_REBUILD_RUNS_DIR / f"{ts}_{source_db_path.stem}_rebuild"
    )
    run_root.mkdir(parents=True, exist_ok=True)

    existing_target = {rec.factor_name: rec for rec in target_store.list_all()}
    results: list[dict[str, Any]] = []
    results.extend(missing_registry_rows)

    total = len(rebuildable_records)
    for idx, rec in enumerate(rebuildable_records, start=1):
        factor_name = rec.factor_name
        print(f"[{idx}/{total}] rebuild {factor_name}")
        result_row: dict[str, Any] = {
            "factor_name": factor_name,
            "source_row_id": rec.id,
            "status": "error",
            "target_db_path": str(target_db_path),
        }
        try:
            if factor_name in existing_target and not replace_existing:
                result_row.update(
                    {
                        "status": "skipped_existing",
                        "target_row_id": existing_target[factor_name].id,
                    }
                )
                results.append(result_row)
                continue

            factor_series = registry.compute(factor_name, data).astype(float).rename(factor_name)
            cache_path = target_cache_dir / _stable_cache_filename(factor_name)
            factor_series.to_frame(name=factor_name).to_parquet(cache_path)

            report_dir = str(Path(rec.report_dir).resolve())
            quick_rank_icir = rec.quick_rank_icir
            quick_rank_ic_mean = None
            quick_ic_mean = None
            quick_icir = None
            net_sharpe = None
            mean_turnover = None
            if recompute_metrics:
                if prepared is None:
                    raise RuntimeError("prepared backtest inputs unexpectedly missing")
                bt = run_factor_backtest_report(
                    factor_series,
                    prepared=prepared,
                    factor_name=factor_name,
                    out_dir=run_root,
                    out_prefix=f"{idx:03d}_{_sanitize_slug(factor_name)}",
                    horizon=int(label_horizon),
                    min_stocks=10,
                    winsorize=False,
                    zscore=False,
                    pure_mode="partial",
                    rolling_window=60,
                    n_groups=5,
                    long_group=None,
                    short_group=None,
                    cost_bps=10.0,
                    enable_alphalens=False,
                )
                summary = summarize_backtest_result(bt)
                stored_metrics = _to_db_jsonable(bt["metrics"])
                report_dir = str(Path(bt["report_dir"]).resolve())
                quick_rank_icir = _safe_float(summary.get("quick_rank_icir"))
                quick_rank_ic_mean = _safe_float(summary.get("quick_rank_ic_mean"))
                quick_ic_mean = _safe_float(summary.get("quick_ic_mean"))
                quick_icir = _safe_float(summary.get("quick_icir"))
                net_sharpe = _safe_float(summary.get("net_sharpe"))
                mean_turnover = _safe_float(summary.get("mean_turnover"))
            else:
                try:
                    parsed_metrics = json.loads(rec.metrics_json)
                except json.JSONDecodeError:
                    parsed_metrics = {}
                stored_metrics = parsed_metrics if isinstance(parsed_metrics, dict) else {}

            if factor_name in existing_target and replace_existing:
                target_store.delete_factors([existing_target[factor_name].id])

            target_row_id = target_store.insert_factor(
                factor_name=factor_name,
                expr_json=rec.expr_json,
                report_dir=report_dir,
                cache_path=str(cache_path.resolve()),
                input_cache_key=input_cache_key,
                quick_rank_icir=quick_rank_icir,
                metrics=stored_metrics,
            )
            result_row.update(
                {
                    "status": "ok",
                    "target_row_id": target_row_id,
                    "cache_path": str(cache_path.resolve()),
                    "report_dir": report_dir,
                    "nonnull": int(factor_series.notna().sum()),
                    "quick_rank_ic_mean": quick_rank_ic_mean,
                    "quick_rank_icir": quick_rank_icir,
                    "quick_ic_mean": quick_ic_mean,
                    "quick_icir": quick_icir,
                    "net_sharpe": net_sharpe,
                    "mean_turnover": mean_turnover,
                    "recompute_metrics": bool(recompute_metrics),
                }
            )
            existing_target[factor_name] = LibraryFactorRecord(
                id=int(target_row_id),
                factor_name=factor_name,
                expr_json=rec.expr_json,
                created_at="",
                report_dir=report_dir,
                cache_path=str(cache_path.resolve()),
                input_cache_key=input_cache_key,
                quick_rank_icir=quick_rank_icir,
                metrics_json=json.dumps(stored_metrics, ensure_ascii=False, allow_nan=True),
            )
        except Exception as exc:
            result_row["error"] = str(exc)
        results.append(result_row)

    results_df = pd.DataFrame(results)
    results_path = run_root / "rebuild_results.csv"
    results_df.to_csv(results_path, index=False)

    summary_payload = {
        "source_db_path": str(source_db_path),
        "target_db_path": str(target_db_path),
        "target_cache_dir": str(target_cache_dir),
        "panel_path": str(panel_path),
        "benchmark_path": str(resolved_benchmark_path) if resolved_benchmark_path else None,
        "data_begin": data_begin,
        "data_end": data_end,
        "input_cache_key": input_cache_key,
        "selected": len(selected_records),
        "rebuildable": len(rebuildable_records),
        "ok": int((results_df["status"] == "ok").sum()) if not results_df.empty else 0,
        "skipped_existing": int((results_df["status"] == "skipped_existing").sum()) if not results_df.empty else 0,
        "errors": int((results_df["status"] == "error").sum()) if not results_df.empty else 0,
        "missing_in_registry": int((results_df["status"] == "missing_in_registry").sum()) if not results_df.empty else 0,
        "results_path": str(results_path),
    }
    summary_path = run_root / "rebuild_summary.json"
    _write_json(summary_path, summary_payload)
    summary_payload["summary_path"] = str(summary_path)
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild cached factor parquet files from an existing library.db into a new or refreshed library."
    )
    parser.add_argument("--source-db", required=True, help="Existing library.db path")
    parser.add_argument("--target-db", default=None, help="Target rebuilt library.db path; defaults to a date-stamped sibling db")
    parser.add_argument("--target-cache-dir", default=None, help="Target feature cache dir; defaults to a date-stamped sibling dir")
    parser.add_argument("--panel-path", default=str(DEFAULT_PANEL_PATH), help="Panel parquet path")
    parser.add_argument("--benchmark-path", default=str(DEFAULT_BENCHMARK_PATH), help="Benchmark csv path")
    parser.add_argument("--data-begin", default=None, help="Rebuild window start date")
    parser.add_argument("--data-end", default=None, help="Rebuild window end date")
    parser.add_argument("--factor-pattern", default="", help="Optional factor_name substring filter")
    parser.add_argument("--factor-names-file", default=None, help="Optional newline-delimited factor names file")
    parser.add_argument("--label-horizon", type=int, default=1, help="Forward return horizon for rebuilt metrics")
    parser.add_argument("--limit", type=int, default=None, help="Optional factor count limit for smoke tests")
    parser.add_argument("--replace-existing", action="store_true", help="Replace same-name rows already present in target db")
    parser.add_argument("--sampling-mode", default="latest_mcap_262", help="Metadata-only sampling mode stored in input_cache_key")
    parser.add_argument("--sampling-seed", type=int, default=42, help="Metadata-only sampling seed stored in input_cache_key")
    parser.add_argument("--run-root", default=None, help="Artifact output dir for rebuild reports")
    parser.add_argument("--recompute-metrics", action="store_true", help="Also rerun backtest metrics; default only rebuilds caches and reuses source metrics")
    args = parser.parse_args()

    summary = rebuild_library_from_db(
        source_db_path=args.source_db,
        target_db_path=args.target_db,
        target_cache_dir=args.target_cache_dir,
        panel_path=args.panel_path,
        benchmark_path=args.benchmark_path,
        data_begin=args.data_begin,
        data_end=args.data_end,
        factor_name_pattern=str(args.factor_pattern or ""),
        factor_names_file=args.factor_names_file,
        label_horizon=int(args.label_horizon),
        limit=args.limit,
        replace_existing=bool(args.replace_existing),
        sampling_mode=str(args.sampling_mode),
        sampling_seed=int(args.sampling_seed),
        run_root=args.run_root,
        recompute_metrics=bool(args.recompute_metrics),
    )
    print(f"[selected] {summary['selected']}")
    print(f"[rebuildable] {summary['rebuildable']}")
    print(f"[ok] {summary['ok']}")
    print(f"[errors] {summary['errors']}")
    print(f"[target_db] {summary['target_db_path']}")
    print(f"[target_cache_dir] {summary['target_cache_dir']}")
    if "summary_path" in summary:
        print(f"[summary] {summary['summary_path']}")


if __name__ == "__main__":
    main()
