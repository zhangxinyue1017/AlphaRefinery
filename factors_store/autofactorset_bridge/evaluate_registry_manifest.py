from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..data import build_data_bundle
from ..eval import prepare_backtest_inputs, run_factor_backtest_report, summarize_backtest_result
from ..registry import create_default_registry

from gp_factor_qlib.autofactorset.config import AutoFactorConfig
from gp_factor_qlib.autofactorset.eval_service import (
    _collect_plot_paths,
    _resolved_input_cache_key,
    _to_db_jsonable,
    _json_float_sanitize,
    evaluate_daily_promotion,
)
from gp_factor_qlib.autofactorset.library_db import LibraryStore, similarity_versus_library


DEFAULT_BENCHMARK_PATH = Path("/root/dmd/BaoStock/Index/sh.000001.csv")
DEFAULT_INGEST_RUNS_DIR = Path("/root/workspace/zxy_workspace/AlphaRefinery/artifacts/runs/autofactorset_ingest")


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("manifest 顶层必须是 YAML object")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise TypeError("manifest.candidates 必须是 list")
    return payload


def _manifest_eval_defaults(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    defaults = payload.get("eval_defaults")
    if not isinstance(defaults, dict):
        raise TypeError("manifest.eval_defaults 必须是 object")
    options = defaults.get("eval_options")
    if not isinstance(options, dict):
        raise TypeError("manifest.eval_defaults.eval_options 必须是 object")
    return defaults, options


def _resolve_required_columns(registry, factor_names: list[str]) -> list[str]:
    cols: set[str] = set()
    for name in factor_names:
        spec = registry.get(name)
        cols.update(str(x) for x in spec.required_fields)
    cols.update({"open", "close", "high", "low", "volume", "amount", "turnover"})
    return sorted(cols)


def _sanitize_slug(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("._-") or "candidate"


def _stable_cache_filename(factor_name: str) -> str:
    digest = hashlib.sha1(factor_name.encode("utf-8")).hexdigest()[:12]
    return f"{_sanitize_slug(factor_name)}_{digest}.parquet"


def _existing_library_names(store: LibraryStore) -> set[str]:
    return {row.factor_name for row in store.list_all()}


def _registry_expr_payload(name: str, registry) -> str:
    spec = registry.get(name)
    payload = {
        "registry_factor_name": name,
        "registry_source": spec.source,
        "expr": spec.expr,
        "required_fields": list(spec.required_fields),
        "notes": spec.notes,
    }
    return json.dumps(payload, ensure_ascii=False)


def _build_ns(defaults: dict[str, Any], options: dict[str, Any], cfg: AutoFactorConfig) -> argparse.Namespace:
    data_source = defaults.get("data_source", cfg.default_data_source)
    data_dir = defaults.get("data_dir", cfg.default_data_dir)
    if not data_source or not data_dir:
        raise ValueError("manifest.eval_defaults 中必须提供 data_source / data_dir")
    return argparse.Namespace(
        data_source=str(data_source),
        data_dir=str(data_dir),
        data_begin=options.get("data_begin"),
        data_end=options.get("data_end"),
        n_days=int(options.get("n_days", 0)),
        n_sh_stocks=int(options.get("n_sh_stocks", 0)),
        n_sz_stocks=int(options.get("n_sz_stocks", 0)),
        sampling_mode=str(options.get("sampling_mode", "latest_mcap_262")),
        sampling_seed=int(options.get("sampling_seed", 42)),
    )


def _safe_float(v: Any) -> float | None:
    try:
        out = float(v)
    except Exception:
        return None
    if pd.isna(out):
        return None
    return out


def evaluate_registry_manifest(
    *,
    manifest_path: str | Path,
    run_root: str | Path | None = None,
    benchmark_path: str | Path | None = DEFAULT_BENCHMARK_PATH,
    label_horizon: int = 1,
    insert_promoted: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    payload = _load_manifest(manifest_path)
    defaults, options = _manifest_eval_defaults(payload)

    cfg = AutoFactorConfig.from_environ()
    store = LibraryStore(cfg.library_db_path)
    store.init_schema()
    existing_names = _existing_library_names(store)

    registry = create_default_registry()
    factor_names = [str(item["factor_name"]).strip() for item in payload["candidates"]]
    if limit is not None and int(limit) > 0:
        factor_names = factor_names[: int(limit)]
    required_columns = _resolve_required_columns(registry, factor_names)

    resolved_benchmark_path = Path(benchmark_path).expanduser().resolve() if benchmark_path else None
    if resolved_benchmark_path is not None and not resolved_benchmark_path.exists():
        resolved_benchmark_path = None

    data, meta = build_data_bundle(
        defaults["data_dir"],
        benchmark_path=resolved_benchmark_path,
        columns=required_columns,
        start=options.get("data_begin"),
        end=options.get("data_end"),
    )
    prepared = prepare_backtest_inputs(data, horizon=int(label_horizon))

    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_root = Path(run_root).expanduser().resolve() if run_root else (
        DEFAULT_INGEST_RUNS_DIR / f"{ts}_{manifest_path.stem}"
    )
    run_root.mkdir(parents=True, exist_ok=True)

    ns = _build_ns(defaults, options, cfg)
    input_cache_key = _resolved_input_cache_key(ns)

    rows: list[dict[str, Any]] = []
    candidate_iter = payload["candidates"]
    if limit is not None and int(limit) > 0:
        candidate_iter = candidate_iter[: int(limit)]

    for idx, candidate in enumerate(candidate_iter, start=1):
        factor_name = str(candidate["factor_name"]).strip()
        print(f"[{idx}/{len(candidate_iter)}] {factor_name}")
        row = dict(candidate)
        row["already_in_library"] = factor_name in existing_names

        try:
            factor = registry.compute(factor_name, data)
            result = run_factor_backtest_report(
                factor.rename(factor_name),
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
            factor_series = result["tables"].get("factor")
            if not isinstance(factor_series, pd.Series):
                raise TypeError("backtest result missing factor series")

            similarity = similarity_versus_library(
                factor_series,
                store,
                input_cache_key,
                cfg.feature_cache_dir,
            )
            promoted, icir_val, promo_detail = evaluate_daily_promotion(
                metrics=result["metrics"],
                tables=result["tables"],
                similarity=similarity,
                main_library_factor_name=cfg.main_library_factor_name,
            )

            inserted = False
            library_row_id: int | None = None
            cache_path_str: str | None = None
            if insert_promoted and promoted and factor_name not in existing_names:
                cfg.feature_cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path = cfg.feature_cache_dir / _stable_cache_filename(factor_name)
                factor_series.to_frame(name=factor_name).to_parquet(cache_path)
                cache_path_str = str(cache_path.resolve())
                try:
                    library_row_id = store.insert_factor(
                        factor_name=factor_name,
                        expr_json=_registry_expr_payload(factor_name, registry),
                        report_dir=str(Path(result["report_dir"]).resolve()),
                        cache_path=cache_path_str,
                        input_cache_key=input_cache_key,
                        quick_rank_icir=icir_val,
                        metrics=_to_db_jsonable(result["metrics"]),
                    )
                    inserted = True
                    existing_names.add(factor_name)
                except sqlite3.IntegrityError:
                    inserted = False

            summary = summarize_backtest_result(result)
            row.update(
                {
                    "status": "ok",
                    "input_cache_key": input_cache_key,
                    "nonnull": int(factor.notna().sum()),
                    "report_dir": str(Path(result["report_dir"]).resolve()),
                    "plot_paths_abs": _collect_plot_paths(Path(result["report_dir"])),
                    "selection_like_rankic": _safe_float(summary.get("quick_rank_ic_mean")),
                    "selection_like_rankicir": _safe_float(summary.get("quick_rank_icir")),
                    "net_ann_return": _safe_float(summary.get("net_ann_return")),
                    "net_sharpe": _safe_float(summary.get("net_sharpe")),
                    "mean_turnover": _safe_float(summary.get("mean_turnover")),
                    "promoted": bool(promoted),
                    "promotion_quick_rank_icir": icir_val,
                    "promotion_checks": _json_float_sanitize(promo_detail),
                    "similarity_to_library": _json_float_sanitize(similarity),
                    "inserted_into_library": inserted,
                    "library_row_id": library_row_id,
                    "feature_cache_path": cache_path_str,
                }
            )
        except Exception as exc:
            row.update(
                {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "input_cache_key": input_cache_key,
                }
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    summary_cols = [
        "factor_name",
        "bucket",
        "family",
        "role",
        "source_model",
        "status",
        "already_in_library",
        "nonnull",
        "selection_like_rankic",
        "selection_like_rankicir",
        "net_ann_return",
        "net_sharpe",
        "mean_turnover",
        "promoted",
        "inserted_into_library",
        "library_row_id",
        "report_dir",
        "error",
    ]
    existing_summary_cols = [c for c in summary_cols if c in df.columns]
    summary_path = run_root / "summary.csv"
    df[existing_summary_cols].to_csv(summary_path, index=False)

    full_path = run_root / "results.json"
    full_path.write_text(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "run_root": str(run_root),
                "label_horizon": int(label_horizon),
                "input_cache_key": input_cache_key,
                "data_meta": {
                    "rows_after_filter": meta.get("rows_after_filter"),
                    "instruments_after_filter": meta.get("instruments_after_filter"),
                },
                "insert_promoted": bool(insert_promoted),
                "rows": _json_float_sanitize(rows),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    ok = int((df["status"] == "ok").sum()) if "status" in df.columns else 0
    promoted = int(df["promoted"].fillna(False).astype(bool).sum()) if "promoted" in df.columns else 0
    inserted = int(df["inserted_into_library"].fillna(False).astype(bool).sum()) if "inserted_into_library" in df.columns else 0
    errors = int((df["status"] == "error").sum()) if "status" in df.columns else 0

    return {
        "manifest_path": str(manifest_path),
        "run_root": str(run_root),
        "summary_path": str(summary_path),
        "results_path": str(full_path),
        "total": len(df),
        "ok": ok,
        "errors": errors,
        "promoted": promoted,
        "inserted": inserted,
        "input_cache_key": input_cache_key,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate factors_store registry factors from manifest and optionally insert into autofactorset library.")
    parser.add_argument("--manifest", required=True, help="YAML manifest path")
    parser.add_argument("--run-root", default=None, help="Output root for bridge run artifacts")
    parser.add_argument("--benchmark-path", default=str(DEFAULT_BENCHMARK_PATH), help="Optional benchmark csv path")
    parser.add_argument("--label-horizon", type=int, default=1, help="Forward return horizon for bridge evaluation")
    parser.add_argument("--insert-promoted", action="store_true", help="Insert promoted factors into autofactorset SQLite library")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on candidate count for smoke testing")
    args = parser.parse_args()

    result = evaluate_registry_manifest(
        manifest_path=args.manifest,
        run_root=args.run_root,
        benchmark_path=args.benchmark_path,
        label_horizon=int(args.label_horizon),
        insert_promoted=bool(args.insert_promoted),
        limit=args.limit,
    )

    print(f"[manifest] {result['manifest_path']}")
    print(f"[run_root] {result['run_root']}")
    print(f"[total] {result['total']}")
    print(f"[ok] {result['ok']}")
    print(f"[errors] {result['errors']}")
    print(f"[promoted] {result['promoted']}")
    print(f"[inserted] {result['inserted']}")
    print(f"[summary] {result['summary_path']}")
    print(f"[results] {result['results_path']}")


if __name__ == "__main__":
    main()
