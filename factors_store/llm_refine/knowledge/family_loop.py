'''Family-loop orchestration helpers and state collection.

Implements anchor candidate collection, true-correlation guards, focused run dispatch, and reports.
'''

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from ...data import build_data_bundle
from ...data_paths import DEFAULT_BENCHMARK_PATH
from ...registry import create_default_registry
from ..config import (
    DEFAULT_FAMILY_LOOP_ANCHOR_MAX_TRUE_PARENT_CORR,
    DEFAULT_FAMILY_LOOP_ANCHOR_MAX_TRUE_SIBLING_CORR,
    DEFAULT_FAMILY_LOOP_RUNS_DIR,
    FAMILY_LOOP_STAGE_PRESETS,
)
from ..core.archive import utc_now_iso
from ..core.seed_loader import load_seed_pool, resolve_family_formula
from ..evaluation.redundancy import factor_series_correlations
from ..parsing.expression_engine import WideExpressionEngine
from ..search import (
    DecisionContext,
    DecisionEngine,
    EvaluationFeedback,
    FamilyDecisionState,
    FamilyState,
    RefinementAction,
    SearchPolicy,
    build_stage_transition_evidence,
    build_stage_transition_shadow,
    resolve_stage_transition_from_state,
)
from ..search.context_resolver import ContextEvidence, resolve_context_profile, resolve_orchestration_profile
from ..search.run_ingest import load_multi_run_candidate_records, resolve_materialized_child_run_dir
from ..search.scoring import pairwise_similarity, safe_float
from ..search.state import SearchNode


_FAMILY_LOOP_SERIES_CACHE_ROOT = Path(__file__).resolve().parents[3] / "artifacts" / "cache" / "family_loop_series"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _family_loop_log(message: str) -> None:
    print(message, flush=True)


def _series_cache_key(
    *,
    family: str,
    factor_name: str,
    expression: str,
    panel_path: str,
    benchmark_path: str,
    start: str,
    end: str,
) -> str:
    payload = {
        "family": str(family or ""),
        "factor_name": str(factor_name or ""),
        "expression": str(expression or ""),
        "panel_path": str(panel_path or ""),
        "benchmark_path": str(benchmark_path or ""),
        "start": str(start or ""),
        "end": str(end or ""),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _series_cache_path(
    *,
    family: str,
    factor_name: str,
    expression: str,
    panel_path: str,
    benchmark_path: str,
    start: str,
    end: str,
) -> Path:
    cache_key = _series_cache_key(
        family=family,
        factor_name=factor_name,
        expression=expression,
        panel_path=panel_path,
        benchmark_path=benchmark_path,
        start=start,
        end=end,
    )
    family_dir = _FAMILY_LOOP_SERIES_CACHE_ROOT / str(family or "unknown_family").strip()
    return family_dir / f"{cache_key}.pkl"


def _load_cached_series(path: Path) -> pd.Series | None:
    try:
        if not path.exists():
            return None
        series = pd.read_pickle(path)
        if isinstance(series, pd.Series):
            return series
    except Exception:
        return None
    return None


def _save_cached_series(path: Path, series: pd.Series) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        series.to_pickle(path)
    except Exception:
        return


def _normalize_models(values: list[str]) -> list[str]:
    out: list[str] = []
    for raw in values:
        for part in str(raw).split(","):
            item = part.strip()
            if item and item not in out:
                out.append(item)
    return out


def resolve_stage_protocol(
    *,
    stage_preset: str,
    policy_preset: str | None = None,
    n_candidates: int | None = None,
    max_rounds: int | None = None,
    stop_if_no_new_winner: int | None = None,
) -> dict[str, Any]:
    preset = dict(FAMILY_LOOP_STAGE_PRESETS.get(str(stage_preset), {}))
    if not preset:
        raise KeyError(f"unknown family loop stage preset: {stage_preset}")
    if policy_preset is not None:
        preset["policy_preset"] = str(policy_preset)
    if n_candidates is not None:
        preset["n_candidates"] = int(n_candidates)
    if max_rounds is not None:
        preset["max_rounds"] = int(max_rounds)
    if stop_if_no_new_winner is not None:
        preset["stop_if_no_new_winner"] = int(stop_if_no_new_winner)
    preset["stage_preset"] = str(stage_preset)
    return preset


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y"}


def _resolve_latest_child_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    children = sorted(p for p in root.iterdir() if p.is_dir())
    return children[-1] if children else None


def _discover_multi_run_dirs(scheduler_dir: Path) -> list[Path]:
    root = scheduler_dir / "multi_runs"
    if not root.exists():
        return []
    out: list[Path] = []
    for summary_path in sorted(root.glob("round_*/*/*/summary.json")):
        out.append(summary_path.parent)
    return out


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def _discover_evaluation_csvs(scheduler_dir: Path) -> list[Path]:
    root = scheduler_dir / "multi_runs"
    if not root.exists():
        return []
    return sorted(root.glob("round_*/*/*/child_runs/*/*/evaluation/family_backtest_summary_full.csv"))


def _load_child_candidate_maps(child_run_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    summary_rows = _load_csv_rows(child_run_dir / "evaluation" / "family_backtest_summary_full.csv")
    by_candidate_id: dict[str, dict[str, Any]] = {}
    parent_row: dict[str, Any] | None = None
    for row in summary_rows:
        candidate_id = str(row.get("candidate_id", "") or "").strip()
        if candidate_id:
            by_candidate_id[candidate_id] = row
            continue
        factor_name = str(row.get("factor_name", "") or "").strip()
        if factor_name and parent_row is None:
            parent_row = row
    return by_candidate_id, parent_row


def _has_any_eval_metric(row: dict[str, Any]) -> bool:
    for key in (
        "quick_rank_ic_mean",
        "quick_rank_icir",
        "net_ann_return",
        "net_excess_ann_return",
        "net_sharpe",
        "mean_turnover",
    ):
        value = row.get(key)
        if value not in (None, ""):
            return True
    return False


def _broad_result_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        safe_float(row.get("net_excess_ann_return"), default=float("-inf")),
        safe_float(row.get("quick_rank_icir"), default=float("-inf")),
        safe_float(row.get("net_sharpe"), default=float("-inf")),
        safe_float(row.get("quick_rank_ic_mean"), default=float("-inf")),
        -safe_float(row.get("mean_turnover"), default=float("inf")),
    )


def _snapshot_broad_results(scheduler_dir: Path) -> dict[str, Any]:
    decision_counts: dict[str, int] = {}
    total_candidates = 0
    promotable_candidates = 0
    research_drop_count = 0
    evaluation_failed_count = 0
    evaluated_candidate_count = 0
    strongest_baseline: dict[str, Any] = {}
    strongest_candidate: dict[str, Any] = {}
    strongest_evaluated: dict[str, Any] = {}

    for path in _discover_evaluation_csvs(scheduler_dir):
        for row in _load_csv_rows(path):
            role = str(row.get("role", "") or "").strip().lower()
            if role == "candidate":
                total_candidates += 1
                decision = str(row.get("decision", "") or "").strip() or "EMPTY"
                decision_counts[decision] = int(decision_counts.get(decision, 0)) + 1
                if decision in {"research_keep", "research_winner", "keep", "winner"}:
                    promotable_candidates += 1
                if decision == "research_drop":
                    research_drop_count += 1
                    reason = str(row.get("decision_reason", "") or "").strip().lower()
                    error_text = str(row.get("error", "") or "").strip()
                    if "evaluation failed" in reason or error_text:
                        evaluation_failed_count += 1
                if _has_any_eval_metric(row):
                    evaluated_candidate_count += 1
                    if not strongest_candidate or _broad_result_sort_key(row) > _broad_result_sort_key(strongest_candidate):
                        strongest_candidate = dict(row)
                    if not strongest_evaluated or _broad_result_sort_key(row) > _broad_result_sort_key(strongest_evaluated):
                        strongest_evaluated = dict(row)
            elif role in {"parent", "peer"} and _has_any_eval_metric(row):
                if not strongest_baseline or _broad_result_sort_key(row) > _broad_result_sort_key(strongest_baseline):
                    strongest_baseline = dict(row)
                if not strongest_evaluated or _broad_result_sort_key(row) > _broad_result_sort_key(strongest_evaluated):
                    strongest_evaluated = dict(row)

    return {
        "total_candidates": total_candidates,
        "promotable_candidates": promotable_candidates,
        "research_drop_count": research_drop_count,
        "evaluation_failed_count": evaluation_failed_count,
        "evaluated_candidate_count": evaluated_candidate_count,
        "decision_counts": decision_counts,
        "strongest_baseline": strongest_baseline,
        "strongest_candidate": strongest_candidate,
        "strongest_evaluated": strongest_evaluated,
    }


def _load_pending_manifest_entries(child_run_dir: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(child_run_dir / "metadata" / "pending_curated_manifest.json")
    out: dict[str, dict[str, Any]] = {}
    for entry in list(payload.get("entries") or []):
        candidate_id = str(entry.get("candidate_id", "") or "").strip()
        if candidate_id:
            out[candidate_id] = dict(entry)
    return out


def _load_auto_applied_entries(child_run_dir: Path) -> set[str]:
    payload = _read_json(child_run_dir / "metadata" / "auto_applied_promotion.json")
    if not payload:
        return set()
    if isinstance(payload, list):
        names: set[str] = set()
        for item in payload:
            if not isinstance(item, dict):
                continue
            if not _safe_bool(item.get("changed")):
                continue
            entries = item.get("entries") or []
            names.update({str(entry).strip() for entry in entries if str(entry).strip()})
        return names
    if not isinstance(payload, dict):
        return set()
    if not _safe_bool(payload.get("changed")):
        return set()
    entries = payload.get("entries") or []
    return {str(item).strip() for item in entries if str(item).strip()}


def _build_corr_data(
    *,
    seed_pool_path: str | Path,
    family_name: str,
    panel_path: str = "",
    benchmark_path: str = "",
    start: str = "",
    end: str = "",
) -> tuple[Any, dict[str, pd.Series], WideExpressionEngine]:
    seed_pool = load_seed_pool(seed_pool_path)
    family = seed_pool.get_family(family_name)
    defaults = dict(seed_pool.evaluation_defaults or {})
    effective_panel_path = str(panel_path or defaults.get("panel_path") or "").strip()
    if not effective_panel_path:
        raise ValueError("panel_path is required to compute true correlations for family loop graduation")
    effective_benchmark = str(benchmark_path or defaults.get("benchmark_path") or "").strip()
    if not effective_benchmark and DEFAULT_BENCHMARK_PATH.exists():
        effective_benchmark = str(DEFAULT_BENCHMARK_PATH)
    data, _meta = build_data_bundle(
        effective_panel_path,
        benchmark_path=effective_benchmark or None,
        start=start or defaults.get("start"),
        end=end or defaults.get("end"),
        apply_filters=bool(defaults.get("apply_filters", False)),
        stock_only=bool(defaults.get("stock_only", False)),
        exclude_st=bool(defaults.get("exclude_st", False)),
        exclude_suspended=bool(defaults.get("exclude_suspended", False)),
        min_listed_days=defaults.get("min_listed_days"),
        drop_limit_move=bool(defaults.get("drop_limit_move", False)),
        min_turnover_quantile=defaults.get("min_turnover_quantile"),
        min_liquidity_quantile=defaults.get("min_liquidity_quantile"),
    )
    return family, data, WideExpressionEngine(data)


def _compute_series_for_factor_or_expression(
    *,
    family: Any,
    factor_name: str,
    expression: str,
    data: dict[str, pd.Series],
    engine: WideExpressionEngine,
    registry: Any,
) -> pd.Series | None:
    name = str(factor_name or "").strip()
    expr = str(expression or "").strip()
    try:
        if name:
            try:
                return registry.compute(name, data)
            except KeyError:
                pass
            if name == family.canonical_seed or name in family.aliases or name in family.formulas:
                resolved = resolve_family_formula(family, name)
                if resolved:
                    return engine.evaluate_series(resolved, name=name)
        if expr:
            return engine.evaluate_series(expr, name=name or "family_loop_candidate")
    except Exception:
        return None
    return None


def _load_or_compute_series_for_factor_or_expression(
    *,
    family_name: str,
    factor_name: str,
    expression: str,
    panel_path: str,
    benchmark_path: str,
    start: str,
    end: str,
    family: Any,
    data: dict[str, pd.Series],
    engine: WideExpressionEngine,
    registry: Any,
) -> tuple[pd.Series | None, bool]:
    cache_path = _series_cache_path(
        family=family_name,
        factor_name=factor_name,
        expression=expression,
        panel_path=panel_path,
        benchmark_path=benchmark_path,
        start=start,
        end=end,
    )
    cached = _load_cached_series(cache_path)
    if cached is not None:
        return cached, True
    computed = _compute_series_for_factor_or_expression(
        family=family,
        factor_name=factor_name,
        expression=expression,
        data=data,
        engine=engine,
        registry=registry,
    )
    if computed is not None:
        _save_cached_series(cache_path, computed)
    return computed, False


def _finite_corr(value: float) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _node_from_payload(payload: dict[str, Any], *, family: str, parent_expression: str = "") -> SearchNode:
    return SearchNode(
        node_id=str(payload.get("candidate_id", "") or payload.get("factor_name", "") or payload.get("expression", "")),
        family=family,
        factor_name=str(payload.get("factor_name", "") or ""),
        expression=str(payload.get("expression", "") or ""),
        candidate_id=str(payload.get("candidate_id", "") or ""),
        parent_candidate_id=str(payload.get("parent_candidate_id", "") or ""),
        motif_signature=str(payload.get("motif_signature", "") or ""),
        status=str(payload.get("status", "") or ""),
        quick_rank_ic_mean=safe_float(payload.get("quick_rank_ic_mean"), default=0.0),
        quick_rank_icir=safe_float(payload.get("quick_rank_icir"), default=0.0),
        net_ann_return=safe_float(payload.get("net_ann_return"), default=0.0),
        net_excess_ann_return=safe_float(payload.get("net_excess_ann_return"), default=0.0),
        net_sharpe=safe_float(payload.get("net_sharpe"), default=0.0),
        mean_turnover=safe_float(payload.get("mean_turnover"), default=0.0),
        metadata={"parent_expression": parent_expression},
    )


def _material_gain(candidate: dict[str, Any], parent_metrics: dict[str, Any], *, excess_min: float, icir_min: float) -> bool:
    cand_excess = safe_float(candidate.get("net_excess_ann_return"), default=float("-inf"))
    cand_icir = safe_float(candidate.get("quick_rank_icir"), default=float("-inf"))
    parent_excess = safe_float(parent_metrics.get("net_excess_ann_return"), default=float("-inf"))
    parent_icir = safe_float(parent_metrics.get("quick_rank_icir"), default=float("-inf"))
    return (cand_excess - parent_excess) >= excess_min or (cand_icir - parent_icir) >= icir_min


def build_scheduler_cmd(
    *,
    python_executable: str,
    family: str,
    stage_mode: str,
    scheduler_runs_dir: Path,
    models: list[str],
    seed_pool: str,
    n_candidates: int,
    runs_dir: str,
    archive_db: str,
    name_prefix: str,
    provider_name: str,
    base_url: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    additional_notes: str,
    policy_preset: str,
    target_profile: str,
    panel_path: str,
    benchmark_path: str,
    start: str,
    end: str,
    max_parallel: int,
    max_rounds: int,
    stop_if_no_new_winner: int,
    skip_eval: bool,
    dry_run: bool,
    auto_apply_promotion: bool,
    disable_mmr_rerank: bool,
    current_parent_name: str = "",
    current_parent_expression: str = "",
) -> list[str]:
    cmd = [
        python_executable,
        "-m",
        "factors_store.llm_refine.cli.run_refine_multi_model_scheduler",
        "--family",
        family,
        "--stage-mode",
        str(stage_mode),
        "--scheduler-runs-dir",
        str(scheduler_runs_dir),
        "--seed-pool",
        str(seed_pool),
        "--n-candidates",
        str(int(n_candidates)),
        "--runs-dir",
        str(runs_dir),
        "--archive-db",
        str(archive_db),
        "--name-prefix",
        str(name_prefix),
        "--provider-name",
        str(provider_name),
        "--base-url",
        str(base_url),
        "--api-key",
        str(api_key),
        "--temperature",
        str(float(temperature)),
        "--max-tokens",
        str(int(max_tokens)),
        "--timeout",
        str(float(timeout)),
        "--policy-preset",
        str(policy_preset),
        "--target-profile",
        str(target_profile),
        "--max-parallel",
        str(int(max_parallel)),
        "--max-rounds",
        str(int(max_rounds)),
        "--stop-if-no-new-winner",
        str(int(stop_if_no_new_winner)),
    ]
    for model in models:
        cmd.extend(["--models", str(model)])
    if str(additional_notes).strip():
        cmd.extend(["--additional-notes", str(additional_notes)])
    if str(panel_path).strip():
        cmd.extend(["--panel-path", str(panel_path)])
    if str(benchmark_path).strip():
        cmd.extend(["--benchmark-path", str(benchmark_path)])
    if str(start).strip():
        cmd.extend(["--start", str(start)])
    if str(end).strip():
        cmd.extend(["--end", str(end)])
    if str(current_parent_name).strip():
        cmd.extend(["--current-parent-name", str(current_parent_name)])
    if str(current_parent_expression).strip():
        cmd.extend(["--current-parent-expression", str(current_parent_expression)])
    if skip_eval:
        cmd.append("--skip-eval")
    if dry_run:
        cmd.append("--dry-run")
    if auto_apply_promotion:
        cmd.append("--auto-apply-promotion")
    if disable_mmr_rerank:
        cmd.append("--disable-mmr-rerank")
    return cmd


def run_scheduler_stage(
    *,
    cmd: list[str],
    stage_root: Path,
    log_path: Path,
) -> tuple[int, Path | None]:
    stage_root.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_fp:
        proc = subprocess.run(cmd, stdout=log_fp, stderr=subprocess.STDOUT, text=True)
    return int(proc.returncode), _resolve_latest_child_dir(stage_root)


def collect_anchor_candidates(
    *,
    archive_db: str,
    scheduler_dir: Path,
    seed_pool_path: str,
    family: str,
    parent_name: str,
    parent_expression: str,
    panel_path: str = "",
    benchmark_path: str = "",
    start: str = "",
    end: str = "",
    parent_metrics: dict[str, Any] | None = None,
    policy: SearchPolicy,
    max_parent_similarity: float,
    max_true_parent_corr: float = DEFAULT_FAMILY_LOOP_ANCHOR_MAX_TRUE_PARENT_CORR,
    max_true_sibling_corr: float = DEFAULT_FAMILY_LOOP_ANCHOR_MAX_TRUE_SIBLING_CORR,
    min_material_excess_gain: float,
    min_material_icir_gain: float,
) -> dict[str, Any]:
    _family_loop_log(
        f"[family_loop] collect_anchor_candidates start family={family} scheduler_dir={scheduler_dir}"
    )
    candidates_by_id: dict[str, dict[str, Any]] = {}
    discovered_parent_metrics = dict(parent_metrics or {})
    parent_node = _node_from_payload(
        {
            "factor_name": parent_name,
            "expression": parent_expression,
            **discovered_parent_metrics,
        },
        family=family,
    )
    corr_context_error = ""
    corr_mode = "heuristic_fallback"
    true_parent_series: pd.Series | None = None
    series_by_candidate_id: dict[str, pd.Series] = {}

    for multi_run_dir in _discover_multi_run_dirs(scheduler_dir):
        rows = load_multi_run_candidate_records(
            archive_db=archive_db,
            multi_run_dir=multi_run_dir,
            family=family,
            statuses=("research_winner", "winner", "research_keep", "keep"),
        )
        inner_summary = _read_json(multi_run_dir / "summary.json")
        child_meta_rows: dict[str, dict[str, Any]] = {}
        pending_entries: dict[str, dict[str, Any]] = {}
        auto_applied_registry_names: set[str] = set()

        for completed in list(inner_summary.get("completed") or []):
            child_run_dir = resolve_materialized_child_run_dir(completed.get("child_runs_dir", ""))
            if child_run_dir is None:
                continue
            meta_rows, parent_row = _load_child_candidate_maps(child_run_dir)
            child_meta_rows.update(meta_rows)
            pending_entries.update(_load_pending_manifest_entries(child_run_dir))
            auto_applied_registry_names.update(_load_auto_applied_entries(child_run_dir))
            if not discovered_parent_metrics and parent_row is not None:
                factor_name = str(parent_row.get("factor_name", "") or "").strip()
                expression = str(parent_row.get("expression", "") or parent_row.get("expr", "") or "").strip()
                if factor_name == parent_name or expression == parent_expression:
                    discovered_parent_metrics = {
                        "quick_rank_ic_mean": parent_row.get("quick_rank_ic_mean"),
                        "quick_rank_icir": parent_row.get("quick_rank_icir"),
                        "net_ann_return": parent_row.get("net_ann_return"),
                        "net_excess_ann_return": parent_row.get("net_excess_ann_return"),
                        "net_sharpe": parent_row.get("net_sharpe"),
                        "mean_turnover": parent_row.get("mean_turnover"),
                    }
                    parent_node = _node_from_payload(
                        {
                            "factor_name": parent_name,
                            "expression": parent_expression,
                            **discovered_parent_metrics,
                        },
                        family=family,
                    )

        for row in rows:
            candidate_id = str(row.get("candidate_id", "") or "").strip()
            if not candidate_id:
                continue
            existing = candidates_by_id.setdefault(candidate_id, dict(row))
            existing.update({k: v for k, v in row.items() if v not in (None, "", [])})
            meta = child_meta_rows.get(candidate_id) or {}
            if meta:
                existing["eligible_for_best_node"] = _safe_bool(meta.get("eligible_for_best_node"))
                existing["winner_score"] = safe_float(meta.get("winner_score"), default=0.0)
                existing["metrics_completeness"] = safe_float(meta.get("metrics_completeness"), default=0.0)
                existing["missing_core_metrics_count"] = int(meta.get("missing_core_metrics_count") or 0)
                existing["decision"] = str(meta.get("decision", "") or existing.get("status", ""))
                existing["decision_reason"] = str(meta.get("decision_reason", "") or "")
            pending = pending_entries.get(candidate_id) or {}
            existing["pending_promotion"] = bool(pending)
            registry_name = str(pending.get("suggested_registry_name", "") or "")
            existing["suggested_registry_name"] = registry_name
            existing["auto_applied_promotion"] = bool(registry_name and registry_name in auto_applied_registry_names)

    if candidates_by_id:
        try:
            _family_loop_log(
                f"[family_loop] building corr context family={family} candidates={len(candidates_by_id)}"
            )
            family_cfg, data, engine = _build_corr_data(
                seed_pool_path=seed_pool_path,
                family_name=family,
                panel_path=panel_path,
                benchmark_path=benchmark_path,
                start=start,
                end=end,
            )
            registry = create_default_registry()
            cache_hits = 0
            true_parent_series, parent_cache_hit = _load_or_compute_series_for_factor_or_expression(
                family_name=family,
                factor_name=parent_name,
                expression=parent_expression,
                panel_path=panel_path,
                benchmark_path=benchmark_path,
                start=start,
                end=end,
                family=family_cfg,
                data=data,
                engine=engine,
                registry=registry,
            )
            cache_hits += int(parent_cache_hit)
            if true_parent_series is not None:
                _family_loop_log(
                    f"[family_loop] parent series ready family={family} parent={parent_name or '(expr-only)'} cache_hit={parent_cache_hit}"
                )
            computed_series_count = 0
            for candidate in candidates_by_id.values():
                candidate_id = str(candidate.get("candidate_id", "") or "").strip()
                if not candidate_id:
                    continue
                factor_series, cache_hit = _load_or_compute_series_for_factor_or_expression(
                    family_name=family,
                    factor_name=str(candidate.get("factor_name", "") or ""),
                    expression=str(candidate.get("expression", "") or ""),
                    panel_path=panel_path,
                    benchmark_path=benchmark_path,
                    start=start,
                    end=end,
                    family=family_cfg,
                    data=data,
                    engine=engine,
                    registry=registry,
                )
                if factor_series is not None:
                    series_by_candidate_id[candidate_id] = factor_series
                    computed_series_count += 1
                    cache_hits += int(cache_hit)
            _family_loop_log(
                f"[family_loop] candidate series ready family={family} computed={computed_series_count}/{len(candidates_by_id)} cache_hits={cache_hits}"
            )
            corr_mode = "true_correlation"
        except Exception as exc:
            corr_context_error = f"{type(exc).__name__}: {exc}"
            _family_loop_log(
                f"[family_loop] corr context fallback family={family} error={corr_context_error}"
            )

    parent_corr_by_candidate_id: dict[str, float] = {}
    if true_parent_series is not None and series_by_candidate_id:
        _family_loop_log(
            f"[family_loop] computing parent corr batch family={family} count={len(series_by_candidate_id)}"
        )
        parent_corr_raw = factor_series_correlations(
            true_parent_series,
            [(candidate_id, series) for candidate_id, series in series_by_candidate_id.items()],
        )
        parent_corr_by_candidate_id = {
            candidate_id: abs(value)
            for candidate_id, value in parent_corr_raw.items()
            if _finite_corr(value) is not None
        }

    enriched: list[dict[str, Any]] = []
    for candidate in candidates_by_id.values():
        candidate = dict(candidate)
        candidate_id = str(candidate.get("candidate_id", "") or "").strip()
        node = _node_from_payload(candidate, family=family, parent_expression=parent_expression)
        similarity_to_parent = pairwise_similarity(node, parent_node, policy)
        material_gain = _material_gain(
            candidate,
            discovered_parent_metrics,
            excess_min=min_material_excess_gain,
            icir_min=min_material_icir_gain,
        ) if discovered_parent_metrics else True
        candidate_series = series_by_candidate_id.get(candidate_id)
        true_corr_to_parent = None
        if candidate_series is not None and true_parent_series is not None:
            true_corr_to_parent = _finite_corr(parent_corr_by_candidate_id.get(candidate_id))
        true_parent_corr_guard_blocked = bool(
            true_corr_to_parent is not None
            and true_corr_to_parent >= float(max_true_parent_corr)
            and not material_gain
        )
        heuristic_parent_guard_blocked = bool(
            true_corr_to_parent is None
            and similarity_to_parent >= float(max_parent_similarity)
            and not material_gain
        )
        candidate["similarity_to_parent"] = similarity_to_parent
        candidate["true_corr_to_parent"] = true_corr_to_parent
        candidate["material_gain_vs_parent"] = material_gain
        candidate["true_parent_corr_guard_blocked"] = true_parent_corr_guard_blocked
        candidate["heuristic_parent_guard_blocked"] = heuristic_parent_guard_blocked
        candidate["stronger_candidate_corr_guard_blocked"] = False
        candidate["true_corr_to_stronger_candidate"] = None
        candidate["true_corr_stronger_factor_name"] = ""
        candidate["material_gain_vs_stronger_candidate"] = False
        candidate["corr_guard_blocked"] = bool(true_parent_corr_guard_blocked or heuristic_parent_guard_blocked)
        enriched.append(candidate)

    def _best_key(item: dict[str, Any]) -> tuple[float, float, float, float, float]:
        return (
            safe_float(item.get("net_excess_ann_return"), default=float("-inf")),
            safe_float(item.get("quick_rank_icir"), default=float("-inf")),
            safe_float(item.get("net_sharpe"), default=float("-inf")),
            safe_float(item.get("winner_score"), default=float("-inf")),
            -safe_float(item.get("mean_turnover"), default=float("inf")),
        )

    enriched_sorted = sorted(enriched, key=_best_key, reverse=True)
    stronger_candidates: list[dict[str, Any]] = []
    if series_by_candidate_id:
        _family_loop_log(
            f"[family_loop] computing stronger-sibling corr family={family} ordered_candidates={len(enriched_sorted)}"
        )
    for candidate in enriched_sorted:
        candidate_id = str(candidate.get("candidate_id", "") or "").strip()
        candidate_series = series_by_candidate_id.get(candidate_id)
        strongest_corr: float | None = None
        strongest_name = ""
        strongest_metrics: dict[str, Any] | None = None
        if candidate_series is not None:
            stronger_refs: list[tuple[str, pd.Series]] = []
            stronger_meta: dict[str, dict[str, Any]] = {}
            for stronger in stronger_candidates:
                stronger_id = str(stronger.get("candidate_id", "") or "").strip()
                stronger_series = series_by_candidate_id.get(stronger_id)
                if stronger_series is None:
                    continue
                stronger_refs.append((stronger_id, stronger_series))
                stronger_meta[stronger_id] = stronger
            if stronger_refs:
                corr_map = factor_series_correlations(candidate_series, stronger_refs)
                for stronger_id, corr_value_raw in corr_map.items():
                    corr_value = _finite_corr(abs(corr_value_raw))
                    if corr_value is None:
                        continue
                    if strongest_corr is None or corr_value > strongest_corr:
                        strongest_corr = corr_value
                        strongest_metrics = stronger_meta.get(stronger_id)
                        strongest_name = str((strongest_metrics or {}).get("factor_name", "") or "")
        if strongest_corr is not None:
            material_gain_vs_stronger = _material_gain(
                candidate,
                strongest_metrics or {},
                excess_min=min_material_excess_gain,
                icir_min=min_material_icir_gain,
            )
            stronger_guard_blocked = bool(
                strongest_corr >= float(max_true_sibling_corr)
                and not material_gain_vs_stronger
            )
            candidate["true_corr_to_stronger_candidate"] = strongest_corr
            candidate["true_corr_stronger_factor_name"] = strongest_name
            candidate["material_gain_vs_stronger_candidate"] = material_gain_vs_stronger
            candidate["stronger_candidate_corr_guard_blocked"] = stronger_guard_blocked
            candidate["corr_guard_blocked"] = bool(candidate.get("corr_guard_blocked") or stronger_guard_blocked)
        stronger_candidates.append(candidate)

    _family_loop_log(
        f"[family_loop] collect_anchor_candidates done family={family} candidates={len(enriched_sorted)} corr_mode={corr_mode}"
    )

    return {
        "family": family,
        "parent_name": parent_name,
        "parent_expression": parent_expression,
        "parent_metrics": discovered_parent_metrics,
        "corr_mode": corr_mode,
        "corr_context_error": corr_context_error,
        "true_parent_corr_threshold": float(max_true_parent_corr),
        "true_sibling_corr_threshold": float(max_true_sibling_corr),
        "candidates": enriched_sorted,
    }


def select_best_anchor(
    *,
    collected: dict[str, Any],
    target_profile: str = "raw_alpha",
    min_icir: float,
    min_sharpe: float,
    max_turnover: float,
    min_metrics_completeness: float,
) -> dict[str, Any]:
    context = DecisionContext.from_runtime(
        family=str(collected.get("family", "") or collected.get("parent_name", "") or ""),
        stage_mode="focused_refine",
        target_profile=str(target_profile or "raw_alpha"),
        policy_preset="balanced",
        context_profile=resolve_context_profile(
            ContextEvidence.from_runtime(
                family=str(collected.get("family", "") or collected.get("parent_name", "") or ""),
                stage_mode="focused_refine",
                target_profile=str(target_profile or "raw_alpha"),
                policy_preset="balanced",
                is_seed_stage=False,
                has_bootstrap_frontier=False,
                has_donor_motifs=False,
                has_decorrelation_targets=False,
                selected_parent_kind="graduated_broad_candidate",
            )
        ),
    )
    engine = DecisionEngine(context)
    return engine.select_anchor(
        collected=collected,
        min_icir=min_icir,
        min_sharpe=min_sharpe,
        max_turnover=max_turnover,
        min_metrics_completeness=min_metrics_completeness,
    )


def build_family_loop_summary(
    *,
    family: str,
    target_profile: str,
    loop_dir: Path,
    broad_stage_preset: str,
    focused_stage_preset: str,
    broad_run_dir: Path | None,
    broad_summary: dict[str, Any],
    anchor_selection: dict[str, Any],
    focused_run_dir: Path | None,
    focused_summary: dict[str, Any],
    broad_returncode: int = 0,
    focused_returncode: int = 0,
) -> dict[str, Any]:
    broad_snapshot = _snapshot_broad_results(broad_run_dir) if broad_run_dir is not None else {}
    state = FamilyDecisionState.from_family_loop_inputs(
        family=family,
        target_profile=target_profile,
        broad_summary=broad_summary,
        broad_snapshot=broad_snapshot,
        anchor_selection=anchor_selection,
        focused_summary=focused_summary,
        broad_returncode=int(broad_returncode),
        focused_returncode=int(focused_returncode),
    )
    broad_display = dict(state.broad_display_node or {})
    if not _has_any_eval_metric(broad_display):
        strongest_evaluated = dict(broad_snapshot.get("strongest_evaluated") or {})
        if strongest_evaluated:
            broad_display = strongest_evaluated
    decision_context = DecisionContext.from_runtime(
        family=family,
        stage_mode="family_loop",
        target_profile=target_profile,
        policy_preset="balanced",
        context_profile=resolve_context_profile(
            ContextEvidence.from_runtime(
                family=family,
                stage_mode="family_loop",
                target_profile=target_profile,
                policy_preset="balanced",
                is_seed_stage=False,
                has_bootstrap_frontier=False,
                has_donor_motifs=False,
                has_decorrelation_targets=False,
                selected_parent_kind="family_loop_anchor",
            )
        ),
    )
    decision_engine = DecisionEngine(decision_context)
    next_action = decision_engine.recommend_next_action(state)
    orchestration_evidence = ContextEvidence.from_runtime(
        family=family,
        stage_mode="family_loop",
        target_profile=target_profile,
        policy_preset="balanced",
        is_seed_stage=False,
        has_bootstrap_frontier=False,
        has_donor_motifs=False,
        has_decorrelation_targets=False,
        selected_parent_kind="family_loop_anchor",
    )
    orchestration_context_profile = resolve_context_profile(orchestration_evidence)
    orchestration_profile = resolve_orchestration_profile(
        evidence=orchestration_evidence,
        context_profile=orchestration_context_profile,
        last_round_status="ok" if int(focused_returncode) == 0 else "failed",
        last_round_search_improved=bool(next_action.get("recommended_next_step") == "continue_focused"),
        last_round_winner=dict(state.focused_best_candidate or state.broad_best_candidate or {}),
        last_round_keep=dict(state.focused_best_keep or state.broad_best_keep or {}),
        recommended_stage_mode_hint=str(next_action.get("recommended_next_stage_preset", "") or ""),
    )
    family_state = FamilyState(
        family_id=family,
        stage="focused_refine" if focused_run_dir is not None else "family_loop",
        target_profile=target_profile,
        best_node=dict(state.focused_best_node or state.broad_best_node or {}),
        redundancy_state={},
        failure_state={},
        budget_state={
            "budget_exhausted": False,
            "frontier_exhausted": False,
            "children_collected": int(dict(broad_snapshot or {}).get("total_candidates") or 0),
            "children_added_to_search": int(dict(broad_snapshot or {}).get("evaluated_candidate_count") or 0),
        },
    )
    refinement_action = RefinementAction(
        stage_mode=family_state.stage,
        target_profile=target_profile,
        policy_preset="balanced",
    )
    evaluation_feedback = EvaluationFeedback(
        status="ok" if int(focused_returncode) == 0 else "failed",
        search_improved=bool(next_action.get("recommended_next_step") == "continue_focused"),
        winner=dict(state.focused_best_candidate or state.broad_best_candidate or {}),
        keep=dict(state.focused_best_keep or state.broad_best_keep or {}),
        best_anchor=dict((anchor_selection or {}).get("best_anchor") or {}),
        passed_anchor_count=len((anchor_selection or {}).get("passed_candidates") or []),
        focused_best_node=dict(state.focused_best_node or {}),
        children_collected=int(dict(broad_snapshot or {}).get("total_candidates") or 0),
        children_added_to_search=int(dict(broad_snapshot or {}).get("evaluated_candidate_count") or 0),
    )
    stage_transition_evidence = build_stage_transition_evidence(
        family_state,
        refinement_action,
        evaluation_feedback,
    )
    stage_transition = resolve_stage_transition_from_state(
        family_state,
        refinement_action,
        evaluation_feedback,
    )
    legacy_decision = {
        **orchestration_profile.to_dict(),
        "recommended_next_step": next_action.get("recommended_next_step", ""),
        "recommended_next_stage_preset": next_action.get("recommended_next_stage_preset", ""),
        "recommended_reason": next_action.get("recommended_reason", ""),
    }
    stage_transition_shadow = build_stage_transition_shadow(
        legacy_decision=legacy_decision,
        family_state_decision=stage_transition,
    )

    return {
        "family": family,
        "target_profile": target_profile,
        "loop_dir": str(loop_dir),
        "generated_at": utc_now_iso(),
        "broad_stage_preset": str(broad_stage_preset),
        "focused_stage_preset": str(focused_stage_preset),
        "broad_run_dir": str(broad_run_dir) if broad_run_dir else "",
        "focused_run_dir": str(focused_run_dir) if focused_run_dir else "",
        "broad_prompt_trace": dict(broad_summary.get("prompt_trace") or {}),
        "focused_prompt_trace": dict(focused_summary.get("prompt_trace") or {}),
        "broad_stop_reason": state.broad_stop_reason,
        "focused_stop_reason": state.focused_stop_reason,
        "broad_best_node": state.broad_best_node,
        "broad_best_candidate": state.broad_best_candidate,
        "broad_best_keep": state.broad_best_keep,
        "broad_display_node": broad_display,
        "broad_candidate_snapshot": broad_snapshot,
        "anchor_selection": anchor_selection,
        "focused_best_node": state.focused_best_node,
        "focused_best_candidate": state.focused_best_candidate,
        "focused_best_keep": state.focused_best_keep,
        "focused_display_node": state.focused_display_node,
        "comparison": dict(next_action.get("comparison") or {}),
        "recommended_next_step": next_action.get("recommended_next_step", ""),
        "recommended_next_stage_preset": next_action.get("recommended_next_stage_preset", ""),
        "recommended_reason": next_action.get("recommended_reason", ""),
        "next_action_mode": next_action.get("next_action_mode", ""),
        "next_action_trace": dict(next_action.get("next_action_trace") or {}),
        "orchestration_context_evidence": orchestration_evidence.to_dict(),
        "orchestration_context_profile": orchestration_context_profile.to_dict(),
        "orchestration_profile": orchestration_profile.to_dict(),
        "family_state": family_state.to_dict(),
        "refinement_action": refinement_action.to_dict(),
        "evaluation_feedback": evaluation_feedback.to_dict(),
        "stage_transition_evidence": stage_transition_evidence.to_dict(),
        "stage_transition": stage_transition.to_dict(),
        "family_state_decision": stage_transition.to_dict(),
        "legacy_orchestration_decision": legacy_decision,
        "stage_transition_shadow": stage_transition_shadow,
        "broad_returncode": int(broad_returncode),
        "focused_returncode": int(focused_returncode),
    }


def render_family_loop_markdown(summary: dict[str, Any]) -> str:
    broad = dict(summary.get("broad_display_node") or summary.get("broad_best_node") or {})
    broad_best_candidate = dict(summary.get("broad_best_candidate") or {})
    broad_best_keep = dict(summary.get("broad_best_keep") or {})
    broad_snapshot = dict(summary.get("broad_candidate_snapshot") or {})
    anchor_selection = dict(summary.get("anchor_selection") or {})
    anchor = dict(anchor_selection.get("best_anchor") or {})
    focused = dict(summary.get("focused_display_node") or summary.get("focused_best_node") or {})
    focused_best_candidate = dict(summary.get("focused_best_candidate") or {})
    focused_best_keep = dict(summary.get("focused_best_keep") or {})
    comparison = dict(summary.get("comparison") or {})
    broad_trace = dict(summary.get("broad_prompt_trace") or {})
    focused_trace = dict(summary.get("focused_prompt_trace") or {})
    orchestration_profile = dict(summary.get("orchestration_profile") or {})
    stage_transition = dict(summary.get("stage_transition") or {})
    stage_transition_shadow = dict(summary.get("stage_transition_shadow") or {})
    stage_transition_tags = list(stage_transition.get("rationale_tags") or [])

    def _metric_line(payload: dict[str, Any]) -> str:
        if not payload:
            return "(empty)"
        return (
            f"IC={safe_float(payload.get('quick_rank_ic_mean'), default=0.0):.4f}, "
            f"ICIR={safe_float(payload.get('quick_rank_icir'), default=0.0):.4f}, "
            f"Ann={safe_float(payload.get('net_ann_return'), default=0.0):.4f}, "
            f"Excess={safe_float(payload.get('net_excess_ann_return'), default=0.0):.4f}, "
            f"Sharpe={safe_float(payload.get('net_sharpe'), default=0.0):.4f}, "
            f"TO={safe_float(payload.get('mean_turnover'), default=0.0):.4f}"
        )

    lines = [
        f"# Family Loop Summary: {summary.get('family', '')}",
        "",
        f"- target_profile: `{summary.get('target_profile', '')}`",
        f"- broad_stage_preset: `{summary.get('broad_stage_preset', '')}`",
        f"- focused_stage_preset: `{summary.get('focused_stage_preset', '')}`",
        f"- broad_run_dir: `{summary.get('broad_run_dir', '')}`",
        f"- focused_run_dir: `{summary.get('focused_run_dir', '')}`",
        f"- recommended_next_step: `{summary.get('recommended_next_step', '')}`",
        f"- recommended_next_stage_preset: `{summary.get('recommended_next_stage_preset', '')}`",
        f"- next_action_mode: `{summary.get('next_action_mode', '')}`",
        f"- reason: {summary.get('recommended_reason', '')}",
        f"- broad_stop_reason: `{summary.get('broad_stop_reason', '')}`",
        f"- focused_stop_reason: `{summary.get('focused_stop_reason', '')}`",
        "",
        "## Prompt Trace",
        f"- broad_stage_mode: `{broad_trace.get('stage_mode', '')}`",
        f"- broad_seed_stage_active: `{broad_trace.get('seed_stage_active', '')}`",
        f"- broad_selected_parent_kind: `{broad_trace.get('selected_parent_kind', '')}`",
        f"- broad_requested_candidate_count: `{broad_trace.get('requested_candidate_count', '')}`",
        f"- broad_bootstrap_frontier_count: `{broad_trace.get('bootstrap_frontier_count', '')}`",
        f"- broad_donor_motifs_count: `{broad_trace.get('donor_motifs_count', '')}`",
        f"- focused_stage_mode: `{focused_trace.get('stage_mode', '')}`",
        f"- focused_seed_stage_active: `{focused_trace.get('seed_stage_active', '')}`",
        f"- focused_selected_parent_kind: `{focused_trace.get('selected_parent_kind', '')}`",
        f"- focused_requested_candidate_count: `{focused_trace.get('requested_candidate_count', '')}`",
        f"- focused_bootstrap_frontier_count: `{focused_trace.get('bootstrap_frontier_count', '')}`",
        f"- focused_donor_motifs_count: `{focused_trace.get('donor_motifs_count', '')}`",
        "",
        "## Orchestration Trace",
        f"- recommended_stage_mode: `{orchestration_profile.get('recommended_stage_mode', '')}`",
        f"- round_strategy: `{orchestration_profile.get('round_strategy', '')}`",
        f"- promotion_bias: `{orchestration_profile.get('promotion_bias', '')}`",
        f"- parent_selection_bias: `{orchestration_profile.get('parent_selection_bias', '')}`",
        f"- termination_bias: `{orchestration_profile.get('termination_bias', '')}`",
        f"- confidence: `{orchestration_profile.get('confidence', '')}`",
        "",
        "## Stage Transition Advisory",
        f"- current_stage: `{stage_transition.get('current_stage', '')}`",
        f"- next_stage: `{stage_transition.get('next_stage', '')}`",
        f"- action: `{stage_transition.get('action', '')}`",
        f"- confidence: `{stage_transition.get('confidence', '')}`",
        f"- termination_bias: `{stage_transition.get('termination_bias', '')}`",
        f"- parent_selection_bias: `{stage_transition.get('parent_selection_bias', '')}`",
        f"- target_profile_bias: `{stage_transition.get('target_profile_bias', '')}`",
        f"- rationale_tags: `{', '.join(str(item) for item in stage_transition_tags)}`",
        f"- reason: {stage_transition.get('reason', '')}",
        "",
        "## Stage Transition Shadow",
        f"- legacy_next_stage: `{stage_transition_shadow.get('legacy_next_stage', '')}`",
        f"- legacy_action: `{stage_transition_shadow.get('legacy_action', '')}`",
        f"- family_state_next_stage: `{stage_transition_shadow.get('family_state_next_stage', '')}`",
        f"- family_state_action: `{stage_transition_shadow.get('family_state_action', '')}`",
        f"- stage_agrees: `{stage_transition_shadow.get('stage_agrees', '')}`",
        f"- action_agrees: `{stage_transition_shadow.get('action_agrees', '')}`",
        "",
        "## Broad Strongest",
        f"- factor: `{broad.get('factor_name', '')}`",
        f"- role: `{broad.get('role', broad.get('status', ''))}`",
        f"- metrics: {_metric_line(broad)}",
        "",
        "## Broad Round-Level Best Candidate",
        f"- factor: `{broad_best_candidate.get('factor_name', '')}`" if broad_best_candidate else "- factor: ``",
        (
            f"- status: `{broad_best_candidate.get('status', '')}`"
            if broad_best_candidate
            else "- status: ``"
        ),
        f"- metrics: {_metric_line(broad_best_candidate)}" if broad_best_candidate else "- metrics: (empty)",
        "",
        "## Broad Global Best Keep",
        f"- factor: `{broad_best_keep.get('factor_name', '')}`" if broad_best_keep else "- factor: ``",
        f"- metrics: {_metric_line(broad_best_keep)}" if broad_best_keep else "- metrics: (empty)",
        "",
        "## Broad Candidate Snapshot",
        f"- total_candidates: `{int(broad_snapshot.get('total_candidates') or 0)}`",
        f"- promotable_candidates: `{int(broad_snapshot.get('promotable_candidates') or 0)}`",
        f"- evaluated_candidate_count: `{int(broad_snapshot.get('evaluated_candidate_count') or 0)}`",
        f"- research_drop_count: `{int(broad_snapshot.get('research_drop_count') or 0)}`",
        f"- evaluation_failed_count: `{int(broad_snapshot.get('evaluation_failed_count') or 0)}`",
        "",
        "## Selected Anchor",
    ]
    if anchor_selection:
        lines.extend(
            [
                f"- anchor_selection_mode: `{anchor_selection.get('anchor_selection_mode', '')}`",
                f"- corr_mode: `{anchor_selection.get('corr_mode', '')}`",
                f"- passed_count: `{len(anchor_selection.get('passed_candidates') or [])}`",
                f"- rejected_count: `{len(anchor_selection.get('rejected_candidates') or [])}`",
            ]
        )
        if str(anchor_selection.get("corr_context_error", "") or "").strip():
            lines.append(f"- corr_context_error: `{anchor_selection.get('corr_context_error', '')}`")
    if anchor:
        lines.extend(
            [
                f"- factor: `{anchor.get('factor_name', '')}`",
                f"- metrics: {_metric_line(anchor)}",
                f"- similarity_to_parent: {safe_float(anchor.get('similarity_to_parent'), default=0.0):.4f}",
                (
                    f"- true_corr_to_parent: {safe_float(anchor.get('true_corr_to_parent'), default=0.0):.4f}"
                    if anchor.get("true_corr_to_parent") is not None
                    else "- true_corr_to_parent: (n/a)"
                ),
                (
                    f"- true_corr_to_stronger_candidate: {safe_float(anchor.get('true_corr_to_stronger_candidate'), default=0.0):.4f}"
                    f" vs `{anchor.get('true_corr_stronger_factor_name', '')}`"
                    if anchor.get("true_corr_to_stronger_candidate") is not None
                    else "- true_corr_to_stronger_candidate: (n/a)"
                ),
                f"- material_gain_vs_parent: {bool(anchor.get('material_gain_vs_parent'))}",
                f"- material_gain_vs_stronger_candidate: {bool(anchor.get('material_gain_vs_stronger_candidate'))}",
                f"- auto_applied_promotion: {bool(anchor.get('auto_applied_promotion'))}",
            ]
        )
    else:
        lines.append("- no anchor passed graduation gate")
    decision_counts = dict(broad_snapshot.get("decision_counts") or {})
    if decision_counts:
        lines.extend(["", "## Broad Decision Counts"])
        for key in sorted(decision_counts):
            lines.append(f"- {key}: {int(decision_counts[key])}")
    lines.extend(
        [
            "",
            "## Focused Best",
            f"- factor: `{focused.get('factor_name', '')}`" if focused else "- factor: ``",
            f"- metrics: {_metric_line(focused)}" if focused else "- metrics: (empty)",
            (
                f"- best_candidate: `{focused_best_candidate.get('factor_name', '')}`"
                if focused_best_candidate
                else "- best_candidate: ``"
            ),
            (
                f"- best_keep: `{focused_best_keep.get('factor_name', '')}`"
                if focused_best_keep
                else "- best_keep: ``"
            ),
            "",
            "## Anchor -> Focused Delta",
        ]
    )
    if comparison:
        for key, value in comparison.items():
            lines.append(f"- {key}: {safe_float(value, default=0.0):.4f}")
    else:
        lines.append("- no comparable focused delta")
    next_action_trace = dict(summary.get("next_action_trace") or {})
    if next_action_trace:
        lines.extend(["", "## Next Action Trace"])
        for key in (
            "target_profile",
            "stage_mode",
            "broad_stop_reason",
            "focused_stop_reason",
            "passed_anchor_candidate_count",
            "strong_anchor",
            "focused_improved_vs_anchor",
        ):
            if key in next_action_trace:
                lines.append(f"- {key}: `{next_action_trace.get(key)}`")
    return "\n".join(lines) + "\n"
