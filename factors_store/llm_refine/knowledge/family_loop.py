from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from ...data import build_data_bundle
from ...registry import create_default_registry
from ..config import (
    DEFAULT_FAMILY_LOOP_ANCHOR_MAX_TRUE_PARENT_CORR,
    DEFAULT_FAMILY_LOOP_ANCHOR_MAX_TRUE_SIBLING_CORR,
    DEFAULT_FAMILY_LOOP_RUNS_DIR,
    FAMILY_LOOP_STAGE_PRESETS,
)
from ..core.archive import utc_now_iso
from ..core.seed_loader import load_seed_pool, resolve_family_formula
from ..evaluation.redundancy import factor_series_correlation
from ..parsing.expression_engine import WideExpressionEngine
from ..search import SearchPolicy
from ..search.run_ingest import load_multi_run_candidate_records, resolve_materialized_child_run_dir
from ..search.scoring import pairwise_similarity, safe_float, winner_improved
from ..search.state import SearchNode

_DEFAULT_BENCHMARK_PATH = Path("/root/dmd/BaoStock/Index/sh.000001.csv")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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
    if not effective_benchmark and _DEFAULT_BENCHMARK_PATH.exists():
        effective_benchmark = str(_DEFAULT_BENCHMARK_PATH)
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
            family_cfg, data, engine = _build_corr_data(
                seed_pool_path=seed_pool_path,
                family_name=family,
                panel_path=panel_path,
                benchmark_path=benchmark_path,
                start=start,
                end=end,
            )
            registry = create_default_registry()
            true_parent_series = _compute_series_for_factor_or_expression(
                family=family_cfg,
                factor_name=parent_name,
                expression=parent_expression,
                data=data,
                engine=engine,
                registry=registry,
            )
            for candidate in candidates_by_id.values():
                candidate_id = str(candidate.get("candidate_id", "") or "").strip()
                if not candidate_id:
                    continue
                factor_series = _compute_series_for_factor_or_expression(
                    family=family_cfg,
                    factor_name=str(candidate.get("factor_name", "") or ""),
                    expression=str(candidate.get("expression", "") or ""),
                    data=data,
                    engine=engine,
                    registry=registry,
                )
                if factor_series is not None:
                    series_by_candidate_id[candidate_id] = factor_series
            corr_mode = "true_correlation"
        except Exception as exc:
            corr_context_error = f"{type(exc).__name__}: {exc}"

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
            true_corr_to_parent = _finite_corr(abs(factor_series_correlation(candidate_series, true_parent_series)))
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
    for candidate in enriched_sorted:
        candidate_id = str(candidate.get("candidate_id", "") or "").strip()
        candidate_series = series_by_candidate_id.get(candidate_id)
        strongest_corr: float | None = None
        strongest_name = ""
        strongest_metrics: dict[str, Any] | None = None
        if candidate_series is not None:
            for stronger in stronger_candidates:
                stronger_series = series_by_candidate_id.get(str(stronger.get("candidate_id", "") or "").strip())
                if stronger_series is None:
                    continue
                corr_value = _finite_corr(abs(factor_series_correlation(candidate_series, stronger_series)))
                if corr_value is None:
                    continue
                if strongest_corr is None or corr_value > strongest_corr:
                    strongest_corr = corr_value
                    strongest_name = str(stronger.get("factor_name", "") or "")
                    strongest_metrics = stronger
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

    return {
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
    min_icir: float,
    min_sharpe: float,
    max_turnover: float,
    min_metrics_completeness: float,
) -> dict[str, Any]:
    passed: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for raw in list(collected.get("candidates") or []):
        item = dict(raw)
        gate_reasons: list[str] = []
        status = str(item.get("status", "") or item.get("decision", "")).strip().lower()
        if status not in {"research_winner", "winner", "research_keep", "keep"}:
            gate_reasons.append("status_not_promotable")
        anchor_eligible = ("winner" in status) or _safe_bool(item.get("eligible_for_best_node", True))
        if not anchor_eligible:
            gate_reasons.append("not_eligible_for_best_node")
        if safe_float(item.get("metrics_completeness"), default=0.0) < float(min_metrics_completeness):
            gate_reasons.append("metrics_incomplete")
        if safe_float(item.get("quick_rank_icir"), default=float("-inf")) < float(min_icir):
            gate_reasons.append("icir_below_threshold")
        if safe_float(item.get("net_sharpe"), default=float("-inf")) < float(min_sharpe):
            gate_reasons.append("sharpe_below_threshold")
        if safe_float(item.get("mean_turnover"), default=float("inf")) > float(max_turnover):
            gate_reasons.append("turnover_above_threshold")
        if _safe_bool(item.get("true_parent_corr_guard_blocked")):
            gate_reasons.append("true_parent_corr_guard_blocked")
        if _safe_bool(item.get("stronger_candidate_corr_guard_blocked")):
            gate_reasons.append("stronger_candidate_corr_guard_blocked")
        if _safe_bool(item.get("heuristic_parent_guard_blocked")):
            gate_reasons.append("heuristic_parent_guard_blocked")
        if _safe_bool(item.get("corr_guard_blocked")) and not any(
            reason.endswith("corr_guard_blocked") for reason in gate_reasons
        ):
            gate_reasons.append("corr_guard_blocked")

        item["graduation_gate_reasons"] = list(gate_reasons)
        item["graduation_passed"] = not gate_reasons
        if item["graduation_passed"]:
            passed.append(item)
        else:
            rejected.append(item)

    def _sort_key(item: dict[str, Any]) -> tuple[float, float, float, float, float, float, float]:
        status_rank = 1.0 if "winner" in str(item.get("status", "")).lower() else 0.0
        return (
            safe_float(item.get("net_excess_ann_return"), default=float("-inf")),
            safe_float(item.get("quick_rank_icir"), default=float("-inf")),
            safe_float(item.get("net_sharpe"), default=float("-inf")),
            safe_float(item.get("winner_score"), default=float("-inf")),
            -safe_float(item.get("mean_turnover"), default=float("inf")),
            status_rank,
            safe_float(item.get("quick_rank_ic_mean"), default=float("-inf")),
        )

    passed_sorted = sorted(passed, key=_sort_key, reverse=True)
    best_anchor = dict(passed_sorted[0]) if passed_sorted else {}
    if best_anchor:
        best_anchor["selected_as_anchor"] = True

    return {
        "parent_name": collected.get("parent_name", ""),
        "parent_expression": collected.get("parent_expression", ""),
        "parent_metrics": dict(collected.get("parent_metrics") or {}),
        "corr_mode": collected.get("corr_mode", ""),
        "corr_context_error": collected.get("corr_context_error", ""),
        "true_parent_corr_threshold": safe_float(collected.get("true_parent_corr_threshold"), default=0.0),
        "true_sibling_corr_threshold": safe_float(collected.get("true_sibling_corr_threshold"), default=0.0),
        "passed_candidates": passed_sorted,
        "rejected_candidates": rejected,
        "best_anchor": best_anchor,
    }


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
    broad_best = dict(broad_summary.get("best_node") or {})
    broad_snapshot = _snapshot_broad_results(broad_run_dir) if broad_run_dir is not None else {}
    focused_best = dict(focused_summary.get("best_node") or {})
    best_anchor = dict(anchor_selection.get("best_anchor") or {})
    broad_stop_reason = str(broad_summary.get("stop_reason", "") or "")
    focused_stop_reason = str(focused_summary.get("stop_reason", "") or "")
    passed_candidates = list(anchor_selection.get("passed_candidates") or [])

    broad_display = dict(broad_best)
    if not _has_any_eval_metric(broad_display):
        strongest_evaluated = dict(broad_snapshot.get("strongest_evaluated") or {})
        if strongest_evaluated:
            broad_display = strongest_evaluated

    def _strong_anchor(payload: dict[str, Any]) -> bool:
        return (
            safe_float(payload.get("net_excess_ann_return"), default=float("-inf")) > 0.0
            and safe_float(payload.get("quick_rank_icir"), default=float("-inf")) >= 0.60
            and safe_float(payload.get("net_sharpe"), default=float("-inf")) >= 4.0
        )

    recommended_stage_preset = ""
    if not best_anchor:
        recommendation = "return_to_broad"
        reason = "broad 阶段没有候选通过 anchor graduation gate"
        recommended_stage_preset = "new_family_broad"
    elif focused_returncode != 0 or not focused_best:
        if _strong_anchor(best_anchor):
            recommendation = "donor_mode"
            reason = "focused 阶段没有形成新的 best_node，但当前 anchor 已足够强，适合转 donor/confirmation"
            recommended_stage_preset = "donor_validation"
        else:
            recommendation = "freeze_anchor"
            reason = "anchor 已选出，但 focused 阶段没有形成可比 best_node"
    else:
        improved = winner_improved(focused_best, best_anchor)
        delta_excess = safe_float(focused_best.get("net_excess_ann_return"), default=0.0) - safe_float(
            best_anchor.get("net_excess_ann_return"), default=0.0
        )
        delta_icir = safe_float(focused_best.get("quick_rank_icir"), default=0.0) - safe_float(
            best_anchor.get("quick_rank_icir"), default=0.0
        )
        delta_sharpe = safe_float(focused_best.get("net_sharpe"), default=0.0) - safe_float(
            best_anchor.get("net_sharpe"), default=0.0
        )
        material_improvement = delta_excess >= 0.02 or delta_icir >= 0.05 or delta_sharpe >= 0.25
        if improved and material_improvement:
            recommendation = "continue_focused"
            reason = "focused best node 相对 broad anchor 仍有实质提升"
            recommended_stage_preset = "focused_refine"
        elif improved:
            recommendation = "confirmation"
            reason = "focused best node 仍优于 anchor，但增益已转小，适合进入轻量确认阶段"
            recommended_stage_preset = "confirmation"
        else:
            if _strong_anchor(best_anchor):
                recommendation = "donor_mode"
                reason = "focused 阶段没有继续抬高 anchor，但当前 anchor 足够强，适合转 donor/confirmation"
                recommended_stage_preset = "donor_validation"
            elif len(passed_candidates) >= 2:
                recommendation = "return_to_broad"
                reason = "focused 未继续改善，而且 broad 阶段仍有其他通过 gate 的候选，建议回到 broad 重新展开"
                recommended_stage_preset = "new_family_broad"
            else:
                recommendation = "freeze_anchor"
                reason = "focused 阶段没有继续抬高 anchor 的综合质量"

    delta = {}
    if best_anchor and focused_best:
        for metric in (
            "quick_rank_ic_mean",
            "quick_rank_icir",
            "net_ann_return",
            "net_excess_ann_return",
            "net_sharpe",
            "mean_turnover",
        ):
            delta[f"delta_anchor_to_focused_{metric}"] = safe_float(
                focused_best.get(metric), default=0.0
            ) - safe_float(best_anchor.get(metric), default=0.0)

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
        "broad_stop_reason": broad_stop_reason,
        "focused_stop_reason": focused_stop_reason,
        "broad_best_node": broad_best,
        "broad_display_node": broad_display,
        "broad_candidate_snapshot": broad_snapshot,
        "anchor_selection": anchor_selection,
        "focused_best_node": focused_best,
        "comparison": delta,
        "recommended_next_step": recommendation,
        "recommended_next_stage_preset": recommended_stage_preset,
        "recommended_reason": reason,
        "broad_returncode": int(broad_returncode),
        "focused_returncode": int(focused_returncode),
    }


def render_family_loop_markdown(summary: dict[str, Any]) -> str:
    broad = dict(summary.get("broad_display_node") or summary.get("broad_best_node") or {})
    broad_snapshot = dict(summary.get("broad_candidate_snapshot") or {})
    anchor_selection = dict(summary.get("anchor_selection") or {})
    anchor = dict(anchor_selection.get("best_anchor") or {})
    focused = dict(summary.get("focused_best_node") or {})
    comparison = dict(summary.get("comparison") or {})
    broad_trace = dict(summary.get("broad_prompt_trace") or {})
    focused_trace = dict(summary.get("focused_prompt_trace") or {})

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
        "## Broad Strongest",
        f"- factor: `{broad.get('factor_name', '')}`",
        f"- role: `{broad.get('role', broad.get('status', ''))}`",
        f"- metrics: {_metric_line(broad)}",
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
            "",
            "## Anchor -> Focused Delta",
        ]
    )
    if comparison:
        for key, value in comparison.items():
            lines.append(f"- {key}: {safe_float(value, default=0.0):.4f}")
    else:
        lines.append("- no comparable focused delta")
    return "\n".join(lines) + "\n"
