from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..config import DEFAULT_FAMILY_LOOP_RUNS_DIR
from ..core.archive import utc_now_iso
from ..search import SearchPolicy
from ..search.run_ingest import load_multi_run_candidate_records, resolve_materialized_child_run_dir
from ..search.scoring import pairwise_similarity, safe_float, winner_improved
from ..search.state import SearchNode


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
    family: str,
    parent_name: str,
    parent_expression: str,
    parent_metrics: dict[str, Any] | None = None,
    policy: SearchPolicy,
    max_parent_similarity: float,
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

    enriched: list[dict[str, Any]] = []
    for candidate in candidates_by_id.values():
        candidate = dict(candidate)
        node = _node_from_payload(candidate, family=family, parent_expression=parent_expression)
        similarity_to_parent = pairwise_similarity(node, parent_node, policy)
        material_gain = _material_gain(
            candidate,
            discovered_parent_metrics,
            excess_min=min_material_excess_gain,
            icir_min=min_material_icir_gain,
        ) if discovered_parent_metrics else True
        candidate["similarity_to_parent"] = similarity_to_parent
        candidate["material_gain_vs_parent"] = material_gain
        candidate["corr_guard_blocked"] = bool(
            similarity_to_parent >= float(max_parent_similarity)
            and not material_gain
        )
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
    return {
        "parent_name": parent_name,
        "parent_expression": parent_expression,
        "parent_metrics": discovered_parent_metrics,
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
        if _safe_bool(item.get("corr_guard_blocked")):
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
        "passed_candidates": passed_sorted,
        "rejected_candidates": rejected,
        "best_anchor": best_anchor,
    }


def build_family_loop_summary(
    *,
    family: str,
    target_profile: str,
    loop_dir: Path,
    broad_run_dir: Path | None,
    broad_summary: dict[str, Any],
    anchor_selection: dict[str, Any],
    focused_run_dir: Path | None,
    focused_summary: dict[str, Any],
) -> dict[str, Any]:
    broad_best = dict(broad_summary.get("best_node") or {})
    focused_best = dict(focused_summary.get("best_node") or {})
    best_anchor = dict(anchor_selection.get("best_anchor") or {})

    if not best_anchor:
        recommendation = "return_to_broad"
        reason = "broad 阶段没有候选通过 anchor graduation gate"
    elif not focused_best:
        recommendation = "freeze_anchor"
        reason = "anchor 已选出，但 focused 阶段没有形成可比 best_node"
    else:
        improved = winner_improved(focused_best, best_anchor)
        if improved:
            recommendation = "continue_focused"
            reason = "focused best node 相对 broad anchor 仍有实质提升"
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
        "broad_run_dir": str(broad_run_dir) if broad_run_dir else "",
        "focused_run_dir": str(focused_run_dir) if focused_run_dir else "",
        "broad_best_node": broad_best,
        "anchor_selection": anchor_selection,
        "focused_best_node": focused_best,
        "comparison": delta,
        "recommended_next_step": recommendation,
        "recommended_reason": reason,
    }


def render_family_loop_markdown(summary: dict[str, Any]) -> str:
    broad = dict(summary.get("broad_best_node") or {})
    anchor_selection = dict(summary.get("anchor_selection") or {})
    anchor = dict(anchor_selection.get("best_anchor") or {})
    focused = dict(summary.get("focused_best_node") or {})
    comparison = dict(summary.get("comparison") or {})

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
        f"- broad_run_dir: `{summary.get('broad_run_dir', '')}`",
        f"- focused_run_dir: `{summary.get('focused_run_dir', '')}`",
        f"- recommended_next_step: `{summary.get('recommended_next_step', '')}`",
        f"- reason: {summary.get('recommended_reason', '')}",
        "",
        "## Broad Strongest",
        f"- factor: `{broad.get('factor_name', '')}`",
        f"- metrics: {_metric_line(broad)}",
        "",
        "## Selected Anchor",
    ]
    if anchor:
        lines.extend(
            [
                f"- factor: `{anchor.get('factor_name', '')}`",
                f"- metrics: {_metric_line(anchor)}",
                f"- similarity_to_parent: {safe_float(anchor.get('similarity_to_parent'), default=0.0):.4f}",
                f"- material_gain_vs_parent: {bool(anchor.get('material_gain_vs_parent'))}",
                f"- auto_applied_promotion: {bool(anchor.get('auto_applied_promotion'))}",
            ]
        )
    else:
        lines.append("- no anchor passed graduation gate")
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
