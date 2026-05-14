'''Build and persist layered memory snapshots.

The snapshot is intentionally an integration layer over existing archive,
reflection, search, and donor-transfer primitives. It does not make decisions
itself; it gives prompt construction, reports, and future managers one shared
view of what the run knew at a point in time.
'''

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.archive import DEFAULT_ARCHIVE_DB, utc_now_iso
from ..core.models import SeedFamily
from ..knowledge.retrieval import build_family_memory_payload

MEMORY_SNAPSHOT_SCHEMA_VERSION = "memory_snapshot.v1"


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if isinstance(value, dict):
        return dict(value)
    return {}


def _metric_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "quick_rank_ic_mean": record.get("quick_rank_ic_mean"),
        "quick_rank_icir": record.get("quick_rank_icir"),
        "net_ann_return": record.get("net_ann_return"),
        "net_excess_ann_return": record.get("net_excess_ann_return"),
        "net_sharpe": record.get("net_sharpe"),
        "mean_turnover": record.get("mean_turnover"),
    }


def _best_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}

    def score(item: dict[str, Any]) -> tuple[float, float, float]:
        def safe_float(value: Any) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return float("-inf")

        return (
            safe_float(item.get("net_sharpe")),
            safe_float(item.get("net_excess_ann_return")),
            safe_float(item.get("quick_rank_icir")),
        )

    return dict(max(records, key=score))


def _donor_families(donor_motifs: list[dict[str, Any]]) -> list[str]:
    families: list[str] = []
    for item in donor_motifs:
        family = str(item.get("source_family") or "").strip()
        if family and family not in families:
            families.append(family)
    return families


def _rejected_motifs(failures: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    rejected: list[dict[str, Any]] = []
    for item in failures[: max(int(limit), 0)]:
        rejected.append(
            {
                "factor_name": item.get("factor_name", ""),
                "status": item.get("status", ""),
                "reason": item.get("reason", ""),
                "operator_skeleton": item.get("operator_skeleton", ""),
                "expression_tags": list(item.get("expression_tags") or ()),
                "mutation_class": item.get("mutation_class", ""),
            }
        )
    return rejected


def _successful_patterns(reflection: dict[str, Any], recent_winners: list[dict[str, Any]]) -> list[dict[str, Any]]:
    patterns = [dict(item) for item in list(reflection.get("top_success_tags") or [])]
    if patterns:
        return patterns
    counts: dict[str, int] = {}
    for item in recent_winners:
        for tag in list(item.get("expression_tags") or []):
            key = str(tag).strip()
            if key:
                counts[key] = counts.get(key, 0) + 1
    return [{"label": key, "count": count} for key, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:3]]


def build_memory_snapshot(
    *,
    family: SeedFamily,
    db_path: str | Path = DEFAULT_ARCHIVE_DB,
    run_id: str = "",
    run_dir: str | Path = "",
    round_id: int = 0,
    stage_mode: str = "auto",
    target_profile: str = "raw_alpha",
    policy_preset: str = "balanced",
    selected_parent: Any = None,
    dual_parent: dict[str, Any] | None = None,
    requested_candidate_count: int = 0,
    final_candidate_target: int = 0,
    role_slots: list[str] | tuple[str, ...] = (),
    bootstrap_frontier: list[dict[str, Any]] | None = None,
    donor_motifs: list[dict[str, Any]] | None = None,
    decorrelation_targets: list[str] | tuple[str, ...] = (),
    prompt_trace: dict[str, Any] | None = None,
    current_model_name: str = "",
    current_parent_candidate_id: str = "",
    child_records: list[dict[str, Any]] | None = None,
    reflection_card: dict[str, Any] | None = None,
    search_summary: dict[str, Any] | None = None,
    saturation_assessment: dict[str, Any] | None = None,
    stage_transition: dict[str, Any] | None = None,
    stop_reason: str = "",
) -> dict[str, Any]:
    parent = _as_dict(selected_parent)
    donor_items = [dict(item) for item in list(donor_motifs or [])]
    children = [dict(item) for item in list(child_records or [])]
    best_child = _best_record(children)
    effective_parent_candidate_id = (
        str(current_parent_candidate_id or "").strip()
        or str(parent.get("candidate_id") or "").strip()
    )
    family_memory_payload = build_family_memory_payload(
        db_path=db_path,
        family=family,
        exclude_run_id=run_id,
        current_model_name=current_model_name,
        current_parent_candidate_id=effective_parent_candidate_id,
    )
    recent_winners = list(family_memory_payload.get("recent_winners") or [])
    recent_keeps = list(family_memory_payload.get("recent_keeps") or [])
    recent_failures = list(family_memory_payload.get("recent_failures") or [])
    latest_reflection = dict(reflection_card or family_memory_payload.get("latest_reflection") or {})
    prompt_trace_payload = dict(prompt_trace or {})
    search_payload = dict(search_summary or {})

    snapshot = {
        "schema_version": MEMORY_SNAPSHOT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "run_id": str(run_id or ""),
        "run_dir": str(run_dir or ""),
        "working_memory": {
            "family": family.family,
            "stage": str(stage_mode or "auto"),
            "target_profile": str(target_profile or "raw_alpha"),
            "policy_preset": str(policy_preset or "balanced"),
            "current_parent": parent,
            "dual_parent": dict(dual_parent or {}),
            "round": {
                "round_id": int(round_id or 0),
                "requested_candidate_count": int(requested_candidate_count or 0),
                "final_candidate_target": int(final_candidate_target or 0),
                "role_slots": list(role_slots or ()),
            },
            "current_candidates": children,
            "evaluation_snapshot": {
                "best_child": best_child,
                "best_child_metrics": _metric_snapshot(best_child) if best_child else {},
                "child_count": len(children),
            },
            "context": {
                "bootstrap_frontier_count": len(list(bootstrap_frontier or [])),
                "donor_motifs_count": len(donor_items),
                "donor_families": _donor_families(donor_items),
                "decorrelation_targets": list(decorrelation_targets or ()),
            },
            "prompt_trace": prompt_trace_payload,
        },
        "short_term_memory": {
            "recent_winners": recent_winners,
            "recent_keeps": recent_keeps,
            "recent_failures": recent_failures,
            "latest_reflection": latest_reflection,
            "recent_stop_reason": str(stop_reason or prompt_trace_payload.get("stop_reason") or ""),
            "recent_donor_usage": [
                {
                    "source_family": item.get("source_family", ""),
                    "source_factor_name": item.get("source_factor_name", ""),
                    "motif_score": item.get("motif_score"),
                    "retrieval_mode": item.get("retrieval_mode", ""),
                }
                for item in donor_items
            ],
            "rejected_motifs": _rejected_motifs(recent_failures),
        },
        "family_long_term_memory": {
            "family_metadata": {
                "canonical_seed": family.canonical_seed,
                "preferred_refine_seed": family.preferred_refine_seed or family.canonical_seed,
                "direction": family.direction,
                "primary_objective": family.primary_objective,
                "secondary_objective": family.secondary_objective,
            },
            "lineage": list(family_memory_payload.get("lineage_trace") or []),
            "successful_patterns": _successful_patterns(latest_reflection, recent_winners),
            "failure_patterns": list(latest_reflection.get("top_failure_reasons") or []),
            "promoted_anchors": recent_winners,
            "family_saturation": dict(saturation_assessment or {}),
            "stage_transition": dict(stage_transition or {}),
            "search_summary": search_payload,
            "model_empirical_summary": dict(family_memory_payload.get("model_empirical_summary") or {}),
        },
        "global_cross_family_memory": {
            "donor_motifs": donor_items,
            "donor_families": _donor_families(donor_items),
            "transfer_plan_refs": [
                {
                    "source_family": item.get("source_family", ""),
                    "source_factor_name": item.get("source_factor_name", ""),
                    "retrieval_mode": item.get("retrieval_mode", "runtime"),
                }
                for item in donor_items
            ],
            "similar_family_successes": donor_items,
            "target_profile_best_patterns": [],
        },
        "source_payloads": {
            "family_memory_payload": family_memory_payload,
        },
    }
    return _jsonable(snapshot)


def write_memory_snapshot(path: str | Path, snapshot: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(snapshot), ensure_ascii=False, indent=2), encoding="utf-8")
    return output
