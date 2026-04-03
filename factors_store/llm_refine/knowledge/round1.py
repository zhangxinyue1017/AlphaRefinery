from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import (
    DEFAULT_ROUND1_BOOTSTRAP_ALIAS_LIMIT,
    DEFAULT_ROUND1_CANDIDATE_ROLES,
    DEFAULT_ROUND1_EXTRA_CANDIDATES,
    DEFAULT_ROUND1_LIGHT_RERANK_MAX_SELECTED_SIMILARITY,
)
from ..core.archive import make_seed_candidate_id
from ..core.models import RefinementCandidate, SeedFamily, SeedPool
from ..core.seed_loader import resolve_family_formula, resolve_preferred_refine_seed
from ..search.policy import SearchPolicy
from ..search.scoring import expression_complexity, pairwise_similarity, safe_float
from ..search.state import SearchNode

_SEED_STAGE_NODE_KINDS = {
    "canonical_seed",
    "preferred_seed",
    "alias_seed",
    "bootstrap_parent",
    "bootstrap_seed",
}
_LEGACY_CANDIDATE_ROLES = ("conservative", "decorrelating", "enhancing")


@dataclass(frozen=True)
class BootstrapParent:
    factor_name: str
    expression: str
    candidate_id: str
    node_kind: str
    status: str
    metrics: dict[str, Any]
    bootstrap_score: tuple[float, float, float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.metrics)
        payload.update(
            {
                "factor_name": self.factor_name,
                "expression": self.expression,
                "candidate_id": self.candidate_id,
                "node_kind": self.node_kind,
                "status": self.status,
                "bootstrap_score": list(self.bootstrap_score),
            }
        )
        return payload


def is_seed_stage_node_kind(node_kind: str) -> bool:
    return str(node_kind or "").strip() in _SEED_STAGE_NODE_KINDS


def resolve_round1_role_slots(
    *,
    family: SeedFamily,
    final_candidate_target: int,
    seed_stage_active: bool,
) -> tuple[str, ...]:
    if seed_stage_active:
        return tuple(DEFAULT_ROUND1_CANDIDATE_ROLES)
    roles = tuple(family.candidate_roles or _LEGACY_CANDIDATE_ROLES)
    return roles[: max(int(final_candidate_target), 1)]


def resolve_requested_candidate_count(
    *,
    final_candidate_target: int,
    role_slots: tuple[str, ...],
    seed_stage_active: bool,
) -> int:
    final_target = max(int(final_candidate_target), 1)
    if not seed_stage_active:
        return final_target
    return max(final_target + int(DEFAULT_ROUND1_EXTRA_CANDIDATES), len(role_slots), final_target)


def build_bootstrap_frontier(
    *,
    seed_pool: SeedPool,
    family: SeedFamily,
    alias_limit: int = DEFAULT_ROUND1_BOOTSTRAP_ALIAS_LIMIT,
) -> list[dict[str, Any]]:
    from ..prompting.prompt_builder import load_prompt_history_row

    candidate_names: list[tuple[str, str]] = []
    preferred = resolve_preferred_refine_seed(family)
    candidate_names.append((preferred, "preferred_seed" if preferred != family.canonical_seed else "canonical_seed"))
    if family.canonical_seed != preferred:
        candidate_names.append((family.canonical_seed, "canonical_seed"))
    for alias in family.aliases[: max(int(alias_limit), 0)]:
        candidate_names.append((alias, "alias_seed"))

    seen: set[str] = set()
    frontier: list[BootstrapParent] = []
    for factor_name, node_kind in candidate_names:
        name = str(factor_name or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        expression = resolve_family_formula(family, name)
        if not expression:
            continue
        metrics = load_prompt_history_row(seed_pool, name, family=family) or {}
        frontier.append(
            BootstrapParent(
                factor_name=name,
                expression=expression,
                candidate_id=make_seed_candidate_id(name),
                node_kind=node_kind,
                status="seed",
                metrics=metrics,
                bootstrap_score=_bootstrap_sort_key(metrics),
            )
        )

    frontier.sort(key=lambda item: item.bootstrap_score, reverse=True)
    return [item.to_dict() for item in frontier]


def select_bootstrap_parent(frontier: list[dict[str, Any]]) -> dict[str, Any]:
    if not frontier:
        return {}
    return dict(frontier[0])


def light_rerank_candidates(
    *,
    candidates: tuple[RefinementCandidate, ...],
    family: SeedFamily,
    parent_factor_name: str,
    parent_expression: str,
    policy: SearchPolicy,
    final_candidate_target: int,
    role_slots: tuple[str, ...],
    max_selected_similarity: float = DEFAULT_ROUND1_LIGHT_RERANK_MAX_SELECTED_SIMILARITY,
) -> tuple[tuple[RefinementCandidate, ...], tuple[dict[str, Any], ...], dict[str, Any]]:
    final_target = max(int(final_candidate_target), 1)
    if len(candidates) <= final_target:
        report = {
            "requested": len(candidates),
            "final_target": final_target,
            "selected_count": len(candidates),
            "dropped_count": 0,
            "role_slots": list(role_slots),
            "selected": [_candidate_report_row(item, role_slots=role_slots, parent_expression=parent_expression, policy=policy) for item in candidates],
            "dropped": [],
        }
        return candidates, (), report

    parent_node = _make_parent_node(
        family=family.family,
        factor_name=parent_factor_name,
        expression=parent_expression,
    )
    ranked_rows = [
        _candidate_report_row(item, role_slots=role_slots, parent_expression=parent_expression, policy=policy, parent_node=parent_node)
        for item in candidates
    ]
    row_by_id = {row["candidate"].candidate_id: row for row in ranked_rows}

    selected: list[RefinementCandidate] = []
    selected_rows: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    dropped_rows: list[dict[str, Any]] = []

    for role in role_slots:
        if len(selected) >= final_target:
            break
        candidates_for_role = [
            row
            for row in ranked_rows
            if row["candidate"].candidate_id not in selected_ids and row["candidate_role"] == role
        ]
        chosen = _pick_non_redundant_row(
            candidates_for_role,
            selected_rows=selected_rows,
            policy=policy,
            max_selected_similarity=max_selected_similarity,
        )
        if chosen is None:
            continue
        chosen["selection_reason"] = f"role_coverage:{role}"
        selected.append(chosen["candidate"])
        selected_rows.append(chosen)
        selected_ids.add(chosen["candidate"].candidate_id)

    fill_pool = [
        row
        for row in sorted(
            ranked_rows,
            key=lambda item: (float(item["selection_score"]), -float(item["max_similarity_to_parent"])),
            reverse=True,
        )
        if row["candidate"].candidate_id not in selected_ids
    ]
    for row in fill_pool:
        if len(selected) >= final_target:
            break
        max_similarity_to_selected = _max_similarity_to_selected(
            row=row,
            selected_rows=selected_rows,
            policy=policy,
        )
        if max_similarity_to_selected > float(max_selected_similarity):
            row["drop_reason"] = (
                f"peer_similarity={max_similarity_to_selected:.3f} > "
                f"{float(max_selected_similarity):.3f}"
            )
            dropped_rows.append(row)
            continue
        row["selection_reason"] = "quality_fill"
        row["max_similarity_to_selected"] = max_similarity_to_selected
        selected.append(row["candidate"])
        selected_rows.append(row)
        selected_ids.add(row["candidate"].candidate_id)

    for row in ranked_rows:
        if row["candidate"].candidate_id in selected_ids:
            continue
        if "drop_reason" not in row:
            row["drop_reason"] = "lower_priority_after_role_coverage"
        row["max_similarity_to_selected"] = _max_similarity_to_selected(
            row=row,
            selected_rows=selected_rows,
            policy=policy,
        )
        dropped_rows.append(row)

    dropped_payloads = tuple(
        {
            "candidate": row["candidate"],
            "filter_stage": "light_rerank",
            "filter_reason": str(row.get("drop_reason", "")),
            "selection_score": float(row.get("selection_score", 0.0)),
            "candidate_role": str(row.get("candidate_role", "")),
            "parent_similarity": float(row.get("max_similarity_to_parent", 0.0)),
            "peer_similarity": float(row.get("max_similarity_to_selected", 0.0)),
        }
        for row in dropped_rows
    )
    report = {
        "requested": len(candidates),
        "final_target": final_target,
        "selected_count": len(selected),
        "dropped_count": len(dropped_payloads),
        "role_slots": list(role_slots),
        "selected": [_serialize_report_row(row) for row in selected_rows],
        "dropped": [_serialize_report_row(row) for row in dropped_rows],
    }
    return tuple(selected), dropped_payloads, report


def light_rerank_markdown(
    *,
    family: SeedFamily,
    parent_factor_name: str,
    parent_expression: str,
    report: dict[str, Any],
) -> str:
    def _row_label(row: dict[str, Any]) -> str:
        return str(
            row.get("factor_name")
            or row.get("candidate_name")
            or row.get("name")
            or row.get("expression")
            or "unknown_candidate"
        )

    def _row_role(row: dict[str, Any]) -> str:
        return str(row.get("candidate_role") or row.get("role") or "")

    def _row_parent_sim(row: dict[str, Any]) -> float:
        return float(row.get("max_similarity_to_parent", row.get("parent_similarity", 0.0)))

    def _row_peer_sim(row: dict[str, Any]) -> float:
        return float(row.get("max_similarity_to_selected", row.get("peer_similarity", 0.0)))

    lines = [
        "# Light Rerank Report",
        "",
        f"- family: `{family.family}`",
        f"- parent: `{parent_factor_name}`",
        f"- parent_expression: `{parent_expression}`",
        f"- requested_candidates: {int(report.get('requested', 0))}",
        f"- final_target: {int(report.get('final_target', 0))}",
        f"- selected_count: {int(report.get('selected_count', 0))}",
        f"- dropped_count: {int(report.get('dropped_count', 0))}",
        f"- role_slots: {', '.join(report.get('role_slots') or []) or '-'}",
        "",
        "## Selected",
        "",
    ]
    selected = list(report.get("selected") or [])
    if not selected:
        lines.append("- (none)")
    else:
        for row in selected:
            lines.append(
                f"- `{_row_label(row)}` | role=`{_row_role(row)}` | "
                f"score={float(row.get('selection_score', 0.0)):.3f} | parent_sim={_row_parent_sim(row):.3f} | "
                f"peer_sim={_row_peer_sim(row):.3f} | reason={row.get('selection_reason', '')}"
            )
    lines.extend(["", "## Dropped", ""])
    dropped = list(report.get("dropped") or [])
    if not dropped:
        lines.append("- (none)")
    else:
        for row in dropped:
            lines.append(
                f"- `{_row_label(row)}` | role=`{_row_role(row)}` | "
                f"score={float(row.get('selection_score', 0.0)):.3f} | parent_sim={_row_parent_sim(row):.3f} | "
                f"peer_sim={_row_peer_sim(row):.3f} | reason={row.get('drop_reason', '')}"
            )
    return "\n".join(lines).strip() + "\n"


def _bootstrap_sort_key(metrics: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    has_metrics = 1.0 if metrics else 0.0
    excess = safe_float(metrics.get("net_excess_ann_return"), default=safe_float(metrics.get("net_ann_return"), default=float("-inf")))
    icir = safe_float(metrics.get("quick_rank_icir"), default=float("-inf"))
    rank_ic = safe_float(metrics.get("quick_rank_ic_mean"), default=float("-inf"))
    sharpe = safe_float(metrics.get("net_sharpe"), default=float("-inf"))
    turnover = safe_float(metrics.get("mean_turnover"), default=float("inf"))
    return (
        has_metrics,
        excess,
        icir,
        rank_ic,
        sharpe,
        -turnover,
    )


def _candidate_report_row(
    candidate: RefinementCandidate,
    *,
    role_slots: tuple[str, ...],
    parent_expression: str,
    policy: SearchPolicy,
    parent_node: SearchNode | None = None,
) -> dict[str, Any]:
    node = _make_candidate_node(candidate=candidate, parent_expression=parent_expression)
    parent = parent_node or _make_parent_node(
        family=candidate.family,
        factor_name=candidate.parent_factor or "parent",
        expression=parent_expression,
    )
    parent_similarity = pairwise_similarity(node, parent, policy)
    parent_complexity = expression_complexity(parent_expression)
    complexity_delta = expression_complexity(candidate.expression) - parent_complexity
    role = str(candidate.candidate_role or "").strip()
    selection_score = _selection_score(
        role=role,
        role_slots=role_slots,
        parent_similarity=parent_similarity,
        complexity_delta=complexity_delta,
    )
    return {
        "candidate": candidate,
        "node": node,
        "candidate_role": role,
        "selection_score": selection_score,
        "complexity_delta": complexity_delta,
        "max_similarity_to_parent": parent_similarity,
        "max_similarity_to_selected": 0.0,
        "selection_reason": "",
        "drop_reason": "",
    }


def _selection_score(
    *,
    role: str,
    role_slots: tuple[str, ...],
    parent_similarity: float,
    complexity_delta: float,
) -> float:
    novelty = 1.0 - float(parent_similarity)
    score = novelty
    if role in role_slots:
        score += 0.2
    if role == "conservative":
        if 0.55 <= parent_similarity <= 0.92:
            score += 0.15
        if complexity_delta <= 0.75:
            score += 0.08
    elif role == "confirmation":
        if 0.45 <= parent_similarity <= 0.9:
            score += 0.12
        if complexity_delta <= 1.0:
            score += 0.08
    elif role == "decorrelating":
        if parent_similarity <= 0.75:
            score += 0.2
    elif role == "donor_transfer":
        if parent_similarity <= 0.88:
            score += 0.14
    elif role == "simplifying":
        if complexity_delta <= 0.0:
            score += 0.2
    elif role == "stretch":
        if complexity_delta > 0.4:
            score += 0.1
        if parent_similarity <= 0.85:
            score += 0.08
    return float(score)


def _pick_non_redundant_row(
    rows: list[dict[str, Any]],
    *,
    selected_rows: list[dict[str, Any]],
    policy: SearchPolicy,
    max_selected_similarity: float,
) -> dict[str, Any] | None:
    ranked = sorted(rows, key=lambda item: float(item["selection_score"]), reverse=True)
    for row in ranked:
        max_similarity_to_selected = _max_similarity_to_selected(
            row=row,
            selected_rows=selected_rows,
            policy=policy,
        )
        if max_similarity_to_selected <= float(max_selected_similarity):
            row["max_similarity_to_selected"] = max_similarity_to_selected
            return row
        row["drop_reason"] = (
            f"peer_similarity={max_similarity_to_selected:.3f} > "
            f"{float(max_selected_similarity):.3f}"
        )
    return None


def _max_similarity_to_selected(
    *,
    row: dict[str, Any],
    selected_rows: list[dict[str, Any]],
    policy: SearchPolicy,
) -> float:
    if not selected_rows:
        return 0.0
    return max(
        float(pairwise_similarity(row["node"], selected_row["node"], policy))
        for selected_row in selected_rows
    )


def _make_parent_node(*, family: str, factor_name: str, expression: str) -> SearchNode:
    return SearchNode(
        node_id=f"parent::{factor_name}",
        family=family,
        factor_name=factor_name,
        expression=expression,
        candidate_id=make_seed_candidate_id(factor_name),
        node_kind="seed",
        status="seed",
        metadata={"parent_expression": ""},
    )


def _make_candidate_node(*, candidate: RefinementCandidate, parent_expression: str) -> SearchNode:
    return SearchNode(
        node_id=candidate.candidate_id or f"candidate::{candidate.name}",
        family=candidate.family,
        factor_name=candidate.name,
        expression=candidate.expression,
        candidate_id=candidate.candidate_id,
        parent_candidate_id=candidate.parent_candidate_id,
        node_kind="candidate",
        status=candidate.status,
        metadata={"parent_expression": parent_expression},
    )


def _serialize_report_row(row: dict[str, Any]) -> dict[str, Any]:
    candidate = row["candidate"]
    return {
        "factor_name": candidate.name,
        "candidate_id": candidate.candidate_id,
        "candidate_role": row["candidate_role"],
        "selection_score": round(float(row["selection_score"]), 6),
        "complexity_delta": round(float(row["complexity_delta"]), 6),
        "max_similarity_to_parent": round(float(row["max_similarity_to_parent"]), 6),
        "max_similarity_to_selected": round(float(row.get("max_similarity_to_selected", 0.0)), 6),
        "selection_reason": str(row.get("selection_reason", "")),
        "drop_reason": str(row.get("drop_reason", "")),
    }
