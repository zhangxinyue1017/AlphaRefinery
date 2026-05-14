'''Prompt rendering for layered memory snapshots.'''

from __future__ import annotations

from typing import Any

from ..knowledge.retrieval import render_family_memory_block
from ..prompting.prompt_plan import PromptMemoryPlan


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def _shorten(text: Any, *, max_chars: int = 120) -> str:
    value = str(text or "").strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def _render_working_context(snapshot: dict[str, Any]) -> str:
    working = dict(snapshot.get("working_memory") or {})
    parent = dict(working.get("current_parent") or {})
    context = dict(working.get("context") or {})
    round_state = dict(working.get("round") or {})
    lines = [
        "当前工作记忆：",
        f"- stage={working.get('stage', '')}, target={working.get('target_profile', '')}, policy={working.get('policy_preset', '')}",
        f"- parent={parent.get('factor_name', '') or parent.get('candidate_name', '') or '(unknown)'}; round={round_state.get('round_id', 0)}",
    ]
    if context.get("donor_motifs_count"):
        lines.append(
            f"- donor_motifs={context.get('donor_motifs_count')} from {', '.join(context.get('donor_families') or []) or 'unknown'}"
        )
    if context.get("decorrelation_targets"):
        lines.append(f"- decorrelation_targets={', '.join(str(item) for item in context.get('decorrelation_targets') or [])}")
    return "\n".join(lines)


def _render_family_long_term(snapshot: dict[str, Any]) -> str:
    memory = dict(snapshot.get("family_long_term_memory") or {})
    saturation = dict(memory.get("family_saturation") or {})
    patterns = list(memory.get("successful_patterns") or [])[:3]
    failures = list(memory.get("failure_patterns") or [])[:3]
    lines: list[str] = []
    if saturation:
        lines.extend(
            [
                "Family long-term signals:",
                f"- saturation={saturation.get('grade', '') or 'NA'} score={_fmt_metric(saturation.get('score'))} escape={saturation.get('recommended_escape_mode', '') or 'NA'}",
            ]
        )
    if patterns:
        if not lines:
            lines.append("Family long-term signals:")
        labels = ", ".join(f"{item.get('label', '')}:{item.get('count', '')}" for item in patterns)
        lines.append(f"- successful_patterns={labels}")
    if failures:
        if not lines:
            lines.append("Family long-term signals:")
        labels = ", ".join(f"{item.get('label', '')}:{item.get('count', '')}" for item in failures)
        lines.append(f"- failure_patterns={labels}")
    return "\n".join(lines)


def _render_global_memory(snapshot: dict[str, Any]) -> str:
    memory = dict(snapshot.get("global_cross_family_memory") or {})
    donor_motifs = list(memory.get("donor_motifs") or [])[:3]
    if not donor_motifs:
        return ""
    lines = ["Global cross-family memory:"]
    for item in donor_motifs:
        lines.append(
            "- "
            f"{item.get('source_family', '')}/{item.get('source_factor_name', '')} "
            f"motif_score={_fmt_metric(item.get('motif_score'))}; "
            f"hint={_shorten(item.get('rationale', ''), max_chars=96)}"
        )
    return "\n".join(lines)


def render_memory_snapshot_prompt_block(
    snapshot: dict[str, Any],
    *,
    plan: PromptMemoryPlan,
) -> str:
    if not plan.include:
        return ""
    payload = dict(dict(snapshot.get("source_payloads") or {}).get("family_memory_payload") or {})
    if not payload:
        payload = {
            "recent_winners": list(dict(snapshot.get("short_term_memory") or {}).get("recent_winners") or []),
            "recent_keeps": list(dict(snapshot.get("short_term_memory") or {}).get("recent_keeps") or []),
            "recent_failures": list(dict(snapshot.get("short_term_memory") or {}).get("recent_failures") or []),
            "latest_reflection": dict(dict(snapshot.get("short_term_memory") or {}).get("latest_reflection") or {}),
            "lineage_trace": list(dict(snapshot.get("family_long_term_memory") or {}).get("lineage") or []),
        }
    blocks = [
        _render_working_context(snapshot),
        render_family_memory_block(
            payload,
            max_winners=plan.max_winners,
            max_keeps=plan.max_keeps,
            max_failures=plan.max_failures,
            include_lineage=plan.include_lineage,
            include_reflection=plan.include_reflection,
        ),
        _render_family_long_term(snapshot),
        _render_global_memory(snapshot),
    ]
    return "\n\n".join(block for block in blocks if str(block or "").strip()).strip()
