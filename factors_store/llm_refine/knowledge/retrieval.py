from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from ..core.archive import DEFAULT_ARCHIVE_DB
from ..core.models import SeedFamily
from .archive_queries import load_candidate_lineage, load_recent_failures, load_recent_keeps, load_recent_winners
from .empirical import build_model_empirical_summary, render_model_empirical_block


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _winner_is_prompt_worthy(item: dict[str, Any]) -> bool:
    excess = _safe_float(item.get("net_excess_ann_return"))
    sharpe = _safe_float(item.get("net_sharpe"))
    if excess is not None:
        return excess >= 0.0
    if sharpe is not None:
        return sharpe >= 2.0
    return True


def _truncate_failure_reason(reason: str, *, max_chars: int = 72) -> str:
    text = str(reason or "").strip() or "(none)"
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _load_latest_reflection_card(
    *,
    db_path: str | Path = DEFAULT_ARCHIVE_DB,
    family: str,
    exclude_run_id: str = "",
    max_runs: int = 12,
) -> dict[str, Any] | None:
    sql = """
        SELECT run_id, run_dir
        FROM runs
        WHERE family = ?
          AND status = 'completed'
        ORDER BY started_at DESC, run_id DESC
        LIMIT ?
    """
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(sql, (family, int(max_runs))).fetchall()
    for run_id, run_dir in rows:
        if exclude_run_id and str(run_id) == str(exclude_run_id):
            continue
        card_path = Path(str(run_dir)) / "metadata" / "reflection_card.json"
        if not card_path.exists():
            continue
        try:
            return json.loads(card_path.read_text(encoding="utf-8"))
        except Exception:
            continue
    return None


def build_family_memory_payload(
    *,
    db_path: str | Path = DEFAULT_ARCHIVE_DB,
    family: SeedFamily,
    limit: int = 3,
    exclude_run_id: str = "",
    current_model_name: str = "",
    current_parent_candidate_id: str = "",
) -> dict[str, Any]:
    recent_winners = load_recent_winners(
        db_path=db_path,
        family=family.family,
        limit=int(limit),
        exclude_run_id=exclude_run_id,
    )
    recent_keeps = load_recent_keeps(
        db_path=db_path,
        family=family.family,
        limit=int(limit),
        exclude_run_id=exclude_run_id,
    )
    recent_failures = load_recent_failures(
        db_path=db_path,
        family=family.family,
        limit=int(limit),
        exclude_run_id=exclude_run_id,
    )
    latest_reflection = _load_latest_reflection_card(
        db_path=db_path,
        family=family.family,
        exclude_run_id=exclude_run_id,
    )
    model_empirical_summary = build_model_empirical_summary(
        db_path=db_path,
        family=family.family,
        current_model_name=current_model_name,
    )
    lineage_trace = load_candidate_lineage(
        db_path=db_path,
        candidate_id=current_parent_candidate_id,
        max_hops=3,
    )
    return {
        "family": family.family,
        "recent_winners": recent_winners,
        "recent_keeps": recent_keeps,
        "recent_failures": recent_failures,
        "latest_reflection": latest_reflection,
        "model_empirical_summary": model_empirical_summary,
        "lineage_trace": lineage_trace,
    }


def _lineage_delta_label(prev: dict[str, Any], current: dict[str, Any]) -> str:
    try:
        prev_sharpe = float(prev.get("net_sharpe"))
        curr_sharpe = float(current.get("net_sharpe"))
    except Exception:
        prev_sharpe = curr_sharpe = 0.0
    try:
        prev_excess = float(prev.get("net_excess_ann_return"))
        curr_excess = float(current.get("net_excess_ann_return"))
    except Exception:
        prev_excess = curr_excess = 0.0
    if curr_sharpe > prev_sharpe and curr_excess >= prev_excess:
        return "improved"
    if curr_sharpe < prev_sharpe and curr_excess < prev_excess:
        return "degraded"
    return "mixed"


def render_family_memory_block(
    payload: dict[str, Any],
    *,
    max_winners: int = 2,
    max_keeps: int = 2,
    max_failures: int = 3,
    include_lineage: bool = True,
    include_reflection: bool = True,
) -> str:
    lines = [
        "最近有效经验（来自 archive 检索，用于启发，不要求机械模仿）：",
        "- 原则：优先吸收“有效改法”和“失败模式”，不要直接复制旧表达式。",
    ]
    raw_winners = payload.get("recent_winners") or []
    winners = [item for item in raw_winners if _winner_is_prompt_worthy(item)][: max(int(max_winners), 0)]
    if winners:
        lines.extend(["", "优先延续的有效模式："])
        for item in winners:
            lines.append(
                "- "
                f"{item.get('factor_name', '')} "
                f"(Sharpe={_fmt_metric(item.get('net_sharpe'))}, "
                f"Excess={_fmt_metric(item.get('net_excess_ann_return'))}, "
                f"tags={','.join(item.get('expression_tags', ())) or 'none'})"
            )
    elif raw_winners:
        lines.extend(["", "优先延续的有效模式：", "- (none after quality filter)"])
    else:
        lines.extend(["", "优先延续的有效模式：", "- (none)"])

    keeps = (payload.get("recent_keeps") or [])[: max(int(max_keeps), 0)]
    if keeps:
        lines.extend(["", "可借鉴但未完全确认的改法："])
        for item in keeps:
            lines.append(
                "- "
                f"{item.get('factor_name', '')} "
                f"(reason={item.get('reason', '') or '(none)'}, "
                f"tags={','.join(item.get('expression_tags', ())) or 'none'})"
            )

    failures = (payload.get("recent_failures") or [])[: max(int(max_failures), 0)]
    if failures:
        lines.extend(["", "近期要避免的失败模式："])
        for item in failures:
            lines.append(
                "- "
                f"{item.get('factor_name', '')} "
                f"[{item.get('status', '')}] "
                f"reason={_truncate_failure_reason(item.get('reason', ''))}"
            )

    lineage = payload.get("lineage_trace") or []
    if include_lineage and lineage:
        lines.extend(["", "当前 parent 最近 lineage："])
        for idx, item in enumerate(lineage[:3]):
            prefix = "seed" if idx == 0 else f"step_{idx}"
            parts = [
                f"{prefix}: {item.get('factor_name', '')}",
                f"mutation={item.get('mutation_class', '')}",
                f"Sharpe={_fmt_metric(item.get('net_sharpe'))}",
            ]
            if idx > 0:
                parts.append(f"delta={_lineage_delta_label(lineage[idx - 1], item)}")
            lines.append("- " + ", ".join(parts))

    reflection = payload.get("latest_reflection") or {}
    if include_reflection and reflection:
        lines.extend(["", "最近一次总结："])
        if reflection.get("summary"):
            lines.append(f"- summary: {reflection.get('summary')}")
        next_focus = reflection.get("suggested_next_focus") or []
        if next_focus:
            lines.append(f"- next_focus: {next_focus[0]}")

    return "\n".join(lines).strip()
