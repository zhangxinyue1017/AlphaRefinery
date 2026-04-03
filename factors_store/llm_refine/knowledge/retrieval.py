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


def render_family_memory_block(payload: dict[str, Any]) -> str:
    lines = [
        "最近有效经验（来自 archive 检索，用于启发，不要求机械模仿）：",
        "- 原则：优先吸收“有效改法”和“失败模式”，不要直接复制旧表达式。",
        "",
        "Recent winners:",
    ]
    winners = payload.get("recent_winners") or []
    if winners:
        for item in winners:
            lines.append(
                "- "
                f"{item.get('factor_name', '')} "
                f"(model={item.get('source_model', '')}, "
                f"Sharpe={_fmt_metric(item.get('net_sharpe'))}, "
                f"Excess={_fmt_metric(item.get('net_excess_ann_return'))}, "
                f"Turn={_fmt_metric(item.get('mean_turnover'))}, "
                f"tags={','.join(item.get('expression_tags', ())) or 'none'})"
            )
    else:
        lines.append("- (none)")

    lines.extend(["", "Recent keeps:"])
    keeps = payload.get("recent_keeps") or []
    if keeps:
        for item in keeps:
            lines.append(
                "- "
                f"{item.get('factor_name', '')} "
                f"(model={item.get('source_model', '')}, "
                f"reason={item.get('reason', '') or '(none)'}, "
                f"tags={','.join(item.get('expression_tags', ())) or 'none'})"
            )
    else:
        lines.append("- (none)")

    lines.extend(["", "Recent failures:"])
    failures = payload.get("recent_failures") or []
    if failures:
        for item in failures:
            lines.append(
                "- "
                f"{item.get('factor_name', '')} "
                f"[{item.get('status', '')}] "
                f"reason={item.get('reason', '') or '(none)'}"
            )
    else:
        lines.append("- (none)")

    lineage = payload.get("lineage_trace") or []
    if lineage:
        lines.extend(["", "Current parent lineage trace:"])
        for idx, item in enumerate(lineage):
            prefix = "seed" if idx == 0 else f"step_{idx}"
            parts = [
                f"{prefix}: {item.get('factor_name', '')}",
                f"mutation={item.get('mutation_class', '')}",
                f"skeleton={item.get('operator_skeleton', '')}",
                f"Sharpe={_fmt_metric(item.get('net_sharpe'))}",
                f"Excess={_fmt_metric(item.get('net_excess_ann_return'))}",
            ]
            if idx > 0:
                parts.append(f"delta={_lineage_delta_label(lineage[idx - 1], item)}")
            lines.append("- " + ", ".join(parts))

    reflection = payload.get("latest_reflection") or {}
    if reflection:
        lines.extend(["", "最近一次 reflection card 摘要："])
        if reflection.get("summary"):
            lines.append(f"- summary: {reflection.get('summary')}")
        top_tags = reflection.get("top_success_tags") or []
        if top_tags:
            lines.append(
                "- top_success_tags: "
                + ", ".join(f"{item['label']}({item['count']})" for item in top_tags[:3])
            )
        top_failures = reflection.get("top_failure_reasons") or []
        if top_failures:
            lines.append(
                "- top_failure_reasons: "
                + ", ".join(f"{item['label']}({item['count']})" for item in top_failures[:3])
            )
        next_focus = reflection.get("suggested_next_focus") or []
        if next_focus:
            lines.append("- suggested_next_focus:")
            lines.extend(f"  - {item}" for item in next_focus[:3])
    model_summary = payload.get("model_empirical_summary") or {}
    if model_summary:
        lines.extend(["", render_model_empirical_block(model_summary)])
    return "\n".join(lines).strip()
