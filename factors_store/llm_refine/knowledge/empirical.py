from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from ..core.archive import DEFAULT_ARCHIVE_DB
from .archive_queries import extract_expression_tags

_WINNER_STATUSES = {"research_winner", "winner"}
_KEEP_STATUSES = {"research_keep", "keep"}
_ACTIVE_STATUSES = _WINNER_STATUSES | _KEEP_STATUSES | {"proposed"}


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def _top_counter_labels(counter: Counter[str], *, limit: int = 3) -> list[str]:
    return [label for label, _ in counter.most_common(limit)]


def build_model_empirical_summary(
    *,
    db_path: str | Path = DEFAULT_ARCHIVE_DB,
    family: str,
    current_model_name: str = "",
    limit: int = 500,
) -> dict[str, Any]:
    sql = """
        SELECT
            c.source_model,
            c.status,
            c.expression,
            COALESCE(e.net_sharpe, NULL) AS net_sharpe,
            COALESCE(e.net_excess_ann_return, NULL) AS net_excess_ann_return,
            COALESCE(e.decision_reason, c.filter_reason, '') AS reason,
            COALESCE(e.evaluated_at, c.created_at) AS event_at
        FROM candidates c
        LEFT JOIN evaluations e
          ON c.candidate_id = e.candidate_id
        WHERE c.family = ?
          AND COALESCE(c.source_model, '') != ''
        ORDER BY event_at DESC, c.candidate_id DESC
        LIMIT ?
    """
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(sql, (family, int(limit))).fetchall()

    per_model: dict[str, dict[str, Any]] = {}
    for source_model, status, expression, net_sharpe, net_excess, reason, _event_at in rows:
        model = str(source_model or "").strip()
        if not model:
            continue
        entry = per_model.setdefault(
            model,
            {
                "model": model,
                "total": 0,
                "winner_count": 0,
                "keep_count": 0,
                "failure_count": 0,
                "winner_sharpes": [],
                "winner_excess": [],
                "success_tags": Counter(),
                "failure_reasons": Counter(),
            },
        )
        entry["total"] += 1
        status_text = str(status or "").strip()
        tags = extract_expression_tags(expression)
        if status_text in _WINNER_STATUSES:
            entry["winner_count"] += 1
            if net_sharpe is not None:
                entry["winner_sharpes"].append(float(net_sharpe))
            if net_excess is not None:
                entry["winner_excess"].append(float(net_excess))
            entry["success_tags"].update(tags)
        elif status_text in _KEEP_STATUSES:
            entry["keep_count"] += 1
            entry["success_tags"].update(tags)
        elif status_text not in _ACTIVE_STATUSES:
            entry["failure_count"] += 1
            if reason:
                entry["failure_reasons"].update([str(reason)])

    models: list[dict[str, Any]] = []
    for item in per_model.values():
        sharpes = item.pop("winner_sharpes")
        excess = item.pop("winner_excess")
        success_tags = item.pop("success_tags")
        failure_reasons = item.pop("failure_reasons")
        item["avg_winner_sharpe"] = sum(sharpes) / len(sharpes) if sharpes else None
        item["avg_winner_excess"] = sum(excess) / len(excess) if excess else None
        item["top_success_tags"] = _top_counter_labels(success_tags)
        item["top_failure_reasons"] = _top_counter_labels(failure_reasons)
        models.append(item)

    models.sort(
        key=lambda item: (
            int(item.get("winner_count", 0)),
            int(item.get("keep_count", 0)),
            float(item.get("avg_winner_sharpe") or float("-inf")),
            -int(item.get("failure_count", 0)),
        ),
        reverse=True,
    )

    current_model_stats = None
    if current_model_name:
        for item in models:
            if str(item.get("model", "")) == str(current_model_name):
                current_model_stats = item
                break

    return {
        "family": family,
        "current_model": current_model_name,
        "current_model_stats": current_model_stats,
        "top_models": models[:3],
        "models": models,
    }


def render_model_empirical_block(summary: dict[str, Any]) -> str:
    lines = [
        "模型经验摘要（同 family 历史统计，仅供参考，不代表硬约束）：",
    ]
    current = summary.get("current_model_stats")
    current_model = str(summary.get("current_model", "")).strip()
    if current_model:
        if current:
            lines.append(
                "- 当前模型 "
                f"`{current_model}`: "
                f"winner={current.get('winner_count', 0)}, "
                f"keep={current.get('keep_count', 0)}, "
                f"failure={current.get('failure_count', 0)}, "
                f"avg_winner_sharpe={_fmt_metric(current.get('avg_winner_sharpe'))}, "
                f"top_success_tags={','.join(current.get('top_success_tags', [])) or 'none'}"
            )
        else:
            lines.append(f"- 当前模型 `{current_model}`: 暂无该 family 的历史统计。")

    top_models = summary.get("top_models") or []
    if top_models:
        lines.append("- peer 模型近期相对更常成功的摘要：")
        for item in top_models[:2]:
            lines.append(
                "  - "
                f"{item.get('model', '')}: "
                f"winner={item.get('winner_count', 0)}, "
                f"keep={item.get('keep_count', 0)}, "
                f"avg_winner_sharpe={_fmt_metric(item.get('avg_winner_sharpe'))}, "
                f"top_success_tags={','.join(item.get('top_success_tags', [])) or 'none'}"
            )
    return "\n".join(lines).strip()
