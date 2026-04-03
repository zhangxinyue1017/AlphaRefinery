from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ..core.archive import DEFAULT_ARCHIVE_DB, utc_now_iso
from ..core.models import SeedFamily
from .archive_queries import (
    load_recent_failures,
    load_recent_keeps,
    load_recent_winners,
    load_run_candidates,
)


def _pick_best(records: list[dict[str, Any]], statuses: tuple[str, ...]) -> dict[str, Any] | None:
    pool = [item for item in records if str(item.get("status", "")) in statuses]
    if not pool:
        return None

    def _score(item: dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            float(item.get("net_sharpe") or float("-inf")),
            float(item.get("net_excess_ann_return") or float("-inf")),
            float(item.get("net_ann_return") or float("-inf")),
            float(item.get("quick_rank_icir") or float("-inf")),
        )

    return max(pool, key=_score)


def _top_counter(records: list[dict[str, Any]], key: str, *, limit: int = 3) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for record in records:
        value = record.get(key)
        if isinstance(value, (tuple, list)):
            counter.update(str(item) for item in value if str(item).strip())
        elif value:
            counter.update([str(value)])
    return [{"label": label, "count": count} for label, count in counter.most_common(limit)]


def _model_contribution(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    relevant = [
        record
        for record in records
        if str(record.get("status", "")) in {"research_winner", "winner", "research_keep", "keep"}
    ]
    if not relevant:
        return []
    counter = Counter(str(record.get("source_model", "")).strip() or "(unknown)" for record in relevant)
    return [{"model": model, "count": count} for model, count in counter.most_common()]


def _suggest_next_focus(
    *,
    current_records: list[dict[str, Any]],
    recent_failures: list[dict[str, Any]],
    recent_winners: list[dict[str, Any]],
) -> list[str]:
    suggestions: list[str] = []
    failure_text = " ".join(str(item.get("reason", "")) for item in [*current_records, *recent_failures]).lower()
    winner_tags = Counter(
        tag
        for item in recent_winners
        for tag in item.get("expression_tags", ())
    )
    if "turnover" in failure_text:
        suggestions.append("优先考虑增加平滑、分母稳定化或更低换手的确认项。")
    if "redundant" in failure_text or "overlap" in failure_text:
        suggestions.append("优先做 decorrelating edit，避免继续沿完全相同的 motif 小修小补。")
    if winner_tags.get("amount_std_normalization", 0) >= 1:
        suggestions.append("`amount/std` 主线仍然有效，可继续尝试不同窗口、EMA 分母或相对量版本。")
    if winner_tags.get("turnover_std_normalization", 0) >= 1:
        suggestions.append("`turnover/std` 支线仍可保留为次优探索方向。")
    if not suggestions:
        suggestions.append("延续当前最强主线，同时保留一个结构上更远的 decorrelating 分支。")
    return suggestions[:3]


def build_reflection_card(
    *,
    db_path: str | Path = DEFAULT_ARCHIVE_DB,
    family: SeedFamily,
    run_id: str,
    selected_parent: dict[str, Any] | None = None,
    recent_limit: int = 3,
) -> dict[str, Any]:
    current_records = load_run_candidates(db_path=db_path, run_id=run_id)
    current_winner = _pick_best(current_records, ("research_winner", "winner"))
    current_keeps = [
        item for item in current_records if str(item.get("status", "")) in {"research_keep", "keep"}
    ][: int(recent_limit)]
    current_failures = [
        item
        for item in current_records
        if str(item.get("status", "")) not in {"research_winner", "winner", "research_keep", "keep", "proposed"}
    ][: int(recent_limit)]

    recent_winners = load_recent_winners(
        db_path=db_path,
        family=family.family,
        limit=int(recent_limit),
        exclude_run_id=run_id,
    )
    recent_keeps = load_recent_keeps(
        db_path=db_path,
        family=family.family,
        limit=int(recent_limit),
        exclude_run_id=run_id,
    )
    recent_failures = load_recent_failures(
        db_path=db_path,
        family=family.family,
        limit=int(recent_limit),
        exclude_run_id=run_id,
    )

    success_tag_records = [item for item in [current_winner, *current_keeps, *recent_winners] if item]
    failure_tag_records = [*current_failures, *recent_failures]

    card = {
        "family": family.family,
        "run_id": run_id,
        "generated_at": utc_now_iso(),
        "selected_parent": selected_parent or {},
        "current_winner": current_winner,
        "current_keeps": current_keeps,
        "current_failures": current_failures,
        "recent_winners": recent_winners,
        "recent_keeps": recent_keeps,
        "recent_failures": recent_failures,
        "top_success_tags": _top_counter(success_tag_records, "expression_tags"),
        "top_failure_reasons": _top_counter(failure_tag_records, "reason"),
        "model_contribution": _model_contribution(current_records),
        "suggested_next_focus": _suggest_next_focus(
            current_records=current_records,
            recent_failures=recent_failures,
            recent_winners=recent_winners,
        ),
    }
    card["summary"] = (
        f"{family.family}: "
        f"winner={str((current_winner or {}).get('factor_name', '(none)'))}, "
        f"keeps={len(current_keeps)}, failures={len(current_failures)}"
    )
    return card


def reflection_markdown(card: dict[str, Any]) -> str:
    lines = [
        "# Reflection Card",
        "",
        f"- family: `{card.get('family', '')}`",
        f"- run_id: `{card.get('run_id', '')}`",
        f"- generated_at: `{card.get('generated_at', '')}`",
        f"- summary: {card.get('summary', '')}",
        "",
        "## Selected Parent",
    ]
    parent = card.get("selected_parent") or {}
    if parent:
        lines.extend(
            [
                f"- factor_name: `{parent.get('factor_name', '')}`",
                f"- candidate_id: `{parent.get('candidate_id', '')}`",
                f"- status: `{parent.get('status', '')}`",
            ]
        )
    else:
        lines.append("- (none)")

    lines.extend(["", "## Current Winner"])
    winner = card.get("current_winner") or {}
    if winner:
        lines.extend(
            [
                f"- factor_name: `{winner.get('factor_name', '')}`",
                f"- model: `{winner.get('source_model', '')}`",
                f"- expression: `{winner.get('expression', '')}`",
                f"- RankICIR: `{winner.get('quick_rank_icir')}`",
                f"- NetSharpe: `{winner.get('net_sharpe')}`",
                f"- NetExcess: `{winner.get('net_excess_ann_return')}`",
                f"- Turnover: `{winner.get('mean_turnover')}`",
            ]
        )
    else:
        lines.append("- (none)")

    for title, key in (
        ("Current Keeps", "current_keeps"),
        ("Current Failures", "current_failures"),
        ("Recent Winners", "recent_winners"),
        ("Recent Keeps", "recent_keeps"),
        ("Recent Failures", "recent_failures"),
    ):
        lines.extend(["", f"## {title}"])
        items = card.get(key) or []
        if not items:
            lines.append("- (none)")
            continue
        for item in items:
            lines.append(
                f"- `{item.get('factor_name', '')}` "
                f"[{item.get('status', '')}] "
                f"model={item.get('source_model', '')} "
                f"reason={item.get('reason', '') or '(none)'}"
            )

    lines.extend(["", "## Top Success Tags"])
    success_tags = card.get("top_success_tags") or []
    if success_tags:
        lines.extend(f"- {item['label']}: {item['count']}" for item in success_tags)
    else:
        lines.append("- (none)")

    lines.extend(["", "## Top Failure Reasons"])
    failure_reasons = card.get("top_failure_reasons") or []
    if failure_reasons:
        lines.extend(f"- {item['label']}: {item['count']}" for item in failure_reasons)
    else:
        lines.append("- (none)")

    lines.extend(["", "## Model Contribution"])
    contribution = card.get("model_contribution") or []
    if contribution:
        lines.extend(f"- {item['model']}: {item['count']}" for item in contribution)
    else:
        lines.append("- (none)")

    lines.extend(["", "## Suggested Next Focus"])
    focus = card.get("suggested_next_focus") or []
    if focus:
        lines.extend(f"- {item}" for item in focus)
    else:
        lines.append("- (none)")
    return "\n".join(lines).strip() + "\n"
