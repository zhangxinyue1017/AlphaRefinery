from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..parsing.expression_engine import guess_required_fields, normalize_expression
from ..core.models import RefinementCandidate, SeedFamily
from ..parsing.parser import expression_dedup_key

_CALL_PATTERN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_ADV_PATTERN = re.compile(r"adv\d+", flags=re.IGNORECASE)
_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z_])(\d+(?:\.\d+)?)(?![A-Za-z_])")


def structure_signature(expression: str) -> dict[str, Any]:
    normalized = normalize_expression(expression).replace(" ", "").lower()
    fields = tuple(sorted(guess_required_fields(expression)))
    operators = tuple(op.lower() for op in _CALL_PATTERN.findall(normalized))
    windows = tuple(sorted(int(float(item)) for item in _NUMBER_PATTERN.findall(normalized) if float(item).is_integer()))
    skeleton = normalized
    for field in sorted(fields, key=len, reverse=True):
        skeleton = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(field)}(?![A-Za-z0-9_])", "<F>", skeleton)
    skeleton = _ADV_PATTERN.sub("<ADV>", skeleton)
    skeleton = _NUMBER_PATTERN.sub("<N>", skeleton)
    return {
        "normalized": normalized,
        "dedup_key": expression_dedup_key(expression),
        "fields": fields,
        "operators": operators,
        "windows": windows,
        "top_operator": operators[0] if operators else "",
        "skeleton": skeleton,
    }


def _windows_are_close(left: Sequence[int], right: Sequence[int], *, max_delta: int = 5) -> bool:
    if not left or not right or len(left) != len(right):
        return False
    return all(abs(int(a) - int(b)) <= max_delta for a, b in zip(sorted(left), sorted(right)))


def _structure_match_reason(current: dict[str, Any], previous: dict[str, Any]) -> str | None:
    if current["skeleton"] == previous["skeleton"] and current["fields"] == previous["fields"]:
        return "same normalized structure skeleton"
    if (
        current["operators"] == previous["operators"]
        and current["fields"] == previous["fields"]
        and _windows_are_close(current["windows"], previous["windows"])
    ):
        return "same operator/field structure; likely only a window-level variant"
    return None


def filter_structurally_redundant(
    candidates: Sequence[RefinementCandidate],
) -> tuple[tuple[RefinementCandidate, ...], tuple[dict[str, Any], ...]]:
    kept: list[RefinementCandidate] = []
    kept_signatures: dict[str, dict[str, Any]] = {}
    dropped: list[dict[str, Any]] = []
    for candidate in candidates:
        current_sig = structure_signature(candidate.expression)
        matched_name = ""
        matched_candidate_id = ""
        matched_reason = None
        for incumbent in kept:
            previous_sig = kept_signatures[incumbent.candidate_id or incumbent.name]
            reason = _structure_match_reason(current_sig, previous_sig)
            if reason:
                matched_name = incumbent.name
                matched_candidate_id = incumbent.candidate_id
                matched_reason = reason
                break
        if matched_reason:
            dropped.append(
                {
                    "candidate": candidate,
                    "filter_stage": "structure",
                    "filter_reason": f"{matched_reason}; overlaps with {matched_name}",
                    "matched_candidate_id": matched_candidate_id,
                    "matched_name": matched_name,
                    "signature": current_sig,
                }
            )
            continue
        kept.append(candidate)
        kept_signatures[candidate.candidate_id or candidate.name] = current_sig
    return tuple(kept), tuple(dropped)


def factor_series_correlation(left: pd.Series, right: pd.Series, *, max_points: int = 200_000) -> float:
    joined = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    if joined.empty or len(joined) < 50:
        return float("nan")
    if len(joined) > max_points:
        joined = joined.sample(max_points, random_state=42)
    corr = joined["left"].corr(joined["right"])
    try:
        return float(corr)
    except Exception:
        return float("nan")


def structure_filter_markdown(
    *,
    family: SeedFamily,
    kept_candidates: Sequence[RefinementCandidate],
    dropped_candidates: Sequence[dict[str, Any]],
) -> str:
    lines = [
        "# Structure Filter Report",
        "",
        f"- family: `{family.family}`",
        f"- canonical_seed: `{family.canonical_seed}`",
        f"- kept_count: `{len(kept_candidates)}`",
        f"- dropped_count: `{len(dropped_candidates)}`",
        "",
        "## Kept Candidates",
    ]
    if kept_candidates:
        lines.extend(f"- `{candidate.name}`: `{candidate.expression}`" for candidate in kept_candidates)
    else:
        lines.append("- (none)")
    lines.extend(["", "## Dropped Candidates"])
    if not dropped_candidates:
        lines.append("- (none)")
    else:
        for item in dropped_candidates:
            candidate = item["candidate"]
            lines.extend(
                [
                    f"### {candidate.name}",
                    f"- expression: `{candidate.expression}`",
                    f"- reason: {item.get('filter_reason', '')}",
                    f"- matched_name: {item.get('matched_name', '') or '(none)'}",
                    "",
                ]
            )
    return "\n".join(lines).strip() + "\n"
