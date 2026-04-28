'''Search metric normalization and historical profile statistics.

Builds percentile or scaled normalizers from archive rows, motif counts, and correlation-risk profiles.
'''

from __future__ import annotations

import bisect
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...knowledge.archive_queries import extract_economic_family_tags, extract_operator_skeleton, infer_mutation_class

_OPERATOR_PATTERN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\(")
_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CORR_VALUE_PATTERN = re.compile(r"corr[^=]*=\s*([0-9.]+)")


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return out


def _expression_depth(expression: str) -> int:
    text = str(expression or "").strip()
    if not text:
        return 0
    depth = 0
    max_depth = 0
    for char in text:
        if char == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif char == ")":
            depth = max(depth - 1, 0)
    return max_depth


def _expression_complexity(expression: str) -> float:
    text = str(expression or "").strip()
    if not text:
        return 0.0
    operator_count = len(_OPERATOR_PATTERN.findall(text))
    token_count = len(_TOKEN_PATTERN.findall(text))
    nesting = _expression_depth(text)
    return float(operator_count) + 0.08 * float(token_count) + 0.18 * float(nesting)


def _percentile_rank(sorted_values: list[float], value: float) -> float:
    if not sorted_values:
        return 0.5
    idx = bisect.bisect_right(sorted_values, value)
    return float(idx) / float(len(sorted_values))


@dataclass
class SearchNormalizer:
    strategy: str = "percentile"
    scope: str = "family_then_global"
    min_samples: int = 40
    metric_samples: dict[str, list[float]] = field(default_factory=dict)
    sample_counts: dict[str, int] = field(default_factory=dict)
    motif_profile_counts: dict[str, int] = field(default_factory=dict)
    corr_risk_by_profile: dict[str, float] = field(default_factory=dict)
    source_family: str = ""
    archive_db: str = ""

    def _samples(self, metric_name: str) -> list[float]:
        return list(self.metric_samples.get(metric_name, []))

    def normalize_signed(self, metric_name: str, raw_value: float) -> float | None:
        samples = self._samples(metric_name)
        if len(samples) < int(self.min_samples):
            return None
        pct = _percentile_rank(samples, raw_value)
        return 2.0 * pct - 1.0

    def normalize_positive(self, metric_name: str, raw_value: float) -> float | None:
        samples = self._samples(metric_name)
        if len(samples) < int(self.min_samples):
            return None
        pct = _percentile_rank(samples, max(raw_value, 0.0))
        return pct

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "scope": self.scope,
            "min_samples": int(self.min_samples),
            "sample_counts": dict(self.sample_counts),
            "motif_profile_counts": dict(self.motif_profile_counts),
            "corr_risk_by_profile": dict(self.corr_risk_by_profile),
            "source_family": self.source_family,
            "archive_db": self.archive_db,
        }


def _load_metric_rows(
    *,
    db_path: str | Path,
    family: str | None,
    limit: int,
) -> list[tuple[Any, ...]]:
    where_sql = ""
    params: list[Any] = []
    if family:
        where_sql = "WHERE c.family = ?"
        params.append(family)
    params.append(int(limit))
    sql = f"""
        SELECT
            e.quick_rank_ic_mean,
            e.quick_rank_icir,
            e.net_ann_return,
            e.net_excess_ann_return,
            e.net_sharpe,
            e.mean_turnover,
            c.expression,
            c.parent_candidate_id,
            p.expression,
            c.status,
            c.filter_reason,
            e.decision_reason
        FROM evaluations e
        JOIN candidates c
          ON c.candidate_id = e.candidate_id
        LEFT JOIN candidates p
          ON p.candidate_id = c.parent_candidate_id
        {where_sql}
        ORDER BY COALESCE(e.evaluated_at, c.created_at) DESC, c.candidate_id DESC
        LIMIT ?
    """
    with sqlite3.connect(str(db_path)) as conn:
        return conn.execute(sql, params).fetchall()


def _build_metric_samples(rows: list[tuple[Any, ...]]) -> dict[str, list[float]]:
    samples: dict[str, list[float]] = {
        "rank_ic_mean": [],
        "rank_icir": [],
        "net_ann_return": [],
        "net_excess_ann_return": [],
        "net_sharpe": [],
        "mean_turnover": [],
        "complexity": [],
        "expression_depth": [],
    }
    for row in rows:
        rank_ic, rank_icir, net_ann, net_excess, sharpe, turnover, expression = row[:7]
        mapping = {
            "rank_ic_mean": rank_ic,
            "rank_icir": rank_icir,
            "net_ann_return": net_ann,
            "net_excess_ann_return": net_excess,
            "net_sharpe": sharpe,
            "mean_turnover": turnover,
        }
        for name, value in mapping.items():
            parsed = _safe_float(value)
            if parsed is not None:
                samples[name].append(parsed)

        expr = str(expression or "").strip()
        if expr:
            samples["complexity"].append(_expression_complexity(expr))
            samples["expression_depth"].append(float(_expression_depth(expr)))
    for key in list(samples):
        samples[key] = sorted(samples[key])
    return samples


def _profile_key(expression: str, parent_expression: str) -> str:
    mutation_class = infer_mutation_class(expression, parent_expression)
    operator_skeleton = extract_operator_skeleton(expression)
    economic_tags = extract_economic_family_tags(expression)
    return "||".join(
        [
            mutation_class or "unknown",
            operator_skeleton or "literal",
            ",".join(economic_tags) or "generic",
        ]
    )


def _corr_risk_value(*, filter_reason: Any, decision_reason: Any, status: Any) -> float:
    text = " ".join(str(item or "") for item in (status, filter_reason, decision_reason)).lower()
    if "redundant" not in text and "corr" not in text:
        return 0.0
    match = _CORR_VALUE_PATTERN.search(text)
    if match:
        try:
            return min(max(float(match.group(1)), 0.0), 1.0)
        except Exception:
            return 0.5
    if "corr" in text or "redundant" in text:
        return 0.5
    return 0.0


def _build_profile_stats(rows: list[tuple[Any, ...]]) -> tuple[dict[str, int], dict[str, float]]:
    profile_counts: dict[str, int] = {}
    corr_risk_values: dict[str, list[float]] = {}
    for row in rows:
        expression = str(row[6] or "").strip()
        parent_expression = str(row[8] or "").strip()
        status = str(row[9] or "").strip()
        filter_reason = row[10]
        decision_reason = row[11]
        if not expression:
            continue
        key = _profile_key(expression, parent_expression)
        profile_counts[key] = profile_counts.get(key, 0) + 1
        risk = _corr_risk_value(
            filter_reason=filter_reason,
            decision_reason=decision_reason,
            status=status,
        )
        if risk > 0.0:
            corr_risk_values.setdefault(key, []).append(risk)
    corr_risk_by_profile = {
        key: sum(values) / float(len(values))
        for key, values in corr_risk_values.items()
        if values
    }
    return profile_counts, corr_risk_by_profile


def build_search_normalizer(
    *,
    db_path: str | Path,
    family: str,
    min_samples: int = 40,
    family_limit: int = 3000,
    global_limit: int = 6000,
) -> SearchNormalizer:
    db_path = str(Path(db_path).expanduser().resolve())
    family_rows = _load_metric_rows(db_path=db_path, family=family, limit=family_limit)
    global_rows = _load_metric_rows(db_path=db_path, family=None, limit=global_limit)

    family_samples = _build_metric_samples(family_rows)
    global_samples = _build_metric_samples(global_rows)
    family_profile_counts, family_corr_risk = _build_profile_stats(family_rows)

    merged_samples: dict[str, list[float]] = {}
    sample_counts: dict[str, int] = {}
    for metric_name, family_values in family_samples.items():
        values = list(family_values)
        if len(values) < int(min_samples):
            needed = int(min_samples) - len(values)
            extra = list(global_samples.get(metric_name, []))
            values.extend(extra[: max(needed, 0)])
            values = sorted(values)
        merged_samples[metric_name] = values
        sample_counts[metric_name] = len(values)

    return SearchNormalizer(
        strategy="percentile",
        scope="family_then_global",
        min_samples=int(min_samples),
        metric_samples=merged_samples,
        sample_counts=sample_counts,
        motif_profile_counts=family_profile_counts,
        corr_risk_by_profile=family_corr_risk,
        source_family=family,
        archive_db=db_path,
    )
