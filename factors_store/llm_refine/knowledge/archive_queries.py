from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from ..core.archive import DEFAULT_ARCHIVE_DB

_WINNER_STATUSES = ("research_winner", "winner")
_KEEP_STATUSES = ("research_keep", "keep")
_ACTIVE_STATUSES = (*_WINNER_STATUSES, *_KEEP_STATUSES, "proposed")
_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TAG_TOKENS = (
    "amount",
    "volume",
    "turnover",
    "ema",
    "decay_linear",
    "ts_mean",
    "ts_std",
    "ts_rank",
    "cs_rank",
    "corr",
    "ts_corr",
    "cov",
    "ts_cov",
    "rel_amount",
    "rel_volume",
    "bucket_sum",
    "where",
    "if_then_else",
)
_RANK_TOKENS = {"ts_rank", "cs_rank", "rank"}
_SMOOTHING_TOKENS = {"ema", "decay_linear"}
_NORMALIZATION_TOKENS = {"ts_std", "std"}
_CONDITIONAL_TOKENS = {"where", "if_then_else", "bucket_sum"}
_RELATIVE_TOKENS = {"rel_amount", "rel_volume"}
_LIQUIDITY_FIELD_TOKENS = {"amount", "volume", "turnover", "rel_amount", "rel_volume"}
_PRICE_FIELD_TOKENS = {"open", "close", "high", "low", "vwap", "returns"}


def _parse_json_list(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = [value]
        if isinstance(parsed, list):
            return tuple(str(item) for item in parsed if str(item).strip())
        return (str(parsed),)
    if isinstance(value, list):
        return tuple(str(item) for item in value if str(item).strip())
    return (str(value),)


def extract_expression_tags(expression: str) -> tuple[str, ...]:
    tokens = {token.lower() for token in _TOKEN_PATTERN.findall(str(expression or ""))}
    tags = [token for token in _TAG_TOKENS if token in tokens]
    if "ts_std" in tags and "amount" in tags:
        tags.append("amount_std_normalization")
    if "ts_std" in tags and "turnover" in tags:
        tags.append("turnover_std_normalization")
    if "ema" in tags:
        tags.append("smoothing")
    return tuple(dict.fromkeys(tags))


def extract_operator_skeleton(expression: str) -> str:
    operators = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\(", str(expression or ""))
    if not operators:
        return "literal"
    normalized: list[str] = []
    for op in operators[:5]:
        low = op.lower()
        if low.startswith("ts_"):
            normalized.append(f"ts:{low[3:]}")
        elif low.startswith("cs_"):
            normalized.append(f"cs:{low[3:]}")
        else:
            normalized.append(low)
    return "|".join(normalized)


def extract_economic_family_tags(expression: str) -> tuple[str, ...]:
    tokens = {token.lower() for token in _TOKEN_PATTERN.findall(str(expression or ""))}
    tags: list[str] = []
    if {"amount", "volume", "turnover"} & tokens:
        tags.append("volume_liquidity")
    if {"returns", "open", "close", "high", "low", "vwap"} & tokens:
        tags.append("price_structure")
    if {"ema", "decay_linear"} & tokens:
        tags.append("smoothing")
    if {"ts_std", "std"} & tokens:
        tags.append("volatility_normalization")
    if {"ts_rank", "cs_rank", "rank"} & tokens:
        tags.append("ranking")
    if {"rel_amount", "rel_volume"} & tokens:
        tags.append("relative_participation")
    if {"where", "if_then_else", "bucket_sum"} & tokens:
        tags.append("conditional_state")
    return tuple(dict.fromkeys(tags)) or ("generic",)


def extract_window_tokens(expression: str) -> set[str]:
    return set(re.findall(r"\b(?:3|5|10|14|15|20|28|40|60|100|120|180|250|375)\b", str(expression or "")))


def _classify_expression_shape(expression: str) -> str:
    text = str(expression or "").lower()
    tokens = {token.lower() for token in _TOKEN_PATTERN.findall(text)}
    windows = len(extract_window_tokens(text))
    if _CONDITIONAL_TOKENS & tokens:
        return "conditionalization"
    if _RANK_TOKENS & tokens:
        return "rank_wrapper"
    if _SMOOTHING_TOKENS & tokens:
        return "smoothing_insertion"
    if _NORMALIZATION_TOKENS & tokens and _LIQUIDITY_FIELD_TOKENS & tokens:
        return "normalization_insertion"
    if _RELATIVE_TOKENS & tokens:
        return "relative_reweighting"
    if windows >= 3:
        return "window_rebalancing"
    return "structural_rewrite"


def infer_mutation_class(expression: str, parent_expression: str = "") -> str:
    child = str(expression or "").strip()
    parent = str(parent_expression or "").strip()
    if not child:
        return "structural_rewrite"
    if not parent:
        return _classify_expression_shape(child)
    if child == parent:
        return "identity"

    child_tokens = {token.lower() for token in _TOKEN_PATTERN.findall(child)}
    parent_tokens = {token.lower() for token in _TOKEN_PATTERN.findall(parent)}
    child_ops = set(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\(", child))
    parent_ops = set(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\(", parent))
    child_windows = extract_window_tokens(child)
    parent_windows = extract_window_tokens(parent)
    child_fields = child_tokens & (_LIQUIDITY_FIELD_TOKENS | _PRICE_FIELD_TOKENS)
    parent_fields = parent_tokens & (_LIQUIDITY_FIELD_TOKENS | _PRICE_FIELD_TOKENS)

    new_tokens = child_tokens - parent_tokens
    removed_tokens = parent_tokens - child_tokens
    new_ops = child_ops - parent_ops
    removed_ops = parent_ops - child_ops

    if (_CONDITIONAL_TOKENS & (new_tokens | new_ops)) and not (_CONDITIONAL_TOKENS & parent_tokens):
        return "conditional_gate_added"
    if (_RANK_TOKENS & (new_tokens | new_ops)) and not (_RANK_TOKENS & parent_tokens):
        return "rank_wrapper_added"
    if (_SMOOTHING_TOKENS & (new_tokens | new_ops)) and not (_SMOOTHING_TOKENS & parent_tokens):
        return "smoothing_added"
    if (_NORMALIZATION_TOKENS & (new_tokens | new_ops)) and not (_NORMALIZATION_TOKENS & parent_tokens):
        return "normalization_added"
    if (_RELATIVE_TOKENS & (new_tokens | removed_tokens)) or (
        child_fields != parent_fields and (_LIQUIDITY_FIELD_TOKENS & (child_fields | parent_fields))
    ):
        return "liquidity_proxy_swap"
    if child_ops == parent_ops and child_fields == parent_fields and child_windows != parent_windows:
        return "window_tuning"
    if ("div" in child_ops and "div" in parent_ops) and (
        (_NORMALIZATION_TOKENS | _SMOOTHING_TOKENS | _RELATIVE_TOKENS) & (new_tokens | removed_tokens | new_ops | removed_ops)
    ):
        return "denominator_rewrite"
    if child_fields != parent_fields:
        return "field_proxy_swap"
    if child_ops != parent_ops and child_windows == parent_windows:
        return "operator_refactor"
    return "structural_rewrite"


def _fetch_candidates(
    *,
    db_path: str | Path = DEFAULT_ARCHIVE_DB,
    family: str,
    limit: int,
    exclude_run_id: str = "",
    statuses: tuple[str, ...] = (),
    exclude_statuses: tuple[str, ...] = (),
    dedup_by_expression: bool = True,
    run_id: str = "",
) -> list[dict[str, Any]]:
    params: list[Any] = []
    clauses: list[str] = []

    if run_id:
        clauses.append("c.run_id = ?")
        params.append(run_id)
    else:
        clauses.append("c.family = ?")
        params.append(family)

    if exclude_run_id:
        clauses.append("c.run_id != ?")
        params.append(exclude_run_id)

    if statuses:
        placeholders = ",".join("?" for _ in statuses)
        clauses.append(f"c.status IN ({placeholders})")
        params.extend(statuses)

    if exclude_statuses:
        placeholders = ",".join("?" for _ in exclude_statuses)
        clauses.append(f"c.status NOT IN ({placeholders})")
        params.extend(exclude_statuses)

    where_sql = " AND ".join(clauses)
    sql = f"""
        SELECT
            c.candidate_id,
            c.run_id,
            c.family,
            c.round_id,
            c.parent_candidate_id,
            p.factor_name AS parent_factor_name,
            p.expression AS parent_expression,
            c.factor_name,
            c.expression,
            c.expression_hash,
            c.candidate_role,
            c.source_model,
            c.source_provider,
            c.validation_warnings,
            c.filter_stage,
            c.filter_reason,
            c.status,
            e.quick_rank_ic_mean,
            e.quick_rank_icir,
            e.net_ann_return,
            e.net_excess_ann_return,
            e.net_sharpe,
            e.mean_turnover,
            e.decision_reason,
            COALESCE(e.evaluated_at, c.created_at) AS event_at
        FROM candidates c
        LEFT JOIN candidates p
          ON p.candidate_id = c.parent_candidate_id
        LEFT JOIN evaluations e
          ON c.candidate_id = e.candidate_id
        WHERE {where_sql}
        ORDER BY event_at DESC, c.candidate_id DESC
    """

    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(sql, params).fetchall()

    out: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for row in rows:
        record = {
            "candidate_id": row[0],
            "run_id": row[1],
            "family": row[2],
            "round_id": row[3],
            "parent_candidate_id": row[4],
            "parent_factor_name": row[5] or "",
            "parent_expression": row[6] or "",
            "factor_name": row[7],
            "expression": row[8],
            "expression_hash": row[9],
            "candidate_role": row[10] or "",
            "source_model": row[11] or "",
            "source_provider": row[12] or "",
            "validation_warnings": _parse_json_list(row[13]),
            "filter_stage": row[14] or "",
            "filter_reason": row[15] or "",
            "status": row[16] or "",
            "quick_rank_ic_mean": row[17],
            "quick_rank_icir": row[18],
            "net_ann_return": row[19],
            "net_excess_ann_return": row[20],
            "net_sharpe": row[21],
            "mean_turnover": row[22],
            "decision_reason": row[23] or "",
            "event_at": row[24] or "",
        }
        record["reason"] = record["decision_reason"] or record["filter_reason"]
        record["expression_tags"] = extract_expression_tags(record["expression"])
        record["operator_skeleton"] = extract_operator_skeleton(record["expression"])
        record["economic_family_tags"] = extract_economic_family_tags(record["expression"])
        record["mutation_class"] = infer_mutation_class(record["expression"], record.get("parent_expression", ""))
        if dedup_by_expression:
            expression_hash = str(record.get("expression_hash", "")).strip()
            if expression_hash and expression_hash in seen_hashes:
                continue
            if expression_hash:
                seen_hashes.add(expression_hash)
        out.append(record)
        if len(out) >= int(limit):
            break
    return out


def load_candidate_lineage(
    *,
    db_path: str | Path = DEFAULT_ARCHIVE_DB,
    candidate_id: str,
    max_hops: int = 3,
) -> list[dict[str, Any]]:
    if not str(candidate_id).strip():
        return []
    sql = """
        SELECT
            c.candidate_id,
            c.parent_candidate_id,
            c.family,
            c.round_id,
            p.factor_name AS parent_factor_name,
            p.expression AS parent_expression,
            c.factor_name,
            c.expression,
            c.candidate_role,
            c.source_model,
            c.status,
            e.quick_rank_ic_mean,
            e.quick_rank_icir,
            e.net_ann_return,
            e.net_excess_ann_return,
            e.net_sharpe,
            e.mean_turnover,
            COALESCE(e.evaluated_at, c.created_at) AS event_at
        FROM candidates c
        LEFT JOIN candidates p
          ON p.candidate_id = c.parent_candidate_id
        LEFT JOIN evaluations e
          ON c.candidate_id = e.candidate_id
        WHERE c.candidate_id = ?
        LIMIT 1
    """
    lineage: list[dict[str, Any]] = []
    current_id = str(candidate_id).strip()
    seen: set[str] = set()
    with sqlite3.connect(str(db_path)) as conn:
        for _ in range(max(int(max_hops), 1)):
            if not current_id or current_id in seen:
                break
            seen.add(current_id)
            row = conn.execute(sql, (current_id,)).fetchone()
            if row is None:
                break
            record = {
                "candidate_id": row[0],
                "parent_candidate_id": row[1] or "",
                "family": row[2] or "",
                "round_id": row[3] or 0,
                "parent_factor_name": row[4] or "",
                "parent_expression": row[5] or "",
                "factor_name": row[6] or "",
                "expression": row[7] or "",
                "candidate_role": row[8] or "",
                "source_model": row[9] or "",
                "status": row[10] or "",
                "quick_rank_ic_mean": row[11],
                "quick_rank_icir": row[12],
                "net_ann_return": row[13],
                "net_excess_ann_return": row[14],
                "net_sharpe": row[15],
                "mean_turnover": row[16],
                "event_at": row[17] or "",
            }
            record["expression_tags"] = extract_expression_tags(record["expression"])
            record["operator_skeleton"] = extract_operator_skeleton(record["expression"])
            record["economic_family_tags"] = extract_economic_family_tags(record["expression"])
            record["mutation_class"] = infer_mutation_class(record["expression"], record.get("parent_expression", ""))
            lineage.append(record)
            current_id = str(record["parent_candidate_id"]).strip()
    lineage.reverse()
    return lineage


def load_run_candidates(
    *,
    db_path: str | Path = DEFAULT_ARCHIVE_DB,
    run_id: str,
    limit: int = 200,
) -> list[dict[str, Any]]:
    return _fetch_candidates(
        db_path=db_path,
        family="",
        run_id=run_id,
        limit=limit,
        dedup_by_expression=False,
    )


def load_recent_winners(
    *,
    db_path: str | Path = DEFAULT_ARCHIVE_DB,
    family: str,
    limit: int = 3,
    exclude_run_id: str = "",
) -> list[dict[str, Any]]:
    return _fetch_candidates(
        db_path=db_path,
        family=family,
        limit=limit,
        exclude_run_id=exclude_run_id,
        statuses=_WINNER_STATUSES,
    )


def load_recent_keeps(
    *,
    db_path: str | Path = DEFAULT_ARCHIVE_DB,
    family: str,
    limit: int = 3,
    exclude_run_id: str = "",
) -> list[dict[str, Any]]:
    return _fetch_candidates(
        db_path=db_path,
        family=family,
        limit=limit,
        exclude_run_id=exclude_run_id,
        statuses=_KEEP_STATUSES,
    )


def load_recent_failures(
    *,
    db_path: str | Path = DEFAULT_ARCHIVE_DB,
    family: str,
    limit: int = 3,
    exclude_run_id: str = "",
) -> list[dict[str, Any]]:
    return _fetch_candidates(
        db_path=db_path,
        family=family,
        limit=limit,
        exclude_run_id=exclude_run_id,
        exclude_statuses=_ACTIVE_STATUSES,
    )
