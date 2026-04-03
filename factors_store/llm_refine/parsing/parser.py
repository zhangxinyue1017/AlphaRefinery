from __future__ import annotations

import ast
import hashlib
import json
import re
from typing import Any

from .expression_engine import normalize_expression
from .expression_repair import repair_expression
from ..core.models import LLMProposal, RefinementCandidate, SeedFamily

_FUNCTION_ALIASES = {
    "corr": "ts_corr",
    "cov": "ts_cov",
    "greater": "rowmax",
    "less": "rowmin",
    "mean": "ts_mean",
    "rank": "cs_rank",
    "std": "ts_std",
    "sum": "ts_sum",
}
_COMMUTATIVE_CALLS = {"rowmax", "rowmin"}
_ADV_PATTERN = re.compile(r"adv(\d+)$", flags=re.IGNORECASE)


def _extract_json_block(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    match = re.search(r"\{.*\}", stripped, flags=re.S)
    if not match:
        raise ValueError("no JSON object found in provider response")
    return match.group(0)


_MISSING_COMMA_BEFORE_KEY = re.compile(
    r'((?:"(?:\\.|[^"\\])*"|true|false|null|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\}|\]))(\s*\n\s*")',
    flags=re.S,
)


def _repair_common_json_issues(text: str) -> str:
    repaired = text.strip()
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    repaired = re.sub(r"(\})(\s*\{)", r"\1,\2", repaired)
    repaired = re.sub(r"(\])(\s*\{)", r"\1,\2", repaired)
    repaired = re.sub(r'(")(\s*\n\s*")', r'\1,\2', repaired)
    prev = None
    while prev != repaired:
        prev = repaired
        repaired = _MISSING_COMMA_BEFORE_KEY.sub(r"\1,\2", repaired)
    return repaired


def _load_json_payload(raw_response: str) -> dict[str, Any]:
    block = _extract_json_block(raw_response)
    try:
        payload = json.loads(block)
    except json.JSONDecodeError as exc:
        repaired = _repair_common_json_issues(block)
        if repaired == block:
            raise ValueError(f"provider response JSON decode failed: {exc}") from exc
        try:
            payload = json.loads(repaired)
        except json.JSONDecodeError as repaired_exc:
            raise ValueError(
                "provider response JSON decode failed after common-issue repair: "
                f"{repaired_exc}"
            ) from repaired_exc
    if not isinstance(payload, dict):
        raise ValueError("provider response root payload must be a JSON object")
    return payload


def _coerce_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _slugify_name(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", str(text).strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug


def _fallback_candidate_name(item: dict[str, Any], *, index: int) -> str:
    for key in ("name", "factor_name", "candidate_name", "id", "title", "label"):
        value = str(item.get(key, "")).strip()
        if value:
            return _slugify_name(value) or f"candidate_{index:02d}"

    expression = str(item.get("expression", "")).strip()
    if expression:
        normalized = expression_dedup_key(expression) or normalize_expression(expression)
        digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()[:8]
        tokens = re.findall(r"[A-Za-z_]+", normalized)
        meaningful = [token.lower() for token in tokens if len(token) > 2 and token.lower() not in {"add", "sub", "mul", "div", "neg"}]
        stem = "_".join(meaningful[:4])
        stem = _slugify_name(stem) if stem else "candidate"
        return f"{stem}_{digest}"

    return f"candidate_{index:02d}"


def _extract_expression(item: dict[str, Any]) -> str:
    for key in ("expression", "expr", "formula", "candidate_formula"):
        value = str(item.get(key, "")).strip()
        if value:
            return value
    raise KeyError("expression")


def _format_constant(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return format(number, ".12g")
    return str(value).strip().lower()


def _extract_numeric(node: ast.AST) -> int | float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _extract_numeric(node.operand)
        if inner is not None:
            return -inner
    return None


def _canonical_call_name(name: str, args: list[ast.AST]) -> str:
    lower = name.lower()
    if lower == "max" and len(args) == 2 and _extract_numeric(args[1]) is not None:
        return "ts_max"
    if lower == "min" and len(args) == 2 and _extract_numeric(args[1]) is not None:
        return "ts_min"
    return _FUNCTION_ALIASES.get(lower, lower)


def _canonicalize_keywords(keywords: list[ast.keyword]) -> list[str]:
    rendered: list[str] = []
    for keyword in keywords:
        if keyword.arg is None:
            continue
        rendered.append(f"{keyword.arg.lower()}={_canonicalize_expression_node(keyword.value)}")
    return sorted(rendered)


def _render_signed(sign: int, body: str) -> str:
    return body if sign >= 0 else f"-{body}"


def _canonicalize_expression_node(node: ast.AST) -> str:
    sign, body = _canonicalize_with_sign(node)
    return _render_signed(sign, body)


def _canonicalize_with_sign(node: ast.AST) -> tuple[int, str]:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        sign, body = _canonicalize_with_sign(node.operand)
        return -sign, body
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id.lower() == "neg" and len(node.args) == 1:
        sign, body = _canonicalize_with_sign(node.args[0])
        return -sign, body
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left_sign, left_body = _canonicalize_with_sign(node.left)
        right_sign, right_body = _canonicalize_with_sign(node.right)
        ordered = sorted((left_body, right_body))
        return left_sign * right_sign, f"({ordered[0]}*{ordered[1]})"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left_sign, left_body = _canonicalize_with_sign(node.left)
        right_sign, right_body = _canonicalize_with_sign(node.right)
        return left_sign * right_sign, f"({left_body}/{right_body})"
    return 1, _canonicalize_without_sign(node)


def _canonicalize_without_sign(node: ast.AST) -> str:
    if isinstance(node, ast.Constant):
        return _format_constant(node.value)
    if isinstance(node, ast.Name):
        token = node.id.strip().lower()
        match = _ADV_PATTERN.fullmatch(token)
        if match:
            return f"ts_mean(volume,{int(match.group(1))})"
        return token
    if isinstance(node, ast.BinOp):
        left = _canonicalize_expression_node(node.left)
        right = _canonicalize_expression_node(node.right)
        if isinstance(node.op, ast.Add):
            return f"({left}+{right})"
        if isinstance(node.op, ast.Sub):
            return f"({left}-{right})"
        if isinstance(node.op, ast.Pow):
            return f"({left}**{right})"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        lower = node.func.id.lower()
        if lower == "neg" and len(node.args) == 1:
            return _canonicalize_expression_node(node.args[0])
        if lower == "add" and len(node.args) == 2:
            return _canonicalize_without_sign(ast.BinOp(left=node.args[0], op=ast.Add(), right=node.args[1]))
        if lower == "sub" and len(node.args) == 2:
            return _canonicalize_without_sign(ast.BinOp(left=node.args[0], op=ast.Sub(), right=node.args[1]))
        if lower == "mul" and len(node.args) == 2:
            sign, body = _canonicalize_with_sign(ast.BinOp(left=node.args[0], op=ast.Mult(), right=node.args[1]))
            return _render_signed(sign, body)
        if lower == "div" and len(node.args) == 2:
            sign, body = _canonicalize_with_sign(ast.BinOp(left=node.args[0], op=ast.Div(), right=node.args[1]))
            return _render_signed(sign, body)
        if lower == "rel_volume" and len(node.args) == 1:
            window = _canonicalize_expression_node(node.args[0])
            return f"(volume/ts_mean(volume,{window}))"
        if lower == "rel_amount" and len(node.args) == 1:
            window = _canonicalize_expression_node(node.args[0])
            return f"(amount/ts_mean(amount,{window}))"
        call_name = _canonical_call_name(node.func.id, node.args)
        args = [_canonicalize_expression_node(arg) for arg in node.args]
        kwargs = _canonicalize_keywords(node.keywords)
        if call_name in _COMMUTATIVE_CALLS and len(args) == 2:
            args = sorted(args)
        return f"{call_name}({','.join([*args, *kwargs])})"
    return normalize_expression(ast.unparse(node)).replace(" ", "").lower()


def _normalize_candidate_role(
    raw_role: Any,
    *,
    index: int,
    allowed_candidate_roles: tuple[str, ...] | None = None,
    default_candidate_roles: tuple[str, ...] | None = None,
) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", "_", str(raw_role or "").strip().lower()).strip("_")
    allowed = tuple(str(item or "").strip() for item in (allowed_candidate_roles or ()) if str(item or "").strip())
    defaults = tuple(str(item or "").strip() for item in (default_candidate_roles or ()) if str(item or "").strip())
    if allowed and normalized in allowed:
        return normalized
    if defaults:
        slot = min(max(int(index) - 1, 0), len(defaults) - 1)
        fallback = defaults[slot]
        if not allowed or fallback in allowed:
            return fallback
    if allowed:
        return allowed[min(max(int(index) - 1, 0), len(allowed) - 1)]
    return normalized


def expression_dedup_key(expr: str) -> str:
    text = normalize_expression(expr).replace("$", "").strip()
    if not text:
        return ""
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return re.sub(r"\s+", "", text).lower()
    return _canonicalize_expression_node(tree.body)


def deduplicate_candidates(
    candidates: tuple[RefinementCandidate, ...],
) -> tuple[tuple[RefinementCandidate, ...], tuple[dict[str, str], ...]]:
    kept: list[RefinementCandidate] = []
    dropped: list[dict[str, str]] = []
    seen_by_key: dict[str, RefinementCandidate] = {}
    for candidate in candidates:
        dedup_key = expression_dedup_key(candidate.expression)
        incumbent = seen_by_key.get(dedup_key)
        if incumbent is None:
            seen_by_key[dedup_key] = candidate
            kept.append(candidate)
            continue
        dropped.append(
            {
                "name": candidate.name,
                "expression": candidate.expression,
                "duplicate_of_name": incumbent.name,
                "duplicate_of_expression": incumbent.expression,
                "dedup_key": dedup_key,
            }
        )
    return tuple(kept), tuple(dropped)


def parse_refinement_response(
    raw_response: str,
    *,
    family: SeedFamily,
    provider_name: str,
    model_name: str,
    allowed_candidate_roles: tuple[str, ...] | None = None,
    default_candidate_roles: tuple[str, ...] | None = None,
) -> LLMProposal:
    payload = _load_json_payload(raw_response)
    candidates: list[RefinementCandidate] = []
    for idx, item in enumerate(payload.get("candidate_formulas", []), start=1):
        item_dict = item if isinstance(item, dict) else {"expression": str(item)}
        candidate_name = _fallback_candidate_name(item_dict, index=idx)
        repair_result = repair_expression(_extract_expression(item_dict))
        candidates.append(
            RefinementCandidate(
                name=candidate_name,
                expression=repair_result.expression,
                explanation=str(item_dict.get("explanation", "")).strip(),
                candidate_role=_normalize_candidate_role(
                    item_dict.get("candidate_role", ""),
                    index=idx,
                    allowed_candidate_roles=allowed_candidate_roles,
                    default_candidate_roles=default_candidate_roles,
                ),
                rationale=str(item_dict.get("rationale", "")).strip(),
                expected_improvement=str(item_dict.get("expected_improvement", "")).strip(),
                risk=str(item_dict.get("risk", "")).strip(),
                source_model=model_name,
                source_provider=provider_name,
                parent_factor=str(payload.get("parent_factor") or family.canonical_seed),
                family=family.family,
                validation_warnings=tuple(f"auto_repair: {item}" for item in repair_result.actions),
            )
        )
    if not candidates:
        raise ValueError("provider response did not contain candidate_formulas")
    return LLMProposal(
        parent_factor=str(payload.get("parent_factor") or family.canonical_seed),
        diagnosed_weaknesses=_coerce_list(payload.get("diagnosed_weaknesses")),
        refinement_rationale=str(payload.get("refinement_rationale", "")).strip(),
        expected_behavior_change=str(payload.get("expected_behavior_change", "")).strip(),
        risk_notes=str(payload.get("risk_notes", "")).strip(),
        candidates=tuple(candidates),
        raw_response=raw_response,
    )
