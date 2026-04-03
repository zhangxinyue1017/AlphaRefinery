from __future__ import annotations

import re

from factors_store.contract import CORE_FIELDS, DERIVED_FIELDS, EXTENDED_DAILY_FIELDS, OPTIONAL_CONTEXT_FIELDS

from .operator_contract import KNOWN_EXTRA_TOKENS, KNOWN_OPERATOR_TOKENS
from ..prompting.prompt_builder import DEFAULT_WINDOWS

_ALLOWED_FIELDS = {str(item) for item in {*CORE_FIELDS, *DERIVED_FIELDS, *EXTENDED_DAILY_FIELDS, *OPTIONAL_CONTEXT_FIELDS}}
_SCIENTIFIC_LITERAL_PATTERN = re.compile(r"(?<![A-Za-z_])(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?\d+")


def validate_expression(expr: str) -> tuple[str, ...]:
    warnings: list[str] = []
    if not expr:
        return ("empty expression",)
    expr = expr.strip()
    if expr.startswith("<") and expr.endswith(">"):
        expr = expr[1:-1].strip()

    sanitized = _SCIENTIFIC_LITERAL_PATTERN.sub("0", expr)

    integers = [int(item) for item in re.findall(r"(?<![A-Za-z_])(\d+)(?![A-Za-z_])", sanitized)]
    invalid_windows = [item for item in integers if item not in DEFAULT_WINDOWS and item not in (0, 1)]
    if invalid_windows:
        warnings.append(f"contains non-default windows: {sorted(set(invalid_windows))}")

    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", sanitized.replace("$", ""))
    unknown: set[str] = set()
    for token in tokens:
        lower = token.lower()
        if lower in KNOWN_OPERATOR_TOKENS:
            continue
        if re.fullmatch(r"adv\d+", lower):
            continue
        if lower in _ALLOWED_FIELDS:
            continue
        if lower in KNOWN_EXTRA_TOKENS:
            continue
        if token.isupper():
            continue
        unknown.add(token)
    if unknown:
        warnings.append(f"contains tokens outside current whitelist: {sorted(unknown)}")
    return tuple(warnings)
