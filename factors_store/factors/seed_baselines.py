'''Seed baseline factor specifications.

Registers canonical seed formulas used as parents and comparison baselines for refinement runs.
'''

from __future__ import annotations

"""Unified baseline registry for seed-family factors.

This module exposes a sign-aware baseline layer on top of the raw source
libraries. The goal is to give llm_refine a stable, family-centered registry
surface for:

- canonical seeds
- aliases
- preferred/oriented refine seeds

Unlike ``llm_refined``, these are not promoted research outputs. They are the
baseline factors we want to compare against consistently across prompt
construction, evaluation, and family-level analysis.
"""

import ast
from pathlib import Path
from typing import Any

import pandas as pd

from ..contract import validate_data
from ..llm_refine.core.seed_loader import (
    DEFAULT_SEED_POOL,
    load_seed_pool,
    resolve_factor_direction,
    resolve_family_formula,
    resolve_preferred_refine_seed,
)
from ..llm_refine.core.models import SeedFamily
from ..registry import FactorRegistry

SEED_BASELINE_SOURCE = "seed_baseline"
_LAST_IMPLEMENTED_FACTORS: tuple[str, ...] = ()
_LAST_SKIPPED_FACTORS: tuple[dict[str, str], ...] = ()


def _seed_target_names(family: SeedFamily) -> tuple[str, ...]:
    names = [family.canonical_seed, *family.aliases]
    preferred = resolve_preferred_refine_seed(family)
    if preferred and preferred not in names:
        names.append(preferred)
    return tuple(dict.fromkeys(str(item).strip() for item in names if str(item).strip()))


def _baseline_registry_name(factor_name: str) -> str:
    return f"{SEED_BASELINE_SOURCE}.{factor_name.replace('.', '_')}"


def seed_baseline_registry_name(factor_name: str) -> str:
    return _baseline_registry_name(factor_name)


def _baseline_legacy_aliases(family: SeedFamily, factor_name: str) -> tuple[str, ...]:
    direction = str(resolve_factor_direction(family, factor_name) or "").strip()
    underscored = factor_name.replace(".", "_")
    aliases: list[str] = []
    if direction == "use_negative_sign":
        aliases.append(f"{SEED_BASELINE_SOURCE}.neg_{underscored}")
    elif direction == "use_positive_sign":
        aliases.append(f"{SEED_BASELINE_SOURCE}.pos_{underscored}")
    return tuple(aliases)


def _baseline_role(family: SeedFamily, factor_name: str) -> str:
    roles: list[str] = []
    if factor_name == family.canonical_seed:
        roles.append("canonical")
    if factor_name in family.aliases:
        roles.append("alias")
    if factor_name == resolve_preferred_refine_seed(family):
        roles.append("preferred")
    return "+".join(roles) or "family_seed"


def _fatal_expression_issues(expression: str) -> tuple[str, ...]:
    from ..llm_refine.parsing.expression_engine import normalize_expression
    from ..llm_refine.parsing.validator import validate_expression

    text = normalize_expression(expression)
    issues: list[str] = []
    try:
        ast.parse(text, mode="eval")
    except SyntaxError as exc:
        issues.append(f"invalid expression syntax: {exc.msg}")
        return tuple(issues)
    for warning in validate_expression(text):
        if warning.startswith("contains dotted factor references"):
            issues.append(warning)
        elif warning.startswith("contains unsupported keyword arguments"):
            issues.append(warning)
        elif warning.startswith("contains tokens outside current whitelist"):
            issues.append(warning)
    return tuple(issues)


def _lookup_registry_factor(registry: FactorRegistry, factor_name: str):
    try:
        return registry.get(factor_name)
    except KeyError:
        return None


def _same_raw_formula_as_canonical(family: SeedFamily, factor_name: str) -> bool:
    raw = str(family.formulas.get(factor_name, "") or "").strip()
    canonical_raw = str(family.formulas.get(family.canonical_seed, "") or "").strip()
    return bool(raw and canonical_raw and raw == canonical_raw)


def _make_signed_registry_wrapper(spec, baseline_name: str, *, direction: str):
    required_fields = tuple(spec.required_fields)

    def _baseline(data: dict[str, pd.Series]) -> pd.Series:
        validate_data(data, required_fields=required_fields)
        series = spec.func(data)
        if direction == "use_negative_sign":
            series = -series
        return series.rename(baseline_name)

    return _baseline, required_fields


def _make_expression_wrapper(expression: str, baseline_name: str):
    from ..llm_refine.parsing.expression_engine import WideExpressionEngine, guess_required_fields

    required_fields = guess_required_fields(expression)

    def _baseline(data: dict[str, pd.Series]) -> pd.Series:
        if required_fields:
            validate_data(data, required_fields=required_fields)
        engine = WideExpressionEngine(data)
        return engine.evaluate_series(expression, name=baseline_name)

    return _baseline, required_fields


def _build_registration_plan(
    registry: FactorRegistry,
    family: SeedFamily,
    factor_name: str,
) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    direction = str(resolve_factor_direction(family, factor_name) or "").strip()
    role = _baseline_role(family, factor_name)
    resolved_expression = str(resolve_family_formula(family, factor_name) or "").strip()
    fatal_issues = _fatal_expression_issues(resolved_expression) if resolved_expression else ("empty resolved expression",)

    source_spec = _lookup_registry_factor(registry, factor_name)
    source_factor_name = factor_name
    impl_mode = "source_registry"

    if source_spec is None and factor_name != family.canonical_seed and _same_raw_formula_as_canonical(family, factor_name):
        source_spec = _lookup_registry_factor(registry, family.canonical_seed)
        source_factor_name = family.canonical_seed
        impl_mode = "canonical_fallback"

    if source_spec is not None:
        baseline_expr = None if fatal_issues else resolved_expression
        return (
            {
                "impl_mode": impl_mode,
                "source_factor_name": source_factor_name,
                "direction": direction,
                "expression": baseline_expr,
                "required_fields": tuple(source_spec.required_fields),
                "make_func": lambda baseline_name, spec=source_spec, signed_direction=direction: _make_signed_registry_wrapper(
                    spec,
                    baseline_name,
                    direction=signed_direction,
                ),
                "notes": (
                    f"family={family.family}; role={role}; source_factor={source_factor_name}; "
                    f"direction={direction or 'positive_as_is'}; impl={impl_mode}"
                ),
            },
            None,
        )

    if not fatal_issues and resolved_expression:
        from ..llm_refine.parsing.expression_engine import guess_required_fields

        required_fields = guess_required_fields(resolved_expression)
        return (
            {
                "impl_mode": "expression_fallback",
                "source_factor_name": factor_name,
                "direction": direction,
                "expression": resolved_expression,
                "required_fields": required_fields,
                "make_func": lambda baseline_name, expression=resolved_expression: _make_expression_wrapper(
                    expression,
                    baseline_name,
                ),
                "notes": (
                    f"family={family.family}; role={role}; source_factor={factor_name}; "
                    f"direction={direction or 'positive_as_is'}; impl=expression_fallback"
                ),
            },
            None,
        )

    return (
        None,
        {
            "family": family.family,
            "factor_name": factor_name,
            "reason": "; ".join(fatal_issues) if fatal_issues else "no backing registry factor",
        },
    )


def register_seed_baselines(
    registry: FactorRegistry,
    *,
    seed_pool_path: str | Path = DEFAULT_SEED_POOL,
) -> int:
    global _LAST_IMPLEMENTED_FACTORS, _LAST_SKIPPED_FACTORS

    pool = load_seed_pool(seed_pool_path)
    implemented: list[str] = []
    skipped: list[dict[str, str]] = []
    seen_names: set[str] = set()

    for family in pool.families:
        for factor_name in _seed_target_names(family):
            plan, skip_payload = _build_registration_plan(registry, family, factor_name)
            if plan is None:
                if skip_payload is not None:
                    skipped.append(skip_payload)
                continue

            baseline_names = (
                _baseline_registry_name(factor_name),
                *_baseline_legacy_aliases(family, factor_name),
            )
            for baseline_name in baseline_names:
                if baseline_name in seen_names:
                    continue
                func, required_fields = plan["make_func"](baseline_name)
                registry.register(
                    baseline_name,
                    func,
                    source=SEED_BASELINE_SOURCE,
                    required_fields=required_fields,
                    expr=plan["expression"],
                    notes=str(plan["notes"]),
                )
                seen_names.add(baseline_name)
                implemented.append(baseline_name)

    _LAST_IMPLEMENTED_FACTORS = tuple(implemented)
    _LAST_SKIPPED_FACTORS = tuple(skipped)
    return len(implemented)


def seed_baseline_catalog(*, seed_pool_path: str | Path = DEFAULT_SEED_POOL) -> tuple[dict[str, Any], ...]:
    pool = load_seed_pool(seed_pool_path)
    rows: list[dict[str, Any]] = []
    for family in pool.families:
        preferred = resolve_preferred_refine_seed(family)
        for factor_name in _seed_target_names(family):
            rows.append(
                {
                    "registry_factor_name": _baseline_registry_name(factor_name),
                    "legacy_registry_aliases": _baseline_legacy_aliases(family, factor_name),
                    "seed_factor_name": factor_name,
                    "family": family.family,
                    "role": _baseline_role(family, factor_name),
                    "preferred_refine_seed": preferred,
                    "is_preferred": factor_name == preferred,
                    "direction": str(resolve_factor_direction(family, factor_name) or "").strip(),
                }
            )
    return tuple(rows)


def seed_baseline_source_info(*, seed_pool_path: str | Path = DEFAULT_SEED_POOL) -> dict[str, object]:
    pool = load_seed_pool(seed_pool_path)
    configured_seed_factors = tuple(
        factor_name
        for family in pool.families
        for factor_name in _seed_target_names(family)
    )
    configured_baselines = tuple(
        dict.fromkeys(
            name
            for family in pool.families
            for factor_name in _seed_target_names(family)
            for name in (_baseline_registry_name(factor_name), *_baseline_legacy_aliases(family, factor_name))
        )
    )
    return {
        "source": SEED_BASELINE_SOURCE,
        "status": "baseline_registry",
        "configured_seed_factors": configured_seed_factors,
        "implemented_factors": _LAST_IMPLEMENTED_FACTORS or configured_baselines,
        "configured_factor_count": len(configured_baselines),
        "family_count": len(pool.families),
        "skipped_factors": _LAST_SKIPPED_FACTORS,
        "notes": (
            "Sign-aware unified baseline layer for llm_refine families. "
            "Includes canonical, alias, and preferred/oriented seed entry points."
        ),
    }
