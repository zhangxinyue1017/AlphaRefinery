'''Seed pool loading and formula resolution helpers.

Reads refinement seed YAML, normalizes family metadata, and expands known formula aliases.
'''

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from ...data_paths import DEFAULT_PANEL_PATH
from ..config import DEFAULT_SEED_POOL_PATH
from .models import EvaluationProtocol, EvaluationWindow, SeedFamily, SeedPool


DEFAULT_SEED_POOL = DEFAULT_SEED_POOL_PATH


def _normalize_formula_text(expression: str) -> str:
    return str(expression or "").strip()


def _qp_price_state_expr(*, side: str, window: int = 20) -> str:
    if side == "low":
        return f"le(close, ts_quantile(close, {int(window)}, 0.3))"
    if side == "high":
        return f"ge(close, ts_quantile(close, {int(window)}, 0.7))"
    raise ValueError(f"unsupported qp price state side: {side}")


def _qp_bucket_volume_expr(*, side: str, volume_field: str = "volume", window: int = 20) -> str:
    state = _qp_price_state_expr(side=side, window=window)
    return f"ts_sum(if_then_else({state}, {volume_field}, 0), {int(window)})"


def _qp_bucket_pv_expr(*, side: str, volume_field: str = "volume", window: int = 20) -> str:
    state = _qp_price_state_expr(side=side, window=window)
    return f"ts_sum(if_then_else({state}, mul(close, {volume_field}), 0), {int(window)})"


def _qp_total_volume_expr(*, volume_field: str = "volume", window: int = 20) -> str:
    return f"ts_sum({volume_field}, {int(window)})"


def _qp_overall_vwap_expr(*, volume_field: str = "volume", window: int = 20) -> str:
    total_pv = f"ts_sum(mul(close, {volume_field}), {int(window)})"
    total_volume = _qp_total_volume_expr(volume_field=volume_field, window=window)
    return f"div({total_pv}, add({total_volume}, 1e-12))"


def _qp_bucket_share_expr(*, side: str, volume_field: str = "volume", window: int = 20) -> str:
    bucket_volume = _qp_bucket_volume_expr(side=side, volume_field=volume_field, window=window)
    total_volume = _qp_total_volume_expr(volume_field=volume_field, window=window)
    return f"div({bucket_volume}, add({total_volume}, 1e-12))"


def _qp_bucket_vwap_expr(*, side: str, volume_field: str = "volume", window: int = 20) -> str:
    bucket_pv = _qp_bucket_pv_expr(side=side, volume_field=volume_field, window=window)
    bucket_volume = _qp_bucket_volume_expr(side=side, volume_field=volume_field, window=window)
    return f"div({bucket_pv}, add({bucket_volume}, 1e-12))"


def _qp_pressure_component_expr(*, kind: str, window: int = 20) -> str:
    overall_vwap = _qp_overall_vwap_expr(window=window)
    if kind == "buy_pressure":
        low_share = _qp_bucket_share_expr(side="low", window=window)
        low_vwap = _qp_bucket_vwap_expr(side="low", window=window)
        return (
            f"mul({low_share}, "
            f"div(sub({overall_vwap}, {low_vwap}), add(abs({overall_vwap}), 1e-12)))"
        )
    if kind == "sell_pressure":
        high_share = _qp_bucket_share_expr(side="high", window=window)
        high_vwap = _qp_bucket_vwap_expr(side="high", window=window)
        return (
            f"mul({high_share}, "
            f"div(sub({high_vwap}, {overall_vwap}), add(abs({overall_vwap}), 1e-12)))"
        )
    if kind == "net_pressure":
        buy_pressure = _qp_pressure_component_expr(kind="buy_pressure", window=window)
        sell_pressure = _qp_pressure_component_expr(kind="sell_pressure", window=window)
        return f"sub({buy_pressure}, {sell_pressure})"
    raise ValueError(f"unsupported qp pressure component kind: {kind}")


_PUBLIC_FORMULA_OVERRIDES = {
    "qp_pressure.low_price_volume_share_20": _qp_bucket_share_expr(side="low", window=20),
    "qp_pressure.high_price_volume_share_20": _qp_bucket_share_expr(side="high", window=20),
    "qp_pressure.low_price_volume_bias_20": (
        f"sub({_qp_bucket_share_expr(side='low', window=20)}, "
        f"{_qp_bucket_share_expr(side='high', window=20)})"
    ),
    "qp_pressure.buy_pressure_20": _qp_pressure_component_expr(kind="buy_pressure", window=20),
    "qp_pressure.sell_pressure_20": _qp_pressure_component_expr(kind="sell_pressure", window=20),
    "qp_pressure.net_pressure_20": _qp_pressure_component_expr(kind="net_pressure", window=20),
}


def _is_already_signed(expression: str, *, negative: bool) -> bool:
    text = _normalize_formula_text(expression)
    if not text:
        return False
    compact = text.replace(" ", "")
    if negative:
        return compact.startswith("-") or compact.startswith("neg(")
    return compact.startswith("+")


def apply_direction_rule(expression: str, direction: str) -> str:
    text = _normalize_formula_text(expression)
    if not text:
        return text
    rule = str(direction or "").strip()
    if rule == "use_negative_sign":
        if _is_already_signed(text, negative=True):
            return text
        return f"neg({text})"
    if rule == "use_positive_sign":
        if _is_already_signed(text, negative=False):
            return text
        return f"+({text})"
    return text


def resolve_factor_direction(family: SeedFamily, factor_name: str) -> str:
    override = str(family.formula_directions.get(factor_name, "") or "").strip()
    if override:
        return override
    return str(family.direction or "").strip()


def resolve_preferred_refine_seed(family: SeedFamily) -> str:
    preferred = str(getattr(family, "preferred_refine_seed", "") or "").strip()
    if preferred and (
        preferred == family.canonical_seed
        or preferred in family.aliases
        or preferred in family.formulas
    ):
        return preferred
    return family.canonical_seed


@lru_cache(maxsize=512)
def _resolve_registry_expression(factor_name: str) -> str:
    text = str(factor_name or "").strip()
    if not text or not text.startswith(("llm_refined.", "seed_baseline.")):
        return ""
    try:
        from ...registry import create_default_registry

        registry = create_default_registry()
        spec = registry.get(text)
        return str(spec.expr or "").strip()
    except Exception:
        return ""


def resolve_family_formula(family: SeedFamily, factor_name: str) -> str:
    raw = _PUBLIC_FORMULA_OVERRIDES.get(factor_name)
    if not raw:
        raw = family.formulas.get(factor_name, "")
    if not raw:
        raw = _resolve_registry_expression(factor_name)
    if not raw:
        canonical_raw = family.formulas.get(family.canonical_seed, "")
        raw = _PUBLIC_FORMULA_OVERRIDES.get(family.canonical_seed, canonical_raw)
    return apply_direction_rule(raw, resolve_factor_direction(family, factor_name))


def _ensure_tuple(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    return tuple(str(item) for item in values)


def _load_window(payload: Any) -> EvaluationWindow:
    data = dict(payload or {})
    return EvaluationWindow(
        start=str(data.get("start", "")),
        end=str(data.get("end", "")),
    )


def _load_protocol(payload: Any) -> EvaluationProtocol | None:
    if not payload:
        return None
    data = dict(payload)
    final_oos_payload = data.get("final_oos")
    final_oos_window = None
    if final_oos_payload:
        candidate = _load_window(final_oos_payload)
        if candidate.start or candidate.end:
            final_oos_window = candidate
    return EvaluationProtocol(
        search=_load_window(data.get("search")),
        selection=_load_window(data.get("selection")),
        final_oos=final_oos_window,
        prompt_history_stage=str(data.get("prompt_history_stage", "search")),
        keep_decision_stage=str(data.get("keep_decision_stage", "selection")),
        promote_stage=str(data.get("promote_stage", "selection")),
    )


def load_seed_pool(path: str | Path = DEFAULT_SEED_POOL) -> SeedPool:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    evaluation_defaults = dict(payload.get("evaluation_defaults", {}))
    if not str(evaluation_defaults.get("panel_path", "") or "").strip():
        evaluation_defaults["panel_path"] = str(DEFAULT_PANEL_PATH)
    families = tuple(
        SeedFamily(
            family=str(item["family"]),
            priority=str(item.get("priority", "medium")),
            canonical_seed=str(item["canonical_seed"]),
            aliases=_ensure_tuple(item.get("aliases")),
            direction=str(item.get("direction", "")),
            implementation_paths=_ensure_tuple(item.get("implementation_paths")),
            formulas={str(k): str(v) for k, v in dict(item.get("formulas", {})).items()},
            interpretation=str(item.get("interpretation", "")),
            likely_weaknesses=_ensure_tuple(item.get("likely_weaknesses")),
            refinement_axes=_ensure_tuple(item.get("refinement_axes")),
            preferred_refine_seed=str(item.get("preferred_refine_seed", "")),
            formula_directions={str(k): str(v) for k, v in dict(item.get("formula_directions", {})).items()},
            primary_objective=str(item.get("primary_objective", "")),
            secondary_objective=str(item.get("secondary_objective", "")),
            hard_constraints=_ensure_tuple(item.get("hard_constraints")),
            candidate_roles=_ensure_tuple(item.get("candidate_roles")),
            anti_patterns=_ensure_tuple(item.get("anti_patterns")),
            allowed_edit_types=_ensure_tuple(item.get("allowed_edit_types")),
            relation_note=str(item.get("relation_note", "")),
        )
        for item in payload.get("seed_groups", [])
    )
    return SeedPool(
        version=int(payload.get("version", 1)),
        created_at=str(payload.get("created_at", "")),
        project=str(payload.get("project", "")),
        purpose=str(payload.get("purpose", "")),
        evaluation_defaults=evaluation_defaults,
        evaluation_protocol=_load_protocol(payload.get("evaluation_protocol")),
        refinement_principles=_ensure_tuple(payload.get("refinement_principles")),
        families=families,
        llm_refinement_template=dict(payload.get("llm_refinement_template", {})),
    )
