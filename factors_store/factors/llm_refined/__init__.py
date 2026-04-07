from __future__ import annotations

"""Family-organized LLM refined factor candidates."""

import inspect
import re

from . import alpha003_family, alpha013_family, alpha016_family, alpha040_family, alpha044_family, alpha069_family, alpha095_family, alpha132_family, amplitude_structure_family, salience_panic_family, gp_relative_volume_pressure_family, gp_downside_price_position_family, weighted_upper_shadow_distribution_family, qp_high_price_distribution_pressure_family, ideal_amplitude_structure_family, qp_low_price_accumulation_pressure_family
from .alpha003_family import *  # noqa: F401,F403
from .alpha013_family import *  # noqa: F401,F403
from .alpha016_family import *  # noqa: F401,F403
from .alpha040_family import *  # noqa: F401,F403
from .alpha044_family import *  # noqa: F401,F403
from .alpha069_family import *  # noqa: F401,F403
from .alpha095_family import *  # noqa: F401,F403
from .alpha132_family import *  # noqa: F401,F403
from .amplitude_structure_family import *  # noqa: F401,F403
from .salience_panic_family import *  # noqa: F401,F403
from .gp_relative_volume_pressure_family import *  # noqa: F401,F403
from .gp_downside_price_position_family import *  # noqa: F401,F403
from .weighted_upper_shadow_distribution_family import *  # noqa: F401,F403
from .qp_high_price_distribution_pressure_family import *  # noqa: F401,F403
from .ideal_amplitude_structure_family import *  # noqa: F401,F403
from .qp_low_price_accumulation_pressure_family import *  # noqa: F401,F403
from .common import LLM_REFINED_SOURCE

FAMILY_MODULES = (
    alpha003_family,
    alpha013_family,
    alpha016_family,
    alpha040_family,
    alpha044_family,
    alpha069_family,
    alpha095_family,
    alpha132_family,
    amplitude_structure_family,
    salience_panic_family,
    gp_relative_volume_pressure_family,
    gp_downside_price_position_family,
    weighted_upper_shadow_distribution_family,
    qp_high_price_distribution_pressure_family,
    ideal_amplitude_structure_family,
    qp_low_price_accumulation_pressure_family,
)

FACTOR_SPECS = tuple(spec for module in FAMILY_MODULES for spec in module.FACTOR_SPECS)

_EXPRESSION_RE = re.compile(r'expression\s*=\s*"(?P<expr>(?:[^"\\]|\\.)*)"')


def _infer_expr_from_func(func) -> str:
    try:
        source = inspect.getsource(func)
    except Exception:
        return ""
    match = _EXPRESSION_RE.search(source)
    if not match:
        return ""
    return bytes(match.group("expr"), "utf-8").decode("unicode_escape").strip()


def register_llm_refined(registry) -> int:
    for spec in FACTOR_SPECS:
        registry.register(
            spec.name,
            spec.func,
            source=LLM_REFINED_SOURCE,
            required_fields=spec.required_fields,
            expr=(str(spec.expr or "").strip() or _infer_expr_from_func(spec.func)),
            notes=spec.notes,
        )
    return len(FACTOR_SPECS)


def llm_refined_source_info() -> dict[str, object]:
    parent_factors = tuple(module.PARENT_FACTOR for module in FAMILY_MODULES)
    family_files = tuple(module.FAMILY_KEY for module in FAMILY_MODULES)
    seed_families = tuple(module.SEED_FAMILY for module in FAMILY_MODULES)
    summary_globs = tuple(module.SUMMARY_GLOB for module in FAMILY_MODULES)
    return {
        "source": LLM_REFINED_SOURCE,
        "status": "active_candidate_pool",
        "implemented_factors": tuple(spec.name for spec in FACTOR_SPECS),
        "family_files": family_files,
        "seed_families": seed_families,
        "parent_factors": parent_factors,
        "summary_globs": summary_globs,
        "notes": (
            "Family-organized bucket for large-model optimized factors. "
            "Docstrings now record parent factor, round, source model, and keep-drop status."
        ),
    }


__all__ = [
    "FACTOR_SPECS",
    "FAMILY_MODULES",
    "llm_refined_source_info",
    "register_llm_refined",
]
for module in FAMILY_MODULES:
    __all__.extend(getattr(module, "__all__", ()))
