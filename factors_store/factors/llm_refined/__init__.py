'''Registry exports for LLM-refined factor families.

Imports refined family specs so promoted formulas are available through the default registry.
'''

from __future__ import annotations

"""Local/private bucket for LLM-refined factor families.

This package is allowed to be empty in the public repository. When local
`*_family.py` modules are present, they are discovered and registered
automatically.
"""

import importlib
import inspect
import pkgutil
import re
from types import ModuleType

from .gp_historical_anchor_ratio_family import *  # noqa: F401,F403
from .qp_amplitude_sliced_momentum_family import *  # noqa: F401,F403
from .gp_return_open_marketfit_family import *  # noqa: F401,F403
from .salience_panic_score_family import *  # noqa: F401,F403
from .ideal_amplitude_structure_family import *  # noqa: F401,F403
from .abnormal_volume_attention_family import *  # noqa: F401,F403
from .qp_apb_price_bias_family import *  # noqa: F401,F403
from .close_volume_covariance_family import *  # noqa: F401,F403
from .open_volume_correlation_family import *  # noqa: F401,F403
from .common import LLM_REFINED_SOURCE

_EXPRESSION_RE = re.compile(r'expression\s*=\s*"(?P<expr>(?:[^"\\]|\\.)*)"')


def _load_family_modules() -> tuple[ModuleType, ...]:
    modules: list[ModuleType] = []
    prefix = f"{__name__}."
    for module_info in pkgutil.iter_modules(__path__):  # type: ignore[name-defined]
        module_name = str(module_info.name)
        if not module_name.endswith("_family"):
            continue
        modules.append(importlib.import_module(f"{prefix}{module_name}"))
    modules.sort(key=lambda module: getattr(module, "FAMILY_KEY", module.__name__))
    return tuple(modules)


FAMILY_MODULES = _load_family_modules()
FACTOR_SPECS = tuple(spec for module in FAMILY_MODULES for spec in getattr(module, "FACTOR_SPECS", ()))


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
    parent_factors = tuple(getattr(module, "PARENT_FACTOR", "") for module in FAMILY_MODULES)
    family_files = tuple(getattr(module, "FAMILY_KEY", module.__name__) for module in FAMILY_MODULES)
    seed_families = tuple(getattr(module, "SEED_FAMILY", "") for module in FAMILY_MODULES)
    summary_globs = tuple(getattr(module, "SUMMARY_GLOB", "") for module in FAMILY_MODULES)
    return {
        "source": LLM_REFINED_SOURCE,
        "status": "local_private_candidate_pool",
        "implemented_factors": tuple(spec.name for spec in FACTOR_SPECS),
        "family_files": family_files,
        "seed_families": seed_families,
        "parent_factors": parent_factors,
        "summary_globs": summary_globs,
        "notes": (
            "Local/private bucket for LLM-refined factors. "
            "The public repository may keep this package empty while local family modules remain usable."
        ),
    }


__all__ = [
    "FACTOR_SPECS",
    "FAMILY_MODULES",
    "LLM_REFINED_SOURCE",
    "llm_refined_source_info",
    "register_llm_refined",
]
