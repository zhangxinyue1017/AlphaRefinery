'''AlphaRefinery factor store package.

Exposes registry, data, evaluation, and factor modules used by the command line tools.
'''

from __future__ import annotations

from .contract import (
    CORE_FIELDS,
    DERIVED_FIELDS,
    EXTENDED_DAILY_FIELDS,
    LIBRARY_OPTIONAL_FIELDS,
    LIBRARY_REQUIRED_FIELDS,
    OPTIONAL_CONTEXT_FIELDS,
    summarize_library_requirements,
)
from .data import (
    apply_main_filters,
    available_fields,
    build_data,
    build_data_bundle,
    load_panel,
    to_worldquant_frame,
    wide_frame_to_series,
)
from .eval import (
    build_price_panel,
    build_proxy_exposures,
    evaluate_factor,
    make_forward_return,
    prepare_backtest_inputs,
    run_factor_backtest,
    run_factor_backtest_report,
    summarize_backtest_result,
)
from .registry import FactorRegistry, create_default_registry
from .llm_refine import (
    OpenAICompatProvider,
    WideExpressionEngine,
    build_refinement_prompt,
    create_run_dir,
    evaluate_refinement_run,
    load_seed_pool,
    register_proposal_candidates,
    write_run_artifacts,
)

__all__ = [
    "CORE_FIELDS",
    "DERIVED_FIELDS",
    "EXTENDED_DAILY_FIELDS",
    "OPTIONAL_CONTEXT_FIELDS",
    "LIBRARY_REQUIRED_FIELDS",
    "LIBRARY_OPTIONAL_FIELDS",
    "summarize_library_requirements",
    "available_fields",
    "apply_main_filters",
    "build_data",
    "build_data_bundle",
    "load_panel",
    "to_worldquant_frame",
    "wide_frame_to_series",
    "evaluate_factor",
    "make_forward_return",
    "build_price_panel",
    "build_proxy_exposures",
    "prepare_backtest_inputs",
    "run_factor_backtest",
    "run_factor_backtest_report",
    "summarize_backtest_result",
    "FactorRegistry",
    "create_default_registry",
    "OpenAICompatProvider",
    "WideExpressionEngine",
    "build_refinement_prompt",
    "create_run_dir",
    "evaluate_refinement_run",
    "load_seed_pool",
    "register_proposal_candidates",
    "write_run_artifacts",
]
