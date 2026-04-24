'''Built-in factor module namespace.

Collects alpha-like, QuantsPlaybook, GP-mined, seed baseline, and llm-refined factor modules.
'''

from .alpha101_like import alpha101_source_info, register_alpha101
from .alpha158_like import register_alpha158
from .alpha191_like import alpha191_source_info, register_alpha191
from .alpha360_like import register_alpha360
from .cicc_daily import cicc_daily_source_info, register_cicc_daily
from .factor365_daily import factor365_daily_source_info, register_factor365_daily
from .factor365_pattern import (
    factor365_pattern_source_info,
    factor365_pattern_v2_source_info,
    register_factor365_pattern,
)
from .gp_mined import gp_mined_source_info, register_gp_mined
from .llm_refined import llm_refined_source_info, register_llm_refined
from .qp_behavior import qp_behavior_source_info, register_qp_behavior
from .qp_chip import qp_chip_source_info, register_qp_chip
from .qp_kline import qp_kline_source_info, register_qp_kline
from .qp_momentum import qp_momentum_source_info, register_qp_momentum
from .qp_path_convexity import (
    qp_path_convexity_source_info,
    qp_path_convexity_v2_source_info,
    register_qp_path_convexity,
)
from .qp_salience import qp_salience_source_info, register_qp_salience
from .seed_baselines import register_seed_baselines, seed_baseline_source_info
from .qp_volatility import qp_volatility_source_info, register_qp_volatility

__all__ = [
    "alpha101_source_info",
    "alpha191_source_info",
    "cicc_daily_source_info",
    "factor365_daily_source_info",
    "factor365_pattern_source_info",
    "factor365_pattern_v2_source_info",
    "register_alpha101",
    "register_alpha191",
    "register_alpha158",
    "register_alpha360",
    "register_cicc_daily",
    "register_factor365_daily",
    "register_factor365_pattern",
    "gp_mined_source_info",
    "llm_refined_source_info",
    "qp_behavior_source_info",
    "qp_chip_source_info",
    "qp_momentum_source_info",
    "qp_path_convexity_source_info",
    "qp_path_convexity_v2_source_info",
    "qp_salience_source_info",
    "seed_baseline_source_info",
    "qp_volatility_source_info",
    "qp_kline_source_info",
    "register_gp_mined",
    "register_llm_refined",
    "register_qp_behavior",
    "register_qp_chip",
    "register_qp_momentum",
    "register_qp_path_convexity",
    "register_qp_salience",
    "register_seed_baselines",
    "register_qp_volatility",
    "register_qp_kline",
]
