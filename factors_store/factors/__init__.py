'''Built-in factor module namespace.

Collects public factor modules and any optional local/private factor modules
that are present in the working tree.
'''

from .alpha101_like import alpha101_source_info, register_alpha101
from .alpha158_like import register_alpha158
from .alpha191_like import alpha191_source_info, register_alpha191
from .alpha360_like import register_alpha360
from .llm_refined import llm_refined_source_info, register_llm_refined
from .seed_baselines import register_seed_baselines, seed_baseline_source_info


def _missing_source_info(source: str) -> dict[str, object]:
    return {
        "source": source,
        "status": "missing_optional_local_module",
        "implemented_factors": (),
    }


def _missing_register(_registry) -> int:
    return 0


try:
    from .cicc_daily import cicc_daily_source_info, register_cicc_daily
except ModuleNotFoundError:
    cicc_daily_source_info = lambda: _missing_source_info("cicc_daily")
    register_cicc_daily = _missing_register

try:
    from .factor365_daily import factor365_daily_source_info, register_factor365_daily
except ModuleNotFoundError:
    factor365_daily_source_info = lambda: _missing_source_info("factor365_daily")
    register_factor365_daily = _missing_register

try:
    from .factor365_pattern import (
        factor365_pattern_source_info,
        factor365_pattern_v2_source_info,
        register_factor365_pattern,
    )
except ModuleNotFoundError:
    factor365_pattern_source_info = lambda: _missing_source_info("factor365_pattern")
    factor365_pattern_v2_source_info = lambda: _missing_source_info("factor365_pattern_v2")
    register_factor365_pattern = _missing_register

try:
    from .gp_mined import gp_mined_source_info, register_gp_mined
except ModuleNotFoundError:
    gp_mined_source_info = lambda: _missing_source_info("gp_mined")
    register_gp_mined = _missing_register

try:
    from .qp_behavior import qp_behavior_source_info, register_qp_behavior
except ModuleNotFoundError:
    qp_behavior_source_info = lambda: _missing_source_info("qp_behavior")
    register_qp_behavior = _missing_register

try:
    from .qp_chip import qp_chip_source_info, register_qp_chip
except ModuleNotFoundError:
    qp_chip_source_info = lambda: _missing_source_info("qp_chip")
    register_qp_chip = _missing_register

try:
    from .qp_kline import qp_kline_source_info, register_qp_kline
except ModuleNotFoundError:
    qp_kline_source_info = lambda: _missing_source_info("qp_kline")
    register_qp_kline = _missing_register

try:
    from .qp_momentum import qp_momentum_source_info, register_qp_momentum
except ModuleNotFoundError:
    qp_momentum_source_info = lambda: _missing_source_info("qp_momentum")
    register_qp_momentum = _missing_register

try:
    from .qp_path_convexity import (
        qp_path_convexity_source_info,
        qp_path_convexity_v2_source_info,
        register_qp_path_convexity,
    )
except ModuleNotFoundError:
    qp_path_convexity_source_info = lambda: _missing_source_info("qp_path_convexity")
    qp_path_convexity_v2_source_info = lambda: _missing_source_info("qp_path_convexity_v2")
    register_qp_path_convexity = _missing_register

try:
    from .qp_pressure import qp_pressure_source_info, register_qp_pressure
except ModuleNotFoundError:
    qp_pressure_source_info = lambda: _missing_source_info("qp_pressure")
    register_qp_pressure = _missing_register

try:
    from .qp_salience import qp_salience_source_info, register_qp_salience
except ModuleNotFoundError:
    qp_salience_source_info = lambda: _missing_source_info("qp_salience")
    register_qp_salience = _missing_register

try:
    from .qp_volatility import qp_volatility_source_info, register_qp_volatility
except ModuleNotFoundError:
    qp_volatility_source_info = lambda: _missing_source_info("qp_volatility")
    register_qp_volatility = _missing_register

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
    "qp_pressure_source_info",
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
    "register_qp_pressure",
    "register_qp_salience",
    "register_seed_baselines",
    "register_qp_volatility",
    "register_qp_kline",
]
