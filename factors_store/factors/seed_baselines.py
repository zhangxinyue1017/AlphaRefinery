from __future__ import annotations

"""Baseline wrappers for seed-family evaluation.

These wrappers expose the actual seed direction used by llm_refine, so staged
search/selection backtests can be run on the same sign convention that later
enters prompt construction.
"""

from ..contract import validate_data
from ..registry import FactorRegistry


def seed_baseline_neg_qp_salience_std_score_60(data):
    validate_data(data, required_fields=("close", "turnover"))
    from .qp_salience import _make_std_score_factor

    factor = _make_std_score_factor(60)
    return (-factor(data)).rename("seed_baseline.neg_qp_salience_std_score_60")


def seed_baseline_neg_qp_salience_terrified_score_60(data):
    validate_data(data, required_fields=("close", "turnover"))
    from .qp_salience import _make_terrified_score_factor

    factor = _make_terrified_score_factor(60)
    return (-factor(data)).rename("seed_baseline.neg_qp_salience_terrified_score_60")


def seed_baseline_neg_qp_volatility_amplitude_spread_5(data):
    validate_data(data, required_fields=("high", "low", "close"))
    from .qp_volatility import _make_amplitude_spread_factor

    factor = _make_amplitude_spread_factor(5)
    return (-factor(data)).rename("seed_baseline.neg_qp_volatility_amplitude_spread_5")


def seed_baseline_neg_qp_volatility_amplitude_high_20(data):
    validate_data(data, required_fields=("high", "low", "close"))
    from .qp_volatility import _make_amplitude_high_factor

    factor = _make_amplitude_high_factor(20)
    return (-factor(data)).rename("seed_baseline.neg_qp_volatility_amplitude_high_20")


def register_seed_baselines(registry: FactorRegistry) -> int:
    registry.register(
        "seed_baseline.neg_qp_salience_std_score_60",
        seed_baseline_neg_qp_salience_std_score_60,
        source="seed_baseline",
        required_fields=("close", "turnover"),
        notes="Negative-direction seed baseline for qp_salience.std_score_60 in salience_panic_score family.",
    )
    registry.register(
        "seed_baseline.neg_qp_salience_terrified_score_60",
        seed_baseline_neg_qp_salience_terrified_score_60,
        source="seed_baseline",
        required_fields=("close", "turnover"),
        notes="Negative-direction seed baseline for qp_salience.terrified_score_60 in salience_panic_score family.",
    )
    registry.register(
        "seed_baseline.neg_qp_volatility_amplitude_spread_5",
        seed_baseline_neg_qp_volatility_amplitude_spread_5,
        source="seed_baseline",
        required_fields=("high", "low", "close"),
        notes="Negative-direction seed baseline for qp_volatility.amplitude_spread_5 in amplitude_structure family.",
    )
    registry.register(
        "seed_baseline.neg_qp_volatility_amplitude_high_20",
        seed_baseline_neg_qp_volatility_amplitude_high_20,
        source="seed_baseline",
        required_fields=("high", "low", "close"),
        notes="Negative-direction seed baseline for qp_volatility.amplitude_high_20 in amplitude_structure family.",
    )
    return 4


def seed_baseline_source_info() -> dict[str, object]:
    return {
        "source": "seed_baseline",
        "status": "initial",
        "implemented_factors": (
            "seed_baseline.neg_qp_salience_std_score_60",
            "seed_baseline.neg_qp_salience_terrified_score_60",
            "seed_baseline.neg_qp_volatility_amplitude_spread_5",
            "seed_baseline.neg_qp_volatility_amplitude_high_20",
        ),
    }
