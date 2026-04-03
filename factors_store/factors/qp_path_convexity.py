from __future__ import annotations

"""Portable price-path convexity factors.

Source reference:
- Huseyin Gulen, Michael Woeppel (2025), Price-Path Convexity and Short-Horizon Return Predictability
- Huatai/HaAn summary note dated 2026-03-26

Paper definition over a window:
1. midpoint = (first close + last close) / 2
2. average = mean(all daily closes in the window)
3. convexity = (midpoint - average) / midpoint

The paper finds that higher convexity predicts lower future returns, so the
registered factors below expose the effective alpha-facing direction:
``neg_convexity = -convexity``.
"""

import numpy as np
import pandas as pd

from ..contract import validate_data
from ..data import wide_frame_to_series
from ..operators import cs_zscore
from ..registry import FactorRegistry
from .qp_kline import _normalized_shadow_components
from .qp_pressure import _pressure_component_frames

QP_PATH_CONVEXITY_FIELDS: tuple[str, ...] = ("close",)
QP_PATH_CONVEXITY_SOURCE = "quants_playbook_path_convexity_v1"
QP_PATH_CONVEXITY_WINDOWS: tuple[int, ...] = (5, 10, 21)
QP_PATH_CONVEXITY_V2_SOURCE = "quants_playbook_path_convexity_v2"
QP_PATH_CONVEXITY_PRESSURE_FIELDS: tuple[str, ...] = ("close", "volume")
QP_PATH_CONVEXITY_UPPER_SHADOW_FIELDS: tuple[str, ...] = ("open", "high", "low", "close")
QP_PATH_CONVEXITY_TURNOVER_FIELDS: tuple[str, ...] = ("close", "turnover")
EPS = 1e-12


def _prepare_close(data: dict[str, pd.Series]) -> pd.DataFrame:
    validate_data(data, required_fields=QP_PATH_CONVEXITY_FIELDS)
    return data["close"].unstack(level="instrument").sort_index()


def _neg_convexity_frame(data: dict[str, pd.Series], *, window: int) -> pd.DataFrame:
    close_df = _prepare_close(data)
    start_df = close_df.shift(window - 1)
    midpoint = (start_df + close_df) / 2.0
    mean_close = close_df.rolling(window).mean()
    convexity = (midpoint - mean_close) / midpoint.where(midpoint.abs() > EPS)
    return -convexity


def _series_from_frame(frame: pd.DataFrame, *, factor_name: str) -> pd.Series:
    return wide_frame_to_series(frame, name=factor_name)


def _zscore_frame(frame: pd.DataFrame, *, factor_name: str) -> pd.DataFrame:
    series = wide_frame_to_series(frame, name=factor_name)
    return cs_zscore(series).unstack(level="instrument").sort_index()


def _interaction_frame(left: pd.DataFrame, right: pd.DataFrame, *, clip: float = 3.0) -> pd.DataFrame:
    return left.clip(-clip, clip) * right.clip(-clip, clip)


def _turnover_mean_frame(data: dict[str, pd.Series], *, window: int = 20) -> pd.DataFrame:
    validate_data(data, required_fields=QP_PATH_CONVEXITY_TURNOVER_FIELDS)
    turnover = data.get("turnover_rate")
    if turnover is None:
        turnover = data["turnover"] / 100.0
    turnover_df = turnover.unstack(level="instrument").sort_index()
    return turnover_df.shift(1).rolling(window).mean()


def _pressure_confirmation_frame(data: dict[str, pd.Series]) -> pd.DataFrame:
    validate_data(data, required_fields=QP_PATH_CONVEXITY_PRESSURE_FIELDS)
    return _pressure_component_frames(data, window=20)["net_pressure"]


def _upper_shadow_confirmation_frame(data: dict[str, pd.Series]) -> pd.DataFrame:
    validate_data(data, required_fields=QP_PATH_CONVEXITY_UPPER_SHADOW_FIELDS)
    return _normalized_shadow_components(data)["upper_shadow_mean20"]


def _neg_convexity_confirmation_combo_frame(
    base_frame: pd.DataFrame,
    confirm_frame: pd.DataFrame,
    *,
    base_name: str,
    confirm_name: str,
) -> pd.DataFrame:
    base_z = _zscore_frame(base_frame, factor_name=base_name)
    confirm_z = _zscore_frame(confirm_frame, factor_name=confirm_name)
    return (base_z + confirm_z) / 2.0


def _neg_convexity_confirmation_interaction_frame(
    base_frame: pd.DataFrame,
    confirm_frame: pd.DataFrame,
    *,
    base_name: str,
    confirm_name: str,
) -> pd.DataFrame:
    base_z = _zscore_frame(base_frame, factor_name=base_name)
    confirm_z = _zscore_frame(confirm_frame, factor_name=confirm_name)
    return _interaction_frame(base_z, confirm_z)


def qp_path_convexity_neg_convexity_5(data: dict[str, pd.Series]) -> pd.Series:
    """5-day effective path-convexity alpha: the negative sign of paper convexity."""
    return _series_from_frame(
        _neg_convexity_frame(data, window=5),
        factor_name="qp_path_convexity.neg_convexity_5",
    )


def qp_path_convexity_neg_convexity_10(data: dict[str, pd.Series]) -> pd.Series:
    """10-day effective path-convexity alpha: the negative sign of paper convexity."""
    return _series_from_frame(
        _neg_convexity_frame(data, window=10),
        factor_name="qp_path_convexity.neg_convexity_10",
    )


def qp_path_convexity_neg_convexity_21(data: dict[str, pd.Series]) -> pd.Series:
    """21-day effective path-convexity alpha: the negative sign of paper convexity."""
    return _series_from_frame(
        _neg_convexity_frame(data, window=21),
        factor_name="qp_path_convexity.neg_convexity_21",
    )


def _make_neg_convexity_factor(window: int):
    def _factor(data: dict[str, pd.Series]) -> pd.Series:
        return _series_from_frame(
            _neg_convexity_frame(data, window=window),
            factor_name=f"qp_path_convexity.neg_convexity_{window}",
        )

    return _factor


def qp_path_convexity_neg_convexity_net_pressure_combo_21_20(data: dict[str, pd.Series]) -> pd.Series:
    """Convexity + pressure confirmation: equal-weight cross-sectional z-score blend of neg_convexity_21 and net_pressure_20."""
    frame = _neg_convexity_confirmation_combo_frame(
        _neg_convexity_frame(data, window=21),
        _pressure_confirmation_frame(data),
        base_name="qp_path_convexity.neg_convexity_21",
        confirm_name="qp_pressure.net_pressure_20",
    )
    return _series_from_frame(frame, factor_name="qp_path_convexity.neg_convexity_net_pressure_combo_21_20")


def qp_path_convexity_neg_convexity_net_pressure_interaction_21_20(data: dict[str, pd.Series]) -> pd.Series:
    """Convexity x pressure confirmation: clipped z-score interaction of neg_convexity_21 and net_pressure_20."""
    frame = _neg_convexity_confirmation_interaction_frame(
        _neg_convexity_frame(data, window=21),
        _pressure_confirmation_frame(data),
        base_name="qp_path_convexity.neg_convexity_21",
        confirm_name="qp_pressure.net_pressure_20",
    )
    return _series_from_frame(
        frame,
        factor_name="qp_path_convexity.neg_convexity_net_pressure_interaction_21_20",
    )


def qp_path_convexity_neg_convexity_upper_shadow_combo_21_20(data: dict[str, pd.Series]) -> pd.Series:
    """Convexity + upper-shadow confirmation: equal-weight z-score blend of neg_convexity_21 and upper_shadow_mean20."""
    frame = _neg_convexity_confirmation_combo_frame(
        _neg_convexity_frame(data, window=21),
        _upper_shadow_confirmation_frame(data),
        base_name="qp_path_convexity.neg_convexity_21",
        confirm_name="qp_kline.upper_shadow_mean20",
    )
    return _series_from_frame(frame, factor_name="qp_path_convexity.neg_convexity_upper_shadow_combo_21_20")


def qp_path_convexity_neg_convexity_upper_shadow_interaction_21_20(data: dict[str, pd.Series]) -> pd.Series:
    """Convexity x upper-shadow confirmation: clipped z-score interaction of neg_convexity_21 and upper_shadow_mean20."""
    frame = _neg_convexity_confirmation_interaction_frame(
        _neg_convexity_frame(data, window=21),
        _upper_shadow_confirmation_frame(data),
        base_name="qp_path_convexity.neg_convexity_21",
        confirm_name="qp_kline.upper_shadow_mean20",
    )
    return _series_from_frame(
        frame,
        factor_name="qp_path_convexity.neg_convexity_upper_shadow_interaction_21_20",
    )


def qp_path_convexity_neg_convexity_turnover_combo_21_20(data: dict[str, pd.Series]) -> pd.Series:
    """Convexity + turnover confirmation: equal-weight z-score blend of neg_convexity_21 and lagged turnover_mean_20."""
    frame = _neg_convexity_confirmation_combo_frame(
        _neg_convexity_frame(data, window=21),
        _turnover_mean_frame(data, window=20),
        base_name="qp_path_convexity.neg_convexity_21",
        confirm_name="qp_turnover.mean_20",
    )
    return _series_from_frame(frame, factor_name="qp_path_convexity.neg_convexity_turnover_combo_21_20")


def qp_path_convexity_neg_convexity_turnover_interaction_21_20(data: dict[str, pd.Series]) -> pd.Series:
    """Convexity x turnover confirmation: clipped z-score interaction of neg_convexity_21 and lagged turnover_mean_20."""
    frame = _neg_convexity_confirmation_interaction_frame(
        _neg_convexity_frame(data, window=21),
        _turnover_mean_frame(data, window=20),
        base_name="qp_path_convexity.neg_convexity_21",
        confirm_name="qp_turnover.mean_20",
    )
    return _series_from_frame(
        frame,
        factor_name="qp_path_convexity.neg_convexity_turnover_interaction_21_20",
    )


def register_qp_path_convexity(registry: FactorRegistry) -> int:
    factor_specs: list[tuple[str, object, str]] = []
    for window in QP_PATH_CONVEXITY_WINDOWS:
        factor_specs.append(
            (
                f"qp_path_convexity.neg_convexity_{window}",
                (
                    qp_path_convexity_neg_convexity_5
                    if window == 5
                    else qp_path_convexity_neg_convexity_10
                    if window == 10
                    else qp_path_convexity_neg_convexity_21
                    if window == 21
                    else _make_neg_convexity_factor(window)
                ),
                f"Negative-sign path convexity over {window} trading days: -((midpoint - mean(close)) / midpoint).",
            )
        )

    for name, func, notes in factor_specs:
        registry.register(
            name,
            func,
            source=QP_PATH_CONVEXITY_SOURCE,
            required_fields=QP_PATH_CONVEXITY_FIELDS,
            notes=notes,
        )

    v2_specs: tuple[tuple[str, object, tuple[str, ...], str], ...] = (
        (
            "qp_path_convexity.neg_convexity_net_pressure_combo_21_20",
            qp_path_convexity_neg_convexity_net_pressure_combo_21_20,
            QP_PATH_CONVEXITY_PRESSURE_FIELDS,
            "Cross-sectional z-score blend of 21-day negative convexity and 20-day net pressure.",
        ),
        (
            "qp_path_convexity.neg_convexity_net_pressure_interaction_21_20",
            qp_path_convexity_neg_convexity_net_pressure_interaction_21_20,
            QP_PATH_CONVEXITY_PRESSURE_FIELDS,
            "Clipped z-score interaction of 21-day negative convexity and 20-day net pressure.",
        ),
        (
            "qp_path_convexity.neg_convexity_upper_shadow_combo_21_20",
            qp_path_convexity_neg_convexity_upper_shadow_combo_21_20,
            QP_PATH_CONVEXITY_UPPER_SHADOW_FIELDS,
            "Cross-sectional z-score blend of 21-day negative convexity and 20-day upper-shadow mean.",
        ),
        (
            "qp_path_convexity.neg_convexity_upper_shadow_interaction_21_20",
            qp_path_convexity_neg_convexity_upper_shadow_interaction_21_20,
            QP_PATH_CONVEXITY_UPPER_SHADOW_FIELDS,
            "Clipped z-score interaction of 21-day negative convexity and 20-day upper-shadow mean.",
        ),
        (
            "qp_path_convexity.neg_convexity_turnover_combo_21_20",
            qp_path_convexity_neg_convexity_turnover_combo_21_20,
            QP_PATH_CONVEXITY_TURNOVER_FIELDS,
            "Cross-sectional z-score blend of 21-day negative convexity and 20-day lagged turnover mean.",
        ),
        (
            "qp_path_convexity.neg_convexity_turnover_interaction_21_20",
            qp_path_convexity_neg_convexity_turnover_interaction_21_20,
            QP_PATH_CONVEXITY_TURNOVER_FIELDS,
            "Clipped z-score interaction of 21-day negative convexity and 20-day lagged turnover mean.",
        ),
    )

    for name, func, required_fields, notes in v2_specs:
        registry.register(
            name,
            func,
            source=QP_PATH_CONVEXITY_V2_SOURCE,
            required_fields=required_fields,
            notes=notes,
        )
    return len(factor_specs) + len(v2_specs)


def qp_path_convexity_source_info() -> dict[str, object]:
    return {
        "source": QP_PATH_CONVEXITY_SOURCE,
        "status": "first_pass_portable",
        "implemented_factors": tuple(
            f"qp_path_convexity.neg_convexity_{window}" for window in QP_PATH_CONVEXITY_WINDOWS
        ),
        "registered_windows": QP_PATH_CONVEXITY_WINDOWS,
        "notes": (
            "Portable first pass of the price-path convexity idea. "
            "These factors expose the effective alpha-facing sign from the paper, "
            "using trailing 5/10/21-day windows on close prices."
        ),
    }


def qp_path_convexity_v2_source_info() -> dict[str, object]:
    return {
        "source": QP_PATH_CONVEXITY_V2_SOURCE,
        "status": "second_pass_portable",
        "implemented_factors": (
            "qp_path_convexity.neg_convexity_net_pressure_combo_21_20",
            "qp_path_convexity.neg_convexity_net_pressure_interaction_21_20",
            "qp_path_convexity.neg_convexity_upper_shadow_combo_21_20",
            "qp_path_convexity.neg_convexity_upper_shadow_interaction_21_20",
            "qp_path_convexity.neg_convexity_turnover_combo_21_20",
            "qp_path_convexity.neg_convexity_turnover_interaction_21_20",
        ),
        "notes": (
            "Second-pass convexity research line combining the 21-day path-convexity signal "
            "with pressure, upper-shadow, and turnover confirmations via z-score blends and interactions."
        ),
    }
