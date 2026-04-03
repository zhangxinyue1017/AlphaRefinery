from __future__ import annotations

"""QuantsPlaybook K-line / candle-shape factors.

Current migration focus:
- 上下影线因子

Source notebook reference:
- QuantsPlaybook/B-因子构建类/上下影线因子/py/上下引线因子.ipynb

The original notebook is monthly and adds size/style neutralization later.
Here we first keep the directly portable OHLC-based core:
- standardized traditional upper/lower shadows
- standardized Williams upper/lower shadows
- 20-day rolling mean/std summaries
- a lightweight UBL-style raw combination without extra neutralization
"""

import numpy as np
import pandas as pd

from ..contract import validate_data
from ..data import to_worldquant_frame, wide_frame_to_series
from ..operators import cs_zscore, wide_sma as Mean, wide_stddev as Std
from ..registry import FactorRegistry

QP_KLINE_REQUIRED_FIELDS: tuple[str, ...] = ("open", "high", "low", "close")
EPS = 1e-12


def _prepare_inputs(data: dict[str, pd.Series]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_data(data, required_fields=QP_KLINE_REQUIRED_FIELDS)
    panel = to_worldquant_frame(data, fields=QP_KLINE_REQUIRED_FIELDS)
    return panel["open"], panel["high"], panel["low"], panel["close"]


def _lagged_rolling_mean(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return Mean(df, window).shift(1)


def _safe_frame_div(numer: pd.DataFrame, denom: pd.DataFrame) -> pd.DataFrame:
    aligned = denom.where(denom.abs() > EPS)
    return numer / aligned


def _normalized_shadow_components(data: dict[str, pd.Series]) -> dict[str, pd.DataFrame]:
    open_df, high_df, low_df, close_df = _prepare_inputs(data)

    upper_shadow = high_df - np.maximum(open_df, close_df)
    lower_shadow = np.minimum(open_df, close_df) - low_df
    williams_upper_shadow = high_df - close_df
    williams_lower_shadow = close_df - low_df

    raw_components = {
        "upper_shadow_norm": _safe_frame_div(upper_shadow, _lagged_rolling_mean(upper_shadow, 5)),
        "lower_shadow_norm": _safe_frame_div(lower_shadow, _lagged_rolling_mean(lower_shadow, 5)),
        "williams_upper_shadow_norm": _safe_frame_div(
            williams_upper_shadow,
            _lagged_rolling_mean(williams_upper_shadow, 5),
        ),
        "williams_lower_shadow_norm": _safe_frame_div(
            williams_lower_shadow,
            _lagged_rolling_mean(williams_lower_shadow, 5),
        ),
    }

    features: dict[str, pd.DataFrame] = {}
    for base_name, frame in raw_components.items():
        stem = base_name.replace("_norm", "")
        features[f"{stem}_mean20"] = Mean(frame, 20)
        features[f"{stem}_std20"] = Std(frame, 20)
    return features


def _series_feature(data: dict[str, pd.Series], feature_name: str, *, factor_name: str) -> pd.Series:
    features = _normalized_shadow_components(data)
    frame = features[feature_name]
    return wide_frame_to_series(frame, name=factor_name)


def qp_kline_upper_shadow_mean20(data: dict[str, pd.Series]) -> pd.Series:
    """20日均值: 传统上影线 `high - max(open, close)` 先做5日滞后均值标准化，再滚动取均值。"""
    return _series_feature(data, "upper_shadow_mean20", factor_name="qp_kline.upper_shadow_mean20")


def qp_kline_upper_shadow_std20(data: dict[str, pd.Series]) -> pd.Series:
    """20日波动: 传统上影线标准化后，统计近20日的离散程度。"""
    return _series_feature(data, "upper_shadow_std20", factor_name="qp_kline.upper_shadow_std20")


def qp_kline_lower_shadow_mean20(data: dict[str, pd.Series]) -> pd.Series:
    """20日均值: 传统下影线 `min(open, close) - low` 标准化后，刻画近一月平均下探力度。"""
    return _series_feature(data, "lower_shadow_mean20", factor_name="qp_kline.lower_shadow_mean20")


def qp_kline_lower_shadow_std20(data: dict[str, pd.Series]) -> pd.Series:
    """20日波动: 传统下影线标准化后，衡量近20日下影线强弱是否稳定。"""
    return _series_feature(data, "lower_shadow_std20", factor_name="qp_kline.lower_shadow_std20")


def qp_kline_williams_upper_shadow_mean20(data: dict[str, pd.Series]) -> pd.Series:
    """20日均值: Williams 上影线 `high - close` 标准化后，衡量收盘离最高点的平均距离。"""
    return _series_feature(
        data,
        "williams_upper_shadow_mean20",
        factor_name="qp_kline.williams_upper_shadow_mean20",
    )


def qp_kline_williams_upper_shadow_std20(data: dict[str, pd.Series]) -> pd.Series:
    """20日波动: Williams 上影线标准化后，衡量冲高后回落幅度的稳定性。"""
    return _series_feature(
        data,
        "williams_upper_shadow_std20",
        factor_name="qp_kline.williams_upper_shadow_std20",
    )


def qp_kline_williams_lower_shadow_mean20(data: dict[str, pd.Series]) -> pd.Series:
    """20日均值: Williams 下影线 `close - low` 标准化后，衡量收盘相对日内低点的平均修复力度。"""
    return _series_feature(
        data,
        "williams_lower_shadow_mean20",
        factor_name="qp_kline.williams_lower_shadow_mean20",
    )


def qp_kline_williams_lower_shadow_std20(data: dict[str, pd.Series]) -> pd.Series:
    """20日波动: Williams 下影线标准化后，衡量日内下探后收复力度是否稳定。"""
    return _series_feature(
        data,
        "williams_lower_shadow_std20",
        factor_name="qp_kline.williams_lower_shadow_std20",
    )


def qp_kline_ubl_raw(data: dict[str, pd.Series]) -> pd.Series:
    """Notebook-inspired UBL proxy without size/style neutralization.

    Inference from source:
    the original notebook combines ``Upper_shadow_std`` and
    ``Williams_lower_shadow_mean`` after neutralization and standardization.
    Here we keep the cross-sectional z-score combination but omit the extra
    style/size neutralization so the factor stays portable on plain OHLC data.

    Intuition:
    combine "上影线波动" with "下探后收复强度" into one candle-structure score.
    """

    upper_std = _series_feature(data, "upper_shadow_std20", factor_name="qp_kline.upper_shadow_std20")
    williams_lower_mean = _series_feature(
        data,
        "williams_lower_shadow_mean20",
        factor_name="qp_kline.williams_lower_shadow_mean20",
    )
    combo = cs_zscore(upper_std) + cs_zscore(williams_lower_mean)
    return combo.rename("qp_kline.ubl_raw")


def register_qp_kline(registry: FactorRegistry) -> int:
    factor_specs = (
        (
            "qp_kline.upper_shadow_mean20",
            qp_kline_upper_shadow_mean20,
            "20-day mean of notebook-style standardized traditional upper shadow.",
        ),
        (
            "qp_kline.upper_shadow_std20",
            qp_kline_upper_shadow_std20,
            "20-day std of notebook-style standardized traditional upper shadow.",
        ),
        (
            "qp_kline.lower_shadow_mean20",
            qp_kline_lower_shadow_mean20,
            "20-day mean of notebook-style standardized traditional lower shadow.",
        ),
        (
            "qp_kline.lower_shadow_std20",
            qp_kline_lower_shadow_std20,
            "20-day std of notebook-style standardized traditional lower shadow.",
        ),
        (
            "qp_kline.williams_upper_shadow_mean20",
            qp_kline_williams_upper_shadow_mean20,
            "20-day mean of notebook-style standardized Williams upper shadow.",
        ),
        (
            "qp_kline.williams_upper_shadow_std20",
            qp_kline_williams_upper_shadow_std20,
            "20-day std of notebook-style standardized Williams upper shadow.",
        ),
        (
            "qp_kline.williams_lower_shadow_mean20",
            qp_kline_williams_lower_shadow_mean20,
            "20-day mean of notebook-style standardized Williams lower shadow.",
        ),
        (
            "qp_kline.williams_lower_shadow_std20",
            qp_kline_williams_lower_shadow_std20,
            "20-day std of notebook-style standardized Williams lower shadow.",
        ),
        (
            "qp_kline.ubl_raw",
            qp_kline_ubl_raw,
            "Portable UBL proxy: cs_zscore(upper_shadow_std20) + cs_zscore(williams_lower_shadow_mean20).",
        ),
    )

    for name, func, notes in factor_specs:
        registry.register(
            name,
            func,
            source="quants_playbook_kline",
            required_fields=QP_KLINE_REQUIRED_FIELDS,
            notes=notes,
        )
    return len(factor_specs)


def qp_kline_source_info() -> dict[str, object]:
    return {
        "source": "quants_playbook_kline",
        "status": "partially_migrated",
        "implemented_factors": (
            "qp_kline.upper_shadow_mean20",
            "qp_kline.upper_shadow_std20",
            "qp_kline.lower_shadow_mean20",
            "qp_kline.lower_shadow_std20",
            "qp_kline.williams_upper_shadow_mean20",
            "qp_kline.williams_upper_shadow_std20",
            "qp_kline.williams_lower_shadow_mean20",
            "qp_kline.williams_lower_shadow_std20",
            "qp_kline.ubl_raw",
        ),
        "notes": (
            "This first pass keeps the daily OHLC-based shadow logic from the notebook. "
            "The original size/style-neutralized monthly composite is approximated here by qp_kline.ubl_raw."
        ),
    }
