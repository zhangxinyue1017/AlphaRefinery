from __future__ import annotations

"""QuantsPlaybook salience-theory factors.

Current migration focus:
- 凸显理论STR因子

Source notebook references:
- QuantsPlaybook/B-因子构建类/凸显理论STR因子/凸显度因子.ipynb
- QuantsPlaybook/B-因子构建类/凸显理论STR因子/scr/core.py

This first pass migrates the core salience-weighted return logic:
- sigma
- salience weight
- STR
- terrified-score family
"""

import numpy as np
import pandas as pd

from ..contract import validate_data
from ..data import wide_frame_to_series
from ..registry import FactorRegistry

QP_SALIENCE_REQUIRED_FIELDS: tuple[str, ...] = ("close", "turnover")
REGISTER_WINDOWS: tuple[int, ...] = (20, 60)
DELTA = 0.7
EPS = 1e-12


def _prepare_returns_turnover(
    data: dict[str, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_data(data, required_fields=QP_SALIENCE_REQUIRED_FIELDS)
    close_df = data["close"].unstack(level="instrument").sort_index()
    ret_df = close_df.pct_change(fill_method=None)
    turnover = data.get("turnover_rate")
    if turnover is None:
        turnover = data["turnover"] / 100.0
    turnover_df = turnover.unstack(level="instrument").sort_index()
    return ret_df, turnover_df


def _calc_sigma(ret_df: pd.DataFrame, bench: pd.Series | None = None) -> pd.DataFrame:
    if bench is None:
        bench = ret_df.mean(axis=1)
    a = ret_df.sub(bench, axis=0).abs()
    b = ret_df.abs().add(bench.abs(), axis=0) + 0.1
    return a.div(b)


def _calc_weight(sigma_df: pd.DataFrame, delta: float = DELTA) -> pd.DataFrame:
    rank = sigma_df.rank(axis=1, ascending=False, method="average")
    powered = np.power(delta, rank)
    scale = powered.mean(axis=1)
    return powered.div(scale.replace(0.0, np.nan), axis=0)


def _str_frame(data: dict[str, pd.Series], *, window: int) -> pd.DataFrame:
    ret_df, _ = _prepare_returns_turnover(data)
    sigma = _calc_sigma(ret_df)
    weight = _calc_weight(sigma)
    return weight.rolling(window).cov(ret_df)


def _avg_score_frame(data: dict[str, pd.Series], *, window: int) -> pd.DataFrame:
    ret_df, _ = _prepare_returns_turnover(data)
    sigma = _calc_sigma(ret_df)
    weighted = sigma * ret_df
    return weighted.rolling(window).mean()


def _std_score_frame(data: dict[str, pd.Series], *, window: int) -> pd.DataFrame:
    ret_df, _ = _prepare_returns_turnover(data)
    sigma = _calc_sigma(ret_df)
    weighted = sigma * ret_df
    return weighted.rolling(window).std()


def _terrified_score_frame(data: dict[str, pd.Series], *, window: int) -> pd.DataFrame:
    avg_score = _avg_score_frame(data, window=window)
    std_score = _std_score_frame(data, window=window)
    return 0.5 * (avg_score + std_score)


def _stv_frame(data: dict[str, pd.Series]) -> pd.DataFrame:
    ret_df, turnover_df = _prepare_returns_turnover(data)
    abs_ret = ret_df.abs()
    return abs_ret.where(abs_ret >= 0.1, turnover_df)


def _series_from_frame(frame: pd.DataFrame, *, factor_name: str) -> pd.Series:
    return wide_frame_to_series(frame, name=factor_name)


def qp_salience_str_20(data: dict[str, pd.Series]) -> pd.Series:
    """STR 因子: 先按凸显度给每日收益加权，再取 20 日滚动协方差。"""
    return _series_from_frame(_str_frame(data, window=20), factor_name="qp_salience.str_20")


def qp_salience_avg_score_20(data: dict[str, pd.Series]) -> pd.Series:
    """20日平均惊恐分: 凸显度乘收益后的滚动均值，刻画持续性的显著收益冲击。"""
    return _series_from_frame(
        _avg_score_frame(data, window=20),
        factor_name="qp_salience.avg_score_20",
    )


def qp_salience_std_score_20(data: dict[str, pd.Series]) -> pd.Series:
    """20日惊恐波动: 凸显度乘收益后的滚动标准差，刻画显著收益冲击是否剧烈。"""
    return _series_from_frame(
        _std_score_frame(data, window=20),
        factor_name="qp_salience.std_score_20",
    )


def qp_salience_terrified_score_20(data: dict[str, pd.Series]) -> pd.Series:
    """20日惊恐度: 平均惊恐分与惊恐波动的等权组合。"""
    return _series_from_frame(
        _terrified_score_frame(data, window=20),
        factor_name="qp_salience.terrified_score_20",
    )


def qp_salience_stv(data: dict[str, pd.Series]) -> pd.Series:
    """STV 代理: 极端收益日用绝对收益，否则用换手率，作为凸显交易活跃度的混合代理。"""
    return _series_from_frame(_stv_frame(data), factor_name="qp_salience.stv")


def _make_str_factor(window: int):
    def _factor(data: dict[str, pd.Series]) -> pd.Series:
        return _series_from_frame(_str_frame(data, window=window), factor_name=f"qp_salience.str_{window}")

    return _factor


def _make_avg_score_factor(window: int):
    def _factor(data: dict[str, pd.Series]) -> pd.Series:
        return _series_from_frame(
            _avg_score_frame(data, window=window),
            factor_name=f"qp_salience.avg_score_{window}",
        )

    return _factor


def _make_std_score_factor(window: int):
    def _factor(data: dict[str, pd.Series]) -> pd.Series:
        return _series_from_frame(
            _std_score_frame(data, window=window),
            factor_name=f"qp_salience.std_score_{window}",
        )

    return _factor


def _make_terrified_score_factor(window: int):
    def _factor(data: dict[str, pd.Series]) -> pd.Series:
        return _series_from_frame(
            _terrified_score_frame(data, window=window),
            factor_name=f"qp_salience.terrified_score_{window}",
        )

    return _factor


def register_qp_salience(registry: FactorRegistry) -> int:
    factor_specs: list[tuple[str, object, str]] = []
    for window in REGISTER_WINDOWS:
        factor_specs.extend(
            (
                (
                    f"qp_salience.str_{window}",
                    qp_salience_str_20 if window == 20 else _make_str_factor(window),
                    f"Salience-weighted rolling covariance between weight and return over {window} days.",
                ),
                (
                    f"qp_salience.avg_score_{window}",
                    qp_salience_avg_score_20 if window == 20 else _make_avg_score_factor(window),
                    f"Rolling mean of salience-weighted returns over {window} days.",
                ),
                (
                    f"qp_salience.std_score_{window}",
                    qp_salience_std_score_20 if window == 20 else _make_std_score_factor(window),
                    f"Rolling std of salience-weighted returns over {window} days.",
                ),
                (
                    f"qp_salience.terrified_score_{window}",
                    qp_salience_terrified_score_20 if window == 20 else _make_terrified_score_factor(window),
                    f"Equal-weight average of avg_score and std_score over {window} days.",
                ),
            )
        )
    factor_specs.append(
        (
            "qp_salience.stv",
            qp_salience_stv,
            "Salience trading-volume proxy mixing abs return and turnover rate.",
        )
    )

    for name, func, notes in factor_specs:
        registry.register(
            name,
            func,
            source="quants_playbook_salience",
            required_fields=QP_SALIENCE_REQUIRED_FIELDS,
            notes=notes,
        )
    return len(factor_specs)


def qp_salience_source_info() -> dict[str, object]:
    return {
        "source": "quants_playbook_salience",
        "status": "partially_migrated",
        "implemented_factors": (
            *(f"qp_salience.str_{window}" for window in REGISTER_WINDOWS),
            *(f"qp_salience.avg_score_{window}" for window in REGISTER_WINDOWS),
            *(f"qp_salience.std_score_{window}" for window in REGISTER_WINDOWS),
            *(f"qp_salience.terrified_score_{window}" for window in REGISTER_WINDOWS),
            "qp_salience.stv",
        ),
        "registered_windows": REGISTER_WINDOWS,
        "notes": "First-pass migration of the salience-theory notebook core formulas.",
    }
