from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

EXPECTED_INDEX_NAMES = ("datetime", "instrument")


class ContractError(ValueError):
    """Raised when A/B module interface contracts are violated."""


def _require_multiindex(index: pd.Index, name: str) -> pd.MultiIndex:
    if not isinstance(index, pd.MultiIndex):
        raise ContractError(f"{name} must use MultiIndex (datetime, instrument)")
    if index.nlevels != 2:
        raise ContractError(f"{name} must have exactly 2 index levels")
    idx = pd.MultiIndex.from_tuples(index.tolist(), names=index.names)
    if tuple(idx.names) != EXPECTED_INDEX_NAMES:
        raise ContractError(
            f"{name} index names must be {EXPECTED_INDEX_NAMES}, got {tuple(idx.names)}"
        )
    if idx.has_duplicates:
        raise ContractError(f"{name} index contains duplicate (datetime, instrument) rows")
    return idx


def _require_series(s: pd.Series, name: str) -> pd.Series:
    if not isinstance(s, pd.Series):
        raise ContractError(f"{name} must be pandas Series")
    _require_multiindex(s.index, f"{name}.index")
    return s


def _require_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise ContractError(f"{name} must be pandas DataFrame")
    _require_multiindex(df.index, f"{name}.index")
    if df.shape[1] == 0:
        raise ContractError(f"{name} must contain at least one factor column")
    return df


def call_barra_exposure_provider(
    provider: Callable[[], pd.DataFrame],
    target_index: pd.MultiIndex | None = None,
    required_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Call and validate the Barra exposure provider contract."""
    if not callable(provider):
        raise ContractError("barra_exposure_provider must be callable with zero arguments")
    exposures = _require_dataframe(provider(), "barra_exposures")
    exposures = exposures.sort_index()
    if target_index is not None:
        _require_multiindex(target_index, "target_index")
        exposures = exposures.reindex(target_index)
    if required_columns is not None:
        missing = [c for c in required_columns if c not in exposures.columns]
        if missing:
            raise ContractError(f"barra_exposures missing required columns: {missing}")
    return exposures


def call_quick_ic(
    quick_ic_fn: Callable[[pd.Series, pd.Series], float],
    factor: pd.Series,
    label: pd.Series,
) -> float:
    """Call and validate quick_ic contract, returning a stable float."""
    if not callable(quick_ic_fn):
        raise ContractError("quick_ic must be callable as quick_ic(factor, label)")
    fac = _require_series(factor, "factor")
    y = _require_series(label, "label")
    if not fac.index.equals(y.index):
        y = y.reindex(fac.index)
    score = float(quick_ic_fn(fac, y))
    if not np.isfinite(score):
        return 0.0
    return score


def call_group_map_provider(
    provider: Callable[[pd.MultiIndex], pd.Series],
    target_index: pd.MultiIndex,
) -> pd.Series:
    """Call and validate group_map provider contract."""
    if not callable(provider):
        raise ContractError("group_map_provider must be callable as provider(target_index)")
    idx = _require_multiindex(target_index, "target_index")
    out = _require_series(provider(idx), "group_map")
    return out.reindex(idx)


def make_static_group_map_provider(industry_static: pd.Series) -> Callable[[pd.MultiIndex], pd.Series]:
    """
    Build a group_map provider from static mapping:
    industry_static index=instrument, value=industry_code/name
    """
    if not isinstance(industry_static, pd.Series):
        raise ContractError("industry_static must be pandas Series(index=instrument)")
    if isinstance(industry_static.index, pd.MultiIndex):
        raise ContractError("industry_static index must be plain instrument index, not MultiIndex")
    if industry_static.index.has_duplicates:
        raise ContractError("industry_static index contains duplicate instruments")
    static = industry_static.copy()

    def _provider(target_index: pd.MultiIndex) -> pd.Series:
        idx = _require_multiindex(target_index, "target_index")
        inst = idx.get_level_values("instrument")
        values = static.reindex(inst)
        return pd.Series(values.to_numpy(), index=idx, name="group")

    return _provider

