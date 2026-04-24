'''Common runtime helpers for LLM-refined expression factors.

Wraps expression evaluation, field preparation, and series naming used by generated family modules.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from ...contract import validate_data
from ...data import wide_frame_to_series
from ...operators import (
    wide_correlation as correlation,
    wide_covariance as covariance,
    wide_decay_linear as decay_linear,
    wide_delta as delta,
    wide_rank as rank,
    wide_scale as scale,
    wide_sma as sma,
    wide_stddev as stddev,
    wide_ts_max as ts_max,
    wide_ts_rank as ts_rank,
)

LLM_REFINED_SOURCE = "llm_refined_candidates"
LLM_REFINED_REQUIRED_FIELDS: tuple[str, ...] = ("open", "amount", "volume")


@dataclass(frozen=True)
class FactorSpec:
    name: str
    func: Callable[[dict[str, pd.Series]], pd.Series]
    required_fields: tuple[str, ...]
    notes: str
    expr: str = ""


def prepare_core_inputs(
    data: dict[str, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_data(data, required_fields=LLM_REFINED_REQUIRED_FIELDS)
    open_df = data["open"].unstack(level="instrument").sort_index()
    amount_df = data["amount"].unstack(level="instrument").sort_index()
    volume_df = data["volume"].unstack(level="instrument").sort_index()
    return open_df, amount_df, volume_df


def frame_to_series(frame: pd.DataFrame, *, factor_name: str) -> pd.Series:
    return wide_frame_to_series(frame, name=factor_name)


def turnover_frame(data: dict[str, pd.Series]) -> pd.DataFrame:
    turnover = data.get("turnover_rate")
    if turnover is None:
        turnover = data["turnover"] / 100.0
    return turnover.unstack(level="instrument").sort_index()


def evaluate_expression_factor(
    data: dict[str, pd.Series],
    *,
    expression: str,
    factor_name: str,
) -> pd.Series:
    from ...llm_refine.parsing.expression_engine import WideExpressionEngine, guess_required_fields

    required_fields = guess_required_fields(expression)
    if required_fields:
        validate_data(data, required_fields=required_fields)
    engine = WideExpressionEngine(data)
    return engine.evaluate_series(expression, name=factor_name)
