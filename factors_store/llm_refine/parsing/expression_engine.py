from __future__ import annotations

import ast
import re
from collections.abc import Mapping

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ...contract import CORE_FIELDS, DERIVED_FIELDS, EXTENDED_DAILY_FIELDS, OPTIONAL_CONTEXT_FIELDS
from ...data import wide_frame_to_series
from ...operators import (
    wide_abs,
    wide_cs_multi_reg_resid,
    wide_cs_reg_resid,
    wide_cs_skew,
    wide_cs_std,
    wide_cs_sum,
    wide_correlation,
    wide_covariance,
    wide_decay_linear,
    wide_delay,
    wide_delta,
    wide_highday,
    wide_inv,
    wide_lowday,
    wide_log,
    wide_max,
    wide_min,
    wide_macd,
    wide_power,
    wide_product,
    wide_quantile,
    wide_rank,
    wide_regbeta,
    wide_regression_residual,
    wide_regression_rsq,
    wide_regression_slope,
    wide_resi,
    wide_rsquare,
    wide_scale,
    wide_sequence,
    wide_sign,
    wide_slope,
    wide_slog1p,
    wide_sma,
    wide_sma_ewm,
    wide_sqrt,
    wide_stddev,
    wide_sumif,
    wide_ts_turnover_ref_price,
    wide_ts_sorted_mean_spread,
    wide_count,
    wide_rolling_cs_spearman_mean,
    wide_ts_count,
    wide_ts_ir,
    wide_ts_kurt,
    wide_ts_mad,
    wide_ts_max_diff,
    wide_ts_med,
    wide_ts_min_diff,
    wide_ts_min_max_diff,
    wide_ts_pct_change,
    wide_ts_exp_weighted_mean_lagged,
    wide_ts_linear_decay_mean,
    wide_ts_skew,
    wide_ts_var,
    wide_ts_argmax,
    wide_ts_argmin,
    wide_ts_max,
    wide_ts_min,
    wide_ts_rank,
    wide_ts_sum,
    wide_upper_shadow,
    wide_wma,
)
from ...factors.alpha101_like import _build_alpha101_group_maps
from .operator_contract import EXPRESSION_TOKEN_ALIASES, FUNCTION_STYLE_BINARY_OPERATORS

AVAILABLE_FIELDS = {str(item) for item in {*CORE_FIELDS, *DERIVED_FIELDS, *EXTENDED_DAILY_FIELDS, *OPTIONAL_CONTEXT_FIELDS}}
_ADV_PATTERN = re.compile(r"adv(\d+)$", flags=re.IGNORECASE)
_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TOKEN_ALIASES = set(EXPRESSION_TOKEN_ALIASES)
_FIELD_ALIASES = {
    "preclose": "pre_close",
}
_RESERVED_FN_ALIASES = {
    "and": "logical_and",
    "or": "logical_or",
    "not": "logical_not",
}


class ExpressionEvaluationError(ValueError):
    """Raised when a generated expression cannot be evaluated safely."""


def normalize_expression(expr: str) -> str:
    text = str(expr or "").strip()
    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1].strip()
    text = text.replace("^", "**")
    for src, dst in _RESERVED_FN_ALIASES.items():
        text = re.sub(rf"\b{src}(?=\s*\()", dst, text)
    return text


def guess_required_fields(expr: str) -> tuple[str, ...]:
    text = normalize_expression(expr)
    tokens = {token.lower() for token in _TOKEN_PATTERN.findall(text)}
    required: set[str] = set()
    required |= {_FIELD_ALIASES.get(token, token) for token in tokens if _FIELD_ALIASES.get(token, token) in AVAILABLE_FIELDS}
    if "midprice" in tokens:
        required.update({"high", "low"})
    if "turnover_rate" in tokens:
        required.add("turnover")
    if "ts_turnover_ref_price" in tokens:
        required.update({"close", "turnover"})
    if any(_ADV_PATTERN.fullmatch(token) for token in tokens) or "rel_volume" in tokens:
        required.add("volume")
    if "rel_amount" in tokens:
        required.add("amount")
    return tuple(sorted(required))


def _wide_cs_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    std = frame.std(axis=1).replace(0.0, np.nan)
    return frame.sub(frame.mean(axis=1), axis=0).div(std, axis=0)


def _wide_cs_mean(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    out = pd.DataFrame(index=frame.index, columns=frame.columns, dtype=float)
    for col in frame.columns:
        out[col] = mean
    return out


def _wide_ema(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    span = max(int(window), 1)
    return frame.ewm(span=span, adjust=False).mean()


def _exp_decay_weights(window: int, half_life: float) -> np.ndarray:
    if float(half_life) <= 0.0:
        return np.ones(int(window), dtype=float)
    age = np.arange(int(window) - 1, -1, -1, dtype=float)
    return 0.5 ** (age / float(half_life))


def _wide_weighted_mean(frame: pd.DataFrame, window: int, *, half_life: float = 10.0) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=frame.index, columns=frame.columns, dtype=float)
    if window <= 0:
        raise ExpressionEvaluationError("weighted_mean window must be positive")
    if len(frame) < window:
        return out

    weights = _exp_decay_weights(window, half_life).reshape(1, 1, window)
    value_view = sliding_window_view(frame.to_numpy(dtype=float, copy=False), window_shape=window, axis=0)
    valid = ~np.isnan(value_view)
    full_valid = valid.all(axis=2)
    numer = np.where(valid, value_view * weights, 0.0).sum(axis=2)
    numer[~full_valid] = np.nan
    out.iloc[window - 1 :] = numer / float(window)
    return out


def _wide_volume_weighted_mean(values: pd.DataFrame, weights: pd.DataFrame, window: int) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=values.index, columns=values.columns, dtype=float)
    if window <= 0:
        raise ExpressionEvaluationError("volume_weighted_mean window must be positive")
    if len(values) < window:
        return out

    value_view = sliding_window_view(values.to_numpy(dtype=float, copy=False), window_shape=window, axis=0)
    weight_view = sliding_window_view(
        weights.reindex_like(values).to_numpy(dtype=float, copy=False),
        window_shape=window,
        axis=0,
    )
    valid = ~np.isnan(value_view) & ~np.isnan(weight_view)
    full_valid = valid.all(axis=2)
    weight_sum = np.where(valid, weight_view, 0.0).sum(axis=2)
    numer = np.where(valid, value_view * weight_view, 0.0).sum(axis=2)
    tail = np.full(weight_sum.shape, np.nan, dtype=float)
    np.divide(numer, weight_sum, out=tail, where=full_valid & (weight_sum > 1e-12))
    out.iloc[window - 1 :] = tail
    return out


class WideExpressionEngine:
    def __init__(self, data: Mapping[str, pd.Series]) -> None:
        self._frames = self._prepare_frames(data)
        if not self._frames:
            raise ExpressionEvaluationError("no usable data fields available for expression evaluation")
        first_frame = next(iter(self._frames.values()))
        self._group_maps = _build_alpha101_group_maps(first_frame.columns)

    def evaluate_frame(self, expr: str) -> pd.DataFrame:
        normalized = normalize_expression(expr)
        try:
            tree = ast.parse(normalized, mode="eval")
        except SyntaxError as exc:  # pragma: no cover - syntax-dependent
            raise ExpressionEvaluationError(f"invalid expression syntax: {exc}") from exc
        value = self._eval_node(tree.body)
        if not isinstance(value, pd.DataFrame):
            raise ExpressionEvaluationError("expression did not evaluate to a wide DataFrame")
        return value.astype(float)

    def evaluate_series(self, expr: str, *, name: str | None = None) -> pd.Series:
        return wide_frame_to_series(self.evaluate_frame(expr), name=name)

    def _prepare_frames(self, data: Mapping[str, pd.Series]) -> dict[str, pd.DataFrame]:
        frames: dict[str, pd.DataFrame] = {}
        for key, value in data.items():
            if not isinstance(value, pd.Series):
                continue
            frames[str(key)] = value.unstack(level="instrument").sort_index().astype(float)
        if "turnover_rate" not in frames and "turnover" in frames:
            frames["turnover_rate"] = frames["turnover"] / 100.0
        if "turnover_rate" in frames:
            frames["turnover"] = frames["turnover_rate"]
        if "high" in frames and "low" in frames:
            frames["midprice"] = (frames["high"] + frames["low"]) / 2.0
        return frames

    def _eval_node(self, node: ast.AST):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return self._resolve_name(node.id)
        if isinstance(node, ast.BoolOp):
            if not node.values:
                raise ExpressionEvaluationError("boolean expression cannot be empty")
            values = [self._ensure_bool_frame(self._eval_node(item)) for item in node.values]
            result = values[0]
            if isinstance(node.op, ast.And):
                for item in values[1:]:
                    result = result & item
                return result
            if isinstance(node.op, ast.Or):
                for item in values[1:]:
                    result = result | item
                return result
            raise ExpressionEvaluationError(f"unsupported boolean operator: {ast.dump(node.op)}")
        if isinstance(node, ast.UnaryOp):
            value = self._eval_node(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return value
            if isinstance(node.op, ast.Not):
                return ~self._ensure_bool_frame(value)
            raise ExpressionEvaluationError(f"unsupported unary operator: {ast.dump(node.op)}")
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
            if isinstance(node.op, ast.BitAnd):
                return self._ensure_bool_frame(left) & self._ensure_bool_frame(right)
            if isinstance(node.op, ast.BitOr):
                return self._ensure_bool_frame(left) | self._ensure_bool_frame(right)
            raise ExpressionEvaluationError(f"unsupported binary operator: {ast.dump(node.op)}")
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise ExpressionEvaluationError("only single comparisons are supported")
            left = self._eval_node(node.left)
            right = self._eval_node(node.comparators[0])
            op = node.ops[0]
            if isinstance(op, ast.Gt):
                return left > right
            if isinstance(op, ast.Lt):
                return left < right
            if isinstance(op, ast.GtE):
                return left >= right
            if isinstance(op, ast.LtE):
                return left <= right
            if isinstance(op, ast.Eq):
                return left == right
            if isinstance(op, ast.NotEq):
                return left != right
            raise ExpressionEvaluationError(f"unsupported comparison operator: {ast.dump(op)}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Constant):
                numeric = node.func.value
                if isinstance(numeric, (int, float)) and float(numeric) == 1.0 and len(node.args) == 1 and not node.keywords:
                    cond = self._ensure_bool_frame(self._eval_node(node.args[0]))
                    return self._coerce_to_frame(1.0, like=cond).where(cond, 0.0)
            if not isinstance(node.func, ast.Name):
                raise ExpressionEvaluationError("only direct function calls are supported")
            name = node.func.id
            args = [self._eval_node(arg) for arg in node.args]
            kwargs = {
                str(keyword.arg): self._eval_node(keyword.value)
                for keyword in node.keywords
                if keyword.arg is not None
            }
            return self._call(name, args, kwargs)
        raise ExpressionEvaluationError(f"unsupported AST node: {ast.dump(node)}")

    def _resolve_name(self, token: str):
        key = str(token)
        lower = key.lower()
        lower = _FIELD_ALIASES.get(lower, lower)
        if key in self._frames:
            return self._frames[key]
        if lower in self._frames:
            return self._frames[lower]
        match = _ADV_PATTERN.fullmatch(lower)
        if match:
            return self._adv(int(match.group(1)))
        if lower in {"industry", "subindustry", "sector"}:
            return lower
        if lower == "true":
            return True
        if lower == "false":
            return False
        if lower == "nan":
            return np.nan
        raise ExpressionEvaluationError(f"unknown token: {token}")

    def _call(self, name: str, args: list[object], kwargs: dict[str, object] | None = None):
        lower = name.lower()
        kwargs = kwargs or {}
        if lower in FUNCTION_STYLE_BINARY_OPERATORS:
            if len(args) != 2:
                raise ExpressionEvaluationError(f"{lower} expects exactly 2 arguments")
            if lower == "add":
                return args[0] + args[1]
            if lower == "sub":
                return args[0] - args[1]
            if lower == "mul":
                return args[0] * args[1]
            return args[0] / args[1]
        if lower in {"gt", "lt", "ge", "le"}:
            if len(args) != 2:
                raise ExpressionEvaluationError(f"{lower} expects exactly 2 arguments")
            if lower == "gt":
                return args[0] > args[1]
            if lower == "lt":
                return args[0] < args[1]
            if lower == "ge":
                return args[0] >= args[1]
            return args[0] <= args[1]
        if lower in {"eq", "ne"}:
            if len(args) != 2:
                raise ExpressionEvaluationError(f"{lower} expects exactly 2 arguments")
            if lower == "eq":
                return args[0] == args[1]
            return args[0] != args[1]
        if lower in {"and", "or", "logical_and", "logical_or"}:
            if len(args) != 2:
                raise ExpressionEvaluationError(f"{lower} expects exactly 2 arguments")
            lhs = self._ensure_bool_frame(self._coerce_bool_operand(args[0], like=args[1] if isinstance(args[1], pd.DataFrame) else args[0]))
            rhs = self._ensure_bool_frame(self._coerce_bool_operand(args[1], like=lhs))
            if lower in {"and", "logical_and"}:
                return lhs & rhs
            return lhs | rhs
        if lower in {"not", "logical_not"}:
            if len(args) != 1:
                raise ExpressionEvaluationError("not expects exactly 1 argument")
            return ~self._ensure_bool_frame(self._coerce_bool_operand(args[0], like=args[0]))
        if lower == "neg":
            if len(args) != 1:
                raise ExpressionEvaluationError("neg expects exactly 1 argument")
            return -args[0]
        if lower in {"where", "if_then_else"}:
            if len(args) != 3:
                raise ExpressionEvaluationError(f"{lower} expects exactly 3 arguments")
            cond = self._ensure_bool_frame(args[0])
            left = self._coerce_to_frame(args[1], like=cond)
            right = self._coerce_to_frame(args[2], like=cond)
            return left.where(cond, right)
        if lower == "cs_mean":
            return _wide_cs_mean(self._ensure_frame(args[0]))
        if lower == "cs_sum":
            return wide_cs_sum(self._ensure_frame(args[0]))
        if lower == "cs_std":
            ddof = int(self._float(args[1])) if len(args) >= 2 else 0
            return wide_cs_std(self._ensure_frame(args[0]), ddof=ddof)
        if lower == "cs_skew":
            min_count = int(self._float(args[1])) if len(args) >= 2 else 3
            return wide_cs_skew(self._ensure_frame(args[0]), min_count=min_count)
        if lower in {"rank", "cs_rank"}:
            return wide_rank(self._ensure_frame(args[0]))
        if lower in {"cs_zscore", "zscore"}:
            return _wide_cs_zscore(self._ensure_frame(args[0]))
        if lower in {"ts_var"}:
            return wide_ts_var(self._ensure_frame(args[0]), self._window(args[1]))
        if lower in {"ts_ir"}:
            return wide_ts_ir(self._ensure_frame(args[0]), self._window(args[1]))
        if lower in {"ts_skew"}:
            return wide_ts_skew(self._ensure_frame(args[0]), self._window(args[1]))
        if lower in {"ts_kurt"}:
            return wide_ts_kurt(self._ensure_frame(args[0]), self._window(args[1]))
        if lower in {"corr", "ts_corr"}:
            return wide_correlation(self._ensure_frame(args[0]), self._ensure_frame(args[1]), self._window(args[2]))
        if lower in {"cov", "ts_cov"}:
            return wide_covariance(self._ensure_frame(args[0]), self._ensure_frame(args[1]), self._window(args[2]))
        if lower in {"std", "ts_std"}:
            return wide_stddev(self._ensure_frame(args[0]), self._window(args[1]))
        if lower in {"mean", "ts_mean"}:
            return wide_sma(self._ensure_frame(args[0]), self._window(args[1]))
        if lower == "sma":
            if len(args) < 2:
                raise ExpressionEvaluationError("sma expects at least 2 arguments")
            m = self._window(args[2]) if len(args) >= 3 else 1
            return wide_sma_ewm(self._ensure_frame(args[0]), self._window(args[1]), m)
        if lower == "wma":
            return wide_wma(self._ensure_frame(args[0]), self._window(args[1]))
        if lower == "product":
            return wide_product(self._ensure_frame(args[0]), self._window(args[1]))
        if lower in {"sum", "ts_sum"}:
            return wide_ts_sum(self._ensure_frame(args[0]), self._window(args[1]))
        if lower == "ts_count":
            return wide_ts_count(self._ensure_frame(args[0]), self._window(args[1]))
        if lower == "ts_med":
            return wide_ts_med(self._ensure_frame(args[0]), self._window(args[1]))
        if lower == "ts_mad":
            return wide_ts_mad(self._ensure_frame(args[0]), self._window(args[1]))
        if lower == "count":
            return wide_count(self._ensure_bool_frame(args[0]), self._window(args[1]))
        if lower == "sumif":
            return wide_sumif(self._ensure_frame(args[0]), self._window(args[1]), self._ensure_bool_frame(args[2]))
        if lower == "bucket_sum":
            if len(args) != 5:
                raise ExpressionEvaluationError("bucket_sum expects exactly 5 arguments")
            return self._bucket_sum(
                self._ensure_frame(args[0]),
                self._ensure_frame(args[1]),
                self._window(args[2]),
                self._float(args[3]),
                str(args[4]),
            )
        if lower == "ts_max":
            return wide_ts_max(self._ensure_frame(args[0]), self._window(args[1]))
        if lower == "ts_min":
            return wide_ts_min(self._ensure_frame(args[0]), self._window(args[1]))
        if lower == "max":
            return self._max_like(args)
        if lower == "min":
            return self._min_like(args)
        if lower == "delay":
            return wide_delay(self._ensure_frame(args[0]), self._window(args[1]))
        if lower == "delta":
            return wide_delta(self._ensure_frame(args[0]), self._window(args[1]))
        if lower in {"ref"}:
            return wide_delay(self._ensure_frame(args[0]), self._window(args[1]))
        if lower == "ts_pct_change":
            return wide_ts_pct_change(self._ensure_frame(args[0]), self._window(args[1]))
        if lower == "ema":
            return _wide_ema(self._ensure_frame(args[0]), self._window(args[1]))
        if lower == "decay_linear":
            return wide_decay_linear(self._ensure_frame(args[0]), self._window(args[1]))
        if lower == "ts_linear_decay_mean":
            return wide_ts_linear_decay_mean(self._ensure_frame(args[0]), self._window(args[1]))
        if lower == "ts_exp_weighted_mean_lagged":
            return wide_ts_exp_weighted_mean_lagged(self._ensure_frame(args[0]), self._window(args[1]))
        if lower == "weighted_mean":
            if len(args) < 2:
                raise ExpressionEvaluationError("weighted_mean expects at least 2 arguments")
            half_life = kwargs.get("half_life", args[2] if len(args) >= 3 else 10.0)
            return _wide_weighted_mean(
                self._ensure_frame(args[0]),
                self._window(args[1]),
                half_life=self._float(half_life),
            )
        if lower == "volume_weighted_mean":
            if len(args) != 3:
                raise ExpressionEvaluationError("volume_weighted_mean expects exactly 3 arguments")
            return _wide_volume_weighted_mean(
                self._ensure_frame(args[0]),
                self._ensure_frame(args[1]),
                self._window(args[2]),
            )
        if lower == "ts_turnover_ref_price":
            if len(args) != 3:
                raise ExpressionEvaluationError("ts_turnover_ref_price expects exactly 3 arguments")
            return wide_ts_turnover_ref_price(
                self._ensure_frame(args[0]),
                self._ensure_frame(args[1]),
                self._window(args[2]),
            )
        if lower == "ts_sorted_mean_spread":
            if len(args) != 4:
                raise ExpressionEvaluationError("ts_sorted_mean_spread expects exactly 4 arguments")
            return wide_ts_sorted_mean_spread(
                self._ensure_frame(args[0]),
                self._ensure_frame(args[1]),
                self._window(args[2]),
                self._float(args[3]),
            )
        if lower == "rolling_cs_spearman_mean":
            if len(args) < 2:
                raise ExpressionEvaluationError("rolling_cs_spearman_mean expects at least 2 arguments")
            min_obs = kwargs.get("min_obs", args[2] if len(args) >= 3 else 10)
            return wide_rolling_cs_spearman_mean(
                self._ensure_frame(args[0]),
                self._window(args[1]),
                min_obs=int(self._float(min_obs)),
            )
        if lower == "cs_reg_resid":
            if len(args) < 2:
                raise ExpressionEvaluationError("cs_reg_resid expects at least 2 arguments")
            min_obs = kwargs.get("min_obs", args[2] if len(args) >= 3 else 20)
            return wide_cs_reg_resid(
                self._ensure_frame(args[0]),
                self._ensure_frame(args[1]),
                min_obs=int(self._float(min_obs)),
            )
        if lower == "cs_multi_reg_resid":
            if len(args) < 2:
                raise ExpressionEvaluationError("cs_multi_reg_resid expects at least one regressor")
            min_obs = kwargs.get("min_obs", 20)
            regressors = list(args[1:])
            if regressors and isinstance(regressors[-1], (int, float, np.integer, np.floating)) and not kwargs.get("min_obs"):
                min_obs = regressors.pop()
            if not regressors:
                raise ExpressionEvaluationError("cs_multi_reg_resid expects at least one regressor")
            return wide_cs_multi_reg_resid(
                self._ensure_frame(args[0]),
                *[self._ensure_frame(item) for item in regressors],
                min_obs=int(self._float(min_obs)),
            )
        if lower == "ts_rank":
            return wide_ts_rank(self._ensure_frame(args[0]), self._window(args[1]))
        if lower == "ts_argmax":
            return wide_ts_argmax(self._ensure_frame(args[0]), self._window(args[1]))
        if lower == "ts_argmin":
            return wide_ts_argmin(self._ensure_frame(args[0]), self._window(args[1]))
        if lower == "highday":
            return wide_highday(self._ensure_frame(args[0]), self._window(args[1]))
        if lower == "lowday":
            return wide_lowday(self._ensure_frame(args[0]), self._window(args[1]))
        if lower == "ts_quantile":
            return wide_quantile(self._ensure_frame(args[0]), self._window(args[1]), self._float(args[2]))
        if lower == "ts_max_diff":
            return wide_ts_max_diff(self._ensure_frame(args[0]), self._window(args[1]))
        if lower == "ts_min_diff":
            return wide_ts_min_diff(self._ensure_frame(args[0]), self._window(args[1]))
        if lower == "ts_min_max_diff":
            return wide_ts_min_max_diff(self._ensure_frame(args[0]), self._window(args[1]))
        if lower == "scale":
            return wide_scale(self._ensure_frame(args[0]))
        if lower == "abs":
            return wide_abs(self._ensure_frame(args[0]))
        if lower == "sign":
            return wide_sign(self._ensure_frame(args[0]))
        if lower == "log":
            return wide_log(self._ensure_frame(args[0]))
        if lower == "slog1p":
            return wide_slog1p(self._ensure_frame(args[0]))
        if lower == "inv":
            return wide_inv(self._ensure_frame(args[0]))
        if lower == "sqrt":
            return wide_sqrt(self._ensure_frame(args[0]))
        if lower == "power":
            return wide_power(self._ensure_frame(args[0]), args[1])
        if lower in {"rowmax", "greater"}:
            return wide_max(args[0], args[1])
        if lower in {"rowmin", "less"}:
            return wide_min(args[0], args[1])
        if lower == "upper_shadow":
            if len(args) != 4:
                raise ExpressionEvaluationError("upper_shadow expects exactly 4 arguments")
            return wide_upper_shadow(
                self._ensure_frame(args[0]),
                self._ensure_frame(args[1]),
                self._ensure_frame(args[2]),
                self._ensure_frame(args[3]),
            )
        if lower == "macd":
            if len(args) != 3:
                raise ExpressionEvaluationError("macd expects exactly 3 arguments")
            return wide_macd(self._ensure_frame(args[0]), self._window(args[1]), self._window(args[2]))
        if lower == "regbeta":
            window = self._window(args[1])
            return wide_regbeta(self._ensure_frame(args[0]), wide_sequence(window))
        if lower == "regression_slope":
            return wide_regression_slope(
                self._ensure_frame(args[0]),
                self._ensure_frame(args[1]),
                self._window(args[2]),
            )
        if lower == "regression_rsq":
            return wide_regression_rsq(
                self._ensure_frame(args[0]),
                self._ensure_frame(args[1]),
                self._window(args[2]),
            )
        if lower == "regression_residual":
            return wide_regression_residual(
                self._ensure_frame(args[0]),
                self._ensure_frame(args[1]),
                self._window(args[2]),
            )
        if lower == "slope":
            return wide_slope(self._ensure_frame(args[0]), self._window(args[1]))
        if lower == "rsquare":
            return wide_rsquare(self._ensure_frame(args[0]), self._window(args[1]))
        if lower == "resi":
            return wide_resi(self._ensure_frame(args[0]), self._window(args[1]))
        if lower == "rel_volume":
            window = self._window(args[0])
            base = self._ensure_frame(self._resolve_name("volume"))
            return base / wide_sma(base, window).replace(0.0, np.nan)
        if lower == "rel_amount":
            window = self._window(args[0])
            base = self._ensure_frame(self._resolve_name("amount"))
            return base / wide_sma(base, window).replace(0.0, np.nan)
        if lower == "indneutralize":
            return self._indneutralize(self._ensure_frame(args[0]), str(args[1]))
        raise ExpressionEvaluationError(f"unsupported function: {name}")

    def _adv(self, window: int) -> pd.DataFrame:
        volume = self._ensure_frame(self._resolve_name("volume"))
        return wide_sma(volume, window)

    def _indneutralize(self, frame: pd.DataFrame, level: str) -> pd.DataFrame:
        groups = self._group_maps.get(str(level).lower())
        if groups is None:
            return frame
        out = pd.DataFrame(index=frame.index, columns=frame.columns, dtype=float)
        for _, cols in groups.groupby(groups).groups.items():
            use_cols = list(cols)
            sub = frame[use_cols]
            out[use_cols] = sub.sub(sub.mean(axis=1), axis=0)
        return out.reindex(columns=frame.columns)

    @staticmethod
    def _ensure_frame(value) -> pd.DataFrame:
        if isinstance(value, pd.DataFrame):
            return value.astype(float, copy=False)
        raise ExpressionEvaluationError(f"expected DataFrame value, got {type(value).__name__}")

    @staticmethod
    def _ensure_bool_frame(value) -> pd.DataFrame:
        if isinstance(value, pd.DataFrame):
            return value.astype(bool, copy=False)
        raise ExpressionEvaluationError(f"expected DataFrame condition, got {type(value).__name__}")

    @staticmethod
    def _coerce_to_frame(value, *, like: pd.DataFrame) -> pd.DataFrame:
        if isinstance(value, pd.DataFrame):
            return value.reindex_like(like).astype(float, copy=False)
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            return pd.DataFrame(float(value), index=like.index, columns=like.columns, dtype=float)
        raise ExpressionEvaluationError(f"expected DataFrame or scalar value, got {type(value).__name__}")

    @staticmethod
    def _coerce_bool_operand(value, like) -> pd.DataFrame:
        if isinstance(like, pd.DataFrame):
            frame_like = like
        elif isinstance(value, pd.DataFrame):
            frame_like = value
        else:
            raise ExpressionEvaluationError("boolean scalar requires a DataFrame counterpart")
        if isinstance(value, pd.DataFrame):
            return value.reindex_like(frame_like).astype(bool, copy=False)
        if isinstance(value, (bool, np.bool_)):
            return pd.DataFrame(bool(value), index=frame_like.index, columns=frame_like.columns, dtype=bool)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return pd.DataFrame(bool(value), index=frame_like.index, columns=frame_like.columns, dtype=bool)
        raise ExpressionEvaluationError(f"expected DataFrame or scalar boolean-like value, got {type(value).__name__}")

    @staticmethod
    def _window(value) -> int:
        if isinstance(value, bool):
            raise ExpressionEvaluationError("window cannot be boolean")
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            return int(round(float(value)))
        raise ExpressionEvaluationError(f"window must be numeric, got {type(value).__name__}")

    @staticmethod
    def _float(value) -> float:
        if isinstance(value, bool):
            raise ExpressionEvaluationError("q cannot be boolean")
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        raise ExpressionEvaluationError(f"expected numeric value, got {type(value).__name__}")

    def _max_like(self, args: list[object]):
        if len(args) != 2:
            raise ExpressionEvaluationError("max expects exactly 2 arguments")
        if isinstance(args[1], (int, float, np.integer, np.floating)):
            return wide_ts_max(self._ensure_frame(args[0]), self._window(args[1]))
        return wide_max(args[0], args[1])

    def _min_like(self, args: list[object]):
        if len(args) != 2:
            raise ExpressionEvaluationError("min expects exactly 2 arguments")
        if isinstance(args[1], (int, float, np.integer, np.floating)):
            return wide_ts_min(self._ensure_frame(args[0]), self._window(args[1]))
        return wide_min(args[0], args[1])

    def _bucket_sum(
        self,
        values: pd.DataFrame,
        key: pd.DataFrame,
        window: int,
        q: float,
        side: str,
    ) -> pd.DataFrame:
        if not 0.0 < q <= 0.5:
            raise ExpressionEvaluationError("bucket_sum requires 0 < q <= 0.5")
        side_norm = str(side).strip().lower()
        if side_norm not in {"low", "high"}:
            raise ExpressionEvaluationError("bucket_sum side must be 'low' or 'high'")
        value_frame = self._ensure_frame(values)
        key_frame = self._ensure_frame(key).reindex_like(value_frame)
        out = pd.DataFrame(np.nan, index=value_frame.index, columns=value_frame.columns, dtype=float)
        if len(value_frame) < window:
            return out

        key_view = sliding_window_view(key_frame.to_numpy(dtype=float, copy=False), window_shape=window, axis=0)
        value_view = sliding_window_view(value_frame.to_numpy(dtype=float, copy=False), window_shape=window, axis=0)
        valid = ~np.isnan(key_view) & ~np.isnan(value_view)
        full_valid = valid.all(axis=2)

        order = np.argsort(key_view, axis=2, kind="mergesort")
        ranks = np.argsort(order, axis=2, kind="mergesort") + 1
        pct_rank = ranks / float(window)
        if side_norm == "low":
            bucket_mask = pct_rank <= q
        else:
            bucket_mask = pct_rank >= (1.0 - q)

        tail = np.where(bucket_mask & valid, value_view, 0.0).sum(axis=2)
        tail[~full_valid] = np.nan
        out.iloc[window - 1 :] = tail
        return out
