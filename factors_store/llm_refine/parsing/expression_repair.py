'''Expression normalization and repair helpers.

Fixes common LLM formula syntax issues before validation and evaluation.
'''

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from ..prompting.prompt_builder import DEFAULT_WINDOWS
from .expression_engine import normalize_expression

_EPSILON = 1e-12
_PUBLIC_RESERVED_FUNCTIONS = {
    "logical_and": "and",
    "logical_or": "or",
    "logical_not": "not",
}
_FUNCTION_ALIASES = {
    "corr": "ts_corr",
    "cov": "ts_cov",
    "greater": "rowmax",
    "less": "rowmin",
    "mean": "ts_mean",
    "rank": "cs_rank",
    "std": "ts_std",
    "sum": "ts_sum",
}
_WINDOW_ARG_INDEXES = {
    "ts_corr": (2,),
    "ts_cov": (2,),
    "ts_var": (1,),
    "ts_ir": (1,),
    "ts_skew": (1,),
    "ts_kurt": (1,),
    "ts_std": (1,),
    "ts_mean": (1,),
    "sma": (1,),
    "wma": (1,),
    "product": (1,),
    "ts_sum": (1,),
    "ts_count": (1,),
    "ts_med": (1,),
    "ts_mad": (1,),
    "ts_linear_decay_mean": (1,),
    "ts_exp_weighted_mean_lagged": (1,),
    "count": (1,),
    "sumif": (1,),
    "bucket_sum": (2,),
    "ts_turnover_ref_price": (2,),
    "ts_sorted_mean_spread": (2,),
    "rolling_cs_spearman_mean": (1,),
    "ts_max": (1,),
    "ts_min": (1,),
    "max": (1,),
    "min": (1,),
    "delay": (1,),
    "ref": (1,),
    "delta": (1,),
    "ts_pct_change": (1,),
    "ema": (1,),
    "decay_linear": (1,),
    "weighted_mean": (1,),
    "volume_weighted_mean": (2,),
    "macd": (1, 2),
    "ts_rank": (1,),
    "ts_argmax": (1,),
    "ts_argmin": (1,),
    "highday": (1,),
    "lowday": (1,),
    "ts_quantile": (1,),
    "ts_max_diff": (1,),
    "ts_min_diff": (1,),
    "ts_min_max_diff": (1,),
    "regbeta": (1,),
    "regression_slope": (2,),
    "regression_rsq": (2,),
    "regression_residual": (2,),
    "slope": (1,),
    "rsquare": (1,),
    "resi": (1,),
    "rel_volume": (0,),
    "rel_amount": (0,),
}
_EXPECTED_ARITY = {
    "neg": 1,
    "cs_mean": 1,
    "rank": 1,
    "cs_rank": 1,
    "cs_zscore": 1,
    "scale": 1,
    "abs": 1,
    "sign": 1,
    "log": 1,
    "rel_volume": 1,
    "rel_amount": 1,
    "add": 2,
    "sub": 2,
    "mul": 2,
    "div": 2,
    "gt": 2,
    "lt": 2,
    "ge": 2,
    "le": 2,
    "eq": 2,
    "ne": 2,
    "and": 2,
    "or": 2,
    "rowmax": 2,
    "rowmin": 2,
    "greater": 2,
    "less": 2,
    "power": 2,
    "ts_var": 2,
    "ts_ir": 2,
    "ts_skew": 2,
    "ts_kurt": 2,
    "ts_std": 2,
    "std": 2,
    "ts_mean": 2,
    "mean": 2,
    "sma": 3,
    "wma": 2,
    "product": 2,
    "ts_sum": 2,
    "sum": 2,
    "ts_count": 2,
    "ts_med": 2,
    "ts_mad": 2,
    "count": 2,
    "ts_max": 2,
    "ts_min": 2,
    "max": 2,
    "min": 2,
    "delay": 2,
    "ref": 2,
    "delta": 2,
    "ts_pct_change": 2,
    "ema": 2,
    "decay_linear": 2,
    "weighted_mean": 3,
    "volume_weighted_mean": 3,
    "ts_turnover_ref_price": 3,
    "ts_sorted_mean_spread": 4,
    "rolling_cs_spearman_mean": 3,
    "cs_reg_resid": 3,
    "ts_rank": 2,
    "ts_argmax": 2,
    "ts_argmin": 2,
    "highday": 2,
    "lowday": 2,
    "ts_max_diff": 2,
    "ts_min_diff": 2,
    "ts_min_max_diff": 2,
    "regbeta": 2,
    "slope": 2,
    "rsquare": 2,
    "resi": 2,
    "where": 3,
    "if_then_else": 3,
    "corr": 3,
    "ts_corr": 3,
    "cov": 3,
    "ts_cov": 3,
    "sumif": 3,
    "ts_quantile": 3,
    "bucket_sum": 5,
    "not": 1,
    "zscore": 1,
    "cs_sum": 1,
    "cs_std": 2,
    "cs_skew": 2,
    "inv": 1,
    "slog1p": 1,
    "sqrt": 1,
    "upper_shadow": 4,
    "macd": 3,
    "ts_linear_decay_mean": 2,
    "ts_exp_weighted_mean_lagged": 2,
    "regression_slope": 3,
    "regression_rsq": 3,
    "regression_residual": 3,
    "cs_multi_reg_resid": None,
}


@dataclass(frozen=True)
class ExpressionRepairResult:
    expression: str
    actions: tuple[str, ...]


def _numeric_value(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _numeric_value(node.operand)
        if inner is not None:
            return -inner
    return None


def _format_number(value: float) -> ast.Constant:
    rounded = round(float(value))
    if abs(float(value) - rounded) < 1e-12:
        return ast.Constant(value=int(rounded))
    return ast.Constant(value=float(value))


def _nearest_default_window(value: float) -> int:
    integer = max(int(round(float(value))), 1)
    if integer <= 1:
        return 1
    if integer in DEFAULT_WINDOWS:
        return integer
    return min(DEFAULT_WINDOWS, key=lambda item: (abs(item - integer), item))


def _is_small_positive_constant(node: ast.AST) -> bool:
    numeric = _numeric_value(node)
    return numeric is not None and 0.0 < abs(numeric) <= 1e-6


def _is_zero_guarded(node: ast.AST) -> bool:
    numeric = _numeric_value(node)
    if numeric is not None:
        return abs(numeric) > 1e-15
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        return _is_small_positive_constant(node.left) or _is_small_positive_constant(node.right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        lower = node.func.id.lower()
        if lower in {"add", "rowmax", "max"}:
            return any(_is_small_positive_constant(arg) for arg in node.args)
    return False


def _wrap_zero_guard(node: ast.AST) -> ast.Call:
    return ast.Call(
        func=ast.Name(id="add", ctx=ast.Load()),
        args=[node, ast.Constant(value=_EPSILON)],
        keywords=[],
    )


class _RepairTransformer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.actions: list[str] = []

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        updated = self.generic_visit(node)
        if isinstance(updated, ast.BinOp) and isinstance(updated.op, ast.Div) and not _is_zero_guarded(updated.right):
            updated.right = _wrap_zero_guard(updated.right)
            self.actions.append("wrapped division denominator with add(..., 1e-12)")
        return updated

    def visit_Call(self, node: ast.Call) -> ast.AST:
        updated = self.generic_visit(node)
        if not isinstance(updated, ast.Call):
            return updated

        if isinstance(updated.func, ast.Constant) and len(updated.args) == 1 and not updated.keywords:
            numeric = _numeric_value(updated.func)
            if numeric is not None and abs(numeric - 1.0) < 1e-12:
                self.actions.append("rewrote 1(cond) shorthand into where(cond, 1, 0)")
                return ast.Call(
                    func=ast.Name(id="where", ctx=ast.Load()),
                    args=[updated.args[0], ast.Constant(value=1), ast.Constant(value=0)],
                    keywords=[],
                )

        if not isinstance(updated.func, ast.Name):
            return updated

        lower = updated.func.id.lower()
        canonical = _FUNCTION_ALIASES.get(lower, lower)
        if canonical != lower:
            self.actions.append(f"normalized function alias {lower} -> {canonical}")
            updated.func.id = canonical
            lower = canonical

        expected_arity = _EXPECTED_ARITY.get(lower)
        if expected_arity is not None and len(updated.args) > expected_arity:
            updated.args = updated.args[:expected_arity]
            self.actions.append(f"trimmed {lower} arguments to expected arity {expected_arity}")

        if lower == "div" and len(updated.args) == 2 and not _is_zero_guarded(updated.args[1]):
            updated.args[1] = _wrap_zero_guard(updated.args[1])
            self.actions.append("wrapped div() denominator with add(..., 1e-12)")

        for arg_index in _WINDOW_ARG_INDEXES.get(lower, ()):
            if arg_index >= len(updated.args):
                continue
            numeric = _numeric_value(updated.args[arg_index])
            if numeric is None:
                continue
            clamped = _nearest_default_window(numeric)
            if clamped != int(round(numeric)):
                updated.args[arg_index] = _format_number(clamped)
                self.actions.append(
                    f"clamped {lower} window from {int(round(numeric))} to default window {clamped}"
                )
        return updated


def repair_expression(expr: str) -> ExpressionRepairResult:
    normalized = normalize_expression(expr).strip().strip("`")
    if not normalized:
        return ExpressionRepairResult(expression=normalized, actions=())
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError:
        return ExpressionRepairResult(expression=normalized, actions=())

    transformer = _RepairTransformer()
    repaired_tree = transformer.visit(tree)
    ast.fix_missing_locations(repaired_tree)
    repaired = normalize_expression(ast.unparse(repaired_tree.body)).strip()
    for internal, public in _PUBLIC_RESERVED_FUNCTIONS.items():
        repaired = repaired.replace(f"{internal}(", f"{public}(")
    actions = tuple(dict.fromkeys(transformer.actions))
    return ExpressionRepairResult(expression=repaired, actions=actions)
