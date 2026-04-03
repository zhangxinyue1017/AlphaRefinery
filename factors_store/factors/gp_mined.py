from __future__ import annotations

"""Curated GP-mined factors extracted from 2026 YAML artifacts.

These factors are representative, de-duplicated GP discoveries selected from
`/root/gp_factor_qlib/artifacts/2026*.yaml` and registered under the
`gp_mined.` namespace so they can be batch-evaluated inside factors_store and
used directly from notebooks.
"""

from dataclasses import dataclass

import pandas as pd

from ..contract import validate_data
from ..registry import FactorRegistry

from gp_factor_qlib.core.expression_tree import ExprNode, parse_qlib_expr  # noqa: E402
from gp_factor_qlib.engine.gp_engine import compute_factor_series  # noqa: E402


@dataclass(frozen=True)
class GPMinedSpec:
    name: str
    expr: str
    readable_expression: str
    qlib_expr: str | None
    source_yaml: str
    original_factor_name: str
    notes: str


GP_MINED_SPECS: tuple[GPMinedSpec, ...] = (
    GPMinedSpec(
        name="gp_mined.cov_ret_open_beta_r2_abs",
        expr="abs(add(ts_cov(id(returns), id(open), 5, 5), regression_rsq(id(returns), id(market_return), 60, 10)))",
        readable_expression="Abs(Cov($returns, $open, 5) + Rsquare($returns, $market_return, 60))",
        qlib_expr=None,
        source_yaml="20260320(1).yaml",
        original_factor_name="candidate_r010_48ce6eb34a",
        notes="Best net-return candidate from 20260320(1); absolute value of short-horizon return/open covariance plus market-fit R^2.",
    ),
    GPMinedSpec(
        name="gp_mined.close_mean60_over_close",
        expr="div(ts_mean(id(close), 60, None), id(close))",
        readable_expression="Mean($close, 60) / $close",
        qlib_expr="Mean($close, 60) / $close",
        source_yaml="20260318(1).yaml",
        original_factor_name="candidate_r004_a69dfa38cc",
        notes="Representative of the repeated 60-day mean-close over current-close family; scale-equivalent duplicates omitted.",
    ),
    GPMinedSpec(
        name="gp_mined.volume_mean60_over_volume",
        expr="div(ts_mean(id(volume), 60, None), add(id(volume), 1e-12))",
        readable_expression="Mean($volume, 60) / ($volume + 1e-12)",
        qlib_expr="Mean($volume, 60) / ($volume + 1e-12)",
        source_yaml="20260320(1).yaml",
        original_factor_name="candidate_r006_1585f2e0dd",
        notes="60-day average volume relative to current volume.",
    ),
    GPMinedSpec(
        name="gp_mined.down_volume_share_30",
        expr="div(ts_sum(greater(sub(delay(id(volume), 1), id(volume)), 0), 30), add(ts_sum(abs(sub(id(volume), delay(id(volume), 1))), 30), 1e-12))",
        readable_expression="30-day shrinking-volume share",
        qlib_expr="Sum(Greater(Ref($volume, 1) - $volume, 0), 30) / (Sum(Abs($volume - Ref($volume, 1)), 30) + 1e-12)",
        source_yaml="20260318(1).yaml",
        original_factor_name="candidate_r006_350fe5c983",
        notes="Ratio of shrinking-volume days to total 30-day absolute volume movement.",
    ),
    GPMinedSpec(
        name="gp_mined.volume_mean30_over_volume",
        expr="div(ts_mean(id(volume), 30, None), add(id(volume), 1e-12))",
        readable_expression="Mean($volume, 30) / ($volume + 1e-12)",
        qlib_expr="Mean($volume, 30) / ($volume + 1e-12)",
        source_yaml="20260318(1).yaml",
        original_factor_name="candidate_r008_aaab7b9a25",
        notes="Representative of the repeated 30-day average-volume over current-volume family; constant-scale variants omitted.",
    ),
    GPMinedSpec(
        name="gp_mined.low_over_high",
        expr="div(id(low), id(high))",
        readable_expression="$low / $high",
        qlib_expr="$low / $high",
        source_yaml="20260320(1).yaml",
        original_factor_name="candidate_r003_4b32c0f931",
        notes="Simple intraday low-to-high ratio; repeatedly shows high IC in the YAML scan.",
    ),
    GPMinedSpec(
        name="gp_mined.low_minus_high_over_prev_close",
        expr="div(sub(id(low), id(high)), add(delay(id(close), 1), 1e-12))",
        readable_expression="($low - $high) / Ref($close, 1)",
        qlib_expr="($low - $high) / Ref($close, 1)",
        source_yaml="20260321.yaml",
        original_factor_name="candidate_r009_db65f0e18f",
        notes="Intraday range scaled by prior close; closely related to low/high family but kept as a distinct signal.",
    ),
    GPMinedSpec(
        name="gp_mined.low5min_over_close",
        expr="div(ts_min(id(low), 5), id(close))",
        readable_expression="Min($low, 5) / $close",
        qlib_expr="Min($low, 5) / $close",
        source_yaml="20260320(1).yaml",
        original_factor_name="candidate_r005_df58110db3",
        notes="5-day rolling low relative to current close.",
    ),
    GPMinedSpec(
        name="gp_mined.low10min_over_close",
        expr="div(ts_min(id(low), 10), id(close))",
        readable_expression="Min($low, 10) / $close",
        qlib_expr="Min($low, 10) / $close",
        source_yaml="20260318(1).yaml",
        original_factor_name="candidate_r003_5148ccd921",
        notes="10-day rolling low relative to current close; repeated several times across YAML runs.",
    ),
    GPMinedSpec(
        name="gp_mined.low_over_close",
        expr="div(id(low), id(close))",
        readable_expression="$low / $close",
        qlib_expr="$low / $close",
        source_yaml="20260317.yaml",
        original_factor_name="gp_r01_70d6b538a2",
        notes="Classic low/close signal that appeared most often across the YAML scan.",
    ),
    GPMinedSpec(
        name="gp_mined.close_quantile10_over_close",
        expr="div(ts_quantile(id(close), 10, 0.2), id(close))",
        readable_expression="Quantile($close, 10, 0.2) / $close",
        qlib_expr="Quantile($close, 10, 0.2) / $close",
        source_yaml="20260317.yaml",
        original_factor_name="gp_r04_fba16d21a6",
        notes="10-day close quantile over current close.",
    ),
    GPMinedSpec(
        name="gp_mined.open_quantile5_over_close",
        expr="div(ts_quantile(id(open), 5, 0.2), id(close))",
        readable_expression="Quantile($open, 5, 0.2) / $close",
        qlib_expr="Quantile($open, 5, 0.2) / $close",
        source_yaml="20260319(1).yaml",
        original_factor_name="candidate_r010_00e8cb0152",
        notes="5-day open-price lower quantile over current close.",
    ),
)


def _iter_leaf_nodes(node):
    if getattr(node, "op", None) == "id":
        yield node
        return
    for arg in getattr(node, "args", []):
        if hasattr(arg, "op") and hasattr(arg, "args"):
            yield from _iter_leaf_nodes(arg)


def _make_expr_factor(parsed_node, factor_name: str, factor_fields: tuple[str, ...]):
    def _factor(data: dict[str, pd.Series]) -> pd.Series:
        validate_data(data, required_fields=factor_fields)
        return compute_factor_series(parsed_node, data).rename(factor_name)

    return _factor


def _cov_ret_open_beta_r2_abs_node():
    return ExprNode(
        "abs",
        [
            ExprNode(
                "add",
                [
                    ExprNode(
                        "ts_cov",
                        [
                            ExprNode("id", ["returns"]),
                            ExprNode("id", ["open"]),
                            5,
                            5,
                        ],
                    ),
                    ExprNode(
                        "regression_rsq",
                        [
                            ExprNode("id", ["returns"]),
                            ExprNode("id", ["market_return"]),
                            60,
                            10,
                        ],
                    ),
                ],
            )
        ],
    )


def _build_node(spec: GPMinedSpec):
    if spec.name == "gp_mined.cov_ret_open_beta_r2_abs":
        return _cov_ret_open_beta_r2_abs_node()
    if spec.qlib_expr is None:
        raise ValueError(f"No qlib parser expression configured for {spec.name}")
    return parse_qlib_expr(spec.qlib_expr)


def register_gp_mined(registry: FactorRegistry) -> int:
    count = 0
    for spec in GP_MINED_SPECS:
        node = _build_node(spec)
        required_fields = tuple(sorted({str(arg.args[0]) for arg in _iter_leaf_nodes(node)}))
        notes = (
            f"readable={spec.readable_expression}; "
            f"source_yaml={spec.source_yaml}; "
            f"original_factor_name={spec.original_factor_name}; "
            f"{spec.notes}"
        )
        registry.register(
            spec.name,
            _make_expr_factor(node, spec.name, required_fields),
            source="gp_mined_2026_yaml_curated",
            required_fields=required_fields,
            expr=spec.expr,
            notes=notes,
        )
        count += 1
    return count


def gp_mined_source_info() -> dict[str, object]:
    return {
        "source": "gp_mined_2026_yaml_curated",
        "status": "curated",
        "factor_count": len(GP_MINED_SPECS),
        "implemented_factors": tuple(spec.name for spec in GP_MINED_SPECS),
        "notes": (
            "These are curated representatives selected from /root/gp_factor_qlib/artifacts/2026*.yaml. "
            "Obvious constant-scale duplicates were intentionally collapsed into one registered signal."
        ),
    }
