from __future__ import annotations

import contextlib
import logging
import os
import warnings
from typing import Any

import pandas as pd

from ..core.expression_tree import ExprNode
from ..core.operators_pro import OPERATOR_REGISTRY_PRO


DEFAULT_EVAL_BACKEND = "python"
_QLIB_RUNTIME_CACHE: dict[str, object] = {
    "provider_uri": None,
    "region": None,
}


def eval_node(node: ExprNode, data_ctx: dict[str, pd.Series]) -> pd.Series:
    if node.op == "id":
        return data_ctx[node.args[0]]

    fn = OPERATOR_REGISTRY_PRO[node.op]
    values: list[Any] = []
    for arg in node.args:
        if isinstance(arg, ExprNode):
            values.append(eval_node(arg, data_ctx))
        else:
            values.append(arg)
    while values and values[-1] is None:
        values.pop()
    return fn(*values)


def _import_qlib_runtime():
    import qlib

    if not hasattr(qlib, "init"):
        raise ImportError(
            "qlib backend requires Microsoft Qlib (`pyqlib`), but the current `qlib` package "
            f"is {getattr(qlib, '__file__', '<unknown>')}"
        )

    from qlib.constant import REG_CN
    from qlib.data import D

    logging.getLogger("qlib").setLevel(logging.WARNING)
    logging.getLogger("qlib.Initialization").setLevel(logging.WARNING)
    logging.getLogger("qlib.workflow").setLevel(logging.WARNING)

    return qlib, REG_CN, D


def _ensure_qlib_provider(qlib_module, provider_uri: str, region) -> None:
    cached_provider = _QLIB_RUNTIME_CACHE.get("provider_uri")
    cached_region = _QLIB_RUNTIME_CACHE.get("region")
    if cached_provider == provider_uri and cached_region == region:
        return
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            qlib_module.init(provider_uri=provider_uri, region=region, skip_if_reg=False)
    _QLIB_RUNTIME_CACHE["provider_uri"] = provider_uri
    _QLIB_RUNTIME_CACHE["region"] = region


def _compute_factor_with_qlib(node: ExprNode, data_ctx: dict[str, object]) -> pd.Series:
    segments = data_ctx.get("_qlib_segments")
    target_index = data_ctx.get("_qlib_target_index")
    if not isinstance(segments, list) or not segments:
        raise ValueError("qlib backend requires data_ctx['_qlib_segments']")
    if not isinstance(target_index, pd.MultiIndex):
        raise ValueError("qlib backend requires data_ctx['_qlib_target_index']")

    expr = node.to_qlib_expr()
    qlib, region, D = _import_qlib_runtime()
    chunks: list[pd.Series] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        provider_uri = str(seg.get("provider_uri") or "").strip()
        instruments = list(seg.get("instruments") or [])
        start_time = str(seg.get("start_time") or "")
        end_time = str(seg.get("end_time") or "")
        if not provider_uri or not instruments or not start_time or not end_time:
            continue
        _ensure_qlib_provider(qlib, provider_uri, region)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="divide by zero encountered in log",
            )
            df = D.features(
                instruments,
                [expr],
                start_time=start_time,
                end_time=end_time,
            )
        df.columns = ["factor"]
        series = df["factor"]
        if isinstance(series.index, pd.MultiIndex) and tuple(series.index.names) == ("instrument", "datetime"):
            series = series.swaplevel(0, 1).sort_index()
            series.index = series.index.set_names(["datetime", "instrument"])
        chunks.append(series)

    if not chunks:
        raise ValueError("qlib backend produced no factor chunks")

    factor = pd.concat(chunks).sort_index()
    factor.index = factor.index.set_names(["datetime", "instrument"])
    return factor.reindex(target_index).rename("factor")


def compute_factor_series(
    node: ExprNode,
    data_ctx: dict[str, object],
    eval_backend: str = DEFAULT_EVAL_BACKEND,
) -> pd.Series:
    backend = (eval_backend or DEFAULT_EVAL_BACKEND).strip().lower()
    if backend == "qlib":
        return _compute_factor_with_qlib(node, data_ctx)
    return eval_node(node, data_ctx)  # type: ignore[arg-type]

