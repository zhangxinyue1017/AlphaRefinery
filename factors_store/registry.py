'''Factor registry construction and lookup utilities.

Registers built-in, seed, and refined factors and exposes compute helpers by factor name.
'''

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import yaml

from ._bootstrap import ensure_project_roots
from .contract import validate_data

ensure_project_roots()

from ._vendor.gpqlib_runtime.core.expression_tree import parse_qlib_expr  # noqa: E402
from ._vendor.gpqlib_runtime.engine.gp_engine import compute_factor_series  # noqa: E402


FactorFunc = Callable[[dict[str, pd.Series]], pd.Series]


@dataclass(frozen=True)
class FactorSpec:
    name: str
    func: FactorFunc
    source: str
    required_fields: tuple[str, ...]
    expr: str | None = None
    notes: str = ""


class FactorRegistry:
    def __init__(self) -> None:
        self._items: dict[str, FactorSpec] = {}
        self.skipped: list[dict[str, str]] = []

    def register(
        self,
        name: str,
        func: FactorFunc,
        *,
        source: str,
        required_fields: tuple[str, ...] | list[str],
        expr: str | None = None,
        notes: str = "",
    ) -> None:
        spec = FactorSpec(
            name=name,
            func=func,
            source=source,
            required_fields=tuple(required_fields),
            expr=expr,
            notes=notes,
        )
        self._items[name] = spec

    def register_expr_library(self, yaml_path: str | Path, *, source: str, prefix: str | None = None) -> int:
        path = Path(yaml_path)
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        count = 0
        for item in payload.get("factors", []):
            expr = str(item["expr"]).strip()
            raw_name = str(item["name"]).strip()
            if not expr or not raw_name:
                continue
            name = f"{prefix}.{raw_name}" if prefix else raw_name
            try:
                node = parse_qlib_expr(expr)
            except Exception as exc:
                self.skipped.append(
                    {
                        "name": name,
                        "source": source,
                        "expr": expr,
                        "reason": str(exc),
                    }
                )
                continue

            required_fields = tuple(sorted({str(arg.args[0]) for arg in _iter_leaf_nodes(node)}))

            def _make_factor(parsed_node, factor_name: str, factor_fields: tuple[str, ...]):
                def _factor(data: dict[str, pd.Series]) -> pd.Series:
                    validate_data(data, required_fields=factor_fields)
                    return compute_factor_series(parsed_node, data).rename(factor_name)

                return _factor

            self.register(
                name,
                _make_factor(node, name, required_fields),
                source=source,
                required_fields=required_fields,
                expr=expr,
            )
            count += 1
        return count

    def get(self, name: str) -> FactorSpec:
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items.keys())

    def find_names(
        self,
        *,
        source: str | None = None,
        source_prefix: str | None = None,
    ) -> list[str]:
        if source is not None and source_prefix is not None:
            raise ValueError("source and source_prefix are mutually exclusive")
        if source is not None:
            return sorted(name for name, spec in self._items.items() if spec.source == source)
        if source_prefix is not None:
            return sorted(name for name, spec in self._items.items() if spec.source.startswith(source_prefix))
        return self.names()

    def compute(self, name: str, data: dict[str, pd.Series]) -> pd.Series:
        spec = self.get(name)
        validate_data(data, required_fields=spec.required_fields)
        return spec.func(data).rename(name)

    def summary(self) -> pd.DataFrame:
        rows = [
            {
                "name": spec.name,
                "source": spec.source,
                "required_fields": ",".join(spec.required_fields),
                "expr": spec.expr or "",
                "notes": spec.notes,
            }
            for spec in self._items.values()
        ]
        return pd.DataFrame(rows).sort_values(["source", "name"]).reset_index(drop=True)


def _iter_leaf_nodes(node):
    if getattr(node, "op", None) == "id":
        yield node
        return
    for arg in getattr(node, "args", []):
        if hasattr(arg, "op") and hasattr(arg, "args"):
            yield from _iter_leaf_nodes(arg)


def create_default_registry() -> FactorRegistry:
    from .factors import (
        register_alpha101,
        register_alpha158,
        register_alpha191,
        register_alpha360,
        register_cicc_daily,
        register_factor365_daily,
        register_factor365_pattern,
        register_gp_mined,
        register_llm_refined,
        register_qp_behavior,
        register_qp_chip,
        register_qp_kline,
        register_qp_momentum,
        register_qp_path_convexity,
        register_qp_pressure,
        register_qp_salience,
        register_qp_volatility,
        register_seed_baselines,
    )

    registry = FactorRegistry()
    register_alpha101(registry)
    register_alpha158(registry)
    register_alpha191(registry)
    register_alpha360(registry)
    register_cicc_daily(registry)
    register_factor365_daily(registry)
    register_factor365_pattern(registry)
    register_gp_mined(registry)
    register_llm_refined(registry)
    register_qp_behavior(registry)
    register_qp_chip(registry)
    register_qp_momentum(registry)
    register_qp_path_convexity(registry)
    register_qp_pressure(registry)
    register_qp_volatility(registry)
    register_qp_kline(registry)
    register_qp_salience(registry)
    register_seed_baselines(registry)
    return registry
