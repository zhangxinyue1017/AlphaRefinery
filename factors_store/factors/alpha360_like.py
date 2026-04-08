from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from ..contract import LIBRARY_REQUIRED_FIELDS, validate_data
from ..data import to_worldquant_frame, wide_frame_to_series
from ..operators import wide_delay as Ref
from ..registry import FactorRegistry

SOURCE_PATH = Path(__file__).resolve().parents[2] / "config" / "factor_manifests" / "alpha360.yaml"
EPS = 1e-12


class Alpha360:
    def __init__(self, df_data: pd.DataFrame):
        self.data_wide = df_data
        self.open_df = df_data["open"]
        self.high_df = df_data["high"]
        self.low_df = df_data["low"]
        self.close_df = df_data["close"]
        self.volume_df = df_data["volume"]
        self.vwap_df = df_data["vwap"]

        self.open = self.open_df
        self.high = self.high_df
        self.low = self.low_df
        self.close = self.close_df
        self.volume = self.volume_df
        self.vwap = self.vwap_df


def _load_factor_manifest() -> list[dict[str, object]]:
    payload = yaml.safe_load(SOURCE_PATH.read_text(encoding="utf-8"))
    return list(payload.get("factors", []))


FACTOR_MANIFEST = _load_factor_manifest()
FACTOR_NAMES = [str(item["name"]) for item in FACTOR_MANIFEST]
FACTOR_EXPRS = {str(item["name"]): str(item["expr"]) for item in FACTOR_MANIFEST}

FIELD_MAP = {
    "CLOSE": ("close", "close"),
    "OPEN": ("open", "close"),
    "HIGH": ("high", "close"),
    "LOW": ("low", "close"),
    "VWAP": ("vwap", "close"),
    "VOLUME": ("volume", "volume"),
}


def _parse_alpha360_name(name: str) -> tuple[str, int]:
    prefix = "".join(ch for ch in name if not ch.isdigit())
    digits = "".join(ch for ch in name if ch.isdigit())
    if prefix not in FIELD_MAP or not digits:
        raise ValueError(f"unsupported alpha360 factor name: {name}")
    return prefix, int(digits)


def _build_alpha360_method(name: str):
    prefix, lag = _parse_alpha360_name(name)
    numerator_field, denominator_field = FIELD_MAP[prefix]

    def _method(self: Alpha360, _lag: int = lag, _num: str = numerator_field, _den: str = denominator_field):
        numerator = getattr(self, _num)
        denominator = getattr(self, _den)
        if _lag == 0:
            return numerator / (denominator + EPS)
        return Ref(numerator, _lag) / (denominator + EPS)

    _method.__name__ = name
    _method.__qualname__ = f"Alpha360.{name}"
    _method.__doc__ = FACTOR_EXPRS[name]
    return _method


for _factor_name in FACTOR_NAMES:
    setattr(Alpha360, _factor_name, _build_alpha360_method(_factor_name))


def _factor_method_names() -> list[str]:
    return list(FACTOR_NAMES)


def _build_alpha360_input(data: dict[str, pd.Series]) -> pd.DataFrame:
    validate_data(data, required_fields=LIBRARY_REQUIRED_FIELDS["alpha360"])
    return to_worldquant_frame(data, fields=LIBRARY_REQUIRED_FIELDS["alpha360"])


def register_alpha360(registry: FactorRegistry) -> int:
    count = 0
    for method_name in _factor_method_names():
        factor_name = f"alpha360.{method_name}"

        def _make_factor(bound_name: str, bound_factor_name: str):
            def _factor(data: dict[str, pd.Series]) -> pd.Series:
                wide = _build_alpha360_input(data)
                engine = Alpha360(wide)
                result = getattr(engine, bound_name)()
                return wide_frame_to_series(result, name=bound_factor_name)

            return _factor

        registry.register(
            factor_name,
            _make_factor(method_name, factor_name),
            source="alpha360",
            required_fields=LIBRARY_REQUIRED_FIELDS["alpha360"],
            expr=FACTOR_EXPRS.get(method_name),
            notes=f"wide_frame_impl::{method_name}",
        )
        count += 1
    return count


def alpha360_source_info() -> dict[str, object]:
    return {
        "source": "alpha360",
        "path": str(SOURCE_PATH),
        "n_defs": len(FACTOR_NAMES),
        "required_fields": LIBRARY_REQUIRED_FIELDS["alpha360"],
        "notes": "内部统一转成 wide DataFrame，并按 Alpha360 模板自动生成命名因子方法。",
    }
