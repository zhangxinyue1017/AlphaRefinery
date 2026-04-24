'''Alpha158-style feature family definitions.

Implements rolling price, volume, and return features used as baseline factor candidates.
'''

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from ..contract import LIBRARY_REQUIRED_FIELDS, validate_data
from ..data import to_worldquant_frame, wide_frame_to_series
from ..operators import (
    wide_abs as Abs,
    wide_correlation as Corr,
    wide_delay as Ref,
    wide_greater as Greater,
    wide_less as Less,
    wide_log as Log,
    wide_quantile as Quantile,
    wide_resi as Resi,
    wide_rsquare as Rsquare,
    wide_slope as Slope,
    wide_sma as Mean,
    wide_stddev as Std,
    wide_ts_argmax as IdxMax,
    wide_ts_argmin as IdxMin,
    wide_ts_max as TsMax,
    wide_ts_min as TsMin,
    wide_ts_rank as Rank,
    wide_ts_sum as Sum,
)
from ..registry import FactorRegistry

SOURCE_PATH = Path(__file__).resolve().parents[2] / "config" / "factor_manifests" / "alpha158.yaml"
EPS = 1e-12


class Alpha158:
    def __init__(self, df_data: pd.DataFrame):
        self.data_wide = df_data
        self.open_df = df_data["open"]
        self.high_df = df_data["high"]
        self.low_df = df_data["low"]
        self.close_df = df_data["close"]
        self.volume_df = df_data["volume"]
        self.vwap_df = df_data["vwap"]

        # Keep the short field names so the formulas read naturally.
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


def _window_from_name(name: str) -> int:
    digits = "".join(ch for ch in name if ch.isdigit())
    if not digits:
        raise ValueError(f"{name} does not contain a rolling window")
    return int(digits)


def _build_alpha158_method(name: str):
    def _kmid(self: Alpha158):
        return (self.close - self.open) / self.open

    def _klen(self: Alpha158):
        return (self.high - self.low) / self.open

    def _kmid2(self: Alpha158):
        return (self.close - self.open) / (self.high - self.low + EPS)

    def _kup(self: Alpha158):
        return (self.high - Greater(self.open, self.close)) / self.open

    def _kup2(self: Alpha158):
        return (self.high - Greater(self.open, self.close)) / (self.high - self.low + EPS)

    def _klow(self: Alpha158):
        return (Less(self.open, self.close) - self.low) / self.open

    def _klow2(self: Alpha158):
        return (Less(self.open, self.close) - self.low) / (self.high - self.low + EPS)

    def _ksft(self: Alpha158):
        return (2 * self.close - self.high - self.low) / self.open

    def _ksft2(self: Alpha158):
        return (2 * self.close - self.high - self.low) / (self.high - self.low + EPS)

    singleton_map = {
        "KMID": _kmid,
        "KLEN": _klen,
        "KMID2": _kmid2,
        "KUP": _kup,
        "KUP2": _kup2,
        "KLOW": _klow,
        "KLOW2": _klow2,
        "KSFT": _ksft,
        "KSFT2": _ksft2,
        "OPEN0": lambda self: self.open / self.close,
        "HIGH0": lambda self: self.high / self.close,
        "LOW0": lambda self: self.low / self.close,
        "VWAP0": lambda self: self.vwap / self.close,
    }
    if name in singleton_map:
        fn = singleton_map[name]
        fn.__name__ = name
        return fn

    window = _window_from_name(name)

    if name.startswith("ROC"):
        return lambda self, w=window: Ref(self.close, w) / self.close
    if name.startswith("MA"):
        return lambda self, w=window: Mean(self.close, w) / self.close
    if name.startswith("STD"):
        return lambda self, w=window: Std(self.close, w) / self.close
    if name.startswith("BETA"):
        return lambda self, w=window: Slope(self.close, w) / self.close
    if name.startswith("RSQR"):
        return lambda self, w=window: Rsquare(self.close, w)
    if name.startswith("RESI"):
        return lambda self, w=window: Resi(self.close, w) / self.close
    if name.startswith("MAX"):
        return lambda self, w=window: TsMax(self.high, w) / self.close
    if name.startswith("MIN"):
        return lambda self, w=window: TsMin(self.low, w) / self.close
    if name.startswith("QTLU"):
        return lambda self, w=window: Quantile(self.close, w, 0.8) / self.close
    if name.startswith("QTLD"):
        return lambda self, w=window: Quantile(self.close, w, 0.2) / self.close
    if name.startswith("RANK"):
        return lambda self, w=window: Rank(self.close, w)
    if name.startswith("RSV"):
        return lambda self, w=window: (self.close - TsMin(self.low, w)) / (TsMax(self.high, w) - TsMin(self.low, w) + EPS)
    if name.startswith("IMAX"):
        return lambda self, w=window: IdxMax(self.high, w) / w
    if name.startswith("IMIN"):
        return lambda self, w=window: IdxMin(self.low, w) / w
    if name.startswith("IMXD"):
        return lambda self, w=window: (IdxMax(self.high, w) - IdxMin(self.low, w)) / w
    if name.startswith("CORR"):
        return lambda self, w=window: Corr(self.close, Log(self.volume + 1.0), w)
    if name.startswith("CORD"):
        return lambda self, w=window: Corr(
            self.close / Ref(self.close, 1),
            Log(self.volume / Ref(self.volume, 1) + 1.0),
            w,
        )
    if name.startswith("CNTP"):
        return lambda self, w=window: (self.close > Ref(self.close, 1)).astype(float).rolling(w).mean()
    if name.startswith("CNTN"):
        return lambda self, w=window: (self.close < Ref(self.close, 1)).astype(float).rolling(w).mean()
    if name.startswith("CNTD"):
        return lambda self, w=window: (
            (self.close > Ref(self.close, 1)).astype(float).rolling(w).mean()
            - (self.close < Ref(self.close, 1)).astype(float).rolling(w).mean()
        )
    if name.startswith("SUMP"):
        return lambda self, w=window: Sum(Greater(self.close - Ref(self.close, 1), 0), w) / (
            Sum(Abs(self.close - Ref(self.close, 1)), w) + EPS
        )
    if name.startswith("SUMN"):
        return lambda self, w=window: Sum(Greater(Ref(self.close, 1) - self.close, 0), w) / (
            Sum(Abs(self.close - Ref(self.close, 1)), w) + EPS
        )
    if name.startswith("SUMD"):
        return lambda self, w=window: (
            Sum(Greater(self.close - Ref(self.close, 1), 0), w)
            - Sum(Greater(Ref(self.close, 1) - self.close, 0), w)
        ) / (Sum(Abs(self.close - Ref(self.close, 1)), w) + EPS)
    if name.startswith("VMA"):
        return lambda self, w=window: Mean(self.volume, w) / (self.volume + EPS)
    if name.startswith("VSTD"):
        return lambda self, w=window: Std(self.volume, w) / (self.volume + EPS)
    if name.startswith("WVMA"):
        return lambda self, w=window: Std(Abs(self.close / Ref(self.close, 1) - 1.0) * self.volume, w) / (
            Mean(Abs(self.close / Ref(self.close, 1) - 1.0) * self.volume, w) + EPS
        )
    if name.startswith("VSUMP"):
        return lambda self, w=window: Sum(Greater(self.volume - Ref(self.volume, 1), 0), w) / (
            Sum(Abs(self.volume - Ref(self.volume, 1)), w) + EPS
        )
    if name.startswith("VSUMN"):
        return lambda self, w=window: Sum(Greater(Ref(self.volume, 1) - self.volume, 0), w) / (
            Sum(Abs(self.volume - Ref(self.volume, 1)), w) + EPS
        )
    if name.startswith("VSUMD"):
        return lambda self, w=window: (
            Sum(Greater(self.volume - Ref(self.volume, 1), 0), w)
            - Sum(Greater(Ref(self.volume, 1) - self.volume, 0), w)
        ) / (Sum(Abs(self.volume - Ref(self.volume, 1)), w) + EPS)

    raise NotImplementedError(f"Unsupported alpha158 factor {name}")


for _factor_name in FACTOR_NAMES:
    _method = _build_alpha158_method(_factor_name)
    _method.__name__ = _factor_name
    _method.__qualname__ = f"Alpha158.{_factor_name}"
    _method.__doc__ = FACTOR_EXPRS[_factor_name]
    setattr(Alpha158, _factor_name, _method)


def _factor_method_names() -> list[str]:
    return list(FACTOR_NAMES)


def _build_alpha158_input(data: dict[str, pd.Series]) -> pd.DataFrame:
    validate_data(data, required_fields=LIBRARY_REQUIRED_FIELDS["alpha158"])
    return to_worldquant_frame(data, fields=LIBRARY_REQUIRED_FIELDS["alpha158"])


def register_alpha158(registry: FactorRegistry) -> int:
    count = 0
    for method_name in _factor_method_names():
        factor_name = f"alpha158.{method_name}"

        def _make_factor(bound_name: str, bound_factor_name: str):
            def _factor(data: dict[str, pd.Series]) -> pd.Series:
                wide = _build_alpha158_input(data)
                engine = Alpha158(wide)
                result = getattr(engine, bound_name)()
                return wide_frame_to_series(result, name=bound_factor_name)

            return _factor

        registry.register(
            factor_name,
            _make_factor(method_name, factor_name),
            source="alpha158",
            required_fields=LIBRARY_REQUIRED_FIELDS["alpha158"],
            expr=FACTOR_EXPRS.get(method_name),
            notes=f"wide_frame_impl::{method_name}",
        )
        count += 1
    return count


def alpha158_source_info() -> dict[str, object]:
    return {
        "source": "alpha158",
        "path": str(SOURCE_PATH),
        "n_defs": len(FACTOR_NAMES),
        "required_fields": LIBRARY_REQUIRED_FIELDS["alpha158"],
        "notes": "内部统一转成 wide DataFrame，并按 Alpha158 模板自动生成命名因子方法。",
    }
