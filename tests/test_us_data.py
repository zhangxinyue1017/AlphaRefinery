from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from factors_store.data import build_data, load_panel
from factors_store.datahub.panel import PANEL_FIELDS, normalize_alpaca_bars, save_panel, validate_panel
from factors_store.eval import prepare_backtest_inputs, run_factor_backtest_dual
from factors_store.operators import wide_correlation
from factors_store.registry import create_default_registry


SYMBOLS = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM", "JNJ", "WMT")


def _synthetic_alpaca_frame(days: int = 16) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=days, tz="America/New_York")
    index = pd.MultiIndex.from_product([SYMBOLS, dates], names=["symbol", "timestamp"])
    rows: list[dict[str, float]] = []
    for symbol_index, _symbol in enumerate(SYMBOLS):
        for day_index, _date in enumerate(dates):
            base = 80.0 + 3.0 * symbol_index + 0.25 * day_index
            open_price = base + ((symbol_index + day_index) % 4) * 0.1
            close_price = base + ((2 * symbol_index + day_index) % 5 - 2) * 0.12
            rows.append(
                {
                    "open": open_price,
                    "high": max(open_price, close_price) + 0.8,
                    "low": min(open_price, close_price) - 0.8,
                    "close": close_price,
                    "volume": float(1_000_000 + 10_000 * symbol_index + 1_000 * day_index),
                    "trade_count": float(10_000 + day_index),
                    "vwap": (open_price + close_price) / 2.0,
                }
            )
    return pd.DataFrame(rows, index=index)


class USDataPanelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = _synthetic_alpaca_frame()
        self.panel = normalize_alpaca_bars(self.raw)

    def test_alpaca_response_normalization(self) -> None:
        self.assertEqual(tuple(self.panel.index.names), ("datetime", "instrument"))
        self.assertEqual(tuple(self.panel.columns), PANEL_FIELDS)
        self.assertIsNone(self.panel.index.get_level_values("datetime").tz)
        self.assertEqual(len(self.panel), len(self.raw))

    def test_index_fields_and_coverage_validation(self) -> None:
        report = validate_panel(self.panel, expected_symbols=SYMBOLS)
        self.assertEqual(report["symbol_count"], len(SYMBOLS))
        self.assertEqual(report["missing_symbols"], [])
        self.assertEqual(report["missing_observations_by_symbol"], {})

        duplicated = pd.concat([self.panel, self.panel.iloc[[0]]]).sort_index()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_panel(duplicated)

        with self.assertRaisesRegex(ValueError, "missing required fields"):
            validate_panel(self.panel.drop(columns="vwap"))

    def test_parquet_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "panel.parquet"
            save_panel(self.panel, path)
            loaded = load_panel(path)
        pd.testing.assert_frame_equal(loaded, self.panel)

    def test_existing_factor_registry_computes_from_us_panel(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "panel.parquet"
            save_panel(self.panel, path)
            data = build_data(path, infer_cap_fields=False)

        self.assertNotIn("cap", data)
        registry = create_default_registry()
        factor = registry.compute("alpha101.alpha003", data)
        self.assertEqual(factor.name, "alpha101.alpha003")
        self.assertEqual(len(factor), len(self.panel))
        self.assertTrue(np.isfinite(factor.dropna()).all())

    def test_us_evaluation_can_disable_proxy_exposures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "panel.parquet"
            save_panel(self.panel, path)
            data = build_data(path, infer_cap_fields=False)
        prepared = prepare_backtest_inputs(data, horizon=5, use_proxy_exposures=False)
        exposures = prepared["exposures"]
        self.assertIsInstance(exposures, pd.DataFrame)
        self.assertEqual(exposures.shape, (len(self.panel), 0))
        self.assertTrue(exposures.index.equals(data["close"].index))

    def test_us_evaluation_can_disable_neutralization(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "panel.parquet"
            save_panel(self.panel, path)
            data = build_data(path, infer_cap_fields=False)
        prepared = prepare_backtest_inputs(data, horizon=5, use_proxy_exposures=False)
        result = run_factor_backtest_dual(
            data["close"],
            prepared=prepared,
            factor_name="test.close",
            horizon=5,
            min_stocks=10,
            n_groups=5,
            enable_neutralization=False,
        )
        self.assertEqual(result["neutralization_status"], "disabled")
        self.assertIsNone(result["neutralized_result"])
        self.assertEqual(result["neutralized_factor_nonnull"], 0)

    def test_zero_variance_rolling_correlation_is_zero(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=12)
        values = np.tile(np.arange(1, 11, dtype=float) / 10.0, (len(dates), 1))
        constant_ranks = pd.DataFrame(values, index=dates, columns=SYMBOLS)
        result = wide_correlation(constant_ranks, constant_ranks, window=10)
        self.assertTrue((result.iloc[9:] == 0.0).all().all())

    def test_rolling_correlation_preserves_mathematical_ties(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=12)
        base = np.arange(len(dates), dtype=float)[:, None]
        scales = np.arange(1, len(SYMBOLS) + 1, dtype=float)[None, :]
        x = pd.DataFrame(base * scales, index=dates, columns=SYMBOLS)
        y = 3.0 * x + 7.0
        result = wide_correlation(x, y, window=10)
        self.assertTrue((result.iloc[9:] == 1.0).all().all())


if __name__ == "__main__":
    unittest.main()
