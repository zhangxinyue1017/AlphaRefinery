from __future__ import annotations

import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from factors_store.llm_refine.cli.run_refine_loop import (
    _build_parse_retry_user_prompt,
    _rebuild_and_validate_candidates,
    _resolve_run_available_fields,
    _validate_focused_refine_request,
)
from factors_store.llm_refine.core.archive import init_archive_db
from factors_store.llm_refine.core.models import LLMProposal, RefinementCandidate
from factors_store.llm_refine.core.seed_loader import load_seed_pool
from factors_store.llm_refine.knowledge.round1 import (
    resolve_requested_candidate_count,
    resolve_round1_role_slots,
)
from factors_store.llm_refine.parsing.validator import validate_expression
from factors_store.llm_refine.prompting.prompt_builder import build_refinement_prompt


REPO_ROOT = Path(__file__).resolve().parents[1]
US_CONFIG = REPO_ROOT / "config" / "us_research_pilot.yaml"


class Phase2DPromptValidationTests(unittest.TestCase):
    def test_current_parse_retry_keeps_english_word_limits(self) -> None:
        retry = _build_parse_retry_user_prompt(
            original_user_prompt="Generate exactly two candidates.",
            parse_error="invalid JSON",
            prompt_template_version="current_compact",
        )
        self.assertIn("at most 24 words", retry)
        self.assertNotIn("Chinese chars", retry)

    def test_current_prompt_is_english_and_uses_supplied_fields(self) -> None:
        seed_pool = replace(load_seed_pool(US_CONFIG), evaluation_protocol=None)
        family = seed_pool.get_family("open_volume_correlation")
        with tempfile.TemporaryDirectory() as tmp:
            archive_db = init_archive_db(Path(tmp) / "archive.db")
            prompt = build_refinement_prompt(
                seed_pool=seed_pool,
                family=family,
                n_candidates=2,
                requested_candidate_count=2,
                final_candidate_target=2,
                role_slots=("conservative",),
                current_parent_name="alpha101.alpha003",
                current_parent_expression="-corr(rank(open), rank(volume), 10)",
                archive_db=archive_db,
                stage_mode="focused_refine",
                prompt_template_version="current_compact",
                available_fields=("open", "high", "low", "close", "volume", "vwap", "returns"),
            )

        assembled = f"{prompt.system_prompt}\n{prompt.user_prompt}"
        self.assertIsNone(re.search(r"[\u4e00-\u9fff]", assembled))
        self.assertIn("Daily US equities using split-adjusted Alpaca bars.", assembled)
        self.assertIn("Signals are available only after the daily bar is complete.", assembled)
        self.assertIn("5-day forward-return horizon", assembled)
        self.assertIn('["close", "high", "low", "open", "returns", "volume", "vwap"]', assembled)
        self.assertNotIn('"pre_close"', assembled)
        self.assertIn("Generate exactly 2 candidates", assembled)
        self.assertNotIn("selection-stage metrics", assembled.lower())

    def test_legacy_prompt_ignores_runtime_field_override(self) -> None:
        seed_pool = replace(load_seed_pool(US_CONFIG), evaluation_protocol=None)
        family = seed_pool.get_family("open_volume_correlation")
        with tempfile.TemporaryDirectory() as tmp:
            archive_db = init_archive_db(Path(tmp) / "archive.db")
            prompt = build_refinement_prompt(
                seed_pool=seed_pool,
                family=family,
                archive_db=archive_db,
                prompt_template_version="legacy_v1",
                available_fields=("close",),
            )

        self.assertIn("A股", prompt.system_prompt)
        self.assertIn('"pre_close"', prompt.user_prompt)
        self.assertIn("40 Chinese characters", prompt.user_prompt)

    def test_actual_bundle_fields_come_from_panel_and_derived_returns(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        index = pd.MultiIndex.from_product(
            [dates, ["AAPL", "MSFT"]], names=["datetime", "instrument"]
        )
        panel = pd.DataFrame(
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1_000.0,
                "vwap": 100.2,
            },
            index=index,
        )
        with tempfile.TemporaryDirectory() as tmp:
            panel_path = Path(tmp) / "panel.parquet"
            panel.to_parquet(panel_path)
            fields = _resolve_run_available_fields(
                seed_pool=SimpleNamespace(
                    evaluation_defaults={"panel_path": str(panel_path), "infer_cap_fields": False}
                ),
                args=SimpleNamespace(panel_path="", benchmark_path=""),
            )

        self.assertEqual(fields, ("close", "high", "low", "open", "returns", "volume", "vwap"))

    def test_unavailable_pre_close_is_rejected_before_evaluation(self) -> None:
        fields = ("open", "high", "low", "close", "volume", "vwap", "returns")
        warnings = validate_expression("div(open, pre_close)", available_fields=fields)
        self.assertIn(
            "uses unavailable fields: ['pre_close']; use delay(close, 1) for previous close",
            warnings,
        )
        self.assertFalse(validate_expression("div(open, delay(close, 1))", available_fields=fields))

        proposal = LLMProposal(
            parent_factor="alpha101.alpha003",
            diagnosed_weaknesses=(),
            refinement_rationale="test",
            candidates=(
                RefinementCandidate(
                    name="bad_previous_close",
                    expression="div(open, pre_close)",
                    explanation="test",
                    family="open_volume_correlation",
                ),
            ),
        )
        _, valid, _, dropped = _rebuild_and_validate_candidates(
            proposal=proposal,
            family_name="open_volume_correlation",
            effective_round_id=1,
            effective_parent_candidate_id="seed:test",
            available_fields=fields,
        )
        self.assertFalse(valid)
        self.assertEqual(len(dropped), 1)
        self.assertIn("delay(close, 1)", dropped[0]["filter_reason"])

    def test_focused_explicit_parent_has_no_seed_expansion(self) -> None:
        family = load_seed_pool(US_CONFIG).get_family("open_volume_correlation")
        roles = resolve_round1_role_slots(
            family=family,
            final_candidate_target=2,
            seed_stage_active=False,
        )
        requested = resolve_requested_candidate_count(
            final_candidate_target=2,
            role_slots=roles,
            seed_stage_active=False,
        )
        self.assertEqual(roles, ("conservative",))
        self.assertEqual(requested, 2)
        _validate_focused_refine_request(
            stage_mode="focused_refine",
            explicit_parent_name="alpha101.alpha003",
            seed_stage_active=False,
            requested_candidate_count=requested,
        )
        with self.assertRaisesRegex(ValueError, "seed-stage role expansion"):
            _validate_focused_refine_request(
                stage_mode="focused_refine",
                explicit_parent_name="alpha101.alpha003",
                seed_stage_active=True,
                requested_candidate_count=2,
            )
        with self.assertRaisesRegex(ValueError, "at most two"):
            _validate_focused_refine_request(
                stage_mode="focused_refine",
                explicit_parent_name="alpha101.alpha003",
                seed_stage_active=False,
                requested_candidate_count=3,
            )


if __name__ == "__main__":
    unittest.main()
