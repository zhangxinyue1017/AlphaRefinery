from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


class RuntimeSmokeTests(unittest.TestCase):
    def test_empty_llm_refined_imports(self) -> None:
        from factors_store.factors import llm_refined

        self.assertTrue(hasattr(llm_refined, "FACTOR_SPECS"))
        self.assertIsInstance(llm_refined.FACTOR_SPECS, tuple)

    def test_default_registry_builds(self) -> None:
        from factors_store.registry import create_default_registry

        registry = create_default_registry()
        self.assertGreater(len(registry.names()), 0)

    def test_core_cli_help_runs(self) -> None:
        env = os.environ.copy()
        env["MPLCONFIGDIR"] = str(REPO_ROOT / ".matplotlib_cache")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "factors_store.llm_refine.cli.run_refine_loop",
                "--help",
            ],
            cwd=REPO_ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--family", result.stdout)

    def test_promotion_defaults_to_pending_only(self) -> None:
        from factors_store.llm_refine.config import DEFAULT_AUTO_APPLY_PROMOTION

        self.assertFalse(DEFAULT_AUTO_APPLY_PROMOTION)

    def test_provider_single_request_mode_records_usage(self) -> None:
        from factors_store.llm_refine.core.models import LLMProviderConfig
        from factors_store.llm_refine.core.providers import OpenAICompatProvider

        response = SimpleNamespace(
            id="response-1",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content='{"ok": true}', reasoning_content=None, refusal=None),
                )
            ],
        )
        client = MagicMock()
        client.chat.completions.create.return_value = response
        with patch.dict(os.environ, {"LLM_DISABLE_PROVIDER_RETRIES": "1"}), patch(
            "openai.OpenAI", return_value=client
        ) as openai_client:
            provider = OpenAICompatProvider(
                LLMProviderConfig(
                    name="openai",
                    base_url="https://example.test/v1",
                    api_key="test",
                    model="gpt-5.6-sol",
                )
            )
            content = provider.generate([{"role": "user", "content": "test"}])

        self.assertEqual(content, '{"ok": true}')
        self.assertEqual(provider.telemetry()["request_count"], 1)
        self.assertEqual(provider.telemetry()["usage"]["total_tokens"], 15)
        self.assertEqual(provider.telemetry()["token_limit_parameter"], "max_completion_tokens")
        self.assertFalse(provider.telemetry()["temperature_sent"])
        self.assertEqual(openai_client.call_args.kwargs["max_retries"], 0)
        request_kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request_kwargs["max_completion_tokens"], 3000)
        self.assertNotIn("max_tokens", request_kwargs)
        self.assertNotIn("temperature", request_kwargs)

    def test_provider_single_request_mode_does_not_retry(self) -> None:
        from factors_store.llm_refine.core.models import LLMProviderConfig
        from factors_store.llm_refine.core.providers import OpenAICompatProvider

        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("429 rate limit")
        with patch.dict(os.environ, {"LLM_DISABLE_PROVIDER_RETRIES": "true"}), patch(
            "openai.OpenAI", return_value=client
        ):
            provider = OpenAICompatProvider(
                LLMProviderConfig(name="test", base_url="https://example.test/v1", api_key="test", model="test-model")
            )
            with self.assertRaisesRegex(RuntimeError, "429"):
                provider.generate([{"role": "user", "content": "test"}])

        self.assertEqual(client.chat.completions.create.call_count, 1)
        self.assertEqual(provider.telemetry()["request_count"], 1)


if __name__ == "__main__":
    unittest.main()
