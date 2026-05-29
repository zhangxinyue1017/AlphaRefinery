from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
