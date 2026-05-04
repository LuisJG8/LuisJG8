from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = ROOT / "scripts" / "update_model_usage.py"
FIXTURES = ROOT / "tests" / "fixtures"


def load_module():
    spec = importlib.util.spec_from_file_location("update_model_usage", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load update_model_usage module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestUpdateModelUsage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_module()

    def make_args(self, **overrides):
        values = {
            "readme": "README.md",
            "json_output": "data/model-usage.json",
            "input_json": None,
            "since": None,
            "until": None,
            "timeout": 120,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def load_fixture(self, name: str):
        with (FIXTURES / name).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_parser_happy_path_breakdown(self):
        payload = self.load_fixture("codex_daily_breakdown.json")
        rows = self.mod.normalize_rows(payload)
        summary = self.mod.build_summary(rows, "mock")

        self.assertEqual(summary["period"]["from"], "2026-03-15")
        self.assertEqual(summary["period"]["to"], "2026-03-16")
        self.assertEqual(summary["totals"]["total_tokens"], 27800)
        self.assertEqual(summary["models"][0]["model"], "gpt-5.3-codex")
        self.assertEqual(summary["models"][0]["total_tokens"], 23900)

    def test_parser_supports_alternate_shapes(self):
        daily_payload = self.load_fixture("codex_daily_model_breakdowns_list.json")
        rows = self.mod.normalize_rows(daily_payload)
        summary = self.mod.build_summary(rows, "mock")
        self.assertEqual(len(summary["models"]), 2)
        self.assertEqual(summary["models"][0]["model"], "gpt-5.3-codex")

        projects_payload = self.load_fixture("codex_projects_shape.json")
        rows = self.mod.normalize_rows(projects_payload)
        summary = self.mod.build_summary(rows, "mock")
        self.assertEqual(summary["totals"]["total_tokens"], 2000)
        self.assertEqual(summary["models"][0]["model"], "gpt-5.3-codex")

    def test_parser_supports_models_dict_shape(self):
        payload = self.load_fixture("codex_daily_models_dict.json")
        rows = self.mod.normalize_rows(payload)
        summary = self.mod.build_summary(rows, "mock")

        self.assertEqual(summary["period"]["from"], "2026-02-02")
        self.assertEqual(summary["period"]["to"], "2026-03-19")
        self.assertEqual(len(summary["models"]), 1)
        self.assertEqual(summary["models"][0]["model"], "gpt-5.2-codex")
        self.assertEqual(summary["models"][0]["input_tokens"], 3500 - 750)
        self.assertEqual(summary["models"][0]["cache_read_tokens"], 750)
        self.assertEqual(summary["models"][0]["reasoning_tokens"], 40)
        self.assertEqual(summary["models"][0]["cost_usd"], 0.44)

    def test_row_cost_distributed_when_model_cost_missing(self):
        payload = {
            "daily": [
                {
                    "date": "2026-03-19",
                    "costUSD": 1.0,
                    "models": {
                        "m1-codex": {"inputTokens": 80, "totalTokens": 80},
                        "m2-codex": {"inputTokens": 20, "totalTokens": 20},
                    },
                }
            ]
        }
        rows = self.mod.normalize_rows(payload)
        summary = self.mod.build_summary(rows, "mock")

        models = {item["model"]: item for item in summary["models"]}
        self.assertAlmostEqual(models["m1-codex"]["cost_usd"], 0.8, places=6)
        self.assertAlmostEqual(models["m2-codex"]["cost_usd"], 0.2, places=6)
        self.assertAlmostEqual(summary["totals"]["cost_usd"], 1.0, places=6)

    def test_only_top_three_codex_models_are_displayed(self):
        payload = {
            "daily": [
                {
                    "date": "2026-03-19",
                    "models": {
                        "gpt-5.4": {"inputTokens": 9999, "totalTokens": 9999},
                        "alpha-codex": {"inputTokens": 3000, "totalTokens": 3000},
                        "beta-codex": {"inputTokens": 2500, "totalTokens": 2500},
                        "gamma-codex": {"inputTokens": 2000, "totalTokens": 2000},
                        "delta-codex": {"inputTokens": 1500, "totalTokens": 1500},
                    },
                }
            ]
        }
        rows = self.mod.normalize_rows(payload)
        summary = self.mod.build_summary(rows, "mock")

        self.assertEqual([m["model"] for m in summary["models"]], ["alpha-codex", "beta-codex", "gamma-codex"])
        self.assertNotIn("gpt-5.4", [m["model"] for m in summary["models"]])
        self.assertEqual(summary["totals"]["total_tokens"], 7500)

    def test_no_data_message(self):
        summary = self.mod.build_summary([], "mock")
        readme_block = self.mod.render_usage_block(summary)
        self.assertIn("No Codex model usage data found yet", readme_block)
        self.assertEqual(summary["models"], [])

    def test_readme_table_contains_input_output_total_columns(self):
        payload = self.load_fixture("codex_daily_breakdown.json")
        rows = self.mod.normalize_rows(payload)
        summary = self.mod.build_summary(rows, "mock")
        readme_block = self.mod.render_usage_block(summary)

        self.assertNotIn("Last updated:", readme_block)
        self.assertNotIn("Showing top", readme_block)
        self.assertNotIn("Coverage:", readme_block)
        self.assertNotIn("Tracked **", readme_block)
        self.assertIn(
            "| Model | Input tokens | Output tokens | Cache read tokens | Total tokens |",
            readme_block,
        )
        self.assertIn("| `gpt-5.3-codex` | 4,100 | 16,900 | 2,700 | 23,900 |", readme_block)
        self.assertNotIn("$", readme_block)

    def test_read_source_json_missing_npx(self):
        args = self.make_args()
        with mock.patch.object(self.mod.subprocess, "run", side_effect=FileNotFoundError):
            with self.assertRaises(SystemExit) as ctx:
                self.mod.read_source_json(args)
        self.assertIn("Unable to run `npx`", str(ctx.exception))

    def test_read_source_json_non_zero_exit(self):
        args = self.make_args()
        completed = subprocess.CompletedProcess(
            args=["npx"], returncode=1, stdout="", stderr="boom"
        )
        with mock.patch.object(self.mod.subprocess, "run", return_value=completed):
            with self.assertRaises(SystemExit) as ctx:
                self.mod.read_source_json(args)
        self.assertIn("Failed to collect Codex usage data", str(ctx.exception))

    def test_read_source_json_invalid_json(self):
        args = self.make_args()
        completed = subprocess.CompletedProcess(
            args=["npx"], returncode=0, stdout="not-json", stderr=""
        )
        with mock.patch.object(self.mod.subprocess, "run", return_value=completed):
            with self.assertRaises(SystemExit) as ctx:
                self.mod.read_source_json(args)
        self.assertIn("did not return valid JSON", str(ctx.exception))

    def test_read_source_json_timeout(self):
        args = self.make_args()
        timeout = subprocess.TimeoutExpired(cmd=["npx"], timeout=10)
        with mock.patch.object(self.mod.subprocess, "run", side_effect=timeout):
            with self.assertRaises(SystemExit) as ctx:
                self.mod.read_source_json(args)
        self.assertIn("Timed out waiting for @ccusage/codex", str(ctx.exception))

    def test_readme_update_is_idempotent(self):
        block = "## Codex Model Spend\n\n_Last updated: 2026-03-19T00:00:00Z_"
        starting = (
            "# Profile\n\n"
            "Text before.\n\n"
            "<!-- MODEL_USAGE:START -->\n"
            "old\n"
            "<!-- MODEL_USAGE:END -->\n\n"
            "Text after.\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            readme_path = Path(temp_dir) / "README.md"
            readme_path.write_text(starting, encoding="utf-8")

            self.mod.update_readme(readme_path, block)
            first = readme_path.read_text(encoding="utf-8")
            self.mod.update_readme(readme_path, block)
            second = readme_path.read_text(encoding="utf-8")

        self.assertEqual(first, second)
        self.assertIn("Text before.", first)
        self.assertIn("Text after.", first)
        self.assertIn("Codex Model Spend", first)

    def test_workflow_contains_commit_guard(self):
        workflow = (ROOT / ".github" / "workflows" / "update-codex-usage.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("git diff --quiet -- README.md data/model-usage.json", workflow)
        self.assertIn("git add README.md data/model-usage.json", workflow)
        self.assertIn("runs-on: [self-hosted, codex-usage]", workflow)


if __name__ == "__main__":
    unittest.main()
