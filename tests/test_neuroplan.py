import json
import tempfile
import unittest
from io import BytesIO, StringIO
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

import httpx
import numpy as np
import pandas as pd
from openai import OpenAI
from pydantic import ValidationError

from neuroplan.analysis import execute
from neuroplan.artifacts import save_run
from neuroplan.cli import main
from neuroplan.data import DATA_COLUMNS, load
from neuroplan.evaluation import _same_answer, reference_result
from neuroplan.planners import (
    DEFAULT_OLLAMA_MODEL, DEFAULT_VLLM_MODEL, KeywordPlanner, OllamaPlanner, OpenAIPlanner, VLLMPlanner,
)
from neuroplan.schema import AnalysisPlan, Decision

ROOT = Path(__file__).resolve().parents[1]
PAIRED = json.loads((ROOT / "examples/visp_visl.json").read_text())


def plan(**kwargs):
    return AnalysisPlan.model_validate_json(json.dumps(PAIRED | kwargs))


def fixture():
    # (unit, session, mouse, area, sorting label, cutoff, presence, ISI, gOSI, gDSI, Hz)
    return pd.DataFrame([
        (1, 1, 1, "VISp", "good", .10, .95, .5, .2, .1, 1.),
        (2, 1, 1, "VISp", "good", .02, 1., .0, .4, .3, 2.),
        (3, 1, 1, "VISl", "good", .02, 1., .0, .1, .4, 3.),
        (4, 2, 2, "VISp", "good", .02, 1., .0, .8, .5, 4.),
        (5, 2, 2, "VISl", "noise", .02, 1., .0, .6, .7, 5.),
        (6, 3, 3, "VISp", "good", .02, 1., .0, np.nan, .4, 5.),
        (7, 3, 3, "VISp", "good", .02, 1., .0, np.inf, .4, 5.),
    ], columns=DATA_COLUMNS)


class AnalysisTests(unittest.TestCase):
    def test_pairing_screen_and_signed_difference(self):
        result = execute(fixture(), plan())
        self.assertEqual(result["n_sessions"], 1)
        self.assertEqual(result["n_units_by_region"], {"VISp": 2, "VISl": 1})
        self.assertAlmostEqual(result["mean_paired_difference"], .2)
        all_result = execute(fixture(), plan(quality="all", regions=["VISl", "VISp"]))
        self.assertEqual(all_result["n_sessions"], 2)
        self.assertAlmostEqual(all_result["mean_paired_difference"], -.2)

    def test_units_minimum_and_missing_area(self):
        result = execute(fixture(), plan(min_units_per_region=2))
        self.assertEqual(result["rows"], [])
        self.assertIsNone(result["mean_paired_difference"])
        summary = execute(fixture(), plan(analysis="summary", regions=["VISal"], min_units_per_region=None))
        self.assertEqual(summary["rows"], [{"region": "VISal", "n_units": 0, "session_ids": [], "value": None}])

    def test_threshold_is_strict_and_metrics_must_be_finite(self):
        result = execute(fixture(), plan(analysis="fraction_above", regions=["VISp"],
                                         statistic=None, threshold=.4, min_units_per_region=None))
        self.assertEqual(result["rows"][0]["n_units"], 3)
        self.assertEqual(result["rows"][0]["n_above"], 1)
        self.assertEqual(result["rows"][0]["value"], 1 / 3)

    def test_impossible_plans_are_rejected(self):
        for changes in ({"regions": ["VISp", "VISp"]}, {"regions": ["CA1"]},
                        {"stimulus": "static_gratings"}, {"min_units_per_region": 0},
                        {"analysis": "fraction_above", "statistic": None, "threshold": 1.5,
                         "min_units_per_region": None}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                plan(**changes)

    def test_repeated_execution_saves_identical_result_and_figure(self):
        with tempfile.TemporaryDirectory() as root:
            a, b = Path(root) / "a", Path(root) / "b"
            manifest = {"sha256": "fixture"}
            save_run(a, fixture(), manifest, plan(), {"backend": "manual"})
            archived = json.loads((a / "run.json").read_text())
            restored = AnalysisPlan.model_validate_json(json.dumps(archived["plan"]))
            save_run(b, fixture(), manifest, restored, {"replayed": True})
            for name in ("result.json", "figure.png"):
                self.assertEqual((a / name).read_bytes(), (b / name).read_bytes())


class PlannerTests(unittest.TestCase):
    def test_keyword_misreads_metric_correction(self):
        response = KeywordPlanner().predict("In V1, report median orientation selectivity, not direction selectivity.")
        self.assertEqual(response.decision.plan.metric.value, "g_dsi_dg")

    def test_openai_structured_output_uses_real_sdk_against_mock_http(self):
        content = Decision(plan=plan(), clarification=None, question=None).model_dump_json()
        sent = []

        def handler(request):
            body = json.loads(request.content)
            sent.append(body)
            return httpx.Response(200, json={
                "id": "resp_test", "object": "response", "model": "test-model",
                "status": "completed", "created_at": 0, "parallel_tool_calls": False,
                "tool_choice": "auto", "tools": [], "output": [{
                    "id": "msg_test", "status": "completed", "type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": content, "annotations": []}],
                }],
            })

        with httpx.Client(transport=httpx.MockTransport(handler)) as http:
            client = OpenAI(api_key="test-key", base_url="https://example.com/v1", http_client=http)
            prediction = OpenAIPlanner("test-model", client).predict("Compare V1 and LM by session.")
        self.assertEqual(prediction.decision.plan, plan())
        self.assertEqual(prediction.provenance["returned_model"], "test-model")
        self.assertEqual(sent[0]["text"]["format"]["type"], "json_schema")
        self.assertTrue(sent[0]["text"]["format"]["strict"])
        self.assertFalse(sent[0]["store"])


class OllamaTests(unittest.TestCase):
    def setUp(self):
        self.model = DEFAULT_OLLAMA_MODEL
        self.models = [{"name": self.model, "digest": "test-digest",
                        "details": {"quantization_level": "Q4_K_M"}}]
        self.reply = {
            "model": self.model, "done": True, "done_reason": "stop",
            "message": {"content": Decision(plan=plan(), clarification=None, question=None).model_dump_json()},
            "prompt_eval_count": 200, "eval_count": 75, "total_duration": 1000,
        }
        self.requests = []
        opener = self.enterContext(patch("neuroplan.planners.build_opener"))
        opener.return_value.open.side_effect = self.mock_response

    def mock_response(self, request, *, timeout):
        self.requests.append(request)
        if request.full_url == "http://127.0.0.1:11434/api/tags":
            self.assertEqual(request.get_method(), "GET")
            return BytesIO(json.dumps({"models": self.models}).encode())
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(request.get_method(), "POST")
        return BytesIO(json.dumps(self.reply).encode())

    def test_cli_ask_and_benchmark_use_ollama_and_save_provenance(self):
        manifest = {"sha256": "fixture"}
        with tempfile.TemporaryDirectory() as root:
            out = Path(root)
            questions = out / "questions.jsonl"
            questions.write_text(json.dumps({"id": "test", "category": "paired",
                                            "question": "Compare V1 and LM by session.",
                                            "plan": PAIRED}) + "\n")
            commands = [
                ["ask", "Compare V1 and LM by session.", "--out", str(out / "ask")],
                ["benchmark", "--questions", str(questions), "--out", str(out / "benchmark")],
            ]
            for command in commands:
                with patch("neuroplan.cli.load", return_value=(fixture(), manifest)), redirect_stdout(StringIO()):
                    with patch("sys.argv", ["neuroplan", *command, "--backend", "ollama"]):
                        self.assertEqual(main(), 0)
            run = json.loads((out / "ask/run.json").read_text())
            report = json.loads((out / "benchmark/benchmark.json").read_text())
        self.assertEqual(run["provenance"]["model_digest"], "test-digest")
        self.assertEqual(run["provenance"]["quantization"], "Q4_K_M")
        self.assertEqual(report["scores"]["numerical_result"], {"correct": 1, "total": 1})
        self.assertEqual(report["cases"][0]["provenance"]["generation"]["num_ctx"], 4096)
        payload = json.loads(self.requests[1].data)
        self.assertEqual(payload["model"], self.model)
        self.assertEqual(payload["format"], Decision.model_json_schema())
        self.assertFalse(payload["think"])
        self.assertFalse(payload["stream"])

    def test_invalid_or_truncated_model_output_is_rejected(self):
        planner = OllamaPlanner()
        self.reply["done_reason"] = "length"
        with self.assertRaisesRegex(ValueError, "did not finish normally"):
            planner.predict("Compare V1 and LM by session.")
        self.reply["done_reason"] = "stop"
        self.reply["message"]["content"] = "{}"
        with self.assertRaises(ValidationError):
            planner.predict("Compare V1 and LM by session.")

    def test_missing_model_reports_pull_command(self):
        self.models.clear()
        with self.assertRaisesRegex(ValueError, "ollama pull qwen3.5:4b"):
            OllamaPlanner()


class VLLMTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.choice = {
            "index": 0, "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": Decision(plan=plan(), clarification=None, question=None).model_dump_json(),
            },
        }
        http = self.enterContext(httpx.Client(transport=httpx.MockTransport(self.mock_response)))
        self.enterContext(patch("openai.OpenAI", side_effect=lambda **kwargs: OpenAI(http_client=http, **kwargs)))
        self.enterContext(patch.dict("os.environ", {
            "VLLM_BASE_URL": "", "VLLM_API_KEY": "", "OPENAI_API_KEY": "unused-cloud-key",
        }))

    def mock_response(self, request):
        self.requests.append(request)
        return httpx.Response(200, json={
            "id": "chatcmpl-test", "object": "chat.completion", "created": 1,
            "model": "test-qwen", "choices": [self.choice], "system_fingerprint": "test-server",
            "usage": {"prompt_tokens": 200, "completion_tokens": 75, "total_tokens": 275},
        })

    def test_cli_ask_and_benchmark_use_schema_and_save_provenance(self):
        question = "Compare V1 and LM within sessions."
        with tempfile.TemporaryDirectory() as root:
            out = Path(root)
            questions = out / "questions.jsonl"
            questions.write_text(json.dumps({"id": "test", "category": "paired",
                                            "question": question, "plan": PAIRED}) + "\n")
            for command in (["ask", question], ["benchmark", "--questions", str(questions)]):
                argv = ["neuroplan", *command, "--backend", "vllm", "--out", str(out / command[0])]
                with (patch("neuroplan.cli.load", return_value=(fixture(), {"sha256": "fixture"})),
                      patch("sys.argv", argv), redirect_stdout(StringIO())):
                    self.assertEqual(main(), 0)
            run = json.loads((out / "ask/run.json").read_text())
            report = json.loads((out / "benchmark/benchmark.json").read_text())
        self.assertEqual(run["provenance"]["system_fingerprint"], "test-server")
        self.assertEqual(run["provenance"]["usage"]["completion_tokens"], 75)
        self.assertEqual(report["scores"]["numerical_result"], {"correct": 1, "total": 1})
        self.assertEqual(len(self.requests), 2)
        request = self.requests[0]
        self.assertEqual(str(request.url), "http://127.0.0.1:8000/v1/chat/completions")
        self.assertEqual(request.headers["authorization"], "Bearer EMPTY")
        payload = json.loads(request.content)
        self.assertEqual(payload["model"], DEFAULT_VLLM_MODEL)
        schema = payload["response_format"]["json_schema"]["schema"]
        executable, clarification = schema["anyOf"]
        self.assertEqual(executable["properties"], {
            "plan": {"$ref": "#/$defs/AnalysisPlan"},
            "clarification": {"type": "null"},
            "question": {"type": "null"},
        })
        self.assertEqual(clarification["properties"], {
            "plan": {"type": "null"},
            "clarification": {"$ref": "#/$defs/Clarification"},
            "question": {"type": "string"},
        })
        for branch in (executable, clarification):
            self.assertCountEqual(branch["required"], ["plan", "clarification", "question"])
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(payload["max_completion_tokens"], 512)

    def test_incomplete_and_invalid_decisions_are_rejected(self):
        planner = VLLMPlanner()
        self.choice["finish_reason"] = "length"
        with self.assertRaisesRegex(ValueError, "finish reason: length"):
            planner.predict("Compare V1 and LM within sessions.")
        self.choice["finish_reason"] = "stop"
        self.choice["message"]["content"] = "{}"
        with self.assertRaises(ValidationError):
            planner.predict("Compare V1 and LM within sessions.")

    def test_mixed_plan_and_question_still_fail_runtime_validation(self):
        response = {"plan": PAIRED, "clarification": None, "question": "Compare within each recording session."}
        self.choice["message"]["content"] = json.dumps(response)
        with self.assertRaisesRegex(ValidationError, "An executable decision has only a plan"):
            VLLMPlanner().predict("Compare V1 and LM within sessions.")


class AllenIntegrationTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / "data/allen_units.csv").exists(), "prepare Allen data first")
    def test_every_reference_plan_agrees_with_independent_reference(self):
        frame, _ = load(ROOT / "data/allen_units.csv")
        records = frame.to_dict("records")
        for line in (ROOT / "examples/benchmark.jsonl").read_text().splitlines():
            case = json.loads(line)
            if "plan" in case:
                with self.subTest(case=case["id"]):
                    expected_plan = AnalysisPlan.model_validate_json(json.dumps(case["plan"]))
                    expected = reference_result(records, expected_plan)
                    self.assertTrue(_same_answer(execute(frame, expected_plan), expected))


if __name__ == "__main__":
    unittest.main()
