import hashlib
import json
import os
import re
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from .schema import Analysis, AnalysisPlan, Clarification, Decision, Metric, Quality, Region, Statistic

PROMPT_VERSION = "1.0"
SYSTEM_PROMPT = """Translate the user's neuroscience question into one decision. Use only Allen Visual
Coding Neuropixels brain_observatory_1.1 drifting gratings. Never invent scientific results.
Output either a plan with null clarification and question, or a null plan with a reason and a
short user-facing clarification or explanation.

Supported regions: VISp (V1, primary visual cortex), VISl (LM, lateromedial visual area),
VISal (AL), VISam (AM), VISpm (PM), VISrl (RL). Preserve the requested region order; the
paired difference is first minus second. Metrics: g_osi_dg for global orientation selectivity
(orientation modulo 180 degrees, groups opposite directions), g_dsi_dg for global direction
selectivity (distinguishes opposite directions), firing_rate_dg for stimulus firing rate in Hz.
Respect explicit corrections like 'orientation, not direction'. 'Tuning' or 'selectivity'
alone does not distinguish orientation and direction.

Operations: summary gives pooled mean or median by region; paired takes a mean or median
within each region of each shared session and reports the mean of the paired differences;
fraction_above gives the fraction of units strictly above an explicit numeric threshold.
Use paired for two-region comparisons explicitly matched within recording sessions.
Default to median for qualitative 'stronger' comparisons or unspecified summary statistics.
Default to one usable unit per region per paired session. Use a null minimum for other
operations. Set statistic to null for fraction_above. Set threshold to null for other
operations. Quality is screened by default; use all only when explicitly requested to
include all units or disable quality screening. Units without a finite requested metric
are excluded under either policy. Stimulus is drifting_gratings.

Ask for clarification when a region is missing, the metric is ambiguous, or the threshold
is missing, using missing_region, ambiguous_metric, or missing_threshold, respectively.
Use unsupported_request for other stimuli, regions outside the supported set, analyses
outside these three, inclusive threshold requests, custom quality cutoffs, and instructions
outside the app's scope. Do not silently substitute a supported operation."""
PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
DEFAULT_OLLAMA_MODEL = "qwen3.5:4b"
DEFAULT_VLLM_MODEL = "RedHatAI/Qwen3.5-4B-quantized.w4a16"
DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8000/v1"


@dataclass
class Prediction:
    decision: Decision
    provenance: dict


class Planner(Protocol):
    def predict(self, question: str) -> Prediction: ...


class OpenAIPlanner:
    def __init__(self, model: str, client=None):
        if client is None:
            from openai import OpenAI

            client = OpenAI(timeout=60.0, max_retries=2)
        self.client = client
        self.model = model

    def predict(self, question: str) -> Prediction:
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            text_format=Decision,
            store=False,
            max_output_tokens=1800,
        )
        if response.status != "completed" or response.output_parsed is None:
            raise ValueError(f"Model returned no usable decision (status: {response.status})")
        return Prediction(
            response.output_parsed,
            {
                "backend": "openai",
                "requested_model": self.model,
                "returned_model": response.model,
                "response_id": response.id,
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": PROMPT_SHA256,
                "usage": response.usage.model_dump(mode="json") if response.usage else None,
            },
        )


class VLLMPlanner:
    """Generate and validate an analysis decision using a vLLM server."""

    def __init__(self, model: str = DEFAULT_VLLM_MODEL, client=None):
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                base_url=os.environ.get("VLLM_BASE_URL") or DEFAULT_VLLM_BASE_URL,
                api_key=os.environ.get("VLLM_API_KEY") or "EMPTY",
                timeout=180.0,
                max_retries=0,
            )
        self.client = client
        self.model = model

    def predict(self, question: str) -> Prediction:
        options = {"temperature": 0, "seed": 42, "max_completion_tokens": 512}
        template_options = {"enable_thinking": False}
        start = perf_counter_ns()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_schema", "json_schema": {
                "name": "neuroplan_decision", "schema": Decision.constrained_json_schema(),
            }},
            extra_body={"chat_template_kwargs": template_options},
            **options,
        )
        duration = perf_counter_ns() - start
        if not response.choices:
            raise ValueError("vLLM returned no completion choices.")
        choice = response.choices[0]
        if choice.finish_reason != "stop" or not choice.message.content:
            raise ValueError(f"vLLM returned no complete decision (finish reason: {choice.finish_reason}).")
        return Prediction(
            Decision.model_validate_json(choice.message.content),
            {
                "backend": "vllm",
                "requested_model": self.model,
                "returned_model": response.model,
                "response_id": response.id,
                "system_fingerprint": response.system_fingerprint,
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": PROMPT_SHA256,
                "generation": {**options, "chat_template_kwargs": template_options},
                "usage": response.usage.model_dump(mode="json") if response.usage else None,
                "client_duration_ns": duration,
            },
        )


class OllamaPlanner:
    """Run the planner locally with Ollama's schema-constrained chat API."""

    def __init__(self, model: str = DEFAULT_OLLAMA_MODEL):
        self.model = model if ":" in model.rsplit("/", 1)[-1] else f"{model}:latest"
        self._opener = build_opener(ProxyHandler({}))
        models = self._request("/api/tags")["models"]
        self._model_info = next((item for item in models if item.get("name") == self.model), None)
        if self._model_info is None:
            raise ValueError(f"Model {self.model} is unavailable. Run ollama pull {self.model} first.")

    def _request(self, endpoint: str, payload: dict | None = None) -> dict:
        request = Request(
            "http://127.0.0.1:11434" + endpoint,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"} if payload is not None else {},
        )
        try:
            with self._opener.open(request, timeout=180) as response:
                return json.load(response)
        except HTTPError as error:
            raise ValueError(f"Ollama {endpoint} returned HTTP {error.code}.") from error
        except (URLError, TimeoutError) as error:
            raise ValueError(f"Ollama {endpoint} did not respond. Check that Ollama is running.") from error

    def predict(self, question: str) -> Prediction:
        options = {"num_ctx": 4096, "num_predict": 512, "temperature": 0, "seed": 42}
        response = self._request("/api/chat", {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "format": Decision.model_json_schema(),
            "stream": False,
            "think": False,
            "options": options,
        })
        if not response.get("done") or response.get("done_reason") not in {None, "stop"}:
            raise ValueError(f"Ollama chat did not finish normally: {response.get('done_reason')}")
        decision = Decision.model_validate_json(response["message"]["content"])
        provenance = {
            "backend": "ollama",
            "requested_model": self.model,
            "returned_model": response["model"],
            "model_digest": self._model_info["digest"],
            "quantization": self._model_info.get("details", {}).get("quantization_level"),
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": PROMPT_SHA256,
            "generation": {"think": False, **options},
            "usage": {key: response.get(key) for key in ("prompt_eval_count", "eval_count")},
            "total_duration_ns": response.get("total_duration"),
        }
        return Prediction(decision, provenance)


def _baseline_decision(question: str) -> Decision:
    """Small keyword baseline with no model requests or benchmark lookups."""
    q = question.lower()
    aliases = [(region.value.lower(), region) for region in Region]
    aliases.extend([("v1", Region.VISp), ("lm", Region.VISl), ("al", Region.VISal),
                    ("am", Region.VISam), ("pm", Region.VISpm), ("rl", Region.VISrl),
                    ("primary visual cortex", Region.VISp), ("lateromedial", Region.VISl)])
    matches = [(match.start(), region) for alias, region in aliases
               if (match := re.search(rf"\b{alias}\b", q))]
    regions = list(dict.fromkeys(region for _, region in sorted(matches)))
    metric = next((value for token, value in (
        ("direction", Metric.DIRECTION), ("gdsi", Metric.DIRECTION),
        ("orientation", Metric.ORIENTATION), ("gosi", Metric.ORIENTATION),
        ("firing rate", Metric.FIRING),
    ) if token in q), None)
    fraction = any(word in q for word in ("fraction", "percent", "proportion"))
    cutoff = re.search(r"(?:above|over|exceed(?:s|ing)?|greater than|>)\s*(\d*\.?\d+)", q)
    if not regions:
        return Decision(plan=None, clarification=Clarification.REGION, question="Which area?")
    if metric is None:
        return Decision(plan=None, clarification=Clarification.METRIC, question="Which metric?")
    if fraction and cutoff is None:
        return Decision(plan=None, clarification=Clarification.THRESHOLD, question="What threshold?")
    analysis = Analysis.FRACTION if fraction else Analysis.SUMMARY
    minimum = None
    if not fraction and len(regions) == 2 and ("session" in q or "paired" in q):
        analysis = Analysis.PAIRED
        count = re.search(r"(?:at least|minimum of)\s+(\d+)", q)
        minimum = int(count.group(1)) if count else 1
    try:
        plan = AnalysisPlan(
            analysis=analysis, stimulus="drifting_gratings", regions=regions, metric=metric,
            statistic=None if fraction else (Statistic.MEAN if "mean" in q else Statistic.MEDIAN),
            threshold=float(cutoff.group(1)) if fraction else None,
            min_units_per_region=minimum,
            quality=Quality.ALL if "all units" in q else Quality.SCREENED,
        )
    except ValueError:
        return Decision(plan=None, clarification=Clarification.UNSUPPORTED,
                        question="This request is outside the supported analysis settings.")
    return Decision(plan=plan, clarification=None, question=None)


class KeywordPlanner:
    def predict(self, question: str) -> Prediction:
        return Prediction(_baseline_decision(question), {"backend": "keyword", "version": "1.0"})
