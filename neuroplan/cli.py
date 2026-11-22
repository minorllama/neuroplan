import argparse
import json
import os
from pathlib import Path

from openai import APIError

from .artifacts import save_run
from .data import load, prepare, write_json
from .evaluation import evaluate
from .planners import (
    DEFAULT_OLLAMA_MODEL, DEFAULT_VLLM_MODEL, KeywordPlanner, OllamaPlanner, OpenAIPlanner, VLLMPlanner,
)
from .schema import AnalysisPlan

DEFAULT_DATA = Path("data/allen_units.csv")


def _backend_options(parser):
    parser.add_argument("--backend", choices=["openai", "ollama", "vllm", "keyword"], default="openai")
    parser.add_argument(
        "--model", help=f"Model ID (Ollama defaults to {DEFAULT_OLLAMA_MODEL}, vLLM to {DEFAULT_VLLM_MODEL})"
    )


def _parser():
    parser = argparse.ArgumentParser(description="Replayable analyses of mouse visual cortex")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="fetch, verify and join the Allen tables")
    prep.add_argument("--data", type=Path, default=DEFAULT_DATA)
    prep.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    for name, help_text in [
        ("ask", "turn a natural-language question into an analysis"),
        ("run-plan", "execute a hand-written plan"),
        ("replay", "rerun an archived plan against the original dataset"),
        ("benchmark", "score plans, answers and clarification decisions"),
    ]:
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--data", type=Path, default=DEFAULT_DATA)
        command.add_argument("--out", type=Path, required=True, help="new output directory")
        if name in {"ask", "benchmark"}:
            _backend_options(command)
        if name == "ask":
            command.add_argument("question")
        elif name in {"run-plan", "replay"}:
            command.add_argument("input", type=Path)
        else:
            command.add_argument("--questions", type=Path, default=Path("examples/benchmark.jsonl"))
    return parser


def _planner(args):
    if args.backend == "keyword":
        return KeywordPlanner()
    if args.backend == "ollama":
        return OllamaPlanner(args.model or DEFAULT_OLLAMA_MODEL)
    if args.backend == "vllm":
        return VLLMPlanner(args.model or DEFAULT_VLLM_MODEL)
    model = args.model or os.environ.get("OPENAI_MODEL")
    if not model:
        raise ValueError("Choose an OpenAI model with --model or OPENAI_MODEL.")
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY to use the OpenAI backend.")
    return OpenAIPlanner(model)


def _run_analysis(args, frame, manifest):
    if args.command == "ask":
        prediction = _planner(args).predict(args.question)
        decision = prediction.decision
        provenance = {**prediction.provenance, "question": args.question}
        if decision.plan is None:
            write_json(args.out / "clarification.json",
                       {"decision": decision.model_dump(mode="json"), "provenance": provenance})
            print(decision.question)
            return
        plan = decision.plan
    elif args.command == "replay":
        archived = json.loads(args.input.read_text())
        if archived["dataset"]["sha256"] != manifest["sha256"]:
            raise ValueError("The current dataset differs from the archived run.")
        plan = AnalysisPlan.model_validate_json(json.dumps(archived["plan"]))
        provenance = {**archived["provenance"], "replayed": True}
    else:
        plan = AnalysisPlan.model_validate_json(args.input.read_text())
        provenance = {"backend": "manual"}
    result = save_run(args.out, frame, manifest, plan, provenance)
    summary = {key: value for key, value in result.items() if key != "rows"}
    print(json.dumps(summary or result, indent=2))


def main():
    parser = _parser()
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            manifest = prepare(args.data, args.raw_dir)
            print(f"Prepared {manifest['rows']:,} units from {len(manifest['session_ids'])} sessions.")
            print(f"Dataset SHA-256: {manifest['sha256']}")
            return 0
        if args.out.exists():
            raise ValueError(f"Output directory already exists: {args.out}")
        frame, manifest = load(args.data)
        if args.command == "benchmark":
            report = evaluate(args.questions, _planner(args), args.backend, frame, manifest, args.out)
            print(json.dumps(report["scores"], indent=2))
            return int(report["scores"]["errors"] > 0)
        _run_analysis(args, frame, manifest)
        return 0
    except APIError as error:
        status = getattr(error, "status_code", None)
        detail = f" (HTTP {status})" if status is not None else ""
        backend = "vLLM" if getattr(args, "backend", None) == "vllm" else "OpenAI"
        parser.exit(2, f"neuroplan: {backend} request failed: {type(error).__name__}{detail}\n")
    except (OSError, KeyError, ValueError) as error:
        parser.exit(2, f"neuroplan: {error}\n")
