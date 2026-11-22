"""Score plan intent and numerical results against hand-written reference plans."""
import json
import math
from collections import defaultdict
from numbers import Real
from pathlib import Path
from statistics import mean, median

from .analysis import execute
from .artifacts import software
from .data import sha256, write_bytes, write_json
from .schema import Analysis, AnalysisPlan, Clarification, Quality, Statistic


def reference_result(records: list[dict], plan: AnalysisPlan) -> dict:
    """Independent Python implementation for generating the benchmark answers."""
    by_region = {region.value: defaultdict(list) for region in plan.regions}
    aggregate = mean if plan.statistic == Statistic.MEAN else median
    for row in records:
        region, value = row["region"], row[plan.metric.value]
        if region not in by_region or not math.isfinite(value):
            continue
        if plan.quality == Quality.SCREENED and not (
            row["quality"] == "good" and row["amplitude_cutoff"] <= 0.1
            and row["presence_ratio"] >= 0.95 and row["isi_violations"] <= 0.5
        ):
            continue
        by_region[region][row["session_id"]].append(value)

    if plan.analysis == Analysis.PAIRED:
        left, right = by_region.values()
        rows = []
        for session in sorted(left.keys() & right.keys()):
            a, b = left[session], right[session]
            if min(len(a), len(b)) >= plan.min_units_per_region:
                x, y = aggregate(a), aggregate(b)
                rows.append(dict(session_id=session, first_value=x, second_value=y,
                                 first_n_units=len(a), second_n_units=len(b), difference=x - y))
        return {"rows": rows, "n_sessions": len(rows),
                "n_units_by_region": {plan.regions[0].value: sum(r["first_n_units"] for r in rows),
                                      plan.regions[1].value: sum(r["second_n_units"] for r in rows)},
                "mean_paired_difference": mean(r["difference"] for r in rows) if rows else None}

    rows = []
    for region, sessions in by_region.items():
        values = [value for group in sessions.values() for value in group]
        row = dict(region=region, n_units=len(values), session_ids=sorted(sessions))
        if plan.analysis == Analysis.FRACTION:
            row["n_above"] = sum(value > plan.threshold for value in values)
            row["value"] = row["n_above"] / len(values) if values else None
        else:
            row["value"] = aggregate(values) if values else None
        rows.append(row)
    return {"rows": rows}


def _same_answer(actual, expected) -> bool:
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and actual.keys() == expected.keys()
                and all(_same_answer(actual[k], v) for k, v in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(_same_answer(a, e) for a, e in zip(actual, expected)))
    if isinstance(expected, float):
        return isinstance(actual, Real) and math.isclose(actual, expected, abs_tol=1e-10, rel_tol=1e-9)
    return actual == expected


def _score_case(case, planner, frame, records):
    expected_json = case.get("plan")
    expected = (AnalysisPlan.model_validate_json(json.dumps(expected_json)) if expected_json else None)
    reason = Clarification(case["clarification"]) if expected is None else None
    reference = reference_result(records, expected) if expected else None
    row = {"id": case["id"], "category": case["category"], "question": case["question"],
           "expected_plan": expected_json, "expected_clarification": reason,
           "reference_result": reference, "prediction": None, "provenance": None,
           "actual_result": None, "error": None, "clarification_decision": False,
           "exact_plan": False if expected else None, "numerical_result": False if expected else None,
           "clarification_reason": False if reason else None}
    try:
        prediction = planner.predict(case["question"])
        decision = prediction.decision
        row["prediction"] = decision.model_dump(mode="json")
        row["provenance"] = prediction.provenance
        row["clarification_decision"] = (decision.plan is None) == (expected is None)
        if expected:
            row["exact_plan"] = decision.plan == expected
            if decision.plan:
                row["actual_result"] = execute(frame, decision.plan)
                row["numerical_result"] = _same_answer(row["actual_result"], reference)
        else:
            row["clarification_reason"] = decision.clarification == reason
    except Exception as error:
        # Keep a complete benchmark even when individual API requests fail.
        row["error"] = type(error).__name__
    return row


def _tallies(rows: list[dict]) -> dict:
    metrics = ("exact_plan", "numerical_result", "clarification_decision", "clarification_reason")
    scores = {name: {"correct": sum(row[name] is True for row in rows),
                     "total": sum(row[name] is not None for row in rows)} for name in metrics}
    scores["errors"] = sum(row["error"] is not None for row in rows)
    return scores


def _markdown(report: dict) -> str:
    lines = ["# NeuroPlan benchmark", "", f"Backend: `{report['backend']}`  ",
             f"Dataset SHA-256: `{report['dataset']['sha256']}`  ",
             f"Question set SHA-256: `{report['benchmark_sha256']}`", "",
             "| Metric | Correct | Total |", "|---|---:|---:|"]
    for name, score in report["scores"].items():
        if isinstance(score, dict):
            lines.append(f"| {name.replace('_', ' ')} | {score['correct']} | {score['total']} |")
    lines += ["", f"Execution errors: {report['scores']['errors']}", "",
              "## Cases that missed an expectation", ""]
    for row in report["cases"]:
        if not any(value is False for value in (
            row["exact_plan"], row["numerical_result"], row["clarification_decision"],
            row["clarification_reason"],
        )) and not row["error"]:
            continue
        actual = row["prediction"]
        expected = row["expected_plan"] or row["expected_clarification"]
        received = (actual["plan"] or actual["clarification"]) if actual else row["error"]
        lines.append(f"- **{row['id']}** {row['question']}")
        lines.append(f"  Expected `{json.dumps(expected)}`, received `{json.dumps(received)}`.")
    return "\n".join(lines) + "\n"


def evaluate(path: Path, planner, backend: str, frame, manifest: dict, out: Path) -> dict:
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Benchmark must contain cases with unique IDs")
    records = frame.to_dict("records")
    rows = [_score_case(case, planner, frame, records) for case in cases]
    by_category = {category: _tallies([row for row in rows if row["category"] == category])
                   for category in sorted({row["category"] for row in rows})}
    report = {"backend": backend, "dataset": manifest, "software": software(),
              "benchmark_sha256": sha256(path), "scores": _tallies(rows),
              "by_category": by_category, "cases": rows}
    write_json(out / "benchmark.json", report)
    write_bytes(out / "benchmark.md", _markdown(report).encode())
    return report
