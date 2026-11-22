"""Deterministic execution of validated, allowlisted plans."""
import numpy as np
import pandas as pd

from .schema import Analysis, AnalysisPlan, Quality


def eligible_units(frame: pd.DataFrame, plan: AnalysisPlan) -> pd.DataFrame:
    """Apply the region, metric-availability and unit-quality policies."""
    metric = plan.metric.value
    keep = frame["region"].isin(plan.regions) & np.isfinite(frame[metric])
    if plan.quality == Quality.SCREENED:
        keep &= (
            frame["quality"].eq("good")
            & frame["amplitude_cutoff"].le(0.1)
            & frame["presence_ratio"].ge(0.95)
            & frame["isi_violations"].le(0.5)
        )
    return frame.loc[keep]


def _region_values(units: pd.DataFrame, plan: AnalysisPlan):
    for region in plan.regions:
        selected = units.loc[units["region"].eq(region.value)]
        values = selected[plan.metric.value]
        metadata = {
            "region": region.value,
            "n_units": len(values),
            "session_ids": sorted(int(value) for value in selected["session_id"].unique()),
        }
        yield metadata, values


def summarize(units: pd.DataFrame, plan: AnalysisPlan) -> dict:
    rows = []
    for row, values in _region_values(units, plan):
        row["value"] = float(values.agg(plan.statistic.value)) if len(values) else None
        rows.append(row)
    return {"rows": rows}


def fraction_above(units: pd.DataFrame, plan: AnalysisPlan) -> dict:
    rows = []
    for row, values in _region_values(units, plan):
        count = int(values.gt(plan.threshold).sum())
        row.update(n_above=count, value=count / len(values) if len(values) else None)
        rows.append(row)
    return {"rows": rows}


def paired_compare(units: pd.DataFrame, plan: AnalysisPlan) -> dict:
    first, second = (region.value for region in plan.regions)
    grouped = units.groupby(["session_id", "region"])[plan.metric.value]
    values = grouped.agg(value=plan.statistic.value, n_units="size")
    rows = []
    for session_id, group in values.groupby(level="session_id", sort=True):
        session = group.droplevel("session_id").reindex([first, second])
        if not session["n_units"].ge(plan.min_units_per_region).all():
            continue
        a, b = map(float, session["value"])
        n_a, n_b = map(int, session["n_units"])
        rows.append({"session_id": int(session_id), "first_value": a, "second_value": b,
                     "first_n_units": n_a, "second_n_units": n_b, "difference": a - b})
    return {
        "rows": rows,
        "n_sessions": len(rows),
        "n_units_by_region": {first: sum(row["first_n_units"] for row in rows),
                              second: sum(row["second_n_units"] for row in rows)},
        "mean_paired_difference": float(np.mean([row["difference"] for row in rows])) if rows else None,
    }


OPERATIONS = {Analysis.SUMMARY: summarize, Analysis.PAIRED: paired_compare,
              Analysis.FRACTION: fraction_above}


def execute(frame: pd.DataFrame, plan: AnalysisPlan) -> dict:
    return OPERATIONS[plan.analysis](eligible_units(frame, plan), plan)
