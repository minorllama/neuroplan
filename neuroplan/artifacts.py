"""Render figures and save the inputs needed to replay an analysis."""
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from platform import python_version

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from . import __version__
from .analysis import execute
from .data import write_bytes, write_json
from .schema import METRIC_LABELS, Analysis, AnalysisPlan


def software() -> dict:
    return {"neuroplan": __version__, "python": python_version(), **{
        name: version(name) for name in ("numpy", "pandas", "matplotlib", "pydantic", "openai")
    }}


def plot_result(plan: AnalysisPlan, result: dict) -> Figure:
    figure = Figure(figsize=(7.3, 4.5), layout="constrained")
    ax = figure.subplots()
    rows = result["rows"]
    label = METRIC_LABELS[plan.metric]
    if plan.analysis == Analysis.PAIRED:
        for row in rows:
            ax.plot([0, 1], [row["first_value"], row["second_value"]], color="#32677c",
                    marker="o", markersize=3, linewidth=0.9, alpha=0.42)
        ax.set_xticks([0, 1], [region.value for region in plan.regions])
        ax.set_xlim(-0.45, 1.45)
        ax.set_title(f"Paired sessions (n = {len(rows)})")
        ax.set_ylabel(f"Within-session {plan.statistic.value} · {label}")
        has_data = bool(rows)
    else:
        available = [row for row in rows if row["value"] is not None]
        if available:
            ax.bar([row["region"] for row in available], [row["value"] for row in available], color="#32677c")
        ax.set_ylabel("Fraction of units" if plan.analysis == Analysis.FRACTION else label)
        title = f"{label} > {plan.threshold:g}" if plan.analysis == Analysis.FRACTION else f"Pooled {plan.statistic.value} by region"
        ax.set_title(title)
        has_data = bool(available)
    if not has_data:
        ax.text(0.5, 0.5, "No eligible observations", transform=ax.transAxes, ha="center")
    ax.spines[["top", "right"]].set_visible(False)
    return figure


def save_run(out: Path, frame, manifest: dict, plan: AnalysisPlan, provenance: dict) -> dict:
    result = execute(frame, plan)
    figure = plot_result(plan, result)
    buffer = BytesIO()
    FigureCanvasAgg(figure).print_png(buffer)
    write_bytes(out / "figure.png", buffer.getvalue())
    write_json(out / "result.json", result)
    write_json(out / "run.json", {"plan": plan.model_dump(mode="json"), "dataset": manifest,
                                 "provenance": provenance, "software": software()})
    return result
