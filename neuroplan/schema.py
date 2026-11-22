from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class Region(StrEnum):
    VISp = "VISp"
    VISl = "VISl"
    VISal = "VISal"
    VISam = "VISam"
    VISpm = "VISpm"
    VISrl = "VISrl"


class Metric(StrEnum):
    ORIENTATION = "g_osi_dg"
    DIRECTION = "g_dsi_dg"
    FIRING = "firing_rate_dg"


METRIC_LABELS = {
    Metric.ORIENTATION: "Global orientation selectivity (gOSI)",
    Metric.DIRECTION: "Global direction selectivity (gDSI)",
    Metric.FIRING: "Drifting-grating firing rate (Hz)",
}


class Analysis(StrEnum):
    SUMMARY = "summary"
    PAIRED = "paired"
    FRACTION = "fraction_above"


class Statistic(StrEnum):
    MEAN = "mean"
    MEDIAN = "median"


class Quality(StrEnum):
    SCREENED = "screened"
    ALL = "all"


class Clarification(StrEnum):
    REGION = "missing_region"
    METRIC = "ambiguous_metric"
    THRESHOLD = "missing_threshold"
    UNSUPPORTED = "unsupported_request"


class AnalysisPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    analysis: Analysis
    stimulus: Literal["drifting_gratings"]
    regions: list[Region]
    metric: Metric
    statistic: Statistic | None
    threshold: float | None
    min_units_per_region: int | None
    quality: Quality

    @model_validator(mode="after")
    def check_arguments(self):
        if not self.regions or len(set(self.regions)) != len(self.regions):
            raise ValueError("Provide at least one region, with no duplicates")
        if self.analysis == Analysis.FRACTION:
            if self.statistic is not None or self.threshold is None:
                raise ValueError("Fraction requires a threshold and a null statistic")
            if self.threshold < 0 or not self.threshold < float("inf"):
                raise ValueError("Threshold must be finite and nonnegative")
            if self.metric != Metric.FIRING and self.threshold > 1:
                raise ValueError("Selectivity thresholds must be in [0, 1]")
        elif self.statistic is None or self.threshold is not None:
            raise ValueError("Summary and paired analysis require a statistic and a null threshold")
        if self.analysis == Analysis.PAIRED:
            if len(self.regions) != 2 or not 1 <= (self.min_units_per_region or 0):
                raise ValueError("Paired analysis needs two regions and a positive minimum")
        elif self.min_units_per_region is not None:
            raise ValueError("Minimum unit count applies only to paired analysis")
        return self


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    plan: AnalysisPlan | None
    clarification: Clarification | None
    question: str | None

    @classmethod
    def constrained_json_schema(cls) -> dict:
        """Expose the plan-or-clarification rule to constrained decoders."""
        schema = cls.model_json_schema()
        alternatives = [
            {"plan": {"$ref": "#/$defs/AnalysisPlan"},
             "clarification": {"type": "null"}, "question": {"type": "null"}},
            {"plan": {"type": "null"},
             "clarification": {"$ref": "#/$defs/Clarification"}, "question": {"type": "string"}},
        ]
        return {
            "$defs": schema["$defs"],
            "anyOf": [
                {"type": "object", "properties": fields, "required": schema["required"],
                 "additionalProperties": False}
                for fields in alternatives
            ],
        }

    @model_validator(mode="after")
    def check_choice(self):
        if self.plan is not None:
            if self.clarification is not None or self.question is not None:
                raise ValueError("An executable decision has only a plan")
        elif self.clarification is None or not (self.question or "").strip():
            raise ValueError("A clarification requires a reason and a message")
        return self
