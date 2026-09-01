"""One call that runs ingest to targets, keeping metadata intact.

pandas drops `.attrs` through most operations, so chronotype and stage
availability were getting lost between stages. Everything that needs to
survive the pipeline is carried explicitly here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .features import engineer, feature_columns
from .ingest import load_nightly
from .schema import MappingReport
from .scoring import attach_targets, compute_score


@dataclass
class Dataset:
    nights: pd.DataFrame
    frame: pd.DataFrame
    mapping: MappingReport
    has_stages: bool
    chronotype_msf: float
    sleep_need_min: float
    notes: list[str] = field(default_factory=list)
    tst_inferred: bool = False

    def predictors(self, target: str) -> list[str]:
        return feature_columns(self.frame, target)

    @property
    def n_nights(self) -> int:
        return len(self.frame)

    @property
    def date_range(self) -> str:
        n = self.frame["night"]
        return f"{n.min():%d %b %Y} to {n.max():%d %b %Y}"

    def readiness(self) -> tuple[str, str]:
        """How much of the pipeline this dataset can honestly support."""
        n = self.n_nights
        if n < 14:
            return "Collecting", (
                f"{n} nights. Enough to chart, not enough to model. "
                "Descriptive views only.")
        if n < 45:
            return "Descriptive", (
                f"{n} nights. Trends and regularity metrics are meaningful. "
                "Prediction would overfit; baselines only.")
        if n < 120:
            return "Provisional", (
                f"{n} nights. A model can be validated, but treat it as provisional "
                "and check it against the naive baselines.")
        return "Established", (
            f"{n} nights. Enough history for walk-forward validation and "
            "seasonal forecasting.")


def build(
    raw: pd.DataFrame,
    sleep_need_min: float = 465.0,
    chronotype_msf: float | None = None,
    weights: dict[str, float] | None = None,
    already_tidy: bool = False,
) -> Dataset:
    """Run the full pipeline and hand back everything downstream needs."""
    notes: list[str] = []

    if already_tidy:
        nights, mapping = raw, MappingReport(
            mapped={c: c for c in raw.columns}, missing=[], unmapped_source_columns=[]
        )
    else:
        nights, mapping = load_nightly(raw)
        notes.append(mapping.summary())

    dropped = len(raw) - len(nights)
    if dropped > 0:
        notes.append(
            f"Dropped {dropped} record(s): duplicate nights, or sessions outside a "
            "plausible 1-18h range.")

    tst_inferred = bool(nights.attrs.get("tst_inferred", False))
    if tst_inferred:
        notes.append(
            "This export reports no separate asleep time, so the whole session is "
            "counted as sleep. Efficiency would be 100% on every night by "
            "construction, so it has been removed from the score.")
        weights = dict(weights or {})
        weights.pop("efficiency", None)

    f = engineer(nights, sleep_need_min=sleep_need_min, chronotype_msf=chronotype_msf)
    has_stages = bool(f.attrs.get("has_stages", False))
    msf = float(f.attrs.get("chronotype_msf", 4.0))

    if not has_stages:
        notes.append(
            "No stage data, so deep/REM features are absent and the score's "
            "restfulness component is redistributed across the others.")

    comps = compute_score(
        f, sleep_need_min=sleep_need_min, weights=weights,
        drop_efficiency=tst_inferred)
    f = pd.concat([f, comps.drop(columns=["sleep_score"])], axis=1)
    f = attach_targets(f, comps["sleep_score"])

    f.attrs["has_stages"] = has_stages
    f.attrs["chronotype_msf"] = msf

    return Dataset(
        nights=nights, frame=f, mapping=mapping, has_stages=has_stages,
        chronotype_msf=msf, sleep_need_min=sleep_need_min, notes=notes,
        tst_inferred=tst_inferred,
    )
