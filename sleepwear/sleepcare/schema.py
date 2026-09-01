"""Canonical night-level schema and tolerant column mapping.

Every exporter names things differently. Health Data Export, Health Sync,
Fitbit's Kaggle dumps and hand-kept spreadsheets all disagree. Rather than
demand one format, we map whatever we're given onto a single schema and
record honestly which fields are missing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# --- canonical fields ------------------------------------------------------
# Required: without these the pipeline cannot run.
REQUIRED = ["night", "bedtime", "waketime"]

# Core: derivable or near-universal.
CORE = ["tst_min", "tib_min"]

# Stage fields. Frequently absent — OHealth's Health Connect write is often
# duration-only, so the app must degrade gracefully when these are missing.
STAGES = ["deep_min", "light_min", "rem_min", "awake_min"]

# Physiology, optional.
PHYSIO = ["rhr", "hrv_rmssd", "spo2", "resp_rate"]

# Daytime context, optional but valuable as next-night predictors.
CONTEXT = ["steps", "active_kcal"]

# Ground truth label, optional but the most trustworthy target we can get.
SUBJECTIVE = ["subjective"]

ALL_FIELDS = REQUIRED + CORE + STAGES + PHYSIO + CONTEXT + SUBJECTIVE

# --- alias table -----------------------------------------------------------
# Keys are canonical names; values are lowercase substrings we accept.
ALIASES: dict[str, list[str]] = {
    "night": ["night", "date", "day", "sleepday", "calendar_date", "start_date"],
    "bedtime": [
        "bedtime", "sleep_start", "start_time", "starttime", "sleeponset",
        "onset", "session_start", "asleep_time", "time_in_bed_start",
    ],
    "waketime": [
        "waketime", "wake_time", "sleep_end", "end_time", "endtime",
        "session_end", "awake_time", "time_in_bed_end", "offset",
    ],
    "tst_min": [
        "tst", "total_sleep", "totalminutesasleep", "minutes_asleep",
        "asleep_minutes", "sleep_duration", "duration_min", "sleep_minutes",
        "asleep", "hours_asleep", "sleep_hours", "sleep_time", "total_hours",
    ],
    "tib_min": [
        "tib", "time_in_bed", "totaltimeinbed", "in_bed_minutes", "bed_minutes",
    ],
    "deep_min": ["deep", "deep_sleep", "slow_wave", "n3"],
    "light_min": ["light", "light_sleep", "n1", "n2"],
    "rem_min": ["rem", "rem_sleep", "paradoxical"],
    "awake_min": ["awake", "wake_minutes", "waso", "restless", "awakenings_min"],
    "rhr": ["rhr", "resting_hr", "resting_heart", "restingheartrate", "hr_min"],
    "hrv_rmssd": ["hrv", "rmssd", "heart_rate_variability"],
    "spo2": ["spo2", "oxygen", "oxygen_saturation", "blood_oxygen"],
    "resp_rate": ["resp", "respiratory", "breathing_rate", "breaths"],
    "steps": ["steps", "step_count", "totalsteps"],
    "active_kcal": ["active_cal", "active_kcal", "calories", "activecalories"],
    "subjective": ["subjective", "rating", "self_report", "how_i_felt", "mood"],
}

# Units that show up in duration columns.
_DURATION_UNIT = re.compile(r"(hour|hr|_h$|sec|second|ms|milli)", re.I)


@dataclass
class MappingReport:
    """What we found, what we guessed, and what simply is not there."""

    mapped: dict[str, str]
    missing: list[str]
    unmapped_source_columns: list[str]

    @property
    def has_stages(self) -> bool:
        return any(f in self.mapped for f in STAGES)

    @property
    def has_physio(self) -> bool:
        return any(f in self.mapped for f in PHYSIO)

    def summary(self) -> str:
        lines = [f"Mapped {len(self.mapped)} of {len(ALL_FIELDS)} canonical fields."]
        if not self.has_stages:
            lines.append(
                "No sleep stages found. Stage-derived features are disabled; "
                "timing and regularity features still work."
            )
        if not self.has_physio:
            lines.append("No overnight physiology found (resting HR, HRV, SpO2).")
        return " ".join(lines)


def _norm(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_")


def guess_mapping(columns: list[str]) -> MappingReport:
    """Best-effort map of source columns onto the canonical schema.

    Longest alias wins, so 'total_sleep_minutes' beats a bare 'sleep' match.
    """
    normed = {c: _norm(c) for c in columns}
    mapped: dict[str, str] = {}
    used: set[str] = set()

    scored: list[tuple[int, str, str]] = []
    for canon, aliases in ALIASES.items():
        for src, n in normed.items():
            for alias in aliases:
                if alias in n:
                    scored.append((len(alias), canon, src))

    # Longest, most specific alias match first.
    for _, canon, src in sorted(scored, key=lambda t: -t[0]):
        if canon in mapped or src in used:
            continue
        mapped[canon] = src
        used.add(src)

    missing = [f for f in ALL_FIELDS if f not in mapped]
    unmapped = [c for c in columns if c not in used]
    return MappingReport(mapped=mapped, missing=missing, unmapped_source_columns=unmapped)


def looks_like_hours(colname: str) -> bool:
    """Heuristic for whether a duration column is in hours rather than minutes."""
    return bool(re.search(r"(hour|hr|_h$)", _norm(colname)))


def looks_like_seconds(colname: str) -> bool:
    return bool(re.search(r"(sec|second)", _norm(colname)))


def looks_like_millis(colname: str) -> bool:
    return bool(re.search(r"(ms$|milli)", _norm(colname)))
