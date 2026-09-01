"""Ingestion: turn whatever the exporter gave us into tidy nightly records.

Also provides a synthetic generator so the pipeline is testable on day one,
before a new watch has accumulated any history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import (
    MappingReport,
    guess_mapping,
    looks_like_hours,
    looks_like_millis,
    looks_like_seconds,
)

# Minutes columns that must be non-negative and bounded.
_MINUTE_CAPS = {
    "tst_min": (0, 1000),
    "tib_min": (0, 1100),
    "deep_min": (0, 400),
    "light_min": (0, 800),
    "rem_min": (0, 400),
    "awake_min": (0, 400),
}


def _to_minutes(series: pd.Series, source_name: str) -> pd.Series:
    """Coerce a duration column to minutes, guessing units from name and scale."""
    s = pd.to_numeric(series, errors="coerce")
    if looks_like_millis(source_name):
        return s / 60000.0
    if looks_like_seconds(source_name):
        return s / 60.0
    if looks_like_hours(source_name):
        return s * 60.0
    # Fall back on magnitude: a median under 24 is almost certainly hours.
    med = s.dropna().median()
    if pd.notna(med) and 0 < med <= 24:
        return s * 60.0
    return s


def load_nightly(
    df: pd.DataFrame, mapping: MappingReport | None = None
) -> tuple[pd.DataFrame, MappingReport]:
    """Normalise an arbitrary export into the canonical nightly frame."""
    mapping = mapping or guess_mapping(list(df.columns))
    out = pd.DataFrame(index=df.index)

    for canon, src in mapping.mapped.items():
        col = df[src]
        if canon in ("bedtime", "waketime"):
            out[canon] = pd.to_datetime(col, errors="coerce")
        elif canon == "night":
            out[canon] = pd.to_datetime(col, errors="coerce").dt.normalize()
        elif canon in _MINUTE_CAPS:
            out[canon] = _to_minutes(col, src)
        else:
            out[canon] = pd.to_numeric(col, errors="coerce")

    if "bedtime" not in out or "waketime" not in out:
        raise ValueError(
            "Need a sleep start and a sleep end column. "
            f"Found only: {sorted(mapping.mapped)}"
        )

    # A session that ends before it starts crossed midnight without a date.
    bad = out["waketime"] <= out["bedtime"]
    out.loc[bad, "waketime"] = out.loc[bad, "waketime"] + pd.Timedelta(days=1)

    # 'Night of' is the evening the sleep belongs to: sessions starting after
    # noon belong to that calendar day, sessions starting before noon (a 2am
    # bedtime) belong to the previous day.
    if "night" not in out or out["night"].isna().all():
        anchor = out["bedtime"]
        out["night"] = np.where(
            anchor.dt.hour >= 12,
            anchor.dt.normalize(),
            (anchor - pd.Timedelta(days=1)).dt.normalize(),
        )
        out["night"] = pd.to_datetime(out["night"])

    span_min = (out["waketime"] - out["bedtime"]).dt.total_seconds() / 60.0
    if "tib_min" not in out or out["tib_min"].isna().all():
        out["tib_min"] = span_min

    tst_inferred = False
    if "tst_min" not in out or out["tst_min"].isna().all():
        stage_cols = [c for c in ("deep_min", "light_min", "rem_min") if c in out]
        if stage_cols:
            out["tst_min"] = out[stage_cols].sum(axis=1, min_count=1)
        else:
            # No stages, no reported asleep time: the whole session is treated
            # as sleep. Efficiency then equals 1.0 by construction, which is an
            # artefact of the export, not a perfect night. Flagged so the score
            # can drop the efficiency component rather than award it full marks.
            out["tst_min"] = span_min
            tst_inferred = True

    # A reported duration can disagree with the timestamps it came with. When
    # it does, trust the clock: an exporter that writes a stale or unit-confused
    # duration field will happily claim a 30-hour night that the start and end
    # times plainly contradict.
    out["_span_min"] = span_min
    disagree = (out["tib_min"] - span_min).abs() > 90
    if disagree.any():
        out.loc[disagree, "tib_min"] = span_min[disagree]
        out.loc[disagree & (out["tst_min"] > span_min), "tst_min"] = span_min[disagree]

    for col, (lo, hi) in _MINUTE_CAPS.items():
        if col in out:
            out[col] = out[col].clip(lo, hi)

    # Sleep can never exceed the session that contains it.
    out["tst_min"] = np.minimum(out["tst_min"], out["tib_min"])

    out = out.dropna(subset=["night", "bedtime", "waketime"])
    out = out.sort_values("night")

    # One row per night: keep the longest session if a night has fragments.
    out = (
        out.sort_values(["night", "tst_min"])
        .groupby("night", as_index=False)
        .last()
        .sort_values("night")
        .reset_index(drop=True)
    )

    # Drop physiologically impossible nights rather than let them poison stats.
    plausible = (
        (out["_span_min"] > 60) & (out["_span_min"] < 1100)
        & (out["tib_min"] > 60) & (out["tib_min"] < 1100)
    )
    out = out[plausible].drop(columns=["_span_min"]).reset_index(drop=True)
    out.attrs["tst_inferred"] = tst_inferred
    return out, mapping


def load_csv(path_or_buffer) -> tuple[pd.DataFrame, MappingReport]:
    raw = pd.read_csv(path_or_buffer)
    return load_nightly(raw)


# --------------------------------------------------------------------------
# Synthetic generator
# --------------------------------------------------------------------------
def generate_synthetic(
    n_nights: int = 420,
    seed: int = 7,
    chronotype_msf: float = 4.1,
    start: str = "2025-06-01",
    include_stages: bool = True,
) -> pd.DataFrame:
    """Generate plausible nightly sleep records with real learnable structure.

    The point is not realism for its own sake. The generator encodes the
    relationships the literature actually supports, so that a model trained
    here has something true to find:

      * circadian misalignment (midpoint drifting from personal chronotype)
        degrades efficiency
      * accumulated sleep debt produces rebound sleep the following night
      * daytime activity modestly improves deep sleep
      * weekends shift the midpoint later, creating social jetlag
      * occasional disrupted nights (travel, illness, a late night out)

    Args:
        chronotype_msf: personal mid-sleep on free days, in decimal hours.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=n_nights, freq="D")

    rows = []
    debt = 0.0  # cumulative sleep debt in minutes
    prev_steps = 8000.0

    # Slow seasonal drift in bedtime, a few weeks long.
    drift = np.cumsum(rng.normal(0, 0.035, n_nights))
    drift = drift - drift.mean()

    for i, d in enumerate(dates):
        weekend = d.dayofweek >= 5
        free_day = weekend

        steps = float(np.clip(rng.normal(9000 if not weekend else 11000, 3200), 800, 26000))
        active_kcal = steps * rng.normal(0.042, 0.005)

        # --- bedtime -------------------------------------------------------
        base_bed = 23.55 + (0.95 if weekend else 0.0)
        late_night = rng.random() < (0.10 if weekend else 0.035)
        bed_dec = base_bed + drift[i] + rng.normal(0, 0.42) + (1.9 if late_night else 0.0)

        # Debt pulls bedtime a little earlier.
        bed_dec -= np.clip(debt / 600.0, 0, 0.8)

        # --- wake ----------------------------------------------------------
        if free_day:
            wake_dec = bed_dec + rng.normal(8.3, 0.85)
        else:
            alarm = 7.0 + rng.normal(0, 0.16)
            wake_dec = min(alarm, bed_dec + rng.normal(9.0, 0.7))
            wake_dec = max(wake_dec, bed_dec + 3.6)

        tib = (wake_dec - bed_dec) * 60.0
        tib = float(np.clip(tib, 220, 700))

        # --- efficiency ----------------------------------------------------
        midpoint = bed_dec + (tib / 60.0) / 2.0
        misalign = abs(((midpoint - chronotype_msf + 12) % 24) - 12)

        eff = 0.925
        eff -= 0.030 * misalign                      # circadian misalignment
        eff += 0.012 * ((prev_steps - 9000) / 6000)  # activity helps
        eff -= 0.020 if late_night else 0.0
        eff += 0.010 * np.clip(debt / 400.0, 0, 1)   # pressure consolidates sleep
        eff += rng.normal(0, 0.028)

        sick = rng.random() < 0.022
        if sick:
            eff -= rng.uniform(0.06, 0.15)

        eff = float(np.clip(eff, 0.58, 0.985))
        tst = tib * eff

        # --- debt dynamics --------------------------------------------------
        debt = float(np.clip(debt + (465.0 - tst), -260, 900))

        # --- physiology -----------------------------------------------------
        rhr = 56 + 5.5 * (1 - eff) * 10 + (3.4 if sick else 0) + rng.normal(0, 1.9)
        rhr -= 0.00016 * (prev_steps - 9000)
        hrv = 52 - 0.62 * (rhr - 56) - (7.5 if sick else 0) + rng.normal(0, 5.2)
        spo2 = 96.6 - (0.9 if sick else 0) + rng.normal(0, 0.55)
        resp = 14.6 + (0.9 if sick else 0) + rng.normal(0, 0.75)

        row = {
            "night": d,
            "bedtime": d + pd.Timedelta(minutes=int(round(bed_dec * 60))),
            "waketime": d + pd.Timedelta(minutes=int(round(wake_dec * 60))),
            "tib_min": round(tib, 1),
            "tst_min": round(tst, 1),
            "rhr": round(float(np.clip(rhr, 40, 95)), 1),
            "hrv_rmssd": round(float(np.clip(hrv, 8, 130)), 1),
            "spo2": round(float(np.clip(spo2, 88, 100)), 1),
            "resp_rate": round(float(np.clip(resp, 9, 24)), 1),
            "steps": int(steps),
            "active_kcal": int(active_kcal),
        }

        if include_stages:
            awake = tib - tst
            deep_frac = np.clip(
                0.175 + 0.03 * ((prev_steps - 9000) / 8000) - 0.012 * misalign
                + rng.normal(0, 0.022),
                0.06, 0.30,
            )
            rem_frac = np.clip(0.215 + 0.02 * np.clip(tst / 480 - 1, -0.5, 0.5)
                               + rng.normal(0, 0.028), 0.08, 0.34)
            deep = tst * deep_frac
            rem = tst * rem_frac
            row.update(
                deep_min=round(deep, 1),
                light_min=round(tst - deep - rem, 1),
                rem_min=round(rem, 1),
                awake_min=round(max(awake, 0.0), 1),
            )

        # Subjective rating: noisy read on efficiency, duration and alignment.
        latent = (
            2.4 * (eff - 0.85) * 10
            + 1.5 * np.clip((tst - 400) / 120, -1.4, 1.4)
            - 0.9 * misalign
            + rng.normal(0, 0.55)
        )
        row["subjective"] = int(np.clip(round(3 + latent / 2.6), 1, 5))

        rows.append(row)
        prev_steps = steps

    df = pd.DataFrame(rows)

    # Real exports have gaps: the watch dies, you forget to wear it.
    keep = rng.random(len(df)) > 0.045
    return df[keep].reset_index(drop=True)
