"""Feature engineering.

Two design decisions drive this module.

First, timing and regularity features come before stage features. Sleep
stages from a wrist wearable disagree with polysomnography often enough
(kappa roughly 0.2-0.5 in published validations) that building a model
primarily on deep-sleep percentage means modelling the device's guesswork.
Timing is measured, not inferred, and regularity of timing is the metric
with the strongest outcome evidence behind it.

Second, everything is computed causally. A feature for night N uses only
data available on the morning of night N. Rolling windows are trailing,
never centred. This is what makes the walk-forward evaluation honest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPOCH_MIN = 5
EPOCHS_PER_DAY = 24 * 60 // EPOCH_MIN

STAGE_COLS = ["deep_min", "light_min", "rem_min", "awake_min"]
PHYSIO_COLS = ["rhr", "hrv_rmssd", "spo2", "resp_rate"]
CONTEXT_COLS = ["steps", "active_kcal"]


# ---------------------------------------------------------------------------
# Circular time helpers
# ---------------------------------------------------------------------------
def to_night_hours(ts: pd.Series) -> pd.Series:
    """Decimal hours on a scale where the night is continuous.

    Midnight is the worst possible place to put a discontinuity when the
    thing you are measuring straddles it. An 11:50pm bedtime and a 12:10am
    bedtime are twenty minutes apart, not twenty-three hours and forty.
    Hours before noon are shifted past 24 so the arithmetic behaves.
    """
    dec = ts.dt.hour + ts.dt.minute / 60.0 + ts.dt.second / 3600.0
    return dec.where(dec >= 12.0, dec + 24.0)


def circ_mean_hours(values: np.ndarray, period: float = 24.0) -> float:
    """Mean of clock times, done on the circle rather than the number line."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    ang = 2 * np.pi * (v % period) / period
    m = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
    return float((m % (2 * np.pi)) * period / (2 * np.pi))


def circ_diff_hours(a: float, b: float, period: float = 24.0) -> float:
    """Signed shortest distance between two clock times."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return np.nan
    d = (a - b + period / 2) % period - period / 2
    return float(d)


# ---------------------------------------------------------------------------
# Epoch grid, for the regularity metrics
# ---------------------------------------------------------------------------
def build_epoch_matrix(df: pd.DataFrame) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Binary sleep/wake matrix of shape (n_days, epochs_per_day).

    SRI, IS and IV are all defined over an evenly sampled binary state
    series, not over nightly summaries, so we reconstruct one from the
    session intervals.
    """
    start = df["bedtime"].min().normalize()
    end = df["waketime"].max().normalize() + pd.Timedelta(days=1)
    idx = pd.date_range(start, end, freq=f"{EPOCH_MIN}min", inclusive="left")
    s = pd.Series(np.zeros(len(idx), dtype=np.int8), index=idx)

    for b, w in zip(df["bedtime"], df["waketime"]):
        s.loc[b:w] = 1

    n_days = len(s) // EPOCHS_PER_DAY
    mat = s.values[: n_days * EPOCHS_PER_DAY].reshape(n_days, EPOCHS_PER_DAY)
    days = pd.DatetimeIndex([start + pd.Timedelta(days=i) for i in range(n_days)])
    return mat, days


def sleep_regularity_index(mat: np.ndarray) -> float:
    """Phillips et al. SRI, scaled -100 (random) to +100 (perfectly regular).

    The probability that you are in the same state now as you were at this
    exact time yesterday, rescaled.
    """
    if mat.shape[0] < 2:
        return np.nan
    agree = (mat[:-1] == mat[1:]).mean()
    return float(200.0 * agree - 100.0)


def interdaily_stability(mat: np.ndarray, bins_per_day: int = 24) -> float:
    """IS: how reliably the rhythm repeats against a 24h template. 0 to 1."""
    if mat.shape[0] < 2:
        return np.nan
    per_bin = mat.shape[1] // bins_per_day
    hourly = mat[:, : per_bin * bins_per_day].reshape(mat.shape[0], bins_per_day, per_bin).mean(axis=2)
    x = hourly.ravel()
    n = x.size
    if n < 2 or np.allclose(x.var(), 0):
        return np.nan
    profile = hourly.mean(axis=0)
    num = n * ((profile - x.mean()) ** 2).sum()
    den = bins_per_day * ((x - x.mean()) ** 2).sum()
    return float(num / den) if den > 0 else np.nan


def intradaily_variability(mat: np.ndarray, bins_per_day: int = 24) -> float:
    """IV: fragmentation. Higher means the rhythm is chopped up."""
    if mat.shape[0] < 2:
        return np.nan
    per_bin = mat.shape[1] // bins_per_day
    hourly = mat[:, : per_bin * bins_per_day].reshape(mat.shape[0], bins_per_day, per_bin).mean(axis=2)
    x = hourly.ravel()
    n = x.size
    den = (n - 1) * ((x - x.mean()) ** 2).sum()
    if den <= 0:
        return np.nan
    num = n * (np.diff(x) ** 2).sum()
    return float(num / den)


def _rolling_regularity(df: pd.DataFrame, windows=(7, 14)) -> pd.DataFrame:
    """Trailing-window SRI / IS / IV aligned back onto nights."""
    mat, days = build_epoch_matrix(df)
    day_pos = {d: i for i, d in enumerate(days)}
    out = pd.DataFrame(index=df.index)

    positions = df["night"].map(day_pos)
    for w in windows:
        sri, ist, ivt = [], [], []
        for p in positions:
            if pd.isna(p) or p < w:
                sri.append(np.nan); ist.append(np.nan); ivt.append(np.nan)
                continue
            p = int(p)
            block = mat[p - w + 1 : p + 1]
            sri.append(sleep_regularity_index(block))
            ist.append(interdaily_stability(block))
            ivt.append(intradaily_variability(block))
        out[f"sri_{w}"] = sri
        out[f"is_{w}"] = ist
        out[f"iv_{w}"] = ivt
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def engineer(
    nights: pd.DataFrame,
    sleep_need_min: float = 465.0,
    chronotype_msf: float | None = None,
) -> pd.DataFrame:
    """Build the modelling frame from tidy nightly records.

    Args:
        sleep_need_min: personal nightly requirement, used as the sleep-debt
            reference point. Population default is a placeholder, not a
            prescription.
        chronotype_msf: personal mid-sleep on free days. Estimated from the
            data when not supplied.
    """
    df = nights.sort_values("night").reset_index(drop=True).copy()
    if len(df) < 8:
        raise ValueError(f"Need at least 8 nights to build features, got {len(df)}.")

    f = pd.DataFrame({"night": df["night"]})
    has_stages = all(c in df.columns and df[c].notna().any() for c in ("deep_min", "rem_min"))

    # --- timing ------------------------------------------------------------
    f["bed_dec"] = to_night_hours(df["bedtime"])
    f["wake_dec"] = to_night_hours(df["waketime"])
    f["mid_dec"] = f["bed_dec"] + (df["tib_min"] / 60.0) / 2.0
    for name, col in (("bed", "bed_dec"), ("mid", "mid_dec")):
        ang = 2 * np.pi * (f[col] % 24) / 24
        f[f"{name}_sin"] = np.sin(ang)
        f[f"{name}_cos"] = np.cos(ang)

    # --- duration and continuity -------------------------------------------
    f["tst_min"] = df["tst_min"]
    f["tib_min"] = df["tib_min"]
    f["efficiency"] = (df["tst_min"] / df["tib_min"]).clip(0, 1)

    # --- calendar ----------------------------------------------------------
    dow = df["night"].dt.dayofweek
    f["dow"] = dow
    f["is_weekend"] = (dow >= 5).astype(int)
    # Tomorrow being free is known tonight, so it is a legitimate predictor.
    f["next_day_free"] = (df["night"].dt.dayofweek.shift(-1).fillna(0) >= 5).astype(int)
    f["days_elapsed"] = (df["night"] - df["night"].min()).dt.days

    # --- stages, only when the exporter actually gave them ------------------
    if has_stages:
        tst = df["tst_min"].replace(0, np.nan)
        f["deep_pct"] = df["deep_min"] / tst
        f["rem_pct"] = df["rem_min"] / tst
        if "light_min" in df:
            f["light_pct"] = df["light_min"] / tst
        f["deep_min"] = df["deep_min"]
        f["rem_min"] = df["rem_min"]
        if "awake_min" in df:
            f["waso_min"] = df["awake_min"]

    # --- physiology and context --------------------------------------------
    for c in PHYSIO_COLS + CONTEXT_COLS:
        if c in df.columns and df[c].notna().any():
            f[c] = df[c]

    # --- trailing rolling statistics ---------------------------------------
    # Time-based windows so that missing nights do not silently stretch them.
    tidx = pd.DatetimeIndex(df["night"])
    def roll(series: pd.Series, days: int, how: str) -> pd.Series:
        s = pd.Series(series.values, index=tidx)
        r = s.rolling(f"{days}D", min_periods=max(3, days // 3))
        return getattr(r, how)().values

    for days in (7, 14):
        f[f"tst_roll{days}_mean"] = roll(f["tst_min"], days, "mean")
        f[f"tst_roll{days}_std"] = roll(f["tst_min"], days, "std")
    f["eff_roll7_mean"] = roll(f["efficiency"], 7, "mean")
    f["bed_roll7_std"] = roll(f["bed_dec"], 7, "std")
    f["wake_roll7_std"] = roll(f["wake_dec"], 7, "std")
    f["mid_roll7_std"] = roll(f["mid_dec"], 7, "std")
    if "steps" in f:
        f["steps_roll7_mean"] = roll(f["steps"], 7, "mean")
    for c in ("rhr", "hrv_rmssd"):
        if c in f:
            base = roll(f[c], 28, "mean")
            f[f"{c}_delta"] = f[c] - base

    # --- regularity indices -------------------------------------------------
    f = pd.concat([f, _rolling_regularity(df)], axis=1)

    # --- chronotype and social jetlag --------------------------------------
    free = f["is_weekend"] == 1
    if chronotype_msf is None:
        chronotype_msf = circ_mean_hours(f.loc[free, "mid_dec"].values) if free.any() \
            else circ_mean_hours(f["mid_dec"].values)
    f["chronotype_msf"] = chronotype_msf
    f["misalignment_h"] = f["mid_dec"].apply(lambda m: abs(circ_diff_hours(m % 24, chronotype_msf % 24)))

    msf_roll = pd.Series(np.where(free, f["mid_dec"], np.nan), index=tidx).rolling("21D", min_periods=2).mean()
    msw_roll = pd.Series(np.where(~free, f["mid_dec"], np.nan), index=tidx).rolling("21D", min_periods=3).mean()
    f["social_jetlag_h"] = (msf_roll.values - msw_roll.values)

    # --- sleep debt ---------------------------------------------------------
    deficit = sleep_need_min - f["tst_min"]
    f["debt_7d_min"] = pd.Series(deficit.values, index=tidx).rolling("7D", min_periods=3).sum().values
    f["debt_14d_min"] = pd.Series(deficit.values, index=tidx).rolling("14D", min_periods=5).sum().values
    baseline = f["tst_roll14_mean"]
    f["tst_z"] = (f["tst_min"] - baseline) / f["tst_roll14_std"].replace(0, np.nan)

    # --- lags ---------------------------------------------------------------
    for lag in (1, 2, 3, 7):
        f[f"tst_lag{lag}"] = f["tst_min"].shift(lag)
    f["eff_lag1"] = f["efficiency"].shift(1)
    f["mid_lag1"] = f["mid_dec"].shift(1)
    f["bed_lag1"] = f["bed_dec"].shift(1)
    if "steps" in f:
        f["steps_lag1"] = f["steps"].shift(1)
    if "rhr" in f:
        f["rhr_lag1"] = f["rhr"].shift(1)

    # Night-to-night swing in timing: the single-night version of regularity.
    f["mid_shift_h"] = (f["mid_dec"] - f["mid_lag1"]).abs()

    if "subjective" in df.columns and df["subjective"].notna().any():
        f["subjective"] = df["subjective"]
        f["subjective_lag1"] = f["subjective"].shift(1)

    f.attrs["has_stages"] = has_stages
    f.attrs["chronotype_msf"] = float(chronotype_msf)
    return f


def feature_columns(f: pd.DataFrame, target: str) -> list[str]:
    """Predictor columns: everything numeric that is not a target or a key."""
    drop = {"night", "chronotype_msf"}
    drop |= {c for c in f.columns if c.startswith("y_")}
    # The label for tonight cannot be used to predict tonight.
    drop |= {target}
    if target == "subjective":
        drop |= {"subjective"}
    return [
        c for c in f.columns
        if c not in drop and pd.api.types.is_numeric_dtype(f[c])
    ]
