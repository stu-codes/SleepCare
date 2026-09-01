"""Regression tests.

Weighted toward the failures that are quiet rather than loud: a bedtime
averaged across midnight, a rolling window that peeks forward, hours read
as minutes. None of these throw. They just produce a confident wrong
answer, which is the only kind of bug that really matters in a tool
meant to give people advice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sleepcare.features import (
    circ_diff_hours, circ_mean_hours, engineer, feature_columns, to_night_hours,
)
from sleepcare.forecast import _to_night_hours, forecast_metric, recommend_bedtime
from sleepcare.ingest import generate_synthetic, load_nightly
from sleepcare.models import walk_forward
from sleepcare.pipeline import build
from sleepcare.unsupervised import find_anomalies, find_archetypes


@pytest.fixture(scope="module")
def ds():
    return build(generate_synthetic(300, seed=11))


# --- circular time ---------------------------------------------------------
def test_night_hours_keeps_the_night_contiguous():
    t = pd.Series(pd.to_datetime(["2025-01-01 23:50", "2025-01-02 00:10"]))
    nh = to_night_hours(t)
    assert abs(nh.iloc[1] - nh.iloc[0]) == pytest.approx(1 / 3, abs=0.01)


def test_circular_mean_straddling_midnight():
    # Naive averaging gives 12:00. The right answer is midnight.
    assert circ_mean_hours(np.array([23.5, 0.5])) % 24 == pytest.approx(0.0, abs=0.05)


def test_circular_difference_is_signed_and_short():
    assert circ_diff_hours(1.0, 23.0) == pytest.approx(2.0, abs=1e-6)
    assert circ_diff_hours(23.0, 1.0) == pytest.approx(-2.0, abs=1e-6)


def test_recommended_bedtime_is_at_night(ds):
    """The bug this exists for: blending 00:06 with 23:47 gave an 11:56 bedtime."""
    rec = recommend_bedtime(ds.frame)
    hour = int(rec.window_start.split(":")[0])
    assert hour >= 19 or hour <= 4, f"recommended {rec.window_start}, which is daytime"


def test_night_hours_helper_is_idempotent_on_evening_times():
    assert _to_night_hours(23.5) == 23.5
    assert _to_night_hours(0.5) == 24.5


# --- causality -------------------------------------------------------------
def test_rolling_features_never_look_forward(ds):
    """Truncating history must not change features on the nights that remain."""
    f_full = ds.frame
    cut = len(ds.nights) - 40
    f_trunc = build(ds.nights.iloc[:cut], already_tidy=True).frame

    check = ["tst_roll7_mean", "bed_roll7_std", "debt_7d_min", "sri_7"]
    for c in check:
        if c not in f_full or c not in f_trunc:
            continue
        a = f_full[c].iloc[:cut].values[-30:]
        b = f_trunc[c].values[-30:]
        both = np.isfinite(a) & np.isfinite(b)
        assert np.allclose(a[both], b[both], atol=1e-6), f"{c} uses future data"


def test_target_is_the_following_night(ds):
    f = ds.frame
    assert np.allclose(
        f["y_score_next"].values[:-1], f["sleep_score"].values[1:], equal_nan=True)


def test_target_excluded_from_predictors(ds):
    cols = ds.predictors("y_score_next")
    assert not any(c.startswith("y_") for c in cols)
    assert "y_score_next" not in cols


# --- units and messy input -------------------------------------------------
def test_hours_column_converted_to_minutes():
    n = 60
    d = pd.date_range("2025-01-01", periods=n)
    bed = [x + pd.Timedelta(hours=23) for x in d]
    raw = pd.DataFrame({
        "date": d, "start_time": bed,
        "end_time": [b + pd.Timedelta(hours=8) for b in bed],
        "TotalHoursAsleep": np.full(n, 7.5),
    })
    nights, _ = load_nightly(raw)
    assert nights["tst_min"].median() == pytest.approx(450, abs=1)


def test_session_crossing_midnight_without_a_date_is_repaired():
    raw = pd.DataFrame({
        "start_time": pd.to_datetime(["2025-03-01 23:30"]),
        "end_time": pd.to_datetime(["2025-03-01 07:00"]),  # earlier than start
    })
    nights, _ = load_nightly(raw)
    assert nights["tib_min"].iloc[0] == pytest.approx(450, abs=1)


def test_duplicate_nights_collapse_to_one():
    raw = generate_synthetic(40, seed=2)
    dup = pd.concat([raw, raw.iloc[:6]], ignore_index=True)
    nights, _ = load_nightly(dup)
    assert nights["night"].duplicated().sum() == 0


def test_impossible_session_is_dropped():
    """A 30-hour night must go, even when a duration column claims otherwise."""
    raw = generate_synthetic(40, seed=5).reset_index(drop=True)
    raw.loc[3, "waketime"] = raw.loc[3, "bedtime"] + pd.Timedelta(hours=30)
    nights, _ = load_nightly(raw)
    assert len(nights) == len(raw) - 1
    assert nights["tib_min"].max() < 1100


def test_duration_column_losing_to_the_clock():
    """When a reported duration contradicts the timestamps, the clock wins."""
    raw = pd.DataFrame({
        "start_time": pd.to_datetime(["2025-03-01 23:00"]),
        "end_time": pd.to_datetime(["2025-03-02 07:00"]),
        "sleep_duration_min": [900.0],  # 15h inside an 8h session
    })
    nights, _ = load_nightly(raw)
    assert nights["tib_min"].iloc[0] == pytest.approx(480, abs=1)
    assert nights["tst_min"].iloc[0] <= nights["tib_min"].iloc[0] + 1e-6


def test_too_few_nights_fails_loudly():
    with pytest.raises(ValueError, match="at least 8 nights"):
        build(generate_synthetic(30).head(5))


# --- graceful degradation --------------------------------------------------
def test_runs_without_stage_data():
    d = build(generate_synthetic(150, include_stages=False))
    assert not d.has_stages
    assert "deep_pct" not in d.frame.columns
    assert d.frame["sleep_score"].notna().sum() > 100


def test_bare_export_drops_efficiency_instead_of_awarding_it():
    """Start and end times only. Efficiency is 1.0 by construction, not merit."""
    n = 100
    d = pd.date_range("2025-01-01", periods=n)
    bed = [x + pd.Timedelta(hours=23, minutes=int(i % 40)) for i, x in enumerate(d)]
    raw = pd.DataFrame({"start_time": bed,
                        "end_time": [b + pd.Timedelta(minutes=450) for b in bed]})
    ds = build(raw)
    assert ds.tst_inferred
    assert "score_efficiency" not in ds.frame.columns
    assert any("efficiency" in n.lower() for n in ds.notes)


def test_readiness_reflects_history_length():
    assert build(generate_synthetic(20)).readiness()[0] in ("Collecting", "Descriptive")
    assert build(generate_synthetic(300)).readiness()[0] == "Established"


# --- modelling -------------------------------------------------------------
def test_walk_forward_folds_do_not_overlap_in_time(ds):
    res = walk_forward(ds.frame, "y_score_next", ds.predictors("y_score_next"))
    assert len(res.folds) > 1
    for a, b in zip(res.folds, res.folds[1:]):
        assert a.dates.max() < b.dates.min()
        assert b.n_train > a.n_train  # expanding, not sliding


def test_baselines_are_always_reported(ds):
    res = walk_forward(ds.frame, "y_score_next", ds.predictors("y_score_next"))
    models = set(res.metrics()["model"])
    assert "Persistence (last night)" in models
    assert "7-night rolling mean" in models


def test_short_history_declines_to_model():
    d = build(generate_synthetic(35))
    res = walk_forward(d.frame, "y_score_next", d.predictors("y_score_next"))
    assert res.folds == []
    assert "not enough data" in res.verdict().lower()


def test_forecast_withholds_prophet_on_short_series():
    d = build(generate_synthetic(60))
    fc = forecast_metric(d.frame, "tst_min")
    assert fc.method != "Prophet"


def test_forecast_horizon_length():
    d = build(generate_synthetic(200))
    assert len(forecast_metric(d.frame, "tst_min", horizon=14).forecast) == 14


# --- unsupervised ----------------------------------------------------------
def test_cluster_names_are_unique(ds):
    cl = find_archetypes(ds.frame)
    assert len(set(cl.names.values())) == len(cl.names)


def test_anomaly_rate_tracks_sensitivity(ds):
    low = find_anomalies(ds.frame, contamination=0.03).flags.mean()
    high = find_anomalies(ds.frame, contamination=0.12).flags.mean()
    assert low < high


# --- regularity metrics ----------------------------------------------------
def test_perfectly_regular_schedule_scores_near_maximum():
    n = 60
    d = pd.date_range("2025-01-01", periods=n)
    bed = [x + pd.Timedelta(hours=23) for x in d]
    raw = pd.DataFrame({"start_time": bed,
                        "end_time": [b + pd.Timedelta(hours=8) for b in bed]})
    f = engineer(load_nightly(raw)[0])
    assert f["sri_7"].dropna().mean() > 90


def test_erratic_schedule_scores_far_lower():
    rng = np.random.default_rng(4)
    n = 60
    d = pd.date_range("2025-01-01", periods=n)
    bed = [x + pd.Timedelta(hours=float(rng.uniform(19, 30))) for x in d]
    raw = pd.DataFrame({"start_time": bed,
                        "end_time": [b + pd.Timedelta(hours=7) for b in bed]})
    f = engineer(load_nightly(raw)[0])
    assert f["sri_7"].dropna().mean() < 75
