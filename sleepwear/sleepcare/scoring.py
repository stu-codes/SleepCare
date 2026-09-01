"""A sleep score you can actually inspect.

OnePlus computes a 0-100 sleep score and shows it in OHealth, but there is
no evidence it crosses the Health Connect boundary, and Health Connect has
no standard field for it. So we compute our own.

The advantage of losing the vendor score is that this one is legible. Every
component is named, weighted visibly, and can be turned off. A score whose
construction you cannot see is not a measurement, it is a mood ring.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {
    "duration": 0.35,
    "efficiency": 0.25,
    "regularity": 0.25,
    "restfulness": 0.15,
}


def _duration_component(tst_min: pd.Series, need: float) -> pd.Series:
    """Full credit at target, tapering both directions.

    Oversleeping is penalised far more gently than undersleeping, because
    the evidence for harm is asymmetric and because a long night after a
    short one is recovery, not a failure.
    """
    ratio = tst_min / need
    under = 100 - 145 * np.clip(1 - ratio, 0, 1)
    over = 100 - 55 * np.clip(ratio - 1.12, 0, 1)
    return pd.Series(np.where(ratio < 1, under, over), index=tst_min.index).clip(0, 100)


def _efficiency_component(eff: pd.Series) -> pd.Series:
    """0.85 is unremarkable, 0.95 is excellent, below 0.75 is a problem."""
    return ((eff - 0.70) / 0.25 * 100).clip(0, 100)


def _regularity_component(mid_shift_h: pd.Series, sri: pd.Series | None) -> pd.Series:
    """Prefers the windowed SRI, falls back to last-night timing swing."""
    if sri is not None and sri.notna().sum() > len(sri) * 0.4:
        comp = ((sri + 20) / 100 * 100).clip(0, 100)
        # Backfill early nights, where the window has not filled yet.
        fallback = (100 - 42 * mid_shift_h.abs()).clip(0, 100)
        return comp.fillna(fallback)
    return (100 - 42 * mid_shift_h.abs()).clip(0, 100)


def _restfulness_component(f: pd.DataFrame) -> pd.Series | None:
    """Deep+REM share if stages exist, otherwise overnight physiology."""
    if "deep_pct" in f and "rem_pct" in f:
        share = f["deep_pct"].fillna(0) + f["rem_pct"].fillna(0)
        # ~0.40 combined is typical for a healthy adult night.
        return (share / 0.42 * 100).clip(0, 100)
    if "rhr_delta" in f and f["rhr_delta"].notna().any():
        # An overnight resting HR above your own baseline reads as strain.
        return (100 - 11 * f["rhr_delta"].clip(-4, 9)).clip(0, 100)
    return None


def compute_score(
    f: pd.DataFrame,
    sleep_need_min: float = 465.0,
    weights: dict[str, float] | None = None,
    drop_efficiency: bool = False,
) -> pd.DataFrame:
    """Return the score plus every component, so nothing is hidden.

    Args:
        drop_efficiency: set when the export gave no separate asleep time, so
            efficiency is 1.0 by construction and would award free marks.
    """
    w = dict(weights or DEFAULT_WEIGHTS)
    if not w:
        w = dict(DEFAULT_WEIGHTS)

    comps = {
        "duration": _duration_component(f["tst_min"], sleep_need_min),
        "regularity": _regularity_component(
            f.get("mid_shift_h", pd.Series(0.0, index=f.index)),
            f.get("sri_7"),
        ),
    }
    if not drop_efficiency:
        comps["efficiency"] = _efficiency_component(f["efficiency"])
    else:
        w.pop("efficiency", None)
    rest = _restfulness_component(f)
    if rest is not None:
        comps["restfulness"] = rest
    else:
        # Redistribute the missing weight rather than silently scoring zero.
        w.pop("restfulness", None)

    w = {k: v for k, v in w.items() if k in comps}
    if not w or sum(w.values()) <= 0:
        w = {k: 1.0 for k in comps}
    total_w = sum(w[k] for k in comps)
    score = sum(comps[k] * w.get(k, 0.0) for k in comps) / total_w

    out = pd.DataFrame({f"score_{k}": v for k, v in comps.items()}, index=f.index)
    out["sleep_score"] = score.clip(0, 100)
    return out


def attach_targets(f: pd.DataFrame, score: pd.Series) -> pd.DataFrame:
    """Add tonight's score and the next-night targets the model predicts."""
    out = f.copy()
    out["sleep_score"] = score
    out["score_lag1"] = out["sleep_score"].shift(1)
    out["y_score_next"] = out["sleep_score"].shift(-1)
    out["y_tst_next"] = out["tst_min"].shift(-1)
    out["y_eff_next"] = out["efficiency"].shift(-1)
    if "subjective" in out.columns:
        out["y_subjective_next"] = out["subjective"].shift(-1)
    return out
