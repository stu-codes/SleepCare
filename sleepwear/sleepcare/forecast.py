"""Short-horizon forecasting, with the method chosen by how much data exists.

Prophet is the tool most people reach for and the wrong one for most
personal sleep series. It is built for long, strongly seasonal business
data; on a few months of nightly records it fits noise and reports
confident intervals around it. So Prophet is offered only once the series
is long enough to justify it, and it always competes against a seasonal
naive baseline that is genuinely hard to beat.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_SM = True
except ImportError:
    HAS_SM = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

PROPHET_MIN_NIGHTS = 180


@dataclass
class Forecast:
    history: pd.Series
    forecast: pd.Series
    lower: pd.Series | None
    upper: pd.Series | None
    method: str
    rationale: str
    backtest_mae: float | None = None
    baseline_mae: float | None = None


def _seasonal_naive(y: pd.Series, horizon: int) -> pd.Series:
    """Next week looks like last week. Hard to beat, easy to understand."""
    idx = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    last = y.iloc[-7:] if len(y) >= 7 else y
    vals = [last.iloc[i % len(last)] for i in range(horizon)]
    return pd.Series(vals, index=idx)


def _backtest(y: pd.Series, fn, horizon: int = 7, folds: int = 3) -> float | None:
    errs = []
    for k in range(folds, 0, -1):
        cut = len(y) - k * horizon
        if cut < 30:
            continue
        try:
            pred = fn(y.iloc[:cut], horizon)
        except Exception:
            continue
        actual = y.iloc[cut : cut + horizon]
        n = min(len(pred), len(actual))
        if n:
            errs.append(np.mean(np.abs(pred.values[:n] - actual.values[:n])))
    return float(np.mean(errs)) if errs else None


def _ets(y: pd.Series, horizon: int) -> pd.Series:
    idx = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    if HAS_SM and len(y) >= 30:
        seasonal = "add" if len(y) >= 28 else None
        m = ExponentialSmoothing(
            y, trend=None, seasonal=seasonal, seasonal_periods=7 if seasonal else None,
            initialization_method="estimated",
        ).fit()
        return pd.Series(np.asarray(m.forecast(horizon)), index=idx)
    return pd.Series(y.ewm(span=7).mean().iloc[-1], index=idx)


def _prophet(y: pd.Series, horizon: int) -> pd.Series:
    d = pd.DataFrame({"ds": y.index, "y": y.values})
    m = Prophet(weekly_seasonality=True, yearly_seasonality=False,
                daily_seasonality=False, changepoint_prior_scale=0.05)
    m.fit(d)
    fut = m.make_future_dataframe(periods=horizon)
    out = m.predict(fut).set_index("ds")["yhat"]
    return out.iloc[-horizon:]


def forecast_metric(
    f: pd.DataFrame, column: str = "tst_min", horizon: int = 7, method: str = "auto"
) -> Forecast:
    """Forecast a nightly metric forward, choosing an appropriate method."""
    y = pd.Series(f[column].values, index=pd.DatetimeIndex(f["night"])).dropna()
    # A complete, evenly spaced daily index. Smoothing models assume regular
    # spacing, and a series with holes in it makes them diverge badly rather
    # than fail loudly.
    y = y.resample("D").mean()
    y = y.interpolate(limit_direction="both")
    y = y.asfreq("D")

    if len(y) < 21:
        fc = _seasonal_naive(y, horizon)
        return Forecast(y, fc, None, None, "Seasonal naive",
                        f"Only {len(y)} nights. Anything more elaborate would be "
                        "fitting noise. Repeating the last week is the honest forecast.")

    candidates: dict[str, callable] = {"Seasonal naive": _seasonal_naive}
    if HAS_SM or True:
        candidates["Exponential smoothing"] = _ets
    if method in ("auto", "prophet") and HAS_PROPHET and len(y) >= PROPHET_MIN_NIGHTS:
        candidates["Prophet"] = _prophet

    if method != "auto" and method in candidates:
        chosen = method
        rationale = f"{method} selected manually."
    else:
        scores = {k: _backtest(y, fn, horizon) for k, fn in candidates.items()}
        scores = {k: v for k, v in scores.items() if v is not None}
        if scores:
            chosen = min(scores, key=scores.get)
            ranked = ", ".join(f"{k} {v:.1f}" for k, v in sorted(scores.items(), key=lambda t: t[1]))
            rationale = f"Chosen by rolling backtest MAE ({ranked})."
        else:
            chosen = "Seasonal naive"
            rationale = "Backtest inconclusive; defaulting to the naive forecast."

    if "Prophet" not in candidates and HAS_PROPHET and len(y) < PROPHET_MIN_NIGHTS:
        rationale += (f" Prophet withheld: it needs roughly {PROPHET_MIN_NIGHTS} nights "
                      f"to model weekly seasonality reliably and you have {len(y)}.")

    fc = candidates[chosen](y, horizon)
    resid = float(np.std(y.diff().dropna())) if len(y) > 3 else 0.0
    band = 1.28 * resid * np.sqrt(np.arange(1, len(fc) + 1)) ** 0.5

    bt = _backtest(y, candidates[chosen], horizon)
    base = _backtest(y, _seasonal_naive, horizon)
    return Forecast(y, fc, fc - band, fc + band, chosen, rationale, bt, base)


# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------
@dataclass
class Recommendation:
    window_start: str
    window_end: str
    target_wake: str
    predicted_gain: float
    reasoning: list[str]
    confidence: str
    evidence: list[str]


def _fmt(h: float) -> str:
    h = h % 24
    return f"{int(h):02d}:{int(round((h - int(h)) * 60)) % 60:02d}"


def _to_night_hours(h: float) -> float:
    """Put a clock hour on the continuous night scale used by bed_dec.

    Averaging 00:06 and 23:47 on the number line gives 11:56, which is how
    an earlier version of this function recommended a midday bedtime. On the
    night scale those two times are 24.1 and 23.78, and the average is the
    23:56 a person would expect.
    """
    return h + 24.0 if h < 12.0 else h


def recommend_bedtime(
    f: pd.DataFrame,
    model_res=None,
    feature_cols: list[str] | None = None,
    sleep_need_min: float = 465.0,
) -> Recommendation:
    """Suggest a bedtime window.

    The recommendation is anchored on regularity and chronotype alignment
    rather than on maximising a predicted deep-sleep figure, because
    regularity is what the outcome evidence actually supports and because
    wearable stage estimates are too noisy to optimise against.

    When a validated model exists, it is used to sweep candidate bedtimes
    counterfactually. When it does not, the recommendation falls back to
    the person's own best nights.
    """
    reasoning: list[str] = []
    recent = f.tail(60)

    msf = float(f.attrs.get("chronotype_msf", recent["mid_dec"].median()))
    need_h = sleep_need_min / 60.0

    # Anchor: the bedtime that centres sleep on your own circadian midpoint.
    # All bedtime arithmetic happens on the night scale, never on raw clock
    # hours, so that times either side of midnight average correctly.
    msf_nh = _to_night_hours(msf)
    anchor_bed = _to_night_hours(msf_nh - need_h / 2.0)
    reasoning.append(
        f"Your mid-sleep on free days sits near {_fmt(msf)}, so a night centred "
        f"there starts around {_fmt(anchor_bed)}."
    )

    # Evidence from your own best nights.
    if "sleep_score" in f.columns and f["sleep_score"].notna().sum() > 20:
        top = f.nlargest(max(10, len(f) // 8), "sleep_score")
        emp_bed = _to_night_hours(float(top["bed_dec"].median()) % 24)
        emp_spread = float(top["bed_dec"].std())
        reasoning.append(
            f"Your highest-scoring nights began around {_fmt(emp_bed)} "
            f"(spread {emp_spread:.1f}h)."
        )
        center = 0.5 * anchor_bed + 0.5 * emp_bed
    else:
        emp_spread = 0.5
        center = anchor_bed

    # Counterfactual sweep, if a model earned the right to be used.
    gain = 0.0
    confidence = "Low"
    if model_res is not None and feature_cols and getattr(model_res, "model", None) is not None:
        if model_res.beats_baseline:
            row = f[feature_cols].iloc[[-1]].copy()
            base_bed = float(f["bed_dec"].iloc[-1])
            best_bed, best_pred = center, -np.inf
            for cand in np.arange(center - 2.0, center + 2.0 + 1e-9, 0.25):
                trial = row.copy()
                if "bed_dec" in trial:
                    trial["bed_dec"] = cand
                if "mid_dec" in trial:
                    trial["mid_dec"] = cand + need_h / 2
                if "mid_shift_h" in trial:
                    trial["mid_shift_h"] = abs((cand + need_h / 2) - float(f["mid_dec"].iloc[-1]))
                if "misalignment_h" in trial:
                    from .features import circ_diff_hours
                    trial["misalignment_h"] = abs(
                        circ_diff_hours((cand + need_h / 2) % 24, msf % 24))
                for nm, col in (("bed_sin", np.sin), ("bed_cos", np.cos)):
                    if nm in trial:
                        trial[nm] = col(2 * np.pi * (cand % 24) / 24)
                try:
                    p = float(model_res.model.predict(trial)[0])
                except Exception:
                    continue
                if p > best_pred:
                    best_pred, best_bed = p, cand
            try:
                base_pred = float(model_res.model.predict(row)[0])
                gain = best_pred - base_pred
                center = 0.6 * best_bed + 0.4 * center
                reasoning.append(
                    f"Sweeping bedtimes through the validated model favours "
                    f"{_fmt(best_bed)}, worth about {gain:+.1f} points against "
                    f"repeating last night's {_fmt(base_bed)}."
                )
                confidence = "Moderate"
            except Exception:
                pass
        else:
            reasoning.append(
                "The trained model did not beat the naive baseline, so it was not "
                "used to pick this window. The recommendation rests on your own "
                "best nights and your chronotype instead."
            )

    # Regularity check: a tight window is the actual intervention.
    reg = float(recent["bed_roll7_std"].dropna().median()) if "bed_roll7_std" in recent else np.nan
    if np.isfinite(reg):
        if reg > 1.0:
            reasoning.append(
                f"Your bedtime varies by about {reg:.1f}h week to week. Narrowing that "
                "is likely to matter more than moving the average."
            )
        else:
            reasoning.append(f"Your bedtime is already consistent (±{reg:.1f}h). Hold it there.")

    # Social jetlag.
    if "social_jetlag_h" in recent and recent["social_jetlag_h"].notna().any():
        sj = float(recent["social_jetlag_h"].dropna().iloc[-1])
        if abs(sj) > 1.0:
            reasoning.append(
                f"Weekend sleep runs {abs(sj):.1f}h later than weekdays. Pulling weekend "
                "nights earlier reduces the Monday realignment cost."
            )

    half = float(np.clip(emp_spread if np.isfinite(emp_spread) else 0.5, 0.25, 0.75))
    if len(f) >= 90 and confidence == "Moderate":
        confidence = "Moderate"
    elif len(f) < 45:
        confidence = "Low"

    return Recommendation(
        window_start=_fmt(center - half),
        window_end=_fmt(center + half),
        target_wake=_fmt(center + need_h),
        predicted_gain=gain,
        reasoning=reasoning,
        confidence=confidence,
        evidence=[
            "Sleep regularity predicts mortality risk more strongly than sleep "
            "duration in large accelerometry cohorts (Windred et al., Sleep, 2024).",
            "Mid-sleep on free days is a standard behavioural marker of circadian "
            "phase (Munich ChronoType Questionnaire).",
        ],
    )
