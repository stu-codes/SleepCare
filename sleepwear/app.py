"""SleepCare — personal sleep analytics for OnePlus Watch 2R exports.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sleepcare.features import build_epoch_matrix
from sleepcare.forecast import forecast_metric, recommend_bedtime
from sleepcare.ingest import generate_synthetic
from sleepcare.models import predict_next, walk_forward
from sleepcare.pipeline import build
from sleepcare.unsupervised import explain_anomaly, find_anomalies, find_archetypes

st.set_page_config(page_title="SleepCare", page_icon="◐", layout="wide",
                   initial_sidebar_state="expanded")

# --------------------------------------------------------------------------
# Visual system
#
# The reference is a sleep lab readout, not a wellness app: instrument
# panels on a dark ward-at-night ground, monospaced numerals because these
# are measurements, and a phosphor green borrowed from old medical
# monitors as the single accent. Stage colours run cool-to-warm from deep
# sleep to wake, which is the convention hypnograms already use.
# --------------------------------------------------------------------------
INK, MUTED, LINE = "#E8ECF8", "#7E8AAB", "#1E2A44"
BG, PANEL = "#070B18", "#101728"
PHOSPHOR, ALERT, AMBER = "#5FE0B4", "#E86A6A", "#E8A33D"
STAGE = {"Deep": "#3B5BDB", "Light": "#5C8AE6", "REM": "#5FE0B4", "Awake": "#E8A33D"}

st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  .stApp {{ background:{BG}; color:{INK}; }}
  html, body, [class*="css"] {{ font-family:'Inter',sans-serif; }}
  h1,h2,h3 {{ font-family:'IBM Plex Mono',monospace !important; letter-spacing:-0.02em; }}
  .masthead {{ border-bottom:1px solid {LINE}; padding:0 0 14px; margin-bottom:22px;
    display:flex; align-items:baseline; gap:18px; flex-wrap:wrap; }}
  .masthead h1 {{ font-size:1.7rem; margin:0; font-weight:600; }}
  .masthead .sub {{ color:{MUTED}; font-family:'IBM Plex Mono',monospace; font-size:.78rem; }}
  .panel {{ background:{PANEL}; border:1px solid {LINE}; border-radius:3px;
    padding:16px 18px; height:100%; }}
  .readout {{ font-family:'IBM Plex Mono',monospace; font-size:2.1rem; font-weight:600;
    line-height:1.1; }}
  .label {{ font-family:'IBM Plex Mono',monospace; font-size:.68rem; letter-spacing:.14em;
    text-transform:uppercase; color:{MUTED}; margin-bottom:6px; }}
  .foot {{ color:{MUTED}; font-size:.78rem; margin-top:4px; }}
  .tag {{ display:inline-block; font-family:'IBM Plex Mono',monospace; font-size:.66rem;
    letter-spacing:.1em; text-transform:uppercase; padding:3px 9px; border-radius:2px;
    border:1px solid {LINE}; color:{MUTED}; }}
  .tag.ok {{ color:{PHOSPHOR}; border-color:{PHOSPHOR}44; }}
  .tag.warn {{ color:{AMBER}; border-color:{AMBER}44; }}
  .tag.bad {{ color:{ALERT}; border-color:{ALERT}44; }}
  .stTabs [data-baseweb="tab-list"] {{ gap:2px; border-bottom:1px solid {LINE}; }}
  .stTabs [data-baseweb="tab"] {{ font-family:'IBM Plex Mono',monospace; font-size:.8rem;
    background:transparent; color:{MUTED}; border-radius:0; padding:9px 16px; }}
  .stTabs [aria-selected="true"] {{ color:{PHOSPHOR}; border-bottom:2px solid {PHOSPHOR}; }}
  section[data-testid="stSidebar"] {{ background:{PANEL}; border-right:1px solid {LINE}; }}
  .stDataFrame {{ border:1px solid {LINE}; }}
  hr {{ border-color:{LINE}; }}
  @media (prefers-reduced-motion: reduce) {{ * {{ animation:none !important; transition:none !important; }} }}
</style>""", unsafe_allow_html=True)


def panel(label: str, value: str, foot: str = "", color: str = INK):
    st.markdown(
        f"<div class='panel'><div class='label'>{label}</div>"
        f"<div class='readout' style='color:{color}'>{value}</div>"
        f"<div class='foot'>{foot}</div></div>", unsafe_allow_html=True)


def style_fig(fig, height=340, legend=True):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Mono, monospace", size=11, color=MUTED),
        margin=dict(l=8, r=8, t=28, b=8), showlegend=legend,
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=10)),
        hoverlabel=dict(bgcolor=PANEL, font_size=11, bordercolor=LINE),
    )
    fig.update_xaxes(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE)
    return fig


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _demo(n, seed, stages):
    return generate_synthetic(n_nights=n, seed=seed, include_stages=stages)


@st.cache_data(show_spinner=False)
def _build(raw, need, weights_t):
    return build(raw, sleep_need_min=need, weights=dict(weights_t))


@st.cache_data(show_spinner=False)
def _model(frame, target, cols):
    return walk_forward(frame, target, cols)


with st.sidebar:
    st.markdown("<div class='label'>Source</div>", unsafe_allow_html=True)
    source = st.radio("Source", ["Demo data", "Upload export"], label_visibility="collapsed")

    raw, is_demo = None, source == "Demo data"
    if is_demo:
        n = st.slider("Nights of history", 21, 500, 300, 7)
        stages_on = st.checkbox("Include sleep stages", value=False,
                                help="Off by default: OHealth's Health Connect write is "
                                     "often duration-only, so this is the realistic case.")
        raw = _demo(n, 7, stages_on)
    else:
        up = st.file_uploader("CSV export", type=["csv"], label_visibility="collapsed")
        if up is not None:
            raw = pd.read_csv(up)
        else:
            st.caption("Export from Health Connect via Health Data Export, "
                       "then drop the CSV here. See the Data tab for the full route.")

    st.markdown("---")
    st.markdown("<div class='label'>Personal settings</div>", unsafe_allow_html=True)
    need_h = st.slider("Nightly sleep need (h)", 6.0, 9.5, 7.75, 0.25)
    st.caption("Your own target, not a prescription.")

    with st.expander("Score weights"):
        w_dur = st.slider("Duration", 0.0, 1.0, 0.35, 0.05)
        w_eff = st.slider("Efficiency", 0.0, 1.0, 0.25, 0.05)
        w_reg = st.slider("Regularity", 0.0, 1.0, 0.25, 0.05)
        w_rest = st.slider("Restfulness", 0.0, 1.0, 0.15, 0.05)

st.markdown(
    "<div class='masthead'><h1>SleepCare</h1>"
    "<span class='sub'>n-of-1 sleep analytics · OnePlus Watch 2R / Health Connect</span></div>",
    unsafe_allow_html=True)

if raw is None:
    st.info("Load a CSV or switch to demo data to begin.")
    st.stop()

weights = (("duration", w_dur), ("efficiency", w_eff),
           ("regularity", w_reg), ("restfulness", w_rest))
try:
    ds = _build(raw, need_h * 60, weights)
except Exception as e:
    st.error(f"Could not read that file: {e}")
    st.caption("SleepCare needs at minimum a sleep start and a sleep end column. "
               "Check the Data tab for accepted column names.")
    st.stop()

f = ds.frame
stage, stage_note = ds.readiness()
target = "y_score_next"
cols = ds.predictors(target)
res = _model(f, target, cols)

if is_demo:
    st.warning(
        "**Demo data.** These nights are simulated with known structure, so the model "
        "scores far better here than it will on your real watch data. Treat the layout "
        "as real and the accuracy as fiction.", icon="⚠")

tabs = st.tabs(["Tonight", "Patterns", "Model", "Forecast", "Anomalies", "Data"])

# ==========================================================================
# TONIGHT
# ==========================================================================
with tabs[0]:
    last = f.iloc[-1]
    pred = predict_next(res, f, cols)
    rec = recommend_bedtime(f, res, cols, sleep_need_min=need_h * 60)

    c = st.columns(4)
    with c[0]:
        sc = last["sleep_score"]
        col = PHOSPHOR if sc >= 75 else (AMBER if sc >= 55 else ALERT)
        panel("Last night", f"{sc:.0f}", f"{last['night']:%a %d %b}", col)
    with c[1]:
        h, m = divmod(int(last["tst_min"]), 60)
        panel("Time asleep", f"{h}h {m:02d}m", f"{last['efficiency']*100:.0f}% efficiency")
    with c[2]:
        panel("Sleep midpoint", f"{int(last['mid_dec']%24):02d}:"
                                f"{int(round((last['mid_dec']%1)*60))%60:02d}",
              f"chronotype {int(ds.chronotype_msf%24):02d}:"
              f"{int(round((ds.chronotype_msf%1)*60))%60:02d}")
    with c[3]:
        if pred is not None and res.folds:
            mae = res.metrics().set_index("model").loc["XGBoost", "MAE"]
            panel("Tonight, predicted", f"{pred:.0f}", f"±{mae:.0f} typical error")
        else:
            panel("Tonight, predicted", "—", "not enough history to validate")

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.15, 1])

    with left:
        st.markdown("### Suggested bedtime window")
        conf_cls = {"Low": "bad", "Moderate": "warn", "High": "ok"}[rec.confidence]
        st.markdown(
            f"<div class='panel'><div class='label'>Aim to be asleep between</div>"
            f"<div class='readout' style='color:{PHOSPHOR}'>{rec.window_start} – {rec.window_end}</div>"
            f"<div class='foot'>Waking near {rec.target_wake} · "
            f"<span class='tag {conf_cls}'>{rec.confidence} confidence</span></div></div>",
            unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        for r in rec.reasoning:
            st.markdown(f"- {r}")
        with st.expander("What this is based on"):
            for e in rec.evidence:
                st.markdown(f"- {e}")
            st.caption(
                "This is a pattern in your own data, not medical advice. Consumer "
                "wearables agree with clinical polysomnography only moderately, and "
                "no wrist device can identify a sleep disorder. Persistent poor sleep, "
                "loud snoring, or daytime sleepiness are worth raising with a doctor.")

    with right:
        st.markdown("### Last 30 nights")
        recent = f.tail(30)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=recent["night"], y=recent["sleep_score"], name="Score",
            marker_color=[PHOSPHOR if v >= 75 else (AMBER if v >= 55 else ALERT)
                          for v in recent["sleep_score"]],
            marker_line_width=0, hovertemplate="%{x|%a %d %b}<br>score %{y:.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=recent["night"], y=recent["sleep_score"].rolling(7, min_periods=3).mean(),
            name="7-night mean", line=dict(color=INK, width=1.6)))
        fig.update_yaxes(range=[0, 100], title="Score")
        st.plotly_chart(style_fig(fig, 300), width="stretch")

    st.markdown("### Score components")
    comp_cols = [c for c in f.columns if c.startswith("score_") and c != "score_lag1"]
    cc = st.columns(len(comp_cols))
    for i, cname in enumerate(comp_cols):
        with cc[i]:
            v = last[cname]
            avg = f[cname].tail(30).mean()
            d = v - avg
            panel(cname.replace("score_", ""), f"{v:.0f}",
                  f"{d:+.0f} vs your 30-night average",
                  PHOSPHOR if d >= 0 else AMBER)

# ==========================================================================
# PATTERNS  — the actogram is the signature view
# ==========================================================================
with tabs[1]:
    st.markdown("### Actogram")
    st.caption(
        "Each row is one day, double-plotted so a 48-hour span runs left to right and "
        "consecutive rows overlap. This is how chronobiologists have read rest-activity "
        "data for decades: a straight vertical edge means a stable rhythm, a ragged or "
        "drifting one means your body clock is being moved around.")

    mat, days = build_epoch_matrix(ds.nights)
    show = min(len(mat) - 1, 120)
    sub, sdays = mat[-show - 1:], days[-show - 1:]
    double = np.hstack([sub[:-1], sub[1:]])

    fig = go.Figure(go.Heatmap(
        z=double, x=np.arange(double.shape[1]) * 5 / 60,
        y=[d.strftime("%d %b") for d in sdays[:-1]],
        colorscale=[[0, "rgba(0,0,0,0)"], [1, PHOSPHOR]], showscale=False,
        hovertemplate="%{y}<br>%{x:.1f}h<extra></extra>"))
    fig.update_xaxes(title="Hours (two days shown per row)",
                     tickvals=list(range(0, 49, 6)))
    fig.update_yaxes(autorange="reversed", showgrid=False,
                     nticks=min(20, len(sdays)))
    st.plotly_chart(style_fig(fig, 560, legend=False), width="stretch")

    st.markdown("---")
    c = st.columns(4)
    latest = f.dropna(subset=["sri_7"]).tail(1)
    with c[0]:
        v = float(latest["sri_7"].iloc[0]) if len(latest) else np.nan
        col = PHOSPHOR if v > 75 else (AMBER if v > 55 else ALERT)
        panel("Sleep Regularity Index", f"{v:.0f}" if np.isfinite(v) else "—",
              "-100 random, +100 identical every night", col)
    with c[1]:
        v = float(latest["is_14"].iloc[0]) if len(latest) and "is_14" in latest else np.nan
        panel("Interdaily stability", f"{v:.2f}" if np.isfinite(v) else "—",
              "rhythm strength against a 24h template")
    with c[2]:
        v = float(latest["iv_14"].iloc[0]) if len(latest) and "iv_14" in latest else np.nan
        panel("Intradaily variability", f"{v:.2f}" if np.isfinite(v) else "—",
              "fragmentation; lower is smoother")
    with c[3]:
        sj = f["social_jetlag_h"].dropna()
        v = float(sj.iloc[-1]) if len(sj) else np.nan
        col = PHOSPHOR if abs(v) < 1 else (AMBER if abs(v) < 2 else ALERT)
        panel("Social jetlag", f"{v:+.1f}h" if np.isfinite(v) else "—",
              "weekend minus weekday mid-sleep", col)

    st.markdown("---")
    lc, rc = st.columns(2)
    with lc:
        st.markdown("### Bedtime and wake drift")
        fig = go.Figure()
        for name, col_, colr in (("Bedtime", "bed_dec", "#5C8AE6"),
                                 ("Wake", "wake_dec", PHOSPHOR)):
            fig.add_trace(go.Scatter(
                x=f["night"], y=f[col_] % 24, mode="markers", name=name,
                marker=dict(size=4, color=colr, opacity=.55)))
            fig.add_trace(go.Scatter(
                x=f["night"], y=(f[col_].rolling(14, min_periods=5).mean()) % 24,
                mode="lines", name=f"{name} trend",
                line=dict(color=colr, width=1.8), showlegend=False))
        fig.update_yaxes(title="Clock hour")
        st.plotly_chart(style_fig(fig), width="stretch")

    with rc:
        st.markdown("### Night archetypes")
        cl = find_archetypes(f)
        if cl is None:
            st.caption("Needs about 20 nights before clusters mean anything.")
        else:
            fig = go.Figure()
            palette = [PHOSPHOR, "#5C8AE6", AMBER, "#B98CE8", ALERT, "#6FD3E0"]
            for k in range(cl.k):
                m = cl.labels == k
                fig.add_trace(go.Scatter(
                    x=cl.coords[m, 0], y=cl.coords[m, 1], mode="markers",
                    name=f"{cl.names[k]} ({m.sum()})",
                    marker=dict(size=7, color=palette[k % len(palette)], opacity=.75)))
            fig.update_xaxes(title="", showticklabels=False)
            fig.update_yaxes(title="", showticklabels=False)
            st.plotly_chart(style_fig(fig), width="stretch")
            st.caption(f"k={cl.k} chosen by silhouette ({cl.silhouette:.2f}). "
                       "Axes are the first two principal components, so distance is "
                       "meaningful but the directions are not.")
            st.dataframe(cl.profile.rename(index=cl.names).round(2),
                         width="stretch")

# ==========================================================================
# MODEL
# ==========================================================================
with tabs[2]:
    st.markdown("### Walk-forward validation")
    st.caption(
        "Each fold trains only on nights before the test block and predicts forward, "
        "the way the model would actually be used. Random cross-validation would let "
        "it train on next week to predict last week and report a much prettier number.")

    if not res.folds:
        st.info(stage_note)
        st.caption(f"Walk-forward needs roughly 57 usable nights. You have {res.n_used}.")
    else:
        m = res.metrics()
        st.dataframe(
            m.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.3f}",
                            "Skill vs best baseline": "{:+.1%}"}),
            width="stretch", hide_index=True)

        verdict = res.verdict()
        (st.success if res.beats_baseline else st.warning)(verdict)

        c = st.columns(3)
        with c[0]:
            panel("Folds", str(len(res.folds)), "expanding window")
        with c[1]:
            panel("Nights used", str(res.n_used), ds.date_range)
        with c[2]:
            panel("Predictors", str(len(cols)),
                  "stages included" if ds.has_stages else "no stage data")

        st.markdown("### Predicted against actual")
        p = res.predictions
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=p.index, y=p["actual"], name="Actual",
                                 line=dict(color=INK, width=1.8)))
        fig.add_trace(go.Scatter(x=p.index, y=p["predicted"], name="Model",
                                 line=dict(color=PHOSPHOR, width=1.8)))
        fig.add_trace(go.Scatter(x=p.index, y=p["persistence"], name="Persistence baseline",
                                 line=dict(color=MUTED, width=1, dash="dot")))
        fig.update_yaxes(title="Next-night score")
        st.plotly_chart(style_fig(fig, 360), width="stretch")

        if res.importances is not None:
            st.markdown("### What the model leans on")
            top = res.importances.head(16).sort_values()
            fig = go.Figure(go.Bar(
                x=top.values, y=top.index, orientation="h",
                marker_color=PHOSPHOR, marker_line_width=0))
            st.plotly_chart(style_fig(fig, 420, legend=False), width="stretch")
            st.caption(
                "Gain importance shows what the trees split on, which is not the same "
                "as what causes good sleep. Correlated features also steal credit from "
                "each other. Read this as a description of the model, not of your body.")

        with st.expander("Why the model is deliberately small"):
            st.markdown(
                "With a few hundred nights from one person, a deep ensemble will "
                "memorise the training set and look excellent right up until it meets "
                "a new night. This one is capped at depth 3 with strong regularisation, "
                "and it is reported against baselines that are genuinely hard to beat. "
                "If the naive baseline wins, the app says so rather than quietly "
                "shipping the fancier model.")

# ==========================================================================
# FORECAST
# ==========================================================================
with tabs[3]:
    metric = st.selectbox("Metric", ["tst_min", "sleep_score", "efficiency", "mid_dec"],
                          format_func=lambda c: {"tst_min": "Time asleep (min)",
                                                 "sleep_score": "Sleep score",
                                                 "efficiency": "Efficiency",
                                                 "mid_dec": "Sleep midpoint"}[c])
    horizon = st.slider("Nights ahead", 3, 21, 7)
    fc = forecast_metric(f, metric, horizon=horizon)

    c = st.columns(3)
    with c[0]:
        panel("Method", fc.method, "selected automatically")
    with c[1]:
        panel("Backtest MAE", f"{fc.backtest_mae:.1f}" if fc.backtest_mae else "—",
              "rolling origin")
    with c[2]:
        panel("Naive MAE", f"{fc.baseline_mae:.1f}" if fc.baseline_mae else "—",
              "seasonal naive")
    st.caption(fc.rationale)

    hist = fc.history.tail(90)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="History",
                             line=dict(color=INK, width=1.5)))
    if fc.lower is not None:
        fig.add_trace(go.Scatter(
            x=list(fc.forecast.index) + list(fc.forecast.index[::-1]),
            y=list(fc.upper.values) + list(fc.lower.values[::-1]),
            fill="toself", fillcolor="rgba(95,224,180,0.10)",
            line=dict(width=0), name="80% interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc.forecast.index, y=fc.forecast.values, name="Forecast",
                             line=dict(color=PHOSPHOR, width=2, dash="dash")))
    st.plotly_chart(style_fig(fig, 380), width="stretch")

    st.info(
        "A sleep forecast predicts your habits, not your future. It assumes next week "
        "resembles recent weeks, which is exactly what a holiday, a deadline or a new "
        "baby will break.", icon="ℹ️")

# ==========================================================================
# ANOMALIES
# ==========================================================================
with tabs[4]:
    st.markdown("### Nights unlike your own pattern")
    sens = st.slider("Sensitivity", 0.02, 0.15, 0.06, 0.01,
                     help="Expected share of nights flagged.")
    an = find_anomalies(f, contamination=sens)

    if an is None:
        st.info("Needs about 25 nights.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=f["night"][~an.flags], y=f["sleep_score"][~an.flags], mode="markers",
            name="Typical", marker=dict(size=5, color=MUTED, opacity=.5)))
        fig.add_trace(go.Scatter(
            x=f["night"][an.flags], y=f["sleep_score"][an.flags], mode="markers",
            name="Flagged", marker=dict(size=10, color=ALERT, symbol="diamond-open",
                                        line=dict(width=1.6))))
        fig.update_yaxes(title="Sleep score")
        st.plotly_chart(style_fig(fig, 320), width="stretch")

        idx = np.where(an.flags)[0]
        order = idx[np.argsort(-an.scores[idx])][:12]
        rows = [{
            "Night": f["night"].iloc[i].strftime("%a %d %b %Y"),
            "Score": round(float(f["sleep_score"].iloc[i]), 1),
            "Asleep": f"{int(f['tst_min'].iloc[i])//60}h {int(f['tst_min'].iloc[i])%60:02d}m",
            "Why it stood out": explain_anomaly(an, f, int(i)),
        } for i in order]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        st.caption(
            "An anomaly means the night was statistically unlike your others. That is "
            "usually a late flight, a drink, a cold, or a child. It is not a clinical "
            "finding, and this tool cannot detect sleep apnea or any other disorder.")

# ==========================================================================
# DATA
# ==========================================================================
with tabs[5]:
    c = st.columns(3)
    with c[0]:
        panel("Nights", str(ds.n_nights), ds.date_range)
    with c[1]:
        panel("Pipeline stage", stage, stage_note.split(". ", 1)[-1])
    with c[2]:
        panel("Stage data", "Yes" if ds.has_stages else "No",
              "deep/REM features active" if ds.has_stages
              else "timing and regularity only")

    for n in ds.notes:
        st.caption(f"· {n}")

    st.markdown("---")
    st.markdown("### Getting data off the Watch 2R")
    st.markdown("""
OHealth has no export function. The only route to a file is Health Connect:

1. **OHealth → Profile → Health Connect → Connect**, then authorise sleep, heart
   rate and blood oxygen individually.
2. Install a Health Connect reader that writes CSV — **Health Data Export** is
   open source and does this well.
3. **Check what actually arrived.** Open Health Connect → Data and access → Sleep
   and look at an OHealth entry. If you see one duration per night rather than
   separate deep, light, REM and awake segments, your stages are not crossing the
   bridge. This app runs either way, but the deep and REM features switch off.
4. **Schedule the export daily, starting now.** Health Connect only serves the
   30 days before a reader was first granted permission. History you do not
   export is not archived somewhere for later — it is gone.
""")
    st.warning(
        "The Watch 2/2R stores health data locally. There is no cloud backup, so a "
        "factory reset erases it. The daily export is your only archive.", icon="⚠")

    st.markdown("### Accepted columns")
    st.caption("Column names are matched loosely; these are examples, not requirements. "
               "Only a sleep start and a sleep end are mandatory.")
    st.code("""night / date          bedtime / sleep_start / start_time
waketime / sleep_end  tst_min / total_sleep / minutes_asleep
tib_min / time_in_bed deep_min, light_min, rem_min, awake_min
rhr, hrv_rmssd, spo2, resp_rate, steps, active_kcal
subjective            (your own 1-5 morning rating)""", language="text")

    st.markdown("### Add your own morning rating")
    st.caption(
        "The most valuable column is the one the watch cannot record. A daily 1–5 on "
        "how you actually felt sidesteps the accuracy problem in wearable sleep "
        "staging entirely, because it is ground truth rather than an estimate. Add a "
        "`subjective` column and it becomes an available prediction target.")

    st.markdown("---")
    with st.expander("Download the processed dataset"):
        st.download_button("Download features CSV",
                           f.to_csv(index=False).encode(),
                           "sleepcare_features.csv", "text/csv")
        st.dataframe(f.tail(30), width="stretch")
