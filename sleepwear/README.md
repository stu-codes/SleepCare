# SleepCare

Personal sleep analytics for a single person's wearable data. Built around
the OnePlus Watch 2R and Android Health Connect, but it will read any nightly
sleep export that has a start time and an end time.

```bash
pip install -r requirements.txt
streamlit run app.py
```

It opens on simulated data, so you can see the whole thing working before you
have collected a single night of your own.

---

## The part that actually blocks you

OHealth has **no export function**. There is no CSV, no JSON, no API. The only
route to a file is Health Connect:

1. **OHealth → Profile → Health Connect → Connect.** Authorise sleep, heart
   rate and blood oxygen separately; granting one does not grant the others.
2. Install a Health Connect reader that writes CSV. **Health Data Export** is
   open source and does this well.
3. **Check what actually arrived.** Open Health Connect → Data and access →
   Sleep and look at an OHealth entry. If you see one duration per night rather
   than separate deep, light, REM and awake segments, your stages are not
   crossing the bridge.
4. **Schedule the export daily, starting today.**

That last step is not housekeeping. Health Connect only serves the 30 days
before a reader app was first granted permission; older records return an error
unless the app holds `PERMISSION_READ_HEALTH_DATA_HISTORY`. The Watch 2/2R also
stores health data locally with no cloud backup, so a factory reset erases it.
History you do not export is not sitting somewhere waiting for you.

### If stages don't sync

They probably won't. The available developer evidence suggests OHealth writes a
duration-only sleep session to Health Connect, and the app is built for that
case: it detects the absence, disables deep and REM features, and redistributes
the score weights instead of quietly scoring zeros.

You lose less than it sounds. Wrist wearables agree with clinical
polysomnography on sleep-versus-wake most of the time but only moderately on
staging — published multi-device validations report Cohen's kappa in the range
of roughly 0.2 to 0.5. Deep sleep in particular is the number these devices are
worst at. Meanwhile the timing features that survive the bridge — regularity,
midpoint, social jetlag — are measured rather than inferred, and regularity is
the metric with the strongest outcome evidence attached to it.

---

## What it does

**Ingest** — Maps arbitrary column names onto one schema, repairs sessions that
cross midnight without a date, converts hours and seconds to minutes, collapses
duplicate nights, and drops implausible sessions. When a reported duration
contradicts the timestamps, the clock wins.

**Features** — Around 60 predictors, all computed causally so that night *N*
uses only what was knowable on the morning of night *N*: sleep midpoint,
Sleep Regularity Index, interdaily stability, intradaily variability, social
jetlag, rolling means and standard deviations, sleep debt, lags, and overnight
physiology where available.

**Score** — A transparent 0–100 composite, since the OnePlus sleep score does
not appear to cross the Health Connect boundary and Health Connect has no field
for it. Every component and weight is visible and adjustable.

**Model** — XGBoost predicting the next night, deliberately shallow and heavily
regularised, evaluated by expanding-window walk-forward validation. Random
k-fold is not offered, because shuffling a time series lets the model train on
next Tuesday to predict last Monday.

**Forecast** — Method selected by rolling backtest. Prophet is withheld until
about 180 nights exist, because on short personal series it fits noise and puts
confident intervals around it.

**Recommendation** — A bedtime window anchored on regularity and chronotype
alignment. When a validated model exists it sweeps candidate bedtimes
counterfactually; when the model loses to the naive baseline, the app says so
and falls back to your own best nights.

---

## The result you should expect

On the demo data the model beats the baselines by a wide margin. That is
because the simulation has known structure in it. **On your real data it may
well lose to "tonight will resemble last night," and if it does, the app will
tell you so rather than shipping the fancier model quietly.**

That is not a failure. Persistence is a genuinely strong forecast for one
person's sleep, and a gradient-boosted ensemble that cannot beat it is an
expensive way to be wrong. Every model is reported against persistence and a
7-night rolling mean, and the skill score is the number that matters.

Roughly: under 45 nights, use the descriptive views only. From 45 to 120,
treat any model as provisional. Past 120, walk-forward validation starts to
mean something.

---

## Add the column your watch cannot record

A daily 1–5 rating of how you actually felt is the most valuable data in the
whole pipeline. It sidesteps the wearable staging accuracy problem entirely,
because it is ground truth rather than an estimate. Add a `subjective` column
and it becomes an available prediction target.

---

## Cold start

If you want to develop against real data before your own accumulates, the
Kaggle *FitBit Fitness Tracker Data* set is CC0 and loads directly. *LifeSnaps*
is longer and richer. *MMASH* on PhysioNet adds beat-to-beat heart rate with
sleep quality and chronotype. SHHS and MESA via the NSRR are the gold standard
for stage labels but require an account and an approved data request, so don't
put them on your critical path.

---

## Tests

```bash
PYTHONPATH=. pytest tests/ -q
```

The suite leans toward bugs that don't throw: bedtimes averaged across
midnight, rolling windows that peek forward, hours parsed as minutes. Those
produce a confident wrong answer, which is the only kind that really matters
in a tool meant to give advice.

---

## Limitations

This is a tool for noticing patterns in your own data. It is not a medical
device and it cannot detect sleep apnea, insomnia, restless legs or any other
disorder. An "anomaly" here means a night that was statistically unlike your
others — usually a late flight, a drink, a cold, or a child.

Persistent poor sleep, loud snoring, gasping at night, or daytime sleepiness
are worth raising with a doctor, and no amount of feature engineering on wrist
data substitutes for that.
