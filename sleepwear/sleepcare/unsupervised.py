"""Pattern discovery without labels.

Clustering answers "what kinds of nights do I have?", which is a more
useful question than "was last night good?" because the answer names
recurring situations you can recognise and act on.

Isolation Forest answers "which nights were unlike me?" — and the honest
framing matters: an anomaly is a night that departs from your own pattern.
It is not a diagnosis, and this tool cannot detect sleep apnea, insomnia
or any other condition.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# A compact, interpretable basis. Clustering on fifty correlated features
# produces clusters no one can describe.
CLUSTER_BASIS = ["tst_min", "efficiency", "mid_dec", "bed_dec", "mid_shift_h"]
OPTIONAL_BASIS = ["deep_pct", "rem_pct", "rhr", "hrv_rmssd", "waso_min"]


@dataclass
class ClusterResult:
    labels: np.ndarray
    k: int
    silhouette: float
    profile: pd.DataFrame
    names: dict[int, str]
    coords: np.ndarray
    basis: list[str]


def _basis(f: pd.DataFrame) -> list[str]:
    cols = [c for c in CLUSTER_BASIS if c in f.columns]
    cols += [c for c in OPTIONAL_BASIS if c in f.columns and f[c].notna().mean() > 0.6]
    return cols


def _name_cluster(row: pd.Series, med: pd.Series) -> str:
    """Describe a cluster in the language a person would use about a night."""
    short = row["tst_min"] < med["tst_min"] - 25
    long_ = row["tst_min"] > med["tst_min"] + 25
    late = row["mid_dec"] > med["mid_dec"] + 0.5
    early = row["mid_dec"] < med["mid_dec"] - 0.5
    poor = row["efficiency"] < med["efficiency"] - 0.02
    erratic = row.get("mid_shift_h", 0) > med.get("mid_shift_h", 0) + 0.55

    if short and late:
        return "Short and late"
    if late and not short:
        return "Late but long enough"
    if short and poor:
        return "Short and broken"
    if long_ and early:
        return "Early and long"
    if poor:
        return "Restless"
    if erratic:
        return "Shifted timing"
    if long_:
        return "Catch-up nights"
    if early:
        return "Early nights"
    return "Typical nights"


def find_archetypes(f: pd.DataFrame, k_range=(2, 6), k: int | None = None) -> ClusterResult | None:
    """K-means over an interpretable basis, k chosen by silhouette."""
    cols = _basis(f)
    if len(cols) < 3 or len(f) < 20:
        return None

    X = f[cols].copy()
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    Xs = pipe.fit_transform(X)

    if k is None:
        best, best_s = 3, -1.0
        for kk in range(k_range[0], min(k_range[1], len(f) // 8) + 1):
            if kk < 2:
                continue
            lab = KMeans(n_clusters=kk, n_init=10, random_state=42).fit_predict(Xs)
            if len(set(lab)) < 2:
                continue
            s = silhouette_score(Xs, lab)
            if s > best_s:
                best, best_s = kk, s
        k = best

    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(Xs)
    sil = silhouette_score(Xs, labels) if len(set(labels)) > 1 else float("nan")

    prof = f[cols].groupby(labels).median()
    prof["nights"] = pd.Series(labels).value_counts().sort_index().values
    prof["share_%"] = (prof["nights"] / len(labels) * 100).round(1)

    med = f[cols].median()
    names = {int(i): _name_cluster(prof.loc[i], med) for i in prof.index}
    # Disambiguate collisions so two clusters never share a label.
    seen: dict[str, int] = {}
    for i in sorted(names):
        base = names[i]
        if base in seen:
            seen[base] += 1
            names[i] = f"{base} ({seen[base]})"
        else:
            seen[base] = 1

    coords = PCA(n_components=2, random_state=42).fit_transform(Xs)
    return ClusterResult(labels, k, float(sil), prof, names, coords, cols)


@dataclass
class AnomalyResult:
    flags: np.ndarray
    scores: np.ndarray
    basis: list[str]
    contributions: pd.DataFrame


def find_anomalies(f: pd.DataFrame, contamination: float = 0.06) -> AnomalyResult | None:
    """Isolation Forest over the same basis, with per-night explanations."""
    cols = _basis(f)
    if len(cols) < 3 or len(f) < 25:
        return None

    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(imp.fit_transform(f[cols]))

    iso = IsolationForest(
        n_estimators=250, contamination=contamination, random_state=42, n_jobs=-1
    )
    flags = iso.fit_predict(Xs) == -1
    # Higher means more anomalous, which is the intuitive direction.
    scores = -iso.score_samples(Xs)

    # Which feature was most extreme relative to the person's own norm.
    contrib = pd.DataFrame(np.abs(Xs), columns=cols, index=f.index)
    return AnomalyResult(flags, scores, cols, contrib)


def explain_anomaly(anom: AnomalyResult, f: pd.DataFrame, i: int, top: int = 2) -> str:
    """Plain-language reason a night was flagged."""
    row = anom.contributions.iloc[i].sort_values(ascending=False)
    labels = {
        "tst_min": "sleep duration", "efficiency": "sleep efficiency",
        "mid_dec": "sleep timing", "bed_dec": "bedtime",
        "mid_shift_h": "shift from the night before", "deep_pct": "deep sleep share",
        "rem_pct": "REM share", "rhr": "resting heart rate",
        "hrv_rmssd": "heart rate variability", "waso_min": "time awake in bed",
    }
    parts = []
    for c in row.index[:top]:
        val = f[c].iloc[i]
        med = f[c].median()
        direction = "well above" if val > med else "well below"
        parts.append(f"{labels.get(c, c)} {direction} your usual")
    return "; ".join(parts)
