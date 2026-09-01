"""Next-night prediction, evaluated the only way that is honest here.

Two rules govern this module.

Random k-fold cross-validation is banned. Shuffling a time series lets the
model train on next Tuesday to predict last Monday, which inflates every
metric and teaches you nothing about tomorrow. Everything here is
walk-forward: train on the past, test on the next block, roll forward.

Baselines are not optional. On a single person's sleep data, "tonight will
resemble last night" is a genuinely strong forecast, and a gradient-boosted
ensemble that cannot beat it is an expensive way to be wrong. Every model
is reported against persistence and a 7-night rolling mean, and the skill
score is the number that matters.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:  # pragma: no cover
    from sklearn.ensemble import GradientBoostingRegressor
    HAS_XGB = False


# Deliberately shallow and heavily regularised. With a few hundred nights
# from one person, model capacity is the enemy, not the goal.
SMALL_N_PARAMS = dict(
    n_estimators=320,
    max_depth=3,
    learning_rate=0.035,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=6,
    reg_lambda=2.5,
    reg_alpha=0.4,
    random_state=42,
)


@dataclass
class FoldResult:
    fold: int
    train_end: pd.Timestamp
    n_train: int
    n_test: int
    y_true: np.ndarray
    y_pred: np.ndarray
    y_persist: np.ndarray
    y_rolling: np.ndarray
    dates: pd.DatetimeIndex


@dataclass
class EvalResult:
    target: str
    folds: list[FoldResult] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    importances: pd.Series | None = None
    model: object | None = None
    n_used: int = 0

    def _cat(self, attr: str) -> np.ndarray:
        return np.concatenate([getattr(f, attr) for f in self.folds]) if self.folds else np.array([])

    @property
    def predictions(self) -> pd.DataFrame:
        if not self.folds:
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "actual": self._cat("y_true"),
                "predicted": self._cat("y_pred"),
                "persistence": self._cat("y_persist"),
                "rolling7": self._cat("y_rolling"),
            },
            index=pd.DatetimeIndex(np.concatenate([f.dates.values for f in self.folds])),
        )

    def metrics(self) -> pd.DataFrame:
        p = self.predictions
        if p.empty:
            return pd.DataFrame()
        rows = []
        for name in ("predicted", "persistence", "rolling7"):
            yt, yp = p["actual"].values, p[name].values
            ok = np.isfinite(yt) & np.isfinite(yp)
            if ok.sum() < 3:
                continue
            rows.append({
                "model": {"predicted": "XGBoost", "persistence": "Persistence (last night)",
                          "rolling7": "7-night rolling mean"}[name],
                "MAE": mean_absolute_error(yt[ok], yp[ok]),
                "RMSE": float(np.sqrt(mean_squared_error(yt[ok], yp[ok]))),
                "R2": r2_score(yt[ok], yp[ok]),
            })
        m = pd.DataFrame(rows)
        best_base = m.loc[m["model"] != "XGBoost", "MAE"].min()
        m["Skill vs best baseline"] = 1 - m["MAE"] / best_base
        return m

    @property
    def beats_baseline(self) -> bool:
        m = self.metrics()
        if m.empty or "XGBoost" not in set(m["model"]):
            return False
        mae = m.set_index("model")["MAE"]
        return bool(mae["XGBoost"] < mae.drop("XGBoost").min())

    def verdict(self) -> str:
        if not self.folds:
            return "Not enough data to validate. Keep collecting nights."
        m = self.metrics().set_index("model")["MAE"]
        gain = (m.drop("XGBoost").min() - m["XGBoost"]) / m.drop("XGBoost").min() * 100
        if gain > 5:
            return (f"The model beats the best naive baseline by {gain:.1f}% on MAE. "
                    "Worth using.")
        if gain > 0:
            return (f"The model edges out the baseline by only {gain:.1f}%. "
                    "That margin is thin enough to be noise. Prefer the simple baseline "
                    "until more nights accumulate.")
        return ("The model does not beat 'tonight will resemble last night'. "
                "Use the baseline. This is a normal and honest result on small "
                "personal datasets, not a bug.")


def _make_model():
    if HAS_XGB:
        return XGBRegressor(objective="reg:squarederror", **SMALL_N_PARAMS)
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(
        n_estimators=SMALL_N_PARAMS["n_estimators"],
        max_depth=SMALL_N_PARAMS["max_depth"],
        learning_rate=SMALL_N_PARAMS["learning_rate"],
        subsample=SMALL_N_PARAMS["subsample"],
        random_state=42,
    )


def walk_forward(
    f: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    n_splits: int = 5,
    min_train: int = 45,
    params: dict | None = None,
) -> EvalResult:
    """Expanding-window validation.

    Fold k trains on everything up to a cut point and tests on the block
    immediately after it. No future information reaches any training set.
    """
    data = f[["night"] + feature_cols + [target]].copy()

    # The naive comparators, built from the same information the model gets.
    base_col = {
        "y_score_next": "sleep_score",
        "y_tst_next": "tst_min",
        "y_eff_next": "efficiency",
        "y_subjective_next": "subjective",
    }.get(target, "tst_min")
    if base_col not in f.columns:
        base_col = "tst_min"
    data["_persist"] = f[base_col].values
    data["_rolling"] = (
        pd.Series(f[base_col].values, index=pd.DatetimeIndex(f["night"]))
        .rolling("7D", min_periods=2).mean().values
    )

    data = data.dropna(subset=[target, "_persist"])
    data = data[np.isfinite(data[target])].reset_index(drop=True)

    res = EvalResult(target=target, feature_names=feature_cols, n_used=len(data))
    if len(data) < min_train + 12:
        return res

    n = len(data)
    test_size = max(7, (n - min_train) // n_splits)
    cuts = list(range(min_train, n - test_size + 1, test_size))[:n_splits]
    if not cuts:
        return res

    X_all = data[feature_cols]
    y_all = data[target]

    for i, cut in enumerate(cuts):
        te = slice(cut, min(cut + test_size, n))
        X_tr, y_tr = X_all.iloc[:cut], y_all.iloc[:cut]
        X_te, y_te = X_all.iloc[te], y_all.iloc[te]
        if len(X_te) < 3:
            continue

        model = _make_model()
        model.set_params(**(params or {}))
        # XGBoost handles NaN natively; the sklearn fallback does not.
        if not HAS_XGB:
            med = X_tr.median()
            X_tr, X_te = X_tr.fillna(med), X_te.fillna(med)
        model.fit(X_tr, y_tr)

        res.folds.append(FoldResult(
            fold=i,
            train_end=data["night"].iloc[cut - 1],
            n_train=cut,
            n_test=len(X_te),
            y_true=y_te.values,
            y_pred=model.predict(X_te),
            y_persist=data["_persist"].iloc[te].values,
            y_rolling=data["_rolling"].iloc[te].fillna(data["_persist"].iloc[te]).values,
            dates=pd.DatetimeIndex(data["night"].iloc[te]),
        ))

    # Final model on all available history, for tonight's actual prediction.
    final = _make_model()
    final.set_params(**(params or {}))
    Xf = X_all if HAS_XGB else X_all.fillna(X_all.median())
    final.fit(Xf, y_all)
    res.model = final

    if hasattr(final, "feature_importances_"):
        res.importances = (
            pd.Series(final.feature_importances_, index=feature_cols)
            .sort_values(ascending=False)
        )
    return res


def predict_next(res: EvalResult, f: pd.DataFrame, feature_cols: list[str]) -> float | None:
    """Prediction for the night after the most recent complete record."""
    if res.model is None or f.empty:
        return None
    row = f[feature_cols].iloc[[-1]]
    if not HAS_XGB:
        row = row.fillna(f[feature_cols].median())
    try:
        return float(res.model.predict(row)[0])
    except Exception:
        return None
