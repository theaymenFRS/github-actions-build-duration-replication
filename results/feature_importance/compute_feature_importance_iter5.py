"""
Compute feature importance for RQ1-style Iter5 (train folds 1..9, test fold 10).

Inputs:
  - Processed per-project datasets (kept in time order) from:
      results/rq2/concept_drift_old_vs_recent_DEFAULTPARAMS/<projet>/*_processed.csv
  - Best hyperparameters from existing RQ1 GA logs (Iter5) under:
      results/rq1/<modèle>/

Outputs (under results/feature_importance/out_iter5/):
  - CSVs (built-in and permutation importance) per (project, model)
  - PNG barplots (top-15 permutation importance) per (project, model)
  - Aggregate CSV across projects per model
"""

from __future__ import annotations

import ast
import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None  # type: ignore

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
PROCESSED_ROOT = RESULTS / "rq2" / "concept_drift_old_vs_recent_DEFAULTPARAMS"
OUT_DIR = Path(__file__).resolve().parent / "out_iter5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS = 10
ITER5_K = 4  # Iter5: train folds 1..9 -> test fold 10
TOPK = 30
PERM_REPEATS = 3
RANDOM_SEED = 42
PERM_CANDIDATES = 20  # compute permutation only on top-k builtin features when available

ModelKey = Literal["RF", "DT", "GBR", "LGBM", "XGB"]

RQ1_MODEL_DIR: dict[ModelKey, str] = {
    "RF": "random_forest",
    "DT": "decision_tree",
    "GBR": "gradient_boosting",
    "LGBM": "lightgbm",
    "XGB": "xgboost",
}

_MODEL_KEYS: tuple[ModelKey, ...] = ("RF", "DT", "GBR", "LGBM", "XGB")


def rq1_log_path(project: str, model_key: ModelKey) -> Path:
    m = RQ1_MODEL_DIR[model_key]
    return RESULTS / "rq1" / m / f"{m}_{project}_RQ1.log"


@dataclass(frozen=True)
class ProjectSpec:
    key: str
    processed_csv: Path
    # log paths per model
    logs: dict[ModelKey, Path]


def _project_spec(key: str, data_dir: str, csv_name: str, log_slug: str) -> ProjectSpec:
    return ProjectSpec(
        key=key,
        processed_csv=PROCESSED_ROOT / data_dir / csv_name,
        logs={mk: rq1_log_path(log_slug, mk) for mk in _MODEL_KEYS},
    )


PROJECTS: list[ProjectSpec] = [
    _project_spec("radare2", "radare2", "radare2_processed.csv", "radare2"),
    _project_spec("daos", "daos", "daos_processed.csv", "daos"),
    _project_spec("ouds-android", "orange", "orange_processed.csv", "orange"),
    _project_spec("bmad", "bmad", "bmad_processed.csv", "bmad"),
    _project_spec("ccpay", "ccpay", "ccpay_processed.csv", "ccpay"),
    _project_spec("filterlists", "filterlists", "filterlists_processed.csv", "filterlists"),
    _project_spec("jod", "jod", "jod_processed.csv", "jod"),
    _project_spec("m2os", "m2os", "m2os_processed.csv", "m2os"),
    _project_spec("bruce", "bruce", "bruce_processed.csv", "bruce"),
    _project_spec("rustlang", "rustlang", "rustlang_processed.csv", "rustlang"),
]


def make_folds(n: int, n_folds: int = N_FOLDS) -> list[np.ndarray]:
    idx = np.arange(n)
    folds = np.array_split(idx, n_folds)
    if any(len(f) == 0 for f in folds):
        raise ValueError("Empty fold encountered")
    return [np.array(f, dtype=int) for f in folds]


def expanding_train_test_indices(folds: list[np.ndarray], k: int) -> tuple[np.ndarray, np.ndarray]:
    train_idx = np.concatenate([folds[i] for i in range(0, 5 + k)])
    test_idx = folds[5 + k]
    return train_idx, test_idx


def winsorize_train(y_tr: pd.Series) -> pd.Series:
    p1, p99 = y_tr.quantile([0.01, 0.99])
    return y_tr.clip(lower=float(p1), upper=float(p99))


def parse_best_params_iter5(log_path: Path) -> dict[str, Any]:
    txt = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    target_block = "[GA DONE][Iter5]"
    best_line_idx = None
    for i, line in enumerate(txt):
        if target_block in line:
            # expect "Best params:" within next ~10 lines
            for j in range(i, min(i + 12, len(txt))):
                if txt[j].startswith("Best params:"):
                    best_line_idx = j
                    break
    if best_line_idx is None:
        raise ValueError(f"Cannot find Iter5 best params in {log_path}")
    line = txt[best_line_idx]
    # Some logs contain invisible characters; be permissive.
    if "Best params:" not in line:
        raise ValueError(f"Malformed Best params line in {log_path}: {line}")
    payload = line.split("Best params:", 1)[1].strip()
    if not (payload.startswith("{") and payload.endswith("}")):
        # fallback: capture dict-like substring
        m = re.search(r"(\\{.*\\})", payload)
        if not m:
            raise ValueError(f"Malformed Best params line in {log_path}: {line}")
        payload = m.group(1)
    return ast.literal_eval(payload)


class LogTargetRegressor:
    """Wraps a regressor trained on log1p(y) to predict on original scale."""

    def __init__(self, base):
        self.base = base

    def fit(self, X, y):
        self.base.fit(X, np.log1p(y))
        return self

    def predict(self, X):
        pred = np.expm1(self.base.predict(X))
        return np.maximum(0.0, pred)

    @property
    def feature_importances_(self):
        return getattr(self.base, "feature_importances_", None)


def build_model(model_key: ModelKey, params: dict[str, Any]):
    if model_key == "RF":
        return RandomForestRegressor(**params, random_state=RANDOM_SEED, n_jobs=-1)
    if model_key == "DT":
        return DecisionTreeRegressor(**params, random_state=RANDOM_SEED)
    if model_key == "GBR":
        return GradientBoostingRegressor(**params, random_state=RANDOM_SEED)
    if model_key == "LGBM":
        if LGBMRegressor is None:
            raise RuntimeError("lightgbm is not installed")
        base = LGBMRegressor(**params, random_state=RANDOM_SEED, n_jobs=-1)
        return LogTargetRegressor(base)
    if model_key == "XGB":
        if XGBRegressor is None:
            raise RuntimeError("xgboost is not installed")
        base = XGBRegressor(
            **params,
            objective="reg:squarederror",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=0,
            tree_method="hist",
        )
        return LogTargetRegressor(base)
    raise ValueError(model_key)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def compute_importances_for_project_model(spec: ProjectSpec, model_key: ModelKey) -> pd.DataFrame:
    df = pd.read_csv(spec.processed_csv, low_memory=False)
    if "build_duration" not in df.columns:
        raise ValueError(f"Missing build_duration in {spec.processed_csv}")
    y = df["build_duration"].astype(float)
    X = df.drop(columns=["build_duration"])

    # Ensure numeric where possible
    Xn = X.apply(pd.to_numeric, errors="coerce")
    # The processed files should already be numeric; keep 0 for leftover NaNs
    Xn = Xn.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    folds = make_folds(len(Xn), N_FOLDS)
    train_idx, test_idx = expanding_train_test_indices(folds, ITER5_K)

    X_tr_df = Xn.iloc[train_idx]
    y_tr = winsorize_train(y.iloc[train_idx]).to_numpy()
    X_te_df = Xn.iloc[test_idx]
    y_te = y.iloc[test_idx].to_numpy()

    params = parse_best_params_iter5(spec.logs[model_key])
    model = build_model(model_key, params)
    model.fit(X_tr_df, y_tr)
    pred = model.predict(X_te_df)
    base_rmse = rmse(y_te, pred)

    builtin = getattr(model, "feature_importances_", None)
    if builtin is None:
        builtin_s = pd.Series(index=Xn.columns, data=np.nan, dtype=float)
        perm_features = list(Xn.columns)
    else:
        builtin_s = pd.Series(index=Xn.columns, data=np.asarray(builtin, dtype=float))
        perm_features = (
            builtin_s.sort_values(ascending=False).head(PERM_CANDIDATES).index.tolist()
        )

    # permutation importance on test fold (manual; estimator expects full feature set)
    rng = np.random.default_rng(RANDOM_SEED)
    perm_deltas: dict[str, list[float]] = {f: [] for f in perm_features}
    for f in perm_features:
        for _ in range(PERM_REPEATS):
            Xp = X_te_df.copy()
            Xp[f] = rng.permutation(Xp[f].to_numpy())
            rmse_p = rmse(y_te, model.predict(Xp))
            perm_deltas[f].append(rmse_p - base_rmse)
    perm_s_mean = pd.Series({k: float(np.mean(v)) for k, v in perm_deltas.items()}, dtype=float)
    perm_s_std = pd.Series({k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in perm_deltas.items()}, dtype=float)

    out = pd.DataFrame({"feature": Xn.columns})
    out["project"] = spec.key
    out["model"] = model_key
    out["builtin_importance"] = out["feature"].map(builtin_s).astype(float)
    out["perm_importance_mean"] = out["feature"].map(perm_s_mean).astype(float)
    out["perm_importance_std"] = out["feature"].map(perm_s_std).astype(float)

    out["iter5_test_rmse"] = base_rmse
    return out


def save_topk_plot(df_imp: pd.DataFrame, out_png: Path, title: str) -> None:
    d = (
        df_imp.dropna(subset=["perm_importance_mean"])
        .sort_values("perm_importance_mean", ascending=False)
        .head(15)
        .iloc[::-1]
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(d["feature"], d["perm_importance_mean"], xerr=d["perm_importance_std"], color="#2c7fb8", alpha=0.9)
    ax.set_xlabel("Permutation importance (Δ RMSE, test fold 10)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--projects",
        nargs="*",
        default=["all"],
        help="Project keys to run (e.g. radare2 daos). Default: all",
    )
    ap.add_argument(
        "--models",
        nargs="*",
        default=["RF"],
        help="Models to run (RF, DT, GBR, LGBM, XGB). Default: RF only",
    )
    args = ap.parse_args()

    wanted_projects = set(args.projects)
    wanted_models = [m.upper() for m in args.models]
    if "ALL" in wanted_projects or "all" in wanted_projects:
        specs = PROJECTS
    else:
        specs = [p for p in PROJECTS if p.key in wanted_projects]

    all_rows: list[pd.DataFrame] = []
    for spec in specs:
        for model_key in wanted_models:
            imp = compute_importances_for_project_model(spec, model_key)  # full features
            all_rows.append(imp)

            # save CSVs
            csv_path = OUT_DIR / f"iter5_{spec.key}_{model_key}_feature_importance.csv"
            imp.sort_values("perm_importance_mean", ascending=False).to_csv(csv_path, index=False)

            # plot top permutation
            png_path = OUT_DIR / f"iter5_{spec.key}_{model_key}_perm_top15.png"
            title = f"{spec.key} — {model_key} — Iter5 feature importance (permutation)"
            save_topk_plot(imp, png_path, title)

    if not all_rows:
        raise SystemExit("No (project, model) selected.")

    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(OUT_DIR / "iter5_feature_importance_long.csv", index=False)

    # aggregate across projects: mean permutation importance per model/feature (rank stability)
    agg = (
        full.groupby(["model", "feature"], as_index=False)[["perm_importance_mean", "perm_importance_std"]]
        .mean(numeric_only=True)
        .sort_values(["model", "perm_importance_mean"], ascending=[True, False])
    )
    agg.to_csv(OUT_DIR / "iter5_aggregate_mean_perm_importance_by_model.csv", index=False)

    print("Done. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()

