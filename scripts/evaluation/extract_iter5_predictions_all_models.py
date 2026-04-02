#!/usr/bin/env python3
"""
Extraire les prédictions build-par-build pour tous les modèles RQ1 (itération 5)
ainsi que les deux baselines :
- baseline lag-1
- baseline moyenne glissante sur les 7 dernières durées

Le script reconstruit le scénario principal de la RQ1 :
train folds 1..9 -> test fold 10.

Modèles supportés :
- decision_tree
- random_forest
- gradient_boosting
- xgboost
- lightgbm

Sorties :
- un CSV par projet et par modèle dans results/qualitative_cases/
- un CSV global agrégé : results/qualitative_cases/iter5_all_models_all_projects.csv
- un CSV d'exemples candidats : results/qualitative_cases/iter5_candidate_examples.csv

Utilisation :
    py scripts/evaluation/extract_iter5_predictions_all_models.py
    py scripts/evaluation/extract_iter5_predictions_all_models.py --model random_forest
    py scripts/evaluation/extract_iter5_predictions_all_models.py --project filterlists
"""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

MAX_DURATION_SEC = 300 * 24 * 60 * 60
N_FOLDS = 10
ITERATION_INDEX = 4  # Iter5 => k=4

PROJECT_TO_CSV = {
    "daos": "daos_wf9020028_fixed.csv",
    "rustlang": "rustlang_wf51073_fixed.csv",
    "orange": "Orange_OpenSourceouds_android_wf108176393_fixed.csv",
    "bmad": "bmad simbmad ecosystem_wf69576399_fixed.csv",
    "ccpay": "ccpay_wf6192976_fixed.csv",
    "filterlists": "collinbarrettFilterLists_wf75763098_fixed.csv",
    "jod": "jod-yksilo-ui_wf83806327_fixed.csv",
    "m2os": "m2Gilesm2os_wf105026558_fixed.csv",
    "bruce": "pr3y_Bruce_wf121541665_fixed.csv",
    "radare2": "radareorg_radare2_wf1989843_fixed.csv",
}

MODEL_SPECS = {
    "decision_tree": {
        "label": "DT",
        "uses_log_target": False,
    },
    "random_forest": {
        "label": "RF",
        "uses_log_target": False,
    },
    "gradient_boosting": {
        "label": "GBR",
        "uses_log_target": False,
    },
    "xgboost": {
        "label": "XGB",
        "uses_log_target": True,
    },
    "lightgbm": {
        "label": "LGBM",
        "uses_log_target": True,
    },
}


def rq1_model_log_path(repo_root: Path, project: str, model_name: str) -> Path:
    """Chemin vers le journal RQ1 aligné sur scripts/modeling/rq1/*.py."""
    return repo_root / "results" / "rq1" / model_name / f"{model_name}_{project}_RQ1.log"

DROP_COLUMNS = [
    "repo", "id_build", "commit_sha", "conclusion", "workflow_name",
    "created_at", "updated_at", "gh_job_id", "fetch_duration",
    "gh_pull_req_number", "tests_passed", "status", "tests_failed",
    "tests_skipped", "tests_total", "gh_first_commit_created_at",
    "job_", "git_merged_with", "job_details", "build_language",
    "test_framework", "languages", "duration_from_ts",
    "build_duration_original", "run_attempt_from_api", "fixed_with_api",
]


class FileTypesBinarizer:
    def __init__(self, sep: str = ","):
        self.sep = sep
        self.classes_: List[str] = []

    def fit_transform(self, values: pd.Series) -> pd.DataFrame:
        classes = set()
        s = values.fillna("").astype(str)
        for val in s:
            for tok in val.split(self.sep):
                tok = tok.strip()
                if tok:
                    classes.add(tok)
        self.classes_ = sorted(classes)

        rows = []
        for val in s:
            toks = {tok.strip() for tok in val.split(self.sep) if tok.strip()}
            rows.append([1 if c in toks else 0 for c in self.classes_])

        cols = [f"ft__{c}" for c in self.classes_]
        return pd.DataFrame(rows, columns=cols, index=values.index)


def make_unique_columns(cols: List[str]) -> List[str]:
    seen = {}
    out = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__dup{seen[c]}")
    return out


def add_lag_features_per_workflow(df: pd.DataFrame) -> pd.DataFrame:
    chunks = []
    for wf_id in df["workflow_id"].dropna().unique():
        w = df[df["workflow_id"] == wf_id].copy()
        w = w.sort_values("created_at")
        w["secs_since_prev"] = (w["created_at"] - w["created_at"].shift(1)).dt.total_seconds()
        for lag in range(1, 8):
            w[f"duration_lag_{lag}"] = w["build_duration"].shift(lag)
        w["window_avg_7"] = w[[f"duration_lag_{i}" for i in range(1, 8)]].mean(axis=1)
        w["window_std_7"] = w[[f"duration_lag_{i}" for i in range(1, 8)]].std(axis=1)
        w["window_avg_3"] = w[[f"duration_lag_{i}" for i in range(1, 4)]].mean(axis=1)
        w["window_std_3"] = w[[f"duration_lag_{i}" for i in range(1, 4)]].std(axis=1)
        w["window_avg_4"] = w[[f"duration_lag_{i}" for i in range(1, 5)]].mean(axis=1)
        w["window_std_4"] = w[[f"duration_lag_{i}" for i in range(1, 5)]].std(axis=1)
        chunks.append(w)
    return pd.concat(chunks, ignore_index=True)


def preprocess_with_metadata(csv_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)

    if "conclusion" in df.columns:
        df = df[df["conclusion"] == "success"].copy()
    if "workflow_event_trigger" in df.columns:
        df = df[df["workflow_event_trigger"].isin(["push", "pull_request"])].copy()

    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True, errors="coerce")
    df["hour"] = df["created_at"].dt.hour
    df["dow"] = df["created_at"].dt.dayofweek
    df["month"] = df["created_at"].dt.month
    df["duration_from_ts"] = (df["updated_at"] - df["created_at"]).dt.total_seconds()

    df = df[df["build_duration"] <= MAX_DURATION_SEC].copy()
    df = df.sort_values("created_at").reset_index(drop=True)

    if "workflow_id" in df.columns:
        counts = df["workflow_id"].value_counts()
        valid_workflows = counts[counts >= 100].index.tolist()
        if valid_workflows:
            df = df[df["workflow_id"].isin(valid_workflows)].copy()
        df = add_lag_features_per_workflow(df)

    meta_cols = [
        "repo", "id_build", "workflow_id", "created_at", "updated_at",
        "build_duration", "workflow_event_trigger", "branch", "issuer_name",
        "gh_files_modified", "gh_lines_added", "gh_lines_deleted",
        "gh_src_churn", "gh_doc_files", "gh_src_files", "gh_other_files",
        "duration_lag_1", "window_avg_7", "secs_since_prev",
    ]
    meta_cols = [c for c in meta_cols if c in df.columns]
    meta_df = df[meta_cols].copy()

    model_df = df.copy()
    model_df.drop(columns=[c for c in DROP_COLUMNS if c in model_df.columns], inplace=True, errors="ignore")

    for col in ["workflow_event_trigger", "issuer_name"]:
        if col in model_df.columns:
            model_df[col] = model_df[col].astype("category").cat.codes

    if "file_types" in model_df.columns:
        ft = FileTypesBinarizer(sep=",")
        ft_df = ft.fit_transform(model_df["file_types"])
        model_df = pd.concat([model_df.drop(columns=["file_types"]), ft_df], axis=1)

    if "branch" in model_df.columns:
        b = model_df["branch"].astype(str)
        model_df["branch"] = np.select(
            [
                b.str.contains("fix", case=False, na=False),
                b.str.contains(r"\b(?:main|master)\b", case=False, na=False, regex=True),
            ],
            [0, 1],
            default=2,
        ).astype(np.int8)

    model_df.columns = [re.sub(r"[\[\]<>]", "_", str(c)) for c in model_df.columns]
    model_df.columns = make_unique_columns(list(model_df.columns))

    valid_mask = model_df.notna().all(axis=1)
    model_df = model_df.loc[valid_mask].copy().reset_index(drop=True)
    meta_df = meta_df.loc[valid_mask].copy().reset_index(drop=True)

    return model_df, meta_df


def make_folds(n: int, n_folds: int = N_FOLDS):
    if n < n_folds:
        return None
    idx = np.arange(n)
    folds = np.array_split(idx, n_folds)
    if any(len(f) == 0 for f in folds):
        return None
    return folds


def expanding_train_test_indices(folds, k: int):
    train_idx = np.concatenate([folds[i] for i in range(0, 5 + k)])
    test_idx = folds[5 + k]
    return train_idx, test_idx


def parse_model_log(log_path: Path) -> Tuple[List[str], Dict]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")

    m_feat = re.search(r"\[FEATURES\].*?\n(\[.*?\])\n", text, flags=re.S)
    if not m_feat:
        raise ValueError(f"Impossible de lire la liste des features dans {log_path}")
    kept_features = ast.literal_eval(m_feat.group(1))

    m_params = re.search(r"\[GA DONE\]\[Iter5\].*?Best params:\s*(\{.*?\})", text, flags=re.S)
    if not m_params:
        raise ValueError(f"Impossible de lire les meilleurs hyperparamètres Iter5 dans {log_path}")
    best_params = ast.literal_eval(m_params.group(1))
    return kept_features, best_params


def build_run_url(repo: str, run_id) -> str:
    return f"https://github.com/{repo}/actions/runs/{int(run_id)}"


def make_model(model_name: str, params: Dict):
    if model_name == "decision_tree":
        return DecisionTreeRegressor(**params, random_state=42)

    if model_name == "random_forest":
        return RandomForestRegressor(**params, random_state=42, n_jobs=-1)

    if model_name == "gradient_boosting":
        return GradientBoostingRegressor(**params, random_state=42)

    if model_name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            **params,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            tree_method="hist",
        )

    if model_name == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            **params,
            objective="regression",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    raise ValueError(f"Modèle non supporté : {model_name}")


def fit_predict_model(model_name: str, params: Dict, X_tr: pd.DataFrame, y_tr: pd.Series, X_te: pd.DataFrame) -> np.ndarray:
    p1, p99 = y_tr.quantile([0.01, 0.99])
    y_tr_clip = y_tr.clip(lower=p1, upper=p99)

    model = make_model(model_name, params)
    if MODEL_SPECS[model_name]["uses_log_target"]:
        model.fit(X_tr.to_numpy(), np.log1p(y_tr_clip).to_numpy())
        pred = np.expm1(model.predict(X_te.to_numpy()))
    else:
        model.fit(X_tr.to_numpy(), y_tr_clip.to_numpy())
        pred = model.predict(X_te.to_numpy())

    pred = np.maximum(0.0, pred)
    return pred


def run_project_model(repo_root: Path, project: str, model_name: str, output_dir: Path) -> pd.DataFrame:
    csv_path = repo_root / "data" / "raw" / PROJECT_TO_CSV[project]
    log_path = rq1_model_log_path(repo_root, project, model_name)

    model_df, meta_df = preprocess_with_metadata(csv_path)
    kept_features, best_params = parse_model_log(log_path)

    kept_features = [c for c in kept_features if c in model_df.columns]
    X = model_df.drop(columns=["build_duration"]).copy()
    y = model_df["build_duration"].copy()

    rows = []
    for wf_id in X["workflow_id"].unique():
        mask = X["workflow_id"] == wf_id
        Xw = X.loc[mask].reset_index(drop=True)
        yw = y.loc[mask].reset_index(drop=True)
        Mw = meta_df.loc[mask].reset_index(drop=True)

        folds = make_folds(len(Xw), N_FOLDS)
        if folds is None:
            continue

        train_idx, test_idx = expanding_train_test_indices(folds, ITERATION_INDEX)
        X_tr = Xw.iloc[train_idx][kept_features]
        y_tr = yw.iloc[train_idx]
        X_te = Xw.iloc[test_idx][kept_features]
        y_te = yw.iloc[test_idx]
        M_te = Mw.iloc[test_idx].copy()

        if len(X_tr) < 30 or len(X_te) < 1:
            continue

        pred = fit_predict_model(model_name, best_params, X_tr, y_tr, X_te)

        out = M_te.copy()
        out["project"] = project
        out["model"] = MODEL_SPECS[model_name]["label"]
        out["model_key"] = model_name
        out["actual_duration"] = y_te.to_numpy()
        out["prediction"] = pred
        out["baseline_lag1"] = X_te["duration_lag_1"].to_numpy() if "duration_lag_1" in X_te.columns else np.nan
        out["baseline_mean7"] = X_te["window_avg_7"].to_numpy() if "window_avg_7" in X_te.columns else np.nan
        out["abs_error_model"] = np.abs(out["actual_duration"] - out["prediction"])
        out["abs_error_lag1"] = np.abs(out["actual_duration"] - out["baseline_lag1"])
        out["abs_error_mean7"] = np.abs(out["actual_duration"] - out["baseline_mean7"])
        out["run_url"] = out.apply(lambda r: build_run_url(r["repo"], r["id_build"]), axis=1)
        rows.append(out)

    if not rows:
        raise RuntimeError(f"Aucune prédiction extraite pour le projet {project}, modèle {model_name}")

    result = pd.concat(rows, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"iter5_{model_name}_{project}_predictions.csv"
    result.to_csv(out_file, index=False, encoding="utf-8")
    print(f"[OK] {project} / {model_name}: {len(result)} builds exportés -> {out_file}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        choices=sorted(PROJECT_TO_CSV.keys()),
        default=None,
        help="Extraire un seul projet (par défaut : tous les projets).",
    )
    parser.add_argument(
        "--model",
        choices=["all"] + list(MODEL_SPECS.keys()),
        default="all",
        help="Modèle à extraire (par défaut : all).",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Racine du dépôt GitHub (par défaut : répertoire courant).",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_dir = repo_root / "results" / "qualitative_cases"

    projects = [args.project] if args.project else list(PROJECT_TO_CSV.keys())
    models = list(MODEL_SPECS.keys()) if args.model == "all" else [args.model]

    all_frames = []
    for model_name in models:
        for project in projects:
            all_frames.append(run_project_model(repo_root, project, model_name, output_dir))

    if all_frames:
        all_df = pd.concat(all_frames, ignore_index=True)
        global_file = output_dir / "iter5_all_models_all_projects.csv"
        all_df.to_csv(global_file, index=False, encoding="utf-8")
        print(f"[OK] Fichier global -> {global_file}")

        summary = []
        for (model_name, project), dfp in all_df.groupby(["model_key", "project"]):
            keep_cols = [
                "model", "model_key", "project", "repo", "id_build", "run_url",
                "actual_duration", "prediction", "baseline_lag1", "baseline_mean7",
                "abs_error_model", "abs_error_lag1", "abs_error_mean7",
            ]
            best_m = dfp.nsmallest(3, "abs_error_model")[keep_cols].assign(case_type="best_model")
            worst_m = dfp.nlargest(3, "abs_error_model")[keep_cols].assign(case_type="worst_model")
            summary.append(best_m)
            summary.append(worst_m)
        summary_df = pd.concat(summary, ignore_index=True)
        summary_file = output_dir / "iter5_candidate_examples.csv"
        summary_df.to_csv(summary_file, index=False, encoding="utf-8")
        print(f"[OK] Exemples candidats -> {summary_file}")


if __name__ == "__main__":
    main()
