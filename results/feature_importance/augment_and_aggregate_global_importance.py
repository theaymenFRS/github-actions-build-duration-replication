"""
Create weighted + normalized (ΔNRMSE) global feature importance per model.

We start from the per-(project, model, feature) file produced by:
  out_iter5/iter5_feature_importance_long.csv

Then we compute per-project test-fold (Iter5, fold 10) stats:
  - n_test
  - std_test (std of y on test fold; ddof=1)

Normalized permutation importance:
  ΔNRMSE = ΔRMSE / std_test

Global aggregation (per model, feature):
  - Unweighted mean across projects (baseline we already had)
  - Weighted mean across projects with weights = n_test

Outputs (out_iter5/):
  - global_perm_importance_all_models_unweighted.csv
  - global_perm_importance_all_models_weighted.csv
  - global_perm_importance_all_models_unweighted_nrmse.csv
  - global_perm_importance_all_models_weighted_nrmse.csv
  - Per-model CSVs + top20 PNGs for the 4 variants above
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out_iter5"
LONG = OUT / "iter5_feature_importance_long.csv"
RESULTS = REPO_ROOT / "results"
PROCESSED_ROOT = RESULTS / "rq2" / "concept_drift_old_vs_recent_DEFAULTPARAMS"

N_FOLDS = 10
ITER5_K = 4
TOP = 20


@dataclass(frozen=True)
class ProjectProcessed:
    key: str
    processed_csv: Path


PROJECTS: list[ProjectProcessed] = [
    ProjectProcessed("radare2", PROCESSED_ROOT / "radare2" / "radare2_processed.csv"),
    ProjectProcessed("daos", PROCESSED_ROOT / "daos" / "daos_processed.csv"),
    ProjectProcessed("ouds-android", PROCESSED_ROOT / "orange" / "orange_processed.csv"),
    ProjectProcessed("bmad", PROCESSED_ROOT / "bmad" / "bmad_processed.csv"),
    ProjectProcessed("ccpay", PROCESSED_ROOT / "ccpay" / "ccpay_processed.csv"),
    ProjectProcessed("filterlists", PROCESSED_ROOT / "filterlists" / "filterlists_processed.csv"),
    ProjectProcessed("jod", PROCESSED_ROOT / "jod" / "jod_processed.csv"),
    ProjectProcessed("m2os", PROCESSED_ROOT / "m2os" / "m2os_processed.csv"),
    ProjectProcessed("bruce", PROCESSED_ROOT / "bruce" / "bruce_processed.csv"),
    ProjectProcessed("rustlang", PROCESSED_ROOT / "rustlang" / "rustlang_processed.csv"),
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


def compute_test_stats(project_key: str, processed_csv: Path) -> dict[str, float]:
    df = pd.read_csv(processed_csv, low_memory=False)
    y = pd.to_numeric(df["build_duration"], errors="coerce").dropna().astype(float)
    folds = make_folds(len(y), N_FOLDS)
    _, test_idx = expanding_train_test_indices(folds, ITER5_K)
    y_te = y.iloc[test_idx].to_numpy()
    n_test = float(len(y_te))
    std_test = float(np.std(y_te, ddof=1)) if len(y_te) > 1 else float("nan")
    return {"project": project_key, "n_test": n_test, "std_test": std_test}


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    m = values.notna() & weights.notna()
    v = values[m].astype(float)
    w = weights[m].astype(float)
    if len(v) == 0:
        return float("nan")
    sw = float(w.sum())
    if sw <= 0:
        return float("nan")
    return float((v * w).sum() / sw)


def plot_top(agg: pd.DataFrame, out_png: Path, title: str, xlabel: str) -> None:
    top = agg.sort_values("importance", ascending=False).head(TOP).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10.5, 7))
    ax.barh(top["feature"], top["importance"], color="#2c7fb8", alpha=0.92)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def aggregate(df: pd.DataFrame, mode: str, normalized: bool) -> pd.DataFrame:
    """
    mode: 'unweighted' or 'weighted'
    normalized: False => ΔRMSE, True => ΔNRMSE
    """
    col = "perm_importance_mean"
    if normalized:
        col = "perm_importance_mean_nrmse"
    if mode == "unweighted":
        out = (
            df.groupby(["model", "feature"], as_index=False)[col]
            .mean()
            .rename(columns={col: "importance"})
        )
        return out
    if mode == "weighted":
        rows = []
        for (model, feature), g in df.groupby(["model", "feature"]):
            rows.append(
                {
                    "model": model,
                    "feature": feature,
                    "importance": weighted_mean(g[col], g["n_test"]),
                }
            )
        return pd.DataFrame(rows)
    raise ValueError(mode)


def main() -> None:
    if not LONG.exists():
        raise SystemExit(f"Missing input: {LONG} (run compute_feature_importance_iter5.py first)")

    base = pd.read_csv(LONG, low_memory=False)
    base["perm_importance_mean"] = pd.to_numeric(base["perm_importance_mean"], errors="coerce")
    base = base.dropna(subset=["perm_importance_mean"])

    # per-project test stats
    stats = pd.DataFrame([compute_test_stats(p.key, p.processed_csv) for p in PROJECTS])
    stats["n_test"] = pd.to_numeric(stats["n_test"], errors="coerce")
    stats["std_test"] = pd.to_numeric(stats["std_test"], errors="coerce")

    df = base.merge(stats, on="project", how="left")
    df["perm_importance_mean_nrmse"] = df["perm_importance_mean"] / df["std_test"]

    # save augmented long
    df.to_csv(OUT / "iter5_feature_importance_long_with_teststats.csv", index=False)

    variants = [
        ("unweighted", False, "global_perm_importance_all_models_unweighted.csv"),
        ("weighted", False, "global_perm_importance_all_models_weighted.csv"),
        ("unweighted", True, "global_perm_importance_all_models_unweighted_nrmse.csv"),
        ("weighted", True, "global_perm_importance_all_models_weighted_nrmse.csv"),
    ]

    for mode, norm, fname in variants:
        agg_all = aggregate(df, mode=mode, normalized=norm)
        agg_all = agg_all.sort_values(["model", "importance"], ascending=[True, False])
        agg_all.to_csv(OUT / fname, index=False)

        for model in sorted(agg_all["model"].unique()):
            sub = agg_all.loc[agg_all["model"] == model].sort_values("importance", ascending=False)
            suf = f"{mode}" + ("_nrmse" if norm else "_rmse")
            sub.to_csv(OUT / f"global_{model}_{suf}.csv", index=False)
            xlabel = (
                "Importance globale (ΔRMSE moyen sur tous les projets)"
                if not norm
                else "Importance globale normalisée (ΔNRMSE moyen sur tous les projets)"
            )
            if mode == "weighted":
                xlabel = xlabel.replace("moyen", "pondéré (poids = taille du test fold)")
            plot_top(
                sub,
                OUT / f"global_{model}_{suf}_top{TOP}.png",
                title=f"Importance globale des variables — {model} ({suf}, top {TOP})",
                xlabel=xlabel,
            )

    print("Done. Outputs in:", OUT)


if __name__ == "__main__":
    main()

