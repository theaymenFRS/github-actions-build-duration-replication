"""
Aggregate "global" feature importance across all projects, per model.

Definition (global):
  For each model and feature, take the mean of permutation importance mean (ΔRMSE)
  across all (project, model) runs where the feature has a value.

Inputs:
  - out_iter5/iter5_feature_importance_long.csv (produced by compute_feature_importance_iter5.py)

Outputs (out_iter5/):
  - global_<MODEL>_perm_importance_mean.csv
  - global_<MODEL>_perm_top20.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
OUT = HERE / "out_iter5"
LONG = OUT / "iter5_feature_importance_long.csv"

TOP = 20


def main() -> None:
    if not LONG.exists():
        raise SystemExit(f"Missing input: {LONG} (run compute_feature_importance_iter5.py first)")

    df = pd.read_csv(LONG, low_memory=False)
    if not {"model", "feature", "perm_importance_mean", "project"}.issubset(df.columns):
        raise SystemExit("Unexpected columns in iter5_feature_importance_long.csv")

    df["perm_importance_mean"] = pd.to_numeric(df["perm_importance_mean"], errors="coerce")
    df = df.dropna(subset=["perm_importance_mean"])

    # mean importance across projects (unweighted: each project counts once)
    agg = (
        df.groupby(["model", "feature"], as_index=False)["perm_importance_mean"]
        .mean()
        .sort_values(["model", "perm_importance_mean"], ascending=[True, False])
    )

    # Save a single CSV with all models as well (handy for LaTeX table)
    agg.to_csv(OUT / "global_perm_importance_mean_all_models.csv", index=False)

    for model in sorted(agg["model"].unique()):
        sub = agg.loc[agg["model"] == model].sort_values("perm_importance_mean", ascending=False)
        sub.to_csv(OUT / f"global_{model}_perm_importance_mean.csv", index=False)

        top = sub.head(TOP).iloc[::-1]  # reverse for barh
        fig, ax = plt.subplots(figsize=(10.5, 7))
        ax.barh(top["feature"], top["perm_importance_mean"], color="#2c7fb8", alpha=0.92)
        ax.set_xlabel("Importance globale (permutation) : ΔRMSE moyen sur tous les projets")
        ax.set_title(f"Importance globale des variables — {model} (top {TOP})")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUT / f"global_{model}_perm_top{TOP}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    print("Done. Outputs in:", OUT)


if __name__ == "__main__":
    main()

