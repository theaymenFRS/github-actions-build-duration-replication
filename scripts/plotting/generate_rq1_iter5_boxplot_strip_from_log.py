"""
Generate ONLY `rq1_iter5_boxplot_strip` from `rq1_iter5_metrics.log`.

Input:
  - src/rq1_iter5_metrics.log

Output:
  - src/figures_rq1_iter5_from_log/rq1_iter5_boxplot_strip.png
  - src/figures_rq1_iter5_from_log/rq1_iter5_boxplot_strip.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "rq1_iter5_metrics.log"
OUT = ROOT / "figures_rq1_iter5_from_log"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["RF", "LGB", "XGB", "GBR", "DT", "Baseline lag-1", "Baseline 7 moy."]


def _normalize_model_label(m: str) -> str:
    m = str(m).strip()
    return "Baseline 7 moy." if m == "Baseline 7 last means" else m


def main() -> None:
    if not LOG.exists():
        raise SystemExit(f"Missing {LOG}. Run rq1_iter5_extract_metrics_log.py first.")

    df = pd.read_csv(LOG)
    df["model"] = df["model"].map(_normalize_model_label)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    palette = ["#2c7fb8"] * 5 + ["#7a7a7a", "#b35806"]

    for ax, metric, t in zip(
        axes,
        ["NRMSE", "R2"],
        ["NRMSE par projet (10 points par modèle)", "R² par projet"],
    ):
        data = [df.loc[df["model"] == m, metric].values for m in MODEL_ORDER]
        bp = ax.boxplot(
            data,
            tick_labels=MODEL_ORDER,
            patch_artist=True,
            showmeans=True,
            meanline=True,
        )
        for patch, c in zip(bp["boxes"], palette):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)

        rng = np.random.default_rng(42)
        for i, m in enumerate(MODEL_ORDER):
            y = df.loc[df["model"] == m, metric].values
            x = np.full(len(y), i + 1) + rng.uniform(-0.09, 0.09, size=len(y))
            ax.scatter(x, y, s=22, c="black", alpha=0.45, zorder=3)

        ax.set_xticklabels(MODEL_ORDER, rotation=25, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(t)
        ax.axhline(0, color="k", linewidth=0.5)

    fig.suptitle(
        "RQ1 — Itération 5 — dispersion entre projets (chaque point = un projet)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT / "rq1_iter5_boxplot_strip.png", dpi=150)
    fig.savefig(OUT / "rq1_iter5_boxplot_strip.pdf")
    plt.close(fig)

    print("Done:", OUT / "rq1_iter5_boxplot_strip.png")


if __name__ == "__main__":
    main()

