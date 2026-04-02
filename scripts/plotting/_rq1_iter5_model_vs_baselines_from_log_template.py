"""
Internal helper template (not intended to be run directly).

We keep a copy here to make the per-model scripts tiny and consistent.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _normalize_model_label(m: str) -> str:
    m = str(m).strip()
    return "Baseline 7 moy." if m == "Baseline 7 last means" else m


def _project_order(df: pd.DataFrame) -> list[str]:
    seen: list[str] = []
    for p in df["project"].tolist():
        if p not in seen:
            seen.append(p)
    return seen


def generate_one_model_figure(
    ml: str,
    log_path: Path,
    out_dir: Path,
    ml_color: str,
    baseline_lag_color: str = "#7a7a7a",
    baseline_b7_color: str = "#b35806",
) -> None:
    df = pd.read_csv(log_path)
    df["model"] = df["model"].map(_normalize_model_label)
    projects = _project_order(df)

    lag, b7 = "Baseline lag-1", "Baseline 7 moy."
    width = 0.26
    x = np.arange(len(projects))

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    series = [
        (ml, ml_color, ml),
        (lag, baseline_lag_color, "Baseline lag-1"),
        (b7, baseline_b7_color, "Baseline 7 moy."),
    ]

    for ax, metric, ylabel in zip(axes, ["NRMSE", "R2"], ["NRMSE", "R²"]):
        for i, (mkey, color, label) in enumerate(series):
            vals = [
                float(df.loc[(df["project"] == p) & (df["model"] == mkey), metric].iloc[0])
                for p in projects
            ]
            off = (i - 1) * width
            ax.bar(
                x + off,
                vals,
                width,
                label=label,
                color=color,
                edgecolor="black",
                linewidth=0.35,
                alpha=0.92,
            )
        if metric == "R2":
            ax.axhline(0, color="k", linewidth=0.5)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    axes[0].set_title(f"{ml} vs baselines — NRMSE")
    axes[1].set_title(f"{ml} vs baselines — R²")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(projects, rotation=30, ha="right")

    fig.suptitle(
        f"RQ1 — Itération 5 — {ml} : comparaison sur les 10 projets (avec baselines)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"rq1_iter5_{ml}_projets_nrmse_r2.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / f"rq1_iter5_{ml}_projets_nrmse_r2.pdf", bbox_inches="tight")
    plt.close(fig)

