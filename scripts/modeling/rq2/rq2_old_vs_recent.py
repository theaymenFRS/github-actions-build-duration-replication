import os
import re
import warnings
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import mannwhitneyu

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


# =========================
# USER SETTINGS
# =========================

def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "data").is_dir() and (p / "scripts").is_dir():
            return p
    raise RuntimeError("Could not locate repo root (expected 'data/' and 'scripts/' dirs).")


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
from preprocessing.preprocess_common import preprocess_data

DATA_RAW_DIR = REPO_ROOT / "data" / "raw"

REQUIRED_FEATURES = ["duration_lag_1", "window_avg_7", "secs_since_prev", "hour", "dow"]

CORR_THR = 0.7
REDUNDANCY_R2_THR = 0.9

N_FOLDS = 10

# RQ2: 4 OLD runs vs 4 RECENT runs (8 total) with the SAME test folds (Fold 7..10)
# - OLD:    train folds 1..5, test fold 7..10  (4 runs)
# - RECENT: train folds 2..6 -> test7, 3..7 -> test8, 4..8 -> test9, 5..9 -> test10  (4 runs)
N_ITERS = 4  # iterations = 0..3 => test fold indices 6..9 (Fold 7..10)

PROJECTS = [
    ("daos",        str(DATA_RAW_DIR / "daos_wf9020028_fixed.csv")),
    ("rustlang",    str(DATA_RAW_DIR / "rustlang_wf51073_fixed.csv")),
    ("orange",      str(DATA_RAW_DIR / "Orange_OpenSourceouds_android_wf108176393_fixed.csv")),
    ("bmad",        str(DATA_RAW_DIR / "bmad simbmad ecosystem_wf69576399_fixed.csv")),
    ("ccpay",       str(DATA_RAW_DIR / "ccpay_wf6192976_fixed.csv")),
    ("filterlists", str(DATA_RAW_DIR / "collinbarrettFilterLists_wf75763098_fixed.csv")),
    ("jod",         str(DATA_RAW_DIR / "jod-yksilo-ui_wf83806327_fixed.csv")),
    ("m2os",        str(DATA_RAW_DIR / "m2Gilesm2os_wf105026558_fixed.csv")),
    ("bruce",       str(DATA_RAW_DIR / "pr3y_Bruce_wf121541665_fixed.csv")),
    ("radare2",     str(DATA_RAW_DIR / "radareorg_radare2_wf1989843_fixed.csv")),
]

MODELS = ["rf", "lgbm", "xgb", "gbr"]

# suppress constant spearman warnings from scipy/pandas internals (doesn't affect correctness)
warnings.filterwarnings("ignore", message=".*ConstantInputWarning.*")
warnings.filterwarnings("ignore", message=".*correlation coefficient is not defined.*")


# =========================
# HELPERS
# =========================
def make_folds(n, n_folds=N_FOLDS):
    """Chronological folds (data already sorted by created_at in preprocess)."""
    if n < n_folds:
        return None
    idx = np.arange(n)
    folds = np.array_split(idx, n_folds)
    if any(len(f) == 0 for f in folds):
        return None
    return folds

def first_window_indices(folds):
    # leakage-safe screening only: use earliest ~40% (folds 1..4)
    if len(folds) < 5:
        return None
    train40 = np.concatenate([folds[i] for i in range(0, 4)])  # 0..3
    return train40

def rq2_old_recent_splits(folds, it):
    """
    it = 0..3

    OLD:
      train = folds 1..5 (0..4)
      test  = folds 7..10 (6..9)  -> uses folds[6+it]

    RECENT:
      train = shifted 5-fold window:
        it=0 => folds 2..6 (1..5)
        it=1 => folds 3..7 (2..6)
        it=2 => folds 4..8 (3..7)
        it=3 => folds 5..9 (4..8)
      test  = same as OLD test: folds[6+it]
    """
    if len(folds) < 10:
        return None

    old_train_idx = np.concatenate([folds[i] for i in range(0, 5)])  # 0..4

    recent_start = 1 + it
    recent_train_idx = np.concatenate([folds[i] for i in range(recent_start, recent_start + 5)])  # 5 folds

    test_idx = folds[6 + it]  # 6..9  (Fold 7..10)
    return old_train_idx, recent_train_idx, test_idx

def _numeric_fill_median(X):
    Xn = X.apply(pd.to_numeric, errors="coerce")
    Xn = Xn.replace([np.inf, -np.inf], np.nan)
    med = Xn.median(numeric_only=True)
    return Xn.fillna(med)


# =========================
# SCREENING (CORR + REDUNDANCY)  -- KEEP AS YOU HAD
# =========================
def spearman_correlation_filter(X_tr, y_tr, feature_cols, always_keep, thr=CORR_THR):
    X = _numeric_fill_median(X_tr[feature_cols])
    std = X.std(axis=0, ddof=0)
    X = X.loc[:, std > 0]  # removes constant columns
    cols = list(X.columns)
    if len(cols) <= 1:
        return cols

    corr = X.corr(method="spearman").abs().fillna(0.0)

    ynum = pd.to_numeric(y_tr, errors="coerce")
    target_corr = {}
    for c in cols:
        v = pd.Series(X[c]).corr(ynum, method="spearman")
        target_corr[c] = 0.0 if pd.isna(v) else float(abs(v))

    keep = set(cols)
    always_keep = set([c for c in always_keep if c in keep])

    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = float(corr.iat[i, j])
            if v > thr:
                pairs.append((cols[i], cols[j], v))
    pairs.sort(key=lambda t: t[2], reverse=True)

    for a, b, _ in pairs:
        if a not in keep or b not in keep:
            continue
        a_req, b_req = (a in always_keep), (b in always_keep)
        if a_req and not b_req:
            keep.remove(b); continue
        if b_req and not a_req:
            keep.remove(a); continue
        if a_req and b_req:
            continue

        if target_corr[a] >= target_corr[b]:
            keep.remove(b)
        else:
            keep.remove(a)

    return [c for c in cols if c in keep]

def redundancy_filter_r2(X_tr, feature_cols, always_keep, thr=REDUNDANCY_R2_THR):
    X = _numeric_fill_median(X_tr[feature_cols])
    cols = list(X.columns)
    always_keep = set([c for c in always_keep if c in cols])

    n = len(X)
    if n < 40:
        return cols

    split = max(20, int(n * 0.8))
    if split >= n:
        return cols

    X_train = X.iloc[:split]
    X_test  = X.iloc[split:]
    drop = set()

    for fi in cols:
        if fi in always_keep or fi in drop:
            continue

        others = [c for c in cols if c != fi and c not in drop]
        if len(others) < 1:
            continue

        model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
        model.fit(X_train[others], X_train[fi])
        pred = model.predict(X_test[others])

        r2v = float(r2_score(X_test[fi], pred)) if len(X_test) > 1 else float("nan")
        if not np.isnan(r2v) and r2v >= thr:
            drop.add(fi)

    return [c for c in cols if c not in drop]

def screen_features_first_window_only(data, feature_cols, y_col="build_duration",
                                     corr_thr=CORR_THR, red_thr=REDUNDANCY_R2_THR,
                                     always_keep=REQUIRED_FEATURES):
    X = data.drop(columns=[y_col]).reset_index(drop=True)
    y = data[y_col].reset_index(drop=True)

    pools_X, pools_y = [], []

    for wf_id in X["workflow_id"].unique():
        m = (X["workflow_id"] == wf_id)
        Xw = X[m].reset_index(drop=True)
        yw = y[m].reset_index(drop=True)

        folds = make_folds(len(Xw), N_FOLDS)
        if folds is None:
            continue

        train40 = first_window_indices(folds)
        if train40 is None:
            continue

        pools_X.append(Xw.iloc[train40][feature_cols])
        pools_y.append(yw.iloc[train40])

    if not pools_X:
        return feature_cols

    X_tr = pd.concat(pools_X, ignore_index=True)
    y_tr = pd.concat(pools_y, ignore_index=True)

    kept = spearman_correlation_filter(X_tr, y_tr, feature_cols, always_keep, thr=corr_thr)
    kept = redundancy_filter_r2(X_tr, kept, always_keep, thr=red_thr)

    kept_set = set(kept)
    for f in always_keep:
        if f in feature_cols:
            kept_set.add(f)

    return [c for c in feature_cols if c in kept_set]


# =========================
# MODELS (DEFAULT PARAMS ONLY)
# =========================
def build_model_default(model_name: str):
    m = model_name.lower()

    if m == "rf":
        return RandomForestRegressor(random_state=42, n_jobs=-1)

    if m == "lgbm":
        return LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)

    if m == "xgb":
        return XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        )

    if m == "gbr":
        return GradientBoostingRegressor(random_state=42)

    raise ValueError(f"Unknown model_name: {model_name}")


# =========================
# METRICS
# =========================
def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    sd = float(np.std(y_true, ddof=1)) if len(y_true) > 1 else 0.0
    nrmse = float(rmse / sd) if sd > 0 else float("nan")

    mu = float(np.mean(y_true)) if len(y_true) > 0 else 0.0
    cvrmse = float(rmse / mu) if mu > 0 else float("nan")

    r2v = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")

    return {"rmse": rmse, "nrmse": nrmse, "cvrmse": cvrmse, "r2": r2v}

def cliffs_delta_from_u(u, n1, n2):
    """Cliff's delta computed from Mann–Whitney U (U for sample a)."""
    if n1 <= 0 or n2 <= 0:
        return float("nan")
    return float((2.0 * u) / (n1 * n2) - 1.0)

def cliffs_delta_magnitude(delta):
    """Magnitude labels (Romano et al. style thresholds)."""
    if np.isnan(delta):
        return "N/A"
    ad = abs(delta)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"

def safe_mwu_stats(a, b):
    a = [v for v in a if np.isfinite(v)]
    b = [v for v in b if np.isfinite(v)]
    if len(a) >= 3 and len(b) >= 3:
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        delta = cliffs_delta_from_u(u, len(a), len(b))
        mag = cliffs_delta_magnitude(delta)
        return float(p), float(delta), mag
    return float("nan"), float("nan"), "N/A"


# =========================
# CORE EVAL (OLD vs RECENT) - 8 runs total (4 old + 4 recent)
# =========================
def run_rq2_old_recent_for_model(X, y, kept_features, model_name, it):
    folds = make_folds(len(X), N_FOLDS)
    if folds is None or len(folds) < 10:
        return None

    split = rq2_old_recent_splits(folds, it)
    if split is None:
        return None
    old_tr_idx, rec_tr_idx, te_idx = split

    X_old, y_old = X.iloc[old_tr_idx][kept_features], y.iloc[old_tr_idx]
    X_rec, y_rec = X.iloc[rec_tr_idx][kept_features], y.iloc[rec_tr_idx]
    X_te,  y_te  = X.iloc[te_idx][kept_features],     y.iloc[te_idx]

    # minimum sizes for stability
    if len(X_old) < 10 or len(X_rec) < 10 or len(X_te) < 5:
        return None

    # OLD model
    mdl_old = build_model_default(model_name)
    mdl_old.fit(X_old, y_old)
    pred_old = np.maximum(0.0, mdl_old.predict(X_te))
    m_old = compute_metrics(y_te, pred_old)

    # RECENT model
    mdl_rec = build_model_default(model_name)
    mdl_rec.fit(X_rec, y_rec)
    pred_rec = np.maximum(0.0, mdl_rec.predict(X_te))
    m_rec = compute_metrics(y_te, pred_rec)

    # fold numbers for logging (1-based for humans)
    test_fold_num = 7 + it  # because te_idx = folds[6+it] => Fold (7+it)
    old_train_folds = "1-5"
    recent_train_folds = f"{2+it}-{6+it}"

    return {
        "it": it + 1,
        "test_fold": test_fold_num,
        "old_train": old_train_folds,
        "recent_train": recent_train_folds,
        "old": m_old,
        "recent": m_rec,
    }


# =========================
# PLOTTING
# =========================
def plot_project_grid(metrics_by_model, model_order, out_png, title):
    """
    metrics_by_model[model] = {
      "rmse":  (old_list, recent_list),
      "nrmse": (old_list, recent_list),
      "r2":    (old_list, recent_list),
      "cvrmse":(old_list, recent_list),
    }
    """
    row_defs = [
        ("RMSE",   "rmse"),
        ("NRMSE",  "nrmse"),
        ("R\u00b2","r2"),
        ("CVRMSE", "cvrmse"),
    ]

    nrows = len(row_defs)
    ncols = len(model_order)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.2*ncols, 10.4))

    fig.suptitle(title, fontsize=16, y=0.99)

    colors = ["#F4A6A6", "#88D1D8"]  # old, recent

    # shared y-lims per metric row (across models) to make comparisons fair
    ylims = {}
    for _, key in row_defs:
        all_vals = []
        for m in model_order:
            old_vals, rec_vals = metrics_by_model[m][key]
            all_vals.extend([v for v in old_vals if np.isfinite(v)])
            all_vals.extend([v for v in rec_vals if np.isfinite(v)])
        if len(all_vals) == 0:
            ylims[key] = None
            continue
        lo = np.percentile(all_vals, 1)
        hi = np.percentile(all_vals, 99)
        pad = 0.08 * (hi - lo) if hi > lo else (1.0 if hi == 0 else abs(hi) * 0.1)
        ylims[key] = (lo - pad, hi + pad)

    for r, (ylabel, key) in enumerate(row_defs):
        for c, model in enumerate(model_order):
            ax = axes[r, c] if ncols > 1 else axes[r]

            old_vals, rec_vals = metrics_by_model[model][key]
            data = [old_vals, rec_vals]

            bp = ax.boxplot(
                data,
                tick_labels=["old", "recent"],
                patch_artist=True,
                showfliers=False,
                widths=0.6
            )

            if ylims[key] is not None:
                ax.set_ylim(*ylims[key])

            for patch, col in zip(bp["boxes"], colors):
                patch.set_facecolor(col)
                patch.set_edgecolor(col)
                patch.set_alpha(0.35)
            for med, col in zip(bp["medians"], colors):
                med.set_color(col)
                med.set_linewidth(2)

            ax.set_title(
                model.upper(),
                fontsize=12,
                bbox=dict(facecolor="#E9E9E9", edgecolor="black", boxstyle="square,pad=0.3")
            )

            if c == 0:
                ax.set_ylabel(ylabel, fontsize=14)
            else:
                ax.set_ylabel("")

            ax.grid(axis="y", alpha=0.25)

            # p-value + decision under subplot
            p, d, mag = safe_mwu_stats(old_vals, rec_vals)

            ptxt = "p=nan" if np.isnan(p) else f"p={p:.3g}"
            dtxt = "d=nan" if np.isnan(d) else f"d={d:+.3f}"

            ax.text(
                0.5, -0.26,
                f"{ptxt} | {dtxt}({mag})",
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=10, fontweight="bold",
                clip_on=False
            )

    fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.08, wspace=0.35, hspace=0.85)
    plt.savefig(out_png, dpi=250)
    plt.close()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # Aligné avec results/rq2/ (voir scripts/README.md et results/README.md)
    root_out = REPO_ROOT / "results" / "rq2" / "concept_drift_old_vs_recent_DEFAULTPARAMS"
    root_out.mkdir(parents=True, exist_ok=True)

    global_log_path = root_out / "rq2_ALL_PROJECTS_RQ2.log"

    # for the final "big figure" (all projects combined)
    overall = {m: {"rmse": ([], []), "nrmse": ([], []), "r2": ([], []), "cvrmse": ([], [])} for m in MODELS}

    with open(global_log_path, "w", encoding="utf-8") as glog:
        glog.write("RQ2 Concept Drift (OLD vs RECENT) — DEFAULT MODEL PARAMETERS\n")
        glog.write(f"N_FOLDS={N_FOLDS}, N_ITERS(old)=4, N_ITERS(recent)=4, test folds = 7..10\n")
        glog.write("=" * 100 + "\n\n")

        for proj_name, file_path in PROJECTS:
            proj_out = root_out / proj_name
            proj_out.mkdir(parents=True, exist_ok=True)

            proj_log_path = proj_out / f"rq2_{proj_name}_RQ2.log"
            processed_out = proj_out / f"{proj_name}_processed.csv"

            # preprocess once
            data = preprocess_data(file_path, output_path=processed_out, verbose=False)
            if data is None or len(data) == 0:
                msg = f"[WARN] {proj_name}: No data after preprocessing\n"
                print(msg.strip())
                glog.write(msg)
                continue

            X = data.drop(columns=["build_duration"])
            y = data["build_duration"]
            feature_cols = [c for c in X.columns if c != "workflow_id"]

            kept_features = screen_features_first_window_only(
                data, feature_cols, y_col="build_duration",
                corr_thr=CORR_THR, red_thr=REDUNDANCY_R2_THR,
                always_keep=REQUIRED_FEATURES
            )

            # collect 4 old + 4 recent per model/metric (each list length 4)
            metrics_by_model = {
                m: {"rmse": ([], []), "nrmse": ([], []), "r2": ([], []), "cvrmse": ([], [])}
                for m in MODELS
            }

            with open(proj_log_path, "w", encoding="utf-8") as plog:
                plog.write(f"PROJECT: {proj_name}\n")
                plog.write(f"CSV: {file_path}\n")
                plog.write(f"[FEATURES] initial={len(feature_cols)} kept={len(kept_features)}\n")
                plog.write("Kept features:\n")
                plog.write(str(kept_features) + "\n")
                plog.write("-" * 100 + "\n\n")

                # run iterations
                for it in range(N_ITERS):  # 0..3 => test folds 7..10
                    plog.write(f"ITERATION {it+1} (Test Fold {7+it})\n")
                    plog.write("-" * 80 + "\n")

                    for model_name in MODELS:
                        res = run_rq2_old_recent_for_model(X, y, kept_features, model_name, it)
                        if res is None:
                            plog.write(f"{model_name.upper()}: SKIP (insufficient data)\n")
                            continue

                        oldm = res["old"]
                        recm = res["recent"]

                        # store for boxplots
                        metrics_by_model[model_name]["rmse"][0].append(oldm["rmse"])
                        metrics_by_model[model_name]["rmse"][1].append(recm["rmse"])

                        metrics_by_model[model_name]["nrmse"][0].append(oldm["nrmse"])
                        metrics_by_model[model_name]["nrmse"][1].append(recm["nrmse"])

                        metrics_by_model[model_name]["r2"][0].append(oldm["r2"])
                        metrics_by_model[model_name]["r2"][1].append(recm["r2"])

                        metrics_by_model[model_name]["cvrmse"][0].append(oldm["cvrmse"])
                        metrics_by_model[model_name]["cvrmse"][1].append(recm["cvrmse"])

                        # log per run
                        plog.write(
                            f"{model_name.upper()} | OLD(train {res['old_train']}) -> test fold {res['test_fold']} | "
                            f"RMSE={oldm['rmse']:.6f} NRMSE={oldm['nrmse']:.6f} R2={oldm['r2']:.6f} CVRMSE={oldm['cvrmse']:.6f}\n"
                        )
                        plog.write(
                            f"{model_name.upper()} | RECENT(train {res['recent_train']}) -> test fold {res['test_fold']} | "
                            f"RMSE={recm['rmse']:.6f} NRMSE={recm['nrmse']:.6f} R2={recm['r2']:.6f} CVRMSE={recm['cvrmse']:.6f}\n"
                        )

                    plog.write("\n")

                # p-values per model/metric
                plog.write("=" * 100 + "\n")
                plog.write("Mann–Whitney U tests (OLD vs RECENT) using the 4 runs per group\n")
                plog.write("=" * 100 + "\n")

                for model_name in MODELS:
                    for metric_key in ["rmse", "nrmse", "r2", "cvrmse"]:
                        old_vals, rec_vals = metrics_by_model[model_name][metric_key]
                        p, d, mag = safe_mwu_stats(old_vals, rec_vals)

                        ptxt = "p=nan" if np.isnan(p) else f"p={p:.3g}"
                        dtxt = "d=nan" if np.isnan(d) else f"d={d:+.3f}"

                        plog.write(
                            f"{model_name.upper():4s} | {metric_key.upper():6s} | "
                            f"old={old_vals} recent={rec_vals} | {ptxt} | {dtxt}({mag})\n"
                        )

                # save project figure
                fig_path = proj_out / f"rq2_{proj_name}_RQ2_GRID.png"
                plot_project_grid(
                    metrics_by_model=metrics_by_model,
                    model_order=MODELS,
                    out_png=fig_path,
                    title=f"{proj_name} — Old vs Recent (Default Params, 4 vs 4 runs)"
                )

            # append to global log + build overall distributions
            glog.write(f"[PROJECT] {proj_name}\n")
            glog.write(f"  kept_features={len(kept_features)}\n")
            for model_name in MODELS:
                for metric_key in ["rmse", "nrmse", "r2", "cvrmse"]:
                    old_vals, rec_vals = metrics_by_model[model_name][metric_key]
                    overall[model_name][metric_key][0].extend(old_vals)
                    overall[model_name][metric_key][1].extend(rec_vals)
            glog.write("\n")

        # final big figure (all projects pooled)
        overall_by_model = {m: overall[m] for m in MODELS}
        overall_fig = root_out / "rq2_ALL_PROJECTS_RQ2_GRID.png"
        plot_project_grid(
            metrics_by_model=overall_by_model,
            model_order=MODELS,
            out_png=overall_fig,
            title="ALL PROJECTS — Old vs Recent (Default Params, pooled runs)"
        )

        # log overall p-values
        glog.write("=" * 100 + "\n")
        glog.write("OVERALL Mann–Whitney U tests (ALL projects pooled)\n")
        glog.write("=" * 100 + "\n")
        for model_name in MODELS:
            for metric_key in ["rmse", "nrmse", "r2", "cvrmse"]:
                old_vals, rec_vals = overall[model_name][metric_key]
                p, d, mag = safe_mwu_stats(old_vals, rec_vals)
                ptxt = "p=nan" if np.isnan(p) else f"p={p:.3g}"
                dtxt = "d=nan" if np.isnan(d) else f"d={d:+.3f}"

                glog.write(
                    f"{model_name.upper():4s} | {metric_key.upper():6s} | "
                    f"n_old={len(old_vals)} n_recent={len(rec_vals)} | {ptxt} | {dtxt}({mag})\n"
                )

    print(f"[DONE] Output folder: {root_out}")
    print(f"[DONE] Global log   : {global_log_path}")
