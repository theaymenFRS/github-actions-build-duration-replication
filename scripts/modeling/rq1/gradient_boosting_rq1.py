import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import random
import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
#from xgboost import XGBRegressor
#from lightgbm import LGBMRegressor
import matplotlib
#from sklearn.ensemble import RandomForestRegressor
import re
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor


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
DATA_PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = "gradient_boosting_placeholder.log"
# === GA feature constraints ===
REQUIRED_FEATURES = [
    "duration_lag_1",
    "window_avg_7",
    "secs_since_prev",
    "hour",
    "dow",
]
# ---- Screening thresholds ----
CORR_THR = 0.7          # Spearman abs corr > 0.7 => correlated
REDUNDANCY_R2_THR = 0.9 # if feature predictable from others with R2 >= 0.9 => redundant

# ---- Fold / drift settings ----
N_FOLDS = 10
N_ITERS = 5

# ---- GA settings (hyperparam tuning only; runs once) ----
GA_GENERATIONS = 15
GA_POP_SIZE = 20
GA_MUTATION_RATE = 0.35
GA_SEED = 42


""""
def log_individual(individual, gen_num, ind_num, workflow_rmse, avg_rmse, workflow_r2, avg_r2, predictions_sample):
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"Generation: {gen_num}, Individual: {ind_num}\n")
        log_file.write(f"Params: {individual['params']}\n")
        log_file.write(f"Selected Features: {individual['features']}\n")
        log_file.write(f"Workflow RMSEs: {workflow_rmse}\n")
        log_file.write(f"Workflow R² Scores: {workflow_r2}\n")
        log_file.write(f"Average RMSE: {avg_rmse:.4f}, Average R²: {avg_r2:.4f}\n")
        log_file.write(f"Sample Predictions (actual -> predicted):\n")
        for actual, pred in predictions_sample:
            log_file.write(f"  {actual:.4f} -> {pred:.4f}\n")
        log_file.write("-" * 50 + "\n")
"""

def make_folds(n, n_folds=N_FOLDS):
    """Split indices 0..n-1 into n_folds contiguous chunks."""
    if n < n_folds:
        return None
    idx = np.arange(n)
    folds = np.array_split(idx, n_folds)
    if any(len(f) == 0 for f in folds):
        return None
    return folds

def first_window_indices(folds):
    """
    Iteration 1 tuning window (1-indexed in your description):
      train = folds 1..4  -> indices 0..3
      val   = fold 5      -> index 4
      test  = fold 6      -> index 5 (NEVER used for tuning)
    """
    if len(folds) < 6:
        return None
    train40 = np.concatenate([folds[i] for i in range(0, 4)])  # 0..3
    val10   = folds[4]                                        # 4
    test10  = folds[5]                                        # 5
    return train40, val10, test10

def expanding_train_test_indices(folds, k):
    """
    Expanding window evaluation (matches your new picture), k=0..4:

      Iter1(k=0): train folds 1..5  (0..4) -> test fold 6  (5)
      Iter2(k=1): train folds 1..6  (0..5) -> test fold 7  (6)
      Iter3(k=2): train folds 1..7  (0..6) -> test fold 8  (7)
      Iter4(k=3): train folds 1..8  (0..7) -> test fold 9  (8)
      Iter5(k=4): train folds 1..9  (0..8) -> test fold 10 (9)
    """
    train_idx = np.concatenate([folds[i] for i in range(0, 5 + k)])
    test_idx  = folds[5 + k]
    return train_idx, test_idx

def compute_baseline_last_k_expanding_folds(X, y, k_roll=7):
    """
    Baseline matching your expanding-window folds:
      Iter1: train folds 1..5 -> test fold 6
      Iter2: train folds 1..6 -> test fold 7
      ...
      Iter5: train folds 1..9 -> test fold 10

    Uses "last-k mean" with shift(1) so it's leakage-safe.
    Returns:
      results[it] = dict with rmse/r2/nrmse/cvrmse lists across workflows
    """
    results = {it: {"rmse": [], "r2": [], "nrmse": [], "cvrmse": []} for it in range(N_ITERS)}

    for wf_id in X["workflow_id"].unique():
        m = (X["workflow_id"] == wf_id)
        yw = y[m].reset_index(drop=True)

        folds = make_folds(len(yw), N_FOLDS)
        if folds is None:
            continue

        # last-k mean baseline using only previous values
        baseline = yw.rolling(window=k_roll, min_periods=1).mean().shift(1)

        for it in range(N_ITERS):
            train_idx, test_idx = expanding_train_test_indices(folds, it)

            y_te = yw.iloc[test_idx]
            y_pred = baseline.iloc[test_idx]

            valid = y_pred.notna()
            y_te = y_te[valid]
            y_pred = y_pred[valid]
            if len(y_te) == 0:
                continue

            rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
            r2v  = float(r2_score(y_te, y_pred)) if len(y_te) > 1 else float("nan")

            sd   = float(np.std(y_te, ddof=1)) if len(y_te) > 1 else 0.0
            mean = float(np.mean(y_te)) if len(y_te) > 0 else 0.0
            nrmse  = rmse / sd if sd > 0 else float("nan")
            cvrmse = rmse / mean if mean > 0 else float("nan")

            results[it]["rmse"].append(rmse)
            results[it]["r2"].append(r2v)
            results[it]["nrmse"].append(nrmse)
            results[it]["cvrmse"].append(cvrmse)

    return results

def _numeric_fill_median(X):
    Xn = X.apply(pd.to_numeric, errors="coerce")
    Xn = Xn.replace([np.inf, -np.inf], np.nan)
    med = Xn.median(numeric_only=True)
    return Xn.fillna(med)

def spearman_correlation_filter(X_tr, y_tr, feature_cols, always_keep, thr=CORR_THR):
    """
    Remove one feature from each highly correlated pair (|rho| > thr),
    keeping the one more correlated with y (Spearman).
    NEVER drops features in always_keep.
    """
    X = _numeric_fill_median(X_tr[feature_cols])

    # drop constant cols
    std = X.std(axis=0, ddof=0)
    X = X.loc[:, std > 0]
    cols = list(X.columns)
    if len(cols) <= 1:
        return cols

    corr = X.corr(method="spearman").abs().fillna(0.0)

    # feature -> |spearman(feature,y)|
    ynum = pd.to_numeric(y_tr, errors="coerce")
    target_corr = {}
    for c in cols:
        col = X[c]
        if isinstance(col, pd.DataFrame):  # happens when duplicate names exist
            col = col.iloc[:, 0]
        v = pd.Series(col).corr(ynum, method="spearman")
        target_corr[c] = 0.0 if pd.isna(v) else float(abs(v))

    keep = set(cols)
    always_keep = set([c for c in always_keep if c in keep])

    # iterate pairs descending corr
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

        # drop weaker wrt target
        if target_corr[a] >= target_corr[b]:
            keep.remove(b)
        else:
            keep.remove(a)

    return [c for c in cols if c in keep]

def redundancy_filter_r2(X_tr, feature_cols, always_keep, thr=REDUNDANCY_R2_THR):
    """
    For each feature fi, predict fi from other features using Ridge on a time-based split.
    If R2 >= thr, drop fi unless required.
    """
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

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0))
        ])

        model.fit(X_train[others], X_train[fi])
        pred = model.predict(X_test[others])

        r2v = float(r2_score(X_test[fi], pred)) if len(X_test) > 1 else float("nan")
        if not np.isnan(r2v) and r2v >= thr:
            drop.add(fi)

    return [c for c in cols if c not in drop]

def screen_features_first_window_only(data, feature_cols, y_col="build_duration",
                                     corr_thr=CORR_THR, red_thr=REDUNDANCY_R2_THR,
                                     always_keep=REQUIRED_FEATURES):
    """
    Screening happens right after cleaning, but done leakage-safe:
    pool TRAIN folds 1..4 (40%) from Iteration 1 only, across workflows.
    """
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

        fw = first_window_indices(folds)
        if fw is None:
            continue
        train40, _, _ = fw

        pools_X.append(Xw.iloc[train40][feature_cols])
        pools_y.append(yw.iloc[train40])

    if not pools_X:
        return feature_cols  # fallback

    X_tr = pd.concat(pools_X, ignore_index=True)
    y_tr = pd.concat(pools_y, ignore_index=True)

    kept = spearman_correlation_filter(X_tr, y_tr, feature_cols, always_keep, thr=corr_thr)
    kept = redundancy_filter_r2(X_tr, kept, always_keep, thr=red_thr)

    # ensure required always kept (if present)
    kept_set = set(kept)
    for f in always_keep:
        if f in feature_cols:
            kept_set.add(f)

    return [c for c in feature_cols if c in kept_set]

def log_individual(individual, gen_num, ind_num,
                   workflow_rmse, avg_rmse,
                   workflow_r2, avg_r2,
                   predictions_sample,
                   workflow_sd, workflow_mean, workflow_median,
                   workflow_nrmse, workflow_cvrmse,
                   avg_nrmse, avg_cvrmse,
                   avg_mean, avg_median):
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"Generation: {gen_num}, Individual: {ind_num}\n")
        log_file.write(f"Params: {individual['params']}\n")
        log_file.write(f"Workflow mean(y): {workflow_mean}\n")
        log_file.write(f"Workflow median(y): {workflow_median}\n")
        log_file.write(f"Workflow SD(y): {workflow_sd}\n")
        log_file.write(f"Workflow RMSEs: {workflow_rmse}\n")
        log_file.write(f"Workflow NRMSE (RMSE/SD): {workflow_nrmse}\n")
        log_file.write(f"Workflow CVRMSE (RMSE/mean): {workflow_cvrmse}\n")
        log_file.write(f"Workflow R² Scores: {workflow_r2}\n")
        log_file.write(f"Averages -> RMSE: {avg_rmse:.4f}, NRMSE: {avg_nrmse:.4f}, CVRMSE: {avg_cvrmse:.4f}, R²: {avg_r2:.4f}\n")
        log_file.write("Sample Predictions (actual -> predicted):\n")
        for actual, pred in predictions_sample:
            log_file.write(f"  {actual:.4f} -> {pred:.4f}\n")
        log_file.write("-" * 50 + "\n")

def ga_split_for_iter(folds, k):
    """
    k=0..4
    Iter1: train=folds 1..4, val=5, test=6
    Iter2: train=1..5, val=6, test=7
    ...
    Iter5: train=1..8, val=9, test=10
    """
    # folds are 0-indexed in code, but described 1-indexed in paper
    train_idx = np.concatenate([folds[i] for i in range(0, 4 + k)])  # 0..(3+k)
    val_idx   = folds[4 + k]                                         # (4+k)
    test_idx  = folds[5 + k]                                         # (5+k)  NEVER used in GA
    return train_idx, val_idx, test_idx


def fitness_ga_iter(individual, X, y, kept_features, gen_num, ind_num, k_iter):
    """
    GA fitness for a given iteration k_iter:
      uses ONLY GA-train + GA-val (inside the white block)
      does NOT touch the yellow test fold.
    """
    params = individual["params"]
    rmses = []

    for wf_id in X["workflow_id"].unique():
        m = (X["workflow_id"] == wf_id)
        Xw = X[m].reset_index(drop=True)
        yw = y[m].reset_index(drop=True)

        folds = make_folds(len(Xw), N_FOLDS)
        if folds is None or len(folds) < (6 + k_iter):
            continue

        tr_idx, va_idx, _ = ga_split_for_iter(folds, k_iter)

        X_tr = Xw.iloc[tr_idx][kept_features]
        y_tr = yw.iloc[tr_idx]
        X_va = Xw.iloc[va_idx][kept_features]
        y_va = yw.iloc[va_idx]

        if len(X_tr) < 30 or len(X_va) < 5:
            continue

        # winsorize train only
        p1, p99 = y_tr.quantile([0.01, 0.99])
        y_tr_clip = y_tr.clip(lower=p1, upper=p99)

        model = GradientBoostingRegressor(
            **params,
            random_state=42
        )

        # For GBR: start WITHOUT log1p(y) (recommended)
        model.fit(X_tr.to_numpy(), y_tr_clip.to_numpy())

        pred = model.predict(X_va.to_numpy())
        pred = np.maximum(0.0, pred)

        rmse = float(np.sqrt(mean_squared_error(y_va, pred)))
        rmses.append(rmse)

    avg_rmse = float(np.mean(rmses)) if rmses else float("inf")

    with open(LOG_FILE, "a") as f:
        f.write(f"[GA][Iter{k_iter+1}] Gen:{gen_num} Ind:{ind_num} RMSE(val)={avg_rmse:.6f}\n")

    return avg_rmse

def enforce_required(bits, feature_cols, required_set):
    """Force required features to 1 in the binary mask."""
    for i, f in enumerate(feature_cols):
        if f in required_set:
            bits[i] = 1
    return bits

def _clip(v, lo, hi):
    return max(lo, min(hi, v))

def generate_individual():
    params = {
        "n_estimators": random.randint(800, 1200),
        "learning_rate": 10 ** random.uniform(-2.2, -0.7),   # ~0.006 to ~0.20
        "max_depth": random.randint(2, 6),                   # depth of each tree
        "min_samples_split": random.randint(2, 30),
        "min_samples_leaf": random.randint(1, 20),
        "subsample": random.uniform(0.6, 1.0),               # stochastic GB
        "max_features": random.choice([None, "sqrt", "log2"]),
    }
    return {"params": params}

def mutate(individual):
    if random.random() > GA_MUTATION_RATE:
        return

    p = individual["params"]
    k = random.choice(list(p.keys()))

    if k == "n_estimators":
        p[k] = int(max(50, min(4000, p[k] + random.randint(-200, 200))))

    elif k == "learning_rate":
        p[k] = float(max(0.001, min(0.3, p[k] * (10 ** random.uniform(-0.25, 0.25)))))

    elif k == "max_depth":
        p[k] = int(max(1, min(10, p[k] + random.choice([-1, 1]))))

    elif k == "min_samples_split":
        p[k] = int(max(2, min(100, p[k] + random.randint(-3, 3))))

    elif k == "min_samples_leaf":
        p[k] = int(max(1, min(50, p[k] + random.randint(-2, 2))))

    elif k == "subsample":
        p[k] = float(max(0.4, min(1.0, p[k] + random.uniform(-0.1, 0.1))))

    elif k == "max_features":
        p[k] = random.choice([None, "sqrt", "log2"])

def crossover(parent1, parent2):
    c = {"params": {}}
    for k in parent1["params"].keys():
        c["params"][k] = parent1["params"][k] if random.random() < 0.5 else parent2["params"][k]
    return c


def plot_generation_rmse(generation_rmse):
    """ Save the average generation MSE plot to a file """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(generation_rmse) + 1), generation_rmse, marker='o', linestyle='-', color='blue')
    plt.title("Average Generation RMSE")
    plt.xlabel("Generation")
    plt.ylabel("Average RMSE")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save the figure
    output_path = "generation_rmse_plot.png"
    plt.savefig(output_path)
    plt.close()


def ga_tune_hyperparams_for_iter(X, y, kept_features, k_iter,
                                num_generations=GA_GENERATIONS,
                                population_size=GA_POP_SIZE):
    random.seed(GA_SEED)
    np.random.seed(GA_SEED)

    population = [generate_individual() for _ in range(population_size)]

    for gen in range(1, num_generations + 1):
        print(f"[GA][Iter{k_iter+1}] Generation {gen}/{num_generations}")

        fitness = [
            fitness_ga_iter(ind, X, y, kept_features, gen, i + 1, k_iter)
            for i, ind in enumerate(population)
        ]

        order = np.argsort(fitness)
        elite = [population[i] for i in order[: max(2, population_size // 2)]]

        children = []
        while len(children) < population_size - len(elite):
            p1, p2 = random.sample(elite, 2)
            child = crossover(p1, p2)
            mutate(child)
            children.append(child)

        population = elite + children

        best_i = int(order[0])
        best_rmse = float(fitness[best_i])
        with open(LOG_FILE, "a") as f:
            f.write(f"[GA][Iter{k_iter+1}] gen={gen} best_val_rmse={best_rmse:.6f}\n")

    # final best
    final_fitness = [
        fitness_ga_iter(ind, X, y, kept_features, num_generations, i + 1, k_iter)
        for i, ind in enumerate(population)
    ]
    best_i = int(np.argmin(final_fitness))
    best_params = population[best_i]["params"]
    best_rmse = float(final_fitness[best_i])

    with open(LOG_FILE, "a") as f:
        f.write(f"\n[GA DONE][Iter{k_iter+1}] Best params on val fold of Iter{k_iter+1}\n")
        f.write(f"Best val RMSE: {best_rmse:.6f}\n")
        f.write(f"Best params: {best_params}\n")
        f.write("-" * 80 + "\n")

    return best_params

def evaluate_expanding_window_retuned_ga(X, y, kept_features):
    results = {k: {"rmse": [], "r2": [], "nrmse": [], "cvrmse": []} for k in range(N_ITERS)}

    for k in range(N_ITERS):
        # --- GA re-tune for this iteration ---
        best_params = ga_tune_hyperparams_for_iter(X, y, kept_features, k_iter=k)

        for wf_id in X["workflow_id"].unique():
            m = (X["workflow_id"] == wf_id)
            Xw = X[m].reset_index(drop=True)
            yw = y[m].reset_index(drop=True)

            folds = make_folds(len(Xw), N_FOLDS)
            if folds is None or len(folds) < (6 + k):
                continue

            # White training pool & Yellow test
            train_idx, test_idx = expanding_train_test_indices(folds, k)

            X_tr = Xw.iloc[train_idx][kept_features]
            y_tr = yw.iloc[train_idx]
            X_te = Xw.iloc[test_idx][kept_features]
            y_te = yw.iloc[test_idx]

            if len(X_tr) < 30 or len(X_te) < 5:
                continue

            p1, p99 = y_tr.quantile([0.01, 0.99])
            y_tr_clip = y_tr.clip(lower=p1, upper=p99)

            model = GradientBoostingRegressor(
                **best_params,
                random_state=42
            )

            model.fit(X_tr.to_numpy(), y_tr_clip.to_numpy())
            pred = model.predict(X_te.to_numpy())
            pred = np.maximum(0.0, pred)

            rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
            r2v  = float(r2_score(y_te, pred)) if len(y_te) > 1 else float("nan")

            sd   = float(np.std(y_te, ddof=1)) if len(y_te) > 1 else 0.0
            mean = float(np.mean(y_te)) if len(y_te) > 0 else 0.0
            nrmse  = rmse / sd if sd > 0 else float("nan")
            cvrmse = rmse / mean if mean > 0 else float("nan")

            results[k]["rmse"].append(rmse)
            results[k]["r2"].append(r2v)
            results[k]["nrmse"].append(nrmse)
            results[k]["cvrmse"].append(cvrmse)

        # log iteration result
        with open(LOG_FILE, "a") as f:
            rmse = float(np.nanmean(results[k]["rmse"])) if results[k]["rmse"] else float("nan")
            r2v  = float(np.nanmean(results[k]["r2"])) if results[k]["r2"] else float("nan")
            nrmse = float(np.nanmean(results[k]["nrmse"])) if results[k]["nrmse"] else float("nan")
            cvrmse = float(np.nanmean(results[k]["cvrmse"])) if results[k]["cvrmse"] else float("nan")

            train_end = 5 + k
            test_fold = 6 + k
            f.write(
                f"[RETUNE GA] Iter{k+1}: train folds 1..{train_end} -> test fold {test_fold}  "
                f"RMSE={rmse:.6f}  NRMSE={nrmse:.6f}  CVRMSE={cvrmse:.6f}  R2={r2v:.6f}\n"
            )

    return results

def compute_baseline_last_k(X, y, k=7, test_frac=0.1):
    """
    Non-ML baseline: predict each build as the mean of the previous k durations (shifted by 1),
    computed separately per workflow_id, using the same 90/10 time-based split.

    Returns:
      per_workflow: list of dicts with metrics per workflow
      avg: dict with average RMSE, R2, NRMSE, CVRMSE across workflows
    """
    per_workflow = []

    for wf_id in X["workflow_id"].unique():
        mask = (X["workflow_id"] == wf_id)
        # Keep original time order (your preprocess kept chronological order / index)
        y_w = y[mask].copy().sort_index()

        n = len(y_w)
        if n < 3:
            continue

        cut = int(n * (1 - test_frac))  # 90% train / 10% test
        # rolling mean of previous k (strictly previous -> shift(1))
        baseline = y_w.rolling(window=k, min_periods=1).mean().shift(1)

        y_test = y_w.iloc[cut:]
        y_pred = baseline.iloc[cut:]

        # drop NaNs that arise at the very start of the series
        valid = y_pred.notna()
        y_test = y_test[valid]
        y_pred = y_pred[valid]
        if len(y_test) == 0:
            continue

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2   = float(r2_score(y_test, y_pred)) if len(y_test) > 1 else float("nan")

        sd   = float(np.std(y_test, ddof=1))
        mean = float(np.mean(y_test))
        nrmse  = float(rmse / sd)   if sd   > 0 else float("nan")
        cvrmse = float(rmse / mean) if mean > 0 else float("nan")

        per_workflow.append({
            "workflow_id": int(wf_id),
            "rmse": rmse, "r2": r2, "nrmse": nrmse, "cvrmse": cvrmse
        })

    # Averages (nan-aware where relevant)
    avg_rmse   = float(np.mean([d["rmse"] for d in per_workflow])) if per_workflow else float("nan")
    avg_r2     = float(np.nanmean([d["r2"] for d in per_workflow])) if per_workflow else float("nan")
    avg_nrmse  = float(np.nanmean([d["nrmse"] for d in per_workflow])) if per_workflow else float("nan")
    avg_cvrmse = float(np.nanmean([d["cvrmse"] for d in per_workflow])) if per_workflow else float("nan")

    return per_workflow, {"rmse": avg_rmse, "r2": avg_r2, "nrmse": avg_nrmse, "cvrmse": avg_cvrmse}

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

if __name__ == "__main__":
    plt.ion()  # interactive on if you still want live plot windows

    out_dir = REPO_ROOT / "results" / "rq1" / "gradient_boosting"
    out_dir.mkdir(parents=True, exist_ok=True)

    for project_name, file_path in PROJECTS:
        print("\n" + "=" * 80)
        print(f"PROJECT: {project_name}")
        print(f"CSV    : {file_path}")
        print("=" * 80)

        # --- Per-project log file ---
        LOG_FILE = str(out_dir / f"gradient_boosting_{project_name}_RQ1.log")

        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)

        # --- Preprocess data for this project ---
        processed_out = str(DATA_PROCESSED_DIR / f"{project_name}_processed_2025.csv")

        # --- Preprocess data for this project and save it ---
        data = preprocess_data(file_path, output_path=processed_out)

        if data is None or len(data) == 0:
            print(f"[WARN] No data after preprocessing for {project_name}, skipping.")
            continue

        X = data.drop(columns=["build_duration"])
        # Use workflow_id only for splitting, not as a feature
        feature_cols = [c for c in X.columns if c != "workflow_id"]
        y = data["build_duration"]

        # --- Baseline for this project ---
        # --- Baselines for this project (same folds as model evaluation) ---
        base_results_lag1 = compute_baseline_last_k_expanding_folds(X, y, k_roll=1)
        base_results_last7 = compute_baseline_last_k_expanding_folds(X, y, k_roll=7)

        with open(LOG_FILE, "a") as log_file:
            log_file.write(f"=== BASELINE 1 (project={project_name}, lag-1, expanding folds) ===\n")
            for it in range(N_ITERS):
                rmse = float(np.nanmean(base_results_lag1[it]["rmse"])) if base_results_lag1[it]["rmse"] else float(
                    "nan")
                nrmse = float(np.nanmean(base_results_lag1[it]["nrmse"])) if base_results_lag1[it]["nrmse"] else float(
                    "nan")
                cvrmse = float(np.nanmean(base_results_lag1[it]["cvrmse"])) if base_results_lag1[it][
                    "cvrmse"] else float("nan")
                r2v = float(np.nanmean(base_results_lag1[it]["r2"])) if base_results_lag1[it]["r2"] else float("nan")

                train_end = 5 + it
                test_fold = 6 + it
                log_file.write(
                    f"Iter{it + 1}: train folds 1..{train_end} -> test fold {test_fold}  "
                    f"RMSE={rmse:.4f}  NRMSE={nrmse:.4f}  CVRMSE={cvrmse:.4f}  R²={r2v:.4f}\n"
                )

            log_file.write(f"\n=== BASELINE 2 (project={project_name}, last-7 mean, expanding folds) ===\n")
            for it in range(N_ITERS):
                rmse = float(np.nanmean(base_results_last7[it]["rmse"])) if base_results_last7[it]["rmse"] else float(
                    "nan")
                nrmse = float(np.nanmean(base_results_last7[it]["nrmse"])) if base_results_last7[it][
                    "nrmse"] else float("nan")
                cvrmse = float(np.nanmean(base_results_last7[it]["cvrmse"])) if base_results_last7[it][
                    "cvrmse"] else float("nan")
                r2v = float(np.nanmean(base_results_last7[it]["r2"])) if base_results_last7[it]["r2"] else float("nan")

                train_end = 5 + it
                test_fold = 6 + it
                log_file.write(
                    f"Iter{it + 1}: train folds 1..{train_end} -> test fold {test_fold}  "
                    f"RMSE={rmse:.4f}  NRMSE={nrmse:.4f}  CVRMSE={cvrmse:.4f}  R²={r2v:.4f}\n"
                )
            log_file.write("-" * 50 + "\n")

        # --- GA for this project (will log into the same LOG_FILE) ---
        # 1) correlation + redundancy screening (done once, right after cleaning)
        kept_features = screen_features_first_window_only(
            data,
            feature_cols,
            y_col="build_duration",
            corr_thr=CORR_THR,
            red_thr=REDUNDANCY_R2_THR,
            always_keep=REQUIRED_FEATURES,
        )

        with open(LOG_FILE, "a") as f:
            f.write(f"[FEATURES] initial={len(feature_cols)} kept_after_screening={len(kept_features)}\n")
            f.write(str(kept_features) + "\n")
            f.write("-" * 80 + "\n")


        evaluate_expanding_window_retuned_ga(X, y, kept_features)

    plt.ioff()
    plt.show()
