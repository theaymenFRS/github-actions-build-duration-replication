import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "data").is_dir() and (p / "scripts").is_dir():
            return p
    raise RuntimeError("Could not locate repo root (expected 'data/' and 'scripts/' dirs).")


REPO_ROOT = _find_repo_root()
DATA_RAW_DIR = REPO_ROOT / "data" / "raw"
MAX_DURATION_SEC = 300 * 24 * 60 * 60

N_FOLDS = 10
N_ITERS = 5

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


def preprocess_for_baseline(file_path):
    df = pd.read_csv(file_path)

    if "conclusion" in df.columns:
        df = df[df["conclusion"] == "success"].copy()

    if "workflow_event_trigger" in df.columns:
        df = df[df["workflow_event_trigger"].isin(["push", "pull_request"])].copy()

    df["build_duration"] = pd.to_numeric(df["build_duration"], errors="coerce")
    df = df.dropna(subset=["build_duration"]).copy()

    df = df[df["build_duration"] <= MAX_DURATION_SEC].copy()

    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        df = df.dropna(subset=["created_at"]).sort_values("created_at").reset_index(drop=True)

    if "workflow_id" not in df.columns:
        df["workflow_id"] = 0

    workflow_counts = df["workflow_id"].value_counts()
    valid_workflows = workflow_counts[workflow_counts >= 100].index.tolist()
    if valid_workflows:
        df = df[df["workflow_id"].isin(valid_workflows)].copy()

    return df.reset_index(drop=True)


def make_folds(n, n_folds=N_FOLDS):
    if n < n_folds:
        return None
    idx = np.arange(n)
    folds = np.array_split(idx, n_folds)
    if any(len(f) == 0 for f in folds):
        return None
    return folds


def expanding_train_test_indices(folds, k):
    train_idx = np.concatenate([folds[i] for i in range(0, 5 + k)])
    test_idx = folds[5 + k]
    return train_idx, test_idx


def compute_lag1_baseline_expanding_folds(df):
    results = {it: {"rmse": [], "r2": [], "nrmse": [], "cvrmse": []} for it in range(N_ITERS)}

    for wf_id in df["workflow_id"].unique():
        yw = df.loc[df["workflow_id"] == wf_id, "build_duration"].reset_index(drop=True)

        folds = make_folds(len(yw), N_FOLDS)
        if folds is None:
            continue

        baseline = yw.shift(1)  # lag-1

        for it in range(N_ITERS):
            _, test_idx = expanding_train_test_indices(folds, it)

            y_te = yw.iloc[test_idx]
            y_pred = baseline.iloc[test_idx]

            valid = y_pred.notna()
            y_te = y_te[valid]
            y_pred = y_pred[valid]

            if len(y_te) == 0:
                continue

            rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
            r2v = float(r2_score(y_te, y_pred)) if len(y_te) > 1 else float("nan")

            sd = float(np.std(y_te, ddof=1)) if len(y_te) > 1 else 0.0
            mean = float(np.mean(y_te)) if len(y_te) > 0 else 0.0
            nrmse = rmse / sd if sd > 0 else float("nan")
            cvrmse = rmse / mean if mean > 0 else float("nan")

            results[it]["rmse"].append(rmse)
            results[it]["r2"].append(r2v)
            results[it]["nrmse"].append(nrmse)
            results[it]["cvrmse"].append(cvrmse)

    return results


def write_lag1_log(log_file, project_name, results):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"=== BASELINE 1 (project={project_name}, lag-1, expanding folds) ===\n")
        for it in range(N_ITERS):
            rmse = float(np.nanmean(results[it]["rmse"])) if results[it]["rmse"] else float("nan")
            nrmse = float(np.nanmean(results[it]["nrmse"])) if results[it]["nrmse"] else float("nan")
            cvrmse = float(np.nanmean(results[it]["cvrmse"])) if results[it]["cvrmse"] else float("nan")
            r2v = float(np.nanmean(results[it]["r2"])) if results[it]["r2"] else float("nan")

            train_end = 5 + it
            test_fold = 6 + it

            f.write(
                f"Iter{it + 1}: train folds 1..{train_end} -> test fold {test_fold}  "
                f"RMSE={rmse:.4f}  NRMSE={nrmse:.4f}  CVRMSE={cvrmse:.4f}  R²={r2v:.4f}\n"
            )


if __name__ == "__main__":
    # Aligné avec results/rq1/ (voir results/README.md et scripts/README.md)
    baseline_root = REPO_ROOT / "results" / "rq1" / "baseline_lag1"

    for project_name, file_path in PROJECTS:
        print("\n" + "=" * 80)
        print(f"PROJECT: {project_name}")
        print(f"CSV    : {file_path}")
        print("=" * 80)

        if not os.path.exists(file_path):
            print("[WARN] File not found, skipping.")
            continue

        df = preprocess_for_baseline(file_path)
        if df.empty:
            print("[WARN] No data after preprocessing, skipping.")
            continue

        results = compute_lag1_baseline_expanding_folds(df)

        out_dir = baseline_root / project_name
        out_dir.mkdir(parents=True, exist_ok=True)
        log_file = out_dir / "baseline_lag1.log"
        if log_file.exists():
            log_file.unlink()

        write_lag1_log(log_file, project_name, results)
        print(f"[SAVED] {log_file}")