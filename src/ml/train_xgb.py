"""
Robust XGBoost baseline for popularity prediction (flickr / digg / higgs / ...).

Examples
--------
# one dataset, default hyperparameters
python train_xgb.py --dataset_name flickr

# all three datasets with the same hyperparameters
python train_xgb.py --dataset_name flickr digg higgs --max_depth 4 --learning_rate 0.05 \
    --n_estimators 2000 --early_stopping_rounds 50 --subsample 0.8 --colsample_bytree 0.8

# random hyper-parameter search (picked on validation only), 30 trials per dataset
python train_xgb.py --dataset_name digg --tune_trials 30

Requires xgboost >= 2.0, scikit-learn, scipy, pandas, pyarrow.
"""
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import ParameterSampler

log = logging.getLogger("train_xgb")

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
DEFAULT_BASE_PATH = "."
DEFAULT_DATASETS = ["flickr"]
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_SEED = 42

HP_DEFAULTS = dict(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.05,
    min_child_weight=1.0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    gamma=0.0,
)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def find_latest_hash_dir(samples_dir: Path) -> str:
    """Most recently finished feature-hash directory (has index.parquet)."""
    if not samples_dir.exists():
        raise FileNotFoundError(f"Missing samples directory {samples_dir}. Run features.py first.")
    subdirs = [d for d in samples_dir.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No hashed feature directories found in {samples_dir}")
    finished = [d for d in subdirs if (d / "index.parquet").exists()]
    if finished:
        latest = max(finished, key=lambda d: (d / "index.parquet").stat().st_mtime)
    else:
        latest = max(subdirs, key=lambda d: d.stat().st_mtime)
    return latest.name


def load_dataset(args, dataset: str):
    base_dir = Path(args.base_path)
    samples_dir = base_dir / args.output_dir / dataset / "samples"
    splits_dir = base_dir / "data" / "splits" / dataset

    config_hash = args.feature_hash or find_latest_hash_dir(samples_dir)
    features_dir = samples_dir / config_hash
    log.info(f"[{dataset}] features dir: {features_dir}")

    features_df = pd.read_parquet(features_dir / "tabular_features.parquet")
    labels_df = pd.read_parquet(splits_dir / "labels.parquet")
    meta_df = pd.read_parquet(splits_dir / "split_meta.parquet")

    # Guard against duplicate ids (would silently blow up the merge)
    for name, d in [("features", features_df), ("labels", labels_df), ("split_meta", meta_df)]:
        if d["post_id"].duplicated().any():
            raise ValueError(f"[{dataset}] duplicate post_id in {name}")

    n_feat = len(features_df)
    df = features_df.merge(labels_df[["post_id", "label"]], on="post_id", how="inner") \
                    .merge(meta_df[["post_id", "split"]], on="post_id", how="inner")
    if len(df) < n_feat:
        log.warning(f"[{dataset}] {n_feat - len(df)} of {n_feat} feature rows had no label/split and were dropped")
    if df.empty:
        raise ValueError(f"[{dataset}] merge produced 0 rows; check post_id dtype/values")

    # Feature columns: numeric / bool only, no id / label / split
    exclude = {"post_id", "label", "split", "target"}
    cand = [c for c in features_df.columns if c not in exclude]
    feature_cols = [c for c in cand if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])]
    skipped = sorted(set(cand) - set(feature_cols))
    if skipped:
        log.warning(f"[{dataset}] skipping non-numeric columns: {skipped}")

    df[feature_cols] = df[feature_cols].astype("float32").replace([np.inf, -np.inf], np.nan)

    # Drop columns that carry no information (all NaN / constant on train)
    train_mask = df["split"] == "train"
    drop = [c for c in feature_cols if df.loc[train_mask, c].nunique(dropna=True) <= 1]
    if drop:
        log.warning(f"[{dataset}] dropping {len(drop)} constant/all-NaN columns on train: {drop[:10]}{'...' if len(drop) > 10 else ''}")
        feature_cols = [c for c in feature_cols if c not in drop]
    if not feature_cols:
        raise ValueError(f"[{dataset}] no usable feature columns")

    # Target: log1p(count). Counts can't be negative, so clip for safety.
    lab = df["label"].astype("float64")
    if (lab < 0).any():
        log.warning(f"[{dataset}] {(lab < 0).sum()} negative labels clipped to 0")
    if lab.isna().any():
        log.warning(f"[{dataset}] {lab.isna().sum()} NaN labels dropped")
        df = df[lab.notna()].copy()
        lab = df["label"].astype("float64")
    df["target"] = np.log1p(lab.clip(lower=0))

    return df, feature_cols


def split_frames(df, feature_cols, dataset):
    out = {}
    for s in ("train", "val", "test"):
        sub = df[df["split"] == s]
        if sub.empty:
            raise ValueError(f"[{dataset}] split '{s}' is empty")
        out[s] = (sub[feature_cols], sub["target"].to_numpy(), sub["post_id"].to_numpy())
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def evaluate(y_log, p_log):
    """y_log, p_log are on log1p scale."""
    p_log = np.clip(p_log, 0, None)  # counts can't be negative
    msle = mean_squared_error(y_log, p_log)
    raw_true, raw_pred = np.expm1(y_log), np.expm1(p_log)
    sp = spearmanr(y_log, p_log).correlation if np.std(p_log) > 0 and np.std(y_log) > 0 else float("nan")
    return {
        "msle": float(msle),
        "rmsle": float(np.sqrt(msle)),
        "mae": float(mean_absolute_error(raw_true, raw_pred)),
        "medae": float(median_absolute_error(raw_true, raw_pred)),
        "r2_log": float(r2_score(y_log, p_log)),
        "spearman": float(sp),
    }


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(hp: dict, args):
    es = args.early_stopping_rounds if args.early_stopping_rounds > 0 else None
    return xgb.XGBRegressor(
        objective=args.objective,
        eval_metric=args.eval_metric,
        tree_method="hist",
        device=args.device,
        n_jobs=args.n_jobs,
        random_state=args.seed,
        early_stopping_rounds=es,
        **hp,
    )


def fit(hp, args, data):
    X_tr, y_tr, _ = data["train"]
    X_va, y_va, _ = data["val"]
    model = build_model(hp, args)
    model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_va, y_va)], verbose=False)
    return model


def sample_search_space(n_trials, seed):
    space = {
        "max_depth": [2, 3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.02, 0.05, 0.1],
        "min_child_weight": [1, 3, 5, 10, 20],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0],
        "reg_alpha": [0, 0.01, 0.1, 1, 10],
        "reg_lambda": [0.1, 1, 5, 10, 50],
        "gamma": [0, 0.1, 0.5, 1, 5],
    }
    return list(ParameterSampler(space, n_iter=n_trials, random_state=seed))


def run_dataset(args, dataset: str):
    t0 = time.time()
    out_dir = Path(args.base_path) / args.output_dir / dataset / "ml"
    out_dir.mkdir(parents=True, exist_ok=True)

    df, feature_cols = load_dataset(args, dataset)
    data = split_frames(df, feature_cols, dataset)
    n = {s: len(data[s][0]) for s in data}
    log.info(f"[{dataset}] samples train/val/test = {n['train']}/{n['val']}/{n['test']} | features = {len(feature_cols)}")

    # Label stats per split (spot distribution shift early)
    for s in ("train", "val", "test"):
        y = np.expm1(data[s][1])
        log.info(f"[{dataset}] {s:5s} label: mean={y.mean():.2f} median={np.median(y):.1f} "
                 f"p90={np.percentile(y, 90):.1f} max={y.max():.0f}")

    # Trivial baseline: predict train mean in log space
    base_pred = np.full_like(data["test"][1], data["train"][1].mean(), dtype=float)
    baseline = evaluate(data["test"][1], base_pred)
    log.info(f"[{dataset}] constant baseline test MSLE = {baseline['msle']:.4f}")

    hp = dict(
        n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.learning_rate,
        min_child_weight=args.min_child_weight, subsample=args.subsample,
        colsample_bytree=args.colsample_bytree, reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda, gamma=args.gamma,
    )

    # Optional random search, selected on VALIDATION only (test is never touched)
    tuning_log = []
    if args.tune_trials > 0:
        best_val, best_hp, best_model = np.inf, None, None
        for i, trial_hp in enumerate(sample_search_space(args.tune_trials, args.seed), 1):
            cur = {**hp, **trial_hp}
            m = fit(cur, args, data)
            v = mean_squared_error(data["val"][1], np.clip(m.predict(data["val"][0]), 0, None))
            tuning_log.append({"trial": i, "val_msle": float(v), **trial_hp})
            log.info(f"[{dataset}] trial {i}/{args.tune_trials} val MSLE={v:.4f} {trial_hp}")
            if v < best_val:
                best_val, best_hp, best_model = v, cur, m
        hp, model = best_hp, best_model
        log.info(f"[{dataset}] best val MSLE={best_val:.4f} with {hp}")
    else:
        model = fit(hp, args, data)

    best_iter = getattr(model, "best_iteration", None)
    log.info(f"[{dataset}] best iteration: {best_iter}")

    # Evaluate (predict uses best iteration automatically when early stopping is on)
    res = {s: evaluate(data[s][1], model.predict(data[s][0])) for s in ("train", "val", "test")}
    for s in ("train", "val", "test"):
        r = res[s]
        log.info(f"[{dataset}] {s:5s} | MSLE {r['msle']:.4f} | RMSLE {r['rmsle']:.4f} | MAE {r['mae']:.2f} | "
                 f"MedAE {r['medae']:.2f} | R2(log) {r['r2_log']:.3f} | Spearman {r['spearman']:.3f}")
    gap = res["test"]["msle"] - res["train"]["msle"]
    if gap > 0.5 * max(res["train"]["msle"], 1e-9) and res["test"]["msle"] > res["val"]["msle"] * 1.3:
        log.warning(f"[{dataset}] large train->test gap / val->test shift: possible overfit or temporal distribution shift")

    # Save everything
    model.save_model(out_dir / "xgboost_best.json")

    imp = model.get_booster().get_score(importance_type="gain")
    imp_df = pd.DataFrame({"feature": feature_cols}).assign(gain=lambda d: d["feature"].map(imp).fillna(0.0)) \
                                                    .sort_values("gain", ascending=False)
    imp_df.to_csv(out_dir / "xgb_feature_importance.csv", index=False)

    test_p = np.clip(model.predict(data["test"][0]), 0, None)
    pd.DataFrame({"post_id": data["test"][2], "y_true_log": data["test"][1], "y_pred_log": test_p,
                  "y_true": np.expm1(data["test"][1]), "y_pred": np.expm1(test_p)}) \
        .to_parquet(out_dir / "xgb_test_predictions.parquet", index=False)

    summary = {
        "dataset": dataset, "n_train": n["train"], "n_val": n["val"], "n_test": n["test"],
        "n_features": len(feature_cols), "best_iteration": best_iter, "hyperparameters": hp,
        "args": {k: v for k, v in vars(args).items()}, "metrics": res,
        "constant_baseline_test": baseline, "tuning": tuning_log,
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(out_dir / "xgb_metrics.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info(f"[{dataset}] saved model/metrics/predictions to {out_dir}")
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Robust XGBoost for popularity prediction (multi-dataset)",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Environment
    p.add_argument("--base_path", type=str, default=DEFAULT_BASE_PATH)
    p.add_argument("--dataset_name", type=str, nargs="+", default=DEFAULT_DATASETS,
                   help="One or more datasets, e.g. flickr digg higgs")
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--feature_hash", type=str, default=None,
                   help="Feature hash dir to use. Default: latest finished one per dataset")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    p.add_argument("--n_jobs", type=int, default=-1)

    # Booster hyperparameters
    p.add_argument("--n_estimators", type=int, default=HP_DEFAULTS["n_estimators"])
    p.add_argument("--max_depth", type=int, default=HP_DEFAULTS["max_depth"])
    p.add_argument("--learning_rate", type=float, default=HP_DEFAULTS["learning_rate"])
    p.add_argument("--min_child_weight", type=float, default=HP_DEFAULTS["min_child_weight"])
    p.add_argument("--subsample", type=float, default=HP_DEFAULTS["subsample"])
    p.add_argument("--colsample_bytree", type=float, default=HP_DEFAULTS["colsample_bytree"])
    p.add_argument("--reg_alpha", type=float, default=HP_DEFAULTS["reg_alpha"], help="L1 regularization")
    p.add_argument("--reg_lambda", type=float, default=HP_DEFAULTS["reg_lambda"], help="L2 regularization")
    p.add_argument("--gamma", type=float, default=HP_DEFAULTS["gamma"], help="Min loss reduction to split")

    # Training control
    p.add_argument("--objective", type=str, default="reg:squarederror",
                   choices=["reg:squarederror", "reg:pseudohubererror", "reg:absoluteerror"])
    p.add_argument("--eval_metric", type=str, default="rmse", help="Early-stopping metric on val (e.g. rmse, mae)")
    p.add_argument("--early_stopping_rounds", type=int, default=50, help="0 disables early stopping")
    p.add_argument("--tune_trials", type=int, default=0,
                   help="Random-search trials selected on validation (0 = no tuning)")
    return p.parse_args()


def main():
    args = parse_args()
    results, failed = [], []
    for ds in args.dataset_name:
        try:
            results.append(run_dataset(args, ds))
        except Exception as e:  # keep going so one bad dataset doesn't kill the whole run
            log.exception(f"[{ds}] FAILED: {e}")
            failed.append(ds)

    if results:
        rows = []
        for r in results:
            m = r["metrics"]
            rows.append({"dataset": r["dataset"], "val_msle": m["val"]["msle"], "test_msle": m["test"]["msle"],
                         "test_rmsle": m["test"]["rmsle"], "test_mae": m["test"]["mae"],
                         "test_r2_log": m["test"]["r2_log"], "test_spearman": m["test"]["spearman"],
                         "baseline_test_msle": r["constant_baseline_test"]["msle"],
                         "best_iter": r["best_iteration"]})
        table = pd.DataFrame(rows)
        print("\n=== FINAL XGBOOST SUMMARY ===")
        print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("=============================\n")
        summ_path = Path(args.base_path) / args.output_dir / "xgb_summary.csv"
        summ_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(summ_path, index=False)
        log.info(f"Summary saved to {summ_path}")
    if failed:
        log.error(f"Failed datasets: {failed}")
        raise SystemExit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()