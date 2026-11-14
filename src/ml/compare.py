from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from torch_geometric.data import Batch

try:
    from src.ml.dataset_ram import get_dataloaders_ram as get_dataloaders
except ImportError:
    from src.ml.dataset import get_dataloaders

from src.ml.models.hgt import (
    DEFAULT_METADATA,
    NUM_HEADS,
    PopularityHGT,
    get_batch_dict,
)
from src.ml.models.rgcn import PopularityRGCN, infer_schema
from src.ml.train_xgb import find_latest_hash_dir

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _extract_state_dict(ckpt: dict | object) -> dict:
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def _get_post_id(g, fallback_idx: int) -> int:
    for obj in (g, getattr(g, "post", None)):
        if obj is not None and hasattr(obj, "post_id"):
            val = getattr(obj, "post_id")
            if isinstance(val, torch.Tensor):
                return int(val.view(-1)[0].item())
            return int(val)
    return int(fallback_idx)


def _infer_hgt_kwargs(
    state_dict: dict,
    ckpt_path: Path,
    default_conv: str,
    sample_metadata: tuple,
) -> dict:
    """Load exact hparams from <run>_config.json if available, else infer from state_dict."""
    cfg_path = ckpt_path.with_name(ckpt_path.stem.replace("_best", "_config") + ".json")
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            saved = json.load(f).get("hparams", {})
        if saved:
            metadata = DEFAULT_METADATA if saved.get("schema") == "fixed" else sample_metadata
            return {
                "metadata": metadata,
                "hidden_channels": saved.get("hidden_channels", 64),
                "num_heads": saved.get("num_heads", NUM_HEADS),
                "num_layers": saved.get("num_layers", 2),
                "dropout": saved.get("dropout", 0.2),
                "conv": saved.get("conv", default_conv),
                "num_bases": saved.get("num_bases"),
                "norm": saved.get("norm", "layer"),
                "act": saved.get("act", "gelu"),
                "residual": saved.get("residual", True),
                "readout": saved.get("readout", "post"),
                "head_layers": saved.get("head_layers", 2),
                "head_hidden": saved.get("head_hidden"),
                "output_act": saved.get("output_act", "none"),
            }

    hidden = next(
        (v.shape[0] for k, v in state_dict.items() if k.startswith("lin_dict.") and k.endswith(".weight")),
        64,
    )
    layers = len({k.split(".")[1] for k in state_dict if k.startswith("convs.")}) or 2
    conv = "rgcn" if any(k.endswith((".weight", ".bases", ".comp")) and "convs." in k and "self_lin" in k for k in state_dict) else default_conv

    num_heads = NUM_HEADS
    for k, v in state_dict.items():
        if k.startswith("convs.0.") and k.endswith(("p_rel", "rel_pri")) and v.ndim == 2:
            num_heads = int(v.shape[1])
            break

    num_bases = state_dict["convs.0.bases"].shape[0] if "convs.0.bases" in state_dict else None

    if "attn_gate.weight" in state_dict:
        readout = "post+user_attn"
    elif "mlp.0.weight" in state_dict and state_dict["mlp.0.weight"].shape[1] == 2 * hidden:
        readout = "post+user_mean"
    else:
        readout = "post"

    mlp_linear_keys = sorted(k for k in state_dict if k.startswith("mlp.") and k.endswith(".weight"))
    head_layers = len(mlp_linear_keys) or 2
    head_hidden = int(state_dict[mlp_linear_keys[0]].shape[0]) if head_layers > 1 else None

    return {
        "metadata": DEFAULT_METADATA if len(sample_metadata[1]) < len(DEFAULT_METADATA[1]) else sample_metadata,
        "hidden_channels": int(hidden),
        "num_heads": num_heads,
        "num_layers": layers,
        "conv": conv,
        "num_bases": num_bases,
        "readout": readout,
        "head_layers": head_layers,
        "head_hidden": head_hidden,
    }


def _load_gnn_model(ckpt_path: Path, default_conv: str, sample_batch: Batch, device: torch.device):
    """Load either PopularityHGT (hgt/rgcn backbone) or standalone PopularityRGCN."""
    if not ckpt_path.exists():
        log.warning("Checkpoint not found at %s. Skipping.", ckpt_path)
        return None, None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = _extract_state_dict(ckpt)

    # Case 1: Standalone PopularityRGCN from rgcn.py (uses 'encoders.<nt>.*')
    if any(k.startswith("encoders.") for k in state_dict):
        schema = infer_schema([sample_batch])
        hidden = state_dict["encoders.post.0.weight"].shape[0]
        layers = len({k.split(".")[1] for k in state_dict if k.startswith("convs.")}) or 2
        post_in_weight_dim = state_dict["encoders.post.0.weight"].shape[1]
        use_degree = post_in_weight_dim > schema["in_channels"]["post"]
        readout_dim = state_dict["head_hidden.0.weight"].shape[0]
        use_pooling = readout_dim > hidden

        model = PopularityRGCN(
            metadata=(schema["node_types"], schema["edge_types"]),
            in_channels=schema["in_channels"],
            hidden_channels=int(hidden),
            num_layers=layers,
            use_degree_features=use_degree,
            use_pooling=use_pooling,
        ).to(device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model, ckpt

    # Case 2: PopularityHGT from hgt.py / train.py (supports both conv='hgt' and conv='rgcn')
    kwargs = _infer_hgt_kwargs(state_dict, ckpt_path, default_conv, sample_batch.metadata())
    model = PopularityHGT(**kwargs).to(device)
    model.eval()
    with torch.no_grad():
        _ = model(sample_batch.x_dict, sample_batch.edge_index_dict, get_batch_dict(sample_batch))
    model.load_state_dict(state_dict, strict=True)
    return model, ckpt


@torch.no_grad()
def _predict_gnn(model, ckpt, batch: Batch, clamp_max_log: float = 20.0) -> list[str] | np.ndarray:
    if model is None:
        return ["N/A"] * batch.num_graphs
    out = model(batch.x_dict, batch.edge_index_dict, get_batch_dict(batch))
    if isinstance(ckpt, dict) and "target_mean" in ckpt and "target_std" in ckpt:
        out = out * ckpt["target_std"] + ckpt["target_mean"]
    out = out.view(-1).clamp(0.0, clamp_max_log)
    return np.expm1(out.detach().cpu().numpy())


def _predict_xgb(
    posts: list,
    post_ids: list[int],
    base_dir: Path,
    output_dir: str,
    dataset_name: str,
    config_hash: str | None = None,
) -> list[str] | np.ndarray:
    ml_dir = base_dir / output_dir / dataset_name / "ml"
    pred_parquet = ml_dir / "xgb_test_predictions.parquet"

    # 1. Use precomputed test predictions from train_xgb.py if all post_ids match
    if pred_parquet.exists():
        pred_df = pd.read_parquet(pred_parquet)
        if "post_id" in pred_df.columns and "y_pred" in pred_df.columns:
            lookup = dict(zip(pred_df["post_id"].astype(int), pred_df["y_pred"].astype(float)))
            if all(pid in lookup for pid in post_ids):
                return np.array([lookup[pid] for pid in post_ids], dtype=float)

    # 2. Run booster on tabular_features.parquet (exact features used in train_xgb.py)
    xgb_path = ml_dir / "xgboost_best.json"
    if not xgb_path.exists():
        log.warning("XGBoost model not found at %s. Skipping.", xgb_path)
        return ["N/A"] * len(posts)

    try:
        booster = xgb.Booster()
        booster.load_model(str(xgb_path))
        samples_dir = base_dir / output_dir / dataset_name / "samples"
        h = config_hash or find_latest_hash_dir(samples_dir)
        tab_path = samples_dir / h / "tabular_features.parquet"

        if tab_path.exists():
            tab_df = pd.read_parquet(tab_path).set_index("post_id")
            if all(pid in tab_df.index for pid in post_ids):
                sub = tab_df.loc[post_ids]
                feat_names = booster.feature_names
                if feat_names:
                    X = sub[feat_names].astype("float32").replace([np.inf, -np.inf], np.nan)
                    dmat = xgb.DMatrix(X, feature_names=feat_names)
                else:
                    exclude = {"label", "split", "target"}
                    cols = [c for c in sub.columns if c not in exclude]
                    X = sub[cols].astype("float32").replace([np.inf, -np.inf], np.nan)
                    dmat = xgb.DMatrix(X)
                preds_log = np.clip(booster.predict(dmat), 0.0, None)
                return np.expm1(preds_log)

        # Fallback: raw post node features
        X_raw = np.vstack([g["post"].x.detach().cpu().numpy().reshape(1, -1) for g in posts])
        preds_log = np.clip(booster.predict(xgb.DMatrix(X_raw), validate_features=False), 0.0, None)
        return np.expm1(preds_log)
    except Exception as e:
        log.warning("XGBoost inference failed: %s", e)
        return ["N/A"] * len(posts)


def _build_comparison_df(
    posts: list,
    post_ids: list[int],
    xgb_preds,
    rgcn_preds,
    hgt_preds,
) -> pd.DataFrame:
    ground_truth = [
        int(np.round(np.expm1(float(g.y.view(-1)[0].item()))))
        for g in posts
    ]

    def _fmt(val):
        return f"{float(val):.1f}" if isinstance(val, (int, float, np.floating)) else str(val)

    rows = []
    for i in range(len(posts)):
        rows.append({
            "Post ID": post_ids[i],
            "Ground Truth": ground_truth[i],
            "XGBoost": _fmt(xgb_preds[i]),
            "R-GCN": _fmt(rgcn_preds[i]),
            "HGT": _fmt(hgt_preds[i]),
        })
    return pd.DataFrame(rows)


def compare_models(
    base_path: str = ".",
    dataset_name: str = "flickr",
    output_dir: str = "outputs",
    config_hash: str | None = None,
    k: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = Path(base_path)
    ml_dir = base / output_dir / dataset_name / "ml"

    log.info("Loading test split for dataset '%s'...", dataset_name)
    _, _, test_loader = get_dataloaders(
        base_path=base,
        dataset_name=dataset_name,
        config_hash=config_hash,
        batch_size=32,
        output_dir=output_dir,
    )

    all_test_graphs = []
    for b in test_loader:
        all_test_graphs.extend(b.to_data_list())

    if not all_test_graphs:
        raise ValueError("Test loader is empty!")

    indexed_graphs = list(enumerate(all_test_graphs))

    # 1. Select Top-K most popular posts (sorted by ground-truth y descending)
    sorted_by_pop = sorted(
        indexed_graphs,
        key=lambda pair: float(pair[1].y.view(-1)[0].item()),
        reverse=True,
    )
    top_pairs = sorted_by_pop[:k]
    top_indices = {idx for idx, _ in top_pairs}

    # 2. Select Random-K posts from the remaining test posts (or full test set if < 2*k)
    rng = random.Random(seed)
    remaining_pairs = [pair for pair in indexed_graphs if pair[0] not in top_indices]
    pool = remaining_pairs if len(remaining_pairs) >= k else indexed_graphs
    random_pairs = rng.sample(pool, k=min(k, len(pool)))

    # Load R-GCN and HGT models once using a sample batch for schema/lazy-layer init
    sample_batch = Batch.from_data_list(all_test_graphs[: min(32, len(all_test_graphs))]).to(device)

    rgcn_candidates = [
        ml_dir / f"rgcn_{dataset_name}_best.pt",
        ml_dir / "rgcn_best.pt",
    ]
    rgcn_path = next((p for p in rgcn_candidates if p.exists()), rgcn_candidates[-1])
    rgcn_model, rgcn_ckpt = _load_gnn_model(rgcn_path, "rgcn", sample_batch, device)

    hgt_candidates = [
        ml_dir / f"hgt_{dataset_name}_best.pt",
        ml_dir / f"hgt__{dataset_name}_best.pt",
        ml_dir / "hgt_best.pt",
    ]
    hgt_path = next((p for p in hgt_candidates if p.exists()), hgt_candidates[-1])
    hgt_model, hgt_ckpt = _load_gnn_model(hgt_path, "hgt", sample_batch, device)

    def _evaluate_subset(pairs: list[tuple[int, object]]) -> pd.DataFrame:
        posts = [g for _, g in pairs]
        post_ids = [_get_post_id(g, idx) for idx, g in pairs]
        sub_batch = Batch.from_data_list(posts).to(device)

        xgb_preds = _predict_xgb(posts, post_ids, base, output_dir, dataset_name, config_hash)
        rgcn_preds = _predict_gnn(rgcn_model, rgcn_ckpt, sub_batch)
        hgt_preds = _predict_gnn(hgt_model, hgt_ckpt, sub_batch)
        return _build_comparison_df(posts, post_ids, xgb_preds, rgcn_preds, hgt_preds)

    df_top = _evaluate_subset(top_pairs)
    df_rand = _evaluate_subset(random_pairs)

    print(f"\n=== TABLE 1: TOP {len(df_top)} POPULAR TEST POSTS ({dataset_name.upper()}) ===")
    print(df_top.to_string(index=False))

    print(f"\n=== TABLE 2: RANDOM {len(df_rand)} TEST POSTS ({dataset_name.upper()}, seed={seed}) ===")
    print(df_rand.to_string(index=False))
    print()

    return df_top, df_rand


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare XGBoost, R-GCN, and HGT on Top-K and Random-K test posts.")
    parser.add_argument("--base-path", "--base_path", dest="base_path", default=".")
    parser.add_argument("--dataset", default="flickr")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--config-hash", default=None)
    parser.add_argument("--top-k", "-k", dest="k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    compare_models(
        base_path=args.base_path,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        config_hash=args.config_hash,
        k=args.k,
        seed=args.seed,
    )