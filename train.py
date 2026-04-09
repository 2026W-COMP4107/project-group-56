"""
PyTorch training harness for tactical intent prediction.

Trains and evaluates:
  - MLP          (static features)
  - LSTM         (sequence features)
  - GRU          (sequence features)
  - Transformer  (sequence features)

Features:
- AdamW optimizer
- Stratified train / validation / test split
- Leave-one-map-out cross-map validation
- Early stopping on validation loss
- Class-weighted CrossEntropyLoss
- Full evaluation metrics:
    accuracy, balanced accuracy, macro precision/recall/F1,
    confusion matrix, top-2 accuracy,
    early-round prediction accuracy at 5s / 10s / 15s / ...
- Per-map test metrics
- Results saved to results/training_results.json
- Cross-map results saved to results/cross_map_results.json
- Best model checkpoints saved to results/

Usage
    # Full pipeline from raw DB:
    python train.py --full-pipeline

    # Train only (features must already exist):
    python train.py

    # Quick smoke test:
    python train.py --epochs 5 --hidden 64

    # Skip cross-map validation:
    python train.py --no-cross-map
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sqlite3
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import label_rounds
import features as feat_mod
from models import MLPClassifier, LSTMClassifier, GRUClassifier, TransformerClassifier


RESULTS_DIR = Path("results")
FEATURES_DIR = Path("features")
DB_PATH = Path("valorant.db")

LABEL_NAMES = {
    0: "A-Site Rush",
    1: "B-Site Rush",
    2: "A Split",
    3: "B Split",
    4: "Default",
    5: "Slow Execute",
}
N_CLASSES = len(LABEL_NAMES)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = len(y) / (n_classes * counts)
    weights = weights / weights.sum() * n_classes
    return torch.tensor(weights, dtype=torch.float32)


def _safe_round(x: float) -> float:
    return round(float(x), 4)


def compute_metrics(y_true: np.ndarray, logits: np.ndarray) -> dict:
    y_pred = logits.argmax(axis=-1)
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    top2_idx = np.argsort(logits, axis=-1)[:, -2:]
    top2_acc = float(np.mean([y_true[i] in top2_idx[i] for i in range(len(y_true))]))

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(N_CLASSES)),
        target_names=list(LABEL_NAMES.values()),
        output_dict=True,
        zero_division=0,
    )

    macro = report.get("macro avg", {})
    weighted = report.get("weighted avg", {})
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES))).tolist()

    return {
        "accuracy": _safe_round(acc),
        "balanced_accuracy": _safe_round(bal_acc),
        "top2_accuracy": _safe_round(top2_acc),
        "macro_precision": _safe_round(macro.get("precision", 0)),
        "macro_recall": _safe_round(macro.get("recall", 0)),
        "macro_f1": _safe_round(macro.get("f1-score", 0)),
        "weighted_f1": _safe_round(weighted.get("f1-score", 0)),
        "confusion_matrix": cm,
        "per_class": {
            LABEL_NAMES[i]: {
                "precision": _safe_round(report.get(LABEL_NAMES[i], {}).get("precision", 0)),
                "recall": _safe_round(report.get(LABEL_NAMES[i], {}).get("recall", 0)),
                "f1": _safe_round(report.get(LABEL_NAMES[i], {}).get("f1-score", 0)),
                "support": int(report.get(LABEL_NAMES[i], {}).get("support", 0)),
            }
            for i in range(N_CLASSES)
        },
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_logits = []
    all_y = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)

        batch_size = xb.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

        all_logits.append(logits.detach().cpu().numpy())
        all_y.append(yb.detach().cpu().numpy())

    avg_loss = total_loss / max(1, total_count)
    y_true = np.concatenate(all_y, axis=0)
    logits_np = np.concatenate(all_logits, axis=0)
    return avg_loss, y_true, logits_np


@torch.no_grad()
def early_round_accuracy(
    model: nn.Module,
    X_seq_eval: np.ndarray,
    y_eval: np.ndarray,
    steps_ms: list[int],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    results = {}

    for k, t_ms in enumerate(steps_ms, start=1):
        padded = np.zeros_like(X_seq_eval, dtype=np.float32)
        padded[:, :k, :] = X_seq_eval[:, :k, :]

        xb = torch.tensor(padded, dtype=torch.float32, device=device)
        logits = model(xb)
        preds = logits.argmax(dim=-1).cpu().numpy()
        acc = float((preds == y_eval).mean())
        results[f"{t_ms // 1000}s"] = _safe_round(acc)

    return results


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def standardize_splits(
    X_s_tr: np.ndarray,
    X_s_val: np.ndarray,
    X_s_te: np.ndarray,
    X_q_tr: np.ndarray,
    X_q_val: np.ndarray,
    X_q_te: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    static_scaler = StandardScaler()
    X_s_tr = static_scaler.fit_transform(X_s_tr).astype(np.float32)
    X_s_val = static_scaler.transform(X_s_val).astype(np.float32)
    X_s_te = static_scaler.transform(X_s_te).astype(np.float32)

    N_tr, T, F = X_q_tr.shape
    seq_scaler = StandardScaler()
    X_q_tr_2d = seq_scaler.fit_transform(X_q_tr.reshape(-1, F)).astype(np.float32)
    X_q_tr = X_q_tr_2d.reshape(N_tr, T, F)

    X_q_val_2d = seq_scaler.transform(X_q_val.reshape(-1, F)).astype(np.float32)
    X_q_val = X_q_val_2d.reshape(X_q_val.shape[0], T, F)

    X_q_te_2d = seq_scaler.transform(X_q_te.reshape(-1, F)).astype(np.float32)
    X_q_te = X_q_te_2d.reshape(X_q_te.shape[0], T, F)

    return X_s_tr, X_s_val, X_s_te, X_q_tr, X_q_val, X_q_te


def load_round_maps(round_ids: list[str], db_path: Path) -> list[str]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    placeholders = ",".join("?" for _ in round_ids)
    q = f"""
        SELECT r.round_id, m.map
        FROM rounds r
        JOIN matches m ON r.match_id = m.match_id
        WHERE r.round_id IN ({placeholders})
    """
    rows = cur.execute(q, round_ids).fetchall()
    conn.close()

    round_to_map = {rid: map_name for rid, map_name in rows}
    return [round_to_map.get(rid, "Unknown") for rid in round_ids]


def split_train_val_test(
    X_static: np.ndarray,
    X_seq: np.ndarray,
    y: np.ndarray,
    maps: np.ndarray,
    seed: int,
    val_size: float,
    test_size: float,
) -> dict:
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be < 1.0")

    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(sss_test.split(X_static, y))

    X_s_trainval, X_s_te = X_static[trainval_idx], X_static[test_idx]
    X_q_trainval, X_q_te = X_seq[trainval_idx], X_seq[test_idx]
    y_trainval, y_te = y[trainval_idx], y[test_idx]
    maps_trainval, maps_te = maps[trainval_idx], maps[test_idx]

    val_relative = val_size / (1.0 - test_size)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_relative, random_state=seed)
    tr_rel_idx, val_rel_idx = next(sss_val.split(X_s_trainval, y_trainval))

    X_s_tr, X_s_val = X_s_trainval[tr_rel_idx], X_s_trainval[val_rel_idx]
    X_q_tr, X_q_val = X_q_trainval[tr_rel_idx], X_q_trainval[val_rel_idx]
    y_tr, y_val = y_trainval[tr_rel_idx], y_trainval[val_rel_idx]
    maps_tr, maps_val = maps_trainval[tr_rel_idx], maps_trainval[val_rel_idx]

    X_s_tr, X_s_val, X_s_te, X_q_tr, X_q_val, X_q_te = standardize_splits(
        X_s_tr, X_s_val, X_s_te, X_q_tr, X_q_val, X_q_te
    )

    return {
        "X_s_tr": X_s_tr,
        "X_s_val": X_s_val,
        "X_s_te": X_s_te,
        "X_q_tr": X_q_tr,
        "X_q_val": X_q_val,
        "X_q_te": X_q_te,
        "y_tr": y_tr,
        "y_val": y_val,
        "y_te": y_te,
        "maps_tr": maps_tr,
        "maps_val": maps_val,
        "maps_te": maps_te,
        "train_idx": trainval_idx[tr_rel_idx],
        "val_idx": trainval_idx[val_rel_idx],
        "test_idx": test_idx,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    device: torch.device,
    model_name: str,
    class_weights: torch.Tensor | None = None,
) -> tuple[nn.Module, list[float], list[float], float]:
    model = model.to(device)

    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses = []
    val_losses = []
    best_val = float("inf")
    best_state = None
    no_improve = 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_size = xb.size(0)
            running_loss += loss.item() * batch_size
            total_count += batch_size

        train_loss = running_loss / max(1, total_count)
        val_loss, y_val_np, val_logits_np = evaluate_model(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if (epoch + 1) % max(1, epochs // 10) == 0:
            val_preds = val_logits_np.argmax(axis=-1)
            val_acc = (val_preds == y_val_np).mean()
            print(
                f"  [{model_name}] epoch {epoch+1:>4}/{epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_acc={val_acc:.3f}"
            )

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [{model_name}] Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = time.time() - t0
    return model, train_losses, val_losses, elapsed


def build_model(model_key: str, X_s_tr: np.ndarray, X_q_tr: np.ndarray, args) -> nn.Module:
    if model_key == "MLP":
        hidden_dims = [args.hidden] * args.n_layers
        return MLPClassifier(
            input_dim=X_s_tr.shape[1],
            hidden_dims=hidden_dims,
            n_classes=N_CLASSES,
            dropout=args.dropout,
        )
    if model_key == "LSTM":
        return LSTMClassifier(
            input_dim=X_q_tr.shape[2],
            hidden_dim=args.hidden,
            n_classes=N_CLASSES,
            n_layers=args.rnn_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
        )
    if model_key == "GRU":
        return GRUClassifier(
            input_dim=X_q_tr.shape[2],
            hidden_dim=args.hidden,
            n_classes=N_CLASSES,
            n_layers=args.rnn_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
        )
    if model_key == "Transformer":
        return TransformerClassifier(
            input_dim=X_q_tr.shape[2],
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.tf_layers,
            dropout=args.dropout,
            n_classes=N_CLASSES,
        )
    raise ValueError(f"Unknown model: {model_key}")


def evaluate_named_split(
    model: nn.Module,
    model_key: str,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    maps: np.ndarray | None = None,
    X_seq_eval: np.ndarray | None = None,
    y_eval: np.ndarray | None = None,
) -> dict:
    split_loss, y_true, logits = evaluate_model(model, loader, criterion, device)
    metrics = compute_metrics(y_true, logits)
    metrics["loss"] = _safe_round(split_loss)

    if model_key in {"LSTM", "GRU", "Transformer"} and X_seq_eval is not None and y_eval is not None:
        metrics["early_round_accuracy"] = early_round_accuracy(
            model, X_seq_eval, y_eval, feat_mod.SEQ_STEPS_MS, device
        )

    if maps is not None:
        per_map = {}
        for map_name in sorted(set(maps.tolist())):
            idx = np.where(maps == map_name)[0]
            if len(idx) == 0:
                continue
            per_map[map_name] = compute_metrics(y_true[idx], logits[idx])
            per_map[map_name]["n_samples"] = int(len(idx))
        metrics["per_map"] = per_map

    return metrics

def train_and_evaluate_one_model(
    model_key: str,
    split_data: dict,
    args,
    device: torch.device,
    save_checkpoint: bool = True,
) -> tuple[nn.Module, dict]:
    X_s_tr = split_data["X_s_tr"]
    X_s_val = split_data["X_s_val"]
    X_s_te = split_data["X_s_te"]
    X_q_tr = split_data["X_q_tr"]
    X_q_val = split_data["X_q_val"]
    X_q_te = split_data["X_q_te"]
    y_tr = split_data["y_tr"]
    y_val = split_data["y_val"]
    y_te = split_data["y_te"]

    class_weights = compute_class_weights(y_tr, N_CLASSES)

    if model_key == "MLP":
        train_loader = make_loader(X_s_tr, y_tr, args.batch_size, shuffle=True)
        val_loader = make_loader(X_s_val, y_val, args.batch_size, shuffle=False)
        test_loader = make_loader(X_s_te, y_te, args.batch_size, shuffle=False)
    else:
        train_loader = make_loader(X_q_tr, y_tr, args.batch_size, shuffle=True)
        val_loader = make_loader(X_q_val, y_val, args.batch_size, shuffle=False)
        test_loader = make_loader(X_q_te, y_te, args.batch_size, shuffle=False)

    model = build_model(model_key, X_s_tr, X_q_tr, args)
    print(f"\n=== Training {model_key} ===")

    model, train_losses, val_losses, elapsed = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=device,
        model_name=model_key,
        class_weights=class_weights,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    if model_key == "MLP":
        val_metrics = evaluate_named_split(model, model_key, val_loader, criterion, device, maps=split_data["maps_val"])
        test_metrics = evaluate_named_split(model, model_key, test_loader, criterion, device, maps=split_data["maps_te"])
    else:
        val_metrics = evaluate_named_split(
            model, model_key, val_loader, criterion, device,
            maps=split_data["maps_val"], X_seq_eval=X_q_val, y_eval=y_val,
        )
        test_metrics = evaluate_named_split(
            model, model_key, test_loader, criterion, device,
            maps=split_data["maps_te"], X_seq_eval=X_q_te, y_eval=y_te,
        )

    result = {
        "train_loss_curve": [_safe_round(x) for x in train_losses],
        "val_loss_curve": [_safe_round(x) for x in val_losses],
        "train_time_s": _safe_round(elapsed),
        "class_weights": [_safe_round(x) for x in class_weights.numpy()],
        "validation": val_metrics,
        "test": test_metrics,
    }

    if save_checkpoint:
        torch.save(model.state_dict(), RESULTS_DIR / f"{model_key.lower()}_best.pt")

    print(
        f"  {model_key:<11} "
        f"val_f1={val_metrics['macro_f1']:.4f}  "
        f"test_f1={test_metrics['macro_f1']:.4f}  "
        f"test_acc={test_metrics['accuracy']:.4f}  "
        f"test_top2={test_metrics['top2_accuracy']:.4f}  "
        f"({elapsed:.1f}s)"
    )

    return model, result


def summarize_best_models(all_results: dict) -> dict:
    ranking = []
    for model_name, res in all_results.items():
        ranking.append({
            "model": model_name,
            "test_macro_f1": res["test"]["macro_f1"],
            "test_accuracy": res["test"]["accuracy"],
            "test_top2_accuracy": res["test"]["top2_accuracy"],
            "balanced_accuracy": res["test"]["balanced_accuracy"],
        })

    ranking = sorted(
        ranking,
        key=lambda x: (x["test_macro_f1"], x["test_accuracy"], x["test_top2_accuracy"]),
        reverse=True,
    )

    return {
        "best_by_test_macro_f1": ranking[0]["model"] if ranking else None,
        "ranking": ranking,
    }


def split_cross_map_train_val(
    X_static: np.ndarray,
    X_seq: np.ndarray,
    y: np.ndarray,
    maps: np.ndarray,
    heldout_map: str,
    seed: int,
    val_size: float,
) -> dict | None:
    test_mask = maps == heldout_map
    train_mask = ~test_mask

    if test_mask.sum() == 0 or train_mask.sum() == 0:
        return None

    X_s_trainfull = X_static[train_mask]
    X_q_trainfull = X_seq[train_mask]
    y_trainfull = y[train_mask]
    train_maps = maps[train_mask]

    X_s_te = X_static[test_mask]
    X_q_te = X_seq[test_mask]
    y_te = y[test_mask]
    maps_te = maps[test_mask]

    unique_classes, counts = np.unique(y_trainfull, return_counts=True)
    if len(unique_classes) < 2:
        return None

    min_count = counts.min()
    if min_count >= 2:
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        tr_idx, val_idx = next(sss_val.split(X_s_trainfull, y_trainfull))
    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(y_trainfull))
        cut = max(1, int(round(len(y_trainfull) * (1.0 - val_size))))
        tr_idx, val_idx = perm[:cut], perm[cut:]

    X_s_tr, X_s_val = X_s_trainfull[tr_idx], X_s_trainfull[val_idx]
    X_q_tr, X_q_val = X_q_trainfull[tr_idx], X_q_trainfull[val_idx]
    y_tr, y_val = y_trainfull[tr_idx], y_trainfull[val_idx]
    maps_tr, maps_val = train_maps[tr_idx], train_maps[val_idx]

    X_s_tr, X_s_val, X_s_te, X_q_tr, X_q_val, X_q_te = standardize_splits(
        X_s_tr, X_s_val, X_s_te, X_q_tr, X_q_val, X_q_te
    )

    return {
        "X_s_tr": X_s_tr,
        "X_s_val": X_s_val,
        "X_s_te": X_s_te,
        "X_q_tr": X_q_tr,
        "X_q_val": X_q_val,
        "X_q_te": X_q_te,
        "y_tr": y_tr,
        "y_val": y_val,
        "y_te": y_te,
        "maps_tr": maps_tr,
        "maps_val": maps_val,
        "maps_te": maps_te,
    }


def aggregate_cross_map_results(cross_map_results: dict) -> dict:
    summary = {}
    for model_name, by_map in cross_map_results.items():
        if not isinstance(by_map, dict) or not by_map:
            continue

        accs, macro_f1s, bal_accs, top2s = [], [], [], []
        for _, res in by_map.items():
            test_res = res["test"]
            accs.append(test_res["accuracy"])
            macro_f1s.append(test_res["macro_f1"])
            bal_accs.append(test_res["balanced_accuracy"])
            top2s.append(test_res["top2_accuracy"])

        summary[model_name] = {
            "mean_accuracy": _safe_round(np.mean(accs)),
            "mean_macro_f1": _safe_round(np.mean(macro_f1s)),
            "mean_balanced_accuracy": _safe_round(np.mean(bal_accs)),
            "mean_top2_accuracy": _safe_round(np.mean(top2s)),
        }

    return summary


def run_cross_map_validation(
    X_static: np.ndarray,
    X_seq: np.ndarray,
    y: np.ndarray,
    maps: np.ndarray,
    args,
    device: torch.device,
) -> dict:
    unique_maps = sorted(set(maps.tolist()))
    model_names = ["MLP", "LSTM", "GRU", "Transformer"]
    cross_map_results = {m: {} for m in model_names}

    print("\n=== Cross-map validation (leave-one-map-out) ===")

    for heldout_map in unique_maps:
        split_data = split_cross_map_train_val(
            X_static=X_static,
            X_seq=X_seq,
            y=y,
            maps=maps,
            heldout_map=heldout_map,
            seed=args.seed,
            val_size=args.cross_map_val_size,
        )

        if split_data is None:
            print(f"  Skipping held-out map {heldout_map} (insufficient data).")
            continue

        print(
            f"\n--- Held out map: {heldout_map} | "
            f"train={len(split_data['y_tr'])} val={len(split_data['y_val'])} test={len(split_data['y_te'])}"
        )

        for model_name in model_names:
            _, result = train_and_evaluate_one_model(
                model_name,
                split_data,
                args,
                device,
                save_checkpoint=False,
            )
            result["heldout_map"] = heldout_map
            result["n_train"] = int(len(split_data["y_tr"]))
            result["n_val"] = int(len(split_data["y_val"]))
            result["n_test"] = int(len(split_data["y_te"]))
            cross_map_results[model_name][heldout_map] = result

    cross_map_results["_summary"] = aggregate_cross_map_results(cross_map_results)
    return cross_map_results


def run_training(args) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = get_device()

    print(f"\n=== Using device: {device} ===")
    print("\n=== Loading features ===")
    try:
        datasets = feat_mod.load_datasets(FEATURES_DIR)
    except FileNotFoundError:
        print("[train] Features not found — building now …")
        datasets = feat_mod.build_datasets(Path(args.db))
        feat_mod.save_datasets(datasets, FEATURES_DIR)

    X_static = datasets["X_static"]
    y = datasets["y_static"]
    X_seq = datasets["X_seq"]
    round_ids = datasets.get("round_ids")

    if round_ids is None:
        round_ids_path = FEATURES_DIR / "round_ids.txt"
        if round_ids_path.exists():
            round_ids = [line.strip() for line in round_ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    if not round_ids or len(round_ids) != len(y):
        raise RuntimeError(
            "round_ids metadata is missing or does not match the feature arrays. "
            "Rebuild features with the current features.py."
        )

    maps = np.array(load_round_maps(round_ids, Path(args.db)))

    print(f"  Loaded {len(y)} samples  |  F_static={X_static.shape[1]}  |  Maps={sorted(set(maps.tolist()))}")

    if len(np.unique(y)) < 2:
        print("[train] ERROR: Only one class in labels — run label_rounds.py --update-db first.")
        return

    split_data = split_train_val_test(
        X_static=X_static,
        X_seq=X_seq,
        y=y,
        maps=maps,
        seed=args.seed,
        val_size=args.val_size,
        test_size=args.test_size,
    )

    print(
        f"  Train: {len(split_data['y_tr'])}  "
        f"Val: {len(split_data['y_val'])}  "
        f"Test: {len(split_data['y_te'])}"
    )
    print(f"  Class dist (train): {dict(zip(*np.unique(split_data['y_tr'], return_counts=True)))}")
    print(f"  Test maps: {dict(zip(*np.unique(split_data['maps_te'], return_counts=True)))}")

    all_results = {
        "_dataset": {
            "n_samples": int(len(y)),
            "feature_dim_static": int(X_static.shape[1]),
            "feature_shape_seq": list(X_seq.shape[1:]),
            "maps": {m: int((maps == m).sum()) for m in sorted(set(maps.tolist()))},
            "split_sizes": {
                "train": int(len(split_data["y_tr"])),
                "val": int(len(split_data["y_val"])),
                "test": int(len(split_data["y_te"])),
            },
            "class_distribution_full": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
            "class_distribution_train": {int(k): int(v) for k, v in zip(*np.unique(split_data["y_tr"], return_counts=True))},
        }
    }

    model_names = ["MLP", "LSTM", "GRU", "Transformer"]
    for model_name in model_names:
        _, result = train_and_evaluate_one_model(model_name, split_data, args, device, save_checkpoint=True)
        all_results[model_name] = result

    all_results["_analysis"] = summarize_best_models({k: v for k, v in all_results.items() if not k.startswith("_")})

    results_path = RESULTS_DIR / "training_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== Test-set Model Comparison ===")
    print(f"  {'Model':<12} {'Acc':>7} {'BalAcc':>8} {'MacroF1':>9} {'Top2':>7}")
    print(f"  {'-'*50}")
    for name in model_names:
        m = all_results[name]["test"]
        print(
            f"  {name:<12} {m['accuracy']:>7.4f} {m['balanced_accuracy']:>8.4f} "
            f"{m['macro_f1']:>9.4f} {m['top2_accuracy']:>7.4f}"
        )

    print(f"\n[train] Results saved to {results_path}")

    if not args.no_cross_map:
        cross_map_results = run_cross_map_validation(
            X_static=X_static,
            X_seq=X_seq,
            y=y,
            maps=maps,
            args=args,
            device=device,
        )

        cross_path = RESULTS_DIR / "cross_map_results.json"
        with open(cross_path, "w", encoding="utf-8") as f:
            json.dump(cross_map_results, f, indent=2)

        print("\n=== Cross-map Summary ===")
        print(f"  {'Model':<12} {'Mean Acc':>9} {'Mean BalAcc':>12} {'Mean MacroF1':>13} {'Mean Top2':>10}")
        print(f"  {'-'*64}")
        for name, summary in cross_map_results.get("_summary", {}).items():
            print(
                f"  {name:<12} {summary['mean_accuracy']:>9.4f} {summary['mean_balanced_accuracy']:>12.4f} "
                f"{summary['mean_macro_f1']:>13.4f} {summary['mean_top2_accuracy']:>10.4f}"
            )

        print(f"\n[train] Cross-map results saved to {cross_path}")

    print(
        f"[train] Saved checkpoints: "
        f"{RESULTS_DIR / 'mlp_best.pt'}, "
        f"{RESULTS_DIR / 'lstm_best.pt'}, "
        f"{RESULTS_DIR / 'gru_best.pt'}, "
        f"{RESULTS_DIR / 'transformer_best.pt'}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tactical intent prediction models with PyTorch.")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--hidden", type=int, default=128)                      # Hidden layer size (MLP) / hidden dim (LSTM, GRU)
    parser.add_argument("--n-layers", type=int, default=2)                      # Number of hidden layers for MLP
    parser.add_argument("--rnn-layers", type=int, default=1)                    # Number of recurrent layers for LSTM / GRU
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--d-model", type=int, default=64)                      # Transformer d_model (must be divisible by --n-heads)
    parser.add_argument("--n-heads", type=int, default=2)                       # Number of attention heads
    parser.add_argument("--tf-layers", type=int, default=1)                     # Number of Transformer encoder layers
    parser.add_argument("--val-size", type=float, default=0.15)                 # Final validation fraction of full dataset
    parser.add_argument("--test-size", type=float, default=0.15)                # Final test fraction of full dataset
    parser.add_argument("--cross-map-val-size", type=float, default=0.15)       # Validation fraction inside non-heldout maps
    parser.add_argument("--no-cross-map", action="store_true")                  # Skip leave-one-map-out cross-map validation
    parser.add_argument("--full-pipeline", action="store_true")                 # Run labeling + feature extraction before training
    args = parser.parse_args()

    if args.full_pipeline:
        print("\n=== Step 1/3: Labeling rounds ===")
        labels = label_rounds.label_all_rounds(Path(args.db))
        if labels.empty:
            raise SystemExit("No rounds could be labelled. Check your database.")
        label_rounds.print_distribution(labels)
        label_rounds.save_labels_to_db(labels, Path(args.db))

        print("\n=== Step 2/3: Extracting features ===")
        datasets = feat_mod.build_datasets(Path(args.db))
        feat_mod.save_datasets(datasets, FEATURES_DIR)

        print("\n=== Step 3/3: Training models ===")

    run_training(args)
