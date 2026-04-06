"""
train.py
--------
Training harness for Weeks 2 & 3.

Trains and evaluates all three model architectures:
  - MLP          (Week 2 baseline)
  - LSTM         (Week 3)
  - GRU          (Week 3)

Features
--------
- Adam optimiser with gradient clipping
- Stratified train / validation split
- Early stopping on validation loss
- Full evaluation metrics:
    accuracy, macro precision/recall/F1,
    confusion matrix, top-2 accuracy,
    early-round prediction accuracy at 5s / 10s / 15s
- Results saved to results/training_results.json

Usage
-----
    # Full pipeline from raw DB:
    python train.py --full-pipeline

    # Train only (features must already be built):
    python train.py

    # Quick smoke-test with tiny settings:
    python train.py --epochs 5 --hidden 32
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

import label_rounds
import features as feat_mod
from models import MLP, LSTMClassifier, GRUClassifier

RESULTS_DIR = Path("results")
FEATURES_DIR = Path("features")
DB_PATH = Path("valorant.db")

LABEL_NAMES = {
    0: "A-Site Rush",
    1: "B-Site Rush",
    2: "A Split",
    3: "B Split",
    4: "Default",
    5: "Fake",
    6: "Slow Execute",
}
N_CLASSES = len(LABEL_NAMES)


# ══════════════════════════════════════════════════════════════════════════════
# Adam Optimiser
# ══════════════════════════════════════════════════════════════════════════════

class Adam:
    """Per-parameter Adam with weight decay."""

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8,
                 weight_decay: float = 1e-4):
        self.lr = lr; self.b1 = beta1; self.b2 = beta2
        self.eps = eps; self.wd = weight_decay
        self.m: dict = {}; self.v: dict = {}; self.t = 0

    def step(self, param_dict: dict) -> dict:
        """
        param_dict : {name: (param_array, grad_array)}
        Returns {name: updated_param}
        """
        self.t += 1
        updated = {}
        for name, (p, g) in param_dict.items():
            g = g + self.wd * p    # weight decay
            if name not in self.m:
                self.m[name] = np.zeros_like(p)
                self.v[name] = np.zeros_like(p)
            self.m[name] = self.b1 * self.m[name] + (1 - self.b1) * g
            self.v[name] = self.b2 * self.v[name] + (1 - self.b2) * g**2
            m_hat = self.m[name] / (1 - self.b1**self.t)
            v_hat = self.v[name] / (1 - self.b2**self.t)
            updated[name] = p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return updated


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, logits: np.ndarray) -> dict:
    """
    Compute accuracy, macro P/R/F1, confusion matrix, top-2 accuracy.
    """
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  accuracy_score)

    y_pred = logits.argmax(axis=-1)
    acc = float(accuracy_score(y_true, y_pred))

    # Top-2 accuracy
    top2_idx  = np.argsort(logits, axis=-1)[:, -2:]
    top2_acc  = float(np.mean([y_true[i] in top2_idx[i] for i in range(len(y_true))]))

    report = classification_report(
        y_true, y_pred,
        labels=list(range(N_CLASSES)),
        target_names=list(LABEL_NAMES.values()),
        output_dict=True,
        zero_division=0,
    )
    macro = report.get("macro avg", {})

    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES))).tolist()

    return {
        "accuracy":       round(acc, 4),
        "top2_accuracy":  round(top2_acc, 4),
        "macro_precision": round(float(macro.get("precision", 0)), 4),
        "macro_recall":    round(float(macro.get("recall",    0)), 4),
        "macro_f1":        round(float(macro.get("f1-score",  0)), 4),
        "confusion_matrix": cm,
        "per_class":       {LABEL_NAMES[i]: {
            "precision": round(float(report.get(LABEL_NAMES[i],{}).get("precision",0)),4),
            "recall":    round(float(report.get(LABEL_NAMES[i],{}).get("recall",0)),4),
            "f1":        round(float(report.get(LABEL_NAMES[i],{}).get("f1-score",0)),4),
            "support":   int(  report.get(LABEL_NAMES[i],{}).get("support",0)),
        } for i in range(N_CLASSES)},
    }


def early_round_accuracy(model, X_seq_val: np.ndarray,
                          y_val: np.ndarray,
                          steps_ms: list[int]) -> dict[str, float]:
    """
    Evaluate model accuracy when only the first k time steps are visible.
    Returns {f"{t//1000}s": accuracy} for each prefix length.
    """
    results = {}
    for k, t_ms in enumerate(steps_ms, start=1):
        X_prefix = X_seq_val[:, :k, :]
        # Pad the remaining steps with zero so shape stays (N, T, F)
        # The model will only "see" real info in the first k positions.
        T = X_seq_val.shape[1]
        padded = np.zeros_like(X_seq_val)
        padded[:, :k, :] = X_prefix
        preds = model.predict(padded)
        acc = float((preds == y_val).mean())
        results[f"{t_ms // 1000}s"] = round(acc, 4)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Training loop  (MLP)
# ══════════════════════════════════════════════════════════════════════════════

def train_mlp(X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray,   y_val: np.ndarray,
              hidden_dims: list[int], epochs: int, batch_size: int,
              lr: float, dropout: float, patience: int,
              seed: int = 42) -> tuple[MLP, list, list]:

    model = MLP(
        input_dim   = X_train.shape[1],
        hidden_dims = hidden_dims,
        n_classes   = N_CLASSES,
        dropout     = dropout,
        seed        = seed,
    )

    opt = Adam(lr=lr)
    train_losses, val_losses = [], []
    best_val, best_state, no_improve = np.inf, None, 0

    for epoch in range(epochs):
        # Shuffle
        idx = np.random.permutation(len(X_train))
        X_tr = X_train[idx]; y_tr = y_train[idx]

        # Mini-batch SGD
        for start in range(0, len(X_tr), batch_size):
            Xb = X_tr[start:start+batch_size]
            yb = y_tr[start:start+batch_size]

            logits_b = model.forward(Xb, training=True)
            dW_list, db_list = model.backward(logits_b, yb)

            params = {}
            for j, (W, dW, b, db) in enumerate(
                    zip(model.weights, dW_list, model.biases, db_list)):
                params[f"W{j}"] = (W, dW)
                params[f"b{j}"] = (b, db)

            updated = opt.step(params)
            for j in range(len(model.weights)):
                model.weights[j][:] = updated[f"W{j}"]
                model.biases[j][:]  = updated[f"b{j}"]

        # Track losses
        tl = model.loss(X_train, y_train)
        vl = model.loss(X_val,   y_val)
        train_losses.append(tl)
        val_losses.append(vl)

        if (epoch + 1) % max(1, epochs // 10) == 0:
            val_acc = (model.predict(X_val) == y_val).mean()
            print(f"  [MLP] epoch {epoch+1:>4}/{epochs}  "
                  f"train_loss={tl:.4f}  val_loss={vl:.4f}  "
                  f"val_acc={val_acc:.3f}")

        # Early stopping
        if vl < best_val - 1e-5:
            best_val = vl
            best_state = model.get_params().copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [MLP] Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.set_params(best_state)

    return model, train_losses, val_losses


# ══════════════════════════════════════════════════════════════════════════════
# Training loop  (LSTM / GRU)
# ══════════════════════════════════════════════════════════════════════════════

def train_rnn(model_cls, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              hidden_dim: int, epochs: int, batch_size: int,
              lr: float, dropout: float, patience: int,
              seed: int = 42):

    model = model_cls(
        input_dim  = X_train.shape[2],
        hidden_dim = hidden_dim,
        n_classes  = N_CLASSES,
        dropout    = dropout,
        seed       = seed,
    )

    train_losses, val_losses = [], []
    best_val, no_improve = np.inf, 0
    best_Wx = best_Wh = best_b_cell = best_Wout = best_bout = None

    def _save():
        return (model._cell.Wx.copy(), model._cell.Wh.copy(),
                model._cell.b.copy(), model.W_out.copy(), model.b_out.copy())

    def _restore(s):
        model._cell.Wx[:] = s[0]; model._cell.Wh[:] = s[1]
        model._cell.b[:]  = s[2]; model.W_out[:]    = s[3]
        model.b_out[:]    = s[4]

    for epoch in range(epochs):
        idx = np.random.permutation(len(X_train))
        X_tr = X_train[idx]; y_tr = y_train[idx]

        for start in range(0, len(X_tr), batch_size):
            Xb = X_tr[start:start+batch_size]
            yb = y_tr[start:start+batch_size]
            grads = model.backward(Xb, yb)
            CLIP = 5.0
            for k, g in grads.items():
                norm = np.linalg.norm(g)
                if norm > CLIP: grads[k] = g * CLIP / norm
            model.apply_grads(grads, lr=lr, grad_clip=CLIP)

        tl = model.loss(X_train, y_train)
        vl = model.loss(X_val,   y_val)
        train_losses.append(tl)
        val_losses.append(vl)

        name = model.__class__.__name__
        if (epoch + 1) % max(1, epochs // 10) == 0:
            val_acc = (model.predict(X_val) == y_val).mean()
            print(f"  [{name}] epoch {epoch+1:>4}/{epochs}  "
                  f"train_loss={tl:.4f}  val_loss={vl:.4f}  "
                  f"val_acc={val_acc:.3f}")

        if vl < best_val - 1e-5:
            best_val = vl
            saved = _save()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [{name}] Early stopping at epoch {epoch+1}")
                break

    if saved is not None:
        _restore(saved)

    return model, train_losses, val_losses


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_training(args):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ── Load features ─────────────────────────────────────────────────────────
    print("\n=== Loading features ===")
    try:
        datasets = feat_mod.load_datasets(FEATURES_DIR)
    except FileNotFoundError:
        print("[train] Features not found — building now …")
        datasets = feat_mod.build_datasets(DB_PATH)
        feat_mod.save_datasets(datasets, FEATURES_DIR)

    X_static = datasets["X_static"]
    y_static = datasets["y_static"]
    X_seq    = datasets["X_seq"]
    y_seq    = datasets["y_seq"]

    # print(f"  Loaded {len(y_static)} samples  |  "
    #       f"F_static={X_static.shape[1]}  T={X_seq.shape[1]}  F_seq={X_seq.shape[2]}")
    print(f"  Loaded {len(y_static)} samples  |  F_static={X_static.shape[1]}")

    if len(np.unique(y_static)) < 2:
        print("[train] ERROR: Only one class in labels — run label_rounds.py --update-db first.")
        return

    # ── Train/val split (stratified 80/20) ───────────────────────────────────
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train_idx, val_idx = next(sss.split(X_static, y_static))

    X_s_tr, X_s_val = X_static[train_idx], X_static[val_idx]
    y_s_tr, y_s_val = y_static[train_idx], y_static[val_idx]
    X_q_tr, X_q_val = X_seq[train_idx],    X_seq[val_idx]
    y_q_tr, y_q_val = y_seq[train_idx],    y_seq[val_idx]

    # ── Standardise static features ───────────────────────────────────────────
    scaler = StandardScaler()
    X_s_tr  = scaler.fit_transform(X_s_tr).astype(np.float32)
    X_s_val = scaler.transform(X_s_val).astype(np.float32)
    # For seq: standardise per-feature across (N*T) samples
    N_tr, T, F = X_q_tr.shape
    seq_scaler  = StandardScaler()
    X_q_tr_2d   = X_q_tr.reshape(-1, F)
    X_q_tr_2d   = seq_scaler.fit_transform(X_q_tr_2d).astype(np.float32)
    X_q_tr      = X_q_tr_2d.reshape(N_tr, T, F)
    N_vl        = X_q_val.shape[0]
    X_q_val_2d  = X_q_val.reshape(-1, F)
    X_q_val_2d  = seq_scaler.transform(X_q_val_2d).astype(np.float32)
    X_q_val     = X_q_val_2d.reshape(N_vl, T, F)

    print(f"  Train: {len(y_s_tr)}  Val: {len(y_s_val)}")
    print(f"  Class dist (train): {dict(zip(*np.unique(y_s_tr, return_counts=True)))}")

    hidden_dims = [args.hidden] * args.n_layers
    all_results = {}

    # ── MLP ───────────────────────────────────────────────────────────────────
    print(f"\n=== Training MLP (hidden={hidden_dims}) ===")
    t0 = time.time()
    mlp, mlp_tl, mlp_vl = train_mlp(
        X_s_tr, y_s_tr, X_s_val, y_s_val,
        hidden_dims  = hidden_dims,
        epochs       = args.epochs,
        batch_size   = args.batch_size,
        lr           = args.lr,
        dropout      = args.dropout,
        patience     = args.patience,
        seed         = args.seed,
    )
    mlp_time = time.time() - t0

    mlp_logits = mlp.forward(X_s_val, training=False)
    mlp_metrics = compute_metrics(y_s_val, mlp_logits)
    mlp_metrics["train_loss_curve"] = [round(x, 5) for x in mlp_tl]
    mlp_metrics["val_loss_curve"]   = [round(x, 5) for x in mlp_vl]
    mlp_metrics["train_time_s"]     = round(mlp_time, 2)
    all_results["MLP"] = mlp_metrics

    print(f"  MLP  acc={mlp_metrics['accuracy']:.4f}  "
          f"macro_f1={mlp_metrics['macro_f1']:.4f}  "
          f"top2_acc={mlp_metrics['top2_accuracy']:.4f}  "
          f"({mlp_time:.1f}s)")

    # # ── LSTM ──────────────────────────────────────────────────────────────────
    # print(f"\n=== Training LSTM (hidden={args.hidden}) ===")
    # t0 = time.time()
    # lstm, lstm_tl, lstm_vl = train_rnn(
    #     LSTMClassifier, X_q_tr, y_q_tr, X_q_val, y_q_val,
    #     hidden_dim = args.hidden,
    #     epochs     = args.epochs,
    #     batch_size = args.batch_size,
    #     lr         = args.lr,
    #     dropout    = args.dropout,
    #     patience   = args.patience,
    #     seed       = args.seed,
    # )
    # lstm_time = time.time() - t0

    # lstm_logits  = lstm.forward(X_q_val, training=False)
    # lstm_metrics = compute_metrics(y_q_val, lstm_logits)
    # lstm_metrics["train_loss_curve"] = [round(x, 5) for x in lstm_tl]
    # lstm_metrics["val_loss_curve"]   = [round(x, 5) for x in lstm_vl]
    # lstm_metrics["train_time_s"]     = round(lstm_time, 2)

    # steps_ms = feat_mod.SEQ_STEPS_MS
    # lstm_metrics["early_round_accuracy"] = early_round_accuracy(
    #     lstm, X_q_val, y_q_val, steps_ms)
    # all_results["LSTM"] = lstm_metrics

    # print(f"  LSTM acc={lstm_metrics['accuracy']:.4f}  "
    #       f"macro_f1={lstm_metrics['macro_f1']:.4f}  "
    #       f"top2_acc={lstm_metrics['top2_accuracy']:.4f}  "
    #       f"({lstm_time:.1f}s)")
    # print(f"  Early-round accuracy: {lstm_metrics['early_round_accuracy']}")

    # # ── GRU ───────────────────────────────────────────────────────────────────
    # print(f"\n=== Training GRU (hidden={args.hidden}) ===")
    # t0 = time.time()
    # gru, gru_tl, gru_vl = train_rnn(
    #     GRUClassifier, X_q_tr, y_q_tr, X_q_val, y_q_val,
    #     hidden_dim = args.hidden,
    #     epochs     = args.epochs,
    #     batch_size = args.batch_size,
    #     lr         = args.lr,
    #     dropout    = args.dropout,
    #     patience   = args.patience,
    #     seed       = args.seed,
    # )
    # gru_time = time.time() - t0

    # gru_logits  = gru.forward(X_q_val, training=False)
    # gru_metrics = compute_metrics(y_q_val, gru_logits)
    # gru_metrics["train_loss_curve"] = [round(x, 5) for x in gru_tl]
    # gru_metrics["val_loss_curve"]   = [round(x, 5) for x in gru_vl]
    # gru_metrics["train_time_s"]     = round(gru_time, 2)

    # gru_metrics["early_round_accuracy"] = early_round_accuracy(
    #     gru, X_q_val, y_q_val, steps_ms)
    # all_results["GRU"] = gru_metrics

    # print(f"  GRU  acc={gru_metrics['accuracy']:.4f}  "
    #       f"macro_f1={gru_metrics['macro_f1']:.4f}  "
    #       f"top2_acc={gru_metrics['top2_accuracy']:.4f}  "
    #       f"({gru_time:.1f}s)")
    # print(f"  Early-round accuracy: {gru_metrics['early_round_accuracy']}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== Model Comparison ===")
    print(f"  {'Model':<8}  {'Acc':>6}  {'Macro F1':>9}  {'Top-2 Acc':>10}")
    print(f"  {'-'*40}")
    for name, m in all_results.items():
        print(f"  {name:<8}  {m['accuracy']:>6.4f}  {m['macro_f1']:>9.4f}  "
              f"{m['top2_accuracy']:>10.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    results_path = RESULTS_DIR / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[train] Results saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tactical intent prediction models.")
    parser.add_argument("--db",           default=str(DB_PATH))
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch-size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--hidden",       type=int,   default=128,
                        help="Hidden layer size (MLP) / hidden dim (LSTM,GRU)")
    parser.add_argument("--n-layers",     type=int,   default=2,
                        help="Number of hidden layers for MLP")
    parser.add_argument("--dropout",      type=float, default=0.3)
    parser.add_argument("--patience",     type=int,   default=15,
                        help="Early stopping patience in epochs")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--full-pipeline", action="store_true",
                        help="Run labeling + feature extraction before training")
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
