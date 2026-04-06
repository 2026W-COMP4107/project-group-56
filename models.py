"""
models.py
---------
Week 2 deliverable: Baseline MLP (Feedforward Neural Network)
Week 3 deliverable: LSTM and GRU recurrent models

All models are implemented from scratch in NumPy so the math is explicit
and the code has no external ML-framework dependency — appropriate for a
course project.

Architecture overview
---------------------
MLP
    Input → Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear → Softmax
    Input dim: F (feature dim)   Output dim: C (num classes)

LSTM
    Input sequence (T, F) → LSTM cell (hidden_size) → final hidden → Linear → Softmax
    Standard LSTM with forget / input / cell / output gates.

GRU
    Input sequence (T, F) → GRU cell (hidden_size) → final hidden → Linear → Softmax
    Standard GRU with reset / update gates.

All models expose:
    forward(X)          → logits (N, C)
    predict(X)          → class indices (N,)
    get_params()        → flat parameter vector
    set_params(vec)     → restore from flat vector (for optimiser)
    grad(X, y)          → parameter gradient (for SGD / Adam)
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# ── Activations ───────────────────────────────────────────────────────────────

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)

def sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    pos = x >= 0
    result = np.empty_like(x)
    result[pos]  = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    result[~pos] = exp_x / (1.0 + exp_x)
    return result

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise numerically stable softmax."""
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

def cross_entropy(logits: np.ndarray, labels: np.ndarray,
                  eps: float = 1e-9) -> float:
    """Mean cross-entropy loss over a batch."""
    probs = softmax(logits)
    N = len(labels)
    return float(-np.log(probs[np.arange(N), labels] + eps).mean())

def cross_entropy_grad(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Gradient of cross-entropy w.r.t. logits (= softmax − one_hot)."""
    probs = softmax(logits)
    N = len(labels)
    probs[np.arange(N), labels] -= 1.0
    return probs / N


# ── Glorot / He initialisation ────────────────────────────────────────────────

def glorot(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)

def he(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(0, std, (fan_in, fan_out)).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# MLP  (Week 2 — Baseline Feedforward Neural Network)
# ══════════════════════════════════════════════════════════════════════════════

class MLP:
    """
    Multi-Layer Perceptron for static snapshot classification.

    Architecture: Input → [Linear → ReLU → Dropout] × n_layers → Linear → Softmax

    Parameters
    ----------
    input_dim   : F (feature dimension)
    hidden_dims : list of hidden layer widths, e.g. [128, 64]
    n_classes   : C (number of output classes)
    dropout     : dropout probability (applied during training forward pass)
    seed        : random seed for reproducibility
    """

    def __init__(self, input_dim: int, hidden_dims: list[int],
                 n_classes: int, dropout: float = 0.3, seed: int = 42):
        self.n_classes = n_classes
        self.dropout   = dropout
        self._training = True
        rng = np.random.default_rng(seed)

        dims = [input_dim] + hidden_dims + [n_classes]
        self.weights: list[np.ndarray] = []
        self.biases:  list[np.ndarray] = []

        for i in range(len(dims) - 1):
            self.weights.append(he(dims[i], dims[i+1], rng))
            self.biases.append(np.zeros(dims[i+1], dtype=np.float32))

        # Cache for backprop
        self._cache: dict = {}

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        """
        X : (N, F)  — batch of feature vectors
        Returns logits (N, C)
        """
        self._training = training
        a = X.astype(np.float32)
        self._cache = {"layers": []}

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b                          # (N, out)
            is_last = (i == len(self.weights) - 1)

            if not is_last:
                a_pre = a
                z_pre = z
                a = relu(z)                        # ReLU hidden
                # Inverted dropout
                if training and self.dropout > 0:
                    mask = (np.random.rand(*a.shape) > self.dropout).astype(np.float32)
                    a = a * mask / (1.0 - self.dropout)
                else:
                    mask = np.ones_like(a)
                self._cache["layers"].append({"a_in": a_pre, "z": z_pre, "mask": mask})
            else:
                self._cache["layers"].append({"a_in": a, "z": z})
                logits = z

        return logits   # (N, C)

    # ── Backward pass ─────────────────────────────────────────────────────────

    def backward(self, logits: np.ndarray,
                 labels: np.ndarray) -> tuple[list, list]:
        """
        Returns (dW_list, db_list) — gradients for each layer.
        """
        dW_list = [None] * len(self.weights)
        db_list = [None] * len(self.biases)

        # Output layer gradient
        delta = cross_entropy_grad(logits, labels)   # (N, C)

        for i in reversed(range(len(self.weights))):
            cache = self._cache["layers"][i]
            a_in  = cache["a_in"]
            dW_list[i] = a_in.T @ delta
            db_list[i] = delta.sum(axis=0)

            if i > 0:
                # Backprop through this layer's weights
                delta = delta @ self.weights[i].T          # (N, hidden)
                # Through dropout mask
                mask = self._cache["layers"][i-1]["mask"]
                delta = delta * relu_grad(self._cache["layers"][i-1]["z"])
                if self.dropout > 0 and self._training:
                    delta = delta * mask / (1.0 - self.dropout)

        return dW_list, db_list

    # ── Prediction helpers ─────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = self.forward(X, training=False)
        return logits.argmax(axis=-1)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        return cross_entropy(self.forward(X, training=False), y)

    # ── Parameter accessors (for optimiser) ───────────────────────────────────

    def get_params(self) -> np.ndarray:
        parts = []
        for W, b in zip(self.weights, self.biases):
            parts.extend([W.ravel(), b.ravel()])
        return np.concatenate(parts)

    def set_params(self, vec: np.ndarray) -> None:
        offset = 0
        for W, b in zip(self.weights, self.biases):
            n = W.size; W[:] = vec[offset:offset+n].reshape(W.shape); offset += n
            n = b.size; b[:] = vec[offset:offset+n];                   offset += n

    def get_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        logits = self.forward(X, training=True)
        dW_list, db_list = self.backward(logits, y)
        parts = []
        for dW, db in zip(dW_list, db_list):
            parts.extend([dW.ravel(), db.ravel()])
        return np.concatenate(parts)


# ══════════════════════════════════════════════════════════════════════════════
# LSTM Cell  (Week 3)
# ══════════════════════════════════════════════════════════════════════════════

class LSTMCell:
    """
    Standard LSTM cell.

    Gates (all of shape (N, H)):
        f  = sigmoid(X @ Wf + h_prev @ Uf + bf)   forget gate
        i  = sigmoid(X @ Wi + h_prev @ Ui + bi)   input gate
        g  = tanh(   X @ Wg + h_prev @ Ug + bg)   cell gate
        o  = sigmoid(X @ Wo + h_prev @ Uo + bo)   output gate

    Cell update:
        c = f * c_prev + i * g
        h = o * tanh(c)
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 rng: np.random.Generator):
        self.H = hidden_dim
        scale = np.sqrt(1.0 / hidden_dim)

        # Input weights  (F, 4H) — all 4 gates packed together
        self.Wx = (rng.normal(0, scale, (input_dim, 4 * hidden_dim))
                   .astype(np.float32))
        # Recurrent weights  (H, 4H)
        # Orthogonal init for recurrent weights → better gradient flow
        Q, _ = np.linalg.qr(rng.normal(0, 1, (4 * hidden_dim, 4 * hidden_dim)))
        self.Wh = Q[:hidden_dim, :4 * hidden_dim].astype(np.float32)   # (H, 4H)
        self.b  = np.zeros(4 * hidden_dim, dtype=np.float32)
        # Forget gate bias initialised to 1 → avoids vanishing gradients early
        self.b[hidden_dim:2*hidden_dim] = 1.0

        self._cache: list = []

    def step(self, x: np.ndarray, h: np.ndarray,
             c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        x : (N, F)
        h : (N, H)
        c : (N, H)
        Returns (h_new, c_new)
        """
        H = self.H
        gates = x @ self.Wx + h @ self.Wh + self.b   # (N, 4H)

        f = sigmoid(gates[:, :H])
        i = sigmoid(gates[:, H:2*H])
        g = tanh(   gates[:, 2*H:3*H])
        o = sigmoid(gates[:, 3*H:])

        c_new = f * c + i * g
        h_new = o * tanh(c_new)

        self._cache.append((x, h, c, f, i, g, o, c_new))
        return h_new, c_new

    def backward_step(self, dh: np.ndarray, dc: np.ndarray,
                      cache_step: tuple) -> tuple:
        """
        Backprop through one LSTM step.
        Returns (dx, dh_prev, dc_prev, dWx, dWh, db)
        """
        x, h_prev, c_prev, f, i, g, o, c_new = cache_step
        H = self.H

        tanh_c_new = tanh(c_new)
        do = dh * tanh_c_new
        dc_new = dc + dh * o * (1 - tanh_c_new ** 2)

        df  = dc_new * c_prev
        di  = dc_new * g
        dg  = dc_new * i
        dc_prev = dc_new * f

        # Gate pre-activation gradients
        d_f_pre = df * f * (1 - f)
        d_i_pre = di * i * (1 - i)
        d_g_pre = dg * (1 - g ** 2)
        d_o_pre = do * o * (1 - o)

        dgates = np.concatenate([d_f_pre, d_i_pre, d_g_pre, d_o_pre], axis=1)  # (N,4H)

        dWx = x.T @ dgates
        dWh = h_prev.T @ dgates
        db  = dgates.sum(axis=0)
        dx  = dgates @ self.Wx.T
        dh_prev = dgates @ self.Wh.T

        return dx, dh_prev, dc_prev, dWx, dWh, db


# ══════════════════════════════════════════════════════════════════════════════
# GRU Cell  (Week 3)
# ══════════════════════════════════════════════════════════════════════════════

class GRUCell:
    """
    Standard GRU cell.

    Gates (all (N, H)):
        r  = sigmoid(X @ Wr + h_prev @ Ur + br)   reset gate
        z  = sigmoid(X @ Wz + h_prev @ Uz + bz)   update gate
        n  = tanh(   X @ Wn + r * (h_prev @ Un) + bn)  candidate

    Hidden update:
        h_new = (1 - z) * n + z * h_prev
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 rng: np.random.Generator):
        self.H = hidden_dim
        scale = np.sqrt(1.0 / hidden_dim)

        # Pack r, z, n weights
        self.Wx = rng.normal(0, scale, (input_dim,  3*hidden_dim)).astype(np.float32)
        self.Wh = rng.normal(0, scale, (hidden_dim, 3*hidden_dim)).astype(np.float32)
        self.b  = np.zeros(3*hidden_dim, dtype=np.float32)

        self._cache: list = []

    def step(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        H = self.H
        xW = x @ self.Wx          # (N, 3H)
        hW = h @ self.Wh          # (N, 3H)

        r = sigmoid(xW[:, :H]  + hW[:, :H]  + self.b[:H])
        z = sigmoid(xW[:, H:2*H] + hW[:, H:2*H] + self.b[H:2*H])
        n = tanh(   xW[:, 2*H:] + r * hW[:, 2*H:] + self.b[2*H:])

        h_new = (1 - z) * n + z * h
        self._cache.append((x, h, r, z, n, xW, hW))
        return h_new

    def backward_step(self, dh_new: np.ndarray,
                      cache_step: tuple) -> tuple:
        x, h_prev, r, z, n, xW, hW = cache_step
        H = self.H

        dz = dh_new * (h_prev - n)
        dn = dh_new * (1 - z)
        dh_prev_z = dh_new * z

        # Through tanh(n)
        dn_pre = dn * (1 - n**2)
        dr_hW  = dn_pre * hW[:, 2*H:]
        dr = dr_hW * r * (1 - r)

        # Through z gate
        dz_pre = dz * z * (1 - z)

        # Combine
        dx  = np.zeros_like(x)
        dx += dn_pre @ self.Wx[:, 2*H:].T
        dx += dr     @ self.Wx[:, :H].T * r * (1-r)    # approx, simplified
        dx += dz_pre @ self.Wx[:, H:2*H].T

        dWx = np.zeros_like(self.Wx)
        dWx[:, :H]   = x.T @ (dr   * r * (1-r))
        dWx[:, H:2*H]= x.T @  dz_pre
        dWx[:, 2*H:] = x.T @  dn_pre

        dWh = np.zeros_like(self.Wh)
        dWh[:, :H]    = h_prev.T @ (dr * r * (1-r))
        dWh[:, H:2*H] = h_prev.T @  dz_pre
        dWh[:, 2*H:]  = (r * h_prev).T @ dn_pre

        db = np.zeros_like(self.b)
        db[:H]    = (dr * r * (1-r)).sum(axis=0)
        db[H:2*H] = dz_pre.sum(axis=0)
        db[2*H:]  = dn_pre.sum(axis=0)

        dh_prev = dh_prev_z.copy()
        dh_prev += (dr * r * (1-r)) @ self.Wh[:, :H].T
        dh_prev += dz_pre           @ self.Wh[:, H:2*H].T
        dh_prev += (dn_pre * r)     @ self.Wh[:, 2*H:].T

        return dx, dh_prev, dWx, dWh, db


# ══════════════════════════════════════════════════════════════════════════════
# Sequence Classifiers  (Week 3)
# ══════════════════════════════════════════════════════════════════════════════

class _SequenceClassifier:
    """
    Base class for LSTM and GRU sequence classifiers.
    Architecture: RNN cell → final hidden → Linear → Softmax
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int,
                 dropout: float = 0.3, seed: int = 42):
        self.hidden_dim = hidden_dim
        self.n_classes  = n_classes
        self.dropout    = dropout
        rng = np.random.default_rng(seed)

        # Output linear layer
        self.W_out = glorot(hidden_dim, n_classes, rng)
        self.b_out = np.zeros(n_classes, dtype=np.float32)

        self._cell = None   # set by subclass

    def _run_sequence(self, X: np.ndarray,
                      training: bool = False) -> np.ndarray:
        """
        X : (N, T, F)
        Returns final hidden state (N, H)
        """
        raise NotImplementedError

    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        """
        X : (N, T, F)
        Returns logits (N, C)
        """
        h_final = self._run_sequence(X, training)

        # Dropout on final hidden
        if training and self.dropout > 0:
            mask = (np.random.rand(*h_final.shape) > self.dropout
                    ).astype(np.float32)
            h_final = h_final * mask / (1.0 - self.dropout)
        self._h_final = h_final
        logits = h_final @ self.W_out + self.b_out
        return logits

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X, training=False).argmax(axis=-1)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        return cross_entropy(self.forward(X, training=False), y)

    def _output_grad(self, logits: np.ndarray,
                     labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (d_h_final, dW_out, db_out)"""
        d_logits = cross_entropy_grad(logits, labels)   # (N, C)
        dW_out   = self._h_final.T @ d_logits
        db_out   = d_logits.sum(axis=0)
        d_h      = d_logits @ self.W_out.T              # (N, H)
        return d_h, dW_out, db_out


class LSTMClassifier(_SequenceClassifier):
    """LSTM sequence classifier."""

    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int,
                 dropout: float = 0.3, seed: int = 42):
        super().__init__(input_dim, hidden_dim, n_classes, dropout, seed)
        rng = np.random.default_rng(seed)
        self._cell = LSTMCell(input_dim, hidden_dim, rng)
        self._Xs: list = []

    def _run_sequence(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        N, T, F = X.shape
        H = self.hidden_dim
        h = np.zeros((N, H), dtype=np.float32)
        c = np.zeros((N, H), dtype=np.float32)
        self._cell._cache = []
        self._Xs = X   # save for backprop
        for t in range(T):
            h, c = self._cell.step(X[:, t, :], h, c)
        self._h_all_last = (h, c)
        return h

    def backward(self, X: np.ndarray, y: np.ndarray) -> dict:
        logits = self.forward(X, training=True)
        d_h, dW_out, db_out = self._output_grad(logits, y)

        T = X.shape[1]
        dh  = d_h
        dc  = np.zeros_like(dh)

        dWx_total = np.zeros_like(self._cell.Wx)
        dWh_total = np.zeros_like(self._cell.Wh)
        db_total  = np.zeros_like(self._cell.b)

        for t in reversed(range(T)):
            cache_step = self._cell._cache[t]
            _, dh, dc, dWx, dWh, db = self._cell.backward_step(dh, dc, cache_step)
            dWx_total += dWx
            dWh_total += dWh
            db_total  += db

        return {
            "cell_Wx": dWx_total,
            "cell_Wh": dWh_total,
            "cell_b":  db_total,
            "W_out":   dW_out,
            "b_out":   db_out,
        }

    def apply_grads(self, grads: dict, lr: float,
                    grad_clip: float = 5.0) -> None:
        for key, g in grads.items():
            norm = np.linalg.norm(g)
            if norm > grad_clip:
                g = g * grad_clip / norm
            if key == "cell_Wx": self._cell.Wx -= lr * g
            elif key == "cell_Wh": self._cell.Wh -= lr * g
            elif key == "cell_b":  self._cell.b  -= lr * g
            elif key == "W_out":   self.W_out    -= lr * g
            elif key == "b_out":   self.b_out    -= lr * g


class GRUClassifier(_SequenceClassifier):
    """GRU sequence classifier."""

    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int,
                 dropout: float = 0.3, seed: int = 42):
        super().__init__(input_dim, hidden_dim, n_classes, dropout, seed)
        rng = np.random.default_rng(seed)
        self._cell = GRUCell(input_dim, hidden_dim, rng)

    def _run_sequence(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        N, T, F = X.shape
        H = self.hidden_dim
        h = np.zeros((N, H), dtype=np.float32)
        self._cell._cache = []
        for t in range(T):
            h = self._cell.step(X[:, t, :], h)
        return h

    def backward(self, X: np.ndarray, y: np.ndarray) -> dict:
        logits = self.forward(X, training=True)
        d_h, dW_out, db_out = self._output_grad(logits, y)

        T = X.shape[1]
        dh = d_h

        dWx_total = np.zeros_like(self._cell.Wx)
        dWh_total = np.zeros_like(self._cell.Wh)
        db_total  = np.zeros_like(self._cell.b)

        for t in reversed(range(T)):
            cache_step = self._cell._cache[t]
            _, dh, dWx, dWh, db = self._cell.backward_step(dh, cache_step)
            dWx_total += dWx
            dWh_total += dWh
            db_total  += db

        return {
            "cell_Wx": dWx_total,
            "cell_Wh": dWh_total,
            "cell_b":  db_total,
            "W_out":   dW_out,
            "b_out":   db_out,
        }

    def apply_grads(self, grads: dict, lr: float,
                    grad_clip: float = 5.0) -> None:
        for key, g in grads.items():
            norm = np.linalg.norm(g)
            if norm > grad_clip:
                g = g * grad_clip / norm
            if key == "cell_Wx": self._cell.Wx -= lr * g
            elif key == "cell_Wh": self._cell.Wh -= lr * g
            elif key == "cell_b":  self._cell.b  -= lr * g
            elif key == "W_out":   self.W_out    -= lr * g
            elif key == "b_out":   self.b_out    -= lr * g
