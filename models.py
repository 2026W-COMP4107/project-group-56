"""
models.py
---------
PyTorch implementations of the tactical intent models.

Models:
- MLPClassifier
- LSTMClassifier
- GRUClassifier

All models return raw logits of shape (N, C).
"""

from __future__ import annotations

from typing import List
import torch
import torch.nn as nn
import math


class MLPClassifier(nn.Module):
    """
    Feedforward classifier for static snapshot features.
    Input:  (N, F)
    Output: (N, C) logits
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        n_classes: int,
        dropout: float = 0.15,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LSTMClassifier(nn.Module):
    """
    Sequence classifier using LSTM.
    Input:  (N, T, F)
    Output: (N, C) logits
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_classes: int,
        n_layers: int = 1,
        dropout: float = 0.15,
        bidirectional: bool = False,
    ):
        super().__init__()

        lstm_dropout = dropout if n_layers > 1 else 0.0
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.n_dirs = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(hidden_dim * self.n_dirs, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # out: (N, T, H * dirs)
        # h_n: (layers * dirs, N, H)
        _, (h_n, _) = self.lstm(x)

        if self.bidirectional:
            # take final forward + final backward states from last layer
            h_forward = h_n[-2]   # (N, H)
            h_backward = h_n[-1]  # (N, H)
            h_final = torch.cat([h_forward, h_backward], dim=1)
        else:
            h_final = h_n[-1]

        h_final = self.dropout(h_final)
        return self.fc(h_final)


class GRUClassifier(nn.Module):
    """
    Sequence classifier using GRU.
    Input:  (N, T, F)
    Output: (N, C) logits
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_classes: int,
        n_layers: int = 1,
        dropout: float = 0.15,
        bidirectional: bool = False,
    ):
        super().__init__()

        gru_dropout = dropout if n_layers > 1 else 0.0
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.n_dirs = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(hidden_dim * self.n_dirs, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # h_n: (layers * dirs, N, H)
        _, h_n = self.gru(x)

        if self.bidirectional:
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            h_final = torch.cat([h_forward, h_backward], dim=1)
        else:
            h_final = h_n[-1]

        h_final = self.dropout(h_final)
        return self.fc(h_final)

class _PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (Vaswani et al., 2017).
    Shape: (1, T, d_model) — broadcast over batch.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0)                        # (1, T, d)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """
    Sequence classifier using a Transformer encoder.

    Architecture
    ------------
    1. Linear input projection  F → d_model
    2. Sinusoidal positional encoding
    3. n_layers × TransformerEncoderLayer  (MHSA + FFN, Pre-LN)
    4. Mean-pool across T time steps → (N, d_model)
    5. Dropout + linear head → (N, C) logits

    Input:  (N, T, F)
    Output: (N, C) logits

    Notes
    -----
    - d_model must be divisible by n_heads.
    - Mean-pooling is preferred over a [CLS] token when T is small (T=6 here).
    - norm_first=True (Pre-LN) gives more stable gradients for shallow encoders.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        n_classes: int = 6,
        max_seq_len: int = 64,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = _PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # (N, T, d_model)
            norm_first=True,    # Pre-LN
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(d_model, n_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, T, F)  →  (N, C)"""
        x = self.input_proj(x)   # (N, T, d_model)
        x = self.pos_enc(x)      # (N, T, d_model)
        x = self.encoder(x)      # (N, T, d_model)
        x = x.mean(dim=1)        # (N, d_model)  ← mean-pool over time
        x = self.dropout(x)
        return self.fc(x)        # (N, C)