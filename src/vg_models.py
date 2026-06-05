"""Voltage-grid LSTM and CNN-LSTM models for notebook 09.

Key design choices
------------------
1.  Bidirectional LSTM, 1 layer, hidden 40, dropout 0.35.
2.  Masked pooling over the output sequence (mean + max, both respecting the
    validity mask) so pad positions never contribute to the representation.
    This fixes the nb07/08 bug where out.mean(dim=1) pooled over all timesteps
    including the zero-fill tail.
3.  NO pack_padded_sequence. The LSTM runs on the full padded input. Packing
    requires a .cpu() call for lengths on every forward pass, which forces a
    GPU→CPU device sync (expensive on MPS). Since padded positions are filled
    with 0.0 (the scaled channel mean) and masked pooling already ignores them,
    packing gives zero quality benefit while adding per-batch sync overhead that
    dominates for small models.
4.  Optional scalar injection via scalar_mode="head" or "lstm_state"; the
    primary scalar-free path uses scalar_mode="none".
5.  CNN-LSTM: 2 Conv1d blocks with GroupNorm (safer than BatchNorm under LOBO
    where batch statistics are computed on held-out-exclusive data) followed by
    the same bidirectional LSTM + masked pooling head.

All modules implement forward(x, mask, s=None) so they are compatible with
the sequence.train_model / predict_vg training harness.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Masked pooling helper
# ---------------------------------------------------------------------------


def masked_pool(
    out: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean + max pooling over the valid (unmasked) timesteps.

    Parameters
    ----------
    out  : (B, L, 2H)  LSTM output tensor (bidirectional → 2*hidden channels)
    mask : (B, L)      bool — True = valid timestep

    Returns
    -------
    pooled : (B, 4H)   concat of [mean_pool, max_pool]
    """
    m = mask.unsqueeze(-1).to(out.dtype)  # (B, L, 1)
    lengths = mask.sum(dim=1, keepdim=True).clamp(min=1).to(out.dtype)  # (B, 1)

    # Mean: sum over valid positions, divide by count
    mean_pool = (out * m).sum(dim=1) / lengths  # (B, 2H)

    # Max: shift invalid positions to a large negative value.
    # Avoids float("-inf") which triggers a slow path on MPS.
    out_masked = out + (~mask.unsqueeze(-1)) * (-1e9)
    max_pool = out_masked.max(dim=1).values  # (B, 2H)

    return torch.cat([mean_pool, max_pool], dim=-1)  # (B, 4H)


# ---------------------------------------------------------------------------
# Bidirectional LSTM
# ---------------------------------------------------------------------------


class VGLSTMReg(nn.Module):
    """Voltage-grid bidirectional LSTM regressor.

    Architecture
    ------------
    Input (B, L, n_features=3)  →  BiLSTM(hidden)  →  masked mean+max pool
    → Dropout(dropout)  →  Linear(4H [+ n_scalars], 1)

    Scalar injection modes
    ----------------------
    "none"       : no scalar features (primary path).
    "head"       : concatenate s to the pooled rep before the FC layer.
    "lstm_state" : project s to (h0, c0) to prime the BiLSTM with context.
    Both injection modes are plumbed but only used when scalar_mode != "none"
    and s is provided to forward().
    """

    def __init__(
        self,
        n_features: int = 3,
        hidden: int = 40,
        dropout: float = 0.35,
        n_scalars: int = 0,
        scalar_mode: str = "none",  # "none" | "head" | "lstm_state"
    ) -> None:
        super().__init__()
        if scalar_mode not in {"none", "head", "lstm_state"}:
            raise ValueError(
                f"scalar_mode must be 'none', 'head', or 'lstm_state'; got {scalar_mode!r}"
            )

        self.hidden = hidden
        self.scalar_mode = scalar_mode
        self.n_scalars = n_scalars

        self.lstm = nn.LSTM(
            n_features,
            hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.drop = nn.Dropout(dropout)

        if scalar_mode == "lstm_state" and n_scalars > 0:
            # 2 directions × 1 layer = 2 hidden vectors each of size H
            self.h0_proj = nn.Linear(n_scalars, 2 * hidden)
            self.c0_proj = nn.Linear(n_scalars, 2 * hidden)

        head_in = 4 * hidden + (n_scalars if scalar_mode == "head" else 0)
        self.fc = nn.Linear(head_in, 1)

    def forward(
        self,
        x: torch.Tensor,  # (B, L, C)
        mask: torch.Tensor,  # (B, L) bool
        s: torch.Tensor | None = None,  # (B, n_scalars) optional
    ) -> torch.Tensor:
        B, L, _ = x.shape

        # --- Optional initial state from scalars ---
        h0, c0 = None, None
        if self.scalar_mode == "lstm_state" and s is not None:
            h0 = torch.tanh(self.h0_proj(s)).view(2, B, self.hidden)
            c0 = torch.tanh(self.c0_proj(s)).view(2, B, self.hidden)

        # Run LSTM on the full padded sequence — no pack_padded_sequence.
        # Padding positions are 0.0 (scaled channel mean) and masked pooling
        # ignores them; packing only adds a .cpu() sync overhead on MPS.
        out, _ = self.lstm(x, (h0, c0) if h0 is not None else None)
        # out: (B, L, 2H)

        # --- Masked mean + max pooling ---
        pooled = masked_pool(out, mask)  # (B, 4H)
        pooled = self.drop(pooled)

        # --- Optional head scalar injection ---
        if self.scalar_mode == "head" and s is not None:
            pooled = torch.cat([pooled, s], dim=-1)

        return self.fc(pooled).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# CNN-LSTM
# ---------------------------------------------------------------------------


class VGCNNLSTM(nn.Module):
    """CNN front-end + bidirectional LSTM for voltage-grid curves.

    Architecture
    ------------
    Input (B, L, n_features)
    → permute (B, n_features, L)
    → ConvBlock(n_features → cnn_ch, kernel=5)
    → ConvBlock(cnn_ch → cnn_ch, kernel=3)
    → permute back (B, L, cnn_ch)
    → BiLSTM(hidden)
    → masked mean+max pool → Dropout → Linear → scalar

    Same-padding (stride=1) keeps L unchanged, so the mask is valid throughout.
    """

    def __init__(
        self,
        n_features: int = 3,
        cnn_ch: int = 8,
        hidden: int = 40,
        dropout: float = 0.35,
        n_scalars: int = 0,
        scalar_mode: str = "none",
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.scalar_mode = scalar_mode
        self.n_scalars = n_scalars

        # GroupNorm stabilises conv output scale regardless of filter initialisation.
        # Without it, 8 filters × random seed = bimodal results (sometimes great, sometimes catastrophic).
        # num_groups=4 → 2 channels per group, works for any cnn_ch divisible by 4.
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, cnn_ch, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=cnn_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            cnn_ch,
            hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.drop = nn.Dropout(dropout)

        if scalar_mode == "lstm_state" and n_scalars > 0:
            self.h0_proj = nn.Linear(n_scalars, 2 * hidden)
            self.c0_proj = nn.Linear(n_scalars, 2 * hidden)

        head_in = 4 * hidden + (n_scalars if scalar_mode == "head" else 0)
        self.fc = nn.Linear(head_in, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        s: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape

        # CNN front-end: (B, L, C) → (B, C, L) → conv blocks → (B, L, cnn_ch)
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)  # (B, L, cnn_ch)

        # Re-zero the padded timesteps before the LSTM ---
        # mask is (B, L) bool. We unsqueeze to (B, L, 1) and multiply.
        x = x * mask.unsqueeze(-1).to(x.dtype)

        # Optional LSTM initial state
        h0, c0 = None, None
        if self.scalar_mode == "lstm_state" and s is not None:
            h0 = torch.tanh(self.h0_proj(s)).view(2, B, self.hidden)
            c0 = torch.tanh(self.c0_proj(s)).view(2, B, self.hidden)

        out, _ = self.lstm(x, (h0, c0) if h0 is not None else None)

        pooled = masked_pool(out, mask)
        pooled = self.drop(pooled)

        if self.scalar_mode == "head" and s is not None:
            pooled = torch.cat([pooled, s], dim=-1)

        return self.fc(pooled).squeeze(-1)


# ---------------------------------------------------------------------------
# Bidirectional GRU
# ---------------------------------------------------------------------------


class VGGRUReg(nn.Module):
    """Voltage-grid bidirectional GRU regressor.

    Architecture
    ------------
    Input (B, L, n_features=3)  →  BiGRU(hidden)  →  masked mean+max pool
    → Dropout(dropout)  →  Linear(4H [+ n_scalars], 1)

    Mirrors VGLSTMReg but uses GRU (3 gates, no cell state).
    Default hidden=47 gives ~14.9k params vs LSTM's ~14.6k at hidden=40.

    Scalar injection modes
    ----------------------
    "none"       : no scalar features (primary path).
    "head"       : concatenate s to the pooled rep before the FC layer.
    "lstm_state" : project s to h0 to prime the BiGRU (no c0 — GRU has none).
    """

    def __init__(
        self,
        n_features: int = 3,
        hidden: int = 47,
        dropout: float = 0.35,
        n_scalars: int = 0,
        scalar_mode: str = "none",  # "none" | "head" | "lstm_state"
    ) -> None:
        super().__init__()
        if scalar_mode not in {"none", "head", "lstm_state"}:
            raise ValueError(
                f"scalar_mode must be 'none', 'head', or 'lstm_state'; got {scalar_mode!r}"
            )

        self.hidden = hidden
        self.scalar_mode = scalar_mode
        self.n_scalars = n_scalars

        self.gru = nn.GRU(
            n_features,
            hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.drop = nn.Dropout(dropout)

        if scalar_mode == "lstm_state" and n_scalars > 0:
            # GRU has no cell state; 2 directions × 1 layer → one h0 vector
            self.h0_proj = nn.Linear(n_scalars, 2 * hidden)

        head_in = 4 * hidden + (n_scalars if scalar_mode == "head" else 0)
        self.fc = nn.Linear(head_in, 1)

    def forward(
        self,
        x: torch.Tensor,  # (B, L, C)
        mask: torch.Tensor,  # (B, L) bool
        s: torch.Tensor | None = None,  # (B, n_scalars) optional
    ) -> torch.Tensor:
        B, L, _ = x.shape

        h0 = None
        if self.scalar_mode == "lstm_state" and s is not None:
            h0 = torch.tanh(self.h0_proj(s)).view(2, B, self.hidden)

        out, _ = self.gru(x, h0)
        # out: (B, L, 2H)

        pooled = masked_pool(out, mask)  # (B, 4H)
        pooled = self.drop(pooled)

        if self.scalar_mode == "head" and s is not None:
            pooled = torch.cat([pooled, s], dim=-1)

        return self.fc(pooled).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# CNN-GRU
# ---------------------------------------------------------------------------


class VGCNNGRU(nn.Module):
    """CNN front-end + bidirectional GRU for voltage-grid curves.

    Architecture
    ------------
    Input (B, L, n_features)
    → permute (B, n_features, L)
    → ConvBlock(n_features → cnn_ch, kernel=5)
    → permute back (B, L, cnn_ch)
    → BiGRU(hidden)
    → masked mean+max pool → Dropout → Linear → scalar

    Mirrors VGCNNLSTM with GRU instead of LSTM.
    Default hidden=47 gives ~16.4k params vs CNN-LSTM's ~16.3k at hidden=40.
    Same-padding keeps L unchanged so the mask remains valid throughout.
    """

    def __init__(
        self,
        n_features: int = 3,
        cnn_ch: int = 8,
        hidden: int = 47,
        dropout: float = 0.35,
        n_scalars: int = 0,
        scalar_mode: str = "none",
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.scalar_mode = scalar_mode
        self.n_scalars = n_scalars

        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, cnn_ch, kernel_size=5, padding=2),
            nn.GroupNorm(num_groups=4, num_channels=cnn_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            cnn_ch,
            hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.drop = nn.Dropout(dropout)

        if scalar_mode == "lstm_state" and n_scalars > 0:
            self.h0_proj = nn.Linear(n_scalars, 2 * hidden)

        head_in = 4 * hidden + (n_scalars if scalar_mode == "head" else 0)
        self.fc = nn.Linear(head_in, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        s: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape

        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)  # (B, L, cnn_ch)

        h0 = None
        if self.scalar_mode == "lstm_state" and s is not None:
            h0 = torch.tanh(self.h0_proj(s)).view(2, B, self.hidden)

        out, _ = self.gru(x, h0)

        pooled = masked_pool(out, mask)
        pooled = self.drop(pooled)

        if self.scalar_mode == "head" and s is not None:
            pooled = torch.cat([pooled, s], dim=-1)

        return self.fc(pooled).squeeze(-1)


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_vg(
    model: nn.Module,
    X: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
    S: np.ndarray | None = None,
) -> np.ndarray:
    """Batched inference with clip to [0.0, 1.1].

    Clips to 1.1 (not 1.0) because fractional rated capacity Ct/C_nominal can
    slightly exceed 1.0 for fresh cells (e.g. 2.003 Ah / 2.0 = 1.0015).

    Parameters
    ----------
    model      : VGLSTMReg or VGCNNLSTM (or any module with same forward signature)
    X          : (n, L, C) float32
    mask       : (n, L) bool
    device     : torch.device
    S          : (n, n_scalars) float32 optional pre-scaled scalar features
    """
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32)
    mt = torch.tensor(mask, dtype=torch.bool)
    St = torch.tensor(S, dtype=torch.float32) if S is not None else None
    preds: list[np.ndarray] = []
    for i in range(0, len(Xt), batch_size):
        xb = Xt[i : i + batch_size].to(device)
        mb = mt[i : i + batch_size].to(device)
        sb = St[i : i + batch_size].to(device) if St is not None else None
        preds.append(model(xb, mb, sb).cpu().numpy())
    return np.clip(np.concatenate(preds), 0.0, 1.1)
