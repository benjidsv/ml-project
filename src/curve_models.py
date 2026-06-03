"""Within-cycle SOH estimation models for notebook 06.

All models share the same calling convention:
    forward(x: (B, L, C), mask: (B, L), s: (B, n_scalars) | None) → (B,)

train_model / predict_model from src/sequence.py work unchanged.

Harness invariants (held constant across all architectures):
    - L1 loss, cosine LR schedule, Adam lr=3e-4, gradient clip norm=1.0
    - concat_pool [mean, max, first, last] for conv models
    - mean + max + final-states pool for RNNs
    - mean + max pool for Transformer
    - dropout=0 on the FC head for all pure-curve models (models are
      underparameterised relative to training set size; regularisation
      showed no benefit — see dl-journal Stage 2)
    - ~10–35k parameters per architecture for capacity-parity comparison
"""

import math

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


# ---------------------------------------------------------------------------
# Shared pooling utility
# ---------------------------------------------------------------------------


def concat_pool(x: torch.Tensor) -> torch.Tensor:
    """Aggregate the time dimension: mean + max + first + last → 4×C.

    Input:  (B, C, L)  — feature map after convolution blocks (time-last layout)
    Output: (B, 4*C)

    first (charge onset) carries the discriminative signal for high-SOH
    batteries: open-circuit-ish starting voltage and initial current acceptance
    separate fresh from aged cells before the curves converge in the CV taper.
    last (CV-taper end) is retained for low-to-mid SOH where taper shape still
    varies, but is not informative at the high-SOH end.
    """
    mean = x.mean(dim=-1)  # (B, C)
    max_ = x.max(dim=-1).values  # (B, C)
    first = x[:, :, 0]  # (B, C) — charge onset
    last = x[:, :, -1]  # (B, C) — CV taper end
    return torch.cat([mean, max_, first, last], dim=-1)  # (B, 4*C)


# ---------------------------------------------------------------------------
# 1D-CNN
# ---------------------------------------------------------------------------


def _num_groups(out_ch: int) -> int:
    """Pick the largest divisor of out_ch that is ≤ 8 (for GroupNorm)."""
    for g in (8, 4, 2, 1):
        if out_ch % g == 0:
            return g
    return 1


class _ConvBlock(nn.Module):
    """Conv1d → GroupNorm → ReLU → Dropout.

    GroupNorm replaces BatchNorm so statistics are computed per-sample, not
    per-batch — it doesn't accumulate running statistics that would fail to
    transfer to a held-out battery in Leave-One-Battery-Out cross-validation.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5, dropout: float = 0.25):
        super().__init__()
        pad = kernel // 2  # same-padding (symmetric, non-causal)
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad),
            nn.GroupNorm(_num_groups(out_ch), out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CNNReg(nn.Module):
    """3-block 1D-CNN for SOH regression from a fixed-length charge curve.

    Input:  (batch, n_points, channels)  → transpose → (batch, channels, n_points)
    Output: (batch,) scalar, clipped to [0,1] at eval.

    Architecture:
        Block1: C → 32 channels, k=5
        Block2: 32 → 64 channels, k=5
        Block3: 64 → 64 channels, k=3
        concat_pool([mean, max, last]) [+ optional scalars] → FC(192[+n_scalars], 1)

    Parameters
    ----------
    n_scalars : int
        Number of hand-crafted scalar features to concatenate with the pooled
        curve embedding before the head.  0 = pure curve model (default).
        When > 0, forward() expects a third argument s of shape (B, n_scalars).
    """

    def __init__(self, n_features: int, dropout: float = 0.25, n_scalars: int = 0):
        super().__init__()
        self.conv = nn.Sequential(
            _ConvBlock(n_features, 32, kernel=5, dropout=dropout),
            _ConvBlock(32, 64, kernel=5, dropout=dropout),
            _ConvBlock(64, 64, kernel=3, dropout=dropout),
        )
        self.head = nn.Linear(64 * 4 + n_scalars, 1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, s: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # (B, L, C) → (B, C, L) for Conv1d
        x = self.conv(x)  # (B, 64, L)
        x = concat_pool(x)  # (B, 192)
        if s is not None:
            x = torch.cat([x, s], dim=-1)
        return self.head(x).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# Non-causal TCN with concat pooling
# ---------------------------------------------------------------------------


class _TCNBlockPooled(nn.Module):
    """Symmetric-pad dilated Conv1d block (non-causal).

    Unlike the causal _TCNBlock in sequence.py, padding is symmetric so the
    receptive field is centred on each timestep.  Appropriate when the full
    curve is available at inference (within-cycle SOH estimation).

    GroupNorm instead of BatchNorm for the same reason as _ConvBlock: LOBO
    shift.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 5,
        dilation: int = 1,
        dropout: float = 0.25,
    ):
        super().__init__()
        pad = (kernel - 1) * dilation // 2  # symmetric dilation padding
        self.conv = weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=pad)
        )
        self.gn = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        # 1×1 residual if channel dims differ
        self.residual = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop(self.relu(self.gn(self.conv(x))))
        return out + self.residual(x)


class TCNRegPooled(nn.Module):
    """3-block non-causal dilated TCN for SOH regression.

    Dilations 1 → 2 → 4 expand the receptive field across the curve.
    concat_pool([mean, max, last]) aggregates the temporal dimension.

    Input:  (batch, n_points, channels)
    Output: (batch,) scalar

    Parameters
    ----------
    n_scalars : int
        See CNNReg for details.  Same hybrid-head pattern.
    """

    def __init__(
        self,
        n_features: int,
        n_channels: int = 48,
        dropout: float = 0.25,
        n_scalars: int = 0,
    ):
        super().__init__()

        dilations = [
            1,
            2,
            4,
            8,
            16,
            32,
        ]  # 6 blocks with exponentially increasing dilation
        layers = []
        in_ch = n_features
        for d in dilations:
            layers.append(
                _TCNBlockPooled(
                    in_ch,
                    n_channels,
                    kernel=3,
                    dilation=d,
                    dropout=dropout,
                )
            )
            in_ch = n_channels

        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(n_channels * 4 + n_scalars, 1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, s: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # (B, L, C) → (B, C, L)
        x = self.net(x)  # (B, n_channels, L)
        x = concat_pool(x)  # (B, n_channels*3)
        if s is not None:
            x = torch.cat([x, s], dim=-1)
        return self.head(x).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# Bidirectional LSTM with full-sequence pooling
# ---------------------------------------------------------------------------


class LSTMRegPooled(nn.Module):
    """Bidirectional LSTM with mean+max pooling over all timesteps.

    For within-cycle charge curves the full 128-step sequence is available,
    so there is no causal constraint.  Two improvements over the last-hidden-
    state LSTM in sequence.py:

    1. Bidirectional: forward arm sees rising voltage → cutoff knee;
       backward arm reads cutoff → start, emphasising different shape features.
    2. Pool all L hidden states instead of only the last:
         [mean(all), max(all), h_fwd_final, h_bwd_final] → 6*H → FC(1)

    Parameters
    ----------
    n_scalars : int
        Optional scalar features concatenated before the head (same hybrid
        pattern as CNNReg / TCNRegPooled).
    """

    def __init__(
        self,
        n_features: int,
        hidden: int = 64,
        dropout: float = 0.25,
        n_scalars: int = 0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        # mean(2H) + max(2H) + [h_fwd, h_bwd](2H) = 6H
        self.fc = nn.Linear(6 * hidden + n_scalars, 1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, s: torch.Tensor | None = None
    ) -> torch.Tensor:
        out, (h, _) = self.lstm(x)  # out:(B,L,2H)  h:(2,B,H)
        mean_pool = out.mean(dim=1)  # (B, 2H)
        max_pool = out.max(dim=1).values  # (B, 2H)
        final = torch.cat([h[0], h[1]], dim=-1)  # (B, 2H)
        rep = self.drop(torch.cat([mean_pool, max_pool, final], dim=-1))  # (B, 6H)
        if s is not None:
            rep = torch.cat([rep, s], dim=-1)
        return self.fc(rep).squeeze(-1)


# ---------------------------------------------------------------------------
# Bidirectional GRU with full-sequence pooling
# ---------------------------------------------------------------------------


class GRURegPooled(nn.Module):
    """Bidirectional GRU with mean+max+final pooling — identical harness to LSTMRegPooled.

    GRU has 3 gates vs LSTM's 4, so it is slightly lighter (~27k vs ~36k params
    at hidden=64) with comparable inductive bias.  Same pooling strategy:
        [mean(all), max(all), h_fwd_final, h_bwd_final] → 6H → FC(1)

    Parameters
    ----------
    n_scalars : int
        Optional scalar features concatenated before the head (same hybrid
        pattern as LSTMRegPooled).
    """

    def __init__(
        self,
        n_features: int,
        hidden: int = 64,
        dropout: float = 0.0,
        n_scalars: int = 0,
    ):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(6 * hidden + n_scalars, 1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, s: torch.Tensor | None = None
    ) -> torch.Tensor:
        out, h = self.gru(x)  # out:(B,L,2H)  h:(2,B,H)
        mean_pool = out.mean(dim=1)  # (B, 2H)
        max_pool = out.max(dim=1).values  # (B, 2H)
        final = torch.cat([h[0], h[1]], dim=-1)  # (B, 2H)
        rep = self.drop(torch.cat([mean_pool, max_pool, final], dim=-1))  # (B, 6H)
        if s is not None:
            rep = torch.cat([rep, s], dim=-1)
        return self.fc(rep).squeeze(-1)


# ---------------------------------------------------------------------------
# Small Transformer encoder with mean+max pooling
# ---------------------------------------------------------------------------


class TransformerReg(nn.Module):
    """Transformer encoder with learnable positional embeddings and mean+max pooling.

    Designed for capacity parity with the RNN models (~21k params at the
    defaults below).  Uses Pre-LayerNorm (norm_first=True) for training
    stability without a warmup schedule.

    Architecture:
        input_proj: C → d_model
        pos_embed:  learnable (n_points, d_model)
        N × TransformerEncoderLayer(d_model, nhead, dim_feedforward, Pre-LN)
        pool: [mean, max] → 2*d_model → FC(1)

    Default param count (C=3, N=128, d_model=32, nhead=4, dim_ff=64, nlayers=2):
        input_proj: 128  pos_embed: 4096  encoder: 17,088  head: 65  → ~21,377

    Parameters
    ----------
    n_points : int
        Fixed sequence length (must match the resampled curve length).  Default 128.
    n_scalars : int
        Optional scalar features (same hybrid pattern).
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 32,
        nhead: int = 4,
        dim_feedforward: int = 64,
        nlayers: int = 2,
        dropout: float = 0.0,
        n_points: int = 128,
        n_scalars: int = 0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        # Learnable positional embeddings: one vector per timestep
        self.pos_embed = nn.Parameter(torch.zeros(n_points, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN: stable without warmup
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * d_model + n_scalars, 1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, s: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x: (B, L, C) — all-valid curves (no padding in the within-cycle task)
        x = self.input_proj(x)  # (B, L, d_model)
        x = x + self.pos_embed[: x.size(1)]  # add positional signal
        # src_key_padding_mask: True = ignore position; our mask is True=valid → invert
        pad_mask = ~mask  # (B, L) all-False for full curves
        x = self.encoder(x, src_key_padding_mask=pad_mask)  # (B, L, d_model)
        mean_pool = x.mean(dim=1)  # (B, d_model)
        max_pool = x.max(dim=1).values  # (B, d_model)
        rep = self.drop(torch.cat([mean_pool, max_pool], dim=-1))  # (B, 2*d_model)
        if s is not None:
            rep = torch.cat([rep, s], dim=-1)
        return self.fc(rep).squeeze(-1)


# ---------------------------------------------------------------------------
# Per-sample normalization
# ---------------------------------------------------------------------------
import numpy as np


def per_sample_scale(X: np.ndarray) -> np.ndarray:
    """Z-score each sample independently over the time axis, per channel.

    Input : (n, L, C)
    Output: (n, L, C) — every sample has mean=0, std=1 per channel.

    Replaces the global channel StandardScaler.  With fixed-length resampled
    charge curves, duration information is already gone; what remains is curve
    shape (CC slope, CV decay, temperature rise), which this normalization
    preserves while removing the between-sample amplitude shift that caused
    high-SOH predictions to be compressed toward the training mean.
    """
    mu = X.mean(axis=1, keepdims=True)  # (n, 1, C)
    sig = X.std(axis=1, keepdims=True).clip(1e-8)  # (n, 1, C)
    return ((X - mu) / sig).astype(np.float32)


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------


def jitter(
    X: torch.Tensor,
    sigma: float = 0.005,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Add small Gaussian noise to each channel independently.

    Applied only to training data; never to validation or test.

    Parameters
    ----------
    X : (batch, n_points, channels) float32
    sigma : standard deviation of the noise (default 0.005 ≈ 5 mV / 5 mA / 0.05 °C)
    rng : optional torch Generator for reproducibility

    Returns
    -------
    Augmented X, same shape.
    """
    noise = torch.zeros_like(X)
    noise.normal_(mean=0.0, std=sigma, generator=rng)
    return X + noise
