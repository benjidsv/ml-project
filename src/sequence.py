"""Sequence models and training utilities for notebook 04 (across-cycle DL forecasting)."""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


def make_sequences(df, channels: list[str], target: str, L: int):
    """
    Build one left-padded sliding window per row (predict target at cycle t from
    the L cycles ending at t, inclusive). Emits exactly len(df) samples.

    Padding strategy: left-pad with zeros so the valid data is always right-aligned
    (position L-1 is always the current cycle). After z-scoring the input df, zeros
    represent the feature mean, which is the least-informative fill value for the model.

    Returns:
        X     : (n, L, C) float32  — feature windows (already scaled by caller)
        y     : (n,)      float32  — targets
        mask  : (n, L)    bool     — True = valid timestep (not pad)
        groups: (n,)      object   — battery_id
        cidx  : (n,)      int64    — cycle_index
    """
    X_list, y_list, mask_list, groups_list, cidx_list = [], [], [], [], []
    C = len(channels)

    for bid, grp in df.groupby("battery_id"):
        grp = grp.sort_values("cycle_index").reset_index(drop=True)
        feat = grp[channels].to_numpy(dtype=np.float32)  # (T, C)
        tgt = grp[target].to_numpy(dtype=np.float32)  # (T,)
        T = len(feat)

        for t in range(T):
            start = max(0, t - L + 1)
            seg = feat[start : t + 1]  # (actual_len, C)
            actual_len = len(seg)
            if actual_len < L:
                seg = np.concatenate(
                    [np.zeros((L - actual_len, C), dtype=np.float32), seg], axis=0
                )  # (L, C)
            mask = np.zeros(L, dtype=bool)
            mask[L - actual_len :] = True  # True = valid

            X_list.append(seg)
            y_list.append(tgt[t])
            mask_list.append(mask)
            groups_list.append(bid)
            cidx_list.append(int(grp["cycle_index"].iloc[t]))

    return (
        np.stack(X_list),  # (n, L, C)
        np.array(y_list, dtype=np.float32),  # (n,)
        np.stack(mask_list),  # (n, L)
        np.array(groups_list),  # (n,)
        np.array(cidx_list, dtype=np.int64),  # (n,)
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class LSTMReg(nn.Module):
    """Single-layer LSTM → last valid hidden state → scalar.

    Parameters
    ----------
    n_scalars : int
        Number of scalar features to concatenate with the last hidden state
        before the FC head.  0 = pure sequence model (default).
        When > 0, forward() expects a third argument s of shape (B, n_scalars).
    """

    def __init__(
        self,
        n_features: int,
        hidden: int = 32,
        dropout: float = 0.25,
        n_scalars: int = 0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden + n_scalars, 1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, s: torch.Tensor | None = None
    ) -> torch.Tensor:
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        h_out = self.drop(h[-1])  # (B, hidden)
        if s is not None:
            h_out = torch.cat([h_out, s], dim=-1)
        return self.fc(h_out).squeeze(-1)


class GRUReg(nn.Module):
    """Single-layer GRU → last valid hidden state → scalar."""

    def __init__(self, n_features: int, hidden: int = 32, dropout: float = 0.25):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        return self.fc(self.drop(h[-1])).squeeze(-1)


class _TCNBlock(nn.Module):
    """One causal dilated conv block with residual connection."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.25,
    ):
        super().__init__()
        # Symmetric padding = (k-1)*d; trimming the right side gives causal output
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)[:, :, : x.shape[2]]  # causal trim: keep only L positions
        return self.act(self.drop(out) + self.res(x))


class TCNReg(nn.Module):
    """Two-block causal TCN → last position features → scalar (nowcast).

    Data is left-padded so position L-1 is always the current (most recent) cycle.
    The causal TCN at position L-1 sees the last (receptive_field) timesteps,
    which is exactly the desired L-cycle lookback.
    """

    def __init__(self, n_features: int, n_channels: int = 32, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            _TCNBlock(n_features, n_channels, dilation=1, dropout=dropout),
            _TCNBlock(n_channels, n_channels, dilation=2, dropout=dropout),
        )
        self.fc = nn.Linear(n_channels, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # (B, L, C) → (B, C, L) for Conv1d; take last position (current cycle)
        return self.fc(self.net(x.transpose(1, 2))[:, :, -1]).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    model: nn.Module,
    X_tr: np.ndarray,
    mask_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    mask_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 30,
    scheduler: str | None = None,
    augment_fn=None,
    S_tr: np.ndarray | None = None,
    S_val: np.ndarray | None = None,
    progress: bool = True,
) -> tuple:
    """Train with early stopping on val MSE loss.

    Parameters
    ----------
    scheduler : "cosine" | "plateau" | None
        "cosine"  — CosineAnnealingLR over the full epoch budget.
        "plateau" — ReduceLROnPlateau (patience=10, factor=0.5) on val loss.
    augment_fn : callable(xb: Tensor) -> Tensor | None
        Applied to each training batch on-device before the forward pass.
        Enables per-iteration augmentation (e.g. jitter) so noise resamples
        every batch rather than being fixed for the whole training run.
    S_tr, S_val : (n, n_scalars) float32 ndarray | None
        Optional scalar features (e.g. the 9 charge scalars) to pass as the
        third argument to model.forward(x, mask, s).  Must be pre-scaled by
        the caller.  None → model called as forward(x, mask) (original behaviour).

    Returns (model_at_best_val, train_loss_curve, val_loss_curve, best_epoch).
    """
    # Pre-move the full train set to device once; index on-device per batch.
    # Eliminates N_batches × N_epochs × N_folds CPU→device transfers (the main bottleneck on MPS).
    Xt = torch.tensor(X_tr, dtype=torch.float32).to(device)
    mt = torch.tensor(mask_tr, dtype=torch.bool).to(device)
    yt = torch.tensor(y_tr, dtype=torch.float32).to(device)
    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    mv = torch.tensor(mask_val, dtype=torch.bool).to(device)
    yv = torch.tensor(y_val, dtype=torch.float32).to(device)
    Sv = (
        torch.tensor(S_val, dtype=torch.float32).to(device)
        if S_val is not None
        else None
    )
    St = (
        torch.tensor(S_tr, dtype=torch.float32).to(device)
        if S_tr is not None
        else None
    )
    n_tr = len(Xt)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.L1Loss()

    sched = None
    if scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs, eta_min=lr * 0.01
        )
    elif scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", patience=10, factor=0.5, min_lr=lr * 0.01
        )

    best_val, best_state, best_ep = float("inf"), None, 0
    train_curve, val_curve = [], []
    no_improve = 0

    epoch_bar = tqdm(
        range(1, epochs + 1),
        leave=False,
        unit="ep",
        bar_format="{l_bar}{bar:20}{r_bar}",
        disable=not progress,
    )
    for ep in epoch_bar:
        model.train()
        ep_loss = 0.0
        perm = torch.randperm(n_tr, device=device)
        for i in range(0, n_tr, batch_size):
            idx = perm[i : i + batch_size]
            xb, mb, yb = Xt[idx], mt[idx], yt[idx]
            sb = St[idx] if St is not None else None
            if augment_fn is not None:
                xb = augment_fn(xb)
            opt.zero_grad()
            loss = loss_fn(model(xb, mb, sb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * len(yb)
        train_curve.append(ep_loss / n_tr)

        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xv, mv, Sv), yv).item()
        val_curve.append(vl)

        if sched is not None:
            sched.step(vl) if scheduler == "plateau" else sched.step()

        epoch_bar.set_postfix(val=f"{vl:.4f}", best=f"{best_val:.4f}", ep=ep)

        if vl < best_val - 1e-6:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_ep = ep
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_curve, val_curve, best_ep


@torch.no_grad()
def predict_model(
    model: nn.Module,
    X: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
    S: np.ndarray | None = None,
) -> np.ndarray:
    """Batched inference; predictions clipped to [0, 1].

    Parameters
    ----------
    S : (n, n_scalars) float32 ndarray | None
        Optional pre-scaled scalar features, passed as model.forward(x, mask, s).
    """
    model.eval()
    preds = []
    Xt = torch.tensor(X, dtype=torch.float32)
    mt = torch.tensor(mask, dtype=torch.bool)
    St = torch.tensor(S, dtype=torch.float32) if S is not None else None
    for i in range(0, len(Xt), batch_size):
        sb = St[i : i + batch_size].to(device) if St is not None else None
        preds.append(
            model(
                Xt[i : i + batch_size].to(device), mt[i : i + batch_size].to(device), sb
            )
            .cpu()
            .numpy()
        )
    return np.clip(np.concatenate(preds), 0.0, 1.0)
