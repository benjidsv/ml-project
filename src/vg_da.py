"""Domain-adaptation training loop for the voltage-grid SOH model.

Provides ``train_model_da``, a sibling to ``sequence.train_model`` that adds
an auxiliary domain-invariance objective on top of the standard L1 regression
loss.  Everything else is identical to the baseline loop:
  - Adam + cosine-anneal LR (1e-3 → 1e-5 over 300 epochs)
  - gradient clipping 1.0
  - smoothed (5-epoch rolling mean) early stopping
  - best-single-epoch model checkpoint restored at the end

Two domain-adaptation modes
----------------------------
``"coral"``  (Deep CORAL, Sun & Saenko 2016)
    Aligns per-domain feature covariances.  Stable and parameter-free beyond
    ``da_weight``.  Use this first.  Total loss = L1 + da_weight * CORAL.

``"dann"``   (DANN, Ganin et al. 2016)
    Gradient-reversal layer + MLP domain classifier.  Lambda is ramped from 0
    to 1 over training so the classifier has time to warm up.
    Total loss = L1 + da_weight * CE(domain_clf(GRL(feats)), domain_ids).

Domain labels
-------------
Use study-group labels (NASA_CTRL, NASA_RW, CALCE_CS2_T1, …), NOT dataset
labels (nasa / calce).  Under LODO the held dataset is absent from training,
so aligning across study protocols achieves the desired protocol-invariance
even when only one ds_group is present in the training pool.

Model requirement
-----------------
The model must support ``forward(x, mask, s, return_features=True)`` which
returns ``(pred, feats)`` where ``feats`` is (B, feat_dim).  ``VGGRUReg``
supports this after the ``return_features`` hook was added to its forward pass.

Domain-balanced batch sampler
------------------------------
CORAL requires ≥2 domains per batch to compute meaningful pairwise covariances.
A domain-balanced sampler draws ~equal numbers of samples from each study group
represented in the training fold, so every batch contains multiple protocols.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Gradient-reversal layer (for DANN)
# ---------------------------------------------------------------------------


class GradientReversal(torch.autograd.Function):
    """Straight-through forward, negated-and-scaled backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:  # type: ignore[override]
        ctx.save_for_backward(torch.tensor(lam))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (lam,) = ctx.saved_tensors
        return -lam.item() * grad_output, None


def grad_reverse(x: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    return GradientReversal.apply(x, lam)


# ---------------------------------------------------------------------------
# CORAL loss
# ---------------------------------------------------------------------------


def coral_loss(features: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
    """Deep CORAL loss: mean pairwise Frobenius distance between covariance matrices.

    Parameters
    ----------
    features   : (B, d) float32 — batch feature vectors (post-dropout pooled rep).
    domain_ids : (B,)   int64  — integer domain labels for each sample.

    Returns
    -------
    loss : scalar tensor.  0.0 when fewer than 2 domains are present in the batch.
    """
    domains = torch.unique(domain_ids)
    # Keep only domains with >=2 samples (needed for unbiased covariance)
    valid = [dom for dom in domains if (domain_ids == dom).sum() >= 2]
    if len(valid) < 2:
        return features.new_zeros(())

    d = features.shape[1]
    K = len(valid)

    # One covariance per valid domain → (K, d, d)  [K kernel calls, no Python pair loop]
    covs = torch.stack([_cov(features[domain_ids == dom]) for dom in valid])

    # All pairwise Frobenius distances via broadcasting — zero Python loops over pairs
    diff = covs.unsqueeze(0) - covs.unsqueeze(1)                         # (K, K, d, d)
    i_idx, j_idx = torch.triu_indices(K, K, offset=1, device=features.device)
    pair_sq_frob = diff[i_idx, j_idx].pow(2).sum(dim=(-2, -1))          # (n_pairs,)

    return pair_sq_frob.mean() / (4.0 * d * d)


def _cov(x: torch.Tensor) -> torch.Tensor:
    """Unbiased covariance matrix (d, d) of a (n, d) matrix."""
    n = x.shape[0]
    xc = x - x.mean(dim=0, keepdim=True)
    return (xc.T @ xc) / (n - 1)


# ---------------------------------------------------------------------------
# Domain-balanced batch sampler
# ---------------------------------------------------------------------------


class _DomainBatchSampler:
    """Precomputes per-domain index buckets once; rebuilds only shuffles each epoch.

    The static per-domain nonzero lookups (O(N) tensor scans) happen once at
    construction.  Each ``__call__`` only re-shuffles the buckets and returns a
    single ``(n_batches, eff_batch_size)`` int64 tensor — no Python list of 160
    separate tensor objects, no repeated domain scans.

    Iterating over the returned tensor in the training loop yields one 1-D index
    tensor per batch (a cheap row-view, not a copy).
    """

    def __init__(
        self,
        domain_ids: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> None:
        unique_domains = torch.unique(domain_ids)
        self.n_domains = int(len(unique_domains))
        self.per_domain = max(batch_size // self.n_domains, 1)
        self.device = device
        # Computed once for the lifetime of training — never rebuilt across epochs
        self._buckets: list[torch.Tensor] = [
            (domain_ids == d).nonzero(as_tuple=False).squeeze(1)
            for d in unique_domains
        ]
        self._max_len = max(len(b) for b in self._buckets)

    def __call__(self) -> torch.Tensor:
        """Shuffle buckets and return a ``(n_batches, eff_batch_size)`` index tensor."""
        per_domain = self.per_domain
        max_len = self._max_len
        n_domains = self.n_domains

        # Shuffle each bucket and pad to max_len by repeating its own samples
        cols: list[torch.Tensor] = []
        for bucket in self._buckets:
            perm = torch.randperm(len(bucket), device=self.device)
            shuffled = bucket[perm]
            reps = math.ceil(max_len / len(shuffled))
            cols.append(shuffled.repeat(reps)[:max_len])

        # Pad each col to the next multiple of per_domain for a clean reshape
        n_chunks = math.ceil(max_len / per_domain)
        pad_to = n_chunks * per_domain
        padded_cols: list[torch.Tensor] = []
        for col in cols:
            extra = pad_to - len(col)
            if extra:
                col = torch.cat([col, col[-extra:]])
            padded_cols.append(col)

        # Single reshape+permute replaces n_chunks Python torch.cat calls
        # (D, pad_to) → (D, K, P) → (K, D, P) → (K, D*P)
        stacked = torch.stack(padded_cols)                              # (D, pad_to)
        batches = (
            stacked
            .reshape(n_domains, n_chunks, per_domain)                  # (D, K, P)
            .permute(1, 0, 2)                                           # (K, D, P)
            .reshape(n_chunks, n_domains * per_domain)                  # (K, D*P)
        )
        return batches


# ---------------------------------------------------------------------------
# Window dropout (mask-aware)
# ---------------------------------------------------------------------------


def _window_dropout(
    xb: torch.Tensor,
    mb: torch.Tensor,
    dropout_frac: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero a contiguous random block of each sample's valid region.

    Parameters
    ----------
    xb : (B, L, C) float32 — input tensor.
    mb : (B, L)    bool   — validity mask.
    dropout_frac  : fraction of each sample's valid length to drop (0.0–0.5).

    Returns modified (xb, mb) copies (no in-place mutation).
    """
    B, L, _ = xb.shape
    lengths = mb.sum(dim=1)                                                   # (B,)
    drop_lens = (lengths.float() * dropout_frac).clamp(min=1).long()         # (B,)
    drop_lens[lengths < 4] = 0                                                # skip short seqs

    # Vectorised random start: uniform in [0, lengths - drop_lens)
    valid_range = (lengths - drop_lens).clamp(min=0).float()                  # (B,)
    starts = (torch.rand(B, device=xb.device) * valid_range).long()           # (B,)

    # Build boolean drop mask without any Python loop
    pos = torch.arange(L, device=xb.device).unsqueeze(0)                      # (1, L)
    drop_mask = (pos >= starts.unsqueeze(1)) & \
                (pos < (starts + drop_lens).unsqueeze(1))                      # (B, L)

    return xb.masked_fill(drop_mask.unsqueeze(-1), 0.0), mb & ~drop_mask


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train_model_da(
    model: nn.Module,
    X_tr: np.ndarray,
    mask_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    mask_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    domains_tr: np.ndarray,             # (n_tr,) int — study-group ids, training fold
    da_mode: str = "coral",             # "coral" | "dann"
    da_weight: float = 0.1,
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 50,
    scheduler: str | None = "cosine",
    augment_fn: Callable | None = None,
    window_dropout: float = 0.0,
    mixup_alpha: float = 0.0,
    progress: bool = True,
    es_window: int = 5,
    use_amp: bool = False,
) -> tuple:
    """Train with L1 regression loss + optional domain-adaptation auxiliary loss.

    Mirrors sequence.train_model exactly for all hyperparameters, early-stopping
    logic, and device handling.  The only additions are:
      - ``domains_tr``   : integer study-group labels for the training fold.
      - ``da_mode``      : "coral" or "dann".
      - ``da_weight``    : weight of the auxiliary loss term.
      - ``window_dropout``: fraction of valid region to randomly zero per batch.
      - ``mixup_alpha``  : Beta distribution α for mixup on (x, y).  0 = off.

    Returns (model_at_best_val, train_loss_curve, val_loss_curve, best_epoch).
    """
    if da_mode not in {"coral", "dann"}:
        raise ValueError(f"da_mode must be 'coral' or 'dann'; got {da_mode!r}")

    # ── pre-move full tensors to device (eliminates per-batch transfers) ──
    Xt = torch.tensor(X_tr,   dtype=torch.float32).to(device)
    mt = torch.tensor(mask_tr, dtype=torch.bool  ).to(device)
    yt = torch.tensor(y_tr,   dtype=torch.float32).to(device)
    Xv = torch.tensor(X_val,  dtype=torch.float32).to(device)
    mv = torch.tensor(mask_val, dtype=torch.bool ).to(device)
    yv = torch.tensor(y_val,  dtype=torch.float32).to(device)
    dt = torch.tensor(domains_tr, dtype=torch.int64).to(device)  # domain ids

    # Encode domain labels as contiguous 0..K-1 (splitter may leave gaps)
    unique_d, dt = torch.unique(dt, return_inverse=True)
    n_domains = int(len(unique_d))

    n_tr = len(Xt)
    feat_dim = 4 * getattr(model, "hidden", 47)  # 4H for BiGRU (mean + max pool)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    l1_fn = nn.L1Loss()

    # DANN: build domain classifier + its optimiser
    dann_clf: nn.Module | None = None
    dann_opt: torch.optim.Optimizer | None = None
    if da_mode == "dann" and da_weight > 0.0 and n_domains >= 2:
        dann_clf = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_domains),
        ).to(device)
        dann_opt = torch.optim.Adam(dann_clf.parameters(), lr=lr, weight_decay=weight_decay)
    ce_fn = nn.CrossEntropyLoss()

    sched = None
    if scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs, eta_min=lr * 0.01
        )
    elif scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", patience=10, factor=0.5, min_lr=lr * 0.01
        )

    # AMP: only meaningful on CUDA; silently disabled on CPU/MPS
    _amp_enabled = use_amp and device.type == "cuda"
    _scaler = torch.amp.GradScaler("cuda", enabled=_amp_enabled)

    best_val, best_state, best_ep = float("inf"), None, 0
    best_smooth = float("inf")
    train_curve, val_curve = [], []
    no_improve = 0

    # Precompute domain bucket membership once — only shuffles are rebuilt per epoch
    domain_sampler = _DomainBatchSampler(dt, batch_size, device)

    epoch_bar = tqdm(
        range(1, epochs + 1),
        leave=False,
        unit="ep",
        bar_format="{l_bar}{bar:20}{r_bar}",
        disable=not progress,
    )

    for ep in epoch_bar:
        model.train()
        if dann_clf is not None:
            dann_clf.train()
        ep_loss = 0.0

        # Build domain-balanced batches for this epoch (buckets precomputed, only shuffle rebuilt)
        batches = domain_sampler()

        # DANN lambda ramp: 0 → 1 over training
        p = (ep - 1) / max(epochs - 1, 1)
        dann_lam = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0

        for idx in batches:
            xb = Xt[idx]
            mb = mt[idx]
            yb = yt[idx]
            db = dt[idx]

            # Mask-aware window dropout
            if window_dropout > 0.0:
                xb, mb = _window_dropout(xb, mb, window_dropout)

            # Standard augmentation (operates on x only)
            if augment_fn is not None:
                xb = augment_fn(xb)

            # Mixup in the input–target space
            if mixup_alpha > 0.0:
                lam_mix = float(torch.distributions.Beta(mixup_alpha, mixup_alpha).sample())
                perm_mix = torch.randperm(len(xb), device=device)
                xb = lam_mix * xb + (1 - lam_mix) * xb[perm_mix]
                yb = lam_mix * yb + (1 - lam_mix) * yb[perm_mix]
                # masks: conservative AND (require both samples to be valid)
                mb = mb & mb[perm_mix]

            opt.zero_grad()
            if dann_opt is not None:
                dann_opt.zero_grad()

            with torch.autocast(device_type=device.type, enabled=_amp_enabled):
                pred, feats = model(xb, mb, return_features=True)
                loss = l1_fn(pred, yb)

                # Domain-adaptation auxiliary loss
                da_loss = Xt.new_zeros(1).squeeze()
                if da_weight > 0.0 and n_domains >= 2:
                    if da_mode == "coral":
                        da_loss = coral_loss(feats, db)
                    elif da_mode == "dann" and dann_clf is not None:
                        feats_rev = grad_reverse(feats, dann_lam)
                        domain_logits = dann_clf(feats_rev)
                        da_loss = ce_fn(domain_logits, db)

                total_loss = loss + da_weight * da_loss

            _scaler.scale(total_loss).backward()
            _scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            _scaler.step(opt)
            if dann_opt is not None:
                _scaler.unscale_(dann_opt)
                nn.utils.clip_grad_norm_(dann_clf.parameters(), 1.0)
                _scaler.step(dann_opt)
            _scaler.update()

            ep_loss += loss.item() * len(yb)  # track regression loss only

        train_curve.append(ep_loss / n_tr)

        model.eval()
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=_amp_enabled):
            pred_val = model(Xv, mv)  # standard forward (no return_features)
            vl = l1_fn(pred_val, yv).item()
        val_curve.append(vl)

        if sched is not None:
            sched.step(vl) if scheduler == "plateau" else sched.step()

        epoch_bar.set_postfix(val=f"{vl:.4f}", best=f"{best_val:.4f}", ep=ep)

        # Checkpoint: raw best single-epoch val
        if vl < best_val - 1e-6:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_ep = ep

        # Patience: smoothed val to avoid resetting on noise troughs
        smooth_val = (
            float(np.mean(val_curve[-es_window:]))
            if len(val_curve) >= es_window
            else vl
        )
        if smooth_val < best_smooth - 1e-6:
            best_smooth = smooth_val
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_curve, val_curve, best_ep


# ---------------------------------------------------------------------------
# GroupDRO training loop
# ---------------------------------------------------------------------------


def train_model_dro(
    model: nn.Module,
    X_tr: np.ndarray,
    mask_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    mask_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    domains_tr: np.ndarray,             # (n_tr,) int — study-group ids
    group_lr: float = 0.01,             # exponential reweighting step size η
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 50,
    scheduler: str | None = "cosine",
    augment_fn: Callable | None = None,
    window_dropout: float = 0.0,
    mixup_alpha: float = 0.0,
    progress: bool = True,
    es_window: int = 5,
    use_amp: bool = False,
) -> tuple:
    """GroupDRO (Sagawa et al. 2020): minimize worst-group risk.

    Unlike CORAL/DANN which require target-domain samples in-batch,
    GroupDRO needs no target data.  It minimises a reweighted objective
    that concentrates training on the hardest protocol — directly aligned
    with LODO, where the objective is cross-dataset transfer.

    Algorithm per batch
    -------------------
    ℓ_g ← mean L1(pred, y) for samples in group g   (present groups only)
    q_g ← q_g · exp(η · ℓ_g)                        (exponential upweight)
    q   ← q / sum(q)                                  (renormalise)
    loss ← Σ_g q_g · ℓ_g                             (weighted objective)

    q is maintained globally across the full training run (not reset per batch).
    Gradient flows through ℓ_g; q is treated as a constant (detached).

    Returns (model_at_best_val, train_curve, val_curve, best_epoch).
    """
    Xt = torch.tensor(X_tr,    dtype=torch.float32).to(device)
    mt = torch.tensor(mask_tr, dtype=torch.bool   ).to(device)
    yt = torch.tensor(y_tr,    dtype=torch.float32).to(device)
    Xv = torch.tensor(X_val,   dtype=torch.float32).to(device)
    mv = torch.tensor(mask_val, dtype=torch.bool  ).to(device)
    yv = torch.tensor(y_val,   dtype=torch.float32).to(device)
    dt = torch.tensor(domains_tr, dtype=torch.int64).to(device)

    unique_d, dt = torch.unique(dt, return_inverse=True)
    n_domains = int(len(unique_d))

    n_tr = len(Xt)

    # Group weights: uniform init, updated across the full training run
    q = torch.ones(n_domains, device=device) / n_domains

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    l1_fn = nn.L1Loss(reduction="none")  # per-sample losses needed for group weighting

    sched = None
    if scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs, eta_min=lr * 0.01
        )
    elif scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", patience=10, factor=0.5, min_lr=lr * 0.01
        )

    _amp_enabled = use_amp and device.type == "cuda"
    _scaler = torch.amp.GradScaler("cuda", enabled=_amp_enabled)

    best_val, best_state, best_ep = float("inf"), None, 0
    best_smooth = float("inf")
    train_curve, val_curve = [], []
    no_improve = 0

    domain_sampler = _DomainBatchSampler(dt, batch_size, device)

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
        batches = domain_sampler()

        for idx in batches:
            xb = Xt[idx]
            mb = mt[idx]
            yb = yt[idx]
            db = dt[idx]

            if window_dropout > 0.0:
                xb, mb = _window_dropout(xb, mb, window_dropout)

            if augment_fn is not None:
                xb = augment_fn(xb)

            if mixup_alpha > 0.0:
                lam_mix = float(torch.distributions.Beta(mixup_alpha, mixup_alpha).sample())
                perm_mix = torch.randperm(len(xb), device=device)
                xb = lam_mix * xb + (1 - lam_mix) * xb[perm_mix]
                yb = lam_mix * yb + (1 - lam_mix) * yb[perm_mix]
                mb = mb & mb[perm_mix]

            opt.zero_grad()

            with torch.autocast(device_type=device.type, enabled=_amp_enabled):
                pred = model(xb, mb)
                per_sample = l1_fn(pred, yb)  # (B,) — gradients flow through this

                # Per-group mean losses (gradient-tracked)
                present, g_losses = [], []
                for g in range(n_domains):
                    mask_g = db == g
                    if mask_g.any():
                        present.append(g)
                        g_losses.append(per_sample[mask_g].mean())

                # Exponential reweight (no gradient through q)
                with torch.no_grad():
                    for i, g in enumerate(present):
                        q[g] = q[g] * math.exp(group_lr * g_losses[i].item())
                    q.clamp_(min=1e-8)  # floor: prevent float-underflow zeroing easy groups
                    q.div_(q.sum())

                loss = sum(q[g].detach() * g_losses[i] for i, g in enumerate(present))

            _scaler.scale(loss).backward()
            _scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            _scaler.step(opt)
            _scaler.update()

            ep_loss += per_sample.detach().mean().item() * len(yb)

        train_curve.append(ep_loss / n_tr)

        model.eval()
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=_amp_enabled):
            pred_val = model(Xv, mv)
            vl = nn.L1Loss()(pred_val, yv).item()
        val_curve.append(vl)

        if sched is not None:
            sched.step(vl) if scheduler == "plateau" else sched.step()

        epoch_bar.set_postfix(val=f"{vl:.4f}", best=f"{best_val:.4f}", ep=ep)

        if vl < best_val - 1e-6:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_ep = ep

        smooth_val = (
            float(np.mean(val_curve[-es_window:]))
            if len(val_curve) >= es_window
            else vl
        )
        if smooth_val < best_smooth - 1e-6:
            best_smooth = smooth_val
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Stash final group weights for observability (sum to 1; should concentrate on hardest group)
    model._dro_q = q.detach().cpu().numpy()
    _q_str = "  ".join(f"{qi:.3f}" for qi in model._dro_q)
    print(f"  DRO q: [{_q_str}]  sum={model._dro_q.sum():.4f}")

    return model, train_curve, val_curve, best_ep
