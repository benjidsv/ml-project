"""On-device augmentation transforms for the voltage-grid representation.

All transforms operate on batched float32 tensors that are already on the
training device (CPU / CUDA / MPS) so they run inside the training loop with
zero host–device transfer overhead.

Channel layout (5-channel representation):
    0  C-rate       |I| / C_nom  (normalised current)
    1  ΔT           T − T_start  (temperature rise from CC onset)
    2  t_elapsed    seconds from CC-segment start
    3  Q_frac       cumulative charge / C_nom  (incremental capacity integral)
    4  dQ/dV        incremental capacity curve  (Ah/V)

Augmentation rationale
----------------------
The primary transfer-gap driver is that channels 0 (C-rate) and 2 (t_elapsed)
are protocol fingerprints: CALCE uses fixed 0.5C/1C/3C; NASA uses CC
controlled/randomized.  rate_warp simulates an alternative charge rate for
the same cycle, teaching the model that SOH-relevant information lives in the
*shape* of Q(V) and dQ/dV rather than absolute rate/timing.  Q_frac and dQdV
are deliberately left unchanged because they are already rate-invariant — a
faster charge still delivers the same Q by the time V reaches V_HI.
"""

from __future__ import annotations

from typing import Callable

import torch


# ---------------------------------------------------------------------------
# Individual transforms
# ---------------------------------------------------------------------------

CH_CRATE    = 0
CH_DT       = 1
CH_TELAPSED = 2
CH_QFRAC    = 3
CH_DQDV     = 4


def rate_warp(
    xb: torch.Tensor,
    lo: float = 0.8,
    hi: float = 1.25,
) -> torch.Tensor:
    """Simulate a different CC charge rate for each sample in the batch.

    Applies a per-sample uniform random scale factor s ~ U(lo, hi):
        C-rate    ← C-rate × s          (faster rate → higher current)
        t_elapsed ← t_elapsed / s       (faster rate → shorter time)
        ΔT, Q_frac, dQ/dV unchanged     (rate-invariant channels)

    The transform is applied in the *scaled* domain (after global_scale), so
    the magnitude of each multiplicative nudge is relative to the standardised
    scale of each channel — residual effect is small and realistic.

    Parameters
    ----------
    xb : (B, L, C) float32 tensor on the training device.
    lo, hi : lower and upper bounds of the per-sample scale factor U distribution.

    Returns
    -------
    xb_aug : (B, L, C) float32 — modified copy (does not mutate xb).
    """
    B = xb.shape[0]
    s = torch.empty(B, device=xb.device, dtype=xb.dtype).uniform_(lo, hi)  # (B,)
    s2d = s.view(B, 1, 1)  # broadcast over (L, C)

    xb_aug = xb.clone()
    xb_aug[:, :, CH_CRATE]    = xb[:, :, CH_CRATE]    * s2d.squeeze(-1)
    xb_aug[:, :, CH_TELAPSED] = xb[:, :, CH_TELAPSED] / s2d.squeeze(-1)
    return xb_aug


def channel_jitter(
    xb: torch.Tensor,
    sigma: float = 0.01,
) -> torch.Tensor:
    """Add i.i.d. Gaussian noise to every channel of every position.

    Parameters
    ----------
    xb    : (B, L, C) float32 tensor.
    sigma : standard deviation of the additive noise (in the scaled domain).

    Returns
    -------
    xb_aug : (B, L, C) — modified copy.
    """
    return xb + torch.randn_like(xb) * sigma


# ---------------------------------------------------------------------------
# Composed augmentation factory
# ---------------------------------------------------------------------------


def time_warp(
    xb: torch.Tensor,
    lo: float = 0.8,
    hi: float = 1.25,
) -> torch.Tensor:
    """Scale the t_elapsed channel by a per-sample random factor.

    Unlike ``rate_warp`` (which jointly scales C-rate and t_elapsed to simulate
    a different charge rate), ``time_warp`` perturbs only t_elapsed.  This
    simulates SOC-window or measurement-timing variation independently of the
    charge rate — a complementary perturbation axis for the aug sweep.

    s ~ U(lo, hi):
        t_elapsed ← t_elapsed × s
    """
    B = xb.shape[0]
    s = torch.empty(B, device=xb.device, dtype=xb.dtype).uniform_(lo, hi)
    xb_aug = xb.clone()
    xb_aug[:, :, CH_TELAPSED] = xb[:, :, CH_TELAPSED] * s.view(B, 1)
    return xb_aug


def channel_dropout(
    xb: torch.Tensor,
    p: float = 0.1,
) -> torch.Tensor:
    """Per-sample channel dropout: zero one random channel with probability p.

    Parameters
    ----------
    xb : (B, L, C) float32 tensor.
    p  : probability per sample that one channel is zeroed.

    Returns
    -------
    xb_aug : (B, L, C) — modified copy.
    """
    B, L, C = xb.shape
    xb_aug = xb.clone()
    do_drop = torch.bernoulli(torch.full((B,), p, device=xb.device)).bool()
    if do_drop.any():
        ch = torch.randint(0, C, (B,), device=xb.device)
        for i in range(B):
            if do_drop[i]:
                xb_aug[i, :, ch[i]] = 0.0
    return xb_aug


# ---------------------------------------------------------------------------
# Mask-aware transforms  (signature: (xb, mb, ...) -> (xb, mb))
#
# These require the validity mask and cannot ride the augment_fn hook
# (which is label/mask-free).  They are called explicitly in train_model
# after augment_fn when their controlling kwargs are non-zero.
# ---------------------------------------------------------------------------


def end_cutout(
    xb: torch.Tensor,
    mb: torch.Tensor,
    frac_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """End-of-discharge masking: crop the low-voltage tail of each sequence.

    In the voltage-grid representation the valid region is a leading block
    [T…T F…F] where index 0 = V_HI (start of discharge) and the last True
    position = V_cutoff (end of discharge).  Cropping the tail creates
    invariance to the CALCE (3.0 V) vs NASA (2.7 V) cutoff gap.

    Per sample: draw f ~ U(0, frac_max); zero the last round(f · len_i)
    valid positions and mark them masked.  Samples with valid length < 4 are
    skipped.

    Parameters
    ----------
    xb      : (B, L, C) float32 tensor.
    mb      : (B, L) bool tensor — True = valid.
    frac_max: maximum fraction of valid positions to crop (e.g. 0.30).

    Returns
    -------
    xb_aug, mb_aug : same shape as inputs.
    """
    B = xb.shape[0]
    xb_aug = xb.clone()
    mb_aug = mb.clone()
    lengths = mb.sum(dim=1)  # (B,)
    fracs = torch.empty(B, device=xb.device).uniform_(0.0, frac_max)
    for i in range(B):
        li = int(lengths[i].item())
        if li < 4:
            continue
        drop = round(float(fracs[i].item()) * li)
        if drop <= 0:
            continue
        start = li - drop
        xb_aug[i, start:li] = 0.0
        mb_aug[i, start:li] = False
    return xb_aug, mb_aug


def grid_shift(
    xb: torch.Tensor,
    mb: torch.Tensor,
    max_shift: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift the valid discharge block along the voltage grid axis.

    Simulates OCV offset (temperature- or chemistry-induced plateau shift).
    Each sample's leading valid block [0..len_i-1] is shifted by a random
    integer k ~ U(-max_shift, +max_shift):

    * k > 0 (shift right / toward lower voltage): valid block moves to
      [k .. min(k+len_i, L)-1]; the first k positions become zero-fill.
      Data truncated if k+len_i > L.
    * k < 0 (shift left / toward higher voltage): the first |k| positions
      of the valid data are lost; the block becomes [0 .. len_i-|k|-1].

    Samples with valid length < 4 are skipped.

    Parameters
    ----------
    xb        : (B, L, C) float32 tensor.
    mb        : (B, L) bool tensor — True = valid.
    max_shift : maximum absolute shift in grid positions.

    Returns
    -------
    xb_aug, mb_aug : same shape as inputs.
    """
    if max_shift == 0:
        return xb, mb
    B, L, _ = xb.shape
    xb_aug = torch.zeros_like(xb)
    mb_aug = torch.zeros_like(mb)
    lengths = mb.sum(dim=1)
    shifts = torch.randint(-max_shift, max_shift + 1, (B,), device=xb.device)
    for i in range(B):
        li = int(lengths[i].item())
        if li < 4:
            continue
        k = int(shifts[i].item())
        if k == 0:
            xb_aug[i, :li] = xb[i, :li]
            mb_aug[i, :li] = True
        elif k > 0:
            new_len = min(li, L - k)
            if new_len > 0:
                xb_aug[i, k:k + new_len] = xb[i, :new_len]
                mb_aug[i, k:k + new_len] = True
        else:
            abs_k = -k
            new_len = li - abs_k
            if new_len > 0:
                xb_aug[i, :new_len] = xb[i, abs_k:abs_k + new_len]
                mb_aug[i, :new_len] = True
    return xb_aug, mb_aug


# ---------------------------------------------------------------------------
# Composed augmentation factory
# ---------------------------------------------------------------------------


def make_augment(
    rate_warp_lo: float = 0.8,
    rate_warp_hi: float = 1.25,
    jitter_sigma: float = 0.01,
    time_warp_lo: float = 1.0,
    time_warp_hi: float = 1.0,
    channel_dropout_p: float = 0.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a composed augmentation function compatible with train_model's
    ``augment_fn`` hook (signature: ``(xb: Tensor) -> Tensor``).

    Parameters
    ----------
    rate_warp_lo / hi    : bounds of the rate-warp U distribution.
        Set both to 1.0 to disable rate warping.
    jitter_sigma         : noise std for channel_jitter.  0.0 disables.
    time_warp_lo / hi    : bounds for time_warp (t_elapsed-only scale).
        Set both to 1.0 to disable (default).
    channel_dropout_p    : per-sample prob of zeroing one random channel.
        0.0 disables (default).

    Apply order: rate_warp → time_warp → channel_jitter → channel_dropout.

    Example
    -------
        aug = make_augment(rate_warp_lo=0.8, rate_warp_hi=1.25, jitter_sigma=0.01)
        results = run_grouped_cv(lambda: VGGRUReg(n_features=4), ...,
                                  train_kwargs={"augment_fn": aug})
    """
    do_warp      = not (rate_warp_lo == 1.0 and rate_warp_hi == 1.0)
    do_jitter    = jitter_sigma > 0.0
    do_time_warp = not (time_warp_lo == 1.0 and time_warp_hi == 1.0)
    do_ch_drop   = channel_dropout_p > 0.0

    def _augment(xb: torch.Tensor) -> torch.Tensor:
        if do_warp:
            xb = rate_warp(xb, lo=rate_warp_lo, hi=rate_warp_hi)
        if do_time_warp:
            xb = time_warp(xb, lo=time_warp_lo, hi=time_warp_hi)
        if do_jitter:
            xb = channel_jitter(xb, sigma=jitter_sigma)
        if do_ch_drop:
            xb = channel_dropout(xb, p=channel_dropout_p)
        return xb

    return _augment
