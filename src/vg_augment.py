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


def make_augment(
    rate_warp_lo: float = 0.8,
    rate_warp_hi: float = 1.25,
    jitter_sigma: float = 0.01,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a composed augmentation function compatible with train_model's
    ``augment_fn`` hook (signature: ``(xb: Tensor) -> Tensor``).

    Parameters
    ----------
    rate_warp_lo / hi : bounds of the rate-warp U distribution.
        Set both to 1.0 to disable rate warping.
    jitter_sigma : noise std for channel_jitter.  0.0 disables jitter.

    Example
    -------
        aug = make_augment(rate_warp_lo=0.8, rate_warp_hi=1.25, jitter_sigma=0.01)
        results = run_grouped_cv(lambda: VGGRUReg(n_features=4), ...,
                                  train_kwargs={"augment_fn": aug})
    """
    do_warp   = not (rate_warp_lo == 1.0 and rate_warp_hi == 1.0)
    do_jitter = jitter_sigma > 0.0

    def _augment(xb: torch.Tensor) -> torch.Tensor:
        if do_warp:
            xb = rate_warp(xb, lo=rate_warp_lo, hi=rate_warp_hi)
        if do_jitter:
            xb = channel_jitter(xb, sigma=jitter_sigma)
        return xb

    return _augment
