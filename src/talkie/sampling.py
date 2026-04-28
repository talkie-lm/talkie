"""Sampling utilities — Gumbel-max, top-k, and top-p (nucleus) filtering."""

from __future__ import annotations

import torch


def sample_gumbel(
    shape: tuple,
    device: torch.device,
    eps: float = 1e-20,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw Gumbel(0, 1) noise for the Gumbel-max trick."""
    u = torch.rand(shape, device=device, generator=generator)
    return -torch.log(-torch.log(u + eps) + eps)


def apply_top_k_top_p(
    logits: torch.Tensor,
    top_p: torch.Tensor | None = None,
    top_k: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply top-k and/or top-p (nucleus) filtering to ``[B, V]`` logits.

    Filtered-out entries are set to ``-inf`` so they cannot be sampled by the
    downstream Gumbel-argmax.  Both *top_p* and *top_k* accept a 0-dim tensor
    (broadcast to all rows) or a 1-D / ``[B, 1]`` tensor (per-row value).
    """
    V = logits.shape[-1]

    if top_k is not None:
        if top_k.ndim == 0:
            top_k = top_k.unsqueeze(0).expand(logits.shape[0])
        k = torch.clamp(top_k, min=1, max=V).to(
            device=logits.device, dtype=torch.long
        )
        sorted_logits, _ = torch.sort(logits, dim=-1, descending=True)
        threshold = sorted_logits.gather(-1, (k - 1).unsqueeze(-1))
        logits = torch.where(
            logits < threshold,
            torch.full_like(logits, float("-inf")),
            logits,
        )

    if top_p is not None:
        p = top_p.to(device=logits.device, dtype=logits.dtype)
        if p.ndim == 0:
            p = p.unsqueeze(0).expand(logits.shape[0])
        if p.ndim == 1:
            p = p.unsqueeze(-1)
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_to_remove = cumulative_probs > p
        sorted_to_remove[..., 1:] = sorted_to_remove[..., :-1].clone()
        sorted_to_remove[..., 0] = False
        to_remove = torch.zeros_like(sorted_to_remove)
        to_remove.scatter_(-1, sorted_indices, sorted_to_remove)
        logits = torch.where(
            to_remove,
            torch.full_like(logits, float("-inf")),
            logits,
        )

    return logits


# ---------------------------------------------------------------------------
# Helpers for converting Python scalars / lists to the tensor format expected
# by apply_top_k_top_p.
# ---------------------------------------------------------------------------


def scalar_top_p_tensor(
    top_p: float | None, device: torch.device
) -> torch.Tensor | None:
    if top_p is None or top_p >= 1.0:
        return None
    return torch.tensor(float(top_p), device=device, dtype=torch.float32)


def scalar_top_k_tensor(
    top_k: int | None, device: torch.device
) -> torch.Tensor | None:
    if top_k is None or top_k <= 0:
        return None
    return torch.tensor(int(top_k), device=device, dtype=torch.long)


def list_top_p_tensor(
    top_ps: list[float | None] | None, device: torch.device
) -> torch.Tensor | None:
    if top_ps is None:
        return None
    if not any(p is not None and p < 1.0 for p in top_ps):
        return None
    return torch.tensor(
        [[p if (p is not None and p < 1.0) else 1.0] for p in top_ps],
        dtype=torch.float32,
        device=device,
    )


def list_top_k_tensor(
    top_ks: list[int | None] | None, vocab_size: int, device: torch.device
) -> torch.Tensor | None:
    if top_ks is None:
        return None
    if not any(k is not None and k > 0 for k in top_ks):
        return None
    return torch.tensor(
        [k if (k is not None and k > 0) else vocab_size for k in top_ks],
        dtype=torch.long,
        device=device,
    )
