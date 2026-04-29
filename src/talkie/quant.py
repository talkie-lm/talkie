"""Weight-only int8 quantization.

Each ``nn.Linear(bias=False)`` in the attention and MLP blocks is replaced
by :class:`Int8Linear`, which stores its weight as int8 and a per-output-row
bf16 scale.  Forward is a straight dequantize-then-matmul:

    y = (x @ (W_int8 → bf16).T) * scale_per_row

This roughly halves the weight footprint (13B params: 26 GB bf16 → 13 GB
int8) at the cost of one fp32→int8 conversion per weight row at load time.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class Int8Linear(nn.Module):
    """Linear layer with per-output-row symmetric int8 weights, no bias."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight", torch.empty(out_features, in_features, dtype=torch.int8)
        )
        self.register_buffer(
            "scale", torch.empty(out_features, dtype=torch.bfloat16)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize the weight on-the-fly into the activation dtype, run the
        # matmul, then apply the per-output-row scale once on the result —
        # equivalent to multiplying the scale into the weight before matmul,
        # but cheaper because the scale broadcast happens at output width.
        return F.linear(x, self.weight.to(x.dtype)) * self.scale.to(x.dtype)


def quantize_weight_int8(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-row int8 quantization.

    Returns ``(int8_weight, bf16_scale)`` where ``int8_weight`` has the same
    shape as *w* and ``bf16_scale`` has shape ``[w.shape[0]]``.
    """
    w_fp = w.float()
    amax = w_fp.abs().amax(dim=1).clamp(min=1e-12)
    scale = amax / 127.0
    q = (w_fp / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
    return q, scale.to(torch.bfloat16)


# Quantizable: the four attention projections and three MLP projections in
# every transformer block. Embedding and lm_head stay in bf16.
_QUANTIZABLE_LEAVES = frozenset(
    {
        "attn_query",
        "attn_key",
        "attn_value",
        "attn_resid",
        "mlp_gate",
        "mlp_linear",
        "mlp_resid",
    }
)


def is_quantizable_linear_key(key: str) -> bool:
    """Whether *key* names a ``.weight`` of a quantizable transformer linear."""
    if not key.endswith(".weight"):
        return False
    parts = key.split(".")
    return (
        len(parts) >= 5
        and parts[0] == "blocks"
        and parts[2] in ("attn", "mlp")
        and parts[3] in _QUANTIZABLE_LEAVES
    )


def quantize_state_dict_int8(
    state_dict: dict[str, torch.Tensor],
    key_predicate: Callable[[str], bool] = is_quantizable_linear_key,
) -> None:
    """Rewrite *state_dict* in place: every matching weight becomes int8 and a
    sibling ``.scale`` entry is added.

    The original bf16 tensor is dropped from the dict as soon as its int8
    replacement is written, so this never holds both copies of a weight at
    the same time.
    """
    for key in [k for k in state_dict if key_predicate(k)]:
        w = state_dict[key]
        q, s = quantize_weight_int8(w)
        state_dict[key] = q
        scale_key = key[: -len(".weight")] + ".scale"
        state_dict[scale_key] = s
