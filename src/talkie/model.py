"""Talkie 13B transformer architecture.

A 40-layer, 40-head decoder-only GPT with RoPE, SwiGLU, RMS normalisation,
embedding skip connections, and per-head / per-layer gain parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from talkie.sampling import apply_top_k_top_p, sample_gumbel


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class GPTConfig:
    vocab_size: int = 65536
    n_layer: int = 40
    n_head: int = 40
    n_embd: int = 5120
    head_dim: int = 128


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


class HeadGain(nn.Module):
    def __init__(self, n_head: int):
        super().__init__()
        self.head_g = nn.Parameter(torch.ones([n_head]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.head_g.type_as(x).view(1, 1, -1, 1)


class WeightGain(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_g = nn.Parameter(torch.ones(1))

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return w * self.w_g.type_as(w)


class ActGain(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.a_g = nn.Parameter(torch.ones(1) * init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.a_g.type_as(x)


# ---------------------------------------------------------------------------
# Attention & MLP
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        n_state = config.n_embd

        self.attn_query = nn.Linear(n_state, n_state, bias=False)
        self.attn_key = nn.Linear(n_state, n_state, bias=False)
        self.attn_value = nn.Linear(n_state, n_state, bias=False)
        self.attn_resid = nn.Linear(n_state, n_state, bias=False)
        self.head_gain = HeadGain(config.n_head)

    def forward(self, x: torch.Tensor, cos_sin: tuple) -> torch.Tensor:
        bsz, seq_len, _ = x.size()
        q = self.attn_query(x).view(bsz, seq_len, self.n_head, self.head_dim)
        k = self.attn_key(x).view(bsz, seq_len, self.n_head, self.head_dim)
        v = self.attn_value(x).view(bsz, seq_len, self.n_head, self.head_dim)

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q = self.head_gain(q)

        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view_as(x)
        return self.attn_resid(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        n_state = config.n_embd
        n_mlp = int(round(((8 / 3) * n_state) / 128) * 128)

        self.mlp_gate = nn.Linear(n_state, n_mlp, bias=False)
        self.mlp_linear = nn.Linear(n_state, n_mlp, bias=False)
        self.mlp_resid = nn.Linear(n_mlp, n_state, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.mlp_gate(x)) * self.mlp_linear(x)
        return self.mlp_resid(x)


# ---------------------------------------------------------------------------
# Transformer block & full model
# ---------------------------------------------------------------------------


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.attn_gain = ActGain((2 * config.n_layer) ** -0.5)
        self.mlp = MLP(config)
        self.mlp_gain = ActGain((2 * config.n_layer) ** -0.5)
        self.embed_skip = ActGain(0.0)

    def forward(
        self, e_x: torch.Tensor, x: torch.Tensor, cos_sin: tuple
    ) -> torch.Tensor:
        x = x + self.attn_gain(self.attn(F.rms_norm(x, (x.shape[-1],)), cos_sin))
        x = x + self.mlp_gain(self.mlp(F.rms_norm(x, (x.shape[-1],))))
        x = x + self.embed_skip(e_x)
        return x


class TalkieModel(nn.Module):
    """Talkie 13B decoder-only transformer."""

    def __init__(
        self, config: GPTConfig, device: torch.device, max_seq_len: int = 2048
    ):
        super().__init__()
        self.config = config
        self.device = device

        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.lm_head = nn.Parameter(torch.zeros(config.vocab_size, config.n_embd))
        self.lm_head_gain = WeightGain()

        cos, sin = self._precompute_rotary_embeddings(max_seq_len, config.head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.suppress_token_ids: set[int] | None = None

    def _precompute_rotary_embeddings(
        self, seq_len: int, head_dim: int, base: int = 1_000_000
    ) -> tuple:
        device = self.embed.weight.device if hasattr(self, "embed") else "cpu"
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return ``[B, V]`` logits for the last position."""
        _, seq_len = input_ids.shape
        cos_sin = self.cos[:, :seq_len], self.sin[:, :seq_len]

        x = self.embed(input_ids)
        x = F.rms_norm(x, (x.shape[-1],))
        e_x = x
        for block in self.blocks:
            x = block(e_x, x, cos_sin)
        x = F.rms_norm(x, (x.shape[-1],))

        return F.linear(x[:, -1, :], self.lm_head_gain(self.lm_head)).float()

    def sample_batch(
        self,
        x: torch.Tensor,
        t: float = 0.7,
        top_p: torch.Tensor | None = None,
        top_k: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample one token per sequence in the batch."""
        logits = self.forward(x)
        if t != 1:
            logits = logits / t
        if top_p is not None or top_k is not None:
            logits = apply_top_k_top_p(logits, top_p=top_p, top_k=top_k)
        logits = logits + sample_gumbel(logits.shape, self.device)
        return torch.argmax(logits, dim=-1)

    def sample_batch_variable_temp(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        top_p: torch.Tensor | None = None,
        top_k: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Like :meth:`sample_batch` but *t* is a ``[B, 1]`` per-sequence temperature."""
        logits = self.forward(x)
        logits = logits / t
        if top_p is not None or top_k is not None:
            logits = apply_top_k_top_p(logits, top_p=top_p, top_k=top_k)
        logits = logits + sample_gumbel(logits.shape, self.device)
        return torch.argmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def resize_model_embeddings(
    model: TalkieModel, new_vocab_size: int, device: torch.device | str
) -> TalkieModel:
    """Grow embedding and lm_head to *new_vocab_size*, keeping old weights."""
    device = torch.device(device)
    old_vocab_size, n_embd = model.embed.weight.shape

    if old_vocab_size >= new_vocab_size:
        return model

    new_embed = nn.Embedding(new_vocab_size, n_embd, device=device)
    new_embed.weight.data[:old_vocab_size] = model.embed.weight.data
    new_embed.weight.data[old_vocab_size:] = (
        torch.randn(new_vocab_size - old_vocab_size, n_embd, device=device) * 0.02
    )
    model.embed = new_embed

    old_lm_head = model.lm_head.data
    new_lm_head = torch.zeros(new_vocab_size, n_embd, device=device)
    new_lm_head[:old_vocab_size] = old_lm_head
    new_lm_head[old_vocab_size:] = (
        torch.randn(new_vocab_size - old_vocab_size, n_embd, device=device) * 0.02
    )
    model.lm_head = nn.Parameter(new_lm_head)

    model.config.vocab_size = new_vocab_size
    return model


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    target_vocab_size: int | None = None,
) -> TalkieModel:
    """Load a Talkie model from a PyTorch checkpoint file.

    Handles ``torch.compile`` key prefixes and optional vocab resizing.
    Builds and converts to bfloat16 on CPU first, then moves to GPU to
    avoid a transient 2x memory spike from float32 initialisation.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    ckpt_vocab_size = state_dict["embed.weight"].shape[0]
    config = GPTConfig(vocab_size=ckpt_vocab_size)

    # Build on CPU, load weights, convert to bfloat16, THEN move to GPU.
    cpu = torch.device("cpu")
    model = TalkieModel(config, cpu)
    model = model.to(dtype=torch.bfloat16)
    model.load_state_dict(state_dict, strict=True)
    del ckpt, state_dict

    if target_vocab_size is not None and ckpt_vocab_size < target_vocab_size:
        model = resize_model_embeddings(model, target_vocab_size, cpu)

    model = model.to(device)
    model.device = device
    model.eval()
    return model
