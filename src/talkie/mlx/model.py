"""MLX implementation of the custom Talkie 13B architecture.

The original model is a decoder-only GPT with RoPE, Q/K RMS normalisation,
SwiGLU MLPs, per-head / activation / LM-head gains, and embedding skip
connections.  This module intentionally keeps the original checkpoint key names
so converted safetensors can be loaded without transposition or key rewriting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math

import mlx.core as mx


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 65540
    n_layer: int = 40
    n_head: int = 40
    n_embd: int = 5120
    head_dim: int = 128
    rope_base: float = 1_000_000.0
    max_seq_len: int = 2048
    dtype: str = "bfloat16"
    style: str = "it"

    @classmethod
    def from_json(cls, path: str | Path) -> "GPTConfig":
        data = json.loads(Path(path).read_text())
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "head_dim": self.head_dim,
            "rope_base": self.rope_base,
            "max_seq_len": self.max_seq_len,
            "dtype": self.dtype,
            "style": self.style,
        }


def _dtype(name: str):
    if name == "bfloat16":
        return mx.bfloat16
    if name == "float16":
        return mx.float16
    if name == "float32":
        return mx.float32
    raise ValueError(f"unsupported dtype {name!r}")


def _rms_norm(x: mx.array) -> mx.array:
    # Match torch.nn.functional.rms_norm for bf16 inputs: PyTorch computes the
    # reduction in fp32 and uses fp32 epsilon when eps=None.
    eps = mx.finfo(mx.float32).eps
    xf = x.astype(mx.float32)
    y = xf * mx.rsqrt(mx.mean(mx.square(xf), axis=-1, keepdims=True) + eps)
    return y.astype(x.dtype)


def _linear(x: mx.array, weight: mx.array) -> mx.array:
    # Checkpoint weights are PyTorch Linear weights with shape [out_features, in_features].
    return mx.matmul(x, weight.T)


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


class TalkieModel:
    def __init__(self, weights: dict[str, mx.array], config: GPTConfig):
        self.weights = weights
        self.config = config
        self.dtype = _dtype(config.dtype)
        self._cos, self._sin = self._precompute_rope(config.max_seq_len)

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "TalkieModel":
        model_dir = Path(model_dir)
        config = GPTConfig.from_json(model_dir / "config.json")
        weights: dict[str, mx.array] = {}
        files = sorted(model_dir.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no .safetensors files found in {model_dir}")
        for file in files:
            weights.update(mx.load(file))
        return cls(weights, config)

    def _precompute_rope(self, seq_len: int) -> tuple[mx.array, mx.array]:
        half = self.config.head_dim // 2
        channel_range = mx.arange(0, self.config.head_dim, 2, dtype=mx.float32)
        inv_freq = 1.0 / (self.config.rope_base ** (channel_range / self.config.head_dim))
        t = mx.arange(seq_len, dtype=mx.float32)
        freqs = t[:, None] * inv_freq[None, :]
        cos = mx.cos(freqs).astype(self.dtype)[None, :, None, :]
        sin = mx.sin(freqs).astype(self.dtype)[None, :, None, :]
        assert cos.shape[-1] == half
        return cos, sin

    def _apply_rope(self, x: mx.array, offset: int) -> mx.array:
        # x: [B, T, H, D].  Talkie rotates first and second halves, matching
        # source/talkie/src/talkie/model.py::apply_rotary_emb.
        seq_len = x.shape[1]
        if offset + seq_len > self._cos.shape[1]:
            raise ValueError(
                f"sequence length {offset + seq_len} exceeds max_seq_len "
                f"{self._cos.shape[1]}"
            )
        d = x.shape[-1] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        cos = self._cos[:, offset : offset + seq_len]
        sin = self._sin[:, offset : offset + seq_len]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return mx.concatenate([y1, y2], axis=-1).astype(x.dtype)

    def _attention(
        self,
        layer: int,
        x: mx.array,
        cache: tuple[mx.array, mx.array] | None,
        offset: int,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        c = self.config
        prefix = f"blocks.{layer}.attn"
        bsz, seq_len, _ = x.shape

        q = _linear(x, self.weights[f"{prefix}.attn_query.weight"])
        k = _linear(x, self.weights[f"{prefix}.attn_key.weight"])
        v = _linear(x, self.weights[f"{prefix}.attn_value.weight"])

        q = q.reshape(bsz, seq_len, c.n_head, c.head_dim)
        k = k.reshape(bsz, seq_len, c.n_head, c.head_dim)
        v = v.reshape(bsz, seq_len, c.n_head, c.head_dim)

        q = _rms_norm(self._apply_rope(q, offset))
        k = _rms_norm(self._apply_rope(k, offset))
        q = q * self.weights[f"{prefix}.head_gain.head_g"].astype(q.dtype).reshape(1, 1, c.n_head, 1)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
            mask = None if seq_len == 1 else "causal"
        else:
            mask = "causal"

        y = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1.0 / math.sqrt(c.head_dim), mask=mask
        )
        y = y.transpose(0, 2, 1, 3).reshape(bsz, seq_len, c.n_embd)
        y = _linear(y, self.weights[f"{prefix}.attn_resid.weight"])
        return y, (k, v)

    def _mlp(self, layer: int, x: mx.array) -> mx.array:
        prefix = f"blocks.{layer}.mlp"
        gate = _linear(x, self.weights[f"{prefix}.mlp_gate.weight"])
        linear = _linear(x, self.weights[f"{prefix}.mlp_linear.weight"])
        return _linear(_silu(gate) * linear, self.weights[f"{prefix}.mlp_resid.weight"])

    def __call__(
        self,
        input_ids: mx.array,
        cache: list[tuple[mx.array, mx.array]] | None = None,
    ) -> tuple[mx.array, list[tuple[mx.array, mx.array]]]:
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        offset = 0 if cache is None else cache[0][0].shape[2]
        x = self.weights["embed.weight"][input_ids]
        x = _rms_norm(x)
        e_x = x

        new_cache: list[tuple[mx.array, mx.array]] = []
        for layer in range(self.config.n_layer):
            layer_cache = None if cache is None else cache[layer]
            attn, kv = self._attention(layer, _rms_norm(x), layer_cache, offset)
            x = x + attn * self.weights[f"blocks.{layer}.attn_gain.a_g"].astype(x.dtype)
            x = x + self._mlp(layer, _rms_norm(x)) * self.weights[f"blocks.{layer}.mlp_gain.a_g"].astype(x.dtype)
            x = x + e_x * self.weights[f"blocks.{layer}.embed_skip.a_g"].astype(x.dtype)
            new_cache.append(kv)

        x = _rms_norm(x)
        lm_head = self.weights["lm_head"] * self.weights["lm_head_gain.w_g"].astype(self.weights["lm_head"].dtype)
        logits = _linear(x[:, -1, :], lm_head).astype(mx.float32)
        return logits, new_cache
