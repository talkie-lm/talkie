"""Generation utilities for the MLX Talkie backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import numpy as np

from talkie.chat import Message, format_chat, format_prompt, truncate_at_stop
from talkie.mlx.model import TalkieModel
from talkie.tokenizer import build_tokenizer


@dataclass
class MLXGenerationConfig:
    max_tokens: int = 200
    temperature: float = 0.7
    top_p: float | None = 0.95
    top_k: int | None = None
    seed: int | None = None


def _sample_numpy(
    logits: mx.array,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    suppress: set[int] | None = None,
) -> int:
    scores = np.array(logits[0], dtype=np.float32)
    if suppress:
        for token_id in suppress:
            if 0 <= token_id < scores.shape[0]:
                scores[token_id] = -np.inf

    if temperature <= 0:
        return int(np.argmax(scores))

    scores = scores / temperature

    if top_k is not None and 0 < top_k < scores.shape[0]:
        keep = np.argpartition(scores, -top_k)[-top_k:]
        masked = np.full_like(scores, -np.inf)
        masked[keep] = scores[keep]
        scores = masked

    finite = np.isfinite(scores)
    if not np.any(finite):
        raise ValueError("all logits were masked")
    scores = scores - np.max(scores[finite])
    probs = np.exp(scores)
    probs[~finite] = 0.0
    probs = probs / probs.sum()

    if top_p is not None and 0.0 < top_p < 1.0:
        order = np.argsort(probs)[::-1]
        sorted_probs = probs[order]
        cdf = np.cumsum(sorted_probs)
        keep_count = int(np.searchsorted(cdf, top_p, side="left") + 1)
        keep = order[:keep_count]
        masked = np.zeros_like(probs)
        masked[keep] = probs[keep]
        probs = masked / masked.sum()

    return int(np.random.choice(scores.shape[0], p=probs))


class MLXTalkie:
    """High-level MLX inference wrapper for a converted Talkie model directory."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.model = TalkieModel.from_pretrained(self.model_dir)
        self.tokenizer = build_tokenizer(
            self.model_dir / "vocab.txt", style=self.model.config.style
        )
        self.stop_token_ids = {
            self.tokenizer.encode_single_token("<|endoftext|>"),
        }
        if self.model.config.style == "it":
            self.stop_token_ids.update(
                {
                    self.tokenizer.encode_single_token("<|end|>"),
                    self.tokenizer.encode_single_token("<|user|>"),
                    self.tokenizer.encode_single_token("<|system|>"),
                }
            )

    def _generate_ids(
        self, prompt: str, config: MLXGenerationConfig
    ) -> Iterable[int]:
        if config.seed is not None:
            np.random.seed(config.seed)
            mx.random.seed(config.seed)

        prompt_ids = self.tokenizer.encode(prompt, allowed_special="all")
        if not prompt_ids:
            raise ValueError("prompt produced no tokens")

        input_ids = mx.array([prompt_ids], dtype=mx.int32)
        logits, cache = self.model(input_ids)
        mx.eval(logits, cache)

        for _ in range(config.max_tokens):
            token_id = _sample_numpy(
                logits,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
            )
            if token_id in self.stop_token_ids:
                break
            yield token_id
            logits, cache = self.model(mx.array([[token_id]], dtype=mx.int32), cache)
            mx.eval(logits, cache)

    def generate(self, prompt: str, config: MLXGenerationConfig | None = None) -> str:
        config = config or MLXGenerationConfig()
        text = self.tokenizer.decode(list(self._generate_ids(prompt, config)))
        text, _ = truncate_at_stop(text)
        return text

    def stream(self, prompt: str, config: MLXGenerationConfig | None = None):
        config = config or MLXGenerationConfig()
        emitted = ""
        for token_id in self._generate_ids(prompt, config):
            piece = self.tokenizer.decode([token_id])
            emitted += piece
            truncated, stopped = truncate_at_stop(emitted)
            if stopped:
                delta = truncated
                if delta:
                    yield delta
                break
            yield piece

    def chat(self, messages: list[Message], config: MLXGenerationConfig | None = None) -> str:
        return self.generate(format_chat(messages), config)

    def prompt(self, prompt: str, config: MLXGenerationConfig | None = None) -> str:
        if self.model.config.style == "it":
            prompt = format_prompt(prompt)
        return self.generate(prompt, config)
