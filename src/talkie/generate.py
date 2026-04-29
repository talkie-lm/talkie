"""High-level inference interface for Talkie models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Literal

import torch

from talkie.chat import (
    STOP_WINDOW,
    Message,
    format_chat,
    format_prompt,
    truncate_at_stop,
)
from talkie.config import MODELS
from talkie.download import get_model_files
from talkie.model import load_checkpoint
from talkie.sampling import (
    list_top_k_tensor,
    list_top_p_tensor,
    scalar_top_k_tensor,
    scalar_top_p_tensor,
)
from talkie.tokenizer import IT_VOCAB_SIZE, build_tokenizer


@dataclass
class GenerationConfig:
    """Sampling parameters for text generation."""

    temperature: float = 0.7
    max_tokens: int = 256
    top_p: float | None = None
    top_k: int | None = None


@dataclass
class GenerationResult:
    """Container for a generated completion."""

    text: str
    token_count: int
    finish_reason: Literal["stop", "length"]


class Talkie:
    """Main inference interface for Talkie 13B models.

    Downloads the model from HuggingFace on first use if not already cached.

    Parameters
    ----------
    model_name:
        One of ``"talkie-1930-13b-base"``, ``"talkie-1930-13b-it"``,
        or ``"talkie-web-13b-base"``.
    device:
        PyTorch device string.  Defaults to ``"cuda"`` if available.
    cache_dir:
        Custom HuggingFace cache directory.
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        cache_dir: str | None = None,
    ):
        if model_name not in MODELS:
            available = ", ".join(sorted(MODELS))
            raise ValueError(
                f"Unknown model {model_name!r}. Available: {available}"
            )

        self.model_name = model_name
        self.spec = MODELS[model_name]
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Download / resolve files.
        ckpt_path, vocab_path = get_model_files(model_name, cache_dir=cache_dir)

        # Build tokenizer.
        self.tokenizer = build_tokenizer(vocab_path, style=self.spec.style)
        target_vocab = IT_VOCAB_SIZE if self.spec.style == "it" else None

        # Load model.
        self.model = load_checkpoint(
            str(ckpt_path), self.device, target_vocab_size=target_vocab
        )

        # Stop tokens.
        self._stop_ids: set[int] = {
            self.tokenizer.encode_single_token("<|endoftext|>")
        }
        if self.spec.style == "it":
            self._stop_ids.add(self.tokenizer.encode_single_token("<|end|>"))

        self._autocast = (
            torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else torch.no_grad()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 256,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> GenerationResult:
        """Generate a completion and return the full result.

        For base models the *prompt* is passed through raw.  For the IT model
        the prompt is automatically wrapped in the chat template.
        """
        tokens: list[str] = []
        for tok in self.stream(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
        ):
            tokens.append(tok)
        text = "".join(tokens)
        # Heuristic: if stream ended before max_tokens the model hit a stop.
        finish = "stop" if len(tokens) < max_tokens else "length"
        return GenerationResult(text=text, token_count=len(tokens), finish_reason=finish)

    def stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 256,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> Generator[str, None, None]:
        """Stream tokens one at a time.

        For base models the *prompt* is passed through raw.  For the IT model
        the prompt is automatically wrapped in the chat template.
        """
        if self.spec.style == "it":
            formatted = format_prompt(prompt)
        else:
            formatted = prompt
        yield from self._stream_raw(
            formatted, temperature, max_tokens, top_p, top_k
        )

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 256,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> GenerationResult:
        """Multi-turn chat completion (IT model only).

        Raises :class:`ValueError` if called on a base model.
        """
        self._require_it("chat")
        tokens: list[str] = []
        for tok in self.chat_stream(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
        ):
            tokens.append(tok)
        text = "".join(tokens)
        finish = "stop" if len(tokens) < max_tokens else "length"
        return GenerationResult(text=text, token_count=len(tokens), finish_reason=finish)

    def chat_stream(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 256,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> Generator[str, None, None]:
        """Stream a multi-turn chat reply token-by-token (IT model only)."""
        self._require_it("chat_stream")
        formatted = format_chat(messages)
        yield from self._stream_raw(
            formatted, temperature, max_tokens, top_p, top_k
        )

    def batch_generate(
        self,
        prompt: str,
        configs: list[GenerationConfig],
    ) -> list[GenerationResult]:
        """Generate multiple completions from the same prompt in parallel."""
        B = len(configs)
        if self.spec.style == "it":
            formatted = format_prompt(prompt)
        else:
            formatted = prompt

        tokens = self.tokenizer.encode(formatted, allowed_special="all")
        prompt_len = len(tokens)
        prompt_tensor = (
            torch.tensor(tokens, dtype=torch.long, device=self.device)
            .unsqueeze(0)
            .expand(B, -1)
            .contiguous()
        )

        temps = torch.tensor(
            [[c.temperature] for c in configs],
            dtype=torch.float32,
            device=self.device,
        )
        top_p_t = list_top_p_tensor([c.top_p for c in configs], self.device)
        top_k_t = list_top_k_tensor(
            [c.top_k for c in configs],
            self.model.config.vocab_size,
            self.device,
        )
        max_per = [c.max_tokens for c in configs]
        global_max = max(max_per)

        finished = [False] * B
        finish_reasons = ["length"] * B
        gen_tokens: list[list[int]] = [[] for _ in range(B)]

        cache = self.model.make_kv_cache(
            batch=B, max_seq_len=prompt_len + global_max
        )

        with torch.no_grad(), self._autocast:
            # Prefill: feed the whole prompt through once, populating the cache.
            next_tokens = self.model.sample_batch_variable_temp(
                prompt_tensor,
                temps,
                top_p=top_p_t,
                top_k=top_k_t,
                kv_cache=cache,
                position=0,
            )
            position = prompt_len

            for step in range(global_max):
                tokens_cpu = next_tokens.tolist()
                for i in range(B):
                    if finished[i]:
                        continue
                    tok = tokens_cpu[i]
                    if tok in self._stop_ids:
                        finished[i] = True
                        finish_reasons[i] = "stop"
                        continue
                    gen_tokens[i].append(tok)
                    if len(gen_tokens[i]) >= max_per[i]:
                        finished[i] = True

                if all(finished) or step == global_max - 1:
                    break

                # Decode: feed only the just-sampled token back in.
                next_tokens = self.model.sample_batch_variable_temp(
                    next_tokens.unsqueeze(1),
                    temps,
                    top_p=top_p_t,
                    top_k=top_k_t,
                    kv_cache=cache,
                    position=position,
                )
                position += 1

        results: list[GenerationResult] = []
        for i in range(B):
            text = self.tokenizer.decode(gen_tokens[i])
            results.append(
                GenerationResult(
                    text=text,
                    token_count=len(gen_tokens[i]),
                    finish_reason=finish_reasons[i],
                )
            )
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_it(self, method: str) -> None:
        if self.spec.style != "it":
            raise ValueError(
                f"{method}() requires an instruction-tuned model.  "
                f"Current model {self.model_name!r} is style={self.spec.style!r}."
            )

    def _stream_raw(
        self,
        formatted_prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float | None,
        top_k: int | None,
    ) -> Generator[str, None, None]:
        """Stream tokens from an already-formatted prompt string."""
        tokens = self.tokenizer.encode(formatted_prompt, allowed_special="all")
        prompt_len = len(tokens)
        prompt_tensor = torch.tensor(
            [tokens], dtype=torch.long, device=self.device
        )

        top_p_t = scalar_top_p_tensor(top_p, self.device)
        top_k_t = scalar_top_k_tensor(top_k, self.device)

        is_it = self.spec.style == "it"
        buffered_text = ""

        cache = self.model.make_kv_cache(
            batch=1, max_seq_len=prompt_len + max_tokens
        )

        with torch.no_grad(), self._autocast:
            # Prefill: process the whole prompt in one forward pass.
            next_token = self.model.sample_batch(
                prompt_tensor,
                t=temperature,
                top_p=top_p_t,
                top_k=top_k_t,
                kv_cache=cache,
                position=0,
            )
            position = prompt_len

            for step in range(max_tokens):
                tok_int = int(next_token.item())

                if tok_int in self._stop_ids:
                    break

                decoded = self.tokenizer.decode([tok_int])

                if is_it:
                    buffered_text += decoded
                    truncated, should_stop = truncate_at_stop(buffered_text)
                    if should_stop:
                        if truncated:
                            yield truncated
                        return
                    flush_upto = max(
                        0, len(buffered_text) - (STOP_WINDOW - 1)
                    )
                    if flush_upto > 0:
                        yield buffered_text[:flush_upto]
                        buffered_text = buffered_text[flush_upto:]
                else:
                    yield decoded

                if step == max_tokens - 1:
                    break

                # Decode step: feed only the just-sampled token.
                next_token = self.model.sample_batch(
                    next_token.view(1, 1),
                    t=temperature,
                    top_p=top_p_t,
                    top_k=top_k_t,
                    kv_cache=cache,
                    position=position,
                )
                position += 1

        # Flush remaining buffer for IT models.
        if is_it and buffered_text:
            truncated, _ = truncate_at_stop(buffered_text)
            if truncated:
                yield truncated
