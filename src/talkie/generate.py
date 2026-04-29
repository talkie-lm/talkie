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


@dataclass
class PerplexityResult:
    """Per-token statistics from one perplexity evaluation pass.

    *log_softmax* holds the full ``[T-1, vocab_size]`` log-probability matrix
    on CPU so that two passes (e.g. bf16 vs int8) can be compared via KL
    divergence.  At fp32 / 65k vocab / 1024 tokens this is ~256 MB.
    """

    perplexity: float
    n_tokens: int
    nll: torch.Tensor
    top1: torch.Tensor
    top5: torch.Tensor
    log_softmax: torch.Tensor


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
    max_seq_len:
        Maximum sequence length the model is built for.  Controls the
        size of the precomputed rotary cos/sin buffers.  Defaults to
        2048; larger values let you feed longer prompts but quality
        past the model's training length depends on RoPE extrapolation.
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        cache_dir: str | None = None,
        max_seq_len: int = 2048,
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
            str(ckpt_path),
            self.device,
            target_vocab_size=target_vocab,
            max_seq_len=max_seq_len,
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

    def perplexity(
        self,
        text: str | None = None,
        max_tokens: int = 1024,
        last_k: int | None = None,
        tokens: list[int] | None = None,
    ) -> PerplexityResult:
        """Evaluate perplexity on a passage of plain text.

        The text is tokenized as raw BPE (no chat template, no special
        tokens), truncated to ``min(max_tokens, model.max_seq_len)``, and
        run through one forward pass.

        Pass *last_k* to score only the final K tokens (the rest of the
        input is context only).  This is the standard setup for
        context-length sweeps: keep the eval window fixed, vary the
        amount of preceding context.

        Pass *tokens* directly to skip re-tokenization (handy when
        sweeping different slices of the same long text).
        """
        if tokens is None:
            if text is None:
                raise ValueError("Provide either text or tokens.")
            tokens = self.tokenizer.encode(text, disallowed_special=())
        n_max = min(max_tokens, self.model.max_seq_len)
        if len(tokens) > n_max:
            tokens = tokens[:n_max]
        if len(tokens) < 2:
            raise ValueError(
                "Need at least 2 tokens to compute perplexity; "
                f"got {len(tokens)}."
            )

        eval_k = (
            len(tokens) - 1 if last_k is None else min(last_k, len(tokens) - 1)
        )

        input_ids = torch.tensor(
            [tokens], dtype=torch.long, device=self.device
        )

        with torch.no_grad(), self._autocast:
            # Need eval_k+1 logits to predict eval_k tokens; the last logit
            # would predict beyond the input and is dropped.
            logits = self.model.forward(input_ids, last_k=eval_k + 1)

        eval_logits = logits[0, :-1]  # [eval_k, V]
        log_softmax = torch.log_softmax(eval_logits.float(), dim=-1)
        targets = torch.tensor(
            tokens[-eval_k:], dtype=torch.long, device=self.device
        )
        nll = -log_softmax.gather(1, targets.unsqueeze(1)).squeeze(1)
        perplexity = nll.mean().exp().item()

        top5 = log_softmax.topk(5, dim=-1).indices
        top1 = top5[:, 0]

        return PerplexityResult(
            perplexity=perplexity,
            n_tokens=eval_k,
            nll=nll.cpu(),
            top1=top1.cpu(),
            top5=top5.cpu(),
            log_softmax=log_softmax.cpu(),
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
        tokens_tensor = (
            torch.tensor(tokens, dtype=torch.long, device=self.device)
            .unsqueeze(0)
            .expand(B, -1)
            .clone()
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
        gen_counts = [0] * B

        with torch.no_grad(), self._autocast:
            for _ in range(global_max):
                next_tokens = self.model.sample_batch_variable_temp(
                    tokens_tensor, temps, top_p=top_p_t, top_k=top_k_t
                )
                next_tokens = next_tokens.unsqueeze(1)
                tokens_tensor = torch.cat([tokens_tensor, next_tokens], dim=1)

                for i in range(B):
                    if finished[i]:
                        continue
                    gen_counts[i] += 1
                    tok = int(next_tokens[i, 0])
                    if tok in self._stop_ids:
                        finished[i] = True
                        finish_reasons[i] = "stop"
                    elif gen_counts[i] >= max_per[i]:
                        finished[i] = True

                if all(finished):
                    break

        results: list[GenerationResult] = []
        for i in range(B):
            gen_tokens = tokens_tensor[
                i, prompt_len : prompt_len + gen_counts[i]
            ].tolist()
            if gen_tokens and gen_tokens[-1] in self._stop_ids:
                gen_tokens = gen_tokens[:-1]
            text = self.tokenizer.decode(gen_tokens)
            results.append(
                GenerationResult(
                    text=text,
                    token_count=len(gen_tokens),
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
        tokens_tensor = (
            torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        )

        top_p_t = scalar_top_p_tensor(top_p, self.device)
        top_k_t = scalar_top_k_tensor(top_k, self.device)

        is_it = self.spec.style == "it"
        buffered_text = ""

        with torch.no_grad(), self._autocast:
            for _ in range(max_tokens):
                next_token = self.model.sample_batch(
                    tokens_tensor, t=temperature, top_p=top_p_t, top_k=top_k_t
                )[0]
                next_token_tensor = torch.tensor(
                    [[next_token]], device=self.device
                )
                tokens_tensor = torch.cat(
                    [tokens_tensor, next_token_tensor], dim=1
                )

                if int(next_token) in self._stop_ids:
                    break

                decoded = self.tokenizer.decode([int(next_token)])

                if is_it:
                    # Buffer output to catch chat-template leaks.
                    buffered_text += decoded
                    truncated, should_stop = truncate_at_stop(buffered_text)
                    if should_stop:
                        if truncated:
                            yield truncated
                        break
                    flush_upto = max(
                        0, len(buffered_text) - (STOP_WINDOW - 1)
                    )
                    if flush_upto > 0:
                        yield buffered_text[:flush_upto]
                        buffered_text = buffered_text[flush_upto:]
                else:
                    yield decoded

        # Flush remaining buffer for IT models.
        if is_it and buffered_text:
            truncated, _ = truncate_at_stop(buffered_text)
            if truncated:
                yield truncated
