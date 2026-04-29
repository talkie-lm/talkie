"""Quality benchmark: bf16 vs int8 on the same passage.

Loads the model in bf16, runs one forward pass over the passage to compute
per-token log-probs and top-k indices, frees the GPU, and repeats with int8.
The two results are compared on perplexity, top-1 / top-5 next-token
agreement, and mean KL divergence of the softmax distributions.

Requires the int8 quantization path (the ``quantize="int8"`` argument on
``Talkie``).
"""

from __future__ import annotations

import gc
import sys
from dataclasses import dataclass

import torch

from talkie.generate import PerplexityResult, Talkie


@dataclass
class BenchmarkReport:
    model_name: str
    n_tokens: int
    perplexity_bf16: float
    perplexity_int8: float
    top1_agreement: float
    top5_agreement: float
    mean_kl: float

    def __str__(self) -> str:
        ppl_gap_pct = (
            100.0
            * (self.perplexity_int8 - self.perplexity_bf16)
            / self.perplexity_bf16
        )
        lines = [
            f"Model:           {self.model_name}",
            f"Eval tokens:     {self.n_tokens}",
            "",
            f"Perplexity bf16: {self.perplexity_bf16:.4f}",
            f"Perplexity int8: {self.perplexity_int8:.4f}",
            f"  -> int8 is {ppl_gap_pct:+.2f}% relative to bf16 "
            f"({'worse' if ppl_gap_pct > 0 else 'better'})",
            "",
            f"Top-1 agreement: {self.top1_agreement * 100:.2f}%  "
            "(bf16 and int8 pick the same most-likely token)",
            f"Top-5 agreement: {self.top5_agreement * 100:.2f}%  "
            "(int8's top pick is in bf16's top-5)",
            f"Mean KL(bf16 || int8): {self.mean_kl:.4f} nats",
        ]
        return "\n".join(lines)


def _release(model: Talkie) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _agreement_stats(
    bf16: PerplexityResult, int8: PerplexityResult
) -> tuple[float, float]:
    top1_agree = (bf16.top1 == int8.top1).float().mean().item()
    top5_agree = (
        (bf16.top5 == int8.top1.unsqueeze(1)).any(dim=1).float().mean().item()
    )
    return top1_agree, top5_agree


def _mean_kl(bf16: PerplexityResult, int8: PerplexityResult) -> float:
    p_bf16 = bf16.log_softmax.exp()
    kl = (p_bf16 * (bf16.log_softmax - int8.log_softmax)).sum(dim=-1)
    return kl.mean().item()


def benchmark(
    model_name: str,
    text: str,
    max_tokens: int = 1024,
    device: str | None = None,
    cache_dir: str | None = None,
) -> BenchmarkReport:
    """Run the bf16 vs int8 comparison and return a populated report."""
    print("Loading bf16 model...", file=sys.stderr, flush=True)
    bf16_model = Talkie(
        model_name, device=device, cache_dir=cache_dir, quantize=None
    )
    print("Computing bf16 perplexity...", file=sys.stderr, flush=True)
    bf16 = bf16_model.perplexity(text, max_tokens=max_tokens)
    print(f"  bf16 perplexity: {bf16.perplexity:.4f}", file=sys.stderr, flush=True)
    _release(bf16_model)

    print("Loading int8 model...", file=sys.stderr, flush=True)
    int8_model = Talkie(
        model_name, device=device, cache_dir=cache_dir, quantize="int8"
    )
    print("Computing int8 perplexity...", file=sys.stderr, flush=True)
    int8 = int8_model.perplexity(text, max_tokens=max_tokens)
    print(f"  int8 perplexity: {int8.perplexity:.4f}", file=sys.stderr, flush=True)
    _release(int8_model)

    top1, top5 = _agreement_stats(bf16, int8)
    kl = _mean_kl(bf16, int8)

    return BenchmarkReport(
        model_name=model_name,
        n_tokens=bf16.n_tokens,
        perplexity_bf16=bf16.perplexity,
        perplexity_int8=int8.perplexity,
        top1_agreement=top1,
        top5_agreement=top5,
        mean_kl=kl,
    )
