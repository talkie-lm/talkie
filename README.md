# talkie - a 13B vintage language model from 1930

<p align="center">
  <img src="assets/identity.png" alt="talkie" width="600" />
</p>

`talkie` is an inference library for the talkie 13B language model family developed by Alec Radford, Nick Levine, and David Duvenaud.

`talkie-1930-13b-base` is a 13b language model trained on pre-1931 English-language text.

`talkie-1930-13b-it` has been instruction-tuned using a novel instruction-following dataset built from pre-1931 reference works including etiquette manuals, letter-writing manuals, encyclopedias, and poetry collections. It has also undergone reinforcement learning using online DPO to improve instruction-following capabilities. 

We also provide a 'modern' base model, `talkie-web-13b-base`, with the same architecture and training FLOPs as `talkie-1930`, but trained on FineWeb, to allow for controlled comparisons between modern and vintage models. Note that we need to be careful about the claims we make contrasting the behavior and capabilities of the models, because temporal coverage is not the only difference in the pretraining corpora. For example, the distribution of subject matters differs significantly. 

See our [blog post](https://talkie-lm.com/) for details.

This package provides a simple Python API and CLI to download models from HuggingFace and run inference.

## Models

| Name | HuggingFace | Style | Description |
|------|-------------|-------|-------------|
| `talkie-1930-13b-base` | [talkie-lm/talkie-1930-13b-base](https://huggingface.co/talkie-lm/talkie-1930-13b-base) | Base | 1930-era base language model |
| `talkie-1930-13b-it` | [talkie-lm/talkie-1930-13b-it](https://huggingface.co/talkie-lm/talkie-1930-13b-it) | IT | 1930-era instruction-tuned model |
| `talkie-web-13b-base` | [talkie-lm/talkie-web-13b-base](https://huggingface.co/talkie-lm/talkie-web-13b-base) | Base | Same architecture as talkie-1930, but trained on FineWeb |

## Installation

```bash
git clone https://github.com/talkie-lm/talkie.git
cd talkie
uv sync
```

### Requirements

- Python >= 3.11
- PyTorch >= 2.1
- CUDA GPU with >= 28 GB VRAM (bfloat16 inference)
- ~26-50 GB disk space per model

### Apple Silicon / MLX

Talkie also includes an optional MLX backend for Apple Silicon Macs. Install the optional extra:

```bash
uv sync --extra mlx
```

Download or convert an MLX-format Talkie directory, then run:

```bash
uv run talkie-mlx --model-dir /path/to/talkie-1930-13b-it-mlx \
  --max-tokens 80 \
  "Write a short note about radio."
```

To convert a PyTorch checkpoint yourself:

```bash
uv run python scripts/convert_to_mlx.py \
  --checkpoint /path/to/rl-refined.pt \
  --vocab /path/to/vocab.txt \
  --out-dir /path/to/talkie-1930-13b-it-mlx \
  --source-repo talkie-lm/talkie-1930-13b-it
```

## Quick Start

### Python API

```python
from talkie import Talkie

# Load a base model (downloads from HuggingFace on first use)
model = Talkie("talkie-1930-13b-base")

# Generate a completion
result = model.generate("If scientists discover life on other planets,", temperature=0.7, max_tokens=300)
print(result.text)

# Stream tokens
for token in model.stream("The effects of the automobile on public morality have"):
    print(token, end="", flush=True)
```

### Chat (instruction-tuned model)

```python
from talkie import Talkie, Message

model = Talkie("talkie-1930-13b-it")

# Single-turn
result = model.generate("Write an essay predicting what life will be like in the year 1960.", max_tokens=600)
print(result.text)

# Multi-turn chat
messages = [
    Message(role="user", content="What were the causes of the French Revolution?"),
]
result = model.chat(messages, temperature=0.7)
print(result.text)

# Stream a chat reply
messages.append(Message(role="assistant", content=result.text))
messages.append(Message(role="user", content="Which of those causes was the most significant?"))
for token in model.chat_stream(messages):
    print(token, end="", flush=True)
```

### Pre-download models

```python
from talkie import download_model

# Download before loading (useful for setup scripts)
download_model("talkie-1930-13b-base")
```

## CLI

```bash
# Generate text
uv run talkie generate "Once upon a time" --model talkie-1930-13b-base -t 0.8

# Interactive chat
uv run talkie chat --model talkie-1930-13b-it

# Download a model
uv run talkie download talkie-1930-13b-base

# Download all models
uv run talkie download all

# List available models
uv run talkie list
```

## License

Apache 2.0
