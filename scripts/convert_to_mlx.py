#!/usr/bin/env python3
"""Convert a Talkie PyTorch checkpoint to MLX-loadable safetensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from safetensors.torch import save_file
import torch


def tensor_nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Talkie .pt/.ckpt checkpoint")
    parser.add_argument("--vocab", required=True, help="Talkie vocab.txt")
    parser.add_argument("--out-dir", required=True, help="output directory")
    parser.add_argument("--source-repo", default=None)
    parser.add_argument("--max-shard-gb", type=float, default=4.0)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    vocab = Path(args.vocab)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {checkpoint} with mmap=True")
    ckpt = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    state = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    config = dict(ckpt.get("config") or {})
    if not config:
        config = {
            "vocab_size": int(state["embed.weight"].shape[0]),
            "n_layer": 40,
            "n_head": 40,
            "n_embd": 5120,
            "head_dim": 128,
        }
    config.update(
        {
            "rope_base": 1_000_000.0,
            "max_seq_len": 2048,
            "dtype": "bfloat16",
            "style": "it" if int(config["vocab_size"]) > 65536 else "base",
            "architectures": ["TalkieModel"],
            "source_repo": args.source_repo,
            "source_checkpoint": checkpoint.name,
            "mlx_runtime": "talkie.mlx",
        }
    )

    for stale in out_dir.glob("*.safetensors"):
        stale.unlink()
    index_path = out_dir / "model.safetensors.index.json"
    if index_path.exists():
        index_path.unlink()

    max_shard_bytes = int(args.max_shard_gb * 1024**3)
    weight_map: dict[str, str] = {}
    total_size = 0
    shard: dict[str, torch.Tensor] = {}
    shard_bytes = 0
    shard_idx = 1

    def flush() -> None:
        nonlocal shard, shard_bytes, shard_idx
        if not shard:
            return
        filename = f"model-{shard_idx:05d}-of-PLACEHOLDER.safetensors"
        path = out_dir / filename
        print(f"Writing {path.name} ({shard_bytes / 1024**3:.2f} GiB, {len(shard)} tensors)")
        save_file(shard, path, metadata={"format": "mlx"})
        for key in shard:
            weight_map[key] = path.name
        shard = {}
        shard_bytes = 0
        shard_idx += 1

    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        tensor = tensor.detach().cpu().contiguous()
        nbytes = tensor_nbytes(tensor)
        if shard and shard_bytes + nbytes > max_shard_bytes:
            flush()
        shard[key] = tensor
        shard_bytes += nbytes
        total_size += nbytes
    flush()

    shard_files = sorted(out_dir.glob("model-*-of-PLACEHOLDER.safetensors"))
    total_shards = len(shard_files)
    rename_map: dict[str, str] = {}
    for file in shard_files:
        final_name = file.name.replace("PLACEHOLDER", f"{total_shards:05d}")
        file.rename(out_dir / final_name)
        rename_map[file.name] = final_name
    weight_map = {k: rename_map[v] for k, v in weight_map.items()}

    write_json(out_dir / "config.json", config)
    write_json(index_path, {"metadata": {"total_size": total_size}, "weight_map": weight_map})
    shutil.copy2(vocab, out_dir / "vocab.txt")
    print(f"Done. Wrote {total_shards} shards to {out_dir}")


if __name__ == "__main__":
    main()
