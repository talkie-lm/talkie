"""Command-line interface for Talkie inference."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="talkie",
        description="Inference CLI for Talkie 13B language models.",
    )
    sub = parser.add_subparsers(dest="command")

    # -- generate ----------------------------------------------------------
    gen = sub.add_parser("generate", help="Generate text from a prompt.")
    gen.add_argument("prompt", help="The text prompt.")
    gen.add_argument(
        "-m", "--model", default="talkie-1930-13b-base", help="Model name."
    )
    gen.add_argument(
        "-t", "--temperature", type=float, default=0.7, help="Sampling temperature."
    )
    gen.add_argument(
        "-n", "--max-tokens", type=int, default=256, help="Max tokens to generate."
    )
    gen.add_argument("--top-p", type=float, default=None, help="Nucleus sampling p.")
    gen.add_argument("--top-k", type=int, default=None, help="Top-k filtering.")
    gen.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducible generation."
    )
    gen.add_argument("--device", default=None, help="Device (cuda / cpu).")
    gen.add_argument("--cache-dir", default=None, help="HuggingFace cache directory.")
    gen.add_argument(
        "--no-stream", action="store_true", help="Print all at once instead of streaming."
    )

    # -- chat --------------------------------------------------------------
    ch = sub.add_parser("chat", help="Interactive multi-turn chat (IT model).")
    ch.add_argument(
        "-m", "--model", default="talkie-1930-13b-it", help="Model name."
    )
    ch.add_argument(
        "-t", "--temperature", type=float, default=0.7, help="Sampling temperature."
    )
    ch.add_argument(
        "-n", "--max-tokens", type=int, default=256, help="Max tokens per reply."
    )
    ch.add_argument("--top-p", type=float, default=None, help="Nucleus sampling p.")
    ch.add_argument("--top-k", type=int, default=None, help="Top-k filtering.")
    ch.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducible generation."
    )
    ch.add_argument("--device", default=None, help="Device (cuda / cpu).")
    ch.add_argument("--cache-dir", default=None, help="HuggingFace cache directory.")
    ch.add_argument("--system", default=None, help="System prompt.")

    # -- download ----------------------------------------------------------
    dl = sub.add_parser("download", help="Download a model from HuggingFace.")
    dl.add_argument(
        "model",
        help='Model name (or "all" to download every model).',
    )
    dl.add_argument("--cache-dir", default=None, help="HuggingFace cache directory.")

    # -- list --------------------------------------------------------------
    sub.add_parser("list", help="List available models.")

    args = parser.parse_args(argv)

    if args.command == "generate":
        _cmd_generate(args)
    elif args.command == "chat":
        _cmd_chat(args)
    elif args.command == "download":
        _cmd_download(args)
    elif args.command == "list":
        _cmd_list()
    else:
        parser.print_help()


# --------------------------------------------------------------------------
# Command implementations
# --------------------------------------------------------------------------


def _cmd_generate(args: argparse.Namespace) -> None:
    from talkie.generate import Talkie

    print(f"Loading {args.model}...", file=sys.stderr)
    model = Talkie(args.model, device=args.device, cache_dir=args.cache_dir)

    if args.no_stream:
        result = model.generate(
            args.prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
        )
        print(result.text)
    else:
        for token in model.stream(
            args.prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
        ):
            print(token, end="", flush=True)
        print()


def _cmd_chat(args: argparse.Namespace) -> None:
    from talkie.chat import Message
    from talkie.generate import Talkie

    print(f"Loading {args.model}...", file=sys.stderr)
    model = Talkie(args.model, device=args.device, cache_dir=args.cache_dir)
    print("Model loaded. Type your message (Ctrl-D to quit).\n", file=sys.stderr)

    messages: list[Message] = []
    if args.system:
        messages.append(Message(role="system", content=args.system))

    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input.strip():
            continue

        messages.append(Message(role="user", content=user_input))
        reply_parts: list[str] = []
        for token in model.chat_stream(
            messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
        ):
            print(token, end="", flush=True)
            reply_parts.append(token)
        print("\n")
        messages.append(Message(role="assistant", content="".join(reply_parts)))


def _cmd_download(args: argparse.Namespace) -> None:
    from talkie.config import MODELS
    from talkie.download import download_model

    names = sorted(MODELS) if args.model == "all" else [args.model]
    for name in names:
        print(f"Downloading {name}...")
        path = download_model(name, cache_dir=args.cache_dir)
        print(f"  -> {path}")
    print("Done.")


def _cmd_list() -> None:
    from talkie.config import MODELS

    print("Available talkie models:\n")
    print(f"  {'Name':<28} {'HuggingFace Repo':<38} {'Style'}")
    print(f"  {'─' * 28} {'─' * 38} {'─' * 5}")
    for name, spec in sorted(MODELS.items()):
        print(f"  {name:<28} {spec.repo_id:<38} {spec.style}")
