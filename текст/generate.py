"""Sample text from a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from transformator.model import GPT, GPTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a checkpoint")
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/model.pt"))
    parser.add_argument("--prompt", type=str, default="The ")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = GPTConfig(**blob["config"])
    model = GPT(config).to(device)
    model.load_state_dict(blob["model"])
    model.eval()

    stoi = blob["tokenizer"]["stoi"]
    itos = {i: ch for ch, i in stoi.items()}
    prompt = args.prompt
    missing = [ch for ch in prompt if ch not in stoi]
    if missing:
        raise SystemExit(
            "prompt has characters the model never saw: "
            + repr("".join(sorted(set(missing))))
        )
    idx = torch.tensor([[stoi[ch] for ch in prompt]], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    text = "".join(itos[int(i)] for i in out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
