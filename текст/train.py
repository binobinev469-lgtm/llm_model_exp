"""Train the tiny transformer on a text file."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from transformator.data import CharDataset
from transformator.model import GPT, GPTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny character-level LLM")
    parser.add_argument("--data", type=Path, default=Path("data/sample.txt"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/model.pt"))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = CharDataset.from_file(args.data, args.block_size)
    config = GPTConfig(
        vocab_size=dataset.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(
        f"device={device} vocab={dataset.vocab_size} "
        f"params={model.num_parameters():,} chars={len(dataset.data):,}"
    )

    model.train()
    for step in range(1, args.steps + 1):
        x, y = dataset.get_batch(args.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.log_every == 0 or step == 1:
            print(f"step {step:4d}  loss {loss.item():.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config.__dict__,
            "model": model.state_dict(),
            "tokenizer": dataset.state_dict(),
        },
        args.out,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
