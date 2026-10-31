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
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--eval-batches", type=int, default=8)
    return parser.parse_args()


def train_val_split(
    data: torch.Tensor, block_size: int, val_fraction: float
) -> tuple[torch.Tensor, torch.Tensor]:
    min_len = block_size + 2
    if not 0 < val_fraction < 0.5:
        raise ValueError("val-fraction must be between 0 and 0.5")
    if len(data) < 2 * min_len:
        raise ValueError(
            f"need at least {2 * min_len} characters to hold out a validation split"
        )
    split = int(len(data) * (1 - val_fraction))
    split = min(max(split, min_len), len(data) - min_len)
    return data[:split], data[split:]


@torch.no_grad()
def estimate_loss(
    model: GPT,
    dataset: CharDataset,
    data: torch.Tensor,
    batch_size: int,
    device: str,
    batches: int,
) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    for _ in range(batches):
        x, y = dataset.get_batch(batch_size, device, data)
        _, loss = model(x, y)
        total += float(loss.item())
    if was_training:
        model.train()
    return total / batches


def save_checkpoint(
    path: Path, config: GPTConfig, model: GPT, dataset: CharDataset
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config.__dict__,
            "model": model.state_dict(),
            "tokenizer": dataset.state_dict(),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = CharDataset.from_file(args.data, args.block_size)
    train_data, val_data = train_val_split(
        dataset.data, args.block_size, args.val_fraction
    )
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
        f"params={model.num_parameters():,} "
        f"train={len(train_data):,} val={len(val_data):,}"
    )

    best_val = float("inf")
    model.train()
    for step in range(1, args.steps + 1):
        x, y = dataset.get_batch(args.batch_size, device, train_data)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.log_every == 0 or step == 1:
            train_loss = estimate_loss(
                model, dataset, train_data, args.batch_size, device, args.eval_batches
            )
            val_loss = estimate_loss(
                model, dataset, val_data, args.batch_size, device, args.eval_batches
            )
            print(f"step {step:4d}  train {train_loss:.4f}  val {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(args.out, config, model, dataset)
                print(f"saved {args.out}  val {val_loss:.4f}")

    print(f"best val {best_val:.4f}")


if __name__ == "__main__":
    main()
