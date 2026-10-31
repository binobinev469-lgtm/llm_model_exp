"""Train the tiny transformer on a text file."""

from __future__ import annotations

import argparse
import math
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
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--eval-batches", type=int, default=8)
    return parser.parse_args()


def learning_rate(step: int, total: int, base: float, min_lr: float, warmup: int) -> float:
    """Linear warmup, then cosine decay from base down to min_lr."""
    if warmup > 0 and step <= warmup:
        return base * step / warmup
    if total <= warmup:
        return base
    progress = (step - warmup) / (total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base - min_lr) * cosine


def adamw_groups(model: GPT, weight_decay: float) -> list[dict]:
    """Decay matrices. Leave layer-norm scales and biases alone."""
    decay: list[torch.nn.Parameter] = []
    quiet: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for param in model.parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))
        if param.ndim >= 2:
            decay.append(param)
        else:
            quiet.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": quiet, "weight_decay": 0.0},
    ]


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


@torch.no_grad()
def holdout_loss(
    model: GPT,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str,
) -> float:
    """Mean loss over the holdout. Stride through long splits so the score stays stable."""
    was_training = model.training
    model.eval()
    n_windows = len(data) - block_size
    stride = 1 if n_windows <= 512 else block_size
    starts = range(0, n_windows, stride)
    total = 0.0
    tokens = 0
    batch: list[int] = []
    for start in starts:
        batch.append(start)
        if len(batch) < batch_size:
            continue
        total, tokens = _add_window_loss(
            model, data, batch, block_size, device, total, tokens
        )
        batch = []
    if batch:
        total, tokens = _add_window_loss(
            model, data, batch, block_size, device, total, tokens
        )
    if was_training:
        model.train()
    return total / tokens


def _add_window_loss(
    model: GPT,
    data: torch.Tensor,
    starts: list[int],
    block_size: int,
    device: str,
    total: float,
    tokens: int,
) -> tuple[float, int]:
    x = torch.stack([data[s : s + block_size] for s in starts]).to(device)
    y = torch.stack([data[s + 1 : s + 1 + block_size] for s in starts]).to(device)
    _, loss = model(x, y)
    count = y.numel()
    return total + float(loss.item()) * count, tokens + count


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
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
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
    optimizer = torch.optim.AdamW(adamw_groups(model, args.weight_decay), lr=args.lr)
    print(
        f"device={device} vocab={dataset.vocab_size} "
        f"params={model.num_parameters():,} "
        f"train={len(train_data):,} val={len(val_data):,} seed={args.seed}"
    )

    best_val = float("inf")
    model.train()
    for step in range(1, args.steps + 1):
        lr = learning_rate(step, args.steps, args.lr, args.min_lr, args.warmup)
        for group in optimizer.param_groups:
            group["lr"] = lr
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
            val_loss = holdout_loss(
                model, val_data, args.block_size, args.batch_size, device
            )
            print(
                f"step {step:4d}  lr {lr:.2e}  train {train_loss:.4f}  val {val_loss:.4f}"
            )
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(args.out, config, model, dataset)
                print(f"saved {args.out}  val {val_loss:.4f}")

    print(f"best val {best_val:.4f}")


if __name__ == "__main__":
    main()
