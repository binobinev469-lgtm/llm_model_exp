"""Character-level text dataset for next-token training."""

from __future__ import annotations

from pathlib import Path

import torch


class CharDataset:
    def __init__(self, text: str, block_size: int) -> None:
        if len(text) < block_size + 2:
            raise ValueError(
                f"text is {len(text)} chars; need at least {block_size + 2}"
            )
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.block_size = block_size
        self.data = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)

    @classmethod
    def from_file(cls, path: str | Path, block_size: int) -> CharDataset:
        text = Path(path).read_text(encoding="utf-8")
        return cls(text, block_size)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> torch.Tensor:
        unknown = [ch for ch in text if ch not in self.stoi]
        if unknown:
            missing = "".join(sorted(set(unknown)))
            raise ValueError(f"characters not in vocabulary: {missing!r}")
        return torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def get_batch(
        self, batch_size: int, device: torch.device | str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hi = len(self.data) - self.block_size - 1
        starts = torch.randint(0, hi, (batch_size,))
        x = torch.stack([self.data[i : i + self.block_size] for i in starts])
        y = torch.stack(
            [self.data[i + 1 : i + 1 + self.block_size] for i in starts]
        )
        return x.to(device), y.to(device)

    def state_dict(self) -> dict:
        return {
            "stoi": self.stoi,
            "block_size": self.block_size,
        }
