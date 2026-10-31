"""Checks for the schedule, causality, and greedy sampling."""

from __future__ import annotations

import unittest

import torch

from transformator.model import GPT, GPTConfig
from train import learning_rate


class LearningRateTest(unittest.TestCase):
    def test_warmup_then_cosine_down(self) -> None:
        base, floor, warmup, total = 3e-4, 3e-5, 10, 100
        first = learning_rate(1, total, base, floor, warmup)
        at_warmup = learning_rate(warmup, total, base, floor, warmup)
        last = learning_rate(total, total, base, floor, warmup)
        self.assertAlmostEqual(first, base / warmup)
        self.assertAlmostEqual(at_warmup, base)
        self.assertAlmostEqual(last, floor)
        self.assertGreater(at_warmup, last)


class ModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = GPTConfig(
            vocab_size=20, block_size=16, n_layer=2, n_head=2, n_embd=32
        )
        self.model = GPT(self.config).eval()

    def test_future_token_does_not_change_earlier_logits(self) -> None:
        torch.manual_seed(0)
        tokens = torch.randint(0, self.config.vocab_size, (2, 16))
        changed = tokens.clone()
        changed[:, -1] = (changed[:, -1] + 1) % self.config.vocab_size
        first, _ = self.model(tokens)
        second, _ = self.model(changed)
        self.assertTrue(torch.allclose(first[:, :-1], second[:, :-1], atol=1e-5))

    def test_greedy_is_deterministic(self) -> None:
        torch.manual_seed(0)
        prompt = torch.randint(0, self.config.vocab_size, (1, 4))
        a = self.model.generate(prompt.clone(), max_new_tokens=6, temperature=0)
        b = self.model.generate(prompt.clone(), max_new_tokens=6, temperature=0)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(a.shape, (1, 10))


if __name__ == "__main__":
    unittest.main()
