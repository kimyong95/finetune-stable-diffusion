import torch
from collections import defaultdict

class RewardNormalizer:
    def __init__(self):
        self.stats = defaultdict(list)  # per-prompt reward history

    def fit(self, keys: torch.Tensor, rewards: torch.Tensor):
        for k, r in zip(keys, rewards):
            self.stats[k.item()].append(r.item())  # keep history as scalars

    def transform(self, keys: torch.Tensor, rewards: torch.Tensor):

        advantages = torch.zeros_like(rewards, dtype=torch.float64)

        # global std from this batch
        std = rewards.std(unbiased=False) + 1e-4  # scalar

        for i, (k, r) in enumerate(zip(keys, rewards)):
            mean = torch.tensor(self.stats[k.item()], dtype=torch.float64).mean()
            advantages[i] = (r - mean) / std

        return advantages

    def clear(self):
        self.stats = defaultdict(list)
