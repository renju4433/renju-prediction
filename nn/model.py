"""Dual-head SE-ResNet (Policy + Value) for Renju/Gomoku with player feature planes."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.features import NUM_INPUT_PLANES


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = x.mean(dim=(2, 3))
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s)).view(b, c, 1, 1)
        return x * s


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(x + out, inplace=True)


class RenjuDualNet(nn.Module):
    def __init__(
        self,
        in_channels: int = NUM_INPUT_PLANES,
        channels: int = 64,
        num_blocks: int = 6,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])

        # Policy head: (B, channels, 15, 15) -> (B, 225)
        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1, bias=True),
        )

        # Value head: (B, channels, 15, 15) -> [tf_win, draw, ts_win] logits
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(self.stem(x))

        # Policy logits (B, 225)
        p = self.policy_conv(h).flatten(1)

        # Value logits (B, 3) using global avg + max pooling
        v = self.value_conv(h)
        v_avg = v.mean(dim=(2, 3))
        v_max = v.amax(dim=(2, 3))
        v_cat = torch.cat([v_avg, v_max], dim=1)
        v_logits = self.value_fc(v_cat)
        return p, v_logits
