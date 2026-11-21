"""Unified SE-ResNet for Renju: one trunk, a 226-way policy (board + swap) and a 3-way value.

Player identity is injected twice:
  * as FiLM modulation of the convolutional trunk (tentative-first / tentative-second embeddings)
  * straight into the swap and value heads
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.features import NUM_BOARD_ACTIONS, NUM_INPUT_PLANES


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


class RenjuUnifiedNet(nn.Module):
    """Policy over 225 board points + swap, and P(tf_win, draw, ts_win)."""

    def __init__(
        self,
        in_channels: int = NUM_INPUT_PLANES,
        channels: int = 96,
        num_blocks: int = 8,
        player_vocab_size: int = 1,
        player_emb_dim: int = 16,
    ) -> None:
        super().__init__()
        self.player_vocab_size = player_vocab_size
        self.player_emb_dim = player_emb_dim

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])

        self.tf_emb = nn.Embedding(player_vocab_size, player_emb_dim)
        self.ts_emb = nn.Embedding(player_vocab_size, player_emb_dim)

        # FiLM: trunk conditioning on both players
        self.film_mlp = nn.Sequential(
            nn.Linear(player_emb_dim * 2, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels * 2),
        )

        # Board policy head: (B, C, 15, 15) -> (B, 225)
        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1, bias=True),
        )

        # Swap head: pooled features + player identity -> 1 logit
        self.swap_head = nn.Sequential(
            nn.Linear(channels * 2 + player_emb_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        # Value head: P(tf_win), P(draw), P(ts_win)
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(64 + player_emb_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )

    def forward(
        self,
        x: torch.Tensor,
        tf_id: torch.Tensor,
        ts_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(self.stem(x))

        tf_e = self.tf_emb(tf_id)
        ts_e = self.ts_emb(ts_id)
        p_feat = torch.cat([tf_e, ts_e], dim=1)

        gamma_beta = self.film_mlp(p_feat)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        h = h * (1.0 + gamma.unsqueeze(-1).unsqueeze(-1)) + beta.unsqueeze(-1).unsqueeze(-1)

        board_logits = self.policy_conv(h).flatten(1)[:, :NUM_BOARD_ACTIONS]

        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        swap_logit = self.swap_head(torch.cat([pooled, p_feat], dim=1))
        policy_logits = torch.cat([board_logits, swap_logit], dim=1)

        v = self.value_conv(h)
        v_cat = torch.cat([v.mean(dim=(2, 3)), v.amax(dim=(2, 3)), p_feat], dim=1)
        value_logits = self.value_fc(v_cat)
        return policy_logits, value_logits
