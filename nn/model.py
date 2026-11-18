"""Dual-head SE-ResNet (Policy + Value) with Player Identity Embeddings + FiLM modulation."""

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

        # Player identity embedding + FiLM conditioning for Policy & Value heads
        self.player_emb = nn.Embedding(player_vocab_size, player_emb_dim)
        self.film_mlp = nn.Sequential(
            nn.Linear(player_emb_dim * 2, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels * 2),
        )

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
            nn.Linear(64 + player_emb_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )

    def forward(
        self,
        x: torch.Tensor,
        mover_id: torch.Tensor,
        opp_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(self.stem(x))

        m_emb = self.player_emb(mover_id)
        o_emb = self.player_emb(opp_id)
        p_feat = torch.cat([m_emb, o_emb], dim=1)  # (B, 2 * player_emb_dim)

        # FiLM modulation: h_mod = h * (1 + gamma) + beta
        gamma_beta = self.film_mlp(p_feat)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        h_mod = h * (1.0 + gamma.unsqueeze(-1).unsqueeze(-1)) + beta.unsqueeze(-1).unsqueeze(-1)

        # Policy logits (B, 225)
        p = self.policy_conv(h_mod).flatten(1)

        # Value logits (B, 3)
        v = self.value_conv(h_mod)
        v_avg = v.mean(dim=(2, 3))
        v_max = v.amax(dim=(2, 3))
        v_cat = torch.cat([v_avg, v_max, p_feat], dim=1)
        v_logits = self.value_fc(v_cat)
        return p, v_logits
