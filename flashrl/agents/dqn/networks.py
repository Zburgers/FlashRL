"""Q-network architectures matched to observation modes."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn


class DuelingHead(nn.Module):
    def __init__(self, in_features: int, n_actions: int) -> None:
        super().__init__()
        self.value = nn.Sequential(nn.Linear(in_features, 128), nn.ReLU(), nn.Linear(128, 1))
        self.advantage = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.value(x)
        advantage = self.advantage(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class StateDQN(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, dueling: bool = False) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.head = DuelingHead(128, n_actions) if dueling else nn.Linear(128, n_actions)

    def forward(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(obs, dict):
            obs = obs["state"]
        return self.head(self.body(obs.float()))


class VisionEncoder(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int]) -> None:
        super().__init__()
        c, h, w = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            sample = torch.zeros(1, c, h, w)
            self.output_dim = int(np.prod(self.conv(sample).shape[1:]))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.conv(image.float() / 255.0).flatten(1)


class VisionDQN(nn.Module):
    def __init__(
        self, input_shape: tuple[int, int, int], n_actions: int, dueling: bool = False
    ) -> None:
        super().__init__()
        self.encoder = VisionEncoder(input_shape)
        self.body = nn.Sequential(nn.Linear(self.encoder.output_dim, 512), nn.ReLU())
        self.head = DuelingHead(512, n_actions) if dueling else nn.Linear(512, n_actions)

    def forward(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        image = obs["image"] if isinstance(obs, dict) else obs
        return self.head(self.body(self.encoder(image)))


class HybridDQN(nn.Module):
    def __init__(
        self,
        image_shape: tuple[int, int, int],
        state_dim: int,
        n_actions: int,
        dueling: bool = False,
    ) -> None:
        super().__init__()
        self.image_encoder = VisionEncoder(image_shape)
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU())
        merged_dim = self.image_encoder.output_dim + 64
        self.body = nn.Sequential(nn.Linear(merged_dim, 512), nn.ReLU())
        self.head = DuelingHead(512, n_actions) if dueling else nn.Linear(512, n_actions)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        image_features = self.image_encoder(obs["image"])
        state_features = self.state_encoder(obs["state"].float())
        return self.head(self.body(torch.cat([image_features, state_features], dim=1)))


def build_q_network(
    observation_space: gym.Space,
    n_actions: int,
    obs_mode: str,
    dueling: bool = False,
) -> nn.Module:
    if obs_mode == "state":
        return StateDQN(int(observation_space.shape[0]), n_actions, dueling=dueling)
    if obs_mode == "vision":
        return VisionDQN(tuple(observation_space.shape), n_actions, dueling=dueling)
    if obs_mode == "hybrid":
        assert isinstance(observation_space, gym.spaces.Dict)
        return HybridDQN(
            image_shape=tuple(observation_space["image"].shape),
            state_dim=int(observation_space["state"].shape[0]),
            n_actions=n_actions,
            dueling=dueling,
        )
    raise ValueError(f"Unsupported obs_mode: {obs_mode}")
