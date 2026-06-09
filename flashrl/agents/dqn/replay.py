"""Replay buffers for DQN."""

from __future__ import annotations

from collections import deque
import random
from typing import Any, NamedTuple

import numpy as np


class Transition(NamedTuple):
    obs: Any
    action: int
    reward: float
    next_obs: Any
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int | None = None) -> None:
        self.capacity = int(capacity)
        self.buffer: list[Transition] = []
        self.position = 0
        self.rng = random.Random(seed)

    def push(self, transition: Transition) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        indices = self.rng.sample(range(len(self.buffer)), batch_size)
        weights = np.ones(batch_size, dtype=np.float32)
        return [self.buffer[i] for i in indices], indices, weights

    def update_priorities(self, indices, priorities) -> None:
        return None

    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        capacity: int,
        seed: int | None = None,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
    ) -> None:
        super().__init__(capacity, seed=seed)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.priorities = np.zeros(capacity, dtype=np.float32)

    def push(self, transition: Transition) -> None:
        max_priority = float(self.priorities.max()) if self.buffer else 1.0
        idx = self.position
        super().push(transition)
        self.priorities[idx] = max(max_priority, 1e-6)

    def sample(self, batch_size: int):
        priorities = self.priorities[: len(self.buffer)]
        probs = priorities**self.alpha
        probs = probs / probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        return [self.buffer[int(i)] for i in indices], indices, weights.astype(np.float32)

    def update_priorities(self, indices, priorities) -> None:
        for idx, priority in zip(indices, priorities):
            self.priorities[int(idx)] = float(abs(priority)) + 1e-6


class NStepBuffer:
    def __init__(self, n: int, gamma: float) -> None:
        self.n = int(n)
        self.gamma = float(gamma)
        self.buffer: deque[Transition] = deque(maxlen=self.n)

    def append(self, transition: Transition) -> Transition | None:
        self.buffer.append(transition)
        if len(self.buffer) < self.n and not transition.done:
            return None
        return self.pop()

    def pop(self) -> Transition | None:
        if not self.buffer:
            return None
        reward = 0.0
        next_obs = self.buffer[-1].next_obs
        done = self.buffer[-1].done
        for idx, item in enumerate(self.buffer):
            reward += (self.gamma**idx) * item.reward
            next_obs = item.next_obs
            done = item.done
            if item.done:
                break
        first = self.buffer.popleft()
        return Transition(first.obs, first.action, reward, next_obs, done)

    def flush(self) -> list[Transition]:
        out: list[Transition] = []
        while self.buffer:
            item = self.pop()
            if item is not None:
                out.append(item)
        return out
