import numpy as np
import pytest

from flashrl.agents.dqn.replay import (
    NStepBuffer,
    PrioritizedReplayBuffer,
    Transition,
)


def transition(reward, *, terminated=False, truncated=False):
    return Transition(
        obs=np.array([reward], dtype=np.float32),
        action=0,
        reward=float(reward),
        next_obs=np.array([reward + 1], dtype=np.float32),
        terminated=terminated,
        truncated=truncated,
        discount=0.99,
    )


def test_n_step_transition_records_effective_discount():
    buffer = NStepBuffer(n=3, gamma=0.9)
    assert buffer.append(transition(1)) is None
    assert buffer.append(transition(2)) is None
    emitted = buffer.append(transition(3))
    assert emitted.reward == pytest.approx(1 + 0.9 * 2 + 0.9**2 * 3)
    assert emitted.discount == pytest.approx(0.9**3)
    assert not emitted.terminated
    assert not emitted.truncated


def test_short_n_step_flush_uses_actual_number_of_steps():
    buffer = NStepBuffer(n=3, gamma=0.9)
    buffer.append(transition(1))
    emitted = buffer.append(transition(2, truncated=True))
    assert emitted.reward == pytest.approx(1 + 0.9 * 2)
    assert emitted.discount == pytest.approx(0.9**2)
    assert not emitted.terminated
    assert emitted.truncated


def test_n_step_preserves_terminal_signal():
    buffer = NStepBuffer(n=3, gamma=0.9)
    buffer.append(transition(1))
    emitted = buffer.append(transition(2, terminated=True))
    assert emitted.terminated
    assert not emitted.truncated


def test_prioritized_replay_sampling_is_seeded():
    first = PrioritizedReplayBuffer(capacity=16, seed=42)
    second = PrioritizedReplayBuffer(capacity=16, seed=42)
    for value in range(10):
        first.push(transition(value))
        second.push(transition(value))
    _, first_indices, _ = first.sample(4)
    _, second_indices, _ = second.sample(4)
    assert np.array_equal(first_indices, second_indices)

