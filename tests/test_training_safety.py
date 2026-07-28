import numpy as np
import pytest

from flashrl.agents.dqn.train import DQNConfig, train_dqn
from flashrl.envs import DinoEnv


def safety_config(tmp_path, run_id):
    return DQNConfig(
        episodes=1,
        max_episode_steps=1,
        batch_size=1,
        warmup_steps=1,
        replay_size=2,
        selection_episodes=1,
        output_dir=str(tmp_path),
        run_id=run_id,
    )


def test_simulator_exception_aborts_training(monkeypatch, tmp_path):
    def fail_step(self, action):
        raise RuntimeError("deliberate simulator failure")

    monkeypatch.setattr(DinoEnv, "_step_sim", fail_step)
    with pytest.raises(RuntimeError, match="deliberate simulator failure"):
        train_dqn(safety_config(tmp_path, "simulator-failure"))


def test_invalid_observation_aborts_before_replay(monkeypatch, tmp_path):
    def invalid_observation(self):
        return np.full(self.observation_space.shape, np.nan, dtype=np.float32)

    monkeypatch.setattr(DinoEnv, "_observe", invalid_observation)
    with pytest.raises(RuntimeError, match="Invalid observation"):
        train_dqn(safety_config(tmp_path, "invalid-observation"))
