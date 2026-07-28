import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from flashrl.envs import DinoEnv
from flashrl.envs.dino_env import Obstacle


def test_state_env_contract():
    env = DinoEnv(obs_mode="state", backend="sim", max_episode_steps=5, seed=123)
    check_env(env, skip_render_check=True)
    env.close()


def test_observation_modes_reset_and_step():
    for obs_mode in ("state", "vision", "hybrid"):
        env = DinoEnv(obs_mode=obs_mode, backend="sim", max_episode_steps=3, seed=1)
        obs, info = env.reset(seed=1)
        assert env.observation_space.contains(obs)
        obs, reward, terminated, truncated, info = env.step(0)
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert info["obs_mode"] == obs_mode
        env.close()


def test_max_episode_truncates():
    env = DinoEnv(obs_mode="state", backend="sim", max_episode_steps=1, seed=1)
    env.reset(seed=1)
    _, _, _, truncated, info = env.step(0)
    assert truncated
    assert info["steps"] == 1
    env.close()


def test_seed_repeats_initial_state():
    env = DinoEnv(obs_mode="state", backend="sim", seed=7)
    obs1, _ = env.reset(seed=7)
    obs2, _ = env.reset(seed=7)
    assert np.allclose(obs1, obs2)
    env.close()


def test_seed_and_action_sequence_repeat_complete_trajectory():
    actions = [0, 0, 1, 0, 0, 3, 0, 2, 3, 0] * 8

    def rollout():
        env = DinoEnv(obs_mode="state", backend="sim", max_episode_steps=100, seed=17)
        obs, _ = env.reset(seed=17)
        trajectory = [obs.copy()]
        endings = []
        for action in actions:
            obs, reward, terminated, truncated, info = env.step(action)
            trajectory.append(obs.copy())
            endings.append((reward, terminated, truncated, info["score"]))
            if terminated or truncated:
                break
        env.close()
        return np.stack(trajectory), endings

    first_obs, first_endings = rollout()
    second_obs, second_endings = rollout()
    assert np.array_equal(first_obs, second_obs)
    assert first_endings == second_endings


def test_state_observation_distinguishes_bird_altitude():
    env = DinoEnv(obs_mode="state", backend="sim", seed=3)
    env.reset(seed=3)
    env.obstacles[0] = Obstacle(x=100, width=34, height=24, type_id=2, y=36)
    low_bird = env._state_vector()
    env.obstacles[0] = Obstacle(x=100, width=34, height=24, type_id=2, y=68)
    high_bird = env._state_vector()
    env.close()
    assert not np.array_equal(low_bird, high_bird)


@pytest.mark.parametrize("obs_mode", ["state", "vision", "hybrid"])
def test_long_successful_episode_stays_inside_observation_space(obs_mode):
    env = DinoEnv(
        obs_mode=obs_mode,
        backend="sim",
        max_episode_steps=3_000,
        seed=11,
    )
    env.reset(seed=11)
    env._detect_collision = lambda: False
    for _ in range(3_000):
        obs, _, _, truncated, _ = env.step(0)
        assert env.observation_space.contains(obs)
    assert truncated
    env.close()
