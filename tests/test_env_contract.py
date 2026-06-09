import numpy as np
from gymnasium.utils.env_checker import check_env

from flashrl.envs import DinoEnv


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
