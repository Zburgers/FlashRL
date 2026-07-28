from flashrl.agents.dqn.networks import build_q_network
from flashrl.envs import DinoEnv


def test_network_matches_observation_modes():
    for obs_mode in ("state", "vision", "hybrid"):
        env = DinoEnv(obs_mode=obs_mode, backend="sim", seed=0)
        model = build_q_network(env.observation_space, env.action_space.n, obs_mode, dueling=True)
        obs, _ = env.reset(seed=0)
        import torch

        from flashrl.agents.dqn.train import obs_to_torch

        with torch.no_grad():
            q_values = model(obs_to_torch(obs, torch.device("cpu")))
        assert q_values.shape == (1, env.action_space.n)
        env.close()
