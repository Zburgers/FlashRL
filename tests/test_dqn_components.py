from flashrl.agents.dqn.networks import build_q_network
from flashrl.agents.dqn.replay import ReplayBuffer, Transition
from flashrl.agents.dqn.train import DQNConfig, optimize
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


def test_training_smoke_performs_real_optimizer_update():
    import torch

    env = DinoEnv(obs_mode="state", backend="sim", seed=5)
    obs, _ = env.reset(seed=5)
    policy = build_q_network(env.observation_space, env.action_space.n, "state", dueling=False)
    target = build_q_network(env.observation_space, env.action_space.n, "state", dueling=False)
    target.load_state_dict(policy.state_dict())
    replay = ReplayBuffer(capacity=4, seed=5)
    for action in (0, 1):
        replay.push(
            Transition(
                obs=obs,
                action=action,
                reward=10.0,
                next_obs=obs,
                terminated=True,
                truncated=False,
                discount=0.99,
            )
        )
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    before = [parameter.detach().clone() for parameter in policy.parameters()]
    loss = optimize(
        policy,
        target,
        optimizer,
        replay,
        DQNConfig(batch_size=2, warmup_steps=1, double_dqn=False, dueling=False),
        torch.device("cpu"),
    )
    assert loss is not None
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, policy.parameters(), strict=True)
    )
    env.close()
