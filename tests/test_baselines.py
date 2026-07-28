from flashrl.agents.baselines import RuleBasedDinoAgent
from flashrl.envs import DinoEnv
from flashrl.envs.dino_env import Obstacle


def test_rule_policy_uses_bird_altitude():
    env = DinoEnv(obs_mode="state", action_mode="full", backend="sim", seed=5)
    env.reset(seed=5)
    policy = RuleBasedDinoAgent(action_mode="full")

    env.obstacles[0] = Obstacle(x=100, width=34, height=24, type_id=2, y=36)
    low_bird_action = policy.act(env._state_vector())
    env.obstacles[0] = Obstacle(x=100, width=34, height=24, type_id=2, y=68)
    high_bird_action = policy.act(env._state_vector())

    env.close()
    assert low_bird_action == 2
    assert high_bird_action == 0
