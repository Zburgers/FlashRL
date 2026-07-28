import torch
import csv

from flashrl.agents.dqn.train import DQNConfig, compute_td_target, train_dqn


def test_terminal_transition_does_not_bootstrap():
    target = compute_td_target(
        rewards=torch.tensor([2.0]),
        discounts=torch.tensor([0.99]),
        next_q=torch.tensor([10.0]),
        terminated=torch.tensor([True]),
    )
    assert target.item() == 2.0


def test_time_limit_transition_bootstraps():
    target = compute_td_target(
        rewards=torch.tensor([2.0]),
        discounts=torch.tensor([0.99]),
        next_q=torch.tensor([10.0]),
        terminated=torch.tensor([False]),
    )
    assert torch.allclose(target, torch.tensor([11.9]))


def test_training_metrics_record_termination_and_truncation(tmp_path):
    result = train_dqn(
        DQNConfig(
            episodes=1,
            max_episode_steps=1,
            batch_size=1,
            warmup_steps=1,
            replay_size=4,
            output_dir=str(tmp_path),
            run_id="ending-signals",
        )
    )
    with open(
        tmp_path / result["run_id"] / "train_metrics.csv",
        newline="",
        encoding="utf-8",
    ) as fh:
        row = next(csv.DictReader(fh))
    assert row["terminated"] == "False"
    assert row["truncated"] == "True"
    assert row["ending_reason"] == "time_limit"
