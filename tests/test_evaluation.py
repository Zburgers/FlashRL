from flashrl.agents.dqn.train import DQNConfig, train_dqn
from flashrl.artifacts import sha256_file
from flashrl.benchmark.evaluate import evaluate_checkpoint


def test_checkpoint_evaluation_propagates_training_identity(tmp_path):
    trained = train_dqn(
        DQNConfig(
            episodes=1,
            max_episode_steps=2,
            batch_size=1,
            warmup_steps=1,
            replay_size=8,
            selection_interval_episodes=1,
            selection_episodes=1,
            output_dir=str(tmp_path),
            run_id="evaluation-source",
        )
    )
    rows = evaluate_checkpoint(
        trained["best_checkpoint_path"],
        episodes=2,
        eval_seed=8_000,
    )
    first, second = rows
    assert first["result_schema_version"] == 2
    assert first["training_run_id"] == trained["run_id"]
    assert first["algorithm_id"] == trained["algorithm_id"]
    assert first["train_frames"] == trained["train_frames"]
    assert first["checkpoint_role"] == "best"
    assert first["checkpoint_sha256"] == sha256_file(tmp_path / trained["run_id"] / "best.pt")
    assert first["episode_seed"] == 8_000
    assert second["episode_seed"] == 8_001
    assert first["wall_clock_episode_s"] >= 0
    assert "wall_clock_eval_s" not in first
