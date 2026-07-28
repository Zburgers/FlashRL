import json

from flashrl.agents.dqn.train import DQNConfig, train_dqn
from flashrl.artifacts import RunManifest, atomic_write_json, sha256_file
from flashrl.schemas import (
    ACTION_SCHEMA_VERSION,
    ENVIRONMENT_ID,
    OBSERVATION_SCHEMA_VERSION,
    REWARD_SCHEMA_VERSION,
)


def test_atomic_json_round_trip(tmp_path):
    path = tmp_path / "nested" / "manifest.json"
    atomic_write_json(path, {"schema_version": 2, "run_id": "run-1"})
    assert json.loads(path.read_text()) == {
        "schema_version": 2,
        "run_id": "run-1",
    }


def test_file_hash_changes_with_content(tmp_path):
    path = tmp_path / "artifact"
    path.write_bytes(b"first")
    first = sha256_file(path)
    path.write_bytes(b"second")
    assert sha256_file(path) != first


def test_manifest_round_trip_preserves_schema_identity():
    manifest = RunManifest(
        run_id="run-1",
        experiment_id="experiment-1",
        algorithm_id="dueling_double_dqn",
        hyperparameter_hash="a" * 16,
        training_seed=7,
        training_git_commit="deadbeef",
        git_dirty=False,
        started_at="2026-07-28T00:00:00+00:00",
    )
    restored = RunManifest.from_dict(manifest.to_dict())
    assert restored == manifest
    assert restored.environment_id == ENVIRONMENT_ID
    assert restored.observation_schema_version == OBSERVATION_SCHEMA_VERSION
    assert restored.action_schema_version == ACTION_SCHEMA_VERSION
    assert restored.reward_schema_version == REWARD_SCHEMA_VERSION


def test_training_finalizes_manifest_with_artifact_hashes(tmp_path):
    result = train_dqn(
        DQNConfig(
            episodes=1,
            max_episode_steps=2,
            batch_size=1,
            warmup_steps=1,
            replay_size=4,
            output_dir=str(tmp_path),
            run_id="manifest-run",
        )
    )
    path = tmp_path / result["run_id"] / "manifest.json"
    manifest = json.loads(path.read_text())
    assert manifest["status"] == "completed"
    assert manifest["train_frames"] == 2
    assert manifest["algorithm_id"] == "dueling_double_dqn"
    assert manifest["artifacts"]["best.pt"]["sha256"] == sha256_file(
        tmp_path / result["run_id"] / "best.pt"
    )
    assert manifest["artifacts"]["last.pt"]["sha256"] == sha256_file(
        tmp_path / result["run_id"] / "last.pt"
    )


def test_training_seed_is_not_part_of_hyperparameter_identity(tmp_path):
    manifests = []
    for seed in (3, 9):
        result = train_dqn(
            DQNConfig(
                episodes=1,
                max_episode_steps=1,
                seed=seed,
                batch_size=1,
                warmup_steps=1,
                replay_size=2,
                selection_episodes=1,
                output_dir=str(tmp_path),
                run_id=f"seed-{seed}",
            )
        )
        manifests.append(json.loads((tmp_path / result["run_id"] / "manifest.json").read_text()))
    assert manifests[0]["experiment_id"] == manifests[1]["experiment_id"]
    assert manifests[0]["hyperparameter_hash"] == manifests[1]["hyperparameter_hash"]
    assert manifests[0]["training_seed"] != manifests[1]["training_seed"]


def test_training_stops_at_exact_environment_frame_budget(tmp_path):
    result = train_dqn(
        DQNConfig(
            episodes=100,
            total_train_frames=7,
            max_episode_steps=100,
            batch_size=1,
            warmup_steps=1,
            replay_size=10,
            selection_episodes=1,
            output_dir=str(tmp_path),
            run_id="frame-budget",
        )
    )
    assert result["train_frames"] == 7
    metrics = (tmp_path / result["run_id"] / "train_metrics.csv").read_text()
    assert "frame_budget" in metrics
