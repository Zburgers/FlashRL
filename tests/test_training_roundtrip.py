import csv
import json

from flashrl.agents.dqn.train import DQNConfig, load_checkpoint, train_dqn


def tiny_config(tmp_path, *, episodes, run_id="roundtrip"):
    return DQNConfig(
        episodes=episodes,
        max_episode_steps=3,
        batch_size=1,
        warmup_steps=1,
        replay_size=16,
        selection_interval_episodes=1,
        selection_episodes=2,
        output_dir=str(tmp_path),
        run_id=run_id,
    )


def test_training_writes_distinct_best_and_last_checkpoints(tmp_path):
    result = train_dqn(tiny_config(tmp_path, episodes=2))
    run_dir = tmp_path / result["run_id"]
    assert (run_dir / "best.pt").is_file()
    assert (run_dir / "last.pt").is_file()
    assert load_checkpoint(run_dir / "best.pt")["role"] == "best"
    assert load_checkpoint(run_dir / "last.pt")["role"] == "last"
    assert result["best_checkpoint_path"] == str(run_dir / "best.pt")
    assert result["last_checkpoint_path"] == str(run_dir / "last.pt")

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert set(manifest["artifacts"]) >= {
        "best.pt",
        "last.pt",
        "config.json",
        "train_metrics.csv",
    }


def test_resume_continues_episode_and_frame_counts(tmp_path):
    first = train_dqn(tiny_config(tmp_path, episodes=1, run_id="resumable"))
    first_frames = first["train_frames"]
    first_manifest = json.loads(
        (tmp_path / "resumable" / "manifest.json").read_text()
    )
    resumed = train_dqn(
        tiny_config(tmp_path, episodes=2, run_id="resumable"),
        resume_path=first["last_checkpoint_path"],
    )
    assert resumed["train_frames"] > first_frames
    resumed_manifest = json.loads(
        (tmp_path / "resumable" / "manifest.json").read_text()
    )
    assert resumed_manifest["experiment_id"] == first_manifest["experiment_id"]
    with open(
        tmp_path / "resumable" / "train_metrics.csv",
        newline="",
        encoding="utf-8",
    ) as fh:
        rows = list(csv.DictReader(fh))
    assert [int(row["episode"]) for row in rows] == [0, 1]
