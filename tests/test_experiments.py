import json

import pytest

from flashrl.experiments import (
    ExperimentConfigurationError,
    execute_experiment,
    expand_jobs,
    load_experiment,
)


def configuration():
    return {
        "name": "pilot",
        "output_dir": "runs/pilot",
        "seeds": [0, 1],
        "evaluation": {"episodes": 10, "seed_base": 10_000},
        "base": {
            "episodes": 20,
            "max_episode_steps": 100,
            "batch_size": 16,
        },
        "variants": [
            {
                "name": "vanilla",
                "overrides": {"double_dqn": False, "dueling": False},
            },
            {
                "name": "double",
                "overrides": {"double_dqn": True, "dueling": False},
            },
        ],
    }


def test_json_compatible_yaml_configuration_loads(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text(json.dumps(configuration()), encoding="utf-8")
    loaded = load_experiment(path)
    assert loaded["name"] == "pilot"
    assert len(loaded["variants"]) == 2


def test_matrix_expansion_is_deterministic_and_descriptive():
    first = expand_jobs(configuration())
    second = expand_jobs(configuration())
    assert first == second
    assert [job.run_id for job in first] == [
        "pilot-vanilla-seed0",
        "pilot-vanilla-seed1",
        "pilot-double-seed0",
        "pilot-double-seed1",
    ]
    assert first[0].config.seed == 0
    assert first[0].config.double_dqn is False
    assert first[-1].evaluation_episodes == 10


def test_duplicate_variant_identity_is_rejected():
    invalid = configuration()
    invalid["variants"].append(invalid["variants"][0])
    with pytest.raises(ExperimentConfigurationError, match="Duplicate"):
        expand_jobs(invalid)


def test_unknown_dqn_configuration_field_is_rejected():
    invalid = configuration()
    invalid["base"]["invented_option"] = True
    with pytest.raises(ExperimentConfigurationError, match="invented_option"):
        expand_jobs(invalid)


def test_dry_run_describes_jobs_without_creating_output(tmp_path):
    config = configuration()
    config["output_dir"] = str(tmp_path / "runs")
    records = execute_experiment(config, dry_run=True)
    assert [record["status"] for record in records] == ["planned"] * 4
    assert records[0]["run_id"] == "pilot-vanilla-seed0"
    assert not (tmp_path / "runs").exists()


def test_resume_skips_only_jobs_with_complete_evaluation(tmp_path, monkeypatch):
    config = configuration()
    config["seeds"] = [0]
    config["variants"] = [config["variants"][0]]
    config["output_dir"] = str(tmp_path / "runs")
    run_dir = tmp_path / "runs" / "pilot-vanilla-seed0"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text('{"status": "completed"}', encoding="utf-8")
    (run_dir / "eval_results.jsonl").write_text("{}\n", encoding="utf-8")

    def unexpected_training(*args, **kwargs):
        raise AssertionError("completed job was retrained")

    monkeypatch.setattr("flashrl.experiments.train_dqn", unexpected_training)
    records = execute_experiment(config, resume=True)
    assert records == [
        {
            "run_id": "pilot-vanilla-seed0",
            "variant": "vanilla",
            "status": "skipped",
            "reason": "completed",
        }
    ]
