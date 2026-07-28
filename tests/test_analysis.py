import csv
import json

import pytest

from flashrl.analysis import (
    area_under_curve,
    bootstrap_ci,
    failure_taxonomy,
    generate_report,
    select_representative_runs,
    standardized_effect_size,
)


def test_area_under_learning_curve_uses_environment_frames():
    points = [(0, 0.0), (100, 10.0), (200, 10.0)]
    assert area_under_curve(points) == pytest.approx(1_500.0)


def test_bootstrap_confidence_interval_is_seeded_and_contains_mean():
    first = bootstrap_ci([10, 20, 30, 40], samples=500, seed=7)
    second = bootstrap_ci([10, 20, 30, 40], samples=500, seed=7)
    assert first == second
    assert first[0] <= 25 <= first[1]


def test_standardized_effect_size_reports_improvement():
    assert standardized_effect_size([20, 22, 24], [10, 12, 14]) > 0


def test_failure_taxonomy_counts_and_normalizes_reasons():
    rows = [
        {"ending_reason": "late_jump"},
        {"ending_reason": "late_jump"},
        {"ending_reason": "bird_no_duck"},
    ]
    taxonomy = failure_taxonomy(rows)
    assert taxonomy["late_jump"] == {"count": 2, "rate": pytest.approx(2 / 3)}
    assert taxonomy["bird_no_duck"]["count"] == 1


def test_representative_runs_select_median_and_best_seed():
    runs = [
        {"training_run_id": "low", "mean_score": 10},
        {"training_run_id": "middle", "mean_score": 30},
        {"training_run_id": "best", "mean_score": 80},
    ]
    selected = select_representative_runs(runs)
    assert selected["median"]["training_run_id"] == "middle"
    assert selected["best"]["training_run_id"] == "best"


def test_report_is_traceable_to_manifests_and_publishes_data(tmp_path):
    run_root = tmp_path / "runs"
    run_dir = run_root / "pilot-double-seed0"
    run_dir.mkdir(parents=True)
    manifest = {
        "status": "completed",
        "run_id": "pilot-double-seed0",
        "experiment_id": "double_dqn-abc",
        "algorithm_id": "double_dqn",
        "hyperparameter_hash": "abc",
        "training_seed": 0,
        "train_frames": 30,
        "wall_clock_train_s": 1.5,
        "artifacts": {"best.pt": {"sha256": "f" * 64}},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with (run_dir / "train_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["episode", "score", "steps", "ending_reason"])
        writer.writeheader()
        writer.writerows(
            [
                {"episode": 0, "score": 10, "steps": 10, "ending_reason": "late_jump"},
                {"episode": 1, "score": 20, "steps": 20, "ending_reason": "time_limit"},
            ]
        )
    (run_dir / "eval_results.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "training_run_id": "pilot-double-seed0",
                    "score": score,
                    "ending_reason": reason,
                }
            )
            for score, reason in [(40, "late_jump"), (60, "time_limit")]
        )
        + "\n",
        encoding="utf-8",
    )
    baseline_dir = run_root / "pilot-random-baseline"
    baseline_dir.mkdir()
    baseline_manifest = manifest | {
        "run_id": "pilot-random-baseline",
        "experiment_id": "random-baseline",
        "algorithm_id": "random",
        "hyperparameter_hash": "",
        "train_frames": 0,
        "wall_clock_train_s": 0,
        "artifacts": {},
    }
    (baseline_dir / "manifest.json").write_text(json.dumps(baseline_manifest), encoding="utf-8")
    (baseline_dir / "eval_results.jsonl").write_text(
        json.dumps(
            {
                "training_run_id": "pilot-random-baseline",
                "score": 12,
                "ending_reason": "late_jump",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "reports" / "pilot_report.md"
    published = tmp_path / "reports"
    result = generate_report(run_root, out=out, publish_data=published)
    assert result["runs"] == 2
    report = out.read_text()
    assert "pilot-double-seed0" in report
    assert "random" in report
    assert "f" * 12 in report
    assert (published / "pilot_runs.csv").is_file()
    assert (published / "pilot_summary.csv").is_file()
    assert (published / "figures" / "learning_curves.svg").is_file()
