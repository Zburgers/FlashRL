"""Small, deterministic statistics for FlashRL experiment reports."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def area_under_curve(points: Sequence[tuple[int, float]]) -> float:
    """Integrate a learning curve over environment frames."""

    if len(points) < 2:
        return 0.0
    ordered = sorted(points)
    if len({frame for frame, _ in ordered}) != len(ordered):
        raise ValueError("Learning-curve frames must be unique")
    return float(
        sum(
            (right_frame - left_frame) * (left_value + right_value) / 2
            for (left_frame, left_value), (right_frame, right_value) in zip(ordered, ordered[1:])
        )
    )


def bootstrap_ci(
    values: Sequence[float], samples: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    """Return a seeded percentile confidence interval for the sample mean."""

    if not values:
        raise ValueError("At least one value is required")
    if samples <= 0:
        raise ValueError("samples must be positive")
    source = np.asarray(values, dtype=np.float64)
    if len(source) == 1:
        return float(source[0]), float(source[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(source, size=(samples, len(source)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def standardized_effect_size(candidate: Sequence[float], baseline: Sequence[float]) -> float:
    """Compute pooled-standard-deviation Cohen's d."""

    if len(candidate) < 2 or len(baseline) < 2:
        raise ValueError("Each group requires at least two observations")
    candidate_variance = statistics.variance(candidate)
    baseline_variance = statistics.variance(baseline)
    pooled = math.sqrt(
        ((len(candidate) - 1) * candidate_variance + (len(baseline) - 1) * baseline_variance)
        / (len(candidate) + len(baseline) - 2)
    )
    difference = statistics.fmean(candidate) - statistics.fmean(baseline)
    if pooled == 0:
        return math.copysign(float("inf"), difference) if difference else 0.0
    return float(difference / pooled)


def failure_taxonomy(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Count and normalize terminal failure reasons."""

    reasons = [str(row.get("ending_reason", "unknown")) for row in rows]
    if not reasons:
        return {}
    counts = Counter(reasons)
    return {
        reason: {"count": count, "rate": count / len(reasons)}
        for reason, count in sorted(counts.items())
    }


def select_representative_runs(
    runs: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Select the lower-median and best run by held-out mean score."""

    if not runs:
        raise ValueError("At least one run is required")
    ordered = sorted(runs, key=lambda run: (float(run["mean_score"]), str(run["training_run_id"])))
    return {"median": ordered[(len(ordered) - 1) // 2], "best": ordered[-1]}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _training_curve(path: Path) -> tuple[list[tuple[int, float]], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    frames = 0
    points: list[tuple[int, float]] = [(0, 0.0)]
    for row in rows:
        frames += int(float(row["steps"]))
        points.append((frames, float(row["score"])))
    return points, rows


def _collect_runs(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for manifest_path in sorted(run_dir.glob("*/manifest.json")):
        child = manifest_path.parent
        eval_path = child / "eval_results.jsonl"
        metrics_path = child / "train_metrics.csv"
        if not eval_path.is_file() or not metrics_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            continue
        evaluation = _load_jsonl(eval_path)
        if not evaluation:
            continue
        curve, training = _training_curve(metrics_path)
        scores = [float(row["score"]) for row in evaluation]
        checkpoint_hash = manifest.get("artifacts", {}).get("best.pt", {}).get("sha256", "")
        run = {
            "experiment_id": manifest["experiment_id"],
            "algorithm_id": manifest["algorithm_id"],
            "hyperparameter_hash": manifest["hyperparameter_hash"],
            "training_run_id": manifest["run_id"],
            "training_seed": manifest["training_seed"],
            "train_frames": manifest.get("train_frames", curve[-1][0]),
            "wall_clock_train_s": manifest.get("wall_clock_train_s", 0.0),
            "evaluation_episodes": len(scores),
            "mean_score": statistics.fmean(scores),
            "median_score": statistics.median(scores),
            "standard_deviation": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "best_score": max(scores),
            "learning_auc": area_under_curve(curve),
            "checkpoint_sha256": checkpoint_hash,
            "manifest_path": str(manifest_path),
        }
        runs.append(run)
        details[manifest["run_id"]] = {
            "evaluation": evaluation,
            "training": training,
            "curve": curve,
        }
    if not runs:
        raise ValueError(f"No completed evaluated runs found in {run_dir}")
    return runs, details


def _summarize_runs(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[str(run["experiment_id"])].append(run)
    summaries: list[dict[str, Any]] = []
    for experiment_id in sorted(grouped):
        group = grouped[experiment_id]
        means = [float(run["mean_score"]) for run in group]
        low, high = bootstrap_ci(means, samples=10_000)
        summaries.append(
            {
                "experiment_id": experiment_id,
                "algorithm_id": group[0]["algorithm_id"],
                "hyperparameter_hash": group[0]["hyperparameter_hash"],
                "training_runs": len(group),
                "evaluation_episodes": sum(int(run["evaluation_episodes"]) for run in group),
                "mean_score": statistics.fmean(means),
                "median_run_score": statistics.median(means),
                "run_standard_deviation": (statistics.stdev(means) if len(means) > 1 else 0.0),
                "ci95_low": low,
                "ci95_high": high,
                "mean_learning_auc": statistics.fmean(float(run["learning_auc"]) for run in group),
                "mean_wall_clock_train_s": statistics.fmean(
                    float(run["wall_clock_train_s"]) for run in group
                ),
            }
        )
    return summaries


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _svg_document(title: str, body: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="520" '
        'viewBox="0 0 960 520" role="img">'
        f"<title>{title}</title>"
        '<rect width="960" height="520" fill="#101216"/>'
        "<style>text{font-family:ui-monospace,monospace;fill:#d9d5ca}"
        ".axis{stroke:#555b62;stroke-width:1}.label{font-size:13px}"
        ".title{font-size:22px;font-weight:700}</style>"
        f'<text x="56" y="42" class="title">{title}</text>{body}</svg>\n'
    )


def _write_figures(
    figure_dir: Path, runs: Sequence[dict[str, Any]], details: Mapping[str, Any]
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#ff7a45", "#4dc9b0", "#73a8ff", "#d28bff", "#f0c75e", "#e8668a"]
    all_points = [point for run in runs for point in details[str(run["training_run_id"])]["curve"]]
    maximum_frame = max(frame for frame, _ in all_points) or 1
    maximum_score = max(score for _, score in all_points) or 1.0
    learning = (
        '<line x1="70" y1="460" x2="920" y2="460" class="axis"/>'
        '<line x1="70" y1="70" x2="70" y2="460" class="axis"/>'
    )
    for index, run in enumerate(runs):
        points = details[str(run["training_run_id"])]["curve"]
        coordinates = " ".join(
            f"{70 + frame / maximum_frame * 850:.1f},{460 - score / maximum_score * 370:.1f}"
            for frame, score in points
        )
        learning += (
            f'<polyline points="{coordinates}" fill="none" '
            f'stroke="{colors[index % len(colors)]}" stroke-width="2" opacity=".8"/>'
        )
    learning += (
        f'<text x="840" y="492" class="label">frames {maximum_frame:,}</text>'
        f'<text x="76" y="88" class="label">score {maximum_score:.0f}</text>'
    )
    (figure_dir / "learning_curves.svg").write_text(
        _svg_document("Learning curves by environment frames", learning),
        encoding="utf-8",
    )

    evaluation = [row for run in runs for row in details[str(run["training_run_id"])]["evaluation"]]
    taxonomy = failure_taxonomy(evaluation)
    failures = ""
    for index, (reason, values) in enumerate(taxonomy.items()):
        y = 90 + index * 52
        width = float(values["rate"]) * 700
        failures += (
            f'<text x="70" y="{y + 20}" class="label">{reason}</text>'
            f'<rect x="220" y="{y}" width="{width:.1f}" height="28" fill="#ff7a45"/>'
            f'<text x="{230 + width:.1f}" y="{y + 20}" class="label">'
            f"{float(values['rate']):.1%}</text>"
        )
    (figure_dir / "failure_taxonomy.svg").write_text(
        _svg_document("Held-out ending reasons", failures), encoding="utf-8"
    )

    distributions = ""
    for index, run in enumerate(runs):
        scores = [float(row["score"]) for row in details[str(run["training_run_id"])]["evaluation"]]
        x = 90 + index * min(130, 800 / max(1, len(runs)))
        height = (
            statistics.fmean(scores)
            / max(
                1.0,
                max(
                    float(row["score"])
                    for item in runs
                    for row in details[str(item["training_run_id"])]["evaluation"]
                ),
            )
            * 330
        )
        distributions += (
            f'<rect x="{x:.1f}" y="{450 - height:.1f}" width="72" height="{height:.1f}" '
            f'fill="{colors[index % len(colors)]}"/>'
            f'<text x="{x:.1f}" y="476" class="label">s{run["training_seed"]}</text>'
        )
    (figure_dir / "score_distributions.svg").write_text(
        _svg_document("Held-out mean score by training seed", distributions),
        encoding="utf-8",
    )


def generate_report(
    run_dir: str | Path,
    *,
    out: str | Path,
    publish_data: str | Path | None = None,
) -> dict[str, Any]:
    """Generate traceable Markdown, CSV, and dependency-free SVG evidence."""

    run_dir = Path(run_dir)
    out = Path(out)
    runs, details = _collect_runs(run_dir)
    summaries = _summarize_runs(runs)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# FlashRL experiment report",
        "",
        f"Source directory: `{run_dir}`",
        "",
        "## Held-out results",
        "",
        "| Algorithm | Runs | Episodes | Mean | 95% CI | Mean AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['algorithm_id']} | {summary['training_runs']} | "
            f"{summary['evaluation_episodes']} | {summary['mean_score']:.2f} | "
            f"[{summary['ci95_low']:.2f}, {summary['ci95_high']:.2f}] | "
            f"{summary['mean_learning_auc']:.0f} |"
        )
    lines.extend(
        [
            "",
            "## Run provenance",
            "",
            "| Run | Seed | Mean | Frames | Checkpoint SHA-256 | Manifest |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for run in runs:
        lines.append(
            f"| {run['training_run_id']} | {run['training_seed']} | "
            f"{run['mean_score']:.2f} | {run['train_frames']} | "
            f"`{str(run['checkpoint_sha256'])[:12]}` | `{run['manifest_path']}` |"
        )
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")

    if publish_data is not None:
        publish_dir = Path(publish_data)
        prefix = out.stem.removesuffix("_report")
        _write_csv(publish_dir / f"{prefix}_runs.csv", runs)
        _write_csv(publish_dir / f"{prefix}_summary.csv", summaries)
        _write_figures(publish_dir / "figures", runs, details)
    print(f"wrote {out}")
    return {"runs": len(runs), "experiments": len(summaries), "report": str(out)}
