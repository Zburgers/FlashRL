"""Aggregate benchmark CSV files into summary tables."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import statistics


def load_rows(paths: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for pattern in paths:
        for path in sorted(Path().glob(pattern)):
            with path.open(newline="", encoding="utf-8") as fh:
                rows.extend(csv.DictReader(fh))
    return rows


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["agent"], row["obs_mode"], row["action_mode"])
        groups[key].append(float(row["score"]))
    out: list[dict[str, str]] = []
    for (agent, obs_mode, action_mode), scores in sorted(groups.items()):
        scores_sorted = sorted(scores)
        out.append(
            {
                "agent": agent,
                "obs_mode": obs_mode,
                "action_mode": action_mode,
                "episodes": str(len(scores)),
                "mean_score": f"{statistics.fmean(scores):.3f}",
                "median_score": f"{statistics.median(scores):.3f}",
                "best_score": f"{max(scores):.3f}",
                "q1_score": f"{scores_sorted[int(0.25 * (len(scores_sorted) - 1))]:.3f}",
                "q3_score": f"{scores_sorted[int(0.75 * (len(scores_sorted) - 1))]:.3f}",
            }
        )
    return out


def write_markdown(rows: list[dict[str, str]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    headers = ["agent", "obs_mode", "action_mode", "episodes", "mean_score", "median_score", "best_score", "q1_score", "q3_score"]
    with out.open("w", encoding="utf-8") as fh:
        fh.write("| " + " | ".join(headers) + " |\n")
        fh.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            fh.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate FlashRL result CSV files")
    parser.add_argument("paths", nargs="+", help="CSV paths or glob patterns")
    parser.add_argument("--out", default="reports/benchmark_summary.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = summarize(load_rows(args.paths))
    write_markdown(rows, Path(args.out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
