"""Aggregate benchmark CSV files into summary tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from flashrl.results import summarize_results


def load_rows(paths: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for pattern in paths:
        for path in sorted(Path().glob(pattern)):
            with path.open(newline="", encoding="utf-8") as fh:
                rows.extend(csv.DictReader(fh))
    return rows


def summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return summarize_results(rows)


def write_markdown(rows: list[dict[str, Any]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "algorithm_id",
        "runs",
        "episodes",
        "mean_score",
        "median_score",
        "standard_deviation",
        "ci95_low",
        "ci95_high",
    ]
    with out.open("w", encoding="utf-8") as fh:
        fh.write("| " + " | ".join(headers) + " |\n")
        fh.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            fh.write("| " + " | ".join(str(row[h]) for h in headers) + " |\n")


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("paths", nargs="+", help="CSV paths or glob patterns")
    parser.add_argument("--out", default="reports/benchmark_summary.md")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate FlashRL result CSV files")
    configure_parser(parser)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_rows, experiment_rows = summarize(load_rows(args.paths))
    write_markdown(experiment_rows, Path(args.out))
    run_path = Path(args.out).with_suffix(".runs.json")
    run_path.write_text(json.dumps(run_rows, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"wrote {run_path}")
    return run_rows, experiment_rows


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
