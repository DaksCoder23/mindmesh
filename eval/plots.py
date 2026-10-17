"""Charts comparing the five methods on each metric.

Grouped bars, one chart per metric, methods on the x axis and cohort size as the
groups. Methods with no successful runs are labelled on the chart rather than dropped
silently.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

METHOD_ORDER = ("cpsat", "random", "self_selection", "marks_sorted", "direct_llm")

CHARTS = (
    ("skill_coverage_rate", "Skill coverage rate", "fraction of team-skill pairs covered", True),
    (
        "composition_score",
        "Composition score",
        "solver objective, evaluated on the assignment",
        True,
    ),
    ("demographic_parity_gap", "Demographic parity gap", "max deviation from cohort share", False),
    ("violations", "Hard constraint violations", "count per run (lower is better)", False),
    ("wall_time_seconds", "Wall clock time", "seconds per run", False),
)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def make_charts(rows: list[dict[str, object]], out_dir: Path) -> list[Path]:
    """One PNG per metric. Returns the paths written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = sorted({int(r["n"]) for r in rows})
    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in rows)]
    written: list[Path] = []

    for column, title, ylabel, _higher_is_better in CHARTS:
        figure, axes = plt.subplots(figsize=(9, 5))
        width = 0.8 / max(1, len(sizes))

        for offset, n in enumerate(sizes):
            heights, labels = [], []
            for method in methods:
                scored = [
                    float(r[column])
                    for r in rows
                    if r["method"] == method and int(r["n"]) == n and int(r["violations"]) >= 0
                ]
                heights.append(_mean(scored))
                labels.append(method)
            positions = [i + offset * width for i in range(len(methods))]
            axes.bar(positions, heights, width=width, label=f"n = {n}")

        axes.set_xticks([i + 0.4 - width / 2 for i in range(len(methods))])
        axes.set_xticklabels(methods, rotation=15)
        axes.set_title(title)
        axes.set_ylabel(ylabel)
        axes.legend()
        axes.grid(axis="y", alpha=0.3)

        # Say so on the chart when a method produced nothing, rather than showing a zero
        # bar that reads as a real measurement.
        for index, method in enumerate(methods):
            if not any(r["method"] == method and int(r["violations"]) >= 0 for r in rows):
                axes.text(
                    index + 0.3,
                    0,
                    "no successful runs",
                    rotation=90,
                    fontsize=8,
                    va="bottom",
                    ha="center",
                    color="crimson",
                )

        figure.tight_layout()
        path = out_dir / f"{column}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        written.append(path)
        logger.info("wrote %s", path)

    return written
