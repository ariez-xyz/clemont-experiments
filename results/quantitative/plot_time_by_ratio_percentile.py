#!/usr/bin/env python3
"""Histogram of average time per max_ratio percentile bin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:  # pragma: no cover - script entry point convenience
    from ._plot_utils import resolve_json_paths
except ImportError:  # pragma: no cover - script executed from repo root
    from _plot_utils import resolve_json_paths

DEFAULT_BINS = 20
DEFAULT_FIGSIZE = (10, 6)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Bin records by their max_ratio percentile and plot the mean time (ms) per bin."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=(
            "Optional list of quant_run_*.json files or directories containing them. "
            "Defaults to all quant_run_*.json alongside this script when omitted."
        ),
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_BINS,
        help=f"Number of percentile bins to use (default: {DEFAULT_BINS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output image path (defaults to <json>_ratio_percentile_time.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively in addition to saving it",
    )
    args = parser.parse_args()

    if args.bins <= 0:
        raise SystemExit("--bins must be greater than zero")

    script_dir = Path(__file__).resolve().parent
    target_paths = args.paths or [script_dir]
    json_paths: list[Path] = []
    for path in target_paths:
        json_paths.extend(resolve_json_paths(path, default_dir=script_dir))

    datasets: list[Tuple[Path, np.ndarray, np.ndarray]] = []
    for json_path in json_paths:
        ratios, times = _load_ratio_time(json_path)
        if ratios.size == 0:
            print(f"No valid ratio/time pairs in {json_path}; skipping")
            continue
        datasets.append((json_path, ratios, times))

    if not datasets:
        raise SystemExit("No usable records found in the supplied paths")

    _plot_histogram(
        datasets=datasets,
        bins=args.bins,
        output_path=args.output,
        show=args.show,
    )


def _load_ratio_time(json_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    ratio_list = []
    time_list = []
    for record in payload.get("records", []):
        ratio = record.get("max_ratio")
        time = record.get("time")
        if ratio is None or time is None:
            continue
        try:
            ratio = float(ratio)
            time = float(time)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(ratio) or not np.isfinite(time):
            continue
        ratio_list.append(ratio)
        time_list.append(time)

    if not ratio_list:
        return np.array([]), np.array([])

    ratios = np.asarray(ratio_list, dtype=float)
    times = np.asarray(time_list, dtype=float)
    return ratios, times


def _plot_histogram(
    *,
    datasets: Sequence[Tuple[Path, np.ndarray, np.ndarray]],
    bins: int,
    output_path: Optional[Path],
    show: bool,
) -> None:
    percentiles = np.linspace(0, 100, bins + 1)
    labels = [f"{percentiles[i]:.1f}-{percentiles[i + 1]:.1f}" for i in range(bins)]

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    alpha = 0.35

    plotted = 0
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    for idx, (json_path, ratios, times) in enumerate(datasets):
        avg_times = _compute_avg_times(ratios, times, percentiles)
        if avg_times is None:
            print(f"Insufficient ratio variation in {json_path}; skipping")
            continue
        color = color_cycle[idx % len(color_cycle)] if color_cycle else None
        ax.bar(
            labels,
            avg_times,
            color=color,
            alpha=alpha,
            edgecolor="black",
            linewidth=0.6,
            label=f"{json_path.stem} (n={len(ratios)})",
        )
        plotted += 1

    if plotted == 0:
        print("No datasets had sufficient variation to plot")
        plt.close(fig)
        return

    ax.set_xlabel("max_ratio percentile bin")
    ax.set_ylabel("Average time (ms)")
    ax.set_title("Average time by ratio percentile")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()

    output = _resolve_output_path(output_path, [path for path, _, _ in datasets])
    fig.savefig(output, dpi=200)
    print(f"Saved ratio percentile histogram to {output}")

    if show:
        plt.show()
    plt.close(fig)

def _compute_avg_times(
    ratios: np.ndarray,
    times: np.ndarray,
    percentiles: np.ndarray,
) -> Optional[np.ndarray]:
    ratio_breaks = np.percentile(ratios, percentiles)

    for idx in range(1, ratio_breaks.size):
        if ratio_breaks[idx] <= ratio_breaks[idx - 1]:
            ratio_breaks[idx] = np.nextafter(ratio_breaks[idx - 1], float("inf"))

    if (not np.isfinite(ratio_breaks).all()) or np.allclose(ratio_breaks[0], ratio_breaks[-1]):
        return None

    bins = percentiles.size - 1
    bin_indices = np.digitize(ratios, ratio_breaks[1:-1], right=True)
    bin_indices = np.clip(bin_indices, 0, bins - 1)

    avg_times = np.zeros(bins, dtype=float)
    counts = np.zeros(bins, dtype=float)

    for idx, time in zip(bin_indices, times):
        avg_times[idx] += time
        counts[idx] += 1

    mask = counts > 0
    avg_times[mask] /= counts[mask]
    avg_times[~mask] = 0.0
    return avg_times


def _resolve_output_path(candidate: Optional[Path], json_paths: Sequence[Path]) -> Path:
    if candidate is None:
        if len(json_paths) == 1:
            base = json_paths[0].parent
            stem = json_paths[0].stem
            return base / f"{stem}_ratio_percentile_time.png"
        base = Path(__file__).resolve().parent
        return base / "ratio_percentile_time_combined.png"
    candidate = candidate.expanduser().resolve()
    if candidate.is_dir():
        if len(json_paths) == 1:
            stem = json_paths[0].stem
            return candidate / f"{stem}_ratio_percentile_time.png"
        return candidate / "ratio_percentile_time_combined.png"
    return candidate


if __name__ == "__main__":
    main()
