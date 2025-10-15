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
    from ._plot_utils import DEFAULT_DPI, DEFAULT_FIGSIZE, resolve_json_paths
except ImportError:  # pragma: no cover - script executed from repo root
    from _plot_utils import DEFAULT_DPI, DEFAULT_FIGSIZE, resolve_json_paths

DEFAULT_BINS = 20


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
        "--fill-percentiles",
        type=str,
        default="10,90",
        help="Comma-separated lower,upper percentiles for the shaded runtime band (default: 10,90)",
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
    parser.add_argument(
        "--no-title",
        action='store_true',
        dest='no_title'
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Comma-separated list of labels for legend.",
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

    fill_bounds = _parse_fill_percentiles(args.fill_percentiles)

    _plot_histogram(
        datasets=datasets,
        bins=args.bins,
        fill_bounds=fill_bounds,
        output_path=args.output,
        show=args.show,
        no_title=args.no_title,
        labels=args.labels,
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


def _parse_fill_percentiles(spec: str) -> Tuple[float, float]:
    parts = [part.strip() for part in spec.split(",") if part.strip()]
    if len(parts) != 2:
        raise SystemExit("--fill-percentiles must contain two comma-separated values, e.g. '10,90'")
    try:
        lower, upper = (float(parts[0]), float(parts[1]))
    except ValueError as exc:
        raise SystemExit("--fill-percentiles values must be numeric") from exc
    if not (0.0 <= lower < upper <= 100.0):
        raise SystemExit("--fill-percentiles must satisfy 0 ≤ lower < upper ≤ 100")
    return lower, upper


def _plot_histogram(
    *,
    datasets: Sequence[Tuple[Path, np.ndarray, np.ndarray]],
    bins: int,
    fill_bounds: Tuple[float, float],
    output_path: Optional[Path],
    show: bool,
    no_title: bool,
    labels: Optional[str],
) -> None:
    percentiles = np.linspace(0, 100, bins + 1)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    x_vals = (percentiles[:-1] + percentiles[1:]) / 2.0

    plotted = 0
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    for idx, (json_path, ratios, times) in enumerate(datasets):
        stats = _compute_stats(ratios, times, percentiles, fill_bounds)
        if stats is None:
            print(f"Insufficient ratio variation in {json_path}; skipping")
            continue
        avg_times, lower_bounds, upper_bounds = stats
        color = color_cycle[idx % len(color_cycle)] if color_cycle else None
        label = f"{json_path.stem} (n={len(ratios)})"
        if labels: label = labels.split(",")[idx]
        ax.plot(x_vals, avg_times, color=color, linewidth=2.0, label=label)
        mask = ~np.isnan(lower_bounds) & ~np.isnan(upper_bounds)
        if mask.any():
            ax.fill_between(
                x_vals,
                lower_bounds,
                upper_bounds,
                where=mask,
                color=color,
                alpha=0.2,
            )
        plotted += 1

    if plotted == 0:
        print("No datasets had sufficient variation to plot")
        plt.close(fig)
        return

    ax.set_xlabel("Robustness score percentile")
    ax.set_ylabel("Time (ms)")
    if not no_title: ax.set_title(
        "Average time by ratio percentile"
        + (f" (fill {fill_bounds[0]:.1f}–{fill_bounds[1]:.1f}%)" if datasets else "")
    )
    ax.set_xlim(percentiles[0], percentiles[-1])
    major_step = max(1, bins // 10)
    ax.set_xticks(percentiles[::major_step])
    ax.set_xticks(percentiles, minor=True)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()

    output = _resolve_output_path(output_path, [path for path, _, _ in datasets])
    fig.savefig(output, dpi=DEFAULT_DPI)
    print(f"Saved ratio percentile histogram to {output}")

    if show:
        plt.show()
    plt.close(fig)

def _compute_stats(
    ratios: np.ndarray,
    times: np.ndarray,
    percentiles: np.ndarray,
    fill_bounds: Tuple[float, float],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    lower_pct, upper_pct = fill_bounds
    ratio_breaks = np.percentile(ratios, percentiles)

    for idx in range(1, ratio_breaks.size):
        if ratio_breaks[idx] <= ratio_breaks[idx - 1]:
            ratio_breaks[idx] = np.nextafter(ratio_breaks[idx - 1], float("inf"))

    if (not np.isfinite(ratio_breaks).all()) or np.allclose(ratio_breaks[0], ratio_breaks[-1]):
        return None

    bins = percentiles.size - 1
    bin_indices = np.digitize(ratios, ratio_breaks[1:-1], right=True)
    bin_indices = np.clip(bin_indices, 0, bins - 1)

    bucket_values: list[list[float]] = [[] for _ in range(bins)]
    for idx, time in zip(bin_indices, times):
        bucket_values[idx].append(float(time))

    avg_times = np.full(bins, np.nan, dtype=float)
    lower_bounds = np.full(bins, np.nan, dtype=float)
    upper_bounds = np.full(bins, np.nan, dtype=float)

    for idx, values in enumerate(bucket_values):
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        avg_times[idx] = arr.mean()
        if arr.size == 1:
            lower_bounds[idx] = upper_bounds[idx] = arr[0]
        else:
            lower_bounds[idx], upper_bounds[idx] = np.percentile(
                arr, [lower_pct, upper_pct]
            )

    return avg_times, lower_bounds, upper_bounds
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
    sumsq_times = np.zeros(bins, dtype=float)
    counts = np.zeros(bins, dtype=float)

    for idx, time in zip(bin_indices, times):
        avg_times[idx] += time
        sumsq_times[idx] += time * time
        counts[idx] += 1

    mask = counts > 0
    avg_times[mask] /= counts[mask]
    avg_times[~mask] = 0.0
    std_times = np.zeros_like(avg_times)
    valid = counts > 1
    variance = np.zeros_like(avg_times)
    variance[valid] = (sumsq_times[valid] / counts[valid]) - avg_times[valid] ** 2
    variance = np.clip(variance, a_min=0.0, a_max=None)
    std_times[valid] = np.sqrt(variance[valid])
    return avg_times, std_times


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
