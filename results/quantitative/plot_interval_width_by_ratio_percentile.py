#!/usr/bin/env python3
"""Boxplots of interval widths at a fixed time budget grouped by ratio percentiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:  # pragma: no cover - script entry convenience
    from ._plot_utils import DEFAULT_DPI, DEFAULT_FIGSIZE, resolve_json_paths
except ImportError:  # pragma: no cover - script executed from repo root
    from _plot_utils import DEFAULT_DPI, DEFAULT_FIGSIZE, resolve_json_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Group records by their max_ratio percentile and plot the distribution of "
            "interval widths achievable within a fixed time budget."
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
        "--time-budget",
        type=float,
        default=10.0,
        help="Maximum milliseconds allowed per record when sampling the interval width (default: 10.0)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of percentile bins to create between 0 and 100 (default: 10)",
    )
    parser.add_argument(
        "--min-bin-size",
        type=int,
        default=1,
        help="Only include bins with at least this many samples (default: 1)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the chart interactively in addition to saving it",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output image path (defaults to <json>_interval_width_by_ratio_percentile.png)",
    )
    args = parser.parse_args()

    if args.time_budget <= 0:
        raise SystemExit("--time-budget must be positive")
    if args.bins <= 0:
        raise SystemExit("--bins must be greater than zero")
    if args.min_bin_size < 1:
        raise SystemExit("--min-bin-size must be at least 1")

    script_dir = Path(__file__).resolve().parent
    target_paths = args.paths or [script_dir]

    json_paths: List[Path] = []
    for candidate in target_paths:
        json_paths.extend(resolve_json_paths(candidate, default_dir=script_dir))

    widths, ratios = _collect_widths(json_paths, time_budget=args.time_budget)

    if not widths:
        raise SystemExit("No interval widths available under the requested constraints")

    figure = _plot_boxplots(
        ratios=ratios,
        widths=widths,
        bins=args.bins,
        min_bin_size=args.min_bin_size,
        time_budget=args.time_budget,
        output_path=args.output,
        json_paths=json_paths,
    )

    if figure is not None and args.show:
        plt.show()


def _collect_widths(
    json_paths: Sequence[Path],
    *,
    time_budget: float,
) -> Tuple[List[float], List[float]]:
    all_widths: List[float] = []
    all_ratios: List[float] = []

    skipped_no_budget = 0

    for json_path in json_paths:
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        for record in payload.get("records", []):
            max_ratio = _safe_float(record.get("max_ratio"))
            ratio_prog = _safe_float_array(record.get("ratio_progression"))
            bound_prog = _safe_float_array(record.get("bound_progression"))
            ms_prog = _safe_float_array(record.get("ms_progression"))

            if (
                max_ratio is None
                or ratio_prog is None
                or bound_prog is None
                or ms_prog is None
            ):
                continue

            limit = min(len(ratio_prog), len(bound_prog), len(ms_prog))
            if limit == 0:
                continue

            ratio_prog = ratio_prog[:limit]
            bound_prog = bound_prog[:limit]
            ms_prog = ms_prog[:limit]

            idx = _last_index_within_budget(ms_prog, time_budget)
            if idx is None:
                skipped_no_budget += 1
                continue

            width = bound_prog[idx] - ratio_prog[idx]
            if not np.isfinite(width):
                continue
            if width < 0:
                width = 0.0

            all_widths.append(float(width))
            all_ratios.append(float(max_ratio))

    if skipped_no_budget:
        print(
            f"Skipped {skipped_no_budget} records with no ms_progression ≤ time budget",
        )

    return all_widths, all_ratios


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _safe_float_array(values: object) -> Optional[np.ndarray]:
    if not isinstance(values, Iterable):
        return None
    try:
        array = np.asarray(list(values), dtype=float)
    except (ValueError, TypeError):
        return None
    if array.size == 0:
        return None
    if not np.all(np.isfinite(array)):
        return None
    return array


def _last_index_within_budget(ms_prog: np.ndarray, budget_ms: float) -> Optional[int]:
    mask = np.where(ms_prog <= budget_ms)[0]
    if mask.size == 0:
        return None
    return int(mask[-1])


def _plot_boxplots(
    *,
    ratios: Sequence[float],
    widths: Sequence[float],
    bins: int,
    min_bin_size: int,
    time_budget: float,
    output_path: Optional[Path],
    json_paths: Sequence[Path],
) -> Optional[plt.Figure]:
    ratios_array = np.asarray(ratios, dtype=float)
    widths_array = np.asarray(widths, dtype=float)

    percentiles = np.linspace(0, 100, bins + 1)
    thresholds = np.percentile(ratios_array, percentiles)

    # Ensure strictly increasing thresholds when data has limited spread.
    thresholds = np.maximum.accumulate(thresholds)

    bin_indices = np.digitize(ratios_array, thresholds[1:-1], right=True)

    grouped: Dict[int, List[float]] = {}
    for idx, width in zip(bin_indices, widths_array):
        grouped.setdefault(int(idx), []).append(float(width))

    positions: List[int] = []
    data: List[List[float]] = []
    labels: List[str] = []

    for bin_idx in range(bins):
        values = grouped.get(bin_idx, [])
        if len(values) < min_bin_size:
            continue
        lower = percentiles[bin_idx]
        upper = percentiles[bin_idx + 1]
        labels.append(f"{lower:.0f}–{upper:.0f}%")
        positions.append(bin_idx)
        data.append(values)

    if not data:
        print("All bins filtered out by --min-bin-size; nothing to plot")
        return None

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    box = ax.boxplot(
        data,
        positions=positions,
        showfliers=False,
        patch_artist=True,
    )

    for patch in box["boxes"]:
        patch.set_alpha(0.6)
    for element in ("whiskers", "caps", "medians"):
        for artist in box[element]:
            artist.set_alpha(0.8)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Interval width (bound - ratio)")
    ax.set_xlabel("max_ratio percentile")
    ax.set_title(
        f"Interval width after {time_budget:.2f} ms budget\n"
        f"Aggregated from {len(json_paths)} file(s)"
    )
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()

    resolved_output = _resolve_output_path(output_path, json_paths)
    fig.savefig(resolved_output, dpi=DEFAULT_DPI, bbox_inches="tight")
    print(f"Saved percentile interval width boxplots to {resolved_output}")

    return fig


def _resolve_output_path(candidate: Optional[Path], json_paths: Sequence[Path]) -> Path:
    if candidate is None:
        if len(json_paths) == 1:
            return json_paths[0].with_name(
                json_paths[0].stem + "_interval_width_by_ratio_percentile.png"
            )
        script_dir = Path(__file__).resolve().parent
        return script_dir / "interval_width_by_ratio_percentile.png"
    candidate = candidate.expanduser()
    if candidate.is_dir():
        return candidate / "interval_width_by_ratio_percentile.png"
    return candidate


if __name__ == "__main__":
    main()
