#!/usr/bin/env python3
"""Plot max_ratio histograms split by a time-percentile threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

try:  # pragma: no cover - script entry point convenience
    from ._plot_utils import resolve_json_paths
except ImportError:  # pragma: no cover - script executed from repo root
    from _plot_utils import resolve_json_paths

DEFAULT_BINS = 60
DEFAULT_SPLIT = 80.0
EPS = 1e-9


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot two histograms of 'max_ratio' from a quantitative monitor JSON "
            "file, colouring records by whether their 'time' falls inside the "
            "specified percentile band."
        ),
    )
    parser.add_argument(
        "json_path",
        nargs="?",
        type=Path,
        help=(
            "Path to quant_run_*.json or a directory containing them. Defaults to "
            "files alongside this script when omitted."
        ),
    )
    parser.add_argument(
        "--split",
        type=float,
        default=DEFAULT_SPLIT,
        help="Percentile threshold (0-100) on 'time' used to split the records (default: 80)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_BINS,
        help=f"Histogram bin count (default: {DEFAULT_BINS})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory for output images (defaults to each JSON's directory)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively in addition to saving it",
    )
    args = parser.parse_args()

    if not 0.0 < args.split < 100.0:
        raise SystemExit("--split must be strictly between 0 and 100")
    if args.bins < 1:
        raise SystemExit("--bins must be at least 1")

    script_dir = Path(__file__).resolve().parent
    json_paths = resolve_json_paths(args.json_path, default_dir=script_dir)

    for json_path in json_paths:
        _plot_histogram(
            json_path=json_path,
            split=args.split,
            bins=args.bins,
            output_dir=args.output_dir,
            show=args.show,
        )


def _plot_histogram(
    *,
    json_path: Path,
    split: float,
    bins: int,
    output_dir: Optional[Path],
    show: bool,
) -> None:
    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    records = payload.get("records", [])
    if not records:
        print(f"No records found in {json_path}")
        return

    times = np.array([record.get("time") for record in records], dtype=float)
    ratios = np.array([record.get("max_ratio") for record in records], dtype=float)

    valid_mask = np.isfinite(times) & np.isfinite(ratios) & (ratios > 0)
    times = times[valid_mask]
    ratios = ratios[valid_mask]

    if ratios.size == 0:
        print(f"No positive finite 'max_ratio' values found in {json_path}")
        return

    cutoff = float(np.percentile(times, split))
    within_mask = times <= cutoff
    group_within = ratios[within_mask]
    group_outside = ratios[~within_mask]

    if group_within.size == 0 or group_outside.size == 0:
        print(
            f"Split at {split}th percentile ({cutoff:.6g}s) collapsed for {json_path}; "
            "adjust --split to produce two non-empty groups."
        )
        return

    bin_edges = _compute_ratio_bins(ratios, bins)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        group_within,
        bins=bin_edges,
        alpha=0.65,
        color="#1f77b4",
        edgecolor="#ffffff",
        label=(
            f"time ≤ {split:.2f}th pct ({group_within.size} records)"
        ),
    )
    ax.hist(
        group_outside,
        bins=bin_edges,
        alpha=0.65,
        color="#ff7f0e",
        edgecolor="#ffffff",
        label=(
            f"time > {split:.2f}th pct ({group_outside.size} records)"
        ),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("max_ratio")
    ax.set_ylabel("Count")
    ax.set_title(
        f"max_ratio split by time percentile\n"
        f"time cutoff = {cutoff:.6g}s (split {split:.2f}th pct)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(bottom=0.8)

    fig.tight_layout()

    output_path = _resolve_output_path(json_path, output_dir, split)
    fig.savefig(output_path, dpi=200)
    print(
        f"Saved time-coloured ratio histogram to {output_path} "
        f"(≤ cutoff: {group_within.size}, > cutoff: {group_outside.size})"
    )

    if show:
        plt.show()

    plt.close(fig)


def _compute_ratio_bins(values: np.ndarray, bins: int) -> np.ndarray:
    vmin = float(values.min())
    vmax = float(values.max())
    if vmin <= 0:
        vmin = EPS
    if np.isclose(vmax, vmin):
        vmax = vmin * (1.0 + EPS)
    return np.logspace(np.log10(vmin), np.log10(vmax), bins + 1)


def _resolve_output_path(
    json_path: Path,
    output_dir: Optional[Path],
    split: float,
) -> Path:
    if output_dir is not None:
        base = output_dir.expanduser().resolve()
        base.mkdir(parents=True, exist_ok=True)
    else:
        base = json_path.parent
    split_token = str(split).replace(".", "p")
    return base / f"{json_path.stem}_time_split_{split_token}.png"


if __name__ == "__main__":
    main()
