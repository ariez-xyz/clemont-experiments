"""Plot histogram of quantitative monitor ratios from a saved JSON run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

try:
    from ._plot_utils import (
        DEFAULT_DPI,
        DEFAULT_FIGSIZE,
        metadata_value,
        resolve_json_paths,
    )
except ImportError:  # pragma: no cover - script-style execution
    from _plot_utils import (
        DEFAULT_DPI,
        DEFAULT_FIGSIZE,
        metadata_value,
        resolve_json_paths,
    )


DEFAULT_MIN_RATIO = 1e-5


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a histogram of max ratios from a quantitative monitor run",
    )
    parser.add_argument(
        "json_path",
        nargs="?",
        type=Path,
        help=(
            "Path to quant_run_*.json or a directory containing them (defaults to "
            "all quant_run_*.json alongside this script)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output image path (defaults to <json_path>_ratios.png)",
    )
    parser.add_argument("--bins", type=int, default=60, help="Number of histogram bins")
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=DEFAULT_MIN_RATIO,
        help="Drop ratios below this value before plotting. Default: 1e-5.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    json_paths = resolve_json_paths(args.json_path, default_dir=script_dir)

    shared_limits = _shared_axis_limits(json_paths, args.bins, args.min_ratio) if len(json_paths) > 1 else None

    for json_path in json_paths:
        _plot_histogram(json_path, args.output, args.bins, args.min_ratio, shared_limits)


def _load_positive_finite_ratios(json_path: Path, min_ratio: float) -> Optional[np.ndarray]:
    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    ratios = np.array([record["max_ratio"] for record in payload["records"]], dtype=float)
    finite_mask = np.isfinite(ratios)
    ratios = ratios[finite_mask & (ratios >= min_ratio)]
    if ratios.size == 0:
        return None
    return ratios


def _shared_axis_limits(
    json_paths: list[Path],
    bins: int,
    min_ratio: float,
) -> Optional[tuple[tuple[float, float], tuple[float, float], np.ndarray]]:
    ratios_by_path = [
        ratios
        for json_path in json_paths
        if (ratios := _load_positive_finite_ratios(json_path, min_ratio)) is not None
    ]
    if not ratios_by_path:
        return None

    global_min = min(float(ratios.min()) for ratios in ratios_by_path)
    global_max = max(float(ratios.max()) for ratios in ratios_by_path)
    if global_min == global_max:
        global_min /= 1.05
        global_max *= 1.05

    bin_edges = _log_bins(global_min, global_max, bins)
    max_count = max(int(np.histogram(ratios, bins=bin_edges)[0].max()) for ratios in ratios_by_path)
    y_top = max(1.0, max_count * 1.15)
    return (global_min, global_max), (0.8, y_top), bin_edges


def _plot_histogram(
    json_path: Path,
    output_path: Optional[Path],
    bins: int,
    min_ratio_cutoff: float,
    shared_limits: Optional[tuple[tuple[float, float], tuple[float, float], np.ndarray]] = None,
) -> None:
    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    ratios = _load_positive_finite_ratios(json_path, min_ratio_cutoff)
    if ratios is None:
        print(f"No finite ratios >= {min_ratio_cutoff:g} available to plot for {json_path}")
        return

    if shared_limits is None:
        xlim = None
        ylim = None
        min_ratio = float(ratios.min())
        max_ratio = float(ratios.max())
        if min_ratio == max_ratio:
            min_ratio /= 1.05
            max_ratio *= 1.05
        bin_edges = _log_bins(min_ratio, max_ratio, bins)
    else:
        xlim, ylim, bin_edges = shared_limits

    plt.figure(figsize=DEFAULT_FIGSIZE)
    plt.hist(ratios, bins=bin_edges, edgecolor="black", alpha=0.75)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Max ratio")
    plt.ylabel("Frequency")

    metadata = payload.get("metadata", {})
    total_time = metadata_value(metadata, "total_time")
    out_metric = metadata_value(metadata, "out_metric")
    exponent = metadata_value(metadata, "output_exponent", fallback_key="input_exponent")
    maxk = metadata_value(metadata, "max_k")

    title_bits = []
    if total_time is not None:
        try:
            title_bits.append(f"{round(float(total_time))}ms")
        except Exception:
            pass
    if maxk is not None:
        title_bits.append(f"max_k={maxk}")
    if out_metric is not None:
        title_bits.append(f"metric={out_metric}")
    if exponent is not None:
        title_bits.append(f"exponent={exponent}")

    title = "Max ratios histogram"
    if title_bits:
        title += f" ({', '.join(title_bits)})"
    
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    else:
        plt.ylim(top=10000)
#    plt.xlim(left=10e-5, right=10e1)

    final_output_path = output_path or json_path.with_name(json_path.stem + "_ratios.png")
    plt.savefig(final_output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    print(f"Saved ratio histogram to {final_output_path}")
    plt.close()


def _log_bins(min_ratio: float, max_ratio: float, bins: int) -> np.ndarray:
    """Return positive logarithmic bin edges for max-ratio histograms."""

    if min_ratio <= 0 or max_ratio <= 0:
        raise ValueError("log-scaled max-ratio histograms require positive ratios")
    edge_count = max(2, int(bins))
    return np.logspace(np.log10(min_ratio), np.log10(max_ratio), edge_count)


if __name__ == "__main__":
    main()
