"""Plot histogram of quantitative monitor ratios from a saved JSON run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

try:
    from ._plot_utils import metadata_value, resolve_json_paths
except ImportError:  # pragma: no cover - script-style execution
    from _plot_utils import metadata_value, resolve_json_paths


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
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    json_paths = resolve_json_paths(args.json_path, default_dir=script_dir)

    for json_path in json_paths:
        _plot_histogram(json_path, args.output, args.bins)


def _plot_histogram(json_path: Path, output_path: Optional[Path], bins: int) -> None:
    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    ratios = np.array([record["max_ratio"] for record in payload["records"]], dtype=float)
    finite_mask = np.isfinite(ratios)
    ratios = ratios[finite_mask & (ratios > 0)]
    if ratios.size == 0:
        print(f"No positive finite ratios available to plot for {json_path}")
        return

    bin_edges = np.logspace(np.log10(ratios.min()), np.log10(ratios.max()), bins)

    plt.figure(figsize=(10, 6))
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
    plt.ylim(top=10000)
#    plt.xlim(left=10e-5, right=10e1)

    final_output_path = output_path or json_path.with_name(json_path.stem + "_ratios.png")
    plt.savefig(final_output_path, dpi=300, bbox_inches="tight")
    print(f"Saved ratio histogram to {final_output_path}")
    plt.close()


if __name__ == "__main__":
    main()
