"""Plot final k progression values from a quantitative monitor run."""

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
        description="Scatter plot of final k values from a quantitative monitor run",
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
        help="Optional output image path (defaults to <json_path>_k_scatter.png)",
    )
    parser.add_argument("--alpha", type=float, default=0.02, help="Scatter alpha value")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    json_paths = resolve_json_paths(args.json_path, default_dir=script_dir)

    for json_path in json_paths:
        _plot_k_scatter(json_path, args.output, args.alpha)


def _plot_k_scatter(json_path: Path, output_path: Optional[Path], alpha: float) -> None:
    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    records = payload["records"]
    final_ks = []
    for record in records:
        k_progression = record.get("k_progression", [])
        if k_progression:
            final_ks.append(k_progression[-1])

    if not final_ks:
        print(f"No k progression data available in the run for {json_path}")
        return

    indices = np.arange(len(final_ks))
    final_ks = np.array(final_ks, dtype=float)

    plt.figure(figsize=(10, 6))
    plt.scatter(indices, final_ks, s=4, alpha=alpha)
    plt.yscale("log")
    plt.xlabel("Record index")
    plt.ylabel("Final k value")

    metadata = payload.get("metadata", {})
    total_time = metadata_value(metadata, "total_time")
    out_metric = metadata_value(metadata, "out_metric")
    exponent = metadata_value(metadata, "output_exponent", fallback_key="input_exponent")
    max_k = metadata_value(metadata, "max_k")

    title_bits = []
    if total_time is not None:
        try:
            title_bits.append(f"{round(float(total_time))}ms")
        except Exception:
            pass
    if max_k is not None:
        title_bits.append(f"max_k={max_k}")
    if out_metric is not None:
        title_bits.append(f"metric={out_metric}")
    if exponent is not None:
        title_bits.append(f"exponent={exponent}")

    title = "Final k progression"
    if title_bits:
        title += f" ({', '.join(title_bits)})"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    final_output_path = output_path or json_path.with_name(json_path.stem + "_k_scatter.png")
    plt.savefig(final_output_path, dpi=300, bbox_inches="tight")
    print(f"Saved k progression scatter plot to {final_output_path}")
    plt.close()


if __name__ == "__main__":
    main()
