"""Plot histogram of quantitative monitor ratios from a saved JSON run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a histogram of max ratios from a quantitative monitor run",
    )
    parser.add_argument(
        "json_path",
        nargs="?",
        type=Path,
        help="Path to quant_run_*.json (defaults to all quant_run_*.json in script directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output image path (defaults to <json_path>_ratios.png)",
    )
    parser.add_argument("--bins", type=int, default=60, help="Number of histogram bins")
    args = parser.parse_args()

    json_paths = _resolve_json_paths(args.json_path)

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

    total_time = round(payload['metadata']['total_time'])
    out_metric = payload['metadata']['out_metric']
    try: # old name, misnamed
        exponent = payload["metadata"]["output_exponent"]
    except KeyError:
        exponent = payload["metadata"]["input_exponent"]
    maxk = payload['metadata']['max_k']
    
    title = f"Max ratios histogram ({total_time}ms, max_k={maxk}, metric={out_metric}, exponent={exponent})"
    
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.ylim(top=10000)
#    plt.xlim(left=10e-5, right=10e1)

    final_output_path = output_path or json_path.with_name(json_path.stem + "_ratios.png")
    plt.savefig(final_output_path, dpi=300, bbox_inches="tight")
    print(f"Saved ratio histogram to {final_output_path}")
    plt.close()


def _resolve_json_paths(candidate: Optional[Path]) -> list[Path]:
    if candidate:
        if not candidate.is_file():
            raise SystemExit(f"JSON file not found: {candidate}")
        return [candidate]

    script_dir = Path(__file__).resolve().parent
    json_files = sorted(script_dir.glob("quant_run_*.json"))
    if not json_files:
        raise SystemExit("No quant_run_*.json files found in script directory")
    return json_files


if __name__ == "__main__":
    main()
