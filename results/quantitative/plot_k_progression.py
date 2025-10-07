"""Plot final k progression values from a quantitative monitor run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scatter plot of final k values from a quantitative monitor run",
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
        help="Optional output image path (defaults to <json_path>_k_scatter.png)",
    )
    parser.add_argument("--alpha", type=float, default=0.02, help="Scatter alpha value")
    args = parser.parse_args()

    json_paths = _resolve_json_paths(args.json_path)

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

    total_time = round(payload["metadata"]["total_time"])
    out_metric = payload["metadata"]["out_metric"]
    try: # old name, misnamed
        exponent = payload["metadata"]["output_exponent"]
    except KeyError:
        exponent = payload["metadata"]["input_exponent"]
    max_k = payload["metadata"]["max_k"]

    title = (
        "Final k progression "
        f"({total_time}ms, max_k={max_k}, metric={out_metric}, exponent={exponent})"
    )
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    final_output_path = output_path or json_path.with_name(json_path.stem + "_k_scatter.png")
    plt.savefig(final_output_path, dpi=300, bbox_inches="tight")
    print(f"Saved k progression scatter plot to {final_output_path}")
    plt.close()


def _resolve_json_paths(candidate: Optional[Path]) -> list[Path]:
    if candidate:
        if not candidate.is_file():
            raise SystemExit(f"JSON file not found: {candidate}")
        return [candidate]

    experiments_dir = Path(__file__).resolve().parent
    json_files = sorted(experiments_dir.glob("quant_run_*.json"))
    if not json_files:
        raise SystemExit("No quant_run_*.json files found in experiments/")
    return json_files


if __name__ == "__main__":
    main()
