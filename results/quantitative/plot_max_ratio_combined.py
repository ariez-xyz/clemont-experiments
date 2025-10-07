"""Combined histogram of quantitative monitor ratios across multiple runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np

try:
    from ._plot_utils import resolve_json_paths
except ImportError:  # pragma: no cover - script-style execution
    from _plot_utils import resolve_json_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot max-ratio histograms for one or more quantitative monitor runs "
            "on a shared axis."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=(
            "Optional list of quant_run_*.json files or directories containing them. "
            "If omitted, all quant_run_*.json alongside this script are used."
        ),
    )
    parser.add_argument("--bins", type=int, default=60, help="Number of logarithmic bins to use")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Opacity for each histogram overlay (default: 0.45)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output image path (default: combined histogram next to this script)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    json_paths = _collect_json_paths(args.paths, default_dir=script_dir)
    if len(json_paths) < 2:
        print("Warning: fewer than two runs provided; plotting available data anyway.")

    datasets = []
    for json_path in json_paths:
        ratios = _load_ratios(json_path)
        if ratios.size == 0:
            print(f"Skipping {json_path} (no positive finite ratios)")
            continue
        datasets.append((json_path, ratios))

    if not datasets:
        raise SystemExit("No usable ratio data found in the supplied runs")

    all_ratios = np.concatenate([ratios for _, ratios in datasets])
    min_ratio = float(np.min(all_ratios))
    max_ratio = float(np.max(all_ratios))
    if min_ratio <= 0 or not np.isfinite(min_ratio):
        min_ratio = float(np.min(all_ratios[all_ratios > 0]))
    if min_ratio <= 0:
        raise SystemExit("Combined ratios contain no positive values")

    if max_ratio == min_ratio:
        min_edge = min_ratio / 1.5 if min_ratio > 0 else 1e-6
        max_edge = max_ratio * 1.5 if max_ratio > 0 else 1e-6
    else:
        min_edge = min_ratio
        max_edge = max_ratio

    bins = np.logspace(np.log10(min_edge), np.log10(max_edge), args.bins)

    plt.figure(figsize=(10, 6))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    for idx, (json_path, ratios) in enumerate(datasets):
        color = color_cycle[idx % len(color_cycle)] if color_cycle else None
        plt.hist(
            ratios,
            bins=bins,
            alpha=args.alpha,
            label=f"{json_path.stem} (n={len(ratios)})",
            edgecolor="black",
            linewidth=0.6,
            color=color,
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Max ratio")
    plt.ylabel("Frequency")
    plt.title("Max ratios histogram (combined)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = _resolve_output_path(args.output, script_dir)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved combined ratio histogram to {output_path}")
    plt.close()


def _collect_json_paths(paths: Iterable[Path], *, default_dir: Path) -> List[Path]:
    if not paths:
        return resolve_json_paths(None, default_dir=default_dir)

    resolved: List[Path] = []
    seen = set()
    for raw_path in paths:
        for candidate in resolve_json_paths(raw_path, default_dir=default_dir):
            if candidate not in seen:
                seen.add(candidate)
                resolved.append(candidate)
    if not resolved:
        raise SystemExit("No quant_run_*.json files resolved from provided paths")
    return resolved


def _load_ratios(json_path: Path) -> np.ndarray:
    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    ratios = np.array([record.get("max_ratio") for record in payload.get("records", [])], dtype=float)
    mask = np.isfinite(ratios) & (ratios > 0)
    return ratios[mask]


def _resolve_output_path(candidate: Optional[Path], script_dir: Path) -> Path:
    if candidate is None:
        return script_dir / "quant_run_combined_ratios.png"
    candidate = candidate.expanduser()
    if candidate.is_dir():
        return candidate / "quant_run_combined_ratios.png"
    parent = candidate.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    return candidate


if __name__ == "__main__":
    main()
