"""Boxplots of bound minus ratio widths grouped by power-of-two k values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from ._plot_utils import metadata_value, resolve_json_paths
except ImportError:  # pragma: no cover - script-style execution
    from _plot_utils import metadata_value, resolve_json_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Draw boxplots of (bound - ratio) widths grouped by power-of-two k values"
        ),
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
        help=(
            "Optional output image path (defaults to <json>_interval_width_boxplots_by_k.png)"
        ),
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Only keep k bins with at least this many samples (default: 1).",
    )
    parser.add_argument(
        "--show-fliers",
        action="store_true",
        help="Display outlier fliers on the boxplots (default: hidden).",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=None,
        help=(
            "Restrict to records whose max_ratio is at or above the given percentile (0-100). "
            "Defaults to using all records."
        ),
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    json_paths = resolve_json_paths(args.json_path, default_dir=script_dir)

    for json_path in json_paths:
        _plot_interval_widths_by_k(
            json_path,
            args.output,
            min_count=args.min_count,
            show_fliers=args.show_fliers,
            percentile=args.percentile,
        )


def _plot_interval_widths_by_k(
    json_path: Path,
    output_path: Optional[Path],
    *,
    min_count: int,
    show_fliers: bool,
    percentile: Optional[float],
) -> None:
    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    records = payload.get("records", [])
    if not records:
        print(f"No records to plot for {json_path}")
        return

    record_summaries = []

    for record in records:
        k_prog = record.get("k_progression") or []
        ratio_prog = record.get("ratio_progression") or []
        bound_prog = record.get("bound_progression") or []

        limit = min(len(k_prog), len(ratio_prog), len(bound_prog))
        if limit == 0:
            continue

        max_ratio_val = None
        try:
            candidate = record.get("max_ratio")
            if candidate is not None:
                candidate = float(candidate)
                if np.isfinite(candidate):
                    max_ratio_val = candidate
        except (TypeError, ValueError):
            max_ratio_val = None

        per_record_widths: List[Tuple[int, float]] = []
        per_record_converged: Dict[int, int] = {}

        for idx in range(limit):
            k_raw = k_prog[idx]
            try:
                k_val = int(k_raw)
            except (TypeError, ValueError):
                continue
            if k_val <= 8:
                continue
            # Only retain exact powers of two for the requested x-axis.
            if k_val & (k_val - 1):
                continue
            if k_val == 1:
                continue

            ratio = ratio_prog[idx]
            bound = bound_prog[idx]
            try:
                ratio_val = float(ratio)
                bound_val = float(bound)
            except (TypeError, ValueError):
                continue
            width = bound_val - ratio_val
            if not np.isfinite(width):
                continue
            if width < 0:
                per_record_converged[k_val] = per_record_converged.get(k_val, 0) + 1
                continue  # point converged

            per_record_widths.append((k_val, width))

        if not per_record_widths and not per_record_converged:
            continue

        record_summaries.append(
            {
                "max_ratio": max_ratio_val,
                "widths": per_record_widths,
                "converged": per_record_converged,
            }
        )

    if not record_summaries:
        print(f"No finite interval widths grouped by powers of two for {json_path}")
        return

    if percentile is not None:
        if not (0 <= percentile <= 100):
            raise ValueError("--percentile must be between 0 and 100")
        max_ratios = [rec["max_ratio"] for rec in record_summaries if rec["max_ratio"] is not None]
        if not max_ratios:
            print(f"No eligible records with finite max_ratio for percentile filter ({json_path})")
            return
        cutoff = np.percentile(max_ratios, percentile)
        record_summaries = [
            rec
            for rec in record_summaries
            if rec["max_ratio"] is not None and rec["max_ratio"] >= cutoff
        ]
        if not record_summaries:
            print(
                f"All records filtered out by percentile={percentile} for {json_path}"
            )
            return

    widths_by_k: Dict[int, List[float]] = {}
    converged_by_k: Dict[int, int] = {}

    for summary in record_summaries:
        for k_val, width in summary["widths"]:
            widths_by_k.setdefault(k_val, []).append(width)
        for k_val, count in summary["converged"].items():
            converged_by_k[k_val] = converged_by_k.get(k_val, 0) + count

    if not widths_by_k:
        print(f"No finite interval widths grouped by powers of two for {json_path}")
        return

    sorted_k = sorted(k for k, vals in widths_by_k.items() if len(vals) >= min_count)
    if not sorted_k:
        print(f"All bins filtered out (min_count={min_count}) for {json_path}")
        return

    data = [widths_by_k[k] for k in sorted_k]
    positions = np.array(sorted_k, dtype=float)

    plt.figure(figsize=(12, 7))
    box = plt.boxplot(
        data,
        positions=positions,
        widths=[max(0.2, 0.3 * k) for k in positions],
        showfliers=show_fliers,
        patch_artist=True,
        manage_ticks=False,
    )

    for b in box["boxes"]:
        b.set_alpha(0.6)
    for whisker in box["whiskers"]:
        whisker.set_alpha(0.8)
    for cap in box["caps"]:
        cap.set_alpha(0.8)
    for median in box["medians"]:
        median.set_linewidth(2)

    ax = plt.gca()
    try:
        ax.set_xscale("log", base=2)
    except TypeError:  # Matplotlib < 3.3
        ax.set_xscale("log")
#    ax.set_yscale("log")
    ax.set_xticks(positions)
    labels = []
    for k in positions:
        k_int = int(k)
        converged_count = converged_by_k.get(k_int, 0)
        labels.append(f"{k_int}\n({converged_count})")
    ax.set_xticklabels(labels)
    ax.set_xlim(min(positions) / 1.5, max(positions) * 1.5)
    ax.set_xlabel("k (in parentheses: number of converged points)")
    ax.set_ylabel("Ratio-bound interval width")
    ax.grid(True, which="both", axis="both", alpha=0.25)

    metadata = payload.get("metadata", {})
    title_bits = []
    total_time = metadata_value(metadata, "total_time")
    out_metric = metadata_value(metadata, "out_metric")
    exponent = metadata_value(metadata, "output_exponent", fallback_key="input_exponent")
    max_k = metadata_value(metadata, "max_k")

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

    title = "Ratio-bound interval width in k"
    if title_bits:
        title += f" ({', '.join(title_bits)})"
    ax.set_title(title)

    plt.tight_layout()
    final_path = output_path or json_path.with_name(
        json_path.stem + "_interval_width_boxplots_by_k.png"
    )
    plt.savefig(final_path, dpi=300, bbox_inches="tight")
    print(f"Saved interval width boxplots to {final_path}")
    plt.close()


if __name__ == "__main__":
    main()
