"""Boxplots of max_ratio grouped by compared_count bins (powers of two),
with optional display of per-bin sample sizes.

Bin index j:
  j = 0 for (0,1], j = 1 for (1,2], j = 2 for (2,4], ...

Count display styles:
  - tick  : append [n=...] to tick or interval labels
  - text  : draw 'n=...' just beyond the right edge of the plot
  - width : scale box heights by sqrt(n)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

try:
    from ._plot_utils import metadata_value, resolve_json_paths
except ImportError:  # pragma: no cover - script-style execution
    from _plot_utils import metadata_value, resolve_json_paths

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw boxplots of max_ratio (log-x) binned by powers-of-two compared_count."
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
        help="Optional output image path (defaults to <json_path>_ratio_boxplots_by_pow2_count.png)",
    )
    parser.add_argument(
        "--min-bin-size",
        type=int,
        default=1,
        help="Only include bins with at least this many points (default: 1).",
    )
    parser.add_argument(
        "--show-fliers",
        action="store_true",
        help="Show outlier fliers on the boxplots (default: hidden).",
    )
    parser.add_argument(
        "--right-bin-labels",
        action="store_true",
        help="Add a secondary right y-axis with human-readable bin intervals.",
    )
    parser.add_argument(
        "--count-style",
        choices=["tick", "text", "width", "none"],
        default="tick",
        help="How to indicate per-bin sample size: in tick labels, as text, by box height, or not at all.",
    )
    parser.add_argument(
        "--count-alpha",
        type=float,
        default=0.9,
        help="Alpha for count text (when --count-style=text).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    json_paths = resolve_json_paths(args.json_path, default_dir=script_dir)
    for json_path in json_paths:
        _plot_ratio_boxplots_by_pow2_count(
            json_path,
            args.output,
            min_bin_size=args.min_bin_size,
            show_fliers=args.show_fliers,
            right_bin_labels=args.right_bin_labels,
            count_style=args.count_style,
            count_alpha=args.count_alpha,
        )


def _plot_ratio_boxplots_by_pow2_count(
    json_path: Path,
    output_path: Optional[Path],
    *,
    min_bin_size: int,
    show_fliers: bool,
    right_bin_labels: bool,
    count_style: str,
    count_alpha: float,
) -> None:
    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    records = payload.get("records", [])
    if not records:
        print(f"No records to plot for {json_path}")
        return

    # Collect valid pairs
    pairs = []
    for r in records:
        x = r.get("max_ratio")
        c = r.get("compared_count")
        if x is None or c is None:
            continue
        try:
            x = float(x)
            c = float(c)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(x) and x > 0 and np.isfinite(c) and c > 0):
            continue
        pairs.append((x, c))

    if not pairs:
        print(f"No finite positive (ratio, count) pairs found for {json_path}")
        return

    # Bin by powers of two: j=0 for (0,1], j=1 for (1,2], j=2 for (2,4], ...
    bin_to_vals: Dict[int, List[float]] = {}
    for x, c in pairs:
        j = 0 if c <= 1 else int(np.ceil(np.log2(c)))
        bin_to_vals.setdefault(j, []).append(x)

    bins = sorted(j for j, vals in bin_to_vals.items() if len(vals) >= min_bin_size)
    if not bins:
        print(f"All bins filtered out (min_bin_size={min_bin_size}) for {json_path}")
        return

    data = [bin_to_vals[j] for j in bins]
    counts = [len(v) for v in data]
    positions = [j for j in bins]  # 0 for (0,1], 1 for (1,2], 2 for (2,4], ...

    # Compute widths if requested (horizontal boxplots: width == box height)
    widths = None
    if count_style == "width":
        max_n = max(counts)
        # Keep boxes within their 1.0-spaced lanes; aim for [0.3, 0.8] using sqrt scaling.
        widths = [
            0.3 + 0.5 * (np.sqrt(n) / np.sqrt(max_n)) if max_n > 0 else 0.3
            for n in counts
        ]

    # Plot
    plt.figure(figsize=(12, 7))
    bp = plt.boxplot(
        data,
        positions=positions,
        vert=False,
        showfliers=show_fliers,
        flierprops={'alpha': 0.1},
        patch_artist=True,
        manage_ticks=False,
        whis=1.5,
        widths=widths,
    )

    # Styling
    for box in bp["boxes"]:
        box.set_alpha(0.6)
    for whisker in bp["whiskers"]:
        whisker.set_alpha(0.8)
    for cap in bp["caps"]:
        cap.set_alpha(0.8)
    for median in bp["medians"]:
        median.set_linewidth(2)

    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_xlabel("Lipschitz ratio")
    ax.set_ylabel("Number of neighbors considered")
    ax.set_yticks(positions)
    ax.set_ylim(min(positions) - 0.5, max(positions) + 0.5)
    ax.grid(True, which="both", axis="both", alpha=0.25)

    # Left tick labels (indices or indices + counts)
    left_labels = [str(2**j) for j in bins]
    if count_style == "tick" and not right_bin_labels:
        left_labels = [f"{l} {f'[n={n}]':<6}" for l, n in zip(left_labels, counts)]
    ax.set_yticklabels(left_labels)

    # Optional right-side interval labels (optionally with counts)
    if right_bin_labels:
        ax_r = ax.twinx()
        ax_r.set_ylim(ax.get_ylim())
        ax_r.set_yticks(positions)
        interval_labels = [_interval_label(j) for j in bins]
        if count_style == "tick":
            interval_labels = [f"{lab}  [n={n}]" for lab, n in zip(interval_labels, counts)]
        ax_r.set_yticklabels(interval_labels)
        ax_r.set_ylabel("compared_count bin")

    # Optional free-floating count texts along the right margin
    if count_style == "text":
        for y, n in zip(positions, counts):
            # x=1.005 in axes coords (just outside right edge), y in data coords
            ax.text(
                1.005, y, f"n={n}",
                transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=9, alpha=count_alpha
            )

    # Title with metadata
    meta = payload.get("metadata", {})
    title_parts = ["Max ratio by compared_count bins (boxplots)"]
    meta_summary = []

    total_time = metadata_value(meta, "total_time")
    max_k = metadata_value(meta, "max_k")
    out_metric = metadata_value(meta, "out_metric")
    exponent = metadata_value(meta, "output_exponent", fallback_key="input_exponent")

    if total_time is not None:
        try:
            meta_summary.append(f"{round(float(total_time))}ms")
        except Exception:
            pass
    if max_k is not None:
        meta_summary.append(f"max_k={max_k}")
    if out_metric is not None:
        meta_summary.append(f"metric={out_metric}")
    if exponent is not None:
        meta_summary.append(f"exponent={exponent}")
    if meta_summary:
        title_parts.append(f"({', '.join(meta_summary)})")
    ax.set_title(" ".join(title_parts))

    final_output_path = output_path or json_path.with_name(
        json_path.stem + "_ratio_boxplots_by_pow2_count.png"
    )
    plt.tight_layout()
    plt.savefig(final_output_path, dpi=300, bbox_inches="tight")
    print(f"Saved boxplots to {final_output_path}")
    plt.close()


def _interval_label(j: int) -> str:
    if j == 0:
        return "(0, 1]"
    lo = int(2 ** (j - 1))
    hi = int(2 ** j)
    return f"({lo}, {hi}]"


if __name__ == "__main__":
    main()
