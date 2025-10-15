#!/usr/bin/env python3
"""Scatter max_ratio vs time with cumulative time overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:  # pragma: no cover - script entry point convenience
    from ._plot_utils import DEFAULT_DPI, DEFAULT_FIGSIZE, resolve_json_paths
except ImportError:  # pragma: no cover - script executed from repo root
    from _plot_utils import DEFAULT_DPI, DEFAULT_FIGSIZE, resolve_json_paths
CUMTIME_COLOR = "#ff7f0e"
SCATTER_COLOR = "#1f77b4"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot max_ratio against per-record time for quantitative runs, "
            "with a cumulative time curve on a secondary axis."
        )
    )
    parser.add_argument(
        "json_path",
        nargs="?",
        type=Path,
        help=(
            "Path to quant_run_*.json or directory containing them. Defaults to "
            "all quant_run_*.json alongside this script when omitted."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output image path (defaults to <json>.cumtime.png)",
    )
    parser.add_argument(
        "--top",
        type=int,
        help="Plot only the first N records after sorting by time (optional)",
    )
    parser.add_argument(
        "--sort",
        choices=("time", "ratio"),
        default="time",
        help="Order records before plotting (default: time)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively as well as saving it",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    json_paths = resolve_json_paths(args.json_path, default_dir=script_dir)

    for json_path in json_paths:
        _plot_single_run(
            json_path=json_path,
            output_path=args.output,
            sort_key=args.sort,
            top=args.top,
            show=args.show,
        )


def _plot_single_run(
    *,
    json_path: Path,
    output_path: Optional[Path],
    sort_key: str,
    top: Optional[int],
    show: bool,
) -> None:
    records = _load_records(json_path)
    if not records:
        print(f"No finite ratio/time records in {json_path}")
        return

    values = _prepare_values(records, sort_key, top)
    x_times, ratios, cum_times = values

    fig, ax_ratio = plt.subplots(figsize=DEFAULT_FIGSIZE)

    scatter = ax_ratio.scatter(
        x_times,
        ratios,
        s=18,
        alpha=0.25,
        color=SCATTER_COLOR,
        label="record max_ratio",
    )
    ax_ratio.set_xlabel("Record time (ms)")
    ax_ratio.set_ylabel("max_ratio")
    ax_ratio.set_title(f"max_ratio vs time with cumulative time\n{json_path.name}")
    ax_ratio.grid(True, alpha=0.3)

    ax_cum = ax_ratio.twinx()
    line, = ax_cum.plot(
        x_times,
        cum_times,
        color=CUMTIME_COLOR,
        linewidth=2.0,
        label="cumulative time",
    )
    ax_cum.set_ylabel("Cumulative time (ms)")
    ax_ratio.legend(handles=[scatter, line], loc="upper left")

    fig.tight_layout()

    resolved_output = _resolve_output_path(output_path, json_path)
    fig.savefig(resolved_output, dpi=DEFAULT_DPI)
    print(
        f"Saved ratio/time scatter to {resolved_output} "
        f"(records plotted: {len(x_times)})"
    )

    if show:
        plt.show()
    plt.close(fig)


def _load_records(json_path: Path) -> List[dict]:
    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    records = []
    for record in payload.get("records", []):
        time = record.get("time")
        ratio = record.get("max_ratio")
        if time is None or ratio is None:
            continue
        try:
            time = float(time)
            ratio = float(ratio)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(time) or not np.isfinite(ratio):
            continue
        records.append({"time": time, "ratio": ratio})
    return records


def _prepare_values(
    records: Sequence[dict],
    sort_key: str,
    top: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = "time" if sort_key == "time" else "ratio"
    sorted_records = sorted(records, key=lambda item: item[key])

    if top is not None and top > 0:
        sorted_records = sorted_records[:top]

    times = np.array([item["time"] for item in sorted_records], dtype=float)
    ratios = np.array([item["ratio"] for item in sorted_records], dtype=float)
    cum_times = np.cumsum(times)

    return times, ratios, cum_times


def _resolve_output_path(candidate: Optional[Path], json_path: Path) -> Path:
    if candidate is None:
        return json_path.parent / f"{json_path.stem}_ratio_time.png"
    candidate = candidate.expanduser().resolve()
    if candidate.is_dir():
        return candidate / f"{json_path.stem}_ratio_time.png"
    return candidate


if __name__ == "__main__":
    main()
