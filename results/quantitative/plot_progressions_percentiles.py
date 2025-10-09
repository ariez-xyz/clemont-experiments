"""Wrapper to plot specific percentile points from quantitative progression data."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np

try:
    from ._plot_utils import resolve_json_paths
except ImportError:  # pragma: no cover - script-style execution
    from _plot_utils import resolve_json_paths

PERCENTILES = (10, 50, 90)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select point IDs at the 10th, 50th, and 90th compared_count percentiles "
            "and forward them to plot_progressions.py via --point-ids."
        )
    )
    parser.add_argument(
        "json_path",
        nargs="?",
        type=Path,
        help=(
            "Path to a quant_run_*.json file or directory containing them (defaults to "
            "the directory of this script)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional output directory forwarded to plot_progressions.py",
    )

    args, extra_args = parser.parse_known_args()

    script_dir = Path(__file__).resolve().parent
    json_paths = resolve_json_paths(args.json_path, default_dir=script_dir)

    for json_path in json_paths:
        point_ids = _select_percentile_point_ids(json_path)
        if not point_ids:
            print(f"No eligible point IDs found for {json_path}, skipping")
            continue

        cmd: List[str] = [
            sys.executable,
            str(script_dir / "plot_progressions.py"),
            str(json_path),
            "--point-ids",
            ",".join(str(pid) for pid in point_ids),
        ]

        if args.output_dir is not None:
            cmd.extend(["--output-dir", str(args.output_dir)])

        cmd.extend(extra_args)
        cmd.extend(["--alpha", str(0.3)])

        print(
            f"Running plot_progressions.py for {json_path.name} "
            f"with point_ids {point_ids}"
        )
        subprocess.run(cmd, check=True)


def _select_percentile_point_ids(json_path: Path) -> List[int]:
    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    records = payload.get("records", [])
    usable = []
    for record in records:
        point_id = record.get("point_id")
        compared = record.get("compared_count")
        try:
            pid = int(point_id)
            count = float(compared)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(count):
            continue
        usable.append((pid, count))

    if not usable:
        return []

    counts = np.array([count for _, count in usable], dtype=float)

    remaining_indices = list(range(len(usable)))
    selected_ids: List[int] = []

    for percentile in PERCENTILES:
        if not remaining_indices:
            break
        target = np.percentile(counts, percentile)
        best_index = min(
            remaining_indices,
            key=lambda idx: (
                abs(usable[idx][1] - target),
                usable[idx][1],
                usable[idx][0],
            ),
        )
        selected_ids.append(usable[best_index][0])
        remaining_indices.remove(best_index)

    # Ensure ids reflect the original percentile ordering and are unique.
    deduped: List[int] = []
    seen = set()
    for pid in selected_ids:
        if pid in seen:
            continue
        seen.add(pid)
        deduped.append(pid)

    return deduped


if __name__ == "__main__":
    main()
