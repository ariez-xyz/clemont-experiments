#!/usr/bin/env python3
"""Plot time-per-sample progressions across batch/max-k and epsilon settings."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt

BATCH_EPS_DIR = "eps_0_05"
DEFAULT_OUTPUT_NAME = "time_per_sample.png"
JSON_PATTERN = "*.json"


@dataclass
class Series:
    label: str
    x: List[int]
    y: List[float]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a time-per-sample plot for RobustTrees Higgs quantitative runs. "
            "For the batch/max-k lines the script reads all runs under eps_0_05. "
            "For the epsilon lines it reads maxk_128/batch_100000 for every epsilon."
        )
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "robusttrees_higgs" / "results",
        help="Directory containing eps_* subdirectories (default: robusttrees_higgs/results alongside this script)",
    )
    parser.add_argument(
        "--walltime",
        required=True,
        help="Walltime identifier such as 03-00-00 or 03:00:00",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output image path (default: <results_dir>/time_per_sample_<walltime>.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively in addition to saving it",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.expanduser().resolve()
    if not results_dir.is_dir():
        raise SystemExit(f"Results directory not found: {results_dir}")

    walltime_key = args.walltime.strip()
    if not walltime_key:
        raise SystemExit("Walltime must be a non-empty string")
    walltime_key = walltime_key.replace(":", "-")

    batch_series = list(_collect_batch_series(results_dir, walltime_key))
    epsilon_series = list(_collect_epsilon_series(results_dir, walltime_key))

    if not batch_series and not epsilon_series:
        raise SystemExit(
            "No matching JSON payloads found for the requested walltime."
        )

    fig, ax = plt.subplots(figsize=(10, 6))

    for series in batch_series:
        ax.plot(series.x, series.y, label=series.label, linewidth=2.0)
    for series in epsilon_series:
        ax.plot(
            series.x,
            series.y,
            label=series.label,
            linewidth=2.0,
            linestyle="--",
        )

    ax.set_xlabel("Processed samples")
    ax.set_ylabel("Time per sample (seconds)")
    ax.set_title(f"Time per sample progression (walltime {walltime_key})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    output_path = _resolve_output_path(args.output, results_dir, walltime_key)
    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")

    if args.show:
        plt.show()


def _collect_batch_series(results_dir: Path, walltime_key: str) -> Iterable[Series]:
    eps_dir = results_dir / BATCH_EPS_DIR
    if not eps_dir.is_dir():
        print(f"Skipping batch/max-k lines: missing {eps_dir}")
        return []

    series_list: List[Series] = []

    for maxk_dir in sorted(p for p in eps_dir.iterdir() if p.is_dir() and p.name.startswith("maxk_")):
        maxk_label = maxk_dir.name.split("_", 1)[1]
        target_dirs = sorted(
            p for p in maxk_dir.iterdir() if p.is_dir() and p.name.endswith(f"wt_{walltime_key}")
        )
        if not target_dirs:
            print(f"No batch directories with walltime {walltime_key} found under {maxk_dir}")
            continue

        for batch_dir in target_dirs:
            batchsize = _parse_batchsize(batch_dir.name)
            payload = _load_latest_payload(batch_dir)
            if payload is None:
                continue
            series = _build_series(
                payload.get("records", []),
                value_key="time",
                label=f"batch {batchsize}, maxk {maxk_label}",
                value_scale=1.0,
            )
            if series:
                series_list.append(series)

    return series_list


def _collect_epsilon_series(results_dir: Path, walltime_key: str) -> Iterable[Series]:
    series_list: List[Series] = []

    for eps_dir in sorted(
        p for p in results_dir.iterdir() if p.is_dir() and p.name.startswith("eps_")
    ):
        epsilon_value = eps_dir.name.split("eps_", 1)[1].replace("_", ".")
        target_dir = (
            eps_dir
            / "maxk_128"
            / f"batch_100000_wt_{walltime_key}"
        )
        if not target_dir.is_dir():
            print(f"Missing epsilon series directory: {target_dir}")
            continue

        payload = _load_latest_payload(target_dir)
        if payload is None:
            continue
        series = _build_series(
            payload.get("records", []),
            value_key="epsilon_monitor_time_ms",
            label=f"epsilon {epsilon_value} (monitor)",
            value_scale=0.001,
        )
        if series:
            series_list.append(series)

    return series_list


def _build_series(
    records: Sequence[dict],
    *,
    value_key: str,
    label: str,
    value_scale: float,
) -> Series | None:
    ordered: List[Tuple[float, float]] = []
    for idx, record in enumerate(records):
        raw_value = record.get(value_key)
        if raw_value is None:
            continue
        try:
            value = float(raw_value) * value_scale
        except (TypeError, ValueError):
            continue
        order_key = _extract_order(record, fallback=idx)
        ordered.append((order_key, value))

    if not ordered:
        print(f"No usable '{value_key}' values for series '{label}'")
        return None

    ordered.sort(key=lambda item: item[0])
    y_values = [value for _, value in ordered]
    x_values = list(range(1, len(y_values) + 1))
    return Series(label=label, x=x_values, y=y_values)


def _extract_order(record: dict, fallback: int) -> float:
    for key in ("index", "point_id"):
        candidate = record.get(key)
        if candidate is None:
            continue
        try:
            return float(candidate)
        except (TypeError, ValueError):
            continue
    return float(fallback)


def _parse_batchsize(name: str) -> str:
    parts = name.split("_")
    if len(parts) >= 2 and parts[0] == "batch":
        return parts[1]
    return name


def _resolve_output_path(output: Path | None, results_dir: Path, walltime_key: str) -> Path:
    if output:
        return output.expanduser().resolve()
    default_name = f"time_per_sample_{walltime_key}.png"
    return results_dir / default_name


def _load_latest_payload(run_dir: Path) -> dict | None:
    json_files = sorted(run_dir.glob(JSON_PATTERN))
    if not json_files:
        print(f"No JSON files found in {run_dir}")
        return None

    chosen = _select_latest_json(json_files)
    others = [p for p in json_files if p != chosen]
    if others:
        other_names = ", ".join(p.name for p in others)
        print(
            f"Multiple JSON files in {run_dir}. Using {chosen.name}; ignoring {other_names}."
        )

    try:
        with chosen.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:  # pragma: no cover - defensive logging only
        print(f"Failed to load {chosen}: {exc}")
        return None


def _select_latest_json(json_files: Sequence[Path]) -> Path:
    def sort_key(path: Path) -> Tuple[str, float]:
        stem = path.stem
        timestamp = ""
        if stem.startswith("quant_run_"):
            timestamp = stem[10:]
        return timestamp, path.stat().st_mtime

    return max(json_files, key=sort_key)


if __name__ == "__main__":
    main()
