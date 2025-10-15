#!/usr/bin/env python3
"""Plot time-per-sample progressions across batch/max-k and epsilon settings."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt

try:  # pragma: no cover - script entry point convenience
    from ._plot_utils import DEFAULT_DPI, DEFAULT_FIGSIZE
except ImportError:  # pragma: no cover - script executed from repo root
    from _plot_utils import DEFAULT_DPI, DEFAULT_FIGSIZE

BATCH_EPS_DIR = "eps_0_05"
DEFAULT_OUTPUT_NAME = "time_per_sample.png"
JSON_PATTERN = "*.json"
ROLLING_SAMPLE_LIMIT = 1000


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
        help="Optional output image path (default: <results_dir>/../time_per_sample_<walltime>.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively in addition to saving it",
    )
    parser.add_argument(
        "--rolling-average",
        type=int,
        help=(
            "Window size for rolling-average smoothing. When set, the script also "
            "down-samples each line to at most 1000 points for plotting efficiency."
        ),
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        help="Filter runs to this batchsize (applies to both batch/max-k and epsilon series).",
    )
    parser.add_argument(
        "--epsilon",
        type=str,
        help="Filter runs to this epsilon.",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.expanduser().resolve()
    if not results_dir.is_dir():
        raise SystemExit(f"Results directory not found: {results_dir}")

    walltime_key = args.walltime.strip()
    if not walltime_key:
        raise SystemExit("Walltime must be a non-empty string")
    walltime_key = walltime_key.replace(":", "-")

    batchsize_filter = str(args.batchsize) if args.batchsize else None
    epsilon_filter = str(args.epsilon) if args.epsilon else None

    batch_series = list(
        _collect_batch_series(
            results_dir,
            walltime_key,
            rolling_window=args.rolling_average,
            batchsize_filter=batchsize_filter,
        )
    )
    epsilon_series = list(
        _collect_epsilon_series(
            results_dir,
            walltime_key,
            rolling_window=args.rolling_average,
            batchsize_filter=batchsize_filter,
            epsilon_filter=epsilon_filter,
        )
    )

    if not batch_series and not epsilon_series:
        raise SystemExit(
            "No matching JSON payloads found for the requested walltime."
        )

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    line_alpha = 0.5

    for series in batch_series:
        ax.plot(series.x, series.y, label=series.label, linewidth=2.0, alpha=line_alpha)
    for series in epsilon_series:
        ax.plot(
            series.x,
            series.y,
            label=series.label,
            linewidth=2.0,
            linestyle="--",
            alpha=line_alpha,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Processed samples")
    ax.set_ylabel("Time per sample (ms)")
    ax.set_title(f"Time per sample progression (walltime {walltime_key})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    output_path = _resolve_output_path(args.output, results_dir, walltime_key)
    fig.savefig(output_path, dpi=DEFAULT_DPI)
    print(f"Saved plot to {output_path}")

    if args.show:
        plt.show()


def _collect_batch_series(
    results_dir: Path,
    walltime_key: str,
    *,
    rolling_window: int | None,
    batchsize_filter: str | None,
) -> Iterable[Series]:
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
            if batchsize == "10000":
                continue
            if batchsize_filter and batchsize != batchsize_filter:
                continue
            payload = _load_latest_payload(batch_dir)
            if payload is None:
                continue
            label = f"quantitative, batch {batchsize}, maxk {maxk_label}"
            if batchsize_filter: label = f"quantitative, max_k={maxk_label}"
            series = _build_series(
                payload.get("records", []),
                value_key="time",
                label=label,
                value_scale=1.0,
                rolling_window=rolling_window,
            )
            if series:
                series_list.append(series)

    return series_list


def _collect_epsilon_series(
    results_dir: Path,
    walltime_key: str,
    *,
    rolling_window: int | None,
    batchsize_filter: str | None,
    epsilon_filter: str | None,
) -> Iterable[Series]:
    if batchsize_filter and batchsize_filter != "100000":
        print(
            "Skipping epsilon series: --batchsize does not match required batch_100000 directories"
        )
        return []

    series_list: List[Series] = []

    for eps_dir in sorted(
        p for p in results_dir.iterdir() if p.is_dir() and p.name.startswith("eps_")
    ):
        epsilon_value = eps_dir.name.split("eps_", 1)[1].replace("_", ".")
        if epsilon_filter and epsilon_value != epsilon_filter:
            continue
        target_dir = (
            eps_dir
            / "maxk_1024"
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
            label=f"$\\epsilon-\\delta$ monitor, $\\epsilon={epsilon_value}$",
            value_scale=1.0,
            rolling_window=rolling_window,
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
    rolling_window: int | None,
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
    x_values, y_values = _postprocess_series(x_values, y_values, rolling_window)
    return Series(label=label, x=x_values, y=y_values)


def _postprocess_series(
    x_values: List[int],
    y_values: List[float],
    rolling_window: int | None,
) -> tuple[List[int], List[float]]:
    if rolling_window is None or rolling_window <= 0 or len(y_values) <= 1:
        return x_values, y_values

    y_values = _apply_rolling_average(y_values, rolling_window)
    if len(y_values) > ROLLING_SAMPLE_LIMIT:
        x_values, y_values = _downsample_series(x_values, y_values, ROLLING_SAMPLE_LIMIT)
    return x_values, y_values


def _apply_rolling_average(values: Sequence[float], window: int) -> List[float]:
    window = max(1, window)
    count = len(values)
    if count <= 1:
        return list(values)

    window = min(window, count)
    cumsum = [0.0]
    for val in values:
        cumsum.append(cumsum[-1] + val)

    averaged: List[float] = []
    for i in range(1, count + 1):
        start = max(0, i - window)
        total = cumsum[i] - cumsum[start]
        samples = i - start
        averaged.append(total / samples)
    return averaged


def _downsample_series(
    x_values: Sequence[int],
    y_values: Sequence[float],
    limit: int,
) -> tuple[List[int], List[float]]:
    total = len(y_values)
    if total <= limit or limit <= 0:
        return list(x_values), list(y_values)

    if limit == 1:
        idx = total - 1
        return [x_values[idx]], [y_values[idx]]

    step = (total - 1) / (limit - 1)
    indices = {min(total - 1, round(i * step)) for i in range(limit)}
    ordered_indices = sorted(indices)
    return [x_values[i] for i in ordered_indices], [y_values[i] for i in ordered_indices]


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
    default_name = f"../time_per_sample_{walltime_key}.png"
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
