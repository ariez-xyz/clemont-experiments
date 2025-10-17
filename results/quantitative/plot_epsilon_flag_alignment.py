#!/usr/bin/env python3
"""Plot alignment between max_ratio and epsilon_monitor_flag across epsilons."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:  # pragma: no cover - script entry point convenience
    from ._plot_utils import DEFAULT_DPI, DEFAULT_FIGSIZE
except ImportError:  # pragma: no cover - script executed from repo root
    from _plot_utils import DEFAULT_DPI, DEFAULT_FIGSIZE


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the per-epsilon mean squared error between normalized max_ratio "
            "values and epsilon_monitor_flag, then plot the result."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=(
            "Optional directories to scan for *maxk_none JSON payloads. Defaults to "
            "the certifair directory alongside this script."
        ),
    )
    parser.add_argument(
        "--pattern",
        default="fair*maxk_none",
        help="Glob pattern used to select epsilon directories when --series is absent (default: fair*maxk_none)",
    )
    parser.add_argument(
        "--series",
        action="append",
        help="Optional labelled pattern in the form label=glob. Provide multiple times for multi-series plots.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output image path (defaults to epsilon_alignment.png alongside the script)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively in addition to saving it",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    base_dirs = args.paths or [script_dir / "certifair"]

    series_specs = _resolve_series_specs(args.series, args.pattern)

    series_payload: List[SeriesData] = []

    for label, pattern in series_specs:
        entries = _collect_entries(base_dirs, pattern=pattern)
        if not entries:
            print(f"Warning: no epsilon directories found for pattern '{pattern}'")
            continue
        epsilons, mses, counts = _compute_series(entries)
        if not epsilons:
            print(f"Warning: pattern '{pattern}' produced no valid statistics")
            continue
        series_payload.append(
            SeriesData(label=label, epsilons=epsilons, mses=mses, counts=counts)
        )

    if not series_payload:
        raise SystemExit("No epsilon settings produced valid statistics")

    fig = _plot_alignment(series_payload, output=args.output)

    if args.show:
        plt.show()


def _collect_entries(
    base_dirs: Sequence[Path],
    *,
    pattern: str,
) -> Dict[float, Path]:
    discovered: Dict[float, Path] = {}
    for root in base_dirs:
        root = root.expanduser().resolve()
        if not root.is_dir():
            print(f"Warning: directory not found {root}")
            continue
        for candidate in sorted(root.glob(pattern)):
            if not candidate.is_dir():
                continue
            epsilon = _parse_epsilon(candidate.name)
            if epsilon is None:
                print(f"Warning: could not parse epsilon from '{candidate.name}'")
                continue
            json_files = sorted(candidate.glob("*.json"))
            if not json_files:
                print(f"Warning: no JSON files found in {candidate}")
                continue
            if len(json_files) > 1:
                json_files.sort(key=lambda f: f.stat().st_mtime)
                print(
                    f"Warning: multiple JSON files in {candidate}; using newest file {json_files[-1].name}"
                )
            discovered[epsilon] = json_files[-1]
    return discovered


def _resolve_series_specs(
    raw_series: Optional[Sequence[str]],
    default_pattern: str,
) -> List[Tuple[str, str]]:
    if raw_series:
        specs: List[Tuple[str, str]] = []
        for token in raw_series:
            if "=" in token:
                label, pattern = token.split("=", 1)
                label = label.strip() or pattern.strip()
                pattern = pattern.strip()
            else:
                pattern = token.strip()
                label = pattern
            if not pattern:
                print("Warning: empty pattern in --series specification; skipping")
                continue
            specs.append((label, pattern))
        return specs
    return [(default_pattern, default_pattern)]


@dataclass
class SeriesData:
    label: str
    epsilons: List[float]
    mses: List[float]
    counts: List[int]


def _compute_series(entries: Dict[float, Path]) -> Tuple[List[float], List[float], List[int]]:
    epsilons: List[float] = []
    mses: List[float] = []
    counts: List[int] = []

    for epsilon, json_path in sorted(entries.items()):
        ratios, flags = _load_pairs(json_path)
        if not ratios:
            print(f"Skipping {json_path}: no usable records")
            continue
        mse = _normalized_mse(ratios, flags)
        epsilons.append(epsilon)
        mses.append(mse)
        counts.append(int(sum(flags)))
    return epsilons, mses, counts


def _parse_epsilon(folder_name: str) -> Optional[float]:
    parts = folder_name.split("_eps_")
    if len(parts) != 2:
        return None
    remainder = parts[1]
    epsilon_token = remainder.split("_", 1)[0]
    epsilon_str = epsilon_token.replace("p", ".")
    try:
        return float(epsilon_str)
    except ValueError:
        return None


def _load_pairs(json_path: Path) -> Tuple[List[float], List[float]]:
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    ratios: List[float] = []
    flags: List[float] = []
    for record in payload.get("records", []):
        ratio = record.get("max_ratio")
        flag = record.get("epsilon_monitor_flag")
        if ratio is None or flag is None:
            continue
        try:
            ratio_val = float(ratio)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(ratio_val):
            continue
        flag_val = _flag_to_float(flag)
        if flag_val is None:
            continue
        ratios.append(ratio_val)
        flags.append(flag_val)
    return ratios, flags


def _flag_to_float(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        try:
            cast = float(value)
        except (TypeError, ValueError):
            return None
        if cast in (0.0, 1.0):
            return cast
        if cast.is_integer() and cast in (0.0, 1.0):
            return cast
        return None
    return None


def _normalized_mse(ratios: Sequence[float], flags: Sequence[float]) -> float:
    ratios_array = np.asarray(ratios, dtype=float)
    flags_array = np.asarray(flags, dtype=float)

    min_ratio = float(np.min(ratios_array))
    max_ratio = float(np.max(ratios_array))
    if max_ratio > min_ratio:
        normalized = (ratios_array - min_ratio) / (max_ratio - min_ratio)
    else:
        normalized = np.zeros_like(ratios_array)

    diff = normalized - flags_array
    return float(np.mean(diff ** 2))


def _plot_alignment(
    series_list: Sequence[SeriesData],
    *,
    output: Optional[Path],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])
    colors = color_cycle if color_cycle else ["#1f77b4"]

    for idx, series in enumerate(series_list):
        ordered = sorted(
            zip(series.epsilons, series.mses, series.counts),
            key=lambda item: item[0],
        )
        eps_arr = np.array([item[0] for item in ordered], dtype=float)
        mse_arr = np.array([item[1] for item in ordered], dtype=float)
        count_arr = np.array([item[2] for item in ordered], dtype=int)

        color = colors[idx % len(colors)]
        ax.plot(
            eps_arr,
            mse_arr,
            color=color,
            linewidth=2.0,
            label=series.label,
        )

#        for pidx, (epsilon, mse, count) in enumerate(zip(eps_arr, mse_arr, count_arr)):
#            ax.annotate(
#                f"{count}" if pidx % 3 == 0 else "",
#                xy=(epsilon, mse),
#                xytext=(0, 6 if idx%2==0 else -10),
#                textcoords="offset points",
#                fontsize="xx-small",
#                ha="center",
#                color=color,
#            )

    ax.set_xlabel("$\\epsilon$")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()

    destination = _resolve_output_path(output)
    fig.savefig(destination, dpi=DEFAULT_DPI, bbox_inches="tight")
    print(f"Saved epsilon alignment plot to {destination}")
    return fig


def _resolve_output_path(candidate: Optional[Path]) -> Path:
    if candidate is None:
        script_dir = Path(__file__).resolve().parent
        return script_dir / "epsilon_alignment.png"
    candidate = candidate.expanduser()
    if candidate.is_dir():
        return candidate / "epsilon_alignment.png"
    return candidate


if __name__ == "__main__":
    main()
