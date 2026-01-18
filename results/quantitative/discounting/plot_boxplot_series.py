"""Plot boxplot series for interpolated quantitative runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PARENT_DIR))

from _plot_utils import DEFAULT_DPI  # noqa: E402


def main() -> None:
    json_paths = sorted(SCRIPT_DIR.rglob("quant_run_*.json"))
    if not json_paths:
        raise SystemExit(f"No quant_run_*.json files found under {SCRIPT_DIR}")

    by_dir: Dict[Path, List[Path]] = {}
    for json_path in json_paths:
        if _should_plot(json_path):
            by_dir.setdefault(json_path.parent, []).append(json_path)

    for folder, paths in by_dir.items():
        groups = _group_by_discount(paths)
        for discount, trio in groups.items():
            ylim = _max_whisker_in_dir(list(trio.values()))
            _plot_triptych(folder, discount, trio, ylim)


def _should_plot(json_path: Path) -> bool:
    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    meta = payload.get("metadata", {})
    bins = meta.get("interpolate_bins")
    return isinstance(bins, int) and bins < 100


def _max_whisker_in_dir(paths: List[Path]) -> float:
    max_ratio = 0.0
    for json_path in paths:
        with json_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        records = payload["records"]
        ratios = np.array([rec["max_ratio"] for rec in records], dtype=float)
        ratios = ratios[np.isfinite(ratios)]
        meta = payload.get("metadata", {})
        bins = meta.get("interpolate_bins")
        if not isinstance(bins, int) or bins <= 0:
            continue
        num_points = len(records)
        bin_idx = (np.arange(num_points) * bins) // num_points
        for idx in range(bins):
            data = ratios[bin_idx == idx]
            if data.size == 0:
                continue
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            upper = q3 + 1.5 * iqr
            eligible = data[data <= upper]
            whisker = float(np.max(eligible)) if eligible.size else float(np.max(data))
            max_ratio = max(max_ratio, whisker)
    return max_ratio * 1.05


def _group_by_discount(paths: List[Path]) -> Dict[float, Dict[str, Path]]:
    grouped: Dict[float, Dict[str, Path]] = {}
    for json_path in paths:
        with json_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        meta = payload.get("metadata", {})
        discount = meta.get("discount_factor")
        weights = meta.get("interpolate_weights")
        if discount is None:
            continue
        if isinstance(weights, list) and all(w == 0 for w in weights):
            kind = "base"
        elif isinstance(weights, list) and all(w == 1 for w in weights):
            kind = "fair"
        else:
            kind = "interp"
        grouped.setdefault(float(discount), {})[kind] = json_path
    return {k: v for k, v in grouped.items() if {"base", "interp", "fair"} <= set(v)}


def _plot_triptych(folder: Path, discount: float, trio: Dict[str, Path], ylim_max: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    order = [("base", "Base only"), ("interp", "Interpolated"), ("fair", "Fair only")]
    for ax, (key, label) in zip(axes, order):
        _plot_boxplots(ax, trio[key], ylim_max, label)
    fig.suptitle(f"Ratio distribution by interpolation bin  |  discount={discount:g}")
    fig.tight_layout()
    output_path = folder / f"discount_{discount:g}_boxplots.png"
    fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved boxplots to {output_path}")


def _plot_boxplots(ax: plt.Axes, json_path: Path, ylim_max: float, subtitle: str) -> None:
    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    records = payload["records"]
    ratios = np.array([rec["max_ratio"] for rec in records], dtype=float)
    finite_mask = np.isfinite(ratios)

    meta = payload.get("metadata", {})
    bins = int(meta["interpolate_bins"])
    num_points = len(records)

    if meta.get("interpolate_weights") is not None:
        weights = np.asarray(meta["interpolate_weights"], dtype=float)
    else:
        weights = np.linspace(0.0, 1.0, bins)
    bin_idx = (np.arange(num_points) * bins) // num_points
    bin_idx = np.clip(bin_idx, 0, bins - 1)

    grouped: List[np.ndarray] = []
    labels: List[str] = []
    for idx in range(bins):
        mask = bin_idx == idx
        mask &= finite_mask
        bin_ratios = ratios[mask]
        if bin_ratios.size == 0:
            bin_ratios = np.array([np.nan], dtype=float)
        grouped.append(bin_ratios)

        start = (idx * num_points) // bins
        end = ((idx + 1) * num_points) // bins
        base_pct = int(round((1.0 - weights[idx]) * 100))
        fair_pct = int(round(weights[idx] * 100))
        labels.append(f"{start}-{end}\n{base_pct}%/{fair_pct}%")

    box = ax.boxplot(
        grouped,
        tick_labels=labels,
        showfliers=True,
        patch_artist=True,
        medianprops={"color": "#111111", "linewidth": 1.2},
        boxprops={"edgecolor": "#2D2A26", "linewidth": 1.0},
        whiskerprops={"color": "#2D2A26", "linewidth": 1.0},
        capprops={"color": "#2D2A26", "linewidth": 1.0},
        flierprops={
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": "#2D2A26",
            "markeredgecolor": "none",
            "alpha": 0.35,
        },
    )
    base_color = np.array([0.12, 0.47, 0.71])
    fair_color = np.array([1.0, 0.5, 0.05])
    mix_colors = (1.0 - weights[:, None]) * base_color + weights[:, None] * fair_color
    for patch, color in zip(box["boxes"], mix_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xlabel("Interpolation bin (samples + base/fair mix)")
    ax.set_ylabel("Max ratio")
    ax.set_title(subtitle)
    ax.grid(True, axis="y", alpha=0.25)
    if ylim_max > 0:
        ax.set_ylim(0, ylim_max)


if __name__ == "__main__":
    main()
