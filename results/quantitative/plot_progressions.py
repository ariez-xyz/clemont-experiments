"""Overlay ratio/bound progression lines for selected quantitative monitor points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np

try:
    from ._plot_utils import metadata_value, resolve_json_paths
except ImportError:  # pragma: no cover - script-style execution
    from _plot_utils import metadata_value, resolve_json_paths

RATIO_COLOR = "#1f77b4"
BOUND_COLOR = "#ff7f0e"
INTERSECTION_COLOR = "#2ca02c"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay the ratio/bound progression curves for high-priority points "
            "in quantitative monitor runs. Generates two plots per run: one for the "
            "records with the largest compared_count values and one for those with "
            "the largest max_ratio values."
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
        "--top-k",
        type=int,
        default=1000,
        help="Number of points to overlay for each chart (default: 1000)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.08,
        help="Line alpha for individual progressions (default: 0.08)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory for output images (defaults to the JSON file's directory)",
    )
    parser.add_argument(
        "--point-ids",
        type=str,
        help="Comma-separated point IDs to plot instead of automatic top selections",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    json_paths = resolve_json_paths(args.json_path, default_dir=script_dir)

    point_ids = _parse_point_ids(args.point_ids)

    for json_path in json_paths:
        _generate_plots(
            json_path,
            top_k=args.top_k,
            alpha=args.alpha,
            output_dir=args.output_dir,
            point_ids=point_ids,
        )


def _generate_plots(
    json_path: Path,
    *,
    top_k: int,
    alpha: float,
    output_dir: Optional[Path],
    point_ids: Optional[list[int]],
) -> None:
    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    records = _filter_records(payload.get("records", []))
    if not records:
        print(f"No progression records found in {json_path}")
        return

    metadata = payload.get("metadata", {})

    if point_ids:
        records_by_id = {}
        for record in records:
            pid = record.get("point_id")
            if pid is None:
                continue
            try:
                pid_int = int(pid)
            except (TypeError, ValueError):
                continue
            records_by_id[pid_int] = record

        missing: list[int] = []
        for pid in point_ids:
            if pid not in records_by_id:
                missing.append(pid)

        if len(missing) == len(point_ids):
            print(f"No matching point_ids found in {json_path} for {point_ids}")
            return
        if missing:
            print(
                f"Warning: the following point_ids were not present in {json_path}: {missing}"
            )

        # Preserve requested order but drop duplicates while keeping first occurrence.
        seen: set[int] = set()
        ordered_records: list[dict] = []
        plotted_ids: list[int] = []
        for pid in point_ids:
            if pid in seen:
                continue
            record = records_by_id.get(pid)
            if record is None:
                continue
            seen.add(pid)
            ordered_records.append(record)
            plotted_ids.append(pid)

        if not ordered_records:
            print(f"No matching point_ids found in {json_path} for {point_ids}")
            return

        output_base = output_dir or json_path.parent
        id_label = ",".join(str(pid) for pid in plotted_ids)
        description = f"point_ids={id_label}"
        output_path = output_base / f"{json_path.stem}_progressions_point_ids.png"
        _render_overlay(
            json_path,
            ordered_records,
            metadata,
            description=description,
            output_path=output_path,
            alpha=alpha,
        )
        return

    top_by_compared = _select_top(records, key="compared_count", limit=top_k)
    top_by_ratio = _select_top(records, key="max_ratio", limit=top_k)

    output_base = output_dir or json_path.parent

    if top_by_compared:
        output_path = output_base / f"{json_path.stem}_progressions_top_compared.png"
        _render_overlay(
            json_path,
            top_by_compared,
            metadata,
            description=f"top {len(top_by_compared)} by compared_count",
            output_path=output_path,
            alpha=alpha,
        )
    else:
        print(f"No records with compared_count data in {json_path}")

    if top_by_ratio:
        output_path = output_base / f"{json_path.stem}_progressions_top_ratio.png"
        _render_overlay(
            json_path,
            top_by_ratio,
            metadata,
            description=f"top {len(top_by_ratio)} by max_ratio",
            output_path=output_path,
            alpha=alpha,
        )
    else:
        print(f"No records with max_ratio data in {json_path}")


def _filter_records(raw_records: Iterable[dict]) -> list[dict]:
    filtered = []
    for record in raw_records:
        k = record.get("k_progression")
        ratio = record.get("ratio_progression")
        bound = record.get("bound_progression")
        if not k or not ratio or not bound:
            continue
        if len(k) != len(ratio) or len(k) != len(bound):
            continue
        filtered.append(record)
    return filtered


def _select_top(records: list[dict], *, key: str, limit: int) -> list[dict]:
    sorted_records = sorted(
        (record for record in records if record.get(key) is not None),
        key=lambda rec: (float(rec.get(key, 0.0)), int(rec.get("point_id", -1))),
        reverse=True,
    )
    if limit <= 0:
        return list(sorted_records)
    return list(sorted_records[:limit])


def _render_overlay(
    json_path: Path,
    records: list[dict],
    metadata: dict,
    *,
    description: str,
    output_path: Path,
    alpha: float,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    min_ratio: Optional[float] = None
    min_bound: Optional[float] = None
    intersection_points_x: list[float] = []
    intersection_points_y: list[float] = []

    for record in records:
        k_vals = np.asarray(record["k_progression"], dtype=float)
        k_vals = list(filter(lambda v: v >= 16, k_vals))
        ratio_vals = np.asarray(record["ratio_progression"], dtype=float)[-len(k_vals):]
        bound_vals = np.asarray(record["bound_progression"], dtype=float)[-len(k_vals):]

        ratio_min = float(np.nanmin(ratio_vals))
        bound_min = float(np.nanmin(bound_vals))
        if (bound_min < 0): print(record)
        min_ratio = ratio_min if min_ratio is None else min(min_ratio, ratio_min)
        min_bound = bound_min if min_bound is None else min(min_bound, bound_min)

        optional_intersection = _add_progression_to_ax(ax, k_vals, ratio_vals, bound_vals, alpha=alpha)
        if optional_intersection:
            ix, iy = optional_intersection
            intersection_points_x.append(ix)
            intersection_points_y.append(iy)

    ax.scatter(
        intersection_points_x,
        intersection_points_y,
        color=INTERSECTION_COLOR,
        s=18,
        marker='x',
        alpha=min(1.0, alpha * 2.5),
        linewidths=1,
        zorder=5,
    )

    _configure_axes(fig, ax, metadata, description, min_ratio, min_bound)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved progression overlay to {output_path}")
    plt.close(fig)


def _add_progression_to_ax(
    ax: plt.Axes,
    k_vals: np.ndarray,
    ratio_vals: np.ndarray,
    bound_vals: np.ndarray,
    *,
    alpha: float,
) -> Optional[tuple[float, float]]:
    ax.plot(k_vals, ratio_vals, color=RATIO_COLOR, linewidth=0.8, alpha=alpha)
    ax.plot(k_vals, bound_vals, color=BOUND_COLOR, linewidth=0.8, alpha=alpha)
    return _find_intersection(k_vals, ratio_vals, bound_vals)

def _find_intersection(
    k_vals: np.ndarray,
    ratio_vals: np.ndarray,
    bound_vals: np.ndarray,
) -> Optional[tuple[float, float]]:
    diff = ratio_vals - bound_vals
    for idx, current in enumerate(diff):
        if current >= 0:
            if idx == 0 or diff[idx - 1] >= 0:
                return float(k_vals[idx]), float(ratio_vals[idx])

            prev_diff = diff[idx - 1]
            if prev_diff == current:
                return float(k_vals[idx]), float(ratio_vals[idx])

            prev_k = k_vals[idx - 1]
            prev_ratio = ratio_vals[idx - 1]
            prev_bound = bound_vals[idx - 1]
            curr_k = k_vals[idx]
            curr_ratio = ratio_vals[idx]
            # Linear interpolation on diff to approximate crossing location.
            denom = prev_diff - current
            if denom == 0:
                return float(curr_k), float(curr_ratio)
            frac = prev_diff / denom
            frac = float(np.clip(frac, 0.0, 1.0))
            k_interp = prev_k + (curr_k - prev_k) * frac
            ratio_interp = prev_ratio + (curr_ratio - prev_ratio) * frac
            bound_interp = prev_bound + (bound_vals[idx] - prev_bound) * frac
            y_value = (ratio_interp + bound_interp) / 2.0
            return float(k_interp), float(y_value)
    return None


def _configure_axes(
    fig: plt.Figure,
    ax: plt.Axes,
    metadata: dict,
    description: str,
    min_ratio: Optional[float],
    min_bound: Optional[float],
) -> None:
    ax.set_xlabel("k (neighbors)")
    ax.set_ylabel("Value")

    try:
        ax.set_xscale("log", base=2)
    except TypeError:
        ax.set_xscale("log")

    finite_mins = [val for val in (min_ratio, min_bound) if val is not None and np.isfinite(val)]
    if finite_mins and min(finite_mins) > 0:
        ax.set_yscale("log")

    total_time = metadata_value(metadata, "total_time")
    out_metric = metadata_value(metadata, "out_metric")
    exponent = metadata_value(metadata, "output_exponent", fallback_key="input_exponent")
    max_k = metadata_value(metadata, "max_k")

    meta_bits = []
    if total_time is not None:
        try:
            meta_bits.append(f"{round(float(total_time))}ms")
        except Exception:
            pass
    if max_k is not None:
        meta_bits.append(f"max_k={max_k}")
    if out_metric is not None:
        meta_bits.append(f"metric={out_metric}")
    if exponent is not None:
        meta_bits.append(f"exponent={exponent}")

    title = "Ratio vs bound progression"
    if description:
        title += f" — {description}"
    if meta_bits:
        title += f" ({', '.join(meta_bits)})"
    ax.set_title(title)

    legend_handles = [
        plt.Line2D([], [], color=RATIO_COLOR, linewidth=1.5, label="Progression of largest seen ratio"),
        plt.Line2D([], [], color=BOUND_COLOR, linewidth=1.5, label="Bound progression"),
        plt.Line2D(
            [],
            [],
            marker="x",
            color=INTERSECTION_COLOR,
            linestyle="",
            markersize=6,
            label="Crossover",
        ),
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize="small")
    ax.grid(True, which="both", alpha=0.15)
    fig.tight_layout()


def _parse_point_ids(raw: Optional[str]) -> Optional[list[int]]:
    if raw is None:
        return None
    tokens = [token.strip() for token in raw.split(",")]
    ids: list[int] = []
    for token in tokens:
        if not token:
            continue
        try:
            ids.append(int(token))
        except ValueError as exc:
            raise SystemExit(f"Invalid point id '{token}' for --point-ids") from exc
    if not ids:
        raise SystemExit("--point-ids provided but no valid IDs parsed")
    return ids


if __name__ == "__main__":
    main()
