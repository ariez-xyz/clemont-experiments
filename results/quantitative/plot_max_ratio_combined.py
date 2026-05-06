"""Combined histogram of quantitative monitor ratios across multiple runs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from ._plot_utils import DEFAULT_DPI, DEFAULT_FIGSIZE, resolve_json_paths
except ImportError:  # pragma: no cover - script-style execution
    from _plot_utils import DEFAULT_DPI, DEFAULT_FIGSIZE, resolve_json_paths


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MIN_RATIO = 1e-5


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
        "--min-ratio",
        type=float,
        default=DEFAULT_MIN_RATIO,
        help="Drop ratios below this value before plotting. Default: 1e-5.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Opacity for each histogram overlay (default: 0.45)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output image path (default: input path if it is a directory, else next to this script)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Comma-separated list of indices. Ratios from each .json file will be divided into splits accordingly. Splits are assigned an individual color",
    )
    parser.add_argument(
        "--no-title",
        action='store_true',
        dest='no_title'
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Max ratios histogram (combined)",
        help="Plot title. Ignored when --no-title is passed.",
    )
    parser.add_argument(
        "--split-epsilon-flagged",
        action='store_true',
        dest='split_epsilon_flagged'
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Comma-separated list of labels for legend.",
    )
    args = parser.parse_args()

    assert not (args.split_epsilon_flagged and args.split)

    script_dir = Path(__file__).resolve().parent
    json_paths = _collect_json_paths(args.paths, default_dir=script_dir)
    if len(json_paths) < 2 and not args.split:
        print("Warning: fewer than two runs provided; plotting available data anyway.")


    datasets = []
    for json_path in json_paths:
        ratios, kept_all = _load_ratios(json_path, min_ratio=args.min_ratio)
        if ratios.size == 0:
            print(f"Skipping {json_path} (no finite ratios >= {args.min_ratio:g})")
            continue
        if args.split:
            assert kept_all, "split indices are offset by infinite or zero results!"
            split_start = 0
            for split_end in [int(i) for i in args.split.split(",")] + [len(ratios)]:
                print(split_start, split_end, len(ratios[split_start:split_end]))
                datasets.append((f'{split_start}-{split_end-1} of {json_path.stem}', ratios[split_start:split_end]))
                split_start = split_end
            print(len(ratios), [len(d) for l,d in datasets])
        elif args.split_epsilon_flagged:
            flags = _load_flags(json_path)
            datasets.append((f'unflagged', ratios[~flags]))
            datasets.append((f'flagged', ratios[flags]))
        else:
            datasets.append((str(json_path.stem), ratios))

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

    plt.figure(figsize=DEFAULT_FIGSIZE)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    for idx, (label, ratios) in enumerate(datasets):
        color = color_cycle[idx % len(color_cycle)] if color_cycle else None
        label = f"{label} (n={len(ratios)})"
        if args.labels: label = args.labels.split(",")[idx]
        plt.hist(
            ratios,
            bins=bins,
            alpha=args.alpha,
            label=label,
            edgecolor="black",
            linewidth=0.6,
            color=color,
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("i.o.r score")
    plt.ylabel("Frequency")
    if not args.no_title:
        plt.title(args.title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = _resolve_output_path(args.output, args.paths, script_dir)
    plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
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


def _load_flags(json_path: Path) -> np.ndarray:
    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    flags = np.array([record.get("epsilon_monitor_flag") for record in payload.get("records", [])], dtype=bool)
    return flags

def _load_ratios(json_path: Path, *, min_ratio: float) -> Tuple[np.ndarray, bool]:
    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    records = list(payload.get("records", []))
    records, dropped_duplicates = _drop_text_duplicate_witness_pairs(json_path, payload, records)
    if dropped_duplicates:
        print(f"{json_path}: dropped {dropped_duplicates} exact duplicate point/witness pairs")
    ratios = np.array([record.get("max_ratio") for record in records], dtype=float)
    mask = np.isfinite(ratios) & (ratios >= min_ratio)
    return ratios[mask], all(mask[1:])


def _drop_text_duplicate_witness_pairs(
    json_path: Path,
    payload: Mapping[str, Any],
    records: List[Mapping[str, Any]],
) -> tuple[List[Mapping[str, Any]], int]:
    input_csv = _resolve_input_csv(json_path, payload)
    if input_csv is None:
        return records, 0
    text_columns = _text_columns(input_csv)
    if text_columns is None:
        return records, 0

    try:
        frame = pd.read_csv(input_csv, usecols=list(text_columns), engine="python")
    except Exception as exc:
        print(f"Warning: could not read text columns from {input_csv}: {exc}")
        return records, 0

    kept: List[Mapping[str, Any]] = []
    dropped = 0
    for record in records:
        point_idx = _int_value(record.get("index", record.get("point_id")), -1)
        witness_idx = _int_value(record.get("witness_id"), None)
        if witness_idx is None:
            kept.append(record)
            continue
        if not (0 <= point_idx < len(frame) and 0 <= witness_idx < len(frame)):
            kept.append(record)
            continue
        point_text = _raw_text(frame.iloc[point_idx], text_columns)
        witness_text = _raw_text(frame.iloc[witness_idx], text_columns)
        if point_text == witness_text:
            dropped += 1
            continue
        kept.append(record)
    return kept, dropped


def _resolve_input_csv(json_path: Path, payload: Mapping[str, Any]) -> Optional[Path]:
    metadata = payload.get("metadata") or {}
    raw = metadata.get("input_csv") if isinstance(metadata, Mapping) else None
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
        parts = path.parts
        if "data" in parts:
            candidates.append(REPO_ROOT / Path(*parts[parts.index("data") :]))
    else:
        candidates.extend([REPO_ROOT / path, json_path.parent / path, Path.cwd() / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _text_columns(input_csv: Path) -> Optional[tuple[str, ...]]:
    try:
        header = set(pd.read_csv(input_csv, nrows=0).columns)
    except Exception:
        return None
    if "user_input" in header:
        return ("user_input",)
    if "review_text" in header:
        return ("review_title", "review_text") if "review_title" in header else ("review_text",)
    return None


def _raw_text(row: Mapping[str, Any], columns: Sequence[str]) -> str:
    if columns == ("user_input",):
        return str(row.get("user_input") or "")
    if "review_text" in columns:
        title = str(row.get("review_title") or "") if "review_title" in columns else ""
        body = str(row.get("review_text") or "")
        return f"{title}\n{body}" if title else body
    return ""


def _int_value(value: Any, default: Optional[int] = 0) -> Optional[int]:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_output_path(candidate: Optional[Path], input_paths: Iterable[Path], script_dir: Path) -> Path:
    if candidate is None:
        input_paths_list = list(input_paths)
        if len(input_paths_list) > 1 or not input_paths_list[0].is_dir():
            return script_dir / "quant_run_combined_ratios.png"
        return input_paths_list[0] / "quant_run_combined_ratios.png"
    candidate = candidate.expanduser()
    if candidate.is_dir():
        return candidate / "quant_run_combined_ratios.png"
    parent = candidate.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    return candidate


if __name__ == "__main__":
    main()
