"""Compare top max-ratio point IDs across quantitative monitor runs."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class TopEntry:
    point_id: int
    max_ratio: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "For every pair of quantitative monitor runs, report the overlap of "
            "top point_ids ranked by max_ratio, along with metadata differences."
        ),
    )
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing quant_run_*.json files (defaults to script directory)",
    )
    parser.add_argument(
        "--pattern",
        default="quant_run_*.json",
        help="Glob pattern used to select JSON files (default: quant_run_*.json)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top point_ids to consider from each run (default: 10)",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=5,
        help="Number of leading entries to display for each run (default: 5, use 0 to skip)",
    )
    args = parser.parse_args()

    json_paths = sorted(args.directory.glob(args.pattern))
    if len(json_paths) < 2:
        raise SystemExit("Need at least two JSON files to compare")

    runs = {path: _load_top_entries(path, args.top_k) for path in json_paths}

    for path_a, path_b in itertools.combinations(json_paths, 2):
        _report_pair(
            path_a,
            path_b,
            runs[path_a],
            runs[path_b],
            preview_count=args.preview_count,
        )


def _load_top_entries(path: Path, top_k: int) -> tuple[Sequence[TopEntry], dict[str, object]]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    records: Iterable[dict[str, object]] = payload.get("records", [])
    sorted_records = sorted(
        (record for record in records if "max_ratio" in record and "point_id" in record),
        key=lambda r: (float(r["max_ratio"]), int(r["point_id"])),
        reverse=True,
    )

    top_entries = [
        TopEntry(point_id=int(record["point_id"]), max_ratio=float(record["max_ratio"]))
        for record in itertools.islice(sorted_records, top_k)
    ]

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    return top_entries, metadata


def _report_pair(
    path_a: Path,
    path_b: Path,
    data_a: tuple[Sequence[TopEntry], dict[str, object]],
    data_b: tuple[Sequence[TopEntry], dict[str, object]],
    *,
    preview_count: int,
) -> None:
    top_a, meta_a = data_a
    top_b, meta_b = data_b

    set_a = {entry.point_id for entry in top_a}
    set_b = {entry.point_id for entry in top_b}
    overlap = set_a & set_b

    divider = "=" * 72
    print(divider)
    print(f"{path_a.name}  <->  {path_b.name}")
    print(divider)

    _print_metadata_diff(meta_a, meta_b)

    _print_top_entries(path_a.name, top_a, preview_count)
    _print_top_entries(path_b.name, top_b, preview_count)

    overlap_size = len(overlap)
    base_size = min(len(top_a), len(top_b))
    if base_size:
        overlap_pct = (overlap_size / base_size) * 100
        print(f"Overlap: {overlap_size} / {base_size} = {overlap_pct:.1f}%")
    else:
        print("Overlap: 0 / 0 (no entries to compare)")

    if overlap_size >= 2:
        spearman = _spearman_rho(top_a, top_b, overlap)
        agreement = _pairwise_order_agreement(top_a, top_b, overlap)
        print(f"Order alignment: spearman_rho={spearman:.3f}  pairwise_agreement={agreement:.3f}")
    elif overlap_size == 1:
        print("Order alignment: only one shared point_id (spearman/pairs undefined)")
    else:
        print("Order alignment: no shared point_ids")

    print()


def _print_metadata_diff(meta_a: dict[str, object], meta_b: dict[str, object]) -> None:
    all_keys = sorted(set(meta_a) | set(meta_b))
    differing = []
    for key in all_keys:
        value_a = meta_a.get(key, "<missing>")
        value_b = meta_b.get(key, "<missing>")
        if value_a != value_b:
            differing.append((key, value_a, value_b))

    if differing:
        print("Metadata differences:")
        for key, value_a, value_b in differing:
            print(f"  - {key}: {value_a!r} vs {value_b!r}")
    else:
        print("Metadata identical")
    print()


def _print_top_entries(label: str, entries: Sequence[TopEntry], preview_count: int) -> None:
    if not entries:
        print(f"Top entries ({label}): none available")
        return

    print(f"Top entries ({label}):")
    if preview_count <= 0:
        print("  (preview suppressed)")
        return

    width = len(str(min(len(entries), preview_count)))
    for idx, entry in enumerate(entries[:preview_count], start=1):
        ratio_repr = f"{entry.max_ratio:.6g}" if entry.max_ratio == entry.max_ratio else "nan"
        print(f"  {idx:>{width}}. point_id={entry.point_id}  max_ratio={ratio_repr}")
    remaining = len(entries) - preview_count
    if remaining > 0:
        print(f"  ... ({remaining} more)")
    print()


def _spearman_rho(
    top_a: Sequence[TopEntry],
    top_b: Sequence[TopEntry],
    overlap: set[int],
) -> float:
    ranks_a = _ranks_for_overlap(top_a, overlap)
    ranks_b = _ranks_for_overlap(top_b, overlap)

    n = len(ranks_a)
    if n < 2:
        raise ValueError("Spearman rho requires at least two overlapping entries")

    diff_sq_sum = sum((ranks_a[i] - ranks_b[i]) ** 2 for i in range(n))
    numerator = 6 * diff_sq_sum
    denominator = n * (n**2 - 1)
    return 1 - numerator / denominator


def _pairwise_order_agreement(
    top_a: Sequence[TopEntry],
    top_b: Sequence[TopEntry],
    overlap: set[int],
) -> float:
    ranks_a = _ranks_for_overlap(top_a, overlap)
    ranks_b = _ranks_for_overlap(top_b, overlap)

    n = len(ranks_a)
    if n < 2:
        raise ValueError("Pairwise agreement requires at least two overlapping entries")

    concordant = 0
    total_pairs = n * (n - 1) // 2
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff_a = ranks_a[i] - ranks_a[j]
            diff_b = ranks_b[i] - ranks_b[j]
            if diff_a == 0 or diff_b == 0:
                concordant += 1
            elif (diff_a > 0 and diff_b > 0) or (diff_a < 0 and diff_b < 0):
                concordant += 1

    return concordant / total_pairs


def _ranks_for_overlap(top_entries: Sequence[TopEntry], overlap: set[int]) -> list[int]:
    rank_by_id = {entry.point_id: idx + 1 for idx, entry in enumerate(top_entries)}
    ordered_ids = sorted(overlap)
    return [rank_by_id[point_id] for point_id in ordered_ids]


if __name__ == "__main__":
    main()
