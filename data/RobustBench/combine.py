#!/usr/bin/env python3
"""Concatenate two CSV files while skipping duplicates from the second file."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Concatenate two CSV files, keeping all rows from the first file and "
            "only those rows from the second file that are not already present in "
            "the first."
        )
    )
    parser.add_argument("first", type=Path, help="Path to the primary CSV file")
    parser.add_argument("second", type=Path, help="Path to the secondary CSV file")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Optional output path (defaults to stdout)",
    )
    return parser.parse_args()


def read_rows(path: Path) -> tuple[list[str], list[Sequence[str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:  # empty file
            raise SystemExit(f"CSV file is empty: {path}") from exc
        rows = [tuple(row) for row in reader]
    return header, rows


def write_rows(
    header: Iterable[str],
    rows: Iterable[Sequence[str]],
    *,
    destination: Path | None,
) -> None:
    if destination is None:
        writer = csv.writer(sys.stdout)
        writer.writerow(header)
        writer.writerows(rows)
        return

    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    first_header, first_rows = read_rows(args.first)
    second_header, second_rows = read_rows(args.second)

    if first_header != second_header:
        raise SystemExit(
            "CSV headers differ between inputs; refusing to merge."
        )

    seen = set(first_rows)
    filtered_second = [row for row in second_rows if tuple(row) not in seen]

    combined = list(first_rows) + filtered_second
    write_rows(first_header, combined, destination=args.output)


if __name__ == "__main__":
    main()
