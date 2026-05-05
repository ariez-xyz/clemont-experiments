#!/usr/bin/env python3
"""Print compact summaries for text quantitative monitor JSON runs."""

from __future__ import annotations

import argparse
import difflib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DISTANCE_THRESHOLDS = (1e-2, 1e-3, 1e-4, 1e-5)
ROBUSTNESS_ROWS = (
    ("n", None),
    ("sum", None),
    ("mean", None),
    ("p25", 25),
    ("p50", 50),
    ("p75", 75),
    ("p90", 90),
    ("p95", 95),
    ("p99", 99),
    ("max", None),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize one monitor JSON, or recursively summarize monitor JSONs "
            "under one or more directories."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Monitor JSON file(s) or directories containing quant_run_*.json files.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of highest robustness-loss point/witness pairs to print. Default: 10.",
    )
    parser.add_argument(
        "--text-chars",
        type=int,
        default=4000,
        help="Maximum characters to show for each point/witness text. Default: 220.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = discover_monitor_jsons(args.paths)
    if not paths:
        raise SystemExit("No monitor JSONs found.")

    for idx, path in enumerate(paths):
        if idx:
            print("\n" + "=" * 100 + "\n")
        summarize_run(path, top_k=args.top_k, text_chars=args.text_chars)


def discover_monitor_jsons(paths: Sequence[Path]) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        path = raw.expanduser()
        if path.is_dir():
            found.extend(sorted(path.rglob("quant_run_*.json")))
        elif path.is_file():
            found.append(path)
        else:
            print(f"Skipping missing path: {path}")
    return sorted({item.resolve() for item in found})


def summarize_run(path: Path, *, top_k: int, text_chars: int) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = normalize_records(payload.get("records", []))
    metadata = payload.get("metadata", {})
    if not records:
        print(f"{short_path(path)}\nNo records.")
        return

    input_csv = resolve_input_csv(metadata.get("input_csv"), monitor_path=path, metadata=metadata, records=records)
    frame = read_relevant_csv(input_csv, metadata=metadata)
    manifest = load_json_if_exists(input_csv.with_suffix(".json"))
    witness_records = exclude_duplicate_witness_text_pairs(frame, records)

    print_run_metadata(path, input_csv, metadata, manifest, records)
    print(f"duplicate point/witness pairs excluded from witness stats: {len(records) - len(witness_records):,} / {len(records):,}")
    print()
    print_accuracy(frame, records)
    print()
    print_robustness_table(witness_records)
    print()
    print_top_pairs(frame, witness_records, top_k=top_k, text_chars=text_chars)


def normalize_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for record in records:
        if "result" in record and isinstance(record["result"], Mapping):
            merged = dict(record["result"])
            merged.update(record)
            normalized.append(merged)
        else:
            normalized.append(dict(record))
    return normalized


def resolve_input_csv(
    value: Any,
    *,
    monitor_path: Path,
    metadata: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> Path:
    if not value:
        raise ValueError(f"{monitor_path}: metadata.input_csv is missing")

    raw = Path(str(value)).expanduser()
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
        parts = raw.parts
        if "data" in parts:
            candidates.append(REPO_ROOT / Path(*parts[parts.index("data") :]))
    else:
        candidates.extend([monitor_path.parent / raw, REPO_ROOT / raw])
    candidates.append(REPO_ROOT / raw.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    inferred = infer_renamed_text_csv(raw, metadata=metadata, records=records)
    if inferred is not None:
        return inferred
    raise FileNotFoundError(f"{monitor_path}: could not resolve input CSV {value!r}")


def infer_renamed_text_csv(
    raw_path: Path,
    *,
    metadata: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> Path | None:
    raw_text = str(raw_path).replace("\\", "/")
    if "data/text/amazon" in raw_text:
        data_dir = REPO_ROOT / "data/text/amazon"
        pattern = "amazon-judge-*.csv"
    elif "data/text/toxic-chat" in raw_text:
        data_dir = REPO_ROOT / "data/text/toxic-chat"
        pattern = "toxic-chat-judge-*.csv"
    else:
        return None
    if not data_dir.exists():
        return None

    expected_classes = len(metadata.get("probability_columns") or [])
    max_index = max((int_value(record.get("index", record.get("point_id")), -1) for record in records), default=-1)
    scored: list[tuple[int, str, Path]] = []
    for candidate in data_dir.glob(pattern):
        try:
            header = list(pd.read_csv(candidate, nrows=0).columns)
            row_count = sum(1 for _ in candidate.open("rb")) - 1
        except Exception:
            continue
        if row_count <= max_index:
            continue
        prob_count = len([col for col in header if re.fullmatch(r"prob_\d+", col)])
        score = 0
        if expected_classes and prob_count == expected_classes:
            score += 10
        if f"n{len(records)}" in candidate.stem:
            score += 5
        scored.append((score, candidate.name, candidate.resolve()))

    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][2]


def read_relevant_csv(path: Path, *, metadata: Mapping[str, Any]) -> pd.DataFrame:
    header = list(pd.read_csv(path, nrows=0).columns)
    desired = {
        "example_id",
        "source_row",
        "rating",
        "rating_value",
        "review_title",
        "review_text",
        "toxicity_label",
        "jailbreaking_label",
        "user_input",
        "judge_answer",
        "first_token",
    }
    desired.update(col for col in header if re.fullmatch(r"prob_\d+", col))
    desired.update(str(col) for col in metadata.get("probability_columns", []) if col in header)
    usecols = [col for col in header if col in desired]
    return pd.read_csv(path, usecols=usecols)


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def print_run_metadata(
    monitor_path: Path,
    input_csv: Path,
    metadata: Mapping[str, Any],
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> None:
    openrouter = manifest.get("openrouter", {}) if isinstance(manifest, Mapping) else {}
    output_transform = metadata.get("output_transform", "n/a")
    output_transform_meta = metadata.get("output_transform_metadata") or {}
    output_detail = ""
    if isinstance(output_transform_meta, Mapping) and output_transform_meta.get("description"):
        output_detail = f" ({output_transform_meta['description']})"

    print(short_path(monitor_path))
    rows = [
        ("input_csv", short_path(input_csv)),
        ("records", f"{len(records):,}"),
        ("timestamp", metadata.get("timestamp", "n/a")),
        ("judge_model", openrouter.get("chat_model", "n/a")),
        ("embedding_model", openrouter.get("embedding_model", "n/a")),
        ("temperature", openrouter.get("temperature", "n/a")),
        ("output_transform", f"{output_transform}{output_detail}"),
        ("output_columns", ", ".join(map(str, metadata.get("output_columns", [])[:8])) or "n/a"),
        ("out_metric", metadata.get("out_metric", "n/a")),
        ("frnn_metric", metadata.get("frnn_metric", "n/a")),
        ("max_k", metadata.get("max_k", "n/a")),
        ("batchsize", metadata.get("batchsize", "n/a")),
        ("walltime_seconds", fmt(metadata.get("walltime_seconds"))),
    ]
    print_table(rows, headers=("metadata", "value"))


def print_accuracy(frame: pd.DataFrame, records: Sequence[Mapping[str, Any]]) -> None:
    labels = probability_labels(frame.columns)
    if not labels:
        print("Accuracy: n/a (no probability columns)")
        return

    pairs: list[tuple[int, Mapping[str, Any]]] = []
    for record in records:
        idx = int_value(record.get("index", record.get("point_id")), -1)
        if 0 <= idx < len(frame):
            pairs.append((idx, frame.iloc[idx]))

    if not pairs:
        print("Accuracy: n/a (no records map to CSV rows)")
        return

    dataset = dataset_kind(frame.columns)
    rows: list[tuple[str, str]] = []
    if dataset == "toxic_chat":
        for threshold in (2, 3):
            correct = total = 0
            for _, row in pairs:
                true = int_value(row.get("toxicity_label"), -1)
                pred = int_value(predicted_label(row, labels), -999)
                if true not in (0, 1) or pred == -999:
                    continue
                total += 1
                correct += int(pred >= threshold) == true
            rows.append((f"toxic if pred >= {threshold}", accuracy_text(correct, total)))
    else:
        correct = total = 0
        for _, row in pairs:
            true = actual_class(row)
            pred = predicted_label(row, labels)
            if true is None or pred is None:
                continue
            total += 1
            correct += str(true) == str(pred)
        rows.append(("exact class", accuracy_text(correct, total)))

    print("Accuracy")
    print_table(rows, headers=("mapping", "accuracy"))


def print_robustness_table(records: Sequence[Mapping[str, Any]]) -> None:
    columns = ["all", *[format_threshold(threshold) for threshold in DISTANCE_THRESHOLDS]]
    score_sets = [filtered_scores(records, None), *[filtered_scores(records, threshold) for threshold in DISTANCE_THRESHOLDS]]
    rows: list[tuple[Any, ...]] = []
    for name, percentile_value in ROBUSTNESS_ROWS:
        values: list[str] = []
        for scores in score_sets:
            if name == "n":
                values.append(f"{len(scores):,}")
            elif not scores:
                values.append("n/a")
            elif name == "sum":
                values.append(fmt(sum(scores)))
            elif name == "mean":
                values.append(fmt(statistics.fmean(scores)))
            elif name == "max":
                values.append(fmt(max(scores)))
            else:
                values.append(fmt(percentile(scores, percentile_value or 0)))
        rows.append((name, *values))

    print(
        "Robustness loss for non-duplicate point/witness pairs after dropping records "
        "where witness input or output distance is at or below threshold"
    )
    print_table(rows, headers=("stat", *columns))


def filtered_scores(records: Sequence[Mapping[str, Any]], threshold: float | None) -> list[float]:
    scores: list[float] = []
    for record in records:
        score = float_value(record.get("max_ratio"))
        in_distance = float_value(record.get("witness_in_distance"))
        out_distance = float_value(record.get("witness_out_distance"))
        if math.isnan(score):
            continue
        if not math.isfinite(in_distance) or not math.isfinite(out_distance):
            continue
        if threshold is not None and (in_distance <= threshold or out_distance <= threshold):
            continue
        scores.append(score)
    return scores


def exclude_duplicate_witness_text_pairs(
    frame: pd.DataFrame,
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for record in records:
        point_idx = int_value(record.get("index", record.get("point_id")), -1)
        witness_idx = int_value(record.get("witness_id"), None)
        if witness_idx is None:
            filtered.append(dict(record))
            continue
        point = row_at(frame, point_idx)
        witness = row_at(frame, witness_idx)
        if raw_dataset_text(point) == raw_dataset_text(witness):
            continue
        filtered.append(dict(record))
    return filtered


def print_top_pairs(
    frame: pd.DataFrame,
    records: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
    text_chars: int,
) -> None:
    labels = probability_labels(frame.columns)
    top_records = sorted(
        [record for record in records if not math.isnan(float_value(record.get("max_ratio")))],
        key=lambda record: float_value(record.get("max_ratio")),
        reverse=True,
    )[:top_k]

    print(f"Top {min(top_k, len(top_records))} highest robustness-loss pairs")
    if not top_records:
        print("n/a")
        return

    for rank, record in enumerate(top_records, start=1):
        point_idx = int_value(record.get("index", record.get("point_id")), -1)
        witness_idx = int_value(record.get("witness_id"), None)
        point = row_at(frame, point_idx)
        witness = row_at(frame, witness_idx)
        print(
            f"\n{rank}. point={point_idx} witness={witness_idx if witness_idx is not None else 'none'} "
            f"loss={fmt(record.get('max_ratio'))} "
            f"in={fmt(record.get('witness_in_distance'))} "
            f"out={fmt(record.get('witness_out_distance'))} "
            f"k={last(record.get('k_progression')) or 'n/a'}"
        )
        print(f"   Point:   {describe_row(point, labels)}")
        print(f"            {compact_text(text_value(point), text_chars)}")
        if witness:
            print(f"   Witness: {describe_row(witness, labels)}")
            print(f"            {compact_text(text_value(witness), text_chars)}")
            print(f"   Diff:    {compact_diff(raw_dataset_text(point), raw_dataset_text(witness), text_chars)}")
        else:
            print("   Witness: none")


def probability_labels(columns: Iterable[str]) -> list[str]:
    labels = [col.removeprefix("prob_") for col in columns if re.fullmatch(r"prob_\d+", str(col))]
    return sorted(labels, key=lambda label: int(label))


def predicted_label(row: Mapping[str, Any], labels: Sequence[str]) -> str | None:
    if not labels:
        return None
    return max(labels, key=lambda label: probability(row, label))


def probability(row: Mapping[str, Any], label: str) -> float:
    return float_value(row.get(f"prob_{label}"))


def actual_class(row: Mapping[str, Any]) -> str | None:
    if "rating_value" in row and math.isfinite(float_value(row.get("rating_value"))):
        return str(int(round(float_value(row.get("rating_value")))))
    if "rating" in row and math.isfinite(float_value(row.get("rating"))):
        return str(int(round(float_value(row.get("rating")))))
    if "toxicity_label" in row:
        value = int_value(row.get("toxicity_label"), None)
        if value in (0, 1):
            return str(value)
    return None


def dataset_kind(columns: Iterable[str]) -> str:
    column_set = set(columns)
    if "toxicity_label" in column_set or "user_input" in column_set:
        return "toxic_chat"
    if "rating_value" in column_set or "review_text" in column_set:
        return "amazon"
    return "unknown"


def describe_row(row: Mapping[str, Any], labels: Sequence[str]) -> str:
    if not row:
        return "missing"
    pred = predicted_label(row, labels)
    actual = actual_class(row)
    example_id = row.get("example_id", "n/a")
    dist = ", ".join(f"{label}:{fmt(probability(row, label))}" for label in labels)
    return f"id={example_id} true={actual or 'n/a'} pred={pred or 'n/a'} probs=[{dist}]"


def text_value(row: Mapping[str, Any]) -> str:
    if not row:
        return ""
    if "review_text" in row:
        title = str(row.get("review_title") or "").strip()
        body = str(row.get("review_text") or "").strip()
        return f"{title}: {body}" if title else body
    if "user_input" in row:
        return str(row.get("user_input") or "")
    return ""


def raw_dataset_text(row: Mapping[str, Any]) -> str:
    if not row:
        return ""
    if "user_input" in row:
        return str(row.get("user_input") or "")
    if "review_text" in row:
        title = str(row.get("review_title") or "")
        body = str(row.get("review_text") or "")
        return f"{title}\n{body}" if title else body
    return text_value(row)


def row_at(frame: pd.DataFrame, idx: int | None) -> dict[str, Any]:
    if idx is None or idx < 0 or idx >= len(frame):
        return {}
    return frame.iloc[idx].to_dict()


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return math.nan
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def compact_text(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) > limit:
        text = text[:limit].rstrip() + "..."
    return text or "(empty)"


def compact_diff(point_text: str, witness_text: str, limit: int) -> str:
    point = " ".join(str(point_text or "").split())
    witness = " ".join(str(witness_text or "").split())
    if not point and not witness:
        return "(empty)"
    if point == witness:
        return "(identical)"

    matcher = difflib.SequenceMatcher(a=point, b=witness, autojunk=False)
    chunks: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        before = snippet(point[max(0, i1 - 24) : i1], prefix=i1 > 24)
        after = snippet(point[i2 : min(len(point), i2 + 24)], suffix=i2 + 24 < len(point))
        removed = point[i1:i2]
        added = witness[j1:j2]
        chunks.append(f"{before}[-{removed}-]{{+{added}+}}{after}")
        if len(" ".join(chunks)) >= limit:
            break

    diff = " ... ".join(chunk for chunk in chunks if chunk)
    if len(diff) > limit:
        diff = diff[:limit].rstrip() + "..."
    return diff or "(diff unavailable)"


def snippet(text: str, *, prefix: bool = False, suffix: bool = False, limit: int = 48) -> str:
    value = text
    if len(value) > limit:
        if prefix and suffix:
            keep = max(8, (limit - 3) // 2)
            value = value[:keep].rstrip() + "..." + value[-keep:].lstrip()
        elif prefix:
            value = "..." + value[-limit:].lstrip()
        elif suffix:
            value = value[:limit].rstrip() + "..."
        else:
            value = value[:limit].rstrip()
    if prefix and not value.startswith("...") and text:
        value = "..." + value
    if suffix and not value.endswith("...") and text:
        value = value + "..."
    return value


def accuracy_text(correct: int, total: int) -> str:
    if total <= 0:
        return "n/a"
    return f"{100.0 * correct / total:.2f}% ({correct:,}/{total:,})"


def print_table(rows: Sequence[Sequence[Any]], *, headers: Sequence[str]) -> None:
    string_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in string_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    print("  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("  ".join("-" * widths[idx] for idx in range(len(headers))))
    for row in string_rows:
        print("  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)))


def short_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def format_threshold(value: float) -> str:
    return f"> {value:.0e}"


def fmt(value: Any) -> str:
    num = float_value(value)
    if math.isinf(num):
        return "inf" if num > 0 else "-inf"
    if not math.isfinite(num):
        return "n/a"
    if abs(num) >= 100000 or (0 < abs(num) < 0.0001):
        return f"{num:.4e}"
    return f"{num:.4f}".rstrip("0").rstrip(".")


def float_value(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def int_value(value: Any, default: int | None = 0) -> int | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def last(values: Any) -> Any:
    return values[-1] if isinstance(values, Sequence) and values else None


if __name__ == "__main__":
    main()
