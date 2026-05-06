#!/usr/bin/env python3
"""Generate a portable PDF report for a text quantitative monitor run."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "clemont-matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown/PDF report for a text monitor JSON run.",
    )
    parser.add_argument("monitor_json", type=Path, help="Path to quant_run_*.json")
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of highest-scoring point/witness pairs to include. Default: 10.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PDF path. Default: <monitor_json stem>_report.pdf next to the JSON.",
    )
    parser.add_argument(
        "--keep-md",
        action="store_true",
        help="Keep the intermediate Markdown file even after successful PDF generation.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Only write Markdown and assets; do not invoke pandoc.",
    )
    parser.add_argument(
        "--pdf-engine",
        default=None,
        help="Pandoc PDF engine override. Default: first available of xelatex, lualatex, pdflatex.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    monitor_path = args.monitor_json.expanduser().resolve()
    payload = json.loads(monitor_path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    records = normalize_records(payload.get("records", []))
    if not records:
        raise ValueError(f"No records found in {monitor_path}")

    input_csv = resolve_path(
        metadata.get("input_csv"),
        relative_to=monitor_path.parent,
        metadata=metadata,
        records=records,
    )
    frame = read_text_csv(input_csv)
    manifest = load_json_if_exists(input_csv.with_suffix(".json"))

    output_pdf = args.output or monitor_path.with_name(f"{monitor_path.stem}_report.pdf")
    output_pdf = output_pdf.expanduser().resolve()
    asset_dir = output_pdf.with_suffix("").with_name(f"{output_pdf.stem}_assets")
    asset_dir.mkdir(parents=True, exist_ok=True)
    histogram_path = asset_dir / "score_histogram.png"
    write_histogram(records, histogram_path)

    markdown = render_markdown(
        monitor_path=monitor_path,
        input_csv=input_csv,
        metadata=metadata,
        manifest=manifest,
        records=records,
        frame=frame,
        histogram_path=histogram_path,
        top_k=args.top_k,
    )
    md_path = output_pdf.with_suffix(".md")
    md_path.write_text(markdown, encoding="utf-8")

    if args.no_pdf:
        print(f"Wrote Markdown report to {md_path}")
        return

    run_pandoc(md_path, output_pdf, pdf_engine=args.pdf_engine)
    if not args.keep_md:
        md_path.unlink(missing_ok=True)
    print(f"Wrote PDF report to {output_pdf}")


def normalize_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for record in records:
        if "result" in record and isinstance(record["result"], Mapping):
            merged = dict(record["result"])
            merged.update(record)
            normalized.append(merged)
        else:
            normalized.append(dict(record))
    return normalized


def resolve_path(
    value: Any,
    *,
    relative_to: Path,
    metadata: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> Path:
    if not value:
        raise ValueError("monitor JSON metadata.input_csv is missing")
    raw = Path(str(value)).expanduser()
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
        parts = raw.parts
        if "data" in parts:
            idx = parts.index("data")
            candidates.append(REPO_ROOT / Path(*parts[idx:]))
    else:
        candidates.extend([relative_to / raw, REPO_ROOT / raw])
    candidates.append(REPO_ROOT / raw.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    inferred = infer_renamed_text_csv(raw, metadata=metadata, records=records)
    if inferred is not None:
        return inferred
    raise FileNotFoundError(f"Could not resolve path from monitor JSON: {value}")


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

    max_index = max(
        [
            int_value(record.get("index", record.get("point_id")), default=-1) or -1
            for record in records
        ],
        default=-1,
    )
    expected_classes = len(metadata.get("probability_columns") or [])
    scored: list[tuple[int, Path]] = []
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
        scored.append((score, candidate.resolve()))

    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1].name), reverse=True)
    return scored[0][1]


def read_text_csv(path: Path) -> pd.DataFrame:
    header = list(pd.read_csv(path, nrows=0).columns)
    usecols = [col for col in header if not re.fullmatch(r"e\d+", col)]
    # The C parser can fail while concatenating chunks for very wide CSVs even
    # when most columns are dropped via usecols. The Python engine is slower but
    # only used for report generation, and avoids that parser bug.
    return pd.read_csv(path, usecols=usecols, engine="python")


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_histogram(records: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    scores = [float_value(record.get("max_ratio")) for record in records]
    scores = [score for score in scores if math.isfinite(score)]
    if not scores:
        scores = [0.0]

    plt.figure(figsize=(7.0, 3.2))
    plt.hist(scores, bins=40, color="#2f6480", edgecolor="#f7f0e3", alpha=0.9)
    plt.xlabel("Robustness score")
    plt.ylabel("Count")
    plt.yscale("log")
    plt.title("Robustness score distribution")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def render_markdown(
    *,
    monitor_path: Path,
    input_csv: Path,
    metadata: Mapping[str, Any],
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    frame: pd.DataFrame,
    histogram_path: Path,
    top_k: int,
) -> str:
    openrouter = manifest.get("openrouter", {}) if isinstance(manifest, Mapping) else {}
    probability_labels = infer_probability_labels(frame.columns)
    output_transform = str(metadata.get("output_transform") or "probs")
    top_records = sorted(
        [record for record in records if math.isfinite(float_value(record.get("max_ratio")))],
        key=lambda record: float_value(record.get("max_ratio")),
        reverse=True,
    )[:top_k]

    lines: list[str] = [
        "---",
        f"title: Text Monitor Report",
        "geometry: margin=0.8in",
        "---",
        "",
        "# Text Monitor Report",
        "",
        "## Run Summary",
        "",
        f"- Monitor JSON: `{monitor_path.relative_to(REPO_ROOT) if is_relative_to(monitor_path, REPO_ROOT) else monitor_path}`",
        f"- Input CSV: `{input_csv.relative_to(REPO_ROOT) if is_relative_to(input_csv, REPO_ROOT) else input_csv}`",
        f"- Points processed: {len(records):,}",
        f"- Output transform: `{output_transform}`",
        f"- Output metric: `{metadata.get('out_metric', 'n/a')}`",
        f"- Input metric: `{metadata.get('frnn_metric', 'n/a')}`",
        f"- Judge model: `{openrouter.get('chat_model', 'n/a')}`",
        f"- Embedding model: `{openrouter.get('embedding_model', 'n/a')}`",
        f"- Top pairs shown: {len(top_records):,}",
        "",
        "## Score Distribution",
        "",
        f"![Robustness score histogram]({histogram_path})",
        "",
        "## Highest-Scoring Pairs",
        "",
    ]

    for rank, record in enumerate(top_records, start=1):
        point_idx = int_value(record.get("index", record.get("point_id")))
        witness_idx = int_value(record.get("witness_id"), default=None)
        point_row = row_at(frame, point_idx)
        witness_row = row_at(frame, witness_idx) if witness_idx is not None else None

        lines.extend(
            [
                f"### {rank}. Point {point_idx}",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Robustness score | {fmt(record.get('max_ratio'))} |",
                f"| Input distance | {fmt(record.get('witness_in_distance'))} |",
                f"| Output distance | {fmt(record.get('witness_out_distance'))} |",
                f"| Witness index | {witness_idx if witness_idx is not None else 'none'} |",
                f"| Compared count | {record.get('compared_count', 'n/a')} |",
                f"| Largest k | {last(record.get('k_progression')) or 'n/a'} |",
                "",
                render_example_block(
                    "Point",
                    point_row,
                    probability_labels,
                    output_transform=output_transform,
                ),
                "",
            ]
        )
        if witness_row is not None:
            lines.extend(
                [
                    render_example_block(
                        "Witness",
                        witness_row,
                        probability_labels,
                        output_transform=output_transform,
                    ),
                    "",
                ]
            )
        else:
            lines.extend(["**Witness:** none", ""])

    return "\n".join(lines).rstrip() + "\n"


def render_example_block(
    title: str,
    row: Mapping[str, Any],
    labels: Sequence[str],
    *,
    output_transform: str,
) -> str:
    dataset = dataset_kind(row)
    text = text_value(row)
    sampled = sampled_token(row, labels)
    true_label = true_label_value(row, dataset)
    lines = [
        f"**{title}**",
        "",
        f"- ID: `{row.get('example_id', 'n/a')}`",
        f"- Sampled token: `{sampled}`",
        f"- True label: {true_label}",
    ]
    if dataset == "amazon" and row.get("review_title"):
        lines.append(f"- Review title: {safe_inline(row.get('review_title'))}")
    lines.extend(["", truncate_block(text), ""])

    if output_transform == "argmax-normalized":
        lines.extend(
            [
                f"- Monitor output: normalized argmax `{fmt(normalized_argmax(row, labels))}`",
                "",
            ]
        )

    lines.extend(["| Label | Probability | Bar |", "| --- | ---: | --- |"])
    for label in labels:
        prob = probability(row, label)
        marker = " *" if label == predicted_label(row, labels) else ""
        lines.append(f"| `{label}{marker}` | {format_percent(prob)} | `{bar(prob)}` |")
    return "\n".join(lines)


def dataset_kind(row: Mapping[str, Any]) -> str:
    if "review_text" in row or "rating_value" in row:
        return "amazon"
    if "user_input" in row or "toxicity_label" in row:
        return "toxic_chat"
    return "text"


def text_value(row: Mapping[str, Any]) -> str:
    if "review_text" in row:
        return str(row.get("review_text") or "")
    if "user_input" in row:
        return str(row.get("user_input") or "")
    return ""


def true_label_value(row: Mapping[str, Any], dataset: str) -> str:
    if dataset == "amazon":
        rating = float_value(row.get("rating_value", row.get("rating")))
        if not math.isfinite(rating):
            return "n/a"
        formatted = str(int(rating)) if rating.is_integer() else fmt(rating)
        return f"{formatted} {'star' if rating == 1 else 'stars'}"
    if dataset == "toxic_chat":
        toxicity = row.get("toxicity_label", "n/a")
        return "toxic" if str(toxicity) == "1" else "safe" if str(toxicity) == "0" else str(toxicity)
    return "n/a"


def infer_probability_labels(columns: Iterable[str]) -> list[str]:
    labels = [col.removeprefix("prob_") for col in columns if re.fullmatch(r"prob_\d+", col)]
    return sorted(labels, key=lambda label: int(label))


def row_at(frame: pd.DataFrame, idx: int | None) -> Mapping[str, Any]:
    if idx is None or idx < 0 or idx >= len(frame):
        return {}
    return frame.iloc[idx].to_dict()


def sampled_token(row: Mapping[str, Any], labels: Sequence[str]) -> str:
    answer = str(row.get("judge_answer") or row.get("first_token") or "").strip()
    return answer if answer in labels else predicted_label(row, labels)


def predicted_label(row: Mapping[str, Any], labels: Sequence[str]) -> str:
    if not labels:
        return "n/a"
    return max(labels, key=lambda label: probability(row, label))


def probability(row: Mapping[str, Any], label: str) -> float:
    return clamp01(float_value(row.get(f"prob_{label}")))


def normalized_argmax(row: Mapping[str, Any], labels: Sequence[str]) -> float:
    numeric = [float(label) for label in labels]
    if len(numeric) < 2:
        return 0.0
    label = float(predicted_label(row, labels))
    lo = min(numeric)
    hi = max(numeric)
    return (label - lo) / (hi - lo) if hi > lo else 0.0


def truncate_block(text: str, limit: int = 1400) -> str:
    text = text.strip()
    if len(text) > limit:
        text = text[:limit].rstrip() + "..."
    if not text:
        text = "(empty)"
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())


def safe_inline(value: Any, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    if len(text) > limit:
        text = text[:limit].rstrip() + "..."
    return text.replace("|", "\\|")


def bar(probability_value: float, width: int = 28) -> str:
    filled = round(clamp01(probability_value) * width)
    return "#" * filled + "-" * (width - filled)


def format_percent(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    if 0 < value < 0.001:
        return "<0.1%"
    return f"{value * 100:.1f}%"


def fmt(value: Any) -> str:
    num = float_value(value)
    if not math.isfinite(num):
        return "n/a"
    if abs(num) >= 100 or (abs(num) < 0.001 and num != 0):
        return f"{num:.2e}"
    return f"{num:.4f}".rstrip("0").rstrip(".")


def float_value(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def int_value(value: Any, default: int | None = 0) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def last(values: Any) -> Any:
    return values[-1] if isinstance(values, Sequence) and values else None


def is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


def run_pandoc(md_path: Path, output_pdf: Path, *, pdf_engine: str | None = None) -> None:
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        raise RuntimeError("pandoc not found; rerun with --no-pdf to keep Markdown only")
    engine = pdf_engine or first_available_pdf_engine()
    cmd = [
        pandoc,
        str(md_path),
        "-o",
        str(output_pdf),
    ]
    if engine:
        cmd.append(f"--pdf-engine={engine}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "pandoc failed; rerun with --keep-md to inspect the generated Markdown"
        ) from exc


def first_available_pdf_engine() -> str | None:
    for engine in ("xelatex", "lualatex", "pdflatex"):
        if shutil.which(engine):
            return engine
    return None


if __name__ == "__main__":
    main()
