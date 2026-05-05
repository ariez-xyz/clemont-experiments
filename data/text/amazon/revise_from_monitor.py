"""Revise Amazon sentiment judge outputs using monitor witness feedback."""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

TEXT_DIR = Path(__file__).resolve().parents[1]
if str(TEXT_DIR) not in sys.path:
    sys.path.insert(0, str(TEXT_DIR))

from openrouter_client import OpenRouterClient  # noqa: E402
from revise_from_monitor_common import (  # noqa: E402
    apply_revisions,
    build_prompt_map,
    class_count_from_probability_columns,
    finite_float,
    format_distribution,
    label_tokens_from_probability_columns,
    load_monitor_payload,
    probability_columns,
    resolve_input_csv,
    run_revision_judging,
    sampled_token,
    select_revision_indices,
    write_revised_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a revised Amazon sentiment CSV from a monitor JSON."
    )
    parser.add_argument("monitor_json", type=Path, help="Baseline monitor JSON.")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--revision-mode",
        choices=("all", "top-k", "top-fraction"),
        default="all",
        help="Rows to revise. Default: all rows with a witness.",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.10,
        help="Fraction used with --revision-mode top-fraction. Default: 0.10.",
    )
    parser.add_argument("--min-robustness-loss", type=float, default=None)
    parser.add_argument(
        "--allow-missing-witness",
        action="store_true",
        help="Also revise rows without monitor witnesses.",
    )
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_monitor_payload(args.monitor_json)
    input_csv = resolve_input_csv(args.monitor_json, payload)
    if is_revised_csv(input_csv):
        raise ValueError(
            f"refusing to revise an already revised CSV: {input_csv}. "
            "Run this script on a baseline monitor JSON."
        )
    frame = pd.read_csv(input_csv)
    records = payload["records"]
    prob_columns = probability_columns(frame)
    label_tokens = label_tokens_from_probability_columns(prob_columns)
    class_count = class_count_from_probability_columns(prob_columns)

    selected = select_revision_indices(
        records,
        mode=args.revision_mode,
        top_k=args.top_k,
        top_fraction=args.top_fraction,
        min_robustness_loss=args.min_robustness_loss,
        require_witness=not args.allow_missing_witness,
    )
    prompts = build_prompt_map(
        frame=frame,
        records=records,
        selected_indices=selected,
        prob_columns=prob_columns,
        prompt_builder=lambda row, witness, record, probs: build_revision_prompt(
            row,
            witness,
            record,
            probs,
            class_count=class_count,
        ),
    )

    client = OpenRouterClient(
        temperature=args.temperature,
        top_p=args.top_p,
        max_workers=args.max_workers,
    )
    print(
        f"Revising {len(prompts)} / {len(frame)} Amazon rows with "
        f"{client.chat_model} using monitor {args.monitor_json}"
    )
    judge_results = run_revision_judging(
        client=client,
        prompts_by_index=prompts,
        label_tokens=label_tokens,
        system_prompt=(
            "You revise sentiment classifier outputs using monitor witness evidence. "
            "Follow the requested output format exactly."
        ),
    )
    revised = apply_revisions(
        frame,
        records=records,
        selected_indices=set(prompts),
        prompts_by_index=prompts,
        judge_results_by_index=judge_results,
        label_tokens=label_tokens,
    )
    output_csv = args.output_csv
    output_json = args.output_json
    if output_csv is None and output_json is None:
        output_csv = input_csv.with_name(
            f"{input_csv.stem}_{revision_suffix(args)}.csv"
        )
        output_json = output_csv.with_suffix(".json")

    metadata = {
        "dataset": "amazon_reviews",
        "task": f"{class_count}class_sentiment_monitor_informed_revision",
        "input_csv": str(input_csv),
        "source_monitor_json": str(args.monitor_json),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "class_count": class_count,
        "label_tokens": {label: label for label in label_tokens},
        "sample_size_actual": int(len(frame)),
        "revision_strategy": "monitor_witness_revision",
        "revision_mode": args.revision_mode,
        "revision_top_k": args.top_k,
        "revision_top_fraction": args.top_fraction,
        "revision_min_robustness_loss": args.min_robustness_loss,
        "revision_require_witness": not args.allow_missing_witness,
        "revision_selected_count": int(len(prompts)),
        "output_columns": list(prob_columns),
        "embedding_columns": [col for col in frame.columns if re.fullmatch(r"e\d+", col)],
    }
    output_json, output_csv = write_revised_output(
        client=client,
        frame=revised,
        input_csv=input_csv,
        output_csv=output_csv,
        output_json=output_json,
        metadata=metadata,
    )
    print(f"Wrote {output_json}")
    print(f"Wrote {output_csv}")


def build_revision_prompt(
    row: pd.Series,
    witness: pd.Series | None,
    record: Mapping[str, Any],
    prob_columns: Sequence[str],
    *,
    class_count: int,
) -> str:
    label_instruction = sentiment_label_instruction(class_count)
    review = amazon_review_text(row)
    robustness_loss = finite_float(record.get("max_ratio"))
    witness_in = finite_float(record.get("witness_in_distance"))
    witness_out = finite_float(record.get("witness_out_distance"))
    witness_block = "No witness was available for this row."
    if witness is not None:
        witness_block = (
            f"Witness review:\n{amazon_review_text(witness)}\n\n"
            f"Witness sampled token: {sampled_token(witness)}\n"
            f"Witness probabilities: {format_distribution(witness, prob_columns)}"
        )

    return (
        "Revise an Amazon review sentiment judgement using monitor feedback.\n\n"
        "The monitor found whether this row's output changed sharply relative to "
        "similar earlier rows. Treat the witness as consistency evidence, not as "
        "automatically correct. Revise only if the evidence suggests the current "
        "decision is inconsistent with the review text.\n\n"
        f"{label_instruction}\n\n"
        f"Current review:\n{review}\n\n"
        f"Current sampled token: {sampled_token(row)}\n"
        f"Current probabilities: {format_distribution(row, prob_columns)}\n"
        f"Robustness loss: {robustness_loss if robustness_loss is not None else 'n/a'}\n"
        f"Input distance to witness: {witness_in if witness_in is not None else 'n/a'}\n"
        f"Output distance to witness: {witness_out if witness_out is not None else 'n/a'}\n\n"
        f"{witness_block}\n\n"
        "Return the revised sentiment score only."
    )


def amazon_review_text(row: pd.Series) -> str:
    title = clean(row.get("review_title", ""))
    text = clean(row.get("review_text", ""))
    if title:
        return f"Title: {title}\nReview: {text}"
    return f"Review: {text}"


def sentiment_label_instruction(class_count: int) -> str:
    if class_count == 10:
        return (
            "Return exactly one digit from 0 to 9: 0 = extremely negative, "
            "5 = mixed or neutral, 9 = extremely positive."
        )
    if class_count == 5:
        return (
            "Return exactly one digit from 1 to 5: 1 = very negative, "
            "3 = mixed or neutral, 5 = very positive."
        )
    return "Return exactly one digit: 0 = negative sentiment, 1 = positive sentiment."


def clean(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def is_revised_csv(path: Path) -> bool:
    return "witness_revised" in path.stem


def revision_suffix(args: argparse.Namespace) -> str:
    if args.revision_mode == "top-k":
        base = f"witness_revised_top{args.top_k}"
    elif args.revision_mode == "top-fraction":
        percent = int(round(args.top_fraction * 100))
        base = f"witness_revised_top{percent}pct"
    else:
        base = "witness_revised_all"
    if args.min_robustness_loss is not None:
        safe = str(args.min_robustness_loss).replace(".", "p")
        base += f"_minloss{safe}"
    return base


if __name__ == "__main__":
    main()
