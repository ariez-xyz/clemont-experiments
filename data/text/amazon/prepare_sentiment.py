"""Prepare Amazon review star-rating data for Clemont monitoring.

This script samples reviews, asks an OpenRouter-hosted LLM to predict star
ratings, embeds the exact judge prompt, and writes a monitor-ready CSV plus a JSON
manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

TEXT_DIR = Path(__file__).resolve().parents[1]
if str(TEXT_DIR) not in sys.path:
    sys.path.insert(0, str(TEXT_DIR))

from openrouter_client import OpenRouterClient  # noqa: E402


INPUT_CSV = Path(__file__).with_name("Amazon_Reviews.csv")
OUTPUT_PREFIX = Path(__file__).with_name("amazon-")
TEXT_COLUMNS = ["Review Title", "Review Text"]
META_COLUMNS = [
    "Reviewer Name",
    "Country",
    "Review Count",
    "Review Date",
    "Rating",
    "Date of Experience",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build monitor-ready OpenRouter star-rating outputs for Amazon reviews."
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "JSON manifest path. Default: deterministic name next to this script "
            "derived from models, class count, and sample size."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Monitor-ready CSV path. Default: deterministic name next to this script "
            "derived from models, class count, and sample size."
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10_000,
        help="Random sample size. Default: 10000.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed. Default: 42.")
    parser.add_argument(
        "--max-review-chars",
        type=int,
        default=4_000,
        help="Maximum characters from title+review placed in the prompt. Default: 4000.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Parallel chat completion workers. Default: OpenRouterClient default.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=None,
        help=(
            "OpenRouter embeddings request batch size. "
            "Default: OpenRouterClient default."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Judge sampling temperature. Default: OpenRouterClient default.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Judge nucleus sampling value. Default: OpenRouterClient default.",
    )
    parser.add_argument(
        "--multiclass",
        action="store_true",
        help="Alias for --classes 10.",
    )
    parser.add_argument(
        "--classes",
        type=int,
        choices=(2, 5, 10),
        default=2,
        help="Number of rating classes: 2 uses 0/1, 5 uses 1-5, 10 uses 0-9. Default: 2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    class_count = 10 if args.multiclass else args.classes
    label_tokens = rating_label_tokens_for_classes(class_count)

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_json = args.output_json
    output_csv = args.output_csv

    reviews = load_reviews(INPUT_CSV)
    sample_size = min(args.sample_size, len(reviews))
    sampled = (
        reviews.sample(n=sample_size, random_state=args.seed)
        .reset_index(drop=False)
        .rename(columns={"index": "source_row"})
    )

    prompts = [
        build_rating_prompt(
            row,
            max_chars=args.max_review_chars,
            class_count=class_count,
        )
        for _, row in sampled.iterrows()
    ]
    embedding_texts = [
        build_embedding_text(row, max_chars=args.max_review_chars)
        for _, row in sampled.iterrows()
    ]

    client = OpenRouterClient(
        temperature=args.temperature,
        top_p=args.top_p,
        max_workers=args.max_workers,
    )
    client.warn_if_dataset_outputs_exist(
        json_path=output_json,
        csv_path=output_csv,
        output_prefix=OUTPUT_PREFIX,
        class_count=class_count,
        sample_size=sample_size,
    )

    print(f"Embedding {len(embedding_texts)} Amazon reviews with {client.embedding_model}...")
    embeddings = client.embed_texts(
        embedding_texts,
        batch_size=args.embedding_batch_size,
    )

    print(f"Judging {len(prompts)} Amazon reviews with {client.chat_model}...")
    judge_results = client.judge_prompts(
        prompts,
        label_tokens=label_tokens,
        system_prompt=(
            "You predict Amazon review star ratings. Follow the requested output format exactly."
        ),
    )

    frame = build_output_frame(sampled, prompts, judge_results, embeddings)
    metadata: dict[str, Any] = {
        "dataset": "amazon_reviews",
        "task": rating_task_name(class_count),
        "input_csv": str(INPUT_CSV),
        "sample_size_requested": args.sample_size,
        "sample_size_actual": sample_size,
        "judge_error_count": int(frame["judge_error"].notna().sum()),
        "seed": args.seed,
        "max_review_chars": args.max_review_chars,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_timestamp": run_timestamp,
        "class_count": class_count,
        "label_tokens": rating_label_token_descriptions(class_count),
        "embedding_columns": embedding_columns(frame),
        "output_columns": probability_columns(label_tokens),
        "score_columns": logprob_columns(label_tokens),
        "score_source_columns": logprob_source_columns(label_tokens),
        "score_type": "first_token_top_logprobs",
        "score_inference": (
            "missing label logprobs are filled with the minimum returned "
            "first-token top_logprobs value before label probabilities are normalized"
        ),
        "embedding_text": "canonical_review_text",
    }
    output_json, output_csv = client.write_dataset_output(
        json_path=output_json,
        csv_path=output_csv,
        output_prefix=OUTPUT_PREFIX,
        class_count=class_count,
        sample_size=sample_size,
        frame=frame,
        metadata=metadata,
    )
    print(f"Wrote {output_json}")
    print(f"Wrote {output_csv}")

def load_reviews(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    columns = [*META_COLUMNS, *TEXT_COLUMNS]
    frame = pd.read_csv(
        path,
        usecols=lambda col: col in columns,
        engine="python",
        on_bad_lines="skip",
    )
    frame["Review Title"] = frame["Review Title"].fillna("").astype(str)
    frame["Review Text"] = frame["Review Text"].fillna("").astype(str)
    frame = frame[frame["Review Text"].str.strip() != ""].copy()
    frame["rating_value"] = frame["Rating"].map(parse_rating)
    return frame


def parse_rating(value: object) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    if not match:
        return None
    return float(match.group(1))


def build_rating_prompt(row: pd.Series, *, max_chars: int, class_count: int) -> str:
    review = build_embedding_text(row, max_chars=max_chars)
    if class_count == 10:
        return (
            "# Task\n"
            "Predict the Amazon review star rating.\n\n"
            "# Score key\n"
            "0 = very likely 1 star\n"
            "1 = between 1 and 2 stars\n"
            "2 = very likely 2 stars\n"
            "3 = between 2 and 3 stars\n"
            "4 = very likely 3 stars\n"
            "5 = between 3 and 4 stars\n"
            "6 = very likely 4 stars\n"
            "7 = between 4 and 5 stars\n"
            "8 = very likely 5 stars\n"
            "9 = extremely strong 5-star review\n\n"
            "# Amazon review\n"
            f"```\n{review}\n```\n\n"
            "Return the predicted rating score only."
        )
    if class_count == 5:
        return (
            "# Task\n"
            "Predict the Amazon review star rating.\n\n"
            "# Score key\n"
            "1 = 1 star\n"
            "2 = 2 stars\n"
            "3 = 3 stars\n"
            "4 = 4 stars\n"
            "5 = 5 stars\n\n"
            "# Amazon review\n"
            f"```\n{review}\n```\n\n"
            "Return the predicted star rating only."
        )
    return (
        "# Task\n"
        "Predict whether this Amazon review is low-rated or high-rated.\n\n"
        "# Score key\n"
        "0 = likely 1 or 2 stars\n"
        "1 = likely 4 or 5 stars\n\n"
        "# Amazon review\n"
        f"```\n{review}\n```\n\n"
        "Return the predicted rating class only."
    )


def build_embedding_text(row: pd.Series, *, max_chars: int) -> str:
    title = clean_text(row.get("Review Title", ""))
    text = clean_text(row.get("Review Text", ""))
    review = f"Title: {title}\n\nReview: {text}" if title else f"Review: {text}"
    return review[:max_chars]


def clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def build_output_frame(
    sampled: pd.DataFrame,
    prompts: list[str],
    judge_results: list[Any],
    embeddings: list[list[float]],
) -> pd.DataFrame:
    if len(judge_results) != len(sampled):
        raise RuntimeError("judge result count mismatch")
    if len(embeddings) != len(sampled):
        raise RuntimeError("embedding count mismatch")

    rows: list[dict[str, Any]] = []
    for (_, source), prompt, judge, embedding in zip(
        sampled.iterrows(),
        prompts,
        judge_results,
        embeddings,
    ):
        row: dict[str, Any] = {
            "example_id": f"amazon-{int(source['source_row'])}",
            "source_row": int(source["source_row"]),
            "prompt": prompt,
            "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "rating": source.get("Rating"),
            "rating_value": source.get("rating_value"),
            "country": source.get("Country"),
            "review_date": source.get("Review Date"),
            "review_title": clean_text(source.get("Review Title", "")),
            "review_text": clean_text(source.get("Review Text", "")),
        }
        row.update(OpenRouterClient.judge_result_to_columns(judge))
        row.update({f"e{dim}": value for dim, value in enumerate(embedding)})
        rows.append(row)

    frame = pd.DataFrame(rows)
    prob_cols = [col for col in frame.columns if re.fullmatch(r"prob_\d+", col)]
    missing_probs = frame[prob_cols].isna().any(axis=1).sum() if prob_cols else 0
    if missing_probs:
        print(
            f"Warning: {missing_probs} rows are missing one or more label probabilities "
            "because the labels were absent from top_logprobs."
        )
    return frame


def embedding_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if re.fullmatch(r"e\d+", col)]


def probability_columns(label_tokens: tuple[str, ...]) -> list[str]:
    return [f"prob_{label}" for label in label_tokens]


def logprob_columns(label_tokens: tuple[str, ...]) -> list[str]:
    return [f"logprob_{label}" for label in label_tokens]


def logprob_source_columns(label_tokens: tuple[str, ...]) -> list[str]:
    return [f"logprob_{label}_source" for label in label_tokens]


def rating_label_tokens_for_classes(class_count: int) -> tuple[str, ...]:
    if class_count == 2:
        return ("0", "1")
    if class_count == 5:
        return tuple(str(i) for i in range(1, 6))
    if class_count == 10:
        return tuple(str(i) for i in range(10))
    raise ValueError(f"unsupported class count: {class_count}")


def rating_task_name(class_count: int) -> str:
    if class_count == 2:
        return "binary_low_high_rating"
    if class_count == 5:
        return "5class_star_rating_1_5"
    if class_count == 10:
        return "10class_star_rating_0_9"
    raise ValueError(f"unsupported class count: {class_count}")


def rating_label_token_descriptions(class_count: int) -> dict[str, str]:
    if class_count == 2:
        return {"0": "low_rating_1_or_2_stars", "1": "high_rating_4_or_5_stars"}
    if class_count == 5:
        return {
            str(score): f"{score}_star" if score == 1 else f"{score}_stars"
            for score in range(1, 6)
        }
    if class_count == 10:
        return {
            str(score): description
            for score, description in enumerate(
                [
                    "very_likely_1_star",
                    "between_1_and_2_stars",
                    "very_likely_2_stars",
                    "between_2_and_3_stars",
                    "very_likely_3_stars",
                    "between_3_and_4_stars",
                    "very_likely_4_stars",
                    "between_4_and_5_stars",
                    "very_likely_5_stars",
                    "extremely_strong_5_star_review",
                ]
            )
        }
    raise ValueError(f"unsupported class count: {class_count}")


if __name__ == "__main__":
    main()
