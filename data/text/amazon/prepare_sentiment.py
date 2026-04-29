"""Prepare Amazon review sentiment data for Clemont monitoring.

This script samples reviews, asks an OpenRouter-hosted LLM to judge sentiment,
embeds the exact judge prompt, and writes a monitor-ready CSV plus a JSON
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
        description="Build monitor-ready OpenRouter sentiment outputs for Amazon reviews."
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
        help="Number of sentiment classes: 2 uses 0/1, 5 uses 1-5, 10 uses 0-9. Default: 2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    class_count = 10 if args.multiclass else args.classes
    label_tokens = sentiment_label_tokens_for_classes(class_count)

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
        build_sentiment_prompt(
            row,
            max_chars=args.max_review_chars,
            class_count=class_count,
        )
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

    print(f"Judging {len(prompts)} Amazon reviews with {client.chat_model}...")
    judge_results = client.judge_prompts(
        prompts,
        label_tokens=label_tokens,
        system_prompt=(
            "You are a sentiment classifier. Follow the requested output format exactly."
        ),
    )

    print(f"Embedding {len(prompts)} prompts with {client.embedding_model}...")
    embeddings = client.embed_texts(prompts, batch_size=args.embedding_batch_size)

    frame = build_output_frame(sampled, prompts, judge_results, embeddings)
    metadata: dict[str, Any] = {
        "dataset": "amazon_reviews",
        "task": sentiment_task_name(class_count),
        "input_csv": str(INPUT_CSV),
        "sample_size_requested": args.sample_size,
        "sample_size_actual": sample_size,
        "judge_error_count": int(frame["judge_error"].notna().sum()),
        "seed": args.seed,
        "max_review_chars": args.max_review_chars,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_timestamp": run_timestamp,
        "class_count": class_count,
        "label_tokens": sentiment_label_token_descriptions(class_count),
        "embedding_columns": embedding_columns(frame),
        "output_columns": probability_columns(label_tokens),
        "score_columns": logprob_columns(label_tokens),
        "score_source_columns": logprob_source_columns(label_tokens),
        "score_type": "first_token_top_logprobs",
        "score_inference": (
            "missing label logprobs are filled with the minimum returned "
            "first-token top_logprobs value before label probabilities are normalized"
        ),
        "prompt_embedding": "exact_prompt_sent_to_judge",
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


def build_sentiment_prompt(row: pd.Series, *, max_chars: int, class_count: int) -> str:
    title = clean_text(row.get("Review Title", ""))
    text = clean_text(row.get("Review Text", ""))
    review = f"Title: {title}\n\nReview: {text}" if title else f"Review: {text}"
    review = review[:max_chars]
    if class_count == 10:
        return (
            "Analyze the sentiment of this Amazon customer review.\n\n"
            "Return exactly one digit from 0 to 9:\n"
            "0 = extremely negative sentiment\n"
            "5 = mixed or neutral sentiment\n"
            "9 = extremely positive sentiment\n\n"
            f"{review}\n\n"
            "Answer with one digit from 0 to 9 only."
        )
    if class_count == 5:
        return (
            "Analyze the sentiment of this Amazon customer review.\n\n"
            "Return exactly one digit from 1 to 5:\n"
            "1 = very negative sentiment\n"
            "3 = mixed or neutral sentiment\n"
            "5 = very positive sentiment\n\n"
            f"{review}\n\n"
            "Answer with one digit from 1 to 5 only."
        )
    return (
        "Analyze the sentiment of this Amazon customer review.\n\n"
        "Return exactly one digit:\n"
        "0 = negative sentiment\n"
        "1 = positive sentiment\n\n"
        f"{review}\n\n"
        "Answer with 0 or 1 only."
    )


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


def sentiment_label_tokens_for_classes(class_count: int) -> tuple[str, ...]:
    if class_count == 2:
        return ("0", "1")
    if class_count == 5:
        return tuple(str(i) for i in range(1, 6))
    if class_count == 10:
        return tuple(str(i) for i in range(10))
    raise ValueError(f"unsupported class count: {class_count}")


def sentiment_task_name(class_count: int) -> str:
    if class_count == 2:
        return "binary_sentiment"
    if class_count == 5:
        return "5class_sentiment_1_5"
    if class_count == 10:
        return "10class_sentiment_0_9"
    raise ValueError(f"unsupported class count: {class_count}")


def sentiment_label_token_descriptions(class_count: int) -> dict[str, str]:
    if class_count == 2:
        return {"0": "negative", "1": "positive"}
    if class_count == 5:
        return {
            str(score): description
            for score, description in zip(
                range(1, 6),
                [
                    "very_negative",
                    "negative",
                    "mixed_or_neutral",
                    "positive",
                    "very_positive",
                ],
            )
        }
    if class_count == 10:
        return {
            str(score): description
            for score, description in enumerate(
                [
                    "extremely_negative",
                    "very_negative",
                    "negative",
                    "somewhat_negative",
                    "slightly_negative",
                    "mixed_or_neutral",
                    "slightly_positive",
                    "somewhat_positive",
                    "positive",
                    "extremely_positive",
                ]
            )
        }
    raise ValueError(f"unsupported class count: {class_count}")


if __name__ == "__main__":
    main()
