"""Prepare Amazon review sentiment data for Clemont monitoring.

This script samples reviews, asks an OpenRouter-hosted LLM to judge binary
sentiment, embeds the exact judge prompt, and writes a monitor-ready CSV plus a
JSON manifest.
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
            "JSON manifest path. Default: timestamped "
            "amazon_sentiment_openrouter_YYYYMMDDTHHMMSSZ.json next to this script."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Monitor-ready CSV path. Default: same stem as --output-json with .csv suffix.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_json = args.output_json or default_output_json(run_timestamp)
    output_csv = args.output_csv or output_json.with_suffix(".csv")

    reviews = load_reviews(INPUT_CSV)
    sample_size = min(args.sample_size, len(reviews))
    sampled = (
        reviews.sample(n=sample_size, random_state=args.seed)
        .reset_index(drop=False)
        .rename(columns={"index": "source_row"})
    )

    prompts = [
        build_sentiment_prompt(row, max_chars=args.max_review_chars)
        for _, row in sampled.iterrows()
    ]

    client = OpenRouterClient(
        temperature=args.temperature,
        top_p=args.top_p,
        max_workers=args.max_workers,
    )

    print(f"Judging {len(prompts)} Amazon reviews with {client.chat_model}...")
    judge_results = client.judge_prompts(
        prompts,
        label_tokens=("0", "1"),
        system_prompt=(
            "You are a sentiment classifier. Follow the requested output format exactly."
        ),
    )

    print(f"Embedding {len(prompts)} prompts with {client.embedding_model}...")
    embeddings = client.embed_texts(prompts, batch_size=args.embedding_batch_size)

    frame = build_output_frame(sampled, prompts, judge_results, embeddings)
    metadata: dict[str, Any] = {
        "dataset": "amazon_reviews",
        "task": "binary_sentiment",
        "input_csv": str(INPUT_CSV),
        "sample_size_requested": args.sample_size,
        "sample_size_actual": sample_size,
        "judge_error_count": int(frame["judge_error"].notna().sum()),
        "seed": args.seed,
        "max_review_chars": args.max_review_chars,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_timestamp": run_timestamp,
        "label_tokens": {"0": "negative", "1": "positive"},
        "embedding_columns": embedding_columns(frame),
        "output_columns": ["prob_0", "prob_1"],
        "score_columns": ["logprob_0", "logprob_1"],
        "score_source_columns": ["logprob_0_source", "logprob_1_source"],
        "score_type": "first_token_top_logprobs",
        "score_inference": (
            "missing binary label logprob is filled with the minimum returned "
            "first-token top_logprobs value when exactly one label is present"
        ),
        "prompt_embedding": "exact_prompt_sent_to_judge",
    }
    client.write_dataset_output(
        json_path=output_json,
        csv_path=output_csv,
        frame=frame,
        metadata=metadata,
    )
    print(f"Wrote {output_json}")
    print(f"Wrote {output_csv}")


def default_output_json(timestamp: str) -> Path:
    return Path(__file__).with_name(f"amazon_sentiment_openrouter_{timestamp}.json")


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


def build_sentiment_prompt(row: pd.Series, *, max_chars: int) -> str:
    title = clean_text(row.get("Review Title", ""))
    text = clean_text(row.get("Review Text", ""))
    review = f"Title: {title}\n\nReview: {text}" if title else f"Review: {text}"
    review = review[:max_chars]
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
    missing_probs = frame[["prob_0", "prob_1"]].isna().any(axis=1).sum()
    if missing_probs:
        print(
            f"Warning: {missing_probs} rows are missing one or both label probabilities "
            "because the labels were absent from top_logprobs."
        )
    return frame


def embedding_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if re.fullmatch(r"e\d+", col)]


if __name__ == "__main__":
    main()
