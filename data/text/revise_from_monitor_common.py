"""Shared utilities for monitor-informed LLM-judge revision passes."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from openrouter_client import OpenRouterClient


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PREFIX_COLUMNS = (
    "judge_answer",
    "first_token",
    "first_token_logprob",
    "top_logprobs_json",
    "label_logprob_floor",
    "judge_finish_reason",
    "judge_model_returned",
    "judge_response_id",
    "judge_error",
)


def load_monitor_payload(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"monitor JSON must contain an object: {path}")
    if not payload.get("records"):
        raise ValueError(f"monitor JSON has no records: {path}")
    return payload


def resolve_input_csv(monitor_path: Path, payload: Mapping[str, Any]) -> Path:
    metadata = payload.get("metadata") or {}
    raw = metadata.get("input_csv")
    if not raw:
        raise ValueError("monitor JSON metadata.input_csv is missing")

    candidate = Path(str(raw))
    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.extend(
            [
                REPO_ROOT / candidate,
                monitor_path.parent / candidate,
                Path.cwd() / candidate,
            ]
        )
    for item in candidates:
        if item.exists():
            return item.resolve()
    tried = ", ".join(str(item) for item in candidates)
    raise FileNotFoundError(f"could not resolve input CSV from monitor JSON; tried {tried}")


def probability_columns(frame: pd.DataFrame) -> list[str]:
    columns = [col for col in frame.columns if re.fullmatch(r"prob_\d+", col)]
    columns.sort(key=lambda col: int(col.removeprefix("prob_")))
    if not columns:
        raise ValueError("input CSV has no probability columns prob_0/prob_1/...")
    return columns


def label_tokens_from_probability_columns(prob_columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(col.removeprefix("prob_") for col in prob_columns)


def class_count_from_probability_columns(prob_columns: Sequence[str]) -> int:
    return len(prob_columns)


def select_revision_indices(
    records: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    top_k: int | None,
    top_fraction: float,
    min_robustness_loss: float | None,
    require_witness: bool,
) -> set[int]:
    scored: list[tuple[int, float]] = []
    for record in records:
        idx = int(record.get("index", record.get("point_id", -1)))
        score = finite_float(record.get("max_ratio"))
        witness_id = record.get("witness_id")
        if idx < 0 or score is None:
            continue
        if require_witness and witness_id is None:
            continue
        if min_robustness_loss is not None and score < min_robustness_loss:
            continue
        scored.append((idx, score))

    if mode == "all":
        return {idx for idx, _ in scored}

    scored.sort(key=lambda item: item[1], reverse=True)
    if mode == "top-k":
        if top_k is None or top_k <= 0:
            raise ValueError("--top-k must be positive with --revision-mode top-k")
        return {idx for idx, _ in scored[:top_k]}

    if mode == "top-fraction":
        if top_fraction <= 0 or top_fraction > 1:
            raise ValueError("--top-fraction must be in (0, 1]")
        count = max(1, math.ceil(len(scored) * top_fraction)) if scored else 0
        return {idx for idx, _ in scored[:count]}

    raise ValueError(f"unsupported revision mode: {mode}")


def apply_revisions(
    frame: pd.DataFrame,
    *,
    records: Sequence[Mapping[str, Any]],
    selected_indices: set[int],
    prompts_by_index: Mapping[int, str],
    judge_results_by_index: Mapping[int, Any],
    label_tokens: Sequence[str],
) -> pd.DataFrame:
    revised = frame.copy()
    active_columns = list(BASELINE_PREFIX_COLUMNS)
    active_columns.extend(f"prob_{label}" for label in label_tokens)
    active_columns.extend(f"logprob_{label}" for label in label_tokens)
    active_columns.extend(f"logprob_{label}_source" for label in label_tokens)
    for column in active_columns:
        if column in revised.columns and f"baseline_{column}" not in revised.columns:
            revised[f"baseline_{column}"] = revised[column]
        if column in revised.columns and (
            column in BASELINE_PREFIX_COLUMNS or column.endswith("_source")
        ):
            revised[column] = revised[column].astype("object")

    record_by_index = {
        int(record.get("index", record.get("point_id", -1))): record for record in records
    }
    revised["revision_selected"] = False
    revised["revision_applied"] = False
    revised["revision_prompt"] = None
    revised["revision_prompt_hash"] = None
    revised["revision_error"] = None
    revised["baseline_robustness_loss"] = None
    revised["baseline_witness_id"] = None
    revised["baseline_witness_in_distance"] = None
    revised["baseline_witness_out_distance"] = None

    for idx, record in record_by_index.items():
        if 0 <= idx < len(revised):
            revised.at[idx, "baseline_robustness_loss"] = finite_float(record.get("max_ratio"))
            revised.at[idx, "baseline_witness_id"] = record.get("witness_id")
            revised.at[idx, "baseline_witness_in_distance"] = finite_float(
                record.get("witness_in_distance")
            )
            revised.at[idx, "baseline_witness_out_distance"] = finite_float(
                record.get("witness_out_distance")
            )

    for idx in sorted(selected_indices):
        if idx < 0 or idx >= len(revised):
            continue
        judge = judge_results_by_index.get(idx)
        if judge is None:
            continue
        columns = OpenRouterClient.judge_result_to_columns(judge)
        revised.at[idx, "revision_selected"] = True
        if not revision_is_monitorable(columns, label_tokens):
            revised.at[idx, "revision_error"] = columns.get(
                "judge_error",
                "revision did not return monitorable label probabilities",
            )
            prompt = prompts_by_index.get(idx)
            if prompt is not None:
                import hashlib

                revised.at[idx, "revision_prompt"] = prompt
                revised.at[idx, "revision_prompt_hash"] = hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest()
            continue

        for column, value in columns.items():
            revised.at[idx, column] = value
        revised.at[idx, "revision_applied"] = True
        prompt = prompts_by_index.get(idx)
        if prompt is not None:
            import hashlib

            revised.at[idx, "revision_prompt"] = prompt
            revised.at[idx, "revision_prompt_hash"] = hashlib.sha256(
                prompt.encode("utf-8")
            ).hexdigest()

    return revised


def revision_is_monitorable(
    columns: Mapping[str, Any],
    label_tokens: Sequence[str],
) -> bool:
    return all(
        finite_float(columns.get(f"prob_{label}")) is not None for label in label_tokens
    )


def run_revision_judging(
    *,
    client: OpenRouterClient,
    prompts_by_index: Mapping[int, str],
    label_tokens: Sequence[str],
    system_prompt: str,
) -> dict[int, Any]:
    ordered = sorted(prompts_by_index)
    prompts = [prompts_by_index[idx] for idx in ordered]
    results = client.judge_prompts(
        prompts,
        label_tokens=tuple(label_tokens),
        system_prompt=system_prompt,
    )
    return dict(zip(ordered, results))


def write_revised_output(
    *,
    client: OpenRouterClient,
    frame: pd.DataFrame,
    input_csv: Path,
    output_csv: Path | None,
    output_json: Path | None,
    metadata: Mapping[str, Any],
) -> tuple[Path, Path]:
    if output_csv is None and output_json is None:
        output_csv = input_csv.with_name(f"{input_csv.stem}_witness_revised.csv")
        output_json = output_csv.with_suffix(".json")
    elif output_csv is None:
        output_csv = output_json.with_suffix(".csv")  # type: ignore[union-attr]
    elif output_json is None:
        output_json = output_csv.with_suffix(".json")

    assert output_csv is not None
    assert output_json is not None
    return client.write_dataset_output(
        json_path=output_json,
        csv_path=output_csv,
        frame=frame,
        metadata=metadata,
    )


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def format_distribution(row: pd.Series, prob_columns: Sequence[str]) -> str:
    parts = []
    for column in prob_columns:
        label = column.removeprefix("prob_")
        value = finite_float(row.get(column))
        if value is None:
            parts.append(f"{label}: n/a")
        else:
            parts.append(f"{label}: {value:.4g}")
    return ", ".join(parts)


def sampled_token(row: pd.Series) -> str:
    value = row.get("judge_answer")
    if pd.isna(value):
        return "n/a"
    return str(value)


def build_prompt_map(
    *,
    frame: pd.DataFrame,
    records: Sequence[Mapping[str, Any]],
    selected_indices: set[int],
    prob_columns: Sequence[str],
    prompt_builder: Callable[[pd.Series, pd.Series | None, Mapping[str, Any], Sequence[str]], str],
) -> dict[int, str]:
    prompts: dict[int, str] = {}
    record_by_index = {
        int(record.get("index", record.get("point_id", -1))): record for record in records
    }
    for idx in sorted(selected_indices):
        if idx < 0 or idx >= len(frame):
            continue
        record = record_by_index.get(idx)
        if record is None:
            continue
        witness_row = None
        witness_id = record.get("witness_id")
        if witness_id is not None:
            witness_idx = int(witness_id)
            if 0 <= witness_idx < len(frame):
                witness_row = frame.iloc[witness_idx]
        prompts[idx] = prompt_builder(frame.iloc[idx], witness_row, record, prob_columns)
    return prompts
