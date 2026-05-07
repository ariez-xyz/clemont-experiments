#!/usr/bin/env python3
"""Interactive terminal browser for text monitor point/witness pairs."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import textwrap
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WIDTH = 100
PROMPT_MODES = ("none", "point", "both")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Browse a text monitor JSON in score order without loading the large "
            "HTML viewers."
        )
    )
    parser.add_argument("monitor_json", type=Path, help="Path to quant_run_*.json.")
    parser.add_argument(
        "--text-chars",
        type=int,
        default=2400,
        help="Maximum characters shown for point/witness text. Default: 2400.",
    )
    parser.add_argument(
        "--prompt-chars",
        type=int,
        default=4000,
        help="Maximum characters shown for each prompt. Default: 4000.",
    )
    parser.add_argument(
        "--no-prompts",
        action="store_true",
        help="Hide prompt blocks by default. Toggle with 't' while browsing.",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not clear the terminal between records.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    monitor_path = args.monitor_json.expanduser().resolve()
    payload = json.loads(monitor_path.read_text(encoding="utf-8"))
    records = normalize_records(payload.get("records", []))
    if not records:
        raise SystemExit(f"No records found in {monitor_path}")

    metadata = payload.get("metadata", {})
    input_csv = resolve_input_csv(metadata.get("input_csv"), monitor_path=monitor_path)
    frame = read_browser_csv(input_csv, metadata=metadata)
    pairs = sorted(
        [record for record in records if finite_float(record.get("max_ratio")) is not None],
        key=lambda record: finite_float(record.get("max_ratio")) or -math.inf,
        reverse=True,
    )
    if not pairs:
        raise SystemExit(f"No scored records found in {monitor_path}")

    state = BrowserState(
        monitor_path=monitor_path,
        input_csv=input_csv,
        frame=frame,
        pairs=pairs,
        prompt_mode="none" if args.no_prompts else "both",
        clear=not args.no_clear,
        text_chars=args.text_chars,
        prompt_chars=args.prompt_chars,
    )
    interactive_loop(state)


class BrowserState:
    def __init__(
        self,
        *,
        monitor_path: Path,
        input_csv: Path,
        frame: pd.DataFrame,
        pairs: list[dict[str, Any]],
        prompt_mode: str,
        clear: bool,
        text_chars: int,
        prompt_chars: int,
    ) -> None:
        self.monitor_path = monitor_path
        self.input_csv = input_csv
        self.frame = frame
        self.pairs = pairs
        self.rank = 0
        self.prompt_mode = prompt_mode
        self.clear = clear
        self.text_chars = text_chars
        self.prompt_chars = prompt_chars


def interactive_loop(state: BrowserState) -> None:
    while True:
        render_current(state)
        command = input("command [Enter/n next, p prev, j <rank>, id <example_id>, t prompts, q quit, h help]> ").strip()
        if command in {"", "n", "next"}:
            state.rank = min(state.rank + 1, len(state.pairs) - 1)
        elif command in {"p", "prev"}:
            state.rank = max(state.rank - 1, 0)
        elif command in {"q", "quit", "exit"}:
            return
        elif command in {"t", "prompt", "prompts"}:
            state.prompt_mode = next_prompt_mode(state.prompt_mode)
        elif command in {"h", "help", "?"}:
            print_help()
            input("press Enter to continue...")
        elif command.startswith("j "):
            jump_to_rank(state, command[2:].strip())
        elif command.startswith("id "):
            jump_to_example_id(state, command[3:].strip())
        else:
            print(f"Unknown command: {command!r}")
            input("press Enter to continue...")


def render_current(state: BrowserState) -> None:
    if state.clear:
        os.system("clear")

    width = terminal_width()
    record = state.pairs[state.rank]
    idx = record_index(record)
    witness_idx = record_witness_index(record)
    row = row_at(state.frame, idx)
    witness = row_at(state.frame, witness_idx) if witness_idx is not None else None

    print(f"{short_path(state.monitor_path)}")
    print(f"input: {short_path(state.input_csv)}")
    print(
        f"rank {state.rank + 1:,}/{len(state.pairs):,} | "
        f"point={idx} | witness={witness_idx if witness_idx is not None else '--'} | "
        f"loss={fmt(record.get('max_ratio'))} | "
        f"in={fmt(record.get('witness_in_distance'))} | "
        f"out={fmt(record.get('witness_out_distance'))} | "
        f"k={record.get('compared_count', record.get('compared', record.get('k', 'n/a')))} | "
        f"prompts={state.prompt_mode}"
    )
    print("=" * min(width, 120))
    print_card("POINT", row, width=width, text_chars=state.text_chars)
    if witness is not None:
        print_card("WITNESS", witness, width=width, text_chars=state.text_chars)
    else:
        print("\nWITNESS\n  n/a")

    if state.prompt_mode in {"point", "both"}:
        print_prompt_block("POINT PROMPT", prompt_text(row), width=width, max_chars=state.prompt_chars)
        if state.prompt_mode == "both" and witness is not None:
            print_prompt_block(
                "WITNESS PROMPT",
                prompt_text(witness),
                width=width,
                max_chars=state.prompt_chars,
            )


def print_card(title: str, row: pd.Series, *, width: int, text_chars: int) -> None:
    print(f"\n{title}")
    header = []
    example_id = get_value(row, "example_id")
    if example_id:
        header.append(f"id={example_id}")
    true_label = true_label_text(row)
    if true_label:
        header.append(f"true={true_label}")
    answer = get_value(row, "judge_answer") or get_value(row, "first_token")
    if answer:
        header.append(f"sampled={answer}")
    if "revision_applied" in row.index and not is_missing(row.get("revision_applied")):
        header.append(f"revision_applied={bool_value(row.get('revision_applied'))}")
    if header:
        print("  " + " | ".join(header))

    text = truncate(text_for_row(row), text_chars)
    print(indent_wrapped(text, width=width, prefix="  "))
    probs = probability_values(row)
    if probs:
        print_probability_bars(probs, width=width)
    baseline = baseline_probability_values(row)
    if baseline:
        print("  baseline:")
        print_probability_bars(baseline, width=width, prefix="    ")


def print_probability_bars(
    probs: Sequence[tuple[str, float]],
    *,
    width: int,
    prefix: str = "  ",
) -> None:
    label_width = max(len(label) for label, _ in probs)
    bar_width = max(10, min(42, width - len(prefix) - label_width - 18))
    for label, value in probs:
        filled = int(round(value * bar_width))
        bar = "#" * filled + "-" * (bar_width - filled)
        print(f"{prefix}{label:>{label_width}} [{bar}] {value:.4f}")


def print_prompt_block(title: str, prompt: str, *, width: int, max_chars: int) -> None:
    print(f"\n{title}")
    if not prompt:
        print("  n/a")
        return
    body = truncate(prompt, max_chars)
    print("```")
    print(indent_wrapped(body, width=width, prefix="", subsequent_prefix=""))
    print("```")


def print_help() -> None:
    print(
        "\nCommands:\n"
        "  Enter, n          next highest-loss pair\n"
        "  p                 previous pair\n"
        "  j <rank>          jump to 1-based score rank, e.g. j 25\n"
        "  id <example_id>   jump to first pair whose point or witness has that id\n"
        "  t                 cycle prompts: none -> point -> point+witness\n"
        "  q                 quit\n"
    )


def next_prompt_mode(current: str) -> str:
    try:
        idx = PROMPT_MODES.index(current)
    except ValueError:
        return "none"
    return PROMPT_MODES[(idx + 1) % len(PROMPT_MODES)]


def jump_to_rank(state: BrowserState, raw: str) -> None:
    try:
        rank = int(raw.replace(",", ""))
    except ValueError:
        print(f"Invalid rank: {raw!r}")
        input("press Enter to continue...")
        return
    state.rank = max(0, min(rank - 1, len(state.pairs) - 1))


def jump_to_example_id(state: BrowserState, example_id: str) -> None:
    if not example_id:
        return
    for rank, record in enumerate(state.pairs):
        idx = record_index(record)
        witness_idx = record_witness_index(record)
        row = row_at(state.frame, idx)
        witness = row_at(state.frame, witness_idx) if witness_idx is not None else None
        if get_value(row, "example_id") == example_id:
            state.rank = rank
            return
        if witness is not None and get_value(witness, "example_id") == example_id:
            state.rank = rank
            return
    print(f"No point/witness example_id found: {example_id}")
    input("press Enter to continue...")


def read_browser_csv(path: Path, *, metadata: Mapping[str, Any]) -> pd.DataFrame:
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
        "prompt",
        "revision_prompt",
        "revision_selected",
        "revision_applied",
        "revision_error",
    }
    desired.update(col for col in header if re.fullmatch(r"prob_\d+", col))
    desired.update(col for col in header if re.fullmatch(r"baseline_prob_\d+", col))
    desired.update(str(col) for col in metadata.get("probability_columns", []) if col in header)
    usecols = [col for col in header if col in desired]
    return pd.read_csv(path, usecols=usecols, engine="python")


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


def resolve_input_csv(value: Any, *, monitor_path: Path) -> Path:
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
        candidates.extend([monitor_path.parent / raw, REPO_ROOT / raw, Path.cwd() / raw])
    candidates.append(REPO_ROOT / raw.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"{monitor_path}: could not resolve input CSV {value!r}")


def row_at(frame: pd.DataFrame, idx: int | None) -> pd.Series | None:
    if idx is None or idx < 0 or idx >= len(frame):
        return None
    return frame.iloc[idx]


def record_index(record: Mapping[str, Any]) -> int:
    return int(record.get("index", record.get("point_id", -1)))


def record_witness_index(record: Mapping[str, Any]) -> int | None:
    value = record.get("witness_id")
    if value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def text_for_row(row: pd.Series | None) -> str:
    if row is None:
        return "n/a"
    if has_column(row, "review_text"):
        title = get_value(row, "review_title")
        text = get_value(row, "review_text")
        return f"{title}: {text}" if title else text
    if has_column(row, "user_input"):
        return get_value(row, "user_input")
    return "n/a"


def prompt_text(row: pd.Series | None) -> str:
    if row is None:
        return ""
    revision = get_value(row, "revision_prompt")
    if revision:
        return revision
    return get_value(row, "prompt")


def true_label_text(row: pd.Series | None) -> str:
    if row is None:
        return ""
    rating = get_value(row, "rating_value")
    if rating:
        try:
            as_float = float(rating)
            if as_float.is_integer():
                return f"{int(as_float)} stars"
        except ValueError:
            pass
        return str(rating)
    toxicity = get_value(row, "toxicity_label")
    if toxicity:
        return "toxic" if str(toxicity) in {"1", "1.0", "True", "true"} else "safe"
    return ""


def probability_values(row: pd.Series | None) -> list[tuple[str, float]]:
    return prefixed_probability_values(row, "prob_")


def baseline_probability_values(row: pd.Series | None) -> list[tuple[str, float]]:
    return prefixed_probability_values(row, "baseline_prob_")


def prefixed_probability_values(row: pd.Series | None, prefix: str) -> list[tuple[str, float]]:
    if row is None:
        return []
    values: list[tuple[str, float]] = []
    for column in row.index:
        if not re.fullmatch(re.escape(prefix) + r"\d+", str(column)):
            continue
        value = finite_float(row.get(column))
        if value is not None:
            values.append((str(column).removeprefix(prefix), value))
    values.sort(key=lambda item: int(item[0]))
    return values


def get_value(row: pd.Series | None, column: str) -> str:
    if row is None or column not in row.index:
        return ""
    value = row.get(column)
    if is_missing(value):
        return ""
    return str(value)


def has_column(row: pd.Series | None, column: str) -> bool:
    return row is not None and column in row.index


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "1.0", "yes"}


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return value is None


def fmt(value: Any) -> str:
    number = finite_float(value)
    if number is None:
        return "n/a"
    if abs(number) < 1e-6:
        number = 0.0
    if abs(number) >= 1000 or (0 < abs(number) < 1e-4):
        return f"{number:.4e}"
    return f"{number:.4f}".rstrip("0").rstrip(".")


def truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def indent_wrapped(
    text: str,
    *,
    width: int,
    prefix: str,
    subsequent_prefix: str | None = None,
) -> str:
    subsequent = prefix if subsequent_prefix is None else subsequent_prefix
    wrapped_lines: list[str] = []
    body_width = max(40, width - len(prefix) - 2)
    for line in text.splitlines() or [""]:
        if not line:
            wrapped_lines.append(prefix)
            continue
        wrapped_lines.extend(
            textwrap.wrap(
                line,
                width=body_width,
                initial_indent=prefix,
                subsequent_indent=subsequent,
                break_long_words=False,
                replace_whitespace=False,
            )
        )
    return "\n".join(wrapped_lines)


def terminal_width() -> int:
    return max(72, shutil.get_terminal_size((DEFAULT_WIDTH, 24)).columns)


def short_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
