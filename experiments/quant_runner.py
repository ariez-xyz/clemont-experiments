"""Quantitative monitor demonstration for datasets with feature/probability columns.

Streams paired feature/probability rows through Clemont's quantitative monitor,
prints every Nth observation (including the final one) with formatted point and
witness data, and concludes with percentile summaries plus representative
samples covering low/medium/high neighbour counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
import sys
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Literal

import numpy as np

from clemont.frnn import KdTreeFRNN, FaissFRNN
from clemont.quantitative_monitor import QuantitativeMonitor, QuantitativeResult
from clemont.monitor import Monitor, ObservationResult

def _csv_list(value: str) -> Tuple[str, ...]:
    items = [item.strip() for item in value.split(",")]
    return tuple(item for item in items if item)


def _csv_float_list(value: str) -> Tuple[float, ...]:
    items = [item.strip() for item in value.split(",")]
    floats: List[float] = []
    for item in items:
        if not item:
            continue
        try:
            floats.append(float(item))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid float value '{item}' in list") from exc
    if not floats:
        raise argparse.ArgumentTypeError("Expected at least one float in list")
    return tuple(floats)


def _csv_int_set(value: str) -> Optional[set[int]]:
    stripped = value.strip()
    if not stripped:
        return None
    selected: set[int] = set()
    for token in stripped.split(","):
        cleaned = token.strip()
        if not cleaned:
            continue
        try:
            selected.add(int(cleaned))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid row id '{cleaned}' in --row-ids") from exc
    if not selected:
        raise argparse.ArgumentTypeError("--row-ids parsed to an empty set of ids")
    return selected


def _optional_path(value: str) -> Optional[Path]:
    lowered = value.strip().lower()
    if lowered in {"", "none", "null"}:
        return None
    return Path(value)


def _parse_walltime(value: str) -> float:
    stripped = value.strip()
    if not stripped:
        raise argparse.ArgumentTypeError("--walltime expects HH:MM:SS, MM:SS, or SS format")
    parts = stripped.split(":")
    if len(parts) > 3:
        raise argparse.ArgumentTypeError("--walltime accepts at most three colon-separated components")
    try:
        components = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid numeric value in --walltime: '{value}'") from exc
    if any(component < 0 for component in components):
        raise argparse.ArgumentTypeError("--walltime components must be non-negative")
    if len(components) == 1:
        total_seconds = components[0]
    elif len(components) == 2:
        minutes, seconds = components
        total_seconds = minutes * 60 + seconds
    else:
        hours, minutes, seconds = components
        total_seconds = hours * 3600 + minutes * 60 + seconds
    if total_seconds <= 0:
        raise argparse.ArgumentTypeError("--walltime must be greater than zero")
    return float(total_seconds)


def _format_seconds(total_seconds: float) -> str:
    total = int(total_seconds)
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


@dataclass
class Config:
    input_csv: Path = Path(__file__).parent.parent / "data" / "toydata" / "inputs_numeric.csv"
    preds_csv: Optional[Path] = Path(__file__).parent.parent / "data" / "toydata" / "predictions_with_probs.csv"
    results_dir: Path = Path(__file__).parent.parent / "results" / "quantitative"
    input_columns: Tuple[str, ...] | None = None
    pred_columns: Tuple[str, ...] | None = None
    ignore_columns: Tuple[str, ...] = ("row_id",)
    frnn_metric: Literal["linf", "l1", "l2", "tv", "cosine"] = "l2"
    out_metric: Literal["linf", "l1", "l2", "tv", "cosine"] = "tv"
    output_transform: Literal["probs", "argmax-normalized"] = "probs"
    backend: Literal["kdtree", "faiss"] = "kdtree"
    display_stride: int = 1000
    frnn_threads: int = 4
    input_exponent: float = 1
    discount_factor: float = 1.0
    batchsize: int = 1000
    initial_k: int = 16
    max_k: Optional[int] = None
    max_rows: Optional[int] = None
    interpolate_csv: Optional[Path] = None
    interpolate_bins: Optional[int] = None
    interpolate_weights: Optional[Tuple[float, ...]] = None
    deduplicate: bool = False
    save_points: bool = False
    normalize: bool = False
    static: bool = False
    shuffle: bool = False
    epsilon: Optional[float] = None
    row_ids: Optional[set[int]] = None
    k_grow_factor: Optional[float] = None
    walltime_seconds: Optional[float] = None


def main() -> None:
    cfg = parse_args()
    inputs, probs, fair_probs, input_names, prob_names, raw_prob_names = load_data(cfg)
    num_points = inputs.shape[0]
    probs_run = probs
    if fair_probs is not None:
        if num_points > 1:
            if cfg.interpolate_bins:
                bins = cfg.interpolate_bins
                if cfg.interpolate_weights is not None:
                    weights = np.asarray(cfg.interpolate_weights, dtype=float)
                else:
                    weights = np.linspace(0.0, 1.0, bins)
                bin_idx = (np.arange(num_points) * bins) // num_points
                blends = weights[bin_idx]
            else:
                blends = np.linspace(0.0, 1.0, num_points)
        else:
            blends = np.array([0.0])
        probs_run = (1.0 - blends[:, None]) * probs + blends[:, None] * fair_probs

    monitor_outputs, output_names, output_transform_metadata = transform_outputs(
        probs_run,
        prob_names,
        raw_prob_names,
        cfg.output_transform,
    )

    if cfg.backend == "kdtree":
        backend_factory = lambda: KdTreeFRNN(
            metric=cfg.frnn_metric,
            batchsize=cfg.batchsize,
            epsilon=cfg.epsilon,
        )
    elif cfg.backend == "faiss":
        backend_factory = lambda: FaissFRNN(
            metric=cfg.frnn_metric,
            epsilon=cfg.epsilon,
        )
    else:
        raise ValueError(f"unknown backend {cfg.backend}")

    monitor = QuantitativeMonitor(
        backend_factory,
        out_metric=cfg.out_metric,
        initial_k=cfg.initial_k,
        max_k=cfg.max_k,
        input_exponent=cfg.input_exponent,
        discount_factor=cfg.discount_factor,
        k_grow_factor=cfg.k_grow_factor,
    )

    epsilon_monitor: Optional[Monitor] = None
    if cfg.epsilon:
        epsilon_monitor = Monitor(backend_factory)

    if cfg.static:
        monitor.batch_add(zip(inputs, monitor_outputs))

    full_records: List[
        Tuple[
            QuantitativeResult,
            np.ndarray,
            np.ndarray,
            float,
            Optional[ObservationResult],
            Optional[float],
        ]
    ] = []
    walltime_reached = False
    walltime_deadline = None
    if cfg.walltime_seconds is not None:
        walltime_deadline = time.time() + cfg.walltime_seconds

    print("=== Streaming quantitative monitoring demo ===")
    print(
        f"Points={num_points}, input-dim={inputs.shape[1]}, output-dim={monitor_outputs.shape[1]}, "
        f"FRNN metric={cfg.frnn_metric}, output metric={cfg.out_metric}"
    )
    print(f"Every {cfg.display_stride}th observation (• denotes early stop via bound):")

    display_indices = set(range(0, num_points, cfg.display_stride))
    display_indices.add(num_points - 1)

    total_time = 0
    total_epsilon_time = 0.0
    epsilon_timings: List[float] = []

    try:
        for idx, (x_vec, y_vec, p_vec) in enumerate(zip(inputs, monitor_outputs, probs_run)):
            start_time = time.time()
            res = monitor.observe(x_vec, y_vec, dry_run=cfg.static)
            iter_time = (time.time() - start_time) * 1000

            eps_res: Optional[ObservationResult] = None
            eps_time_ms: Optional[float] = None
            if epsilon_monitor:
                eps_start = time.time()
                eps_res = epsilon_monitor.observe(x_vec, int(np.argmax(p_vec)))
                eps_time_ms = (time.time() - eps_start) * 1000
                total_epsilon_time += eps_time_ms
                epsilon_timings.append(eps_time_ms)

            full_records.append((res, x_vec, y_vec, iter_time, eps_res, eps_time_ms))
            total_time += iter_time

            if idx in display_indices:
                _print_observation(
                    idx,
                    res,
                    x_vec,
                    y_vec,
                    inputs,
                    monitor_outputs,
                    output_names,
                    input_names,
                )

            if walltime_deadline is not None and time.time() >= walltime_deadline:
                walltime_reached = True
                print(
                    f"\nReached walltime limit ({_format_seconds(cfg.walltime_seconds)}) after processing {idx + 1} rows."
                )
                break
    except KeyboardInterrupt:
        pass

    ratios = np.array([rec[0].max_ratio for rec in full_records], dtype=float)
    compared = np.array([rec[0].compared_count for rec in full_records], dtype=int)

    if len(full_records) == 0:
        print("No records processed; exiting early.")
        return

    early_stop = sum(rec[0].stopped_by_bound for rec in full_records)
    max_depth = max((rec[0].k_progression[-1] for rec in full_records if rec[0].k_progression), default=0)

    print("\n=== Summary ===")
    print(f"completed in {round(total_time/1000, 2)}s")
    print(f"Processed {len(full_records)} / {num_points} rows")
    if walltime_reached and cfg.walltime_seconds is not None:
        print(f"Stopped early due to walltime limit of {_format_seconds(cfg.walltime_seconds)}")
    if epsilon_monitor and epsilon_timings:
        avg_eps = total_epsilon_time / len(epsilon_timings)
        print(
            f"Epsilon monitor: total {round(total_epsilon_time/1000, 2)}s, "
            f"avg {round(avg_eps, 3)}ms over {len(epsilon_timings)} observations"
        )
    finite_mask = np.isfinite(ratios)
    if finite_mask.any():
        _print_percentiles("Ratio", ratios[finite_mask])
    else:
        print("Ratio     : all observations produced infinite ratios")

    _print_percentiles("Compared", compared.astype(float))
    print(f"Infinities : {(~finite_mask).sum()} occurrences")
    print(f"Early stops: {early_stop} / {num_points} observations")
    print(f"Largest k  : {max_depth}")

    # Representative examples across neighbour counts
    print("\nSample observations by neighbour count:")
    sorted_indices = np.argsort(compared)
    l = len(sorted_indices)
    sample_positions = [0, max(l // 2, 0), l - 1]
    labels = ["low-k", "mid-k", "high-k"]
    seen: set[int] = set()
    for label, pos in zip(labels, sample_positions):
        idx = int(sorted_indices[pos])
        if idx in seen:
            continue
        seen.add(idx)
        res, x_vec, p_vec, iter_time, eps_res, eps_time = full_records[idx]
        print(f"-- {label} (index {idx}, compared={res.compared_count} in {iter_time})")
        _print_observation(
            idx,
            res,
            x_vec,
            p_vec,
            inputs,
            monitor_outputs,
            output_names,
            input_names,
        )

    # Random sample from high-ratio tail (>= 90th percentile)
    high_ratio_candidates: List[int] = []
    if finite_mask.any():
        ratio_threshold = float(np.percentile(ratios[finite_mask], 90))
        for idx, value in enumerate(ratios):
            if value >= ratio_threshold:
                high_ratio_candidates.append(idx)
    else:
        # All ratios are infinite; include every observation
        high_ratio_candidates = list(range(len(ratios)))

    if high_ratio_candidates:
        print("\nRandom sample of high-ratio observations (>= 90th percentile):")
        sample_size = min(3, len(high_ratio_candidates))
        sampled_indices = sorted(random.sample(high_ratio_candidates, sample_size))
        for idx in sampled_indices:
            res, x_vec, p_vec, iter_time, eps_res, eps_time = full_records[idx]
            _print_observation(
                idx,
                res,
                x_vec,
                p_vec,
                inputs,
                monitor_outputs,
                output_names,
                input_names,
            )
    else:
        print("\nNo observations qualified for the high-ratio sample.")

    output_path = save_results_json(
        cfg,
        inputs,
        monitor_outputs,
        full_records,
        input_names,
        output_names,
        output_transform_metadata,
        total_time,
        total_epsilon_time,
        len(epsilon_timings),
    )
    print(f"\nSaved run to {output_path}")


def parse_args() -> Config:
    defaults = Config()
    parser = argparse.ArgumentParser(description="Run Clemont's quantitative monitor on CSV data.")

    parser.add_argument("--input-csv", dest="input_csv", type=Path, default=argparse.SUPPRESS,
                        help=f"Path to feature CSV (default: {defaults.input_csv})")
    parser.add_argument(
        "--preds-csv",
        dest="preds_csv",
        type=_optional_path,
        default=argparse.SUPPRESS,
        help=(
            "Path to probabilities CSV. Use 'none' to indicate the columns live in the input file. "
            f"Default: {defaults.preds_csv}"
        ),
    )
    parser.add_argument("--results-dir", dest="results_dir", type=Path, default=argparse.SUPPRESS,
                        help=f"Directory for JSON outputs (default: {defaults.results_dir})")
    parser.add_argument(
        "--input-cols",
        dest="input_columns",
        type=_csv_list,
        default=argparse.SUPPRESS,
        help="Comma-separated feature columns to load (default: all, minus ignored/prediction columns)",
    )
    parser.add_argument(
        "--pred-cols",
        dest="pred_columns",
        type=_csv_list,
        default=argparse.SUPPRESS,
        help="Comma-separated probability columns (required when using a single CSV without prob_* columns)",
    )
    parser.add_argument(
        "--ignore-cols",
        dest="ignore_columns",
        type=_csv_list,
        default=argparse.SUPPRESS,
        help=f"Comma-separated column names to ignore across inputs/preds (default: {','.join(defaults.ignore_columns)})",
    )
    parser.add_argument(
        "--frnn-metric",
        dest="frnn_metric",
        choices=["linf", "l1", "l2", "tv", "cosine"],
        default=argparse.SUPPRESS,
        help=f"FRNN metric (default: {defaults.frnn_metric})",
    )
    parser.add_argument(
        "--out-metric",
        dest="out_metric",
        choices=["linf", "l1", "l2", "tv", "cosine"],
        default=argparse.SUPPRESS,
        help=f"Output metric (default: {defaults.out_metric})",
    )
    parser.add_argument(
        "--output-transform",
        dest="output_transform",
        choices=["probs", "argmax-normalized"],
        default=argparse.SUPPRESS,
        help=(
            "Transform probability columns before quantitative monitoring. "
            "'probs' uses the distribution directly; 'argmax-normalized' uses "
            "the normalized ordinal argmax label as a one-dimensional output."
            "For example, with 5 class probabilities, 'probs' treats this as 5D output,"
            "argmax-normalized takes the argmax and maps it to [0,1], if class 3"
            "is the most likely then output is taken to be 0.75"
        ),
    )
    parser.add_argument("--display-stride", dest="display_stride", type=int, default=argparse.SUPPRESS,
                        help=f"Print every Nth observation (default: {defaults.display_stride})")
    parser.add_argument("--backend", dest="backend", type=str, default=argparse.SUPPRESS,
                        help=f"kNN backend to use (default: {defaults.backend})")
    parser.add_argument("--frnn-threads", dest="frnn_threads", type=int, default=argparse.SUPPRESS,
                        help=f"Thread hint for FRNN backends (default: {defaults.frnn_threads})")
    parser.add_argument("--input-exponent", dest="input_exponent", type=float, default=argparse.SUPPRESS,
                        help=f"Input exponent for monitor (default: {defaults.input_exponent})")
    parser.add_argument("--discount-factor", dest="discount_factor", type=float, default=argparse.SUPPRESS,
                        help=f"Discount factor for monitor (default: {defaults.discount_factor})")
    parser.add_argument("--batchsize", dest="batchsize", type=int, default=argparse.SUPPRESS,
                        help=f"Batch size for batched kNN backends (default: {defaults.batchsize})")
    parser.add_argument("--initial-k", dest="initial_k", type=int, default=argparse.SUPPRESS,
                        help=f"Initial k value for repeated kNN queries (default: {defaults.initial_k})")
    parser.add_argument("--normalize", dest="normalize", action="store_true",
                        help=f"Normalize input columns to length 1 in the L2 norm (default: {defaults.normalize})")
    parser.add_argument("--save-points", dest="save_points", action="store_true",
                        help=f"Whether to write raw input and output points to .json log (default: {defaults.save_points})")
    parser.add_argument("--shuffle", dest="shuffle", action="store_true",
                        help=f"shuffle the data before run to compute (seed 42, default: {defaults.shuffle})")
    parser.add_argument("--static", dest="static", action="store_true",
                        help=f"preloads the data before run to compute (default: {defaults.static})")
    parser.add_argument("--epsilon", dest="epsilon", type=float, default=argparse.SUPPRESS,
                        help=f"epsilon value. If not None, also computes epsilon-monitor results (default: {defaults.epsilon})")
    parser.add_argument("--k-grow-factor", dest="k_grow_factor", type=float, default=argparse.SUPPRESS,
                        help=f"Grow factor for k on repeated kNN queries (default: {defaults.k_grow_factor})")
    parser.add_argument(
        "--row-ids",
        dest="row_ids",
        type=_csv_int_set,
        default=argparse.SUPPRESS,
        help="Comma-separated row identifiers to include (1-based). When provided, only matching rows are loaded.",
    )
    parser.add_argument(
        "--walltime",
        dest="walltime_seconds",
        type=_parse_walltime,
        default=argparse.SUPPRESS,
        help="Stop after the given HH:MM:SS, MM:SS, or SS duration (graceful early exit).",
    )
    parser.add_argument(
        "--max-k",
        dest="max_k",
        type=int,
        default=argparse.SUPPRESS,
        help="Optional cap for repeated kNN queries (default: no cap)",
    )
    parser.add_argument(
        "--max-n",
        dest="max_rows",
        type=int,
        default=argparse.SUPPRESS,
        help=(
            "Maximum number of rows to load from the CSVs. "
            f"Default: {defaults.max_rows if defaults.max_rows is not None else 'no limit'}"
        ),
    )
    parser.add_argument(
        "--interpolate",
        dest="interpolate_csv",
        type=_optional_path,
        default=argparse.SUPPRESS,
        help=(
            "Path to a second combined CSV with fair model probabilities. "
            "Rows are matched by input features (ignoring output columns), then "
            "probabilities are linearly interpolated from baseline to fair over time."
        ),
    )
    parser.add_argument(
        "--interpolate-bins",
        dest="interpolate_bins",
        type=int,
        default=argparse.SUPPRESS,
        help=(
            "Use stepwise interpolation with the given number of bins. "
            "Bins include both endpoints (e.g. 5 bins -> 0.0, 0.25, 0.5, 0.75, 1.0)."
        ),
    )
    parser.add_argument(
        "--interpolate-weights",
        dest="interpolate_weights",
        type=_csv_float_list,
        default=argparse.SUPPRESS,
        help=(
            "Comma-separated interpolation weights to override bin weights. "
            "Length must match --interpolate-bins."
        ),
    )
    parser.add_argument(
        "--deduplicate",
        dest="deduplicate",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Drop duplicate feature rows when loading data (default: false unless --interpolate is set).",
    )

    parsed = parser.parse_args()
    provided = vars(parsed)
    cfg_values = {field.name: getattr(defaults, field.name) for field in fields(Config)}

    for key, value in provided.items():
        cfg_values[key] = value

    if "pred_columns" in provided and "preds_csv" not in provided:
        cfg_values["preds_csv"] = None

    if (
        cfg_values.get("output_transform") == "argmax-normalized"
        and "out_metric" not in provided
    ):
        cfg_values["out_metric"] = "linf"

    if cfg_values.get("interpolate_csv") is not None:
        if "preds_csv" not in provided:
            cfg_values["preds_csv"] = None
        if cfg_values.get("preds_csv") is not None:
            raise ValueError("--interpolate is incompatible with --preds-csv")
        if cfg_values.get("row_ids") is not None:
            raise ValueError("--interpolate is incompatible with --row-ids")
        cfg_values["deduplicate"] = True
        cfg_values["shuffle"] = True
        bins = cfg_values.get("interpolate_bins")
        if bins is not None and bins < 2:
            raise ValueError("--interpolate-bins must be >= 2")
        weights = cfg_values.get("interpolate_weights")
        if weights is not None:
            if bins is None:
                raise ValueError("--interpolate-weights requires --interpolate-bins")
            if len(weights) != bins:
                raise ValueError("--interpolate-weights length must match --interpolate-bins")
            if any(w < 0.0 or w > 1.0 for w in weights):
                raise ValueError("--interpolate-weights values must be in [0, 1]")
    elif cfg_values.get("interpolate_bins") is not None:
        raise ValueError("--interpolate-bins requires --interpolate")
    elif cfg_values.get("interpolate_weights") is not None:
        raise ValueError("--interpolate-weights requires --interpolate")

    return Config(**cfg_values)


def load_data(
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], List[str], List[str], List[str]]:
    """Load feature and probability data from one or two CSV files."""

    def _ensure_exists(path: Path) -> Path:
        if not path.exists():
            raise FileNotFoundError(
                f"Expected file at {path.resolve()} (adjust --input-csv/--preds-csv)"
            )
        return path

    def _clean_prob_names(columns: Sequence[str]) -> List[str]:
        cleaned: List[str] = []
        for name in columns:
            alias = name
            if alias.startswith("prob_"):
                alias = alias.replace("prob_", "p(") + ")"
            cleaned.append(alias.lower())
        return cleaned

    def _resolve_feature_columns(
        fieldnames: Sequence[str], *, exclude: Sequence[str]
    ) -> List[str]:
        if cfg.input_columns:
            candidates = [col for col in cfg.input_columns if col not in ignore]
        else:
            candidates = [
                col for col in fieldnames if col not in ignore and col not in exclude
            ]
        missing = [col for col in candidates if col not in fieldnames]
        if missing:
            raise KeyError(
                f"Columns {missing} missing in {input_path}. Available: {list(fieldnames)}"
            )
        return candidates

    def _resolve_prob_columns(fieldnames: Sequence[str]) -> List[str]:
        if cfg.pred_columns:
            candidates = [col for col in cfg.pred_columns if col not in ignore]
        else:
            candidates = [
                col
                for col in fieldnames
                if col.lower().startswith("prob") and col not in ignore
            ]
        missing = [col for col in candidates if col not in fieldnames]
        if missing:
            raise KeyError(
                f"Columns {missing} missing in probability file. Available: {list(fieldnames)}"
            )
        if not candidates:
            raise ValueError(
                "Could not identify probability columns; provide them explicitly via --pred-cols"
            )
        return candidates

    def _read_combined_rows(
        path: Path, feature_cols: Sequence[str], prob_cols: Sequence[str]
    ) -> Tuple[List[List[float]], List[List[float]], set[int]]:
        feats: List[List[float]] = []
        probs_local: List[List[float]] = []
        seen: set[int] = set()
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for idx, row in enumerate(reader, start=1):
                if cfg.row_ids and idx not in cfg.row_ids:
                    continue
                feats.append([float(row[col]) for col in feature_cols])
                probs_local.append([float(row[col]) for col in prob_cols])
                seen.add(idx)
        return feats, probs_local, seen

    def _read_combined_rows_with_keys(
        path: Path,
        feature_cols: Sequence[str],
        prob_cols: Sequence[str],
        key_cols: Sequence[str],
    ) -> Tuple[List[List[float]], List[List[float]], List[Tuple[str, ...]]]:
        feats: List[List[float]] = []
        probs_local: List[List[float]] = []
        keys: List[Tuple[str, ...]] = []
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for idx, row in enumerate(reader, start=1):
                if cfg.row_ids and idx not in cfg.row_ids:
                    continue
                feats.append([float(row[col]) for col in feature_cols])
                probs_local.append([float(row[col]) for col in prob_cols])
                keys.append(tuple(row[col] for col in key_cols))
        return feats, probs_local, keys

    def _dedupe_aligned_rows(
        keys: Sequence[Tuple[str, ...]],
        feats: Sequence[List[float]],
        probs_local: Sequence[List[float]],
        fair_local: Sequence[List[float]],
    ) -> Tuple[List[Tuple[str, ...]], List[List[float]], List[List[float]], List[List[float]], int]:
        dedup_keys: List[Tuple[str, ...]] = []
        dedup_feats: List[List[float]] = []
        dedup_probs: List[List[float]] = []
        dedup_fair: List[List[float]] = []
        removed = 0
        last_key: Optional[Tuple[str, ...]] = None
        for key, feat, prob, fair in zip(keys, feats, probs_local, fair_local):
            if key == last_key:
                removed += 1
                continue
            dedup_keys.append(key)
            dedup_feats.append(feat)
            dedup_probs.append(prob)
            dedup_fair.append(fair)
            last_key = key
        return dedup_keys, dedup_feats, dedup_probs, dedup_fair, removed

    def _read_columns(
        path: Path, columns: Sequence[str]
    ) -> Tuple[List[List[float]], set[int]]:
        rows: List[List[float]] = []
        seen: set[int] = set()
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for idx, row in enumerate(reader, start=1):
                if cfg.row_ids and idx not in cfg.row_ids:
                    continue
                rows.append([float(row[col]) for col in columns])
                seen.add(idx)
        return rows, seen

    input_path = _ensure_exists(cfg.input_csv)
    preds_path = cfg.preds_csv
    if preds_path is not None and preds_path != cfg.input_csv:
        preds_path = _ensure_exists(preds_path)

    interpolate_path = cfg.interpolate_csv
    if interpolate_path is not None:
        interpolate_path = _ensure_exists(interpolate_path)

    ignore = set(cfg.ignore_columns)
    combined_file = preds_path is None or preds_path == cfg.input_csv
    fair_probs: Optional[List[List[float]]] = None

    with input_path.open(newline="") as fh:
        input_reader = csv.DictReader(fh)
        input_fields = input_reader.fieldnames or []

    if combined_file:
        prob_columns = _resolve_prob_columns(input_fields)
        feature_columns = _resolve_feature_columns(
            input_fields, exclude=prob_columns
        )
        overlap_columns = set(feature_columns) & set(prob_columns)
        if overlap_columns:
            raise ValueError(
                f"Columns {sorted(overlap_columns)} cannot be both feature and prediction columns"
            )
        if interpolate_path is None:
            inputs, probs, seen_input_ids = _read_combined_rows(
                input_path, feature_columns, prob_columns
            )
        else:
            key_columns = [col for col in feature_columns if col.lower() != "pred"]
            inputs, probs, keys_base = _read_combined_rows_with_keys(
                input_path, feature_columns, prob_columns, key_columns
            )
            with interpolate_path.open(newline="") as fh:
                reader = csv.DictReader(fh)
                interp_fields = reader.fieldnames or []
            if list(interp_fields) != list(input_fields):
                raise ValueError(
                    "Interpolate CSV must have the same columns as input CSV"
                )
            _, fair_probs, keys_fair = _read_combined_rows_with_keys(
                interpolate_path, feature_columns, prob_columns, key_columns
            )
            order_base = sorted(range(len(keys_base)), key=lambda i: (keys_base[i], i))
            order_fair = sorted(range(len(keys_fair)), key=lambda i: (keys_fair[i], i))
            sorted_keys_base = [keys_base[i] for i in order_base]
            sorted_keys_fair = [keys_fair[i] for i in order_fair]
            if sorted_keys_base != sorted_keys_fair:
                raise ValueError(
                    "Interpolate CSV rows do not match input CSV rows when aligned by features"
                )
            inputs = [inputs[i] for i in order_base]
            probs = [probs[i] for i in order_base]
            fair_probs = [fair_probs[i] for i in order_fair]
            if cfg.deduplicate:
                (
                    sorted_keys_base,
                    inputs,
                    probs,
                    fair_probs,
                    removed,
                ) = _dedupe_aligned_rows(
                    sorted_keys_base, inputs, probs, fair_probs
                )
                if removed:
                    print(
                        f"Warning: removed {removed} duplicate rows while aligning interpolation data"
                    )
            seen_input_ids = set(range(1, len(inputs) + 1))
        pred_seen_ids = seen_input_ids
    else:
        feature_columns = _resolve_feature_columns(input_fields, exclude=())
        inputs, seen_input_ids = _read_columns(input_path, feature_columns)

        with preds_path.open(newline="") as fh:
            preds_reader = csv.DictReader(fh)
            pred_fields = preds_reader.fieldnames or []
        prob_columns = _resolve_prob_columns(pred_fields)
        probs, pred_seen_ids = _read_columns(preds_path, prob_columns)

        if cfg.row_ids:
            missing_inputs = sorted(cfg.row_ids - seen_input_ids)
            missing_probs = sorted(cfg.row_ids - pred_seen_ids)
            if missing_inputs or missing_probs:
                issues = []
                if missing_inputs:
                    issues.append(f"input CSV missing {missing_inputs}")
                if missing_probs:
                    issues.append(f"preds CSV missing {missing_probs}")
                raise ValueError(
                    "Rows requested via --row-ids not found in CSV(s): " + ", ".join(issues)
                )

    if cfg.row_ids:
        missing_inputs = sorted(cfg.row_ids - seen_input_ids)
        if missing_inputs:
            raise ValueError(
                f"Rows requested via --row-ids not found in input CSV: {missing_inputs}"
            )

    overlap = min(len(inputs), len(probs))
    if fair_probs is not None:
        overlap = min(overlap, len(fair_probs))
    if cfg.row_ids and overlap == 0:
        raise ValueError(
            "--row-ids filtering removed all rows; verify the ids and the CSV contents"
        )
    if overlap == 0:
        raise ValueError("No overlapping rows between feature and probability files")
    if len(inputs) != len(probs):
        print(
            f"Warning: trimming to {overlap} rows (features={len(inputs)}, probabilities={len(probs)})"
        )

    if cfg.max_rows is not None and overlap > cfg.max_rows:
        print(
            f"Warning: limiting to {cfg.max_rows} rows due to --max-n (available overlap={overlap})"
        )
        overlap = cfg.max_rows

    inputs = inputs[:overlap]
    probs = probs[:overlap]
    fair_probs_array: Optional[np.ndarray] = None

    input_array = np.asarray(inputs, dtype=np.float32)
    probs_array = np.asarray(probs, dtype=np.float64)
    if interpolate_path is not None:
        fair_probs_array = np.asarray(fair_probs[:overlap], dtype=np.float64)

    if cfg.shuffle:
        rng = np.random.default_rng(seed=42)
        indices = np.arange(overlap)
        rng.shuffle(indices)
        input_array = input_array[indices]
        probs_array = probs_array[indices]
        if fair_probs_array is not None:
            fair_probs_array = fair_probs_array[indices]

    if cfg.normalize:
        norms = np.linalg.norm(input_array, axis=1, keepdims=True)
        nonzero_mask = (norms > 0).squeeze(axis=1)
        if nonzero_mask.any():
            input_array[nonzero_mask] /= norms[nonzero_mask]

    prob_names = _clean_prob_names(prob_columns)

    return input_array, probs_array, fair_probs_array, list(feature_columns), prob_names, list(prob_columns)


def transform_outputs(
    probs: np.ndarray,
    prob_names: Sequence[str],
    raw_prob_names: Sequence[str],
    output_transform: Literal["probs", "argmax-normalized"],
) -> tuple[np.ndarray, list[str], dict[str, object]]:
    """Transform probability outputs before they are passed to the monitor."""

    if output_transform == "probs":
        return probs, list(prob_names), {
            "name": "probs",
            "description": "raw probability vector",
            "source_probability_columns": list(prob_names),
            "raw_probability_columns": list(raw_prob_names),
        }

    if output_transform != "argmax-normalized":
        raise ValueError(f"unsupported output_transform: {output_transform}")

    labels = numeric_labels_from_probability_names(raw_prob_names)
    if len(labels) != probs.shape[1]:
        raise ValueError(
            f"probability label count mismatch: {len(labels)} labels for {probs.shape[1]} columns"
        )

    label_min = min(labels)
    label_max = max(labels)
    label_range = label_max - label_min
    if label_range <= 0:
        raise ValueError("--output-transform argmax-normalized requires at least two distinct numeric labels")

    argmax_positions = np.argmax(probs, axis=1)
    argmax_labels = np.asarray([labels[pos] for pos in argmax_positions], dtype=np.float64)
    normalized = ((argmax_labels - label_min) / label_range).reshape(-1, 1)
    return normalized, ["normalized_argmax_label"], {
        "name": "argmax-normalized",
        "description": "argmax label normalized to [0, 1]",
        "source_probability_columns": list(prob_names),
        "raw_probability_columns": list(raw_prob_names),
        "label_values": labels,
        "label_min": label_min,
        "label_max": label_max,
        "output_columns": ["normalized_argmax_label"],
    }


def numeric_labels_from_probability_names(prob_names: Sequence[str]) -> list[float]:
    labels: list[float] = []
    for name in prob_names:
        if name.startswith("prob_"):
            raw = name.removeprefix("prob_")
        elif name.startswith("p(") and name.endswith(")"):
            raw = name[2:-1]
        else:
            raise ValueError(
                f"Cannot infer numeric label from probability column '{name}'. "
                "Expected names like prob_0, prob_1, ..."
            )
        try:
            labels.append(float(raw))
        except ValueError as exc:
            raise ValueError(f"Probability label '{raw}' is not numeric") from exc
    return labels


def save_results_json(
    cfg: Config,
    inputs: np.ndarray,
    probs: np.ndarray,
    records: Sequence[
        Tuple[QuantitativeResult, np.ndarray, np.ndarray, float, Optional[ObservationResult], Optional[float]]
    ],
    feature_names: Sequence[str],
    prob_names: Sequence[str],
    output_transform_metadata: Mapping[str, object],
    total_time: float,
    total_epsilon_time: float,
    epsilon_count: int,
) -> Path:
    """Serialize the run into experiments/<timestamp>.json."""

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_path = cfg.results_dir / f"quant_run_{timestamp}.json"
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    serializable_records = []
    for idx, (result, point_vec, prob_vec, time, eps_res, eps_time) in enumerate(records):
        record_dict = asdict(result)
        record_dict["k_progression"] = list(result.k_progression)
        record_dict["time"] = time

        if eps_res:
            record_dict["epsilon_monitor_flag"] = True if eps_res.counterexamples else False
            record_dict["epsilon_monitor_result"] = asdict(eps_res)
        else:
            record_dict["epsilon_monitor_result"] = None

        record_dict["epsilon_monitor_time_ms"] = eps_time

        if cfg.save_points:
            record_dict["point_vector"] = [float(v) for v in point_vec]
            record_dict["prob_vector"] = [float(v) for v in prob_vec]
            witness_id = result.witness_id
            if witness_id is not None and 0 <= witness_id < inputs.shape[0]:
                record_dict["witness_point_vector"] = [float(v) for v in inputs[witness_id]]
                record_dict["witness_prob_vector"] = [float(v) for v in probs[witness_id]]
            else:
                record_dict["witness_point_vector"] = None
                record_dict["witness_prob_vector"] = None

        record_dict["index"] = idx
        serializable_records.append(record_dict)

    payload = {
        "metadata": {
            "commandline": " ".join(sys.argv),
            "timestamp": timestamp,
            "input_csv": str(cfg.input_csv),
            "preds_csv": str(cfg.preds_csv) if cfg.preds_csv is not None else None,
            "interpolate_csv": str(cfg.interpolate_csv) if cfg.interpolate_csv is not None else None,
            "deduplicate": cfg.deduplicate,
            "interpolate_bins": cfg.interpolate_bins,
            "interpolate_weights": list(cfg.interpolate_weights) if cfg.interpolate_weights is not None else None,
            "input_exponent": cfg.input_exponent,
            "discount_factor": cfg.discount_factor,
            "frnn_metric": cfg.frnn_metric,
            "out_metric": cfg.out_metric,
            "output_transform": cfg.output_transform,
            "output_transform_metadata": dict(output_transform_metadata),
            "display_stride": cfg.display_stride,
            "backend": cfg.backend,
            "batchsize": cfg.batchsize,
            "frnn_threads": cfg.frnn_threads,
            "initial_k": cfg.initial_k,
            "max_k": cfg.max_k,
            "k_grow_factor": cfg.k_grow_factor,
            "total_time": total_time,
            "max_rows": cfg.max_rows,
            "row_ids": cfg.row_ids,
            "walltime_seconds": cfg.walltime_seconds,
            "save_points": cfg.save_points,
            "static": cfg.static,
            "epsilon": cfg.epsilon,
            "normalize": cfg.normalize,
            "epsilon_monitor_total_time_ms": total_epsilon_time if epsilon_count else None,
            "epsilon_monitor_avg_time_ms": (total_epsilon_time / epsilon_count) if epsilon_count else None,
            "epsilon_monitor_observations": epsilon_count if epsilon_count else None,
            "feature_columns": list(feature_names),
            "probability_columns": list(
                output_transform_metadata.get("source_probability_columns", prob_names)
            ),
            "output_columns": list(prob_names),
            "ignore_columns": list(cfg.ignore_columns),
        },
        "records": serializable_records,
    }

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(_json_finite(payload), fh, indent=2, allow_nan=False)

    return output_path


def _json_finite(value):
    """Convert non-finite floats to null so browser JSON.parse can read outputs."""

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    return value


def _print_observation(
    idx: int,
    res: QuantitativeResult,
    x_vec: np.ndarray,
    p_vec: np.ndarray,
    inputs: np.ndarray,
    probs: np.ndarray,
    prob_names: Sequence[str],
    feature_names: Sequence[str],
) -> None:
    TRUNCATE = 200

    witness = res.witness_id if res.witness_id is not None else None
    ratio_disp = "inf" if math.isinf(res.max_ratio) else f"{res.max_ratio:8.4f}"
    flag = "•" if res.stopped_by_bound else " "
    print(f"  [{idx:05d}] ratio={ratio_disp} compared={res.compared_count:5d} witness={witness if witness is not None else '--':>6} d_out={round(res.witness_out_distance, 4)} d_in={round(res.witness_in_distance, 4)} {flag}" if res.witness_in_distance and res.witness_out_distance else f"  [{idx:05d}] ratio={ratio_disp} compared={res.compared_count:5d} witness={witness if witness is not None else '--':>6} {flag}")

    if len(x_vec) > 40: return # Don't print high-dimensional data

    columns = list(prob_names) + list(feature_names)
    header = "                " + " ".join((name.rjust(10) + "  " * 10)[:10] for name in columns)
    point_row = _format_row(np.concatenate([p_vec, x_vec]))
    print(header[:TRUNCATE + 20]) # Some extra to align with rows
    print(f"      point   {point_row[:TRUNCATE]}")

    if witness is not None and 0 <= witness < inputs.shape[0]:
        witness_row = _format_row(np.concatenate([probs[witness], inputs[witness]]))
        print(f"      witness {witness_row[:TRUNCATE]}")
    else:
        print("      witness --")


def _format_row(values: Iterable[float]) -> str:
    return " ".join(f"{float(val):10.4f}".rjust(10)[:10] for val in values)


def _print_percentiles(label: str, values: np.ndarray) -> None:
    perc_points = [0, 50, 90, 95, 99, 100]
    percs = np.percentile(values, perc_points)
    stats = (
        f"min={percs[0]:.4f} median={percs[1]:.4f} p90={percs[2]:.4f} "
        f"p95={percs[3]:.4f} p99={percs[4]:.4f} max={percs[5]:.4f}"
    )
    print(f"{label:<10}: {stats} mean={values.mean():.4f}")


if __name__ == "__main__":
    main()
