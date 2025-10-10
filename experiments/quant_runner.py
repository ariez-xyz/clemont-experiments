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
from typing import Iterable, List, Optional, Sequence, Tuple, Literal

import numpy as np

from clemont.frnn import KdTreeFRNN, FaissFRNN
from clemont.quantitative_monitor import QuantitativeMonitor, QuantitativeResult
from clemont.monitor import Monitor, ObservationResult

def _csv_list(value: str) -> Tuple[str, ...]:
    items = [item.strip() for item in value.split(",")]
    return tuple(item for item in items if item)


def _optional_path(value: str) -> Optional[Path]:
    lowered = value.strip().lower()
    if lowered in {"", "none", "null"}:
        return None
    return Path(value)


@dataclass
class Config:
    input_csv: Path = Path("..") / "data" / "toydata" / "inputs_numeric.csv"
    preds_csv: Optional[Path] = Path("..") / "data" / "toydata" / "predictions_with_probs.csv"
    results_dir: Path = Path("..") / "results" / "quantitative"
    input_columns: Tuple[str, ...] | None = None
    pred_columns: Tuple[str, ...] | None = None
    ignore_columns: Tuple[str, ...] = ("row_id",)
    frnn_metric: Literal["linf", "l1", "l2", "tv", "cosine"] = "l2"
    out_metric: Literal["linf", "l1", "l2", "tv", "cosine"] = "l2"
    backend: Literal["kdtree", "faiss"] = "kdtree"
    display_stride: int = 1000
    frnn_threads: int = 4
    input_exponent: float = 1
    batchsize: int = 1000
    initial_k: int = 16
    max_k: Optional[int] = None
    max_rows: Optional[int] = None
    save_points: bool = False
    static: bool = False
    epsilon: Optional[float] = None


def main() -> None:
    cfg = parse_args()
    inputs, probs, input_names, prob_names = load_data(cfg)
    num_points = inputs.shape[0]

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
    )

    epsilon_monitor: Optional[Monitor] = None
    if cfg.epsilon:
        epsilon_monitor = Monitor(backend_factory)

    if cfg.static: monitor.batch_add(zip(inputs, probs))

    full_records: List[Tuple[QuantitativeResult, np.ndarray, np.ndarray, float, Optional[ObservationResult]]] = []

    print("=== Streaming quantitative monitoring demo ===")
    print(
        f"Points={num_points}, input-dim={inputs.shape[1]}, probs-dim={probs.shape[1]}, "
        f"FRNN metric={cfg.frnn_metric}, output metric={cfg.out_metric}"
    )
    print(f"Every {cfg.display_stride}th observation (• denotes early stop via bound):")

    display_indices = set(range(0, num_points, cfg.display_stride))
    display_indices.add(num_points - 1)

    total_time = 0

    try:
        for idx, (x_vec, p_vec) in enumerate(zip(inputs, probs)):
            start_time = time.time()
            res = monitor.observe(x_vec, p_vec, dry_run=cfg.static)
            iter_time = (time.time() - start_time) * 1000

            eps_res: Optional[ObservationResult] = None if not epsilon_monitor else epsilon_monitor.observe(x_vec, np.argmax(p_vec))

            full_records.append((res, x_vec, p_vec, iter_time, eps_res))
            total_time += iter_time

            if idx in display_indices:
                _print_observation(
                    idx,
                    res,
                    x_vec,
                    p_vec,
                    inputs,
                    probs,
                    prob_names,
                    input_names,
                )
    except KeyboardInterrupt:
        pass

    ratios = np.array([rec[0].max_ratio for rec in full_records], dtype=float)
    compared = np.array([rec[0].compared_count for rec in full_records], dtype=int)
    early_stop = sum(rec[0].stopped_by_bound for rec in full_records)
    max_depth = max((rec[0].k_progression[-1] for rec in full_records if rec[0].k_progression), default=0)

    print("\n=== Summary ===")
    print(f"completed in {round(total_time/1000, 2)}s")
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
        res, x_vec, p_vec, iter_time, eps_res = full_records[idx]
        print(f"-- {label} (index {idx}, compared={res.compared_count} in {iter_time})")
        _print_observation(
            idx,
            res,
            x_vec,
            p_vec,
            inputs,
            probs,
            prob_names,
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
            res, x_vec, p_vec, iter_time, eps_res = full_records[idx]
            _print_observation(
                idx,
                res,
                x_vec,
                p_vec,
                inputs,
                probs,
                prob_names,
                input_names,
            )
    else:
        print("\nNo observations qualified for the high-ratio sample.")

    output_path = save_results_json(cfg, inputs, probs, full_records, input_names, prob_names, total_time)
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
    parser.add_argument("--display-stride", dest="display_stride", type=int, default=argparse.SUPPRESS,
                        help=f"Print every Nth observation (default: {defaults.display_stride})")
    parser.add_argument("--backend", dest="backend", type=str, default=argparse.SUPPRESS,
                        help=f"kNN backend to use (default: {defaults.backend})")
    parser.add_argument("--frnn-threads", dest="frnn_threads", type=int, default=argparse.SUPPRESS,
                        help=f"Thread hint for FRNN backends (default: {defaults.frnn_threads})")
    parser.add_argument("--input-exponent", dest="input_exponent", type=float, default=argparse.SUPPRESS,
                        help=f"Input exponent for monitor (default: {defaults.input_exponent})")
    parser.add_argument("--batchsize", dest="batchsize", type=int, default=argparse.SUPPRESS,
                        help=f"Batch size for batched kNN backends (default: {defaults.batchsize})")
    parser.add_argument("--initial-k", dest="initial_k", type=int, default=argparse.SUPPRESS,
                        help=f"Initial k value for repeated kNN queries (default: {defaults.initial_k})")
    parser.add_argument("--save-points", dest="save_points", action="store_true",
                        help=f"Whether to write raw input and output points to .json log (default: {defaults.save_points})")
    parser.add_argument("--static", dest="static", action="store_true",
                        help=f"preloads the data before run to compute (default: {defaults.save_points})")
    parser.add_argument("--epsilon", dest="epsilon", type=float, default=argparse.SUPPRESS,
                        help=f"epsilon value. If set, also computes epsilon-monitor results (default: {defaults.save_points})")
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

    parsed = parser.parse_args()
    provided = vars(parsed)
    cfg_values = {field.name: getattr(defaults, field.name) for field in fields(Config)}

    for key, value in provided.items():
        cfg_values[key] = value

    if "pred_columns" in provided and "preds_csv" not in provided:
        cfg_values["preds_csv"] = None

    return Config(**cfg_values)


def load_data(cfg: Config) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Load feature and probability data from one or two CSV files."""

    def _ensure_exists(path: Path) -> Path:
        if not path.exists():
            raise FileNotFoundError(f"Expected file at {path.resolve()} (adjust --input-csv/--preds-csv)")
        return path

    def _clean_prob_names(columns: Sequence[str]) -> List[str]:
        prob_names: List[str] = []
        for name in columns:
            cleaned = name
            if name.startswith("prob_"):
                cleaned = name.replace("prob_", "p(") + ")"
            prob_names.append(cleaned.lower())
        return prob_names

    input_path = _ensure_exists(cfg.input_csv)
    preds_path = cfg.preds_csv
    if preds_path is not None and preds_path != cfg.input_csv:
        preds_path = _ensure_exists(preds_path)

    ignore = set(cfg.ignore_columns)

    if preds_path is None or preds_path == cfg.input_csv:
        with input_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames or []
            available = [col for col in fieldnames if col not in ignore]

            if cfg.pred_columns:
                pred_cols = [col for col in cfg.pred_columns if col not in ignore]
            else:
                pred_cols = [col for col in available if col.lower().startswith("prob")]

            missing_probs = [col for col in pred_cols if col not in fieldnames]
            if missing_probs:
                raise KeyError(
                    f"Columns {missing_probs} missing in {input_path}. Available: {fieldnames}"
                )
            if not pred_cols:
                raise ValueError(
                    "Could not identify probability columns; provide them explicitly via --pred-cols"
                )

            if cfg.input_columns:
                feature_columns = [col for col in cfg.input_columns if col not in ignore]
            else:
                feature_columns = [col for col in available if col not in pred_cols]

            missing_features = [col for col in feature_columns if col not in fieldnames]
            if missing_features:
                raise KeyError(
                    f"Columns {missing_features} missing in {input_path}. Available: {fieldnames}"
                )
            overlap = set(feature_columns) & set(pred_cols)
            if overlap:
                raise ValueError(
                    f"Columns {sorted(overlap)} cannot be both feature and prediction columns"
                )

            inputs: List[List[float]] = []
            probs: List[List[float]] = []
            for row in reader:
                inputs.append([float(row[col]) for col in feature_columns])
                probs.append([float(row[col]) for col in pred_cols])

        prob_names = _clean_prob_names(pred_cols)
    else:
        with input_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames or []
            if cfg.input_columns:
                feature_columns = [col for col in cfg.input_columns if col not in ignore]
            else:
                feature_columns = [col for col in fieldnames if col not in ignore]

            missing_features = [col for col in feature_columns if col not in fieldnames]
            if missing_features:
                raise KeyError(
                    f"Columns {missing_features} missing in {input_path}. Available: {fieldnames}"
                )

            inputs = [[float(row[col]) for col in feature_columns] for row in reader]

        with preds_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames or []
            if cfg.pred_columns:
                prob_cols = [col for col in cfg.pred_columns if col not in ignore]
            else:
                prob_cols = [col for col in fieldnames if col.lower().startswith("prob") and col not in ignore]

            missing_probs = [col for col in prob_cols if col not in fieldnames]
            if missing_probs:
                raise KeyError(
                    f"Columns {missing_probs} missing in {preds_path}. Available: {fieldnames}"
                )
            if not prob_cols:
                raise ValueError(
                    "Could not identify probability columns; provide them explicitly via --pred-cols"
                )

            probs = [[float(row[col]) for col in prob_cols] for row in reader]

        prob_names = _clean_prob_names(prob_cols)

    overlap = min(len(inputs), len(probs))
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

    return (
        np.asarray(inputs[:overlap], dtype=np.float32),
        np.asarray(probs[:overlap], dtype=np.float64),
        list(feature_columns),
        prob_names,
    )


def save_results_json(
    cfg: Config,
    inputs: np.ndarray,
    probs: np.ndarray,
    records: Sequence[Tuple[QuantitativeResult, np.ndarray, np.ndarray, float, Optional[ObservationResult]]],
    feature_names: Sequence[str],
    prob_names: Sequence[str],
    total_time: float,
) -> Path:
    """Serialize the run into experiments/<timestamp>.json."""

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_path = cfg.results_dir / f"quant_run_{timestamp}.json"
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    serializable_records = []
    for idx, (result, point_vec, prob_vec, time, eps_res) in enumerate(records):
        record_dict = asdict(result)
        record_dict["k_progression"] = list(result.k_progression)
        record_dict["time"] = time

        if eps_res: record_dict["epsilon_monitor_result"] = asdict(eps_res)

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
            "frnn_metric": cfg.frnn_metric,
            "out_metric": cfg.out_metric,
            "display_stride": cfg.display_stride,
            "frnn_threads": cfg.frnn_threads,
            "input_exponent": cfg.input_exponent,
            "batchsize": cfg.batchsize,
            "initial_k": cfg.initial_k,
            "backend": cfg.backend,
            "feature_columns": list(feature_names),
            "probability_columns": list(prob_names),
            "ignore_columns": list(cfg.ignore_columns),
            "total_time": total_time,
            "max_k": cfg.max_k,
            "max_rows": cfg.max_rows,
            "save_points": cfg.save_points,
            "static": cfg.static,
        },
        "records": serializable_records,
    }

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    return output_path


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
