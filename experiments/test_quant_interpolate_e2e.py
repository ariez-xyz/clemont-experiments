import csv
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


class TestQuantRunnerInterpolationE2E(unittest.TestCase):
    def test_interpolation_end_to_end(self) -> None:
        repo_root = Path(__file__).parent.parent
        base_csv = repo_root / "data" / "toydata" / "interp_base.csv"
        fair_csv = repo_root / "data" / "toydata" / "interp_fair.csv"
        runner = repo_root / "experiments" / "quant_runner.py"

        results_dir = repo_root / "results" / "quantitative" / "interp_test"
        results_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                sys.executable,
                str(runner),
                "--input-csv",
                str(base_csv),
                "--interpolate",
                str(fair_csv),
                "--save-points",
                "--results-dir",
                str(results_dir),
                "--display-stride",
                "1000",
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )

        output_path = None
        for line in result.stdout.splitlines():
            print(line)
            if line.startswith("Saved run to "):
                output_path = line.replace("Saved run to ", "").strip()
                break

        self.assertIsNotNone(output_path, "quant_runner did not report output path")
        out_path = Path(output_path)
        self.assertTrue(out_path.exists(), f"missing output json: {out_path}")

        with out_path.open() as fh:
            payload = json.load(fh)

        records = payload["records"]
        self.assertEqual(len(records), 9)

        feature_columns = payload["metadata"]["feature_columns"]
        self.assertEqual(feature_columns, ["f1"])
        self.assertTrue(payload["metadata"]["deduplicate"])

        def load_prob_map(path: Path) -> dict[float, list[list[float]]]:
            with path.open(newline="") as fh:
                reader = csv.DictReader(fh)
                fieldnames = reader.fieldnames or []
                prob_cols = [col for col in fieldnames if col.lower().startswith("prob")]
                key_cols = [col for col in fieldnames if col not in prob_cols]
                if key_cols != ["f1"]:
                    raise AssertionError(f"unexpected key columns: {key_cols}")
                mapping: dict[float, list[list[float]]] = {}
                for row in reader:
                    f1 = round(float(row["f1"]), 1)
                    probs = [float(row[col]) for col in prob_cols]
                    mapping.setdefault(f1, []).append(probs)
            return mapping

        base_map = load_prob_map(base_csv)
        fair_map = load_prob_map(fair_csv)

        num_points = len(records)
        print("\n=== Interpolation check (features, base->fair blend) ===")
        print(f"{'row':>3} {'blend':>5} {'f1':>4} {'base_csv':>18} {'fair_csv':>18} {'expected':>18} {'actual':>18} {'ratio':>10}")
        for idx, record in enumerate(records):
            blend = idx / (num_points - 1) if num_points > 1 else 0.0
            actual = np.asarray(record["prob_vector"], dtype=float)
            f1 = float(record["point_vector"][0])
            f1_key = round(f1, 1)
            base_vals = base_map.get(f1_key)
            fair_vals = fair_map.get(f1_key)
            base_ref = base_vals[0] if base_vals else [0.0, 1.0]
            expected_candidates = []
            if fair_vals:
                for fair in fair_vals:
                    expected_candidates.append(
                        (1.0 - blend) * np.asarray(base_ref) + blend * np.asarray(fair)
                    )
            else:
                expected_candidates.append(np.array([blend * f1, 1.0 - blend * f1], dtype=float))
            print(
                f"{idx:3d} {blend:5.2f} {f1:4.1f} "
                f"{str(base_vals):>18} {str(fair_vals):>18} "
                f"{str([round(x, 6) for x in expected_candidates[0].tolist()]):>18} "
                f"{str([round(x, 6) for x in actual.tolist()]):>18} "
                f"{record['max_ratio']:10.6f}"
            )
            self.assertTrue(
                any(np.allclose(actual, candidate, atol=1e-6) for candidate in expected_candidates)
            )


if __name__ == "__main__":
    unittest.main()
