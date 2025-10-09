#!/usr/bin/env python3
import csv
from pathlib import Path

f1 = Path("cifar10-Bartoldson2024Adversarial_WRN-94-16-adv.csv")
f2 = Path("cifar10-Bartoldson2024Adversarial_WRN-94-16.csv")
out1 = Path(f1.stem + ".diff.csv")
out2 = Path(f2.stem + ".diff.csv")

with f1.open(newline="") as a, f2.open(newline="") as b, \
     out1.open("w", newline="") as oa, out2.open("w", newline="") as ob:
    ra, rb = csv.reader(a), csv.reader(b)
    wa, wb = csv.writer(oa), csv.writer(ob)

    # Read headers (if present and identical, keep them; otherwise omit)
    row_a = next(ra, None)
    row_b = next(rb, None)
    if row_a is not None and row_b is not None and row_a == row_b:
        wa.writerow(row_a); wb.writerow(row_b)
    else:
        # No/unequal headers: treat those rows as data
        if row_a is not None and row_b is not None and row_a != row_b:
            wa.writerow(row_a); wb.writerow(row_b)

    diff_count = 0
    line_idx = 2  # 1-based with header; adjust as you like

    for ra_row, rb_row in zip(ra, rb):
        if ra_row != rb_row:
            wa.writerow(ra_row)
            wb.writerow(rb_row)
            diff_count += 1
        line_idx += 1

print(f"Wrote {diff_count} differing rows to:\n  {out1}\n  {out2}")
