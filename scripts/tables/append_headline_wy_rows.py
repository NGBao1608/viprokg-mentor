"""Restore the MENTOR headline W/Y rows in hits_bang.tsv.

The values are frozen published numbers and are NOT recomputed:
  MKG-W (OOF gate, lambda 4):   MRR 42.50 +-0.48 CI (SE 0.11), H@1 35.89, H@3 45.26, H@10 54.89
  MKG-Y (OOF gate, lambda 0.1): MRR 44.80 +-1.24 CI (SE 0.29), H@1 40.07, H@3 47.00, H@10 53.27
The config key is "MENTOR headline". The "V3_19b (2 expert)" rows (42.46/45.87, not the headline)
are removed. hits_bang.tsv is backed up as hits_bang.tsv.bak-<date> before writing. Idempotent.
"""

from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TSV = ROOT / "data" / "experimental" / "v3" / "hits_bang.tsv"
CONFIG_KEY = "MENTOR headline"

# dataset -> {metric: (mean %, lo %, hi %)}; for H@k the CI equals the point value
# (the main table only prints a CI for MRR).
VALS = {
    "MKG-W": {
        "n": 8548,
        "MRR": (42.50, 42.02, 42.98),
        "H@1": (35.89, 35.89, 35.89),
        "H@3": (45.26, 45.26, 45.26),
        "H@10": (54.89, 54.89, 54.89),
    },
    "MKG-Y": {
        "n": 5326,
        "MRR": (44.80, 43.56, 46.04),
        "H@1": (40.07, 40.07, 40.07),
        "H@3": (47.00, 47.00, 47.00),
        "H@10": (53.27, 53.27, 53.27),
    },
}


def main() -> int:
    lines = TSV.read_text(encoding="utf-8").splitlines()
    # back up before writing
    bak = TSV.with_name(f"{TSV.name}.bak-{dt.date.today():%Y%m%d}")
    shutil.copy2(TSV, bak)
    print(f"backup -> {bak.name}")

    # drop V3_19b and any previous MENTOR headline rows (idempotent)
    keep = [
        line
        for line in lines
        if not line.startswith("V3_19b (2 expert)\t") and not line.startswith(f"{CONFIG_KEY}\t")
    ]
    n_dropped = len(lines) - len(keep)
    print(f"dropped {n_dropped} line(s) (V3_19b + any previous {CONFIG_KEY})")

    new = []
    for dataset, d in VALS.items():
        n = d["n"]
        for metric in ("MRR", "H@1", "H@3", "H@10"):
            mean, lo, hi = d[metric]
            new.append(
                f"{CONFIG_KEY}\t{dataset}\t tổng\t{metric}\t{n}\t{mean/100:.5f}\t[{lo/100:.5f}; {hi/100:.5f}]".replace(
                    "\t tổng\t", "\ttổng\t"
                )
            )
    out = keep + new
    TSV.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"added {len(new)} {CONFIG_KEY} line(s) (W/Y x 4 metrics). Total {len(out)} lines.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
