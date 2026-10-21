"""Score one MENTOR_base+L ("A3'") run: select g* on validation, report test MRR at g*.

`sweep_gL.tsv` (validation) and `sweep_gL_test.tsv` (test) hold per-key ranks, one column per
lexical weight g (`g0.0`, `g0.025`, ...). g* maximises validation MRR; the test MRR is read at g*
only (the other test columns are never used for selection).
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path


def read_columns(path: Path) -> dict[str, list[float]]:
    """Reciprocal ranks per g column of a sweep TSV (comment lines start with '#')."""
    with open(path, encoding="utf-8") as f:
        header = f.readline()
        while header.startswith("#"):
            header = f.readline()
        names = header.rstrip("\n").split("\t")
        cols = {n: [] for n in names if n.startswith("g")}
        idx = {n: names.index(n) for n in cols}
        for line in f:
            if line.strip():
                v = line.rstrip("\n").split("\t")
                for n, i in idx.items():
                    cols[n].append(1.0 / float(v[i]))
    return cols


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="directory holding sweep_gL.tsv and sweep_gL_test.tsv")
    ap.add_argument("--out_json", default="")
    a = ap.parse_args()
    run = Path(a.run_dir)
    valid = {k: st.fmean(v) for k, v in read_columns(run / "sweep_gL.tsv").items()}
    gstar = max(valid, key=valid.get)
    test = read_columns(run / "sweep_gL_test.tsv")
    res = {
        "g_star": gstar[1:],
        "valid_mrr_at_g_star": valid[gstar],
        "test_mrr_at_g_star": st.fmean(test[gstar]),
        "n_test_keys": len(test[gstar]),
    }
    print(json.dumps(res, indent=1))
    if a.out_json:
        Path(a.out_json).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
