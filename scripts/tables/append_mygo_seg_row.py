"""Append the `MyGO+seg` rows (ViProKG, 3 groups x MRR/H@1/H@3/H@10/MR) to `hits_bang.tsv`.

The ViProKG table uses MyGO with word-segmented text. Same instruments as build_hits_table:
`read_ranks`, `compute_metrics` (REF.metrics), `query_groups_v2` and `conf_interval` (t, df=2
over 3 seeds). Segmented-text ranks come from
`data/experimental/mygo_seg/MyGO_ViMMKG2_seg_s{seed}/ranks_test.tsv`.
Idempotent: if MyGO+seg rows already exist nothing is written (no double counting).
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_hits_table import (
    OUT_DIR,
    compute_metrics,
    conf_interval,
    query_groups_v2,
    read_ranks,
)  # noqa: E402

SEG = HERE.parents[1] / "data" / "experimental" / "mygo_seg"
SEEDS = ("42", "2024", "7")
METRICS = ("MRR", "H@1", "H@3", "H@10", "MR")
GROUPS = ("tổng", "thường", "lát")
CONFIG_NAME, DATASET = "MyGO+seg", "ViMMKG2"


def main() -> int:
    tsv = Path(OUT_DIR)
    cur = tsv.read_text(encoding="utf-8")
    if f"{CONFIG_NAME}\t{DATASET}\t" in cur:
        print(f"hits_bang.tsv already has {CONFIG_NAME} rows; not writing again (idempotent).")
        return 0

    slice_sets = query_groups_v2()
    selector = {
        "tổng": lambda k: True,
        "thường": lambda k: k[:3] in slice_sets["thường"],
        "lát": lambda k: k[:3] in slice_sets["lát"],
    }

    rows = []
    for group in GROUPS:
        # per-seed metrics on the key set of the group
        per = {d: [] for d in METRICS}
        n_keys = None
        for s in SEEDS:
            p = SEG / f"MyGO_ViMMKG2_seg_s{s}" / "ranks_test.tsv"
            if not p.exists():
                raise SystemExit(f"missing {p}: fewer than 3 seeds. STOP.")
            r = read_ranks(str(p))
            sub = {k: v for k, v in r.items() if selector[group](k)}
            m = compute_metrics(sub)
            if n_keys is None:
                n_keys = m["n"]
            assert m["n"] == n_keys, f"{group}/s{s}: n mismatch {m['n']} != {n_keys}"
            for d in METRICS:
                per[d].append(m[d])
        for d in METRICS:
            mean, lo, hi = conf_interval(per[d])
            rows.append(
                f"{CONFIG_NAME}\t{DATASET}\t{group}\t{d}\t{n_keys}\t{mean:.5f}\t[{lo:.5f}; {hi:.5f}]"
            )
            print(f"  {group:7} {d:5} n={n_keys:>5} tb={mean:.5f} [{lo:.5f}; {hi:.5f}]")

    with tsv.open("a", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    print(f"added {len(rows)} {CONFIG_NAME} rows to {tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
