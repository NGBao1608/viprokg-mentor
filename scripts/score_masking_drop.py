"""Answer-name masking: MRR drop on the mention-overlap vs mention-free split (ViProKG).

Reads per-key test ranks of the unmasked run (`dump_test_s<seed>.tsv`) and of the same run
re-scored with answer names masked (`dump_test_che_s<seed>.tsv`), 3 seeds, and reports the mean
MRR drop (percentage points) overall, on mention-overlap pairs and on mention-free pairs.
A pair (h, t) is mention-overlap when the unordered pair is in the v2 mention-edge file.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "data" / "experimental" / "v3"
SEEDS = (42, 7, 2024)


def read_ranks(path: Path) -> dict:
    """Per-key reciprocal ranks; supports the long (`huong`) and wide (`head_rank`) formats."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            k3 = (r["s_qid"], r["p_pid"], r["o_qid"])
            if "huong" in r:
                out[k3 + (r["huong"],)] = 1.0 / float(r["rank_gold"])
            else:
                out[k3 + ("head",)] = 1.0 / float(r["head_rank"])
                out[k3 + ("tail",)] = 1.0 / float(r["tail_rank"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dumps", default=str(V3 / "che_v2_dumps"))
    ap.add_argument("--ext", default=str(V3 / "ext" / "ViMMKG2"))
    ap.add_argument("--edges", default=str(V3 / "canh_v2" / "v10_canh_nhac_ten_ViMMKG2_v2.tsv"))
    ap.add_argument("--out_json", default="")
    a = ap.parse_args()

    with open(Path(a.ext) / "entity2id.tsv", encoding="utf-8") as f:
        next(f)
        ent = {}
        for line in f:
            if line.strip():
                v = line.rstrip("\n").split("\t")
                ent[v[0]] = int(v[1])
    with open(a.edges, encoding="utf-8") as f:
        next(f)
        edges = {frozenset((int(x), int(y))) for x, y in (ln.split("\t") for ln in f if ln.strip())}

    def overlap(k) -> bool:
        return frozenset((ent.get(k[0], -1), ent.get(k[2], -2))) in edges

    def drop(sel) -> float:
        per_seed = []
        for s in SEEDS:
            base = read_ranks(Path(a.dumps) / f"dump_test_s{s}.tsv")
            masked = read_ranks(Path(a.dumps) / f"dump_test_che_s{s}.tsv")
            keys = set(base) & set(masked) & {k for k in base if sel(k)}
            per_seed.append(
                (st.fmean(base[k] for k in keys) - st.fmean(masked[k] for k in keys)) * 100
            )
        return st.fmean(per_seed)

    res = {
        "drop_total": drop(lambda k: True),
        "drop_overlap": drop(overlap),
        "drop_free": drop(lambda k: not overlap(k)),
    }
    res["ratio"] = res["drop_overlap"] / res["drop_free"]
    print(json.dumps(res, indent=1))
    if a.out_json:
        Path(a.out_json).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
