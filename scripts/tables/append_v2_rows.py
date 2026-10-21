"""Append the ViProKG rows to `hits_bang.tsv`: A3'* lambda 8 (ours), the MENTOR-style
baseline, MoMoK, DSoM re-impl. and AdaMF (3 groups x MRR/H@1/H@3/H@10/MR, 3 seeds, t-CI df=2).

These rows used to be written by an ad-hoc appender; re-running build_hits_table overwrites
`hits_bang.tsv` and would drop them. This script rebuilds them reproducibly from the rank files
copied to `data/experimental/v3/ranks_v2/`, with the same instruments as build_hits_table
(`read_ranks` / `compute_metrics` (REF.metrics) + `query_groups_v2` + `conf_interval`).
Idempotent per configuration.

AdaMF ranks are in a long format (`key<TAB>direction<TAB>rank...`, key = numeric ids `s|r|o`), so
a dedicated parser maps them to the common `{(qid, pid, qid, direction): rank}` dict through the
id -> qid maps (EXT2). Check: the AdaMF overall MRR must match `adamf_v2_kq.json` (35.572) +-0.01.
"""

from __future__ import annotations

import csv
import datetime as _dt
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_hits_table import (
    EXT2,
    OUT_DIR,
    compute_metrics,
    conf_interval,
    query_groups_v2,
    read_ranks,
)  # noqa: E402

RANKS_DIR = HERE.parents[1] / "data" / "experimental" / "v3" / "ranks_v2"
SEEDS = ("42", "2024", "7")
METRICS = ("MRR", "H@1", "H@3", "H@10", "MR")
GROUPS = ("tổng", "thường", "lát")
DATASET = "ViMMKG2"

# configuration -> (rank file template, format). "wide" = ViProKG ranks_test; "adamf" = long
# format; "dump" = luoi_c dump.
CONFIGS = [
    ("A3′* λ8 (ours)", "A3pLC_s{}/ranks_test.tsv", "wide"),
    (
        "nền text+ảnh (MENTOR)",
        "nen_l8_s{}/dump_test.tsv",
        "dump",
    ),  # lambda-8 baseline (40.89), not V203
    ("MoMoK", "momok_s{}/ranks_test.tsv", "wide"),
    ("DSoM re-impl.", "dsom_s{}/ranks_test.tsv", "wide"),
    ("AdaMF", "adamf_s{}/rank.tsv", "adamf"),
]


def read_dump(p: str) -> dict:
    """luoi_c dump: s_qid,p_pid,o_qid,huong,rank_gold -> {(s, p, o, direction): rank}."""
    out = {}
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            out[(r["s_qid"], r["p_pid"], r["o_qid"], r["huong"])] = int(float(r["rank_gold"]))
    return out


def _id_to_qid() -> tuple[dict, dict]:
    """Numeric id -> qid/pid maps (EXT2/entity2id.tsv, relation2id.tsv; same source as query_groups_v2)."""

    def read_map(fname):
        m = {}
        with open(f"{EXT2}/{fname}", encoding="utf-8") as f:
            next(f)
            for line in f:
                if line.strip():
                    v = line.rstrip("\n").split("\t")
                    m[v[1]] = v[0]
        return m

    return read_map("entity2id.tsv"), read_map("relation2id.tsv")


def read_adamf(p: str, e: dict, r: dict) -> dict:
    """AdaMF long-format ranks -> {(qid, pid, qid, direction): rank}; key = 's|r|o', direction in {head, tail}."""
    out: dict = {}
    with open(p, encoding="utf-8") as f:
        next(f)
        for line in f:
            if not line.strip():
                continue
            c = line.rstrip("\n").split("\t")
            key, huong, rank = c[0], c[1], c[2]
            si, ri, oi = key.split("|")
            k = (e[si], r[ri], e[oi], huong)
            out[k] = int(rank)
    return out


def main() -> int:
    tsv = Path(OUT_DIR)
    # back up before writing
    bak = tsv.with_name(f"{tsv.name}.bak-{_dt.date.today():%Y%m%d}-themv2")
    if not bak.exists():
        shutil.copy2(tsv, bak)
    # REPLACE: drop every row of the configurations in CONFIGS before rewriting (idempotent)
    config_names = {c[0] for c in CONFIGS}
    kept = [
        line
        for line in tsv.read_text(encoding="utf-8").splitlines()
        if not any(line.startswith(f"{t}\t{DATASET}\t") for t in config_names)
    ]
    tsv.write_text("\n".join(kept) + "\n", encoding="utf-8")
    cur = ""  # all CONFIGS rows were removed, so always rewrite
    slice_sets = query_groups_v2()
    selector = {
        "tổng": lambda k: True,
        "thường": lambda k: k[:3] in slice_sets["thường"],
        "lát": lambda k: k[:3] in slice_sets["lát"],
    }
    e, r = _id_to_qid()

    new_rows = []
    for config, template, fmt in CONFIGS:
        if f"{config}\t{DATASET}\t" in cur:
            print(f"Skip: {config} rows already exist (idempotent).")
            continue
        # load the 3 seeds
        ranks = {}
        for s in SEEDS:
            p = RANKS_DIR / template.format(s)
            if not p.exists():
                raise SystemExit(f"missing {p}: fewer than 3 seeds. STOP.")
            if fmt == "adamf":
                ranks[s] = read_adamf(str(p), e, r)
            elif fmt == "dump":
                ranks[s] = read_dump(str(p))
            else:
                ranks[s] = read_ranks(str(p))
        # check the values against the published sources
        import statistics as st

        mrr3 = st.fmean(compute_metrics(dict(ranks[s]))["MRR"] for s in SEEDS) * 100
        if fmt == "adamf":
            assert abs(mrr3 - 35.57218) < 0.01, f"AdaMF overall MRR {mrr3:.5f} != 35.572 (kq.json)"
            print(f"  AdaMF overall MRR {mrr3:.5f} matches kq.json")
        if config == "nền text+ảnh (MENTOR)":
            assert (
                abs(mrr3 - 40.89) < 0.05
            ), f"lambda-8 baseline overall MRR {mrr3:.5f} != 40.89 (nen_l8.0)"
            print(f"  lambda-8 baseline overall MRR {mrr3:.5f} matches 40.89")
        rows = []
        for group in GROUPS:
            per = {d: [] for d in METRICS}
            n_keys = None
            for s in SEEDS:
                sub = {k: v for k, v in ranks[s].items() if selector[group](k)}
                assert sub, f"{config}/{group}/s{s}: EMPTY intersection - check key normalisation"
                m = compute_metrics(sub)
                if n_keys is None:
                    n_keys = m["n"]
                assert m["n"] == n_keys, f"{config}/{group}/s{s}: n mismatch {m['n']} != {n_keys}"
                for d in METRICS:
                    per[d].append(m[d])
            for d in METRICS:
                mean, lo, hi = conf_interval(per[d])
                rows.append(
                    f"{config}\t{DATASET}\t{group}\t{d}\t{n_keys}\t{mean:.5f}\t[{lo:.5f}; {hi:.5f}]"
                )
            print(f"  {config:16} {group:7} MRR={conf_interval(per['MRR'])[0]:.5f}  (n={n_keys})")
        new_rows.extend(rows)

    if new_rows:
        with tsv.open("a", encoding="utf-8") as f:
            f.write("\n".join(new_rows) + "\n")
        print(f"added {len(new_rows)} rows to {tsv}")
    else:
        print("No new rows (all already exist).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
