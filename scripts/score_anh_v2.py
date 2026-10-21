"""Image controls on ViProKG: paired difference real images - (permuted images | images off).

Input: per-key test dumps of the two branches, 3 seeds (`dump_test_real_s<seed>.tsv`, `dump_test_<tag_b>_s<seed>.tsv`
with `--tag_b perm` (default) or `--tag_b off`,
columns `s_qid p_pid o_qid huong rank_gold`). For every group (overall / ordinary / slice x all / mention-overlap /
mention-free) it reports the MRR of both branches and the paired difference delta = MRR(real) - MRR(permuted), in
points, with

* the mean over seeds and the 95% across-seed CI (t, df = 2),
* a query-level paired bootstrap CI (B = 1000): keys are resampled with replacement, each key contributes the mean over
  seeds of (1/rank_real - 1/rank_perm).

Pre-specified criterion (printed, no group thresholds): |delta overall| <= 0.65 points. The level of the real branch is
compared with the signed MENTOR_base+L value (informational only).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "data" / "experimental" / "v3"
SEEDS = (42, 7, 2024)
T975_DF2 = 4.302652729911275
THRESHOLD = 0.65
REAL_REFERENCE = 41.20  # signed MENTOR_base+L (lambda 8) total MRR on ViProKG


def read_dump(path: Path) -> dict[tuple, float]:
    """{(s_qid, p_pid, o_qid, direction): reciprocal rank}."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            out[(r["s_qid"], r["p_pid"], r["o_qid"], r["huong"])] = 1.0 / float(r["rank_gold"])
    return out


def read_id_map(path: Path, key_col: int, val_col: int) -> dict:
    m = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            if line.strip():
                v = line.rstrip("\n").split("\t")
                m[v[key_col]] = v[val_col]
    return m


def build_groups(ext: Path, edges: Path):
    """Return predicates on a key (s, p, o, direction) for slice / ordinary and overlap / free."""
    ent_id = {k: int(v) for k, v in read_id_map(ext / "entity2id.tsv", 0, 1).items()}  # qid -> id
    id_ent, id_rel = read_id_map(ext / "entity2id.tsv", 1, 0), read_id_map(
        ext / "relation2id.tsv", 1, 0
    )

    def triples(name: str) -> set:
        out = set()
        for line in open(ext / name, encoding="utf-8"):
            if line.strip():
                h, r, t = line.rstrip("\n").split("\t")[:3]
                out.add((id_ent[h], id_rel[r], id_ent[t]))
        return out

    slice_t, ordinary_t = triples("test_lat.txt"), triples("test_thuong.txt")
    with open(edges, encoding="utf-8") as f:
        next(f)
        pairs = {
            frozenset((int(a), int(b))) for a, b in (ln.split("\t")[:2] for ln in f if ln.strip())
        }

    def overlap(k) -> bool:
        return frozenset((ent_id.get(k[0], -1), ent_id.get(k[2], -2))) in pairs

    return {
        "overall": lambda k: True,
        "ordinary": lambda k: k[:3] in ordinary_t,
        "slice": lambda k: k[:3] in slice_t,
    }, {"all": lambda k: True, "overlap": overlap, "free": lambda k: not overlap(k)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dumps", help="directory with dump_test_{real,perm}_s<seed>.tsv")
    ap.add_argument("--ext", default=str(V3 / "ext" / "ViMMKG2"))
    ap.add_argument("--edges", default=str(V3 / "canh_v2" / "v10_canh_nhac_ten_ViMMKG2_v2.tsv"))
    ap.add_argument("--tag_b", default="perm", choices=("perm", "off"), help="second branch")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out_json", default="")
    a = ap.parse_args()

    tops, subs = build_groups(Path(a.ext), Path(a.edges))
    data = {
        s: {
            tag: read_dump(Path(a.dumps) / f"dump_test_{tag}_s{s}.tsv") for tag in ("real", a.tag_b)
        }
        for s in SEEDS
    }
    keys0 = set(data[SEEDS[0]]["real"])
    for s in SEEDS:  # paired design: identical key sets everywhere
        for tag in ("real", a.tag_b):
            assert set(data[s][tag]) == keys0, f"key set differs: seed {s} {tag}"
    keys = sorted(keys0)
    rng = np.random.default_rng(0)
    res: dict = {"n_keys": len(keys), "seeds": list(SEEDS), "tag_b": a.tag_b, "cells": {}}
    for tn, tp in tops.items():
        for sn, sp in subs.items():
            ks = [k for k in keys if tp(k) and sp(k)]
            if not ks:
                continue
            real = np.array([[data[s]["real"][k] for k in ks] for s in SEEDS])  # seeds x keys
            perm = np.array([[data[s][a.tag_b][k] for k in ks] for s in SEEDS])
            d_seed = (real - perm).mean(axis=1) * 100
            m, sd = float(d_seed.mean()), float(d_seed.std(ddof=1))
            half = T975_DF2 * sd / len(SEEDS) ** 0.5
            per_key = (real - perm).mean(axis=0) * 100
            idx = rng.integers(0, len(ks), size=(a.boot, len(ks)))
            boots = per_key[idx].mean(axis=1)
            res["cells"][f"{tn}|{sn}"] = {
                "n": len(ks),
                "mrr_real": float(real.mean() * 100),
                "mrr_b": float(perm.mean() * 100),
                "mrr_perm": float(perm.mean() * 100),
                "delta_per_seed": [float(x) for x in d_seed],
                "delta": m,
                "ci_seed": [m - half, m + half],
                "ci_boot": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            }
    overall = res["cells"]["overall|all"]
    res["criterion"] = {
        "text": f"|delta overall (real - {'permuted' if a.tag_b == 'perm' else a.tag_b})| <= {THRESHOLD}",
        "delta_overall": overall["delta"],
        "pass": abs(overall["delta"]) <= THRESHOLD,
    }
    res["informational_real_level"] = {
        "mrr_real": overall["mrr_real"],
        "signed_value": REAL_REFERENCE,
        "diff": overall["mrr_real"] - REAL_REFERENCE,
        "within_0.65": abs(overall["mrr_real"] - REAL_REFERENCE) <= THRESHOLD,
    }
    print(
        f"{'group':20s} {'n':>5s} {'real':>7s} {a.tag_b:>7s} {'delta':>8s}  {'CI seeds (t,df=2)':>20s}  {'CI key bootstrap':>20s}"
    )
    for name, c in res["cells"].items():
        print(
            f"{name:20s} {c['n']:5d} {c['mrr_real']:7.2f} {c['mrr_perm']:7.2f} {c['delta']:+8.3f}  "
            f"[{c['ci_seed'][0]:+.3f}; {c['ci_seed'][1]:+.3f}]  [{c['ci_boot'][0]:+.3f}; {c['ci_boot'][1]:+.3f}]"
        )
    print(
        f"\nCRITERION |delta overall| <= {THRESHOLD}: delta = {overall['delta']:+.3f} -> "
        f"{'PASS' if res['criterion']['pass'] else 'FAIL'}"
    )
    il = res["informational_real_level"]
    print(
        f"informational (no gate): real branch {il['mrr_real']:.2f} vs signed {il['signed_value']:.2f} "
        f"(diff {il['diff']:+.2f}, within 0.65: {il['within_0.65']})"
    )
    if a.out_json:
        Path(a.out_json).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
