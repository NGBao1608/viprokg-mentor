#!/usr/bin/env python3
"""OOF gate over K=3 experts {baseline (S), regularised (L), relation prior}, score level.

Compared in the same pipeline against the 2-expert gate {baseline, regularised}. One dataset.
No GPU (reads `.pt` score dumps and counts the prior).

Extends the 2-expert rank-PICK gate with a third expert, the relation prior:
`prior_scores = baseline_scores + mu * log P(e | r, dir)`, then rank again (same path as
`relation_prior.mrr_from_dump`). OOF gate: each fold learns, per group (with-edge /
mention-free), which expert has the highest MRR on the other 2 folds (VALID), and applies it to
test by group (majority over 3 folds). `(alpha, mu)` is chosen on VALID (argmax of the total MRR of
the 3-expert gate, pooled over 3 seeds), then frozen; test is only read.

Inputs: --base_dir --a3_dir (each holds scores_full_{valid,test}_s{seed}.pt) --count_triples (prior counts)
        --data <dataset dir> --bo --edges --seeds --alpha --mu --out_json
Output: JSON {gate2, gate3, nen (baseline), a3, prior} test MRR by group (3 seeds + CI df=2) + chosen (alpha, mu).
The prior is counted on `--count_triples`; the scored set (read from the blob) must be DISJOINT from
it (assert 0) - this is the leak gate.
"""
from __future__ import annotations

import argparse  # noqa: E401
import json
import math
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mention_free_screen import load_mention_edges  # noqa: E402
from relation_prior import count_relation_prior, load_triples_ids, smoothed_log_prior  # noqa: E402

T975 = {2: 4.302704, 1: 12.706}


def load_e2i(p):
    return {
        l.split("\t")[0]: int(l.split("\t")[1]) for l in Path(p).read_text().splitlines()[1:] if l
    }


def ranks_from_blob(blob, e2i, r2i=None, lp=None, mu=0.0):
    """{(s,p,o,direction): rank_gold}. lp given => add mu * lp[ri, dir] (prior expert).

    'head' dir=0 (gold=s, scores_head) · 'tail' dir=1 (gold=o, scores_tail) - same as
    relation_prior.mrr_from_dump. Rank = 1 + #{j: score_j > score_gold} (other gold answers are
    already -inf).
    """
    out = {}
    keys = blob["keys"]
    for huong, skey, gpick, dir_ in (("head", "scores_head", 0, 0), ("tail", "scores_tail", 2, 1)):
        if skey not in blob:
            continue
        S = blob[skey].float()
        for i, k in enumerate(keys):
            gcol = e2i.get(k[gpick], -1)
            if gcol < 0 or gcol >= S.shape[1]:
                continue
            row = S[i]
            if lp is not None:
                r = k[1]
                ri = int(r) if str(r).lstrip("-").isdigit() else (r2i or {}).get(r)
                if ri is None:
                    continue
                row = row + mu * lp[ri, dir_]
            out[(k[0], k[1], k[2], huong)] = int((row > row[gcol]).sum().item()) + 1
    return out


def mrr(d, ks):
    return st.fmean(1.0 / d[k] for k in ks) if ks else float("nan")


def oof_pick(experts_valid, co, vks, group_want):
    """OOF K-expert: each fold picks the expert with the highest MRR on the other 2 folds (VALID)
    for one group. Returns the majority expert name over 3 folds.
    `experts_valid` = {name: {key: rank}}."""
    import random

    rng = random.Random(42)
    fold = {k: rng.randrange(3) for k in vks}
    name0 = next(iter(experts_valid))  # FIRST expert = baseline (default on ties / empty group)
    picks = []
    for f in range(3):
        tr = [k for k in vks if fold[k] != f and co(k) == group_want]
        best, bm = name0, float("-inf")
        for name, dd in experts_valid.items():
            m = mrr(dd, tr)
            if (
                m == m and m > bm + 1e-12
            ):  # m==m skips nan (empty group); ties keep the FIRST expert
                bm, best = m, name
        picks.append(best)
    return max(set(picks), key=picks.count)


def agg(xs):
    xs = [x for x in xs if x == x]  # drop nan (empty group at one seed)
    if not xs:
        return {"mean": None, "ktc": [None, None]}
    m = st.fmean(xs)
    if len(xs) < 2:
        return {"mean": round(m, 4), "ktc": [None, None]}
    se = st.stdev(xs) / math.sqrt(len(xs))
    t = T975.get(len(xs) - 1, 1.96)
    return {
        "mean": round(m, 4),
        "ktc": [round(m - t * se, 4), round(m + t * se, 4)],
        "seeds": [round(x, 4) for x in xs],
    }


def gate_mrr(experts_v, experts_t, co, vks, tks):
    """Build the per-group gate from the experts and return (total, with-edge, mention-free) test MRR in %.
    experts_* = {name: {key: rank}}."""
    pick = {nh: oof_pick(experts_v, co, vks, want) for nh, want in (("co", True), ("khong", False))}
    gate_ranks = {k: experts_t[pick["co" if co(k) else "khong"]][k] for k in tks}
    return (
        mrr(gate_ranks, tks) * 100,
        mrr(gate_ranks, [k for k in tks if co(k)]) * 100,
        mrr(gate_ranks, [k for k in tks if not co(k)]) * 100,
    ), pick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", "--nen_dir", required=True)
    ap.add_argument("--a3_dir", required=True)
    ap.add_argument("--count_triples", "--train_dem", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--bo", "--dataset", required=True)
    ap.add_argument("--edges", "--canh", required=True)
    ap.add_argument("--seeds", default="42,2024,7")
    ap.add_argument("--out_json", "--ra", required=True)
    ap.add_argument("--alpha", nargs="*", type=float, default=[0.01, 0.1, 1.0])
    ap.add_argument("--mu", nargs="*", type=float, default=[0.1, 0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--assert_overlap", "--assert_trung", type=int, default=0)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    dd = Path(a.data)
    e2i = load_e2i(dd / "entity2id.tsv")
    r2i = {
        l.split("\t")[0]: int(l.split("\t")[1])
        for l in (dd / "relation2id.tsv").read_text().splitlines()[1:]
        if l
    }
    n_ent, n_rel = len(e2i), len(r2i)
    edges = load_mention_edges(Path(a.edges))

    def co(k):
        return frozenset((e2i.get(k[0], -1), e2i.get(k[2], -2))) in edges

    tri = load_triples_ids(Path(a.count_triples))
    c = count_relation_prior(tri, n_ent, n_rel)

    def load(d, seed, split):
        p = Path(d) / f"scores_full_{split}_s{seed}.pt"
        try:
            return torch.load(p, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(p, map_location="cpu")

    # leak gate: the scored set (test blob) must be disjoint from the prior count set
    b0 = load(a.a3_dir, seeds[0], "test")
    _scored_keys = set()
    for h, r, t in b0["keys"]:
        ri = int(r) if str(r).lstrip("-").isdigit() else r2i.get(r)
        if ri is not None and h in e2i and t in e2i:
            _scored_keys.add((e2i[h], ri, e2i[t]))
    overlap = len(set(map(tuple, tri)) & _scored_keys)
    if a.assert_overlap >= 0 and overlap != a.assert_overlap:
        raise SystemExit(
            f"LEAK: {overlap} triples of the scored set (test) are in the prior count set "
            f"!= {a.assert_overlap} => STOP."
        )
    print(
        f"[leak gate] test {len(_scored_keys)} triples · overlap with count set {overlap} "
        f"(allowed {a.assert_overlap}) OK"
    )

    # baseline / a3 ranks (independent of alpha, mu) - computed once per seed/split
    R = {}  # R[(seed,split,'nen'|'a3')] = {key:rank}
    for s in seeds:
        for split in ("valid", "test"):
            R[(s, split, "nen")] = ranks_from_blob(load(a.base_dir, s, split), e2i)
            R[(s, split, "a3")] = ranks_from_blob(load(a.a3_dir, s, split), e2i)

    # CHOOSE (alpha, mu) on VALID: argmax of the total MRR of the 3-expert gate, pooled over seeds
    best = None
    for al in a.alpha:
        lp = smoothed_log_prior(c, al)
        for mu in a.mu:
            vtot = []
            for s in seeds:
                nv = load(a.base_dir, s, "valid")
                pv = ranks_from_blob(nv, e2i, r2i, lp, mu)
                vks = sorted(set(R[(s, "valid", "nen")]) & set(R[(s, "valid", "a3")]) & set(pv))
                ev = {"nen": R[(s, "valid", "nen")], "a3": R[(s, "valid", "a3")], "prior": pv}
                (tt, _, _), _ = gate_mrr(ev, ev, co, vks, vks)  # valid scores itself
                vtot.append(tt)
            m = st.fmean(vtot)
            if best is None or m > best[2]:
                best = (al, mu, m)
    al, mu = best[0], best[1]
    lp = smoothed_log_prior(c, al)
    print(f"* {a.bo}: (alpha, mu)* = ({al}, {mu}) · 3-expert gate VALID total {best[2]:.4f}")

    # TEST: 2-expert gate {baseline, a3} vs 3-expert gate {baseline, a3, prior}, 3 seeds
    acc = {k: {"tong": [], "co": [], "khong": []} for k in ("gate2", "gate3", "nen", "a3", "prior")}
    picks3 = []
    for s in seeds:
        nv, nt = load(a.base_dir, s, "valid"), load(a.base_dir, s, "test")
        pv = ranks_from_blob(nv, e2i, r2i, lp, mu)
        pt = ranks_from_blob(nt, e2i, r2i, lp, mu)
        ev2 = {"nen": R[(s, "valid", "nen")], "a3": R[(s, "valid", "a3")]}
        et2 = {"nen": R[(s, "test", "nen")], "a3": R[(s, "test", "a3")]}
        ev3 = {**ev2, "prior": pv}
        et3 = {**et2, "prior": pt}
        vks = sorted(set.intersection(*[set(d) for d in ev3.values()]))
        tks = sorted(set.intersection(*[set(d) for d in et3.values()]))
        (g2t, g2c, g2k), _ = gate_mrr(ev2, et2, co, vks, tks)
        (g3t, g3c, g3k), p3 = gate_mrr(ev3, et3, co, vks, tks)
        picks3.append(p3)
        for nm, (tt, cc, kk) in (("gate2", (g2t, g2c, g2k)), ("gate3", (g3t, g3c, g3k))):
            acc[nm]["tong"].append(tt)
            acc[nm]["co"].append(cc)
            acc[nm]["khong"].append(kk)
        for nm, dct in (("nen", et3["nen"]), ("a3", et3["a3"]), ("prior", et3["prior"])):
            acc[nm]["tong"].append(mrr(dct, tks) * 100)
            acc[nm]["co"].append(mrr(dct, [k for k in tks if co(k)]) * 100)
            acc[nm]["khong"].append(mrr(dct, [k for k in tks if not co(k)]) * 100)

    res = {
        "bo": a.bo,
        "alpha": al,
        "mu": mu,
        "picks3_per_seed": picks3,
        **{nm: {g: agg(acc[nm][g]) for g in ("tong", "co", "khong")} for nm in acc},
    }
    # in-pipeline comparison: delta (3-expert gate - 2-expert gate), paired per seed
    for g in ("tong", "co", "khong"):
        res[f"delta_g3_g2_{g}"] = agg([x - y for x, y in zip(acc["gate3"][g], acc["gate2"][g])])
    Path(a.out_json).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(
        f"[k3] {a.bo} gate2 total {res['gate2']['tong']['mean']:.4f} · gate3 total "
        f"{res['gate3']['tong']['mean']:.4f} · delta(g3-g2) total "
        f"{res['delta_g3_g2_tong']['mean']:+.4f} {res['delta_g3_g2_tong']['ktc']}"
    )
    print(
        f"     delta(g3-g2) mention-free {res['delta_g3_g2_khong']['mean']:+.4f} "
        f"{res['delta_g3_g2_khong']['ktc']}"
    )
    print(f"-> {a.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
