#!/usr/bin/env python3
"""Hits@1/@3/@10 and MR for every reported configuration, computed from per-key ranks (no GPU).

Metrics are computed with a verbatim copy of the MyGO reference implementation
(`verify/mygo_eval_reference.py`). The script also recomputes MRR and compares it with the
reported value; a deviation above 5e-5 stops the run, since Hits are only trustworthy when
MRR is reproduced from the same rank vectors.

Note: `v3_18c/v16_hang_cong_*.tsv` files without the `_TEST` suffix are VALID-split ranks;
the TEST ranks are in `v3_18d/..._TEST.tsv`.

Output: `data/experimental/v3/hits_bang.tsv` (and a printed table).
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "verify"))
import mygo_eval_reference as REF  # noqa: E402

V18C = f"{ROOT}/data/from_kaggle/v3_18c"
V18D = f"{ROOT}/data/from_kaggle/v3_18d"
V203 = f"{ROOT}/data/from_kaggle/v2_03/result"
V1B = f"{ROOT}/data/from_kaggle/v1b"
EXT2 = f"{ROOT}/data/experimental/v3/ext/ViMMKG2"
MAIN_DATASETS = ["MKG-W", "MKG-Y", "DB15K"]
SEED = [42, 7, 2024]
OUT_DIR = f"{ROOT}/data/experimental/v3/hits_bang.tsv"

# Published MRR values, used as a consistency check.
PUBLISHED_MRR = {
    ("A3′", "MKG-W"): 0.41067,
    ("A3′", "MKG-Y"): 0.44767,
    ("A3′", "DB15K"): 0.39757,
    ("A3′ + cổng", "MKG-W"): 0.41279,
    ("A3′ + cổng", "MKG-Y"): 0.45867,
    ("A3′ + cổng", "DB15K"): 0.39841,
    # original split: published 0.4026 [0.3998; 0.4054]; five digits are 0.40263.
    ("A3′ (split gốc)", "DB15K"): 0.40263,
    ("V3_19b (2 expert)", "MKG-W"): 0.42458,
    ("V3_19b (2 expert)", "MKG-Y"): 0.44796,
    ("nền text+ảnh (MENTOR)", "ViMMKG2"): 0.39084,
    ("A3′ (nền+L)", "ViMMKG2"): 0.37813,
    ("MyGO", "ViMMKG2"): 0.37974,
    ("TuckER (chỉ cấu trúc)", "ViMMKG2"): 0.36799,
}


def t_crit(df: int) -> float:
    return {1: 12.706, 2: 4.303, 6: 2.447}.get(df, 1.96)


def conf_interval(xs: list[float]) -> tuple[float, float, float]:
    assert (
        xs
    ), "conf_interval() got an EMPTY list; the group filter probably produced an empty intersection"
    xs = [float(x) for x in xs]  # np.float64 makes statistics.stdev fail obscurely
    m = statistics.mean(xs)
    if len(xs) < 2:
        return m, float("nan"), float("nan")
    se = statistics.stdev(xs) / math.sqrt(len(xs))
    h = t_crit(len(xs) - 1) * se
    return m, m - h, m + h


# ------------------------------------------------------------------ per-key rank readers
def read_sweep(p: str, g: str) -> dict:
    out = {}
    with open(p, encoding="utf-8") as f:
        head = f.readline()
        while head.startswith("#"):
            head = f.readline()
        col = head.rstrip("\n").split("\t").index(f"g{g}")
        for line in f:
            if line.strip():
                v = line.rstrip("\n").split("\t")
                out[(v[0], v[1], v[2], v[3])] = int(float(v[col]))
    return out


def read_rank_cols(p: str, col_name: str) -> tuple[dict, dict]:
    """Per-key rank TSV with a `co_canh_cap` column -> ({key: rank}, {key: has_mention})."""
    ranks, has_mention = {}, {}
    with open(p, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        i_rank, i_mention = header.index(col_name), header.index("co_canh_cap")
        for line in f:
            if not line.strip():
                continue
            v = line.rstrip("\n").split("\t")
            k = (v[0], v[1], v[2], v[3])
            ranks[k] = int(float(v[i_rank]))
            has_mention[k] = int(v[i_mention])
    return ranks, has_mention


def read_ranks(p: str) -> dict:
    """ViProKG ranks_test.tsv: one triple per row, with head and tail rank columns."""
    out = {}
    with open(p, encoding="utf-8") as f:
        next(f)
        for line in f:
            if line.strip():
                s, pr, o, hr, tr = line.rstrip("\n").split("\t")[:5]
                out[(s, pr, o, "head")] = int(hr)
                out[(s, pr, o, "tail")] = int(tr)
    return out


def g_sao(run: str) -> str:
    p = f"{run}/g_sao.json"
    if os.path.exists(p):
        return str(json.load(open(p, encoding="utf-8"))["g_sao"])
    head = [x for x in open(f"{run}/sweep_gL.tsv", encoding="utf-8") if not x.startswith("#")][0]
    g_values = [c[1:] for c in head.rstrip("\n").split("\t")[4:]]
    best_name, best_val = None, -1.0
    for g in g_values:
        m = statistics.mean(1.0 / v for v in read_sweep(f"{run}/sweep_gL.tsv", g).values())
        if m > best_val:
            best_name, best_val = g, m
    return best_name


def compute_metrics(ranks: dict) -> dict:
    """Verbatim MyGO metrics() applied to a rank vector."""
    arr = np.array(sorted(ranks.values()), dtype=float)
    mr, mrr, h10, h3, h1 = REF.metrics(arr)
    return {
        "n": len(arr),
        "MR": float(mr),
        "MRR": float(mrr),
        "H@10": float(h10),
        "H@3": float(h3),
        "H@1": float(h1),
    }


# ------------------------------------------------------------------ per-configuration loading
def _id_to_name(dataset: str, name: str) -> dict:
    m = {}
    with open(f"{ROOT}/data/experimental/v3/ext/{dataset}/{name}", encoding="utf-8") as f:
        next(f)
        for line in f:
            if line.strip():
                v = line.rstrip("\n").split("\t")
                m[v[1]] = v[0]
    return m


def load_all() -> dict:
    """{(configuration, dataset): {seed: ({key: rank}, {key: has_mention} | None)}}

    All keys are normalised to (qid, pid, qid, direction) before joining.
    """
    E = {b: _id_to_name(b, "entity2id.tsv") for b in MAIN_DATASETS}
    R = {b: _id_to_name(b, "relation2id.tsv") for b in MAIN_DATASETS}
    d: dict = {}
    for b in MAIN_DATASETS:
        for s in SEED:
            run = f"{V18C}/A3p_{b}_s{s}"
            p = f"{V18D}/v16_hang_cong_{b}_v318_{b}_A3p_{b}_s{s}_TEST.tsv"
            # The `co_canh_cap` label is a property of the triple (derived from text), not of
            # the model, so it can be shared with plain A3', whose sweep has no such column.
            # `sweep_gL_test.tsv` keys are QID/PID, `v16_hang_cong_*.tsv` keys are numeric ids;
            # map before joining, otherwise the intersection is empty (checked below).
            has_mention = read_rank_cols(p, "hang_cong")[1] if os.path.exists(p) else None
            if has_mention is not None:
                has_mention = {
                    (E[b][k[0]], R[b][k[1]], E[b][k[2]], k[3]): v for k, v in has_mention.items()
                }
            d.setdefault(("A3′", b), {})[s] = (
                read_sweep(f"{run}/sweep_gL_test.tsv", g_sao(run)),
                has_mention,
            )

            def _to_qid(t):
                h, c = t
                mp = lambda k: (E[b][k[0]], R[b][k[1]], E[b][k[2]], k[3])
                return ({mp(k): v for k, v in h.items()}, {mp(k): v for k, v in c.items()})

            if os.path.exists(p):
                d.setdefault(("A3′ + cổng", b), {})[s] = _to_qid(read_rank_cols(p, "hang_cong"))
            p2 = f"{V18D}/v16_hang_cong2_{b}_v319b_{b}_A3p_{b}_s{s}_TEST.tsv"
            if os.path.exists(p2):
                d.setdefault(("V3_19b (2 expert)", b), {})[s] = _to_qid(
                    read_rank_cols(p2, "hang_cong2")
                )
            # The main table uses the ORIGINAL DB15K split (0.4026 [0.3998; 0.4054]), not the
            # clean-valid number 0.39757, for the same protocol as external references.
            # Per-key ranks are in v1b/sweep_gL_test.tsv, same 19 804 test keys, so the
            # `co_canh_cap` label from v3_18d can be reused (it is a property of the triple).
            orig_run = f"{V1B}/A3p_orig_{b}_s{s}"
            if os.path.isdir(orig_run):
                d.setdefault(("A3′ (split gốc)", b), {})[s] = (
                    read_sweep(f"{orig_run}/sweep_gL_test.tsv", g_sao(orig_run)),
                    has_mention,
                )
    # The "nền text+ảnh (MENTOR)" baseline is NOT read from text_ViMMKG2 (V203, 39.08, wrong
    # headline); the lambda-8 baseline (40.89) is written by them_hang_v2.py, so it is skipped here.
    v2 = [
        ("A3′ (nền+L)", "A3p_ViMMKG2_s{}"),
        ("MyGO", "MyGO_ViMMKG2_s{}"),
        ("TuckER (chỉ cấu trúc)", "D4_tucker_s{}"),
    ]
    for name, template in v2:
        for s in SEED:
            g = glob.glob(f"{V203}/{template.format(s)}/*rank*.tsv")
            if g:
                d.setdefault((name, "ViMMKG2"), {})[s] = (read_ranks(g[0]), None)
    # TSAM on ViProKG (baselines/tsam/tsam_ViMMKG2_s*), read with the same read_ranks
    for s in SEED:
        p = f"{ROOT}/baselines/tsam/tsam_ViMMKG2_s{s}/ranks_test.tsv"
        if os.path.exists(p):
            d.setdefault(("TSAM", "ViMMKG2"), {})[s] = (read_ranks(p), None)
    return d


# ------------------------------------------------------------------ ViProKG groups
def query_groups_v2() -> dict:
    """{group: set of (qid, pid, qid) keys}.

    Keys are normalised before joining: `ranks_test.tsv` uses QID/PID (`Q1002930 P17 Q881`)
    while `test_lat.txt` uses numeric ids (`74 77 8238`). Joining them directly gives an empty
    intersection and NaN results instead of an error.
    """

    def read_id_map(name: str) -> dict:
        m = {}
        with open(f"{EXT2}/{name}", encoding="utf-8") as f:
            next(f)
            for line in f:
                if line.strip():
                    v = line.rstrip("\n").split("\t")
                    m[v[1]] = v[0]  # numeric id -> qid/pid
        return m

    e, r = read_id_map("entity2id.tsv"), read_id_map("relation2id.tsv")

    def read_triples(p):
        out = set()
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            h, rr, t = line.rstrip("\n").split("\t")[:3]
            out.add((e[h], r[rr], e[t]))
        return out

    return {
        "lát": read_triples(f"{EXT2}/test_lat.txt"),
        "thường": read_triples(f"{EXT2}/test_thuong.txt"),
    }


def main() -> int:
    d = load_all()
    GROUPS = query_groups_v2()
    rows, mismatches = [], []
    print("=" * 108)
    print("HITS@k + MR: all reported configurations, verbatim MyGO metrics(), 3 seeds, t-CI df=2")
    print("=" * 108)
    print(
        f"{'config':24} {'dataset':8} {'n':>6} {'MRR':>9} {'H@1':>9} {'H@3':>9} {'H@10':>9} {'MR':>9}"
    )
    for (name, b), ds in d.items():
        if len(ds) < 3:
            mismatches.append(f"{name} · {b}: only {len(ds)} seed(s)")
        per = {k: [] for k in ("MRR", "H@1", "H@3", "H@10", "MR")}
        n = 0
        for s in sorted(ds):
            m = compute_metrics(ds[s][0])
            n = m["n"]
            for k in per:
                per[k].append(m[k])
        stats = {k: conf_interval(v) for k, v in per.items()}
        # guard: recomputed MRR must match the published number
        published = PUBLISHED_MRR.get((name, b))
        if published is not None and abs(stats["MRR"][0] - published) > 5e-5:
            mismatches.append(
                f"{name} · {b}: recomputed MRR {stats['MRR'][0]:.5f} != published {published}"
            )
        print(
            f"{name:24} {b:8} {n:>6} {stats['MRR'][0]:>9.5f} {stats['H@1'][0]:>9.5f} "
            f"{stats['H@3'][0]:>9.5f} {stats['H@10'][0]:>9.5f} {stats['MR'][0]:>9.2f}"
        )
        for k in ("MRR", "H@1", "H@3", "H@10", "MR"):
            m, lo, hi = stats[k]
            rows.append((name, b, "tổng", k, n, f"{m:.5f}", f"[{lo:.5f}; {hi:.5f}]"))
        # ---- with / without mention edge: H@1 (and MRR for reference)
        if ds[sorted(ds)[0]][1] is not None:
            _h, _c = ds[sorted(ds)[0]]
            _missing = set(_h) - set(_c)
            assert not _missing, (
                f"{name}/{b}: {len(_missing)} ranked keys have no mention label "
                f"- the two files cover different key sets and cannot be joined"
            )
            for label, co in (("CÓ-cạnh", 1), ("KHÔNG-cạnh", 0)):
                pk = {k: [] for k in ("MRR", "H@1")}
                n_sub = 0
                for s in sorted(ds):
                    ranks, has_mention = ds[s]
                    sub = {k: v for k, v in ranks.items() if has_mention[k] == co}
                    sub_metrics = compute_metrics(sub)
                    n_sub = sub_metrics["n"]
                    for k in pk:
                        pk[k].append(sub_metrics[k])
                for k in pk:
                    m, lo, hi = conf_interval(pk[k])
                    rows.append((name, b, label, k, n_sub, f"{m:.5f}", f"[{lo:.5f}; {hi:.5f}]"))
        # ---- ViProKG: attribute-slice / regular split
        if b == "ViMMKG2":
            for label, key_set in GROUPS.items():
                pk = {k: [] for k in ("MRR", "H@1", "H@3", "H@10")}
                n_sub = 0
                for s in sorted(ds):
                    sub = {k: v for k, v in ds[s][0].items() if (k[0], k[1], k[2]) in key_set}
                    assert sub, f"{name}/{b}/{label}: EMPTY intersection - check key normalisation"
                    sub_metrics = compute_metrics(sub)
                    n_sub = sub_metrics["n"]
                    for k in pk:
                        pk[k].append(sub_metrics[k])
                for k in pk:
                    m, lo, hi = conf_interval(pk[k])
                    rows.append((name, b, label, k, n_sub, f"{m:.5f}", f"[{lo:.5f}; {hi:.5f}]"))
    # ---------------------------------------------------------------- check two pre-registered predictions
    ix = {(r[0], r[1], r[2], r[3]): float(r[5]) for r in rows}
    print()
    print("=" * 108)
    print("PRE-REGISTERED PREDICTIONS - recorded as holds/fails, predictions are not edited")
    print("=" * 108)
    print("\nPrediction 2: the with-mention H@1 gap is larger than the mention-free gap")
    print("  reading A: is the gap between the two groups larger at H@1 than at MRR?")
    print(f"  {'dataset':8} {'config':20} {'gap MRR':>10} {'gap H@1':>10}  verdict")
    ok_a = []
    for b in MAIN_DATASETS:
        for c in ("A3′", "A3′ + cổng", "V3_19b (2 expert)"):
            k = lambda nh, compute_metrics: ix.get((c, b, nh, compute_metrics))
            if k("CÓ-cạnh", "MRR") is None:
                continue
            gm = k("CÓ-cạnh", "MRR") - k("KHÔNG-cạnh", "MRR")
            gh = k("CÓ-cạnh", "H@1") - k("KHÔNG-cạnh", "H@1")
            ok = gh > gm
            ok_a.append(ok)
            print(f"  {b:8} {c:20} {gm:>+10.5f} {gh:>+10.5f}  {'holds' if ok else 'fails'}")
    print(f"  => reading A: {sum(ok_a)}/{len(ok_a)} hold")
    print("\n  reading B: is the gain over A3' at H@1 larger in the with-mention group?")
    print(f"  {'dataset':8} {'config':20} {'dH@1 with':>10} {'dH@1 free':>12}  verdict")
    ok_b = []
    for b in MAIN_DATASETS:
        for c in ("A3′ + cổng", "V3_19b (2 expert)"):
            if ix.get((c, b, "CÓ-cạnh", "H@1")) is None:
                continue
            dc = ix[(c, b, "CÓ-cạnh", "H@1")] - ix[("A3′", b, "CÓ-cạnh", "H@1")]
            dk = ix[(c, b, "KHÔNG-cạnh", "H@1")] - ix[("A3′", b, "KHÔNG-cạnh", "H@1")]
            ok = dc > dk
            ok_b.append(ok)
            print(f"  {b:8} {c:20} {dc:>+10.5f} {dk:>+12.5f}  {'holds' if ok else 'fails'}")
    print(f"  => reading B: {sum(ok_b)}/{len(ok_b)} hold")
    print("\nPrediction 1: our H@1 gap to the reference on MKG-Y is larger than the MRR gap")
    print("  NOT CHECKABLE: needs H@1 of the external reference, and `paper_b/sota_scan.tsv`")
    print("     has no Hits column and the repository has no reference PDFs.")

    os.makedirs(os.path.dirname(OUT_DIR), exist_ok=True)
    with open(OUT_DIR, "w", encoding="utf-8") as f:
        f.write("cau_hinh\tbo\tnhom\tdo_do\tn\ttb\tKTC95_t_df2\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    print(f"\nwrote {os.path.relpath(OUT_DIR, ROOT)} ({len(rows)} rows)")
    if mismatches:
        print("\nGUARD MISMATCH - do not use the Hits numbers until this is resolved:")
        for x in mismatches:
            print("  ", x)
        return 1
    print(
        "OK: recomputed MRR matches every published number, so Hits on the same rank vectors are trustworthy"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
