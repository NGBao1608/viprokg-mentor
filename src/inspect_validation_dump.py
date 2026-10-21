#!/usr/bin/env python3
"""Dissect the VALIDATION dump of the three external datasets (1 seed). No GPU.

Pre-registered protocol (applied as is):
- Per-query delta-RR = RR_concat - RR_base (seed 42). A single seed only describes the SHAPE of
  the distribution: no "effect / no effect" statement and no CI for delta.
- A relation gets its own row when n >= 30 per direction; the rest is pooled as "khac".
- |A| bins: {1} {2-5} {6-20} {21-100} {101-1000} {>1000}; gold train-degree bins: {<=2, 3-10, >10}.
- `m(e)` = structural role = binary vector over R, `m(e)[r] = 1` if e is the HEAD of r in train.
  "Same type" = cos(m_top1, m_gold) >= 0.5; the zero vector (entity never seen as head) counts as
  "different type".
- Four flags on the wrong cases of the concat model (rank > 1), per direction:
  (a) top-1 has the gold's type AND top-1 is in Dom(r) · (b) top-1 has a different type ·
  (c) top-1 is among the 5 most frequent heads (tails) of r in train ·
  (d) gold train degree <= 2 OR gold has no image. The external datasets have no text, so the
      "no description" half is always true and the criterion reduces to "no image".
- (A) "role": the 2 top relations hold >= 70 % of the total POSITIVE delta-RR in the head
  direction AND the degree <= 2 bin has a larger mean delta-RR than the other two bins.
  (B) "spread": the 2 top relations < 50 %, OR delta does not skew by degree. Between the two =
  undecided. Applied per dataset, no extra criterion.
- `d_vai_tro` = 1 - cos(pi_(r,t), m(gold)); pi_(r,t) = MEAN role of the known heads of (r,t) in
  train. Only a histogram + Spearman(d, delta-RR) + bootstrap CI (1 000 resamples, seed 0); NO
  bins and NO criterion are derived from this histogram.

Output: data/experimental/v3/ext/<dataset>/{delta_quanhe,delta_bin,co_4,mau_doc_tay}.tsv
        baocao/fig/ext_dvaitro.png · baocao/fig/ext_delta_quanhe.png
"""
from __future__ import annotations

import csv
import math
import os
import random
import statistics
from collections import Counter, defaultdict

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXT_DIR = os.path.join(ROOT_DIR, "data", "experimental", "v3", "ext")
RESULTS_DIR = os.path.join(ROOT_DIR, "data", "from_kaggle", "v3_ext")
FIG = os.path.join(ROOT_DIR, "baocao", "fig")
DATASETS = ["MKG-W", "MKG-Y", "DB15K"]
BIN = [
    (1, 1, "1"),
    (2, 5, "2-5"),
    (6, 20, "6-20"),
    (21, 100, "21-100"),
    (101, 1000, "101-1000"),
    (1001, 10**9, ">1000"),
]
MIN_N = 30
SAME_TYPE_COS = 0.5
N_BOOT = 1000


def bin_name(n: int) -> str:
    """Name of the |A| bin."""
    for lo, hi, t in BIN:
        if lo <= n <= hi:
            return t
    raise ValueError(n)


def degree_bin(d: int) -> str:
    """Bin of the gold train degree."""
    return "<=2" if d <= 2 else ("3-10" if d <= 10 else ">10")


def load_triples_text(p: str) -> list[tuple[str, str, str]]:
    """Read a split `h\\tr\\tt` of numeric ids."""
    out = []
    with open(p, encoding="utf-8") as f:
        for l in f:
            a = l.strip().split("\t")
            if len(a) == 3:
                out.append(tuple(a))
    return out


def split_top_names(s: str, vocab: set[str], k: int | None = None) -> list[str]:
    """Split the `top10_qid` column into entity names.

    Older dumps join with "," while YAGO/DBpedia names CAN contain commas (e.g. `Girl, Interrupted`,
    `Handa,_Aichi`), so recover by dynamic programming: split the pieces into EXACTLY `k`
    contiguous groups, each group rejoined being a real name. `k` = number of fields of
    `top10_score` (no commas), a firm constraint. Greedy longest-name matching is the fallback when
    `k` is missing or the DP has no solution. Newer dumps use U+001F (`evaluate.py: SEP`), split directly.
    """
    if "\x1f" in s:
        return s.split("\x1f")
    pieces = s.split(",")
    if k is not None and 0 < k <= len(pieces):
        # dp[(i, j)] = length of the first group when pieces[i:] splits into j valid groups
        dp: dict[tuple[int, int], int] = {}

        def solve(i: int, j: int) -> int | None:
            if (i, j) in dp:
                return dp[(i, j)] or None
            if j == 0:
                return 0 if i == len(pieces) else None
            for L in range(len(pieces) - i - (j - 1), 0, -1):  # try LONG groups first
                if ",".join(pieces[i : i + L]) in vocab and solve(i + L, j - 1) is not None:
                    dp[(i, j)] = L
                    return L
            dp[(i, j)] = 0
            return None

        if solve(0, k) is not None:
            out, i, j = [], 0, k
            while j:
                L = dp[(i, j)]
                out.append(",".join(pieces[i : i + L]))
                i += L
                j -= 1
            return out
    out, i = [], 0  # fallback: greedy longest name
    while i < len(pieces):
        for j in range(len(pieces) - 1, i - 1, -1):
            if ",".join(pieces[i : j + 1]) in vocab:
                out.append(",".join(pieces[i : j + 1]))
                i = j + 1
                break
        else:
            out.append(pieces[i])
            i += 1
    return out


def write_tsv(p: str, columns: list[str], rows: list[list]) -> None:
    """Write a TSV."""
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(columns)
        w.writerows(rows)
    print(f"   wrote {os.path.basename(p)} ({len(rows)} rows)")


def cos(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine of two sparse vectors (dict). Zero vector -> 0.0."""
    if not a or not b:
        return 0.0
    common = sum(a[k] * b[k] for k in a if k in b)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return common / (na * nb) if na and nb else 0.0


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rho."""
    from scipy.stats import spearmanr

    return float(spearmanr(x, y).statistic)


def boot(x: list[float], y: list[float], seed: int = 0) -> tuple[float, float]:
    """95% percentile CI of rho: N_BOOT bootstrap resamples of the queries."""
    rng = random.Random(seed)
    m = len(x)
    r = sorted(
        spearman([x[i] for i in idx], [y[i] for i in idx])
        for idx in ([rng.randrange(m) for _ in range(m)] for _ in range(N_BOOT))
    )
    return r[int(0.025 * N_BOOT)], r[int(0.975 * N_BOOT) - 1]


def analyse_dataset(bo: str) -> dict:
    """Dissect one dataset; return the data for plotting and the verdict."""
    print(f"\n=== {bo}")
    d = os.path.join(EXT_DIR, bo)
    tr = load_triples_text(os.path.join(d, "train.txt"))
    valid_name = "valid_sach.txt" if bo == "DB15K" else "valid.txt"  # DB15K uses the clean valid
    va = load_triples_text(os.path.join(d, valid_name))
    te = load_triples_text(os.path.join(d, "test.txt"))
    all_triples = tr + va + te  # same as the run's filter_dict
    with open(os.path.join(d, "entity2id.tsv"), encoding="utf-8") as f:
        ten_e = {r["id"]: r["qid"] for r in csv.DictReader(f, delimiter="\t")}
    with open(os.path.join(d, "relation2id.tsv"), encoding="utf-8") as f:
        ten_r = {r["id"]: r["pid"] for r in csv.DictReader(f, delimiter="\t")}
    with open(os.path.join(d, "image_index.tsv"), encoding="utf-8") as f:
        # keyed by NUMERIC id (gold in the dump was mapped back to ids), NOT by name
        has_image = {r["id"]: r["has_image"] == "1" for r in csv.DictReader(f, delimiter="\t")}

    ah, at = defaultdict(set), defaultdict(set)
    for h, r, t in all_triples:
        ah[(r, t)].add(h)
        at[(h, r)].add(t)
    deg = Counter()
    dom, ran = defaultdict(set), defaultdict(set)
    role_vec: dict[str, dict[str, float]] = defaultdict(dict)  # m(e): binary head role
    freq_h, freq_t = defaultdict(Counter), defaultdict(Counter)
    for h, r, t in tr:
        deg[h] += 1
        deg[t] += 1
        dom[r].add(h)
        ran[r].add(t)
        role_vec[h][r] = 1.0
        freq_h[r][h] += 1
        freq_t[r][t] += 1
    top5_h = {r: {e for e, _ in c.most_common(5)} for r, c in freq_h.items()}
    top5_t = {r: {e for e, _ in c.most_common(5)} for r, c in freq_t.items()}
    # pi_(r,t) = mean role of the known heads of (r,t) IN TRAIN
    head_tr = defaultdict(set)
    for h, r, t in tr:
        head_tr[(r, t)].add(h)

    def read_dump(branch: str) -> dict[tuple[str, str, str, str], dict]:
        p = os.path.join(RESULTS_DIR, f"{bo}_{branch}_s42", "dump_valid.tsv")
        out = {}
        with open(p, encoding="utf-8") as f:
            for x in csv.DictReader(f, delimiter="\t"):
                out[(x["s_qid"], x["p_pid"], x["o_qid"], x["huong"])] = x
        return out

    # the dump stores NAMES (qid/pid); map back to numeric ids to join with the graph
    id_e = {v: k for k, v in ten_e.items()}
    id_r = {v: k for k, v in ten_r.items()}
    db, dc = read_dump("base"), read_dump("concat")
    vocab = set(ten_e.values())
    assert set(db) == set(dc), f"{bo}: base/concat keys differ"

    query_rows = []
    for k in db:
        s, p, o, huong = k
        h, r, t = id_e[s], id_r[p], id_e[o]
        rb, rc = int(db[k]["rank_gold"]), int(dc[k]["rank_gold"])
        gold = h if huong == "head" else t
        size_a = len(ah[(r, t)]) if huong == "head" else len(at[(h, r)])
        n_scores = len(dc[k]["top10_score"].replace("\x1f", ",").split(","))
        top1 = id_e[split_top_names(dc[k]["top10_qid"], vocab, n_scores)[0]]
        same_type = cos(role_vec.get(top1, {}), role_vec.get(gold, {})) >= SAME_TYPE_COS
        in_domain = top1 in (dom[r] if huong == "head" else ran[r])
        is_frequent = top1 in (top5_h.get(r, set()) if huong == "head" else top5_t.get(r, set()))
        pi = defaultdict(float)
        if huong == "head":
            hs = head_tr.get((r, t), set())
            for e in hs:
                for rr in role_vec.get(e, {}):
                    pi[rr] += 1.0 / len(hs)
        dvt = 1.0 - cos(dict(pi), role_vec.get(gold, {})) if pi else float("nan")
        query_rows.append(
            {
                "h": h,
                "r": r,
                "t": t,
                "huong": huong,
                "gold": gold,
                "top1": top1,
                "rb": rb,
                "rc": rc,
                "drr": 1.0 / rc - 1.0 / rb,
                "A": size_a,
                "deg": deg.get(gold, 0),
                "d_vai_tro": dvt,
                "sai": rc > 1,
                "a": bool(rc > 1 and same_type and in_domain),
                "b": bool(rc > 1 and not same_type),
                "c": bool(rc > 1 and is_frequent),
                "d": bool(rc > 1 and (deg.get(gold, 0) <= 2 or not has_image.get(gold, False))),
            }
        )
    print(
        f"   {len(query_rows)} queries ({valid_name}) · wrong cases of concat: "
        f"{sum(x['sai'] for x in query_rows)}"
    )

    # ---- 1. table by relation + % contribution to the sum of positive delta-RR (head direction)
    columns = [
        "quan_he",
        "ten",
        "huong",
        "n",
        "MRR_base",
        "MRR_concat",
        "dRR_tb",
        "tong_dRR_duong",
        "pc_dong_gop_duong",
    ]
    rows = []
    for huong in ("head", "tail"):
        q = [x for x in query_rows if x["huong"] == huong]
        total_pos = sum(x["drr"] for x in q if x["drr"] > 0)
        by_rel = defaultdict(list)
        for x in q:
            by_rel[x["r"]].append(x)
        small = [x for r, v in by_rel.items() if len(v) < MIN_N for x in v]
        kept = [(r, v) for r, v in by_rel.items() if len(v) >= MIN_N]
        if small:
            kept.append(("khac", small))
        for r, v in kept:
            dp = sum(x["drr"] for x in v if x["drr"] > 0)
            rows.append(
                [
                    r,
                    (
                        ten_r.get(r, "khac")
                        if r != "khac"
                        else f"gộp {len(by_rel) - len(kept) + (1 if small else 0)} quan hệ n<30"
                    ),
                    huong,
                    len(v),
                    f"{statistics.mean(1/x['rb'] for x in v):.5f}",
                    f"{statistics.mean(1/x['rc'] for x in v):.5f}",
                    f"{statistics.mean(x['drr'] for x in v):+.5f}",
                    f"{dp:.5f}",
                    f"{dp/total_pos:.4f}" if total_pos else "",
                ]
            )
    rows.sort(key=lambda z: (z[2], -float(z[8] or 0)))
    write_tsv(os.path.join(d, "delta_quanhe.tsv"), columns, rows)

    # ---- 2. table by |A| bin and by degree
    cot2 = ["huong", "loai_bin", "bin", "n", "MRR_base", "MRR_concat", "dRR_tb"]
    rows2 = []
    for huong in ("head", "tail"):
        q = [x for x in query_rows if x["huong"] == huong]
        for kind, key_fn, order in (
            ("A", lambda x: bin_name(x["A"]), [b for _l, _h, b in BIN]),
            ("degree", lambda x: degree_bin(x["deg"]), ["<=2", "3-10", ">10"]),
        ):
            for b in order:
                v = [x for x in q if key_fn(x) == b]
                if not v:
                    rows2.append([huong, kind, b, 0, "", "", ""])
                    continue
                rows2.append(
                    [
                        huong,
                        kind,
                        b,
                        len(v),
                        f"{statistics.mean(1/x['rb'] for x in v):.5f}",
                        f"{statistics.mean(1/x['rc'] for x in v):.5f}",
                        f"{statistics.mean(x['drr'] for x in v):+.5f}",
                    ]
                )
    write_tsv(os.path.join(d, "delta_bin.tsv"), cot2, rows2)

    # ---- 3. cross matrix of the 4 flags on the wrong cases of concat
    cot3 = ["huong", "n_ca_sai", "co", "n", "ty_le"]
    rows3 = []
    for huong in ("head", "tail"):
        s = [x for x in query_rows if x["huong"] == huong and x["sai"]]
        for k in ("a", "b", "c", "d"):
            rows3.append(
                [
                    huong,
                    len(s),
                    k,
                    sum(x[k] for x in s),
                    f"{sum(x[k] for x in s)/len(s):.4f}" if s else "",
                ]
            )
        for k1 in ("a", "b", "c", "d"):
            for k2 in ("a", "b", "c", "d"):
                if k1 < k2:
                    n = sum(x[k1] and x[k2] for x in s)
                    rows3.append([huong, len(s), f"{k1}&{k2}", n, f"{n/len(s):.4f}" if s else ""])
        rows3.append(
            [
                huong,
                len(s),
                "khong_co_nao",
                sum(not any(x[k] for k in "abcd") for x in s),
                f"{sum(not any(x[k] for k in 'abcd') for x in s)/len(s):.4f}" if s else "",
            ]
        )
    write_tsv(os.path.join(d, "co_4.tsv"), cot3, rows3)

    # ---- 5. Spearman(d_vai_tro, delta-RR), head direction, NO bins
    hq = [x for x in query_rows if x["huong"] == "head" and not math.isnan(x["d_vai_tro"])]
    rho = (
        spearman([x["d_vai_tro"] for x in hq], [x["drr"] for x in hq])
        if len(hq) > 2
        else float("nan")
    )
    lo, hi = (
        boot([x["d_vai_tro"] for x in hq], [x["drr"] for x in hq])
        if len(hq) > 2
        else (float("nan"),) * 2
    )
    print(
        f"   d_vai_tro: n={len(hq)} · rho={rho:+.4f} · 95%-CI (bootstrap {N_BOOT}, seed 0)="
        f"[{lo:+.4f}; {hi:+.4f}]"
    )

    # ---- 6. sample of 100 wrong cases, stratified by the 4 flags
    rng = random.Random(42)
    s = [x for x in query_rows if x["sai"]]
    o = defaultdict(list)
    for x in s:
        o[tuple(x[k] for k in "abcd")].append(x)
    sample, remaining = [], 100
    for k in sorted(o, key=lambda z: -len(o[z])):
        take = max(1, round(100 * len(o[k]) / len(s)))
        sample += rng.sample(o[k], min(take, len(o[k]), remaining))
        remaining = 100 - len(sample)
        if remaining <= 0:
            break
    write_tsv(
        os.path.join(d, "mau_doc_tay.tsv"),
        [
            "huong",
            "s",
            "p",
            "o",
            "gold",
            "top1",
            "rank_base",
            "rank_concat",
            "dRR",
            "A",
            "degree_gold",
            "co_a",
            "co_b",
            "co_c",
            "co_d",
        ],
        [
            [
                x["huong"],
                ten_e[x["h"]],
                ten_r[x["r"]],
                ten_e[x["t"]],
                ten_e[x["gold"]],
                ten_e[x["top1"]],
                x["rb"],
                x["rc"],
                f"{x['drr']:+.5f}",
                x["A"],
                x["deg"],
                int(x["a"]),
                int(x["b"]),
                int(x["c"]),
                int(x["d"]),
            ]
            for x in sample[:100]
        ],
    )

    # ---- 4. apply (A)/(B)
    hd = [z for z in rows if z[2] == "head" and z[0] != "khac"]
    hd.sort(key=lambda z: -float(z[8] or 0))
    pc2 = sum(float(z[8] or 0) for z in hd[:2])
    dbin = {z[2]: float(z[6]) for z in rows2 if z[0] == "head" and z[1] == "degree" and z[6]}
    skewed = dbin.get("<=2", -9) > max(dbin.get("3-10", -9), dbin.get(">10", -9))
    if pc2 >= 0.70 and skewed:
        pq = "(A) vai trò"
    elif pc2 < 0.50 or not skewed:
        pq = "(B) trải đều"
    else:
        pq = "CHƯA PHÂN XỬ"
    print(
        f"   top-2 relations = {pc2:.1%} of the positive delta-RR sum, head "
        f"({hd[0][0]}/{ten_r.get(hd[0][0],'')}, "
        f"{hd[1][0]}/{ten_r.get(hd[1][0],'')}) · degree<=2 {dbin.get('<=2',float('nan')):+.5f} vs "
        f"3-10 {dbin.get('3-10',float('nan')):+.5f} / >10 {dbin.get('>10',float('nan')):+.5f} "
        f"· skewed={skewed}"
    )
    print(f"   => VERDICT {bo}: {pq}")
    return {
        "bo": bo,
        "d": [x["d_vai_tro"] for x in hq],
        "drr": [x["drr"] for x in hq],
        "rho": rho,
        "lo": lo,
        "hi": hi,
        "pc2": pc2,
        "pq": pq,
        "hd": hd[:8],
        "dbin": dbin,
        "ten_r": ten_r,
    }


def main() -> None:
    os.makedirs(FIG, exist_ok=True)
    results = [analyse_dataset(b) for b in DATASETS]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for a, k in zip(ax, results):
        a.hist(k["d"], bins=50, color="#1f77b4")
        a.set_title(f"{k['bo']}  (n={len(k['d'])}, ρ={k['rho']:+.3f})", fontsize=10)
        a.set_xlabel("d_vai_tro = 1 − cos(π(r,t), m(gold))")
    ax[0].set_ylabel("number of queries (head, validation)")
    fig.suptitle("Distribution of d_vai_tro (histogram only, no bins)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ext_dvaitro.png"), dpi=150)
    print(f"\nwrote {os.path.join(FIG, 'ext_dvaitro.png')}")

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a, k in zip(ax, results):
        h = k["hd"][:8][::-1]
        a.barh([f"{z[1][:22]}" for z in h], [float(z[8] or 0) * 100 for z in h], color="#d62728")
        a.set_title(f"{k['bo']} — {k['pq']} ({k['pc2']:.0%} top-2)", fontsize=10)
        a.set_xlabel("% of positive delta-RR sum (head)")
        a.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ext_delta_quanhe.png"), dpi=150)
    print(f"wrote {os.path.join(FIG, 'ext_delta_quanhe.png')}")
    print("\n=== VERDICT ===")
    for k in results:
        print(f"  {k['bo']:7} {k['pq']}")


if __name__ == "__main__":
    main()
