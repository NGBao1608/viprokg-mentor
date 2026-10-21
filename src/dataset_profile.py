#!/usr/bin/env python3
"""In-depth profile of the three standard benchmarks DB15K / MKG-W / MKG-Y.

Only READS `repos/MyGO-main` (+ `repos/NATIVE-main` for entity / relation names; ids were checked
to coincide across the three repos). Reuses `paper_fair/code_release/degeneracy_profile.py` for
the degeneracy profile. Downloads nothing and modifies no input file.

Output: data/experimental/v3/kiemke_{db15k,mkgw,mkgy}_{tomtat,quanhe,bin,kieu}.tsv
"""
from __future__ import annotations

import csv
import json
import os
import statistics
import sys
import zipfile
from collections import Counter, defaultdict

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT_DIR, "paper_fair", "code_release"))
import degeneracy_profile as DP  # noqa: E402  (reused as is)

MYGO = os.path.join(ROOT_DIR, "repos", "MyGO-main")
OPENKE = os.path.join(ROOT_DIR, "repos", "NATIVE-main", "benchmarks")
OUT_DIR = os.path.join(ROOT_DIR, "data", "experimental", "v3")
DATASETS = [("DB15K", "db15k"), ("MKG-W", "mkgw"), ("MKG-Y", "mkgy")]
BIN = [
    (1, 1, "1"),
    (2, 5, "2-5"),
    (6, 20, "6-20"),
    (21, 100, "21-100"),
    (101, 1000, "101-1000"),
    (1001, 10**9, ">1000"),
]
# relations of TYPE kind: exact name match, fixed in advance
TYPE_RELATION_NAMES = {
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
    "http://www.wikidata.org/entity/P31",
    "P31",
    "isA",
    "rdf:type",
    "type",
}


def bin_name(n: int) -> str:
    """Name of the |A| bin."""
    for lo, hi, t in BIN:
        if lo <= n <= hi:
            return t
    raise ValueError(n)


def read_openke_names(path: str) -> dict[str, str]:
    """Read an OpenKE-style `<name> <id>` file (first line is a count) -> {id: name}."""
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        first_line = f.readline()
        if len(first_line.split()) != 1:
            f.seek(0)
        for line in f:
            a = line.rstrip("\n").split()
            if len(a) >= 2:
                out[a[-1]] = " ".join(a[:-1])
    return out


def load_split(bo: str, ten: str) -> list[tuple[str, str, str]]:
    """Read a MyGO split: each line `h\\tr\\tt` of numeric ids (as in MyGO `dataset.py`)."""
    out = []
    with open(os.path.join(MYGO, "data", bo, ten), encoding="utf-8") as f:
        for line in f:
            a = line.strip().split("\t")
            if len(a) == 3:
                out.append((a[0], a[1], a[2]))
    return out


def write_tsv(path: str, columns: list[str], rows: list[list]) -> None:
    """Write a TSV."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(columns)
        w.writerows(rows)
    print(f"  wrote {os.path.basename(path)} ({len(rows)} rows)")


def image_token_stats(bo: str) -> tuple[int, Counter]:
    """Number of keys with image tokens + distribution of images per entity (196 codes = 1 image)."""
    p = os.path.join(MYGO, "tokens", f"{bo}-visual.json.zip")
    with zipfile.ZipFile(p) as z:
        ten = [n for n in z.namelist() if not n.startswith("__")][0]
        with z.open(ten) as g:
            d = json.load(g)
    c = Counter(len(v) // 196 for v in d.values())
    n = len(d)
    del d
    return n, c


def text_token_stats(bo: str) -> tuple[str, int, Counter] | None:
    """Text tokens: which file, number of keys, length distribution."""
    for suffix in (
        "-textual.json",
        "-textual-v2.json",
        "-textual-llama.json",
        "-textual-roberta.json",
    ):
        p = os.path.join(MYGO, "tokens", bo + suffix)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            c = Counter(len(v) for v in d.values())
            n = len(d)
            del d
            return bo + suffix, n, c
    return None


def profile_dataset(bo: str, slug: str) -> dict[str, object]:
    """Profile one dataset, write 4 TSVs, return a summary for the printed table."""
    print(f"== {bo}")
    tr = load_split(bo, "train.txt")
    va = load_split(bo, "valid.txt")
    te = load_split(bo, "test.txt")
    ent = [
        l.strip()
        for l in open(os.path.join(MYGO, "data", bo, "entities.txt"), encoding="utf-8")
        if l.strip() != ""
    ]
    rel = [
        l.strip()
        for l in open(os.path.join(MYGO, "data", bo, "relations.txt"), encoding="utf-8")
        if l.strip() != ""
    ]
    id2e = read_openke_names(os.path.join(OPENKE, bo, "entity2id.txt"))
    id2r = read_openke_names(os.path.join(OPENKE, bo, "relation2id.txt"))
    all_triples = tr + va + te

    # ---- degeneracy: |A| in both directions over train + valid + test
    ah: dict[tuple[str, str], set[str]] = defaultdict(set)
    at: dict[tuple[str, str], set[str]] = defaultdict(set)
    for h, r, t in all_triples:
        ah[(r, t)].add(h)
        at[(h, r)].add(t)
    a_head = [len(ah[(r, t)]) for _h, r, t in te]
    a_tail = [len(at[(h, r)]) for h, r, _t in te]
    bin_rows = []
    for huong, a in (("head", a_head), ("tail", a_tail)):
        c = Counter(bin_name(x) for x in a)
        for _lo, _hi, b in BIN:
            bin_rows.append([huong, b, c[b], f"{c[b]/len(a):.4f}"])
        bin_rows.append([huong, "TONG", len(a), "1.0000"])
        bin_rows.append(
            [huong, "ty_le_A>1", sum(x > 1 for x in a), f"{sum(x > 1 for x in a)/len(a):.4f}"]
        )
    write_tsv(
        os.path.join(OUT_DIR, f"kiemke_{slug}_bin.tsv"),
        ["huong", "bin_A", "n_query_test", "ty_le"],
        bin_rows,
    )
    # cross-check with degeneracy_profile.py of the earlier paper (threshold 10, head direction)
    n_dp, share_dp, med_dp = DP.profile(os.path.join(MYGO, "data", bo) + "/", 10)

    # ---- relation table
    tr_r, te_r = Counter(r for _h, r, _t in tr), Counter(r for _h, r, _t in te)
    hd: dict[str, set[str]] = defaultdict(set)
    td: dict[str, set[str]] = defaultdict(set)
    for h, r, t in all_triples:
        hd[r].add(h)
        td[r].add(t)
    ah_r: dict[str, list[int]] = defaultdict(list)
    at_r: dict[str, list[int]] = defaultdict(list)
    for i, (_h, r, _t) in enumerate(te):
        ah_r[r].append(a_head[i])
        at_r[r].append(a_tail[i])
    rel_rows = []
    for r in sorted(rel, key=lambda x: -tr_r[x]):
        rel_rows.append(
            [
                r,
                id2r.get(r, ""),
                tr_r[r],
                te_r[r],
                len(hd[r]),
                len(td[r]),
                f"{statistics.median(ah_r[r]):.0f}" if ah_r[r] else "",
                f"{statistics.median(at_r[r]):.0f}" if at_r[r] else "",
                "1" if id2r.get(r, "") in TYPE_RELATION_NAMES else "0",
            ]
        )
    write_tsv(
        os.path.join(OUT_DIR, f"kiemke_{slug}_quanhe.tsv"),
        [
            "rel_id",
            "ten",
            "n_train",
            "n_test",
            "head_rieng",
            "tail_rieng",
            "A_head_trung_vi",
            "A_tail_trung_vi",
            "la_quan_he_kieu",
        ],
        rel_rows,
    )

    # ---- TYPE relation
    type_rels = [r for r in rel if id2r.get(r, "") in TYPE_RELATION_NAMES]
    type_rows = []
    if type_rels:
        rk = type_rels[0]
        pairs = [(h, t) for h, r, t in all_triples if r == rk]
        phu = {h for h, _t in pairs}
        gt = Counter(t for _h, t in pairs)
        for t, n in gt.most_common(20):
            type_rows.append([rk, id2r.get(rk, ""), t, id2e.get(t, ""), n, f"{n/len(pairs):.4f}"])
        kieu = {
            "co": True,
            "rel": id2r.get(rk, ""),
            "phu": len(phu),
            "ty_le_phu": len(phu) / len(ent),
            "K": len(gt),
            "n_triple": len(pairs),
        }
    else:
        # no type relation -> relation signature (set of relations the entity is head of in train)
        cs: dict[str, frozenset] = {}
        tmp: dict[str, set[str]] = defaultdict(set)
        for h, r, _t in tr:
            tmp[h].add(r)
        for h, s in tmp.items():
            cs[h] = frozenset(s)
        sig_counts = Counter(cs.values())
        for s, n in sig_counts.most_common(20):
            type_rows.append(
                [
                    "(chu_ky)",
                    "khong_co_quan_he_kieu",
                    "|".join(sorted(id2r.get(x, x) for x in s)),
                    "",
                    n,
                    f"{n/len(cs):.4f}",
                ]
            )
        kieu = {
            "co": False,
            "rel": "",
            "phu": len(cs),
            "ty_le_phu": len(cs) / len(ent),
            "K": len(sig_counts),
            "n_triple": 0,
        }
    write_tsv(
        os.path.join(OUT_DIR, f"kiemke_{slug}_kieu.tsv"),
        ["rel_id", "ten_quan_he_kieu", "gia_tri", "ten_gia_tri", "n", "ty_le"],
        type_rows,
    )

    # ---- leakage
    pair_train = set()
    for h, _r, t in tr:
        pair_train.add((h, t))
    tr_set = set(tr)

    def leak_rate(ds: list[tuple[str, str, str]]) -> tuple[float, float]:
        pair_hits = sum(1 for h, _r, t in ds if (h, t) in pair_train or (t, h) in pair_train)
        exact_hits = sum(1 for x in ds if x in tr_set)
        return pair_hits / len(ds), exact_hits / len(ds)

    ro_v, ngv = leak_rate(va)
    ro_t, ngt = leak_rate(te)

    # ---- features
    n_vis, pb_anh = image_token_stats(bo)
    tx = text_token_stats(bo)

    # ---- anomalies
    e_train = {x for h, _r, t in tr for x in (h, t)}
    e_test = {x for h, _r, t in te for x in (h, t)}
    e_all = {x for h, _r, t in all_triples for x in (h, t)}
    head_only = [r for r in rel if len(td[r]) == 1]
    tail_only = [r for r in rel if len(hd[r]) == 1]

    return {
        "bo": bo,
        "E": len(ent),
        "R": len(rel),
        "train": len(tr),
        "valid": len(va),
        "test": len(te),
        "id2e_mau": next(iter(id2e.values())),
        "id2r_mau": next(iter(id2r.values())),
        "dp_n": n_dp,
        "dp_share": share_dp,
        "dp_med": med_dp,
        "kieu": kieu,
        "ro_valid_cap": ro_v,
        "ro_test_cap": ro_t,
        "ng_valid": ngv,
        "ng_test": ngt,
        "n_vis": n_vis,
        "pb_anh": dict(sorted(pb_anh.items())),
        "txt": tx[0] if tx else "",
        "n_txt": tx[1] if tx else 0,
        "txt_len_pho_bien": tx[2].most_common(3) if tx else [],
        "e_khong_trong_train": len(e_test - e_train),
        "e_co_canh": len(e_all),
        "e_co_lap": len(ent) - len(e_all),
        "rel_1_tail": len(head_only),
        "rel_1_head": len(tail_only),
    }


def main() -> None:
    summary = [profile_dataset(b, s) for b, s in DATASETS]
    columns = list(summary[0].keys())
    for i, (_b, s) in enumerate(DATASETS):
        write_tsv(
            os.path.join(OUT_DIR, f"kiemke_{s}_tomtat.tsv"),
            ["chi_tieu", "gia_tri"],
            [[k, str(summary[i][k])] for k in columns],
        )
    print("\n### Summary of the 3 datasets")
    for k in columns:
        print(f"{k:22} | " + " | ".join(str(t[k]) for t in summary))


if __name__ == "__main__":
    main()
