#!/usr/bin/env python3
"""Screening table for the regularised configuration: validation total / mention-pair /
mention-free-pair MRR, delta against the baseline, and a pass/fail verdict.

Pre-registered criterion:
    PASS  <=>  delta(mention-free pair) >= +0.008  AND  delta(total) >= -0.003
A pair is "mention-free" when the (h, t) pair of the triple itself has no name-mention edge in
either direction; the grouping is read from `canh_nhac_ten.tsv`, never recomputed here.

Reads `sweep_gL.tsv` (validation) of every run at the `g*` column chosen on that same
validation split. The baseline comes from the reference run of the same notebook, so that both
sides of every comparison use the same instrument.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

THRESH_MENTION_FREE, THRESH_TOTAL = 0.008, -0.003

# Reference sizes of the mention-free-pair group on VALIDATION. Unit test: if the grouping
# drifts to the query-level definition instead of the pair-level one, n is off by 6.5 / 1.8 / 145x.
# Two references, one per mention-edge rule (v1 / v2); `--luat` is mandatory so the script never
# has to guess which one applies.
N_MENTION_FREE_V1 = {"MKG-W": 5216, "MKG-Y": 3552, "DB15K": 5666}
# v2 = edge rule v2 with a zipf threshold of 4.8 on the matched form and a proper-name
# whitelist. Measured with `measure_run` below (same instrument as the check itself), not copied
# from another script: other scripts count on the triple file and drop self-loops, this one
# counts on the dump keys and keeps them, so their numbers differ (e.g. MKG-W 6 028 vs 6 048).
N_MENTION_FREE_V2: "dict[str, int | None]" = {"MKG-W": 5968, "MKG-Y": 3506, "DB15K": 5706}

# Same reference on the TEST split (4 274 triples x 2 directions for MKG-W). A different split
# means a different number, so it must not be mixed with the validation one. The count does not
# depend on the seed. DB15K test has 19 804 keys (9 902 x 2) under both DB15K splits.
N_MENTION_FREE_V2_TEST: "dict[str, int | None]" = {"MKG-W": 5910, "MKG-Y": 3562, "DB15K": 14292}

REFERENCE_COUNTS = {"v1": N_MENTION_FREE_V1, "v2": N_MENTION_FREE_V2}


def read_sweep(p: Path) -> tuple[dict, list[str]]:
    """Read a sweep TSV (comment lines start with '#'); return rows and the g columns."""
    with p.open(encoding="utf-8") as f:
        rows = list(csv.DictReader((l for l in f if not l.startswith("#")), delimiter="\t"))
    g_cols = [c for c in rows[0] if c.startswith("g")]
    return rows, g_cols


def mrr(rows: list[dict], g_cols: str, select=None) -> tuple[float, int]:
    """MRR over the rows kept by `select` (all rows if None); `g_cols` is one column name."""
    v = [1.0 / float(r[g_cols]) for r in rows if select is None or select(r)]
    return (sum(v) / len(v) if v else float("nan")), len(v)


def load_mention_edges(p: Path) -> set:
    """Undirected name-mention edges as a set of frozenset({a, b}) (numeric ids, header skipped)."""
    with p.open(encoding="utf-8") as f:
        return {
            frozenset((int(a), int(b)))
            for a, b in (l.split("\t") for l in f.read().splitlines()[1:] if l)
        }


def measure_run(
    run: Path, edges: set, e2i: dict | None, sweep_name: str = "sweep_gL.tsv"
) -> dict | None:
    """Return total / mention-pair / mention-free MRR at every g column, plus the chosen `g_sao`.

    All columns are returned because the verdict is read at two values of g: the run's own `g*`
    and a COMMON g (the baseline's `g*`), which removes the artefact of the baseline sitting at a
    lower g* inside a sub-group. `sweep_name` defaults to the validation sweep; pass
    `sweep_gL_test.tsv` to measure on TEST.
    """
    sweep_path = run / sweep_name
    if not sweep_path.exists():
        return None
    rows, g_cols = read_sweep(sweep_path)

    def is_mention_pair(r):
        a = e2i.get(r["s_qid"], -1) if e2i else int(r["s_qid"])
        b = e2i.get(r["o_qid"], -2) if e2i else int(r["o_qid"])
        return frozenset((a, b)) in edges

    bang = {}
    for c in g_cols:
        t, n_t = mrr(rows, c)
        co, n_c = mrr(rows, c, is_mention_pair)
        k, n_k = mrr(rows, c, lambda r: not is_mention_pair(r))
        bang[c] = {"tong": t, "co": co, "khong": k, "n": n_t, "n_co": n_c, "n_khong": n_k}
    g_sao = max(g_cols, key=lambda c: bang[c]["tong"])
    return {"g_sao": g_sao, "bang": bang, **bang[g_sao]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", "--ket_qua", required=True)
    ap.add_argument("--out_path", "--ra", required=True)
    ap.add_argument(
        "--data", default="", help="data/<dataset> root (to read entity2id + mention edges)"
    )
    ap.add_argument(
        "--baseline_run",
        "--moc",
        default="",
        help="name pattern of the BASELINE run, e.g. `mocp_{bo}_s42`. Empty = auto-detect "
        "`cong_moi_{bo}_s42`, then `hoiquy_{bo}_s42`. The masked branch MUST pass "
        "`--baseline_run mocp_{bo}_s42` (it is scored on the new text context; comparing it "
        "to the old-tensor baseline would mix two instruments).",
    )
    # Mandatory, no default: a default here would invite comparing a v2 table against a v1
    # reference without anyone noticing.
    ap.add_argument(
        "--luat",
        "--rule",
        required=True,
        choices=["v1", "v2"],
        help="name-mention edge rule of `--canh_nhac_ten` used when the runs were produced",
    )
    # Expected number of arms (e.g. 15 = 5 configurations x 3 datasets). Fewer scored arms makes
    # `hop_le` false, so an empty pass list is never read as "no arm passed".
    ap.add_argument(
        "--n_arms",
        "--n_canh",
        type=int,
        default=0,
        help="expected number of arms; fewer scored => hop_le=False (empty list is NOT a null "
        "result); 0 = no check",
    )
    ap.add_argument(
        "--only",
        "--chi",
        nargs="*",
        default=[],
        help="score only these arms (e.g. `mask`); empty = all arms",
    )
    a = ap.parse_args()
    results = Path(a.results_dir)

    runs = sorted(d for d in results.iterdir() if d.is_dir())
    datasets_found = sorted({d.name.rsplit("_", 2)[-2] for d in runs if "_s" in d.name})
    out_path, table_rows = [], []  # noqa: F841
    for bo in datasets_found:
        data_dir = Path(a.data or results.parent / "data") / bo
        edges = (
            load_mention_edges(data_dir / "canh_nhac_ten.tsv")
            if (data_dir / "canh_nhac_ten.tsv").exists()
            else set()
        )
        e2i = None
        if (data_dir / "entity2id.tsv").exists():
            e2i = {
                r.split("\t")[0]: int(r.split("\t")[1])
                for r in (data_dir / "entity2id.tsv").read_text(encoding="utf-8").splitlines()[1:]
            }
        # Baseline run names changed over time: try the current name first, then fall back to the
        # old one so archived result packs can still be read. A missing baseline must never turn
        # into an empty pass list that reads as "0 arms passed".
        _candidates = (
            [a.baseline_run.format(bo=bo)]
            if a.baseline_run
            else [f"cong_moi_{bo}_s42", f"hoiquy_{bo}_s42"]
        )
        baseline_name = next(
            (t for t in _candidates if (results / t / "sweep_gL.tsv").exists()), None
        )
        baseline_run = measure_run(results / baseline_name, edges, e2i) if baseline_name else None
        if baseline_run is None:
            print(f"WARN {bo}: baseline run NOT found (tried {_candidates}) - dataset skipped.")
            print(f"   A skipped dataset has no rows; this is not a null result.")  # noqa: F541
            continue
        print(f"[baseline] {bo}: {baseline_name}")
        _ref_counts = REFERENCE_COUNTS[a.luat]
        if _ref_counts.get(bo) is None:
            raise SystemExit(
                f"no reference count for rule {a.luat!r} on {bo}: fill `N_MENTION_FREE_{a.luat.upper()}` "
                f"after rebuilding the edges. Refusing to run with an empty reference."
            )
        if baseline_run["n_khong"] != _ref_counts[bo]:
            raise SystemExit(
                f"reference count mismatch: {bo} mention-free n = {baseline_run['n_khong']}, "
                f"reference for rule {a.luat!r} is {_ref_counts[bo]}. Wrong group definition or "
                f"wrong edge rule - stop."
            )
        print(
            f"OK {bo} (rule {a.luat}): mention-free n {baseline_run['n_khong']} matches reference"
        )
        table_rows.append((bo, "A3′ (mốc)", baseline_run, None, None, ""))
        g_chung = baseline_run["g_sao"]  # common g = g* of the baseline
        for d in runs:
            # skip the baseline itself and gate/deterministic runs: not arms of the screening grid
            if not d.name.endswith(f"_{bo}_s42") or d.name.startswith(
                ("hoiquy", "cong_cu", "cong_moi", "tatdinh", "mocp")
            ):
                continue
            if a.only and d.name[: -len(f"_{bo}_s42")] not in a.only:
                continue
            m = measure_run(d, edges, e2i)
            if m is None:
                continue
            # (1) at the arm's own g*, against the baseline at the baseline's g*
            d_free_own = m["khong"] - baseline_run["khong"]
            d_tot_own = m["tong"] - baseline_run["tong"]
            # (2) at the COMMON g: both sides use the same column, so the g-selection gap cancels
            a_common = m["bang"].get(g_chung)
            if a_common is None:
                print(f"WARN {d.name}: missing column {g_chung} - common-g reading skipped")
                d_free_common = d_tot_common = float("nan")
            else:
                d_free_common = a_common["khong"] - baseline_run["bang"][g_chung]["khong"]
                d_tot_common = a_common["tong"] - baseline_run["bang"][g_chung]["tong"]
            pass_own = d_free_own >= THRESH_MENTION_FREE and d_tot_own >= THRESH_TOTAL
            pass_common = d_free_common >= THRESH_MENTION_FREE and d_tot_common >= THRESH_TOTAL
            # PASS only when BOTH readings satisfy the criterion.
            dat = (
                "ĐẠT"
                if (pass_own and pass_common)
                else ("nửa" if (pass_own or pass_common) else "không")
            )
            table_rows.append(
                (
                    bo,
                    d.name[: -len(f"_{bo}_s42")],
                    m,
                    (d_free_own, d_tot_own),
                    (d_free_common, d_tot_common),
                    dat,
                )
            )

    hdr = [
        "bo",
        "cau_hinh",
        "g_sao",
        "n",
        "mrr_tong",
        "n_co",
        "mrr_co",
        "n_khong",
        "mrr_khong",
        "d_khong_gsao",
        "d_tong_gsao",
        "d_khong_gchung",
        "d_tong_gchung",
        "phan",
    ]
    f2 = lambda x: "" if x is None else f"{x:+.5f}"
    with open(a.out_path, "w", encoding="utf-8") as f:
        f.write("\t".join(hdr) + "\n")
        for bo, name, m, own_gain, common_gain, dat in table_rows:
            f.write(
                "\t".join(
                    str(x)
                    for x in [
                        bo,
                        name,
                        m["g_sao"],
                        m["n"],
                        f"{m['tong']:.5f}",
                        m["n_co"],
                        f"{m['co']:.5f}",
                        m["n_khong"],
                        f"{m['khong']:.5f}",
                        f2(None if own_gain is None else own_gain[0]),
                        f2(None if own_gain is None else own_gain[1]),
                        f2(None if common_gain is None else common_gain[0]),
                        f2(None if common_gain is None else common_gain[1]),
                        dat,
                    ]
                )
                + "\n"
            )

    print(
        f"\n{'dataset':8}{'config':15}{'g*':7}{'total':>9}{'FREE':>9}"
        f"{'dFree@g*':>10}{'dTot@g*':>9}{'dFree@gc':>10}{'dTot@gc':>9}  verdict"
    )
    for bo, name, m, own_gain, common_gain, dat in table_rows:
        sr = "" if own_gain is None else f"{own_gain[0]:+10.5f}{own_gain[1]:+9.5f}"
        sc = "" if common_gain is None else f"{common_gain[0]:+10.5f}{common_gain[1]:+9.5f}"
        print(f"{bo:8}{name:15}{m['g_sao']:7}{m['tong']:9.5f}{m['khong']:9.5f}{sr}{sc}  {dat}")

    # The pass list is generated automatically from validation; never filled by hand.
    pass_list = [[bo, name] for bo, name, *_x, d in table_rows if d == "ĐẠT"]
    bo_co_moc = sorted({bo for bo, *_r in table_rows})
    bo_bi_bo = [b for b in datasets_found if b not in bo_co_moc]
    path_pass = os.path.join(os.path.dirname(a.out_path), "dat.json")
    # Also record which datasets had a baseline and how many arms were scored: otherwise
    # `"dat": []` cannot distinguish "0 arms passed" (a result) from "could not score because the
    # baseline was missing or the grid stopped early" (an infrastructure failure).
    json.dump(
        {
            "tieu_chi": {
                "nguong_khong_canh": THRESH_MENTION_FREE,
                "nguong_tong": THRESH_TOTAL,
                "luat": "ĐẠT <=> thoả ở CẢ g* riêng VÀ g chung (R1)",
            },
            "bo_co_moc": bo_co_moc,
            "bo_bi_bo": bo_bi_bo,
            "n_canh_cham": sum(
                1 for _b, _t, _m, own_gain, _c, _d in table_rows if own_gain is not None
            ),
            # `hop_le` must also check the number of arms that actually ran, not only "every
            # dataset has a baseline": a grid that stopped after 1/15 runs would otherwise
            # report an empty pass list as a genuine null result.
            "n_canh_cho_doi": a.n_arms or None,
            "hop_le": (
                (not bo_bi_bo)
                and bool(table_rows)
                and (
                    not a.n_arms
                    or sum(1 for _b, _t, _m, own_gain, _c, _d in table_rows if own_gain is not None)
                    >= a.n_arms
                )
            ),
            "dat": pass_list,
        },
        open(path_pass, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=1,
    )
    _n_scored = sum(1 for _b, _t, _m, own_gain, _c, _d in table_rows if own_gain is not None)
    if a.n_arms and _n_scored < a.n_arms:
        print(
            f"\nONLY {_n_scored}/{a.n_arms} ARMS SCORED - an empty `dat` here is NOT a null "
            f"result, the grid simply has not finished. hop_le=False."
        )
    print(f"\nwrote {path_pass}: {len(pass_list)} arms PASS -> {pass_list}")
    if bo_bi_bo or not table_rows:
        print(
            f"hop_le=false - skipped datasets: {bo_bi_bo}; scored arms: "
            f"{sum(1 for _b, _t, _m, own_gain, _c, _d in table_rows if own_gain is not None)}."
        )
        print("   An empty `dat` here is an INFRASTRUCTURE failure, not a null result.")
    n_pass = len(pass_list)
    print(
        f"\nPre-registered criterion: delta mention-free >= {THRESH_MENTION_FREE:+} "
        f"AND delta total >= {THRESH_TOTAL:+}"
    )
    print("Must hold at BOTH the arm's own `g*` and the common g (= baseline g*).")
    print(
        "  'nửa' = only one reading holds => NOT a pass, printed to expose g-selection artefacts."
    )
    print(f"PASS: {n_pass} dataset-configurations")
    print(
        "Pre-registered conclusion: K1 if >= 2 datasets have a passing arm with CI above 0 · "
        "K2 if 1 dataset · K3 if 0"
    )
    print(f"wrote {a.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
