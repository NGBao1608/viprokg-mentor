"""Acceptance gate for baseline rows: check a row directory submitted by a baseline run (see
`baselines/HOP_DONG.md`). Runs independently for each dataset.

Four checks, in order:
  (G1) `config.json` structure: are `lenh`, `seed` and a non-empty `hash_dau_vao` present? Missing = RED.
  (G2) `ranks_test.tsv`: 5-column header with the exact names, ranks >= 1, no duplicate (s,p,o)? Wrong = RED.
  (G3) T12 (mention-free group): is n equal to the frozen v2 TEST reference count
       (`mention_free_screen.N_MENTION_FREE_V2_TEST`) with tolerance 0? Mismatch = RED.
  (G4) published reference (SOFT, non-blocking): |MRR - reference| <= tolerance gives "khớp";
       outside it the row is labelled "re-impl.".

`rc = 1` if any of G1/G2/G3 is RED; `rc = 0` if G1-G3 are green (even if G4 says re-impl.).

Notes: stored per-key ranks are read; the group is measured with the same `load_mention_edges` +
frozenset of ids as the other scripts; the header is read by column NAME, not by fixed position;
the gate messages point at the real cause; the frozen T12 reference of a real dataset cannot be
overridden.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mention_free_screen import N_MENTION_FREE_V2_TEST, load_mention_edges  # noqa: E402

HEADER = ["s_qid", "p_pid", "o_qid", "head_rank", "tail_rank"]
REQUIRED_CONFIG_KEYS = ("lenh", "seed", "hash_dau_vao")


def read_ranks(p: Path) -> tuple[dict, list[str]]:
    """Return ({(s, p, o, direction): rank}, [errors]). G2 errors are collected, not raised."""
    loi: list[str] = []
    out: dict = {}
    with p.open(encoding="utf-8", newline="") as f:
        rd = csv.reader(f, delimiter="\t")
        head = next(rd, None)
        if head != HEADER:
            loi.append(f"header {head} != {HEADER}")
            return out, loi
        seen_spo: set = set()
        for i, row in enumerate(rd, start=2):
            if len(row) != 5:
                loi.append(f"line {i}: {len(row)} columns != 5")
                continue
            s, p_, o, hr, tr = row
            spo = (s, p_, o)
            if spo in seen_spo:
                loi.append(f"line {i}: duplicate (s,p,o) {spo}")
                continue
            seen_spo.add(spo)
            for direction, v in (("head", hr), ("tail", tr)):
                try:
                    r = float(v)
                except ValueError:
                    loi.append(f"line {i}: {direction} rank {v!r} is not a number")
                    continue
                if r < 1 or r != int(r):
                    loi.append(f"line {i}: {direction} rank={v} is not an integer >= 1")
                out[spo + (direction,)] = r
    if not out and not loi:
        loi.append("empty")
    return out, loi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_dir", "--hang", required=True, help="directory with ranks_test.tsv + config.json"
    )
    ap.add_argument(
        "--data", required=True, help="<dir>/<dataset>/{entity2id.tsv,canh_nhac_ten.tsv}"
    )
    ap.add_argument("--bo", required=True)
    ap.add_argument(
        "--ref_mrr",
        "--moc_goc",
        type=float,
        default=None,
        help="MRR of the original paper for G4 (soft); if absent the label is 'chưa có mốc gốc'",
    )
    ap.add_argument(
        "--tolerance", "--dung_sai", type=float, default=0.02, help="default +-2 MRR points"
    )
    ap.add_argument(
        "--expected_n_free",
        "--moc_t12",
        type=int,
        default=None,
        help="ONLY for datasets NOT in N_MENTION_FREE_V2_TEST (smoke / unknown dataset); "
        "a real dataset has a frozen reference and passing this flag stops the run",
    )
    ap.add_argument("--out", "--ra", required=True)
    a = ap.parse_args()
    run_dir, data_dir = Path(a.run_dir), Path(a.data)
    print(f"[source] {run_dir.resolve()}\n[source] {(data_dir / a.bo).resolve()}")

    result: dict = {"bo": a.bo, "do": 0}

    # -- G1: config structure --
    cfg_p = run_dir / "config.json"
    g1_errors: list[str] = []
    if not cfg_p.exists():
        g1_errors.append("missing config.json")
    else:
        try:
            cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001 - only reported, then G1 is RED
            cfg, g1_errors = {}, [f"config.json is broken: {e}"]
        for k in REQUIRED_CONFIG_KEYS:
            if not str(cfg.get(k, "")).strip():
                g1_errors.append(f"missing/empty '{k}'")
    result["G1_config"] = {"dat": not g1_errors, "loi": g1_errors}
    print(f"  [G1 config]  {'OK' if not g1_errors else 'FAIL ' + '; '.join(g1_errors)}")

    # -- G2: ranks --
    rp = run_dir / "ranks_test.tsv"
    ranks: dict = {}
    g2_errors: list[str] = []
    if not rp.exists():
        g2_errors = ["missing ranks_test.tsv"]
    else:
        ranks, g2_errors = read_ranks(rp)
    result["G2_ranks"] = {
        "dat": not g2_errors,
        "n_khoa": len(ranks),
        "loi": g2_errors[:10],
        "n_loi": len(g2_errors),
    }
    print(
        f"  [G2 ranks]   {'OK ' + str(len(ranks)) + ' keys' if not g2_errors else 'FAIL ' + '; '.join(g2_errors[:5])}"
    )

    # -- G3: T12 - only when G2 is green (the group needs valid ranks) --
    moc = N_MENTION_FREE_V2_TEST.get(a.bo)
    if moc is not None and a.expected_n_free is not None and a.expected_n_free != moc:
        raise SystemExit(
            f"'{a.bo}' is a real dataset (frozen reference {moc}); "
            f"--expected_n_free {a.expected_n_free} would override it: STOP."
        )
    if moc is None:
        if a.expected_n_free is None:
            raise SystemExit(
                f"'{a.bo}' has no frozen T12 reference; --expected_n_free is required."
            )
        moc = a.expected_n_free
    mrr = None
    if not g2_errors:
        edges = load_mention_edges(data_dir / a.bo / "canh_nhac_ten.tsv")
        e2i = {
            r.split("\t")[0]: int(r.split("\t")[1])
            for r in (data_dir / a.bo / "entity2id.tsv")
            .read_text(encoding="utf-8")
            .splitlines()[1:]
            if r
        }

        def has_edge(k) -> bool:
            return frozenset((e2i.get(k[0], -1), e2i.get(k[2], -2))) in edges

        n_free = sum(1 for k in ranks if not has_edge(k))
        mrr = st.fmean(1.0 / v for v in ranks.values())
        g3 = n_free == moc
        result["G3_t12"] = {"dat": g3, "n_khong_canh": n_free, "moc": moc}
        print(
            f"  [G3 T12]     {'OK' if g3 else 'FAIL'} n_KC={n_free} "
            f"{'' if g3 else '!= reference ' + str(moc)}"
        )
    else:
        result["G3_t12"] = {"dat": False, "ghi_chu": "G2 ĐỎ ⇒ không tính nhóm"}
        g3 = False
        print("  [G3 T12]     RED: skipped (G2 is RED)")

    # -- G4: original reference (SOFT) --
    if mrr is None:
        result["G4_moc_goc"] = {"nhan": "không đo (G2 ĐỎ)"}
    elif a.ref_mrr is None:
        result["G4_moc_goc"] = {"mrr": round(mrr, 5), "nhan": "chưa có mốc gốc"}
    else:
        lech = abs(mrr - a.ref_mrr)
        matches = lech <= a.tolerance
        result["G4_moc_goc"] = {
            "mrr": round(mrr, 5),
            "moc_goc": a.ref_mrr,
            "lech": round(lech, 5),
            "dung_sai": a.tolerance,
            "nhan": "khớp mốc gốc" if matches else "re-impl., KHÔNG khớp mốc gốc",
        }
    print(
        f"  [G4 reference] {result['G4_moc_goc'].get('nhan')} "
        f"(MRR={result['G4_moc_goc'].get('mrr')})"
    )

    nhan = result["G1_config"]["dat"] and result["G2_ranks"]["dat"] and g3
    result["nhan_duoc"] = nhan
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{'ACCEPTED' if nhan else 'REJECTED'} (G1 and G2 and G3) -> wrote {a.out}")
    return 0 if nhan else 1


if __name__ == "__main__":
    raise SystemExit(main())
