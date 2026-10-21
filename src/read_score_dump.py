#!/usr/bin/env python3
"""Read a `dump_scores_full` blob: top-K candidates with mixed scores, plus a reference gate.

`--dump_scores_full` stores the exact FILTERED vector used for ranking (fp16, per-query max
subtracted). This script:
  1. recomputes the gold rank with the exact formula of `evaluate._filtered_rank`
     (`greater + (equal + 1) // 2`) - no new ranking rule;
  2. REFERENCE GATE: the MRR of the mention-free-pair group (edge rule v2) must equal
     `--gate_mrr +- eps` and its size must equal `--gate_n`; otherwise exit code 1. The gate also
     checks the assumption that fp16 does not change ranks: if it did, the MRR would drift;
  3. writes top-K (ids + scores) to TSV immediately so later steps read them back instead of
     re-scoring.

The `.pt` stays on the GPU host; only the top-K TSVs travel on.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch


def load_mention_edges(p: Path) -> set:
    with p.open(encoding="utf-8") as f:
        return {
            frozenset((int(a), int(b)))
            for a, b in (l.split("\t") for l in f.read().splitlines()[1:] if l)
        }


def filtered_rank(vec: torch.Tensor, gold: int) -> int:
    """Copy of `evaluate._filtered_rank`: the vector is already filtered, so only counting is left."""
    g = vec[gold].item()
    greater = int((vec > g).sum().item())
    equal = int((vec == g).sum().item())
    return greater + (equal + 1) // 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--data", required=True, help="dataset directory (entity2id.tsv)")
    ap.add_argument(
        "--edges", "--canh", required=True, help="name-mention edges v2 (a\\tb, numeric ids)"
    )
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--out_topk", "--ra_topk", required=True)
    ap.add_argument("--out_keys", "--ra_khoa", required=True)
    # `--gate_mrr <= 0` drops the MRR gate and keeps only the `n` gate. A baseline from another
    # seed is a different model, so its MRR is not the reference value; the `n` gate checks the
    # group split and stays mandatory for every seed.
    ap.add_argument(
        "--gate_mrr",
        "--cong_mrr",
        type=float,
        default=0.34681,
        help="<= 0 drops the MRR gate (baseline of another seed); the n gate always stays",
    )
    # `--gate_n <= 0` drops the n gate. Use it exactly once, for a run that MEASURES a reference
    # count that does not exist yet. The convention must live here (the consumer), not only in
    # the calling notebook.
    ap.add_argument(
        "--gate_n",
        "--cong_n",
        type=int,
        default=5968,
        help="<= 0 drops the n gate (only for the run that measures a new reference); default keeps it",
    )
    ap.add_argument("--eps", type=float, default=1e-4)
    a = ap.parse_args()
    data_dir = Path(a.data)
    print(
        f"PATHS: pt={Path(a.pt).resolve()} · data={data_dir.resolve()} · edges={Path(a.edges).resolve()}"
    )

    # torch >= 2.6 defaults to `weights_only=True`; this blob holds a list of string tuples, so it
    # usually loads either way, but older versions do not take the keyword.
    try:
        blob = torch.load(a.pt, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(a.pt, map_location="cpu")
    print(
        f"blob: split={blob.get('split')} · huong={blob.get('huong')} · "
        f"N={blob.get('N')} · keys={len(blob['keys'])}"
    )
    if blob.get("huong") != "ca_hai":
        raise SystemExit(
            f"need `--scores_full_huong ca_hai`, blob has {blob.get('huong')!r} "
            f"=> one direction is missing and every later number would cover the wrong scope"
        )
    e2i = {
        r.split("\t")[0]: int(r.split("\t")[1])
        for r in (data_dir / "entity2id.tsv").read_text(encoding="utf-8").splitlines()[1:]
    }
    edges = load_mention_edges(Path(a.edges))

    rows_topk, rows_keys = [], []
    for direction in ("head", "tail"):
        S, MX = blob.get(f"scores_{direction}"), blob.get(f"max_{direction}")
        if S is None:
            raise SystemExit(f"blob lacks scores_{direction}")
        for i, (h, r, t) in enumerate(blob["keys"]):
            gold = e2i[h] if direction == "head" else e2i[t]
            vec = S[i].float()
            gold_rank = filtered_rank(vec, gold)
            # The (h, t) PAIR of the triple itself, not the query-level flag.
            has_edge = frozenset((e2i[h], e2i[t])) in edges
            v, idx = torch.topk(vec, min(a.K, vec.numel()))
            mx = float(MX[i]) if MX is not None else 0.0
            # `q_id` = entity on the QUERY side (symmetric to gold); downstream reranking uses
            # this column directly instead of deriving it again.
            q_id = e2i[t] if direction == "head" else e2i[h]
            rows_keys.append(
                {
                    "s_qid": h,
                    "p_pid": r,
                    "o_qid": t,
                    "huong": direction,
                    "gold_id": gold,
                    "q_id": q_id,
                    "hang_gold": gold_rank,
                    "co_canh_cap": int(has_edge),
                    "s_max": mx,
                    "gold_trong_topK": int(gold_rank <= a.K),
                }
            )
            for pos, (sc, ci) in enumerate(zip(v.tolist(), idx.tolist())):
                rows_topk.append(
                    {
                        "s_qid": h,
                        "p_pid": r,
                        "o_qid": t,
                        "huong": direction,
                        "pos": pos,
                        "cand_id": ci,
                        "s_mix": sc + mx,
                        "la_gold": int(ci == gold),
                    }
                )

    def mrr(rows, select=None):
        v = [1.0 / r["hang_gold"] for r in rows if select is None or select(r)]
        return (sum(v) / len(v) if v else float("nan")), len(v)

    m_t, n_t = mrr(rows_keys)
    m_k, n_k = mrr(rows_keys, lambda r: not r["co_canh_cap"])
    m_c, n_c = mrr(rows_keys, lambda r: r["co_canh_cap"])
    print(f"\n== MRR recomputed from the dump ==")  # noqa: F541
    print(f"  total         {m_t:.5f}  (n {n_t})")
    print(f"  mention-free  {m_k:.5f}  (n {n_k})   <- GATE")
    print(f"  with-edge     {m_c:.5f}  (n {n_c})")
    print(
        f"  gold in top-{a.K}: "
        f"{sum(r['gold_trong_topK'] for r in rows_keys) / max(1, n_t):.2%} (all keys) · "
        f"{sum(r['gold_trong_topK'] for r in rows_keys if not r['co_canh_cap']) / max(1, n_k):.2%}"
        f" (mention-free)"
    )

    for p, rows, columns in (
        (
            a.out_keys,
            rows_keys,
            [
                "s_qid",
                "p_pid",
                "o_qid",
                "huong",
                "gold_id",
                "q_id",
                "hang_gold",
                "co_canh_cap",
                "s_max",
                "gold_trong_topK",
            ],
        ),
        (
            a.out_topk,
            rows_topk,
            ["s_qid", "p_pid", "o_qid", "huong", "pos", "cand_id", "s_mix", "la_gold"],
        ),
    ):
        with Path(p).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        print(f"-> {p} ({len(rows)} rows)")

    errors = []
    if a.gate_mrr <= 0:
        print(
            f"MRR gate dropped by flag (`--gate_mrr {a.gate_mrr}`): baseline of another seed, "
            f"so only the n gate applies. Measured mention-free MRR: {m_k:.5f}"
        )
    elif abs(m_k - a.gate_mrr) > a.eps:
        errors.append(
            f"mention-free MRR {m_k:.5f} off by {abs(m_k - a.gate_mrr):.2e} > {a.eps:.0e} "
            f"from gate {a.gate_mrr}"
        )
    if a.gate_n <= 0:
        # No verdict: this run MEASURES the reference. Printed loudly so the log is not read as a pass.
        print(
            f"n gate dropped by flag (`--gate_n {a.gate_n}`): measuring a new reference, nothing "
            f"to compare. Measured mention-free n: {n_k} (total {n_t} · with-edge {n_c})"
        )
    elif n_k != a.gate_n:
        errors.append(f"mention-free n {n_k} != gate {a.gate_n}")
    if errors:
        print("\nGATE FAILED - stop, report, do not loosen:")
        for l in errors:
            print("   " + l)
        print("   => this dump is NOT the published baseline; every later probe number is void.")
        return 1
    print(
        f"\nGATE PASSED: "  # noqa: F541
        + (
            f"mention-free MRR {m_k:.5f} (gate {a.gate_mrr} ±{a.eps:.0e}) · "
            if a.gate_mrr > 0
            else f"(MRR gate dropped; measured MRR {m_k:.5f}) · "
        )
        + (f"n {n_k} = {a.gate_n}" if a.gate_n > 0 else f"(n gate dropped; measured n {n_k})")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
