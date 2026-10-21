#!/usr/bin/env python3
"""Relation-prior channel. No GPU.

`P(t | r, dir) = (n + alpha) / (N + alpha * |E|)` with backoff to `P(t)`; `z(prior) = log P`.
The prior is mixed linearly, with no gate, over the FULL candidate set (no top-K):
`z(S) + mu * z(prior)` - the same mixing path as the LLM-rerank script, with two differences:
no top-K restriction, and the prior is an `(r, dir) x |E|` matrix instead of a per-candidate score.

Two passes, two count sources:
 (a) SELECTION pass - the prior is counted on the pure train split (DB15K: 75 261) and scored on
     `valid_sach`, with an assert of 0 overlapping triples. `(alpha, mu)` is chosen once and
     shared by the three seeds.
 (b) TEST pass - the prior is counted on the original train split (79 222 = train + valid_sach),
     `(alpha, mu)` is frozen from (a), test is read once, and the reference is S-only of the same run.
Required wording when citing: "hyper-parameters transferred from the clean-valid protocol".
Reason: under the original split 3 961/3 961 = 100% of `valid_sach` lies inside train, so selecting
`(alpha, mu)` there would select on data the prior has already seen; leave-one-out does not fix it
because the other 3 960 triples remain in the counts.
"""
from __future__ import annotations

import argparse
import math  # noqa: F401
from pathlib import Path

import torch

GRID_ALPHA = (0.01, 0.1, 1.0)  # alpha: covers both sides
GRID_MU = (0.1, 0.25, 0.5, 1.0, 2.0)  # mu


def load_triples_ids(p: Path) -> list[tuple[int, int, int]]:
    out = []
    for l in p.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        h, r, t = l.split("\t")[:3]
        out.append((int(h), int(r), int(t)))
    return out


def count_relation_prior(
    tri: list, n_ent: int, n_rel: int, permute_relations: bool = False, seed: int = 42
) -> torch.Tensor:
    """Count `(r, dir) -> vector over |E|`. `permute_relations=True` is the BREAK TEST: shuffle relation labels.

    Returns a raw count tensor `[n_rel, 2, n_ent]` (unsmoothed); smoothing happens in
    `smoothed_log_prior` so one count serves every alpha and all alphas see the same table.
    """
    r_map = list(range(n_rel))
    if permute_relations:
        g = torch.Generator().manual_seed(seed)
        r_map = torch.randperm(n_rel, generator=g).tolist()
    c = torch.zeros(n_rel, 2, n_ent, dtype=torch.float32)
    for h, r, t in tri:
        rr = r_map[r]
        c[rr, 1, t] += 1.0  # dir=1: (h, r, ?) -> tail
        c[rr, 0, h] += 1.0  # dir=0: (?, r, t) -> head
    return c


def smoothed_log_prior(c: torch.Tensor, alpha: float) -> torch.Tensor:
    """`log P(e | r, dir)` with alpha smoothing and backoff to `P(e)` when `(r, dir)` is empty."""
    n_ent = c.shape[-1]
    tong = c.sum(-1, keepdim=True)
    p = (c + alpha) / (tong + alpha * n_ent)
    # backoff: (r, dir) with no observation uses the marginal distribution of the same count table
    marginal = c.sum((0, 1))
    pb = (marginal + alpha) / (marginal.sum() + alpha * n_ent)
    p = torch.where(tong > 0, p, pb.expand_as(p))
    return p.log()


def _count_lines(data: str) -> str:
    """Number of lines of `valid.txt` in the dataset directory, used only inside an error message.

    Wrapped in try/except: an error message must not raise on its own, otherwise it hides the
    reference-gate failure it was meant to report.
    """
    try:
        with (Path(data) / "valid.txt").open(encoding="utf-8") as f:
            return str(sum(1 for l in f if l.strip()))
    except OSError as e:
        return f"(cannot read: {e.__class__.__name__})"


def load_mention_edges_v2(p: Path) -> set:
    """Copy of `mention_free_screen.load_mention_edges` / `read_score_dump.load_mention_edges`."""
    with p.open(encoding="utf-8") as f:
        return {
            frozenset((int(x), int(y)))
            for x, y in (l.split("\t") for l in f.read().splitlines()[1:] if l)
        }


def filtered_rank(scores: torch.Tensor, gold: int) -> int:
    """Copy of `read_score_dump.filtered_rank`: the vector is already filtered, so only counting is left."""
    g = scores[gold].item()
    return 1 + int((scores > g).sum().item())


def mrr_from_dump(
    blob: dict,
    e2i: dict,
    lp: torch.Tensor | None,
    mu: float,
    r2i: dict | None = None,
    edges: set | None = None,
) -> dict:
    """MRR over the FULL candidate set, split by with-edge / mention-free (edge rule v2).

    `lp=None` gives S-only (the baseline). Groups use the (h, t) PAIR of the triple itself, as in
    `read_score_dump`: `frozenset((e2i[h], e2i[t])) in edges`.
    `r2i` maps pid -> numeric id: `blob["keys"]` stores relations as PID strings in some dumps and
    as numeric ids in others, so they must be normalised before indexing `lp`.
    """
    acc = {"tong": [0.0, 0], "co": [0.0, 0], "khong": [0.0, 0]}
    for ch, dir_ in (("head", 0), ("tail", 1)):
        S = blob.get(f"scores_{ch}")
        if S is None:
            continue
        for i, (h, r, t) in enumerate(blob["keys"]):
            gold = e2i[h] if ch == "head" else e2i[t]
            v = S[i].float()
            if lp is not None:
                ri = int(r) if str(r).lstrip("-").isdigit() else (r2i or {}).get(r)
                if ri is None:
                    raise SystemExit(
                        f"cannot map relation {r!r} to an id; "
                        f"pass `--data` with the `relation2id.tsv` of the right dataset"
                    )
                v = v + mu * lp[ri, dir_]
            rr = 1.0 / filtered_rank(v, gold)
            acc["tong"][0] += rr
            acc["tong"][1] += 1
            if edges is not None:
                k = "co" if frozenset((e2i[h], e2i[t])) in edges else "khong"
                acc[k][0] += rr
                acc[k][1] += 1
    out = {"mrr": acc["tong"][0] / max(1, acc["tong"][1]), "n": acc["tong"][1]}
    for k in ("co", "khong"):
        out[f"mrr_{k}"] = acc[k][0] / acc[k][1] if acc[k][1] else float("nan")
        out[f"n_{k}"] = acc[k][1]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--count_triples", "--train_dem", required=True, help="triple file the prior is COUNTED on"
    )
    ap.add_argument("--pt", required=True, help="dump scores_full (.pt)")
    ap.add_argument("--data", required=True, help="dataset directory (entity2id.tsv)")
    ap.add_argument(
        "--scored_file",
        "--khoa_cham",
        default="",
        help="triple file of the split being SCORED - to assert no overlap with --count_triples",
    )
    ap.add_argument(
        "--assert_overlap",
        "--assert_trung",
        type=int,
        default=-1,
        help="allowed number of overlapping triples; 0 = selection pass; -1 = no check",
    )
    ap.add_argument("--alpha", nargs="*", type=float, default=list(GRID_ALPHA))
    ap.add_argument("--mu", nargs="*", type=float, default=list(GRID_MU))
    ap.add_argument(
        "--permute_relations",
        "--hoan_vi",
        type=int,
        default=0,
        help="1 = BREAK TEST: shuffle relation labels",
    )
    ap.add_argument(
        "--edges",
        "--canh",
        default="",
        help="name-mention edges v2 => split with-edge / mention-free",
    )
    # Reference gate: stop BEFORE mixing if the baseline is not the expected dataset.
    ap.add_argument("--ref_n", "--moc_n", type=int, default=0, help="expected n keys; 0 = skip")
    ap.add_argument(
        "--ref_mrr", "--moc_mrr", type=float, default=0.0, help="expected baseline MRR; 0 = skip"
    )
    ap.add_argument(
        "--ref_n_free",
        "--moc_nkc",
        type=int,
        default=0,
        help="expected mention-free n; 0 = skip",
    )
    ap.add_argument("--ref_eps", "--moc_eps", type=float, default=0.01)
    ap.add_argument("--out_tsv", "--ra", default="")
    # `--out_3seed`: write a TSV in the exact format of the 3-seed table script, so that script
    # can be called as is (it reads columns `{group}_{metric}` with group in {tat, co, khong} plus
    # `bien_the`, `lam`, `khong_n`). An adapter is used instead of recomputing the CI here:
    # the other script already has the correct t quantile for df=2.
    ap.add_argument("--out_3seed", "--ra_3seed", default="", help="TSV in the 3-seed table format")
    ap.add_argument(
        "--variant_name", "--bien_the_ten", default="PRIOR", help="variant label in --out_3seed"
    )
    a = ap.parse_args()

    d = Path(a.data)
    e2i = {
        r.split("\t")[0]: int(r.split("\t")[1])
        for r in (d / "entity2id.tsv").read_text(encoding="utf-8").splitlines()[1:]
    }
    n_ent = len(e2i)
    n_rel = len((d / "relation2id.tsv").read_text(encoding="utf-8").splitlines()) - 1
    tri = load_triples_ids(Path(a.count_triples))
    print(
        f"COUNT prior on {Path(a.count_triples).resolve()} - {len(tri)} triples · "
        f"|E|={n_ent} · |R|={n_rel}"
    )

    c = count_relation_prior(tri, n_ent, n_rel, bool(a.permute_relations))
    if a.permute_relations:
        print("   BREAK TEST: relation labels PERMUTED => delta must be <= 0")
    try:
        blob = torch.load(a.pt, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(a.pt, map_location="cpu")
    print(f"blob: split={blob.get('split')} · huong={blob.get('huong')} · keys={len(blob['keys'])}")

    r2i = {
        r.split("\t")[0]: int(r.split("\t")[1])
        for r in (d / "relation2id.tsv").read_text(encoding="utf-8").splitlines()[1:]
    }
    edges = load_mention_edges_v2(Path(a.edges)) if a.edges else None
    # ---- LEAK GATE: compare against the set ACTUALLY scored (read from the blob) ----
    # An earlier version compared the count set against a FILE (`--scored_file`), i.e. against the
    # set we BELIEVED was scored; it passed while the dump scored the ORIGINAL 9 904-triple valid
    # split, 88% of which lies inside train. The gate must read `blob["keys"]`.
    _scored_keys = set()
    for h, r, t in blob["keys"]:
        ri = int(r) if str(r).lstrip("-").isdigit() else r2i.get(r)
        if ri is None or h not in e2i or t not in e2i:
            continue
        _scored_keys.add((e2i[h], ri, e2i[t]))
    if a.assert_overlap >= 0:
        overlap = len(set(map(tuple, tri)) & _scored_keys)
        print(
            f"   [leak gate] scored set ACTUALLY used (from blob) {len(_scored_keys)} triples · "
            f"overlap with count set: {overlap} (allowed {a.assert_overlap})"
        )
        if overlap != a.assert_overlap:
            raise SystemExit(
                f"LEAK: {overlap}/{len(_scored_keys)} triples of the SCORED set are in the count set "
                f"!= {a.assert_overlap} => STOP.\n"
                f"   count set  = {Path(a.count_triples).name} ({len(tri)} triples)\n"
                f"   scored set = blob {a.pt} (split={blob.get('split')}, "
                f"{len(blob['keys'])} keys)\n"
                f"   => check whether `valid.txt` in `--data` really is `valid_sach`."
            )

    # ---- REFERENCE GATE - stop BEFORE mixing ----
    base = mrr_from_dump(blob, e2i, None, 0.0, r2i, edges)
    _errors = []
    if a.ref_n and base["n"] != a.ref_n:
        _errors.append(f"n keys {base['n']} != reference {a.ref_n}")
    if a.ref_mrr and abs(base["mrr"] - a.ref_mrr) > a.ref_eps:
        _errors.append(
            f"baseline MRR {base['mrr']:.5f} off by {abs(base['mrr'] - a.ref_mrr):.5f} "
            f"> {a.ref_eps} from reference {a.ref_mrr}"
        )
    if a.ref_n_free and base.get("n_khong") != a.ref_n_free:
        _errors.append(f"mention-free n {base.get('n_khong')} != reference {a.ref_n_free}")
    if _errors:
        raise SystemExit(
            "REFERENCE GATE FAILED - STOP BEFORE MIXING:\n   "
            + "\n   ".join(_errors)
            + f"\n   scored blob: {a.pt} (split={blob.get('split')}, "
            f"{len(blob['keys'])} keys · huong={blob.get('huong')})\n"
            f"   `valid.txt` in {Path(a.data).resolve()} has {_count_lines(a.data)} lines\n"
            f"   => DB15K clean-valid reference: n 7922, MRR ~ 0.4020, mention-free n 5706. "
            f"n = 19808 means the ORIGINAL 9904-triple valid split is being scored."
        )
    if a.ref_n or a.ref_mrr or a.ref_n_free:
        print(
            f"   [reference gate] PASSED - n {base['n']} · baseline MRR {base['mrr']:.5f} · "
            f"mention-free n {base.get('n_khong')}"
        )
    print(
        f"\nS-only (baseline): MRR {base['mrr']:.5f} · n {base['n']}"
        + (
            f" · with-edge {base['mrr_co']:.5f} (n {base['n_co']})"
            f" · mention-free {base['mrr_khong']:.5f} (n {base['n_khong']})"
            if edges
            else ""
        )
    )
    rows, best = [], None
    for al in a.alpha:
        lp = smoothed_log_prior(c, al)
        for mu in a.mu:
            m = mrr_from_dump(blob, e2i, lp, mu, r2i, edges)
            delta = m["mrr"] - base["mrr"]
            d_with = m.get("mrr_co", float("nan")) - base.get("mrr_co", float("nan"))
            d_free = m.get("mrr_khong", float("nan")) - base.get("mrr_khong", float("nan"))
            rows.append((al, mu, m["mrr"], delta, d_with, d_free))
            print(
                f"  alpha={al:<5} mu={mu:<5} MRR {m['mrr']:.5f}  delta {delta:+.5f}"
                + (f"  d_with {d_with:+.5f}  d_free {d_free:+.5f}" if edges else "")
            )
            if best is None or delta > best[3]:
                best = (al, mu, m["mrr"], delta)
    print(
        f"\nbest (alpha, mu): alpha={best[0]} mu={best[1]} · MRR {best[2]:.5f} · delta {best[3]:+.5f}"
    )
    # Also report delta at the FIXED mu=2: delta at the optimum cannot tell "no signal" from "harmful".
    at = [r for r in rows if r[1] == 2.0]
    if at:
        b = max(at, key=lambda r: r[3])
        print(
            f"  delta @ fixed mu=2 (best alpha {b[0]}): {b[3]:+.5f}   [read together with delta@mu*]"
        )
    if a.out_tsv:
        with open(a.out_tsv, "w", encoding="utf-8") as f:
            f.write("alpha\tmu\tmrr\tdelta\td_co\td_khong\tmrr_nen\tn\tn_co\tn_khong\thoan_vi\n")
            for al, mu, mr, delta, d_with, d_free in rows:
                f.write(
                    f"{al}\t{mu}\t{mr:.6f}\t{delta:.6f}\t{d_with:.6f}\t{d_free:.6f}"
                    f"\t{base['mrr']:.6f}\t{base['n']}\t{base.get('n_co', '')}"
                    f"\t{base.get('n_khong', '')}\t{a.permute_relations}\n"
                )
        print(f"-> {a.out_tsv} ({len(rows)} rows)")

    if a.out_3seed:
        # Only the frozen (alpha, mu) pair is written - the 3-seed round does not re-select.
        if len(rows) != 1:
            raise SystemExit(
                f"`--out_3seed` needs EXACTLY ONE frozen (alpha, mu) pair, got "
                f"{len(rows)} => pass a single `--alpha X --mu Y`."
            )
        al, mu, mr, delta, d_with, d_free = rows[0]
        with open(a.out_3seed, "w", encoding="utf-8") as f:
            columns = [
                "bien_the",
                "lam",
                "cach_doc",
                "alpha",
                "tat_delta",
                "co_delta",
                "khong_delta",
                "tat_nen_mrr",
                "co_nen_mrr",
                "khong_nen_mrr",
                "tat_sau_mrr",
                "co_sau_mrr",
                "khong_sau_mrr",
                "tat_n",
                "co_n",
                "khong_n",
            ]
            f.write("\t".join(columns) + "\n")
            v = {
                "bien_the": a.variant_name,
                "lam": mu,
                "cach_doc": "lamchung",
                "alpha": al,
                "tat_delta": delta,
                "co_delta": d_with,
                "khong_delta": d_free,
                "tat_nen_mrr": base["mrr"],
                "co_nen_mrr": base.get("mrr_co", ""),
                "khong_nen_mrr": base.get("mrr_khong", ""),
                "tat_sau_mrr": base["mrr"] + delta,
                "co_sau_mrr": (base.get("mrr_co", 0) + d_with) if edges else "",
                "khong_sau_mrr": (base.get("mrr_khong", 0) + d_free) if edges else "",
                "tat_n": base["n"],
                "co_n": base.get("n_co", ""),
                "khong_n": base.get("n_khong", ""),
            }
            f.write("\t".join(str(v[c]) for c in columns) + "\n")
        print(f"-> {a.out_3seed} (3-seed table format, variant {a.variant_name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
