"""Mention-only baseline (pre-registered). No GPU.

Ranking rule: candidates whose label occurs verbatim in the query-side text come FIRST; within
each block they are ordered by a text score. Reports validation MRR on the three datasets,
split by gold degree bin.

The full text-only score does not exist (only a narrow dump covering one relation, 829 queries,
and missing DB15K), so two variants are reported:
  A `ngau_nhien` - ties inside a block are broken at RANDOM. Closed form, exact, independent of
     any model, works in both directions. This is "mention only" in the literal sense.
  B `h1`        - ties broken by the scores of run H1 (same architecture as the text-only run,
     only the schedule differs; MRR delta vs the base is -0.001...+0.001 on all 3 datasets).
     This is a substitute that needs sign-off. Head direction only (the full score dump stores
     head only).

Filtering: same convention as `_filtered_rank` - other correct answers are dropped; ties give
rank `greater + (equal + 1) // 2`.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402

import inspect_validation_dump as M  # noqa: E402
from label_leak_probe import (  # noqa: E402,F401
    DATASETS,
    EXT_DIR,
    OUT_DIR,
    TOKENS_DIR,
    label_from_key,
    normalize_words,
)
from make_masked_text import MAX_LABEL_WORDS, load_labels  # noqa: E402

Q08 = os.path.join(M.ROOT_DIR, "data", "from_kaggle", "v3_08")
NEN = os.path.join(M.ROOT_DIR, "data", "from_kaggle", "v3_ext")
DEGREE_BINS = ["<=2", "3-10", ">10"]
PROXY = "H1"  # run used in place of the text score in variant B


def harmonic(n: int) -> list[float]:
    """Harmonic numbers H[0..n], used to get E[1/rank] inside a randomly ordered block."""
    h = [0.0] * (n + 1)
    for i in range(1, n + 1):
        h[i] = h[i - 1] + 1.0 / i
    return h


def load_label_mentions(dataset: str, tok) -> tuple[dict[int, set[int]], dict[int, list[str]]]:
    """mentions[e] = ids of entities whose label occurs verbatim in text(e)."""
    labels, e2i = load_labels(dataset)
    lookup: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for i, v in labels.items():
        if v and len(v) <= MAX_LABEL_WORDS:
            lookup[tuple(v)].append(i)
    max_len = max((len(k) for k in lookup), default=1)
    token_ids = json.load(
        open(os.path.join(TOKENS_DIR, f"{dataset}-textual.json"), encoding="utf-8")
    )
    mentions: dict[int, set[int]] = {}
    for k, ids in token_ids.items():
        e = int(k) if dataset == "DB15K" else e2i.get(k)
        if e is None:
            continue
        words = normalize_words(
            tok.decode([x for x in ids if x < tok.vocab_size], skip_special_tokens=True)
        )
        s: set[int] = set()
        for i in range(len(words)):
            for n in range(1, min(max_len, len(words) - i) + 1):
                g = lookup.get(tuple(words[i : i + n]))
                if g:
                    s.update(g)
        mentions[e] = s
    return mentions, labels


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Mention-only baseline (closed-form MRR, ties broken at random)."
    )
    ap.add_argument(
        "--text_encoder",
        default="bert-base-uncased",
        help="tokenizer: a LOCAL PATH on machines without internet, or a model name (needs internet or a cache)",
    )
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.text_encoder)
    rows = []
    for dataset in DATASETS:
        d = os.path.join(EXT_DIR, dataset)
        N = sum(1 for _ in open(os.path.join(d, "entities.txt"), encoding="utf-8"))
        H = harmonic(N + 1)
        tr = M.load_triples_text(os.path.join(d, "train.txt"))
        va_f = "valid_sach.txt" if dataset == "DB15K" else "valid.txt"
        va = M.load_triples_text(os.path.join(d, va_f))
        te = M.load_triples_text(os.path.join(d, "test.txt"))
        deg: dict[int, int] = defaultdict(int)
        for h, _r, t in tr:
            deg[int(h)] += 1
            deg[int(t)] += 1
        heads_of, tails_of = defaultdict(set), defaultdict(set)
        for h, r, t in tr + va + te:
            heads_of[(int(r), int(t))].add(int(h))
            tails_of[(int(h), int(r))].add(int(t))
        mentions, _labels = load_label_mentions(dataset, tok)
        print(f"-- {dataset}: N={N} - valid {len(va)} - text {len(mentions)}", flush=True)

        # ---------- A: mention flag only, random ties (closed form, both directions)
        rr: dict[str, list[float]] = {b: [] for b in DEGREE_BINS}
        total: list[float] = []
        by_direction: dict[str, list[float]] = {"head": [], "tail": []}
        for h, r, t in va:
            h, r, t = int(h), int(r), int(t)
            for direction in ("head", "tail"):
                source, gold = (t, h) if direction == "head" else (h, t)
                excluded = (heads_of[(r, t)] if direction == "head" else tails_of[(h, r)]) - {gold}
                n_cand = N - len(excluded)
                m = mentions.get(source, set())
                n1 = len(m - excluded)
                has_gold = gold in m
                if has_gold:
                    o, b = 0, max(n1, 1)
                else:
                    o, b = n1, max(n_cand - n1, 1)
                e = (H[min(o + b, N + 1)] - H[min(o, N + 1)]) / b  # E[1/rank] inside the block
                total.append(e)
                by_direction[direction].append(e)
                rr[M.degree_bin(deg.get(gold, 0))].append(e)
        for direction in ("head", "tail"):
            v = by_direction[direction]
            rows.append(
                [
                    dataset,
                    "A_ngau_nhien",
                    direction,
                    len(v),
                    f"{sum(v)/len(v):.5f}",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )
            print(f"   A {direction:4}       : MRR {sum(v)/len(v):.5f} (n={len(v)})", flush=True)
        rows.append(
            [dataset, "A_ngau_nhien", "head+tail", len(total), f"{sum(total)/len(total):.5f}"]
            + [f"{sum(rr[b])/len(rr[b]):.5f}" if rr[b] else "" for b in DEGREE_BINS]
            + [str(len(rr[b])) for b in DEGREE_BINS]
        )
        print(f"   A random ties : MRR {sum(total)/len(total):.5f} (n={len(total)})", flush=True)

        # ---------- B: mention flag + ties broken by PROXY scores (head only)
        p = os.path.join(Q08, f"{PROXY}_{dataset}_s42", "scores_full.pt")
        if not os.path.exists(p):
            print(f"   B missing {p}")
            continue
        blob = torch.load(p, map_location="cpu")
        assert blob["huong"] == "head" and blob["N"] == N
        S = blob["scores_head"].float()
        has_mention = torch.zeros(len(blob["keys"]), N, dtype=torch.bool)
        q2i = {
            q: i
            for i, q in enumerate(
                open(os.path.join(d, "entity2id.tsv"), encoding="utf-8").read().splitlines()[1:]
            )
        }
        e2i = {r.split("\t")[0]: int(r.split("\t")[1]) for r in q2i}
        gold_ids, degs = [], []
        for j, (s_q, p_q, o_q) in enumerate(blob["keys"]):
            g = e2i[s_q]
            gold_ids.append(g)
            degs.append(deg.get(g, 0))
            idx = sorted(mentions.get(e2i[o_q], set()))
            if idx:
                has_mention[j, torch.tensor(idx)] = True
        BIG = 1e4
        K = S + has_mention.float() * BIG
        K[S == float("-inf")] = float("-inf")
        gt = K[torch.arange(len(gold_ids)), torch.tensor(gold_ids)].unsqueeze(1)
        rank = (K > gt).sum(1) + ((K == gt).sum(1) + 1) // 2
        rr2: dict[str, list[float]] = {b: [] for b in DEGREE_BINS}
        inv = (1.0 / rank.float()).tolist()
        for v, dg in zip(inv, degs):
            rr2[M.degree_bin(dg)].append(v)
        rows.append(
            [dataset, f"B_hoa_{PROXY}", "head", len(inv), f"{sum(inv)/len(inv):.5f}"]
            + [f"{sum(rr2[b])/len(rr2[b]):.5f}" if rr2[b] else "" for b in DEGREE_BINS]
            + [str(len(rr2[b])) for b in DEGREE_BINS]
        )
        print(
            f"   B ties by {PROXY}    : MRR {sum(inv)/len(inv):.5f} (head, n={len(inv)})",
            flush=True,
        )
        del S, K, has_mention

    p = os.path.join(OUT_DIR, "v9a_baseline_ten.tsv")
    with open(p, "w", encoding="utf-8") as f:
        f.write(
            "bo\tbien_the\thuong\tn\tMRR\tMRR_deg_le2\tMRR_deg_3_10\tMRR_deg_tren10"
            "\tn_le2\tn_3_10\tn_tren10\n"
        )
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"wrote {p} ({len(rows)} rows)")


if __name__ == "__main__":
    sys.exit(main())
