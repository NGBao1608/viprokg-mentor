"""Generate local data (no GPU) for the mention-edge, pseudo-text and BM25 steps.

G1 `v10_canh_nhac_ten_<dataset>.tsv`: edge (a, b) when the label of b appears verbatim in the
                                  text of a. Derived from TEXT, not from triples. Reports the
                                  overlap with valid/test pairs.
G2 `<dataset>-textual-pseudo.json`: real text if available; entities without text get a text
                                  generated from **train.txt** only.
X2 `v10_bm25_<dataset>.npz`:      top-100 BM25 neighbours per entity + normalised scores.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_leak_probe import DATASETS, EXT_DIR, OUT_DIR, TOKENS_DIR, normalize_words  # noqa: E402
from make_masked_text import MAX_LABEL_WORDS, load_labels  # noqa: E402,F401
from mention_only_baseline import load_label_mentions  # noqa: E402

TOP_BM25 = 100
K1, B = 1.5, 0.75  # BM25 defaults (Robertson), NOT tuned


def load_split_triples(dataset: str, split: str) -> list[tuple[int, int, int]]:
    f = "valid_sach.txt" if (dataset == "DB15K" and split == "valid") else f"{split}.txt"
    p = os.path.join(EXT_DIR, dataset, f)
    return [
        tuple(int(x) for x in r.split("\t"))
        for r in open(p, encoding="utf-8").read().splitlines()
        if r.strip()
    ]


def g1(dataset: str, mentions: dict[int, set[int]]) -> list:
    """Mention edges + overlap with valid/test pairs (REPORTED, not filtered)."""
    edges = sorted((a, b) for a, s in mentions.items() for b in s if b != a)
    train_pairs = {(h, t) for h, _r, t in load_split_triples(dataset, "train")}
    valid_test_pairs = {
        (h, t)
        for h, _r, t in load_split_triples(dataset, "valid") + load_split_triples(dataset, "test")
    }
    pairs = {(a, b) for a, b in edges} | {(b, a) for a, b in edges}
    p = os.path.join(OUT_DIR, f"v10_canh_nhac_ten_{dataset}.tsv")
    with open(p, "w", encoding="utf-8") as f:
        f.write("a\tb\n")
        for a, b in edges:
            f.write(f"{a}\t{b}\n")
    n_vt = sum(1 for e in valid_test_pairs if e in pairs)
    n_tr = sum(1 for e in train_pairs if e in pairs)
    print(
        f"  G1 {dataset:7}: {len(edges):7} edges · covers {n_tr}/{len(train_pairs)} train pairs "
        f"({100*n_tr/len(train_pairs):.1f} %) "
        f"· overlaps {n_vt}/{len(valid_test_pairs)} valid+test pairs "
        f"({100*n_vt/len(valid_test_pairs):.1f} %)",
        flush=True,
    )
    return [
        dataset,
        len(edges),
        len(train_pairs),
        n_tr,
        len(valid_test_pairs),
        n_vt,
        f"{100*n_tr/len(train_pairs):.2f}",
        f"{100*n_vt/len(valid_test_pairs):.2f}",
    ]


def g2(dataset: str, tok, labels: dict[int, list[str]], e2i: dict) -> list:
    """Pseudo-text for entities without a description; reads train.txt ONLY."""
    tk = json.load(open(os.path.join(TOKENS_DIR, f"{dataset}-textual.json"), encoding="utf-8"))
    rel_names = {
        int(r.split("\t")[1]): r.split("\t")[0]
        for r in open(os.path.join(EXT_DIR, dataset, "relation2id.tsv"), encoding="utf-8")
        .read()
        .splitlines()[1:]
    }
    desc: dict[int, str] = {}
    for k, ids in tk.items():
        i = int(k) if dataset == "DB15K" else e2i.get(k)
        if i is not None:
            desc[i] = tok.decode([x for x in ids if x < tok.vocab_size], skip_special_tokens=True)
    neighbors = defaultdict(list)
    for h, r, t in load_split_triples(dataset, "train"):  # train ONLY
        neighbors[h].append((rel_names.get(r, "rel"), t))
        neighbors[t].append(("nghich " + rel_names.get(r, "rel"), h))
    out, n_generated, n_isolated = {}, 0, 0
    _i2k = {
        (int(k) if dataset == "DB15K" else e2i.get(k)): k
        for k in ([str(v) for v in e2i.values()] if dataset == "DB15K" else list(e2i))
    }
    for k, i in ((str(v), v) for v in e2i.values()) if dataset == "DB15K" else e2i.items():
        if i in desc:
            out[k] = desc[i]
            continue
        parts = [
            " ".join(normalize_words(labels.get(i, ["entity"]) and " ".join(labels[i]) or "entity"))
        ]
        for rl, nb in neighbors.get(i, [])[:12]:
            parts.append(" ".join(normalize_words(rl)) + " " + " ".join(labels.get(nb, ["entity"])))
        out[k] = " . ".join(c for c in parts if c.strip())
        n_generated += 1
        n_isolated += int(not neighbors.get(i))
    p = os.path.join(OUT_DIR, f"{dataset}-textual-pseudo.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False)
    print(
        f"  G2 {dataset:7}: {len(out)} keys · newly generated {n_generated} · of which "
        f"{n_isolated} entities have NO train neighbours (label only)",
        flush=True,
    )
    return [dataset, len(out), n_generated, n_isolated]


def build_bm25_topk(
    dataset: str, tok, e2i: dict, tok_json: str | None = None, out: str | None = None
) -> list:
    """BM25 top-K between entity descriptions.

    `tok_json`/`out` = None: old behaviour. Given: build BM25 on the **masked text**.
    K1/B/TOP_BM25 are unchanged; only the input differs.
    """
    tk = json.load(
        open(tok_json or os.path.join(TOKENS_DIR, f"{dataset}-textual.json"), encoding="utf-8")
    )
    N = len(e2i)
    docs: dict[int, list[str]] = {}
    for k, ids in tk.items():
        i = int(k) if dataset == "DB15K" else e2i.get(k)
        if i is not None:
            docs[i] = normalize_words(
                tok.decode([x for x in ids if x < tok.vocab_size], skip_special_tokens=True)
            )
    ids_with_text = sorted(docs)
    df = Counter()
    for i in ids_with_text:
        df.update(set(docs[i]))
    n_doc = len(ids_with_text)
    avgdl = sum(len(docs[i]) for i in ids_with_text) / max(n_doc, 1)
    idf = {w: math.log(1 + (n_doc - c + 0.5) / (c + 0.5)) for w, c in df.items()}
    # inverted index: word -> [(doc, bm25 weight of that word in the doc)]
    inverted = defaultdict(list)
    for i in ids_with_text:
        tf = Counter(docs[i])
        dl = len(docs[i])
        for w, f_ in tf.items():
            w_ = idf[w] * f_ * (K1 + 1) / (f_ + K1 * (1 - B + B * dl / max(avgdl, 1e-9)))
            inverted[w].append((i, w_))
    DF_SKIP = 2000  # skip words that are too common, to bound memory
    idx = np.zeros((N, TOP_BM25), dtype=np.int32)
    val = np.zeros((N, TOP_BM25), dtype=np.float32)
    for i in ids_with_text:
        scores = defaultdict(float)
        for w in set(docs[i]):
            ds = inverted[w]
            if len(ds) > DF_SKIP:
                continue
            for j, w_ in ds:
                if j != i:
                    scores[j] += w_
        if not scores:
            continue
        top = sorted(scores.items(), key=lambda x: -x[1])[:TOP_BM25]
        mx = top[0][1] or 1.0
        for c, (j, v) in enumerate(top):
            idx[i, c] = j
            val[i, c] = v / mx  # normalised to [0,1] per row
    p = out or os.path.join(OUT_DIR, f"v10_bm25_{dataset}.npz")
    np.savez_compressed(p, idx=idx, val=val)
    n_covered = int((val.sum(1) > 0).sum())
    print(
        f"  X2 {dataset:7}: top-{TOP_BM25}, {n_covered}/{N} entities have BM25 neighbours "
        f"({100*n_covered/N:.1f} %) -> {os.path.getsize(p)/1048576:.0f} MB",
        flush=True,
    )
    return [dataset, N, n_covered, TOP_BM25]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build BM25 top-K candidate files for the lexical channel (3 standard datasets)."
    )
    ap.add_argument(
        "--text_encoder",
        default="bert-base-uncased",
        help="tokenizer: a LOCAL PATH on machines without internet, or a model name (needs internet or a cache)",
    )
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.text_encoder)
    b1, b2, b3 = [], [], []
    for dataset in DATASETS:
        print(f"== {dataset}", flush=True)
        labels, e2i = load_labels(dataset)
        mentions, _ = load_label_mentions(dataset, tok)
        b1.append(g1(dataset, mentions))
        b2.append(g2(dataset, tok, labels, e2i))
        b3.append(build_bm25_topk(dataset, tok, e2i))
    with open(os.path.join(OUT_DIR, "v10_sinh_du_lieu.tsv"), "w", encoding="utf-8") as f:
        f.write("# G1\nbo\tn_canh\tn_cap_train\tphu_train\tn_cap_vt\ttrung_vt\tpc_train\tpc_vt\n")
        for r in b1:
            f.write("\t".join(map(str, r)) + "\n")
        f.write("# G2\nbo\tn_khoa\tn_sinh_moi\tn_khong_lang_gieng\n")
        for r in b2:
            f.write("\t".join(map(str, r)) + "\n")
        f.write("# X2\nbo\tN\tn_co_lang_gieng\ttop_k\n")
        for r in b3:
            f.write("\t".join(map(str, r)) + "\n")
    print("wrote v10_sinh_du_lieu.tsv")


if __name__ == "__main__":
    sys.exit(main())
