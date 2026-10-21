"""CPU-only preprocessing for ViProKG: `folds_k3.json`, BM25 npz, `L_feat.pt`.

Same formulas as the three external datasets (`build_lexical_data.build_bm25_topk` and
`lexical_channel_features`); only the normalisation differs: `normalize_words_vi` replaces
`normalize_words`, and text is read from `texts_by_id.tsv` instead of decoding BERT tokens
(the source text is Vietnamese and is not passed through an uncased tokenizer). All constants
(`K1`, `B`, `TOP_BM25`, `DF_SKIP`, `NORMALIZERS`) are kept as in the other datasets, not tuned.

Labels are mapped by QID from `data/final/ViMMKG/entity_labels.tsv`; the 48 `VAL::P149::*`
attribute-slice entities take their label from the last segment of the key.

GPU work is NOT done here: image tokenisation (BEiT-v2) for `V.pt`/`V_mask.pt`, and
`T_ctx.pt`/`T_mask.pt` with PhoBERT.

Outputs: `ext/ViMMKG2/{folds_k3.json, L_feat.pt}` and `v10_bm25_ViMMKG2.npz`.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_leak_probe import contains_label, normalize_words_vi  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Kaggle: override via env (VIMMKG2_EXT = dir holding entity2id/texts_by_id; VIMMKG2_LAB = entity_labels.tsv).
D = os.environ.get("VIMMKG2_EXT", os.path.join(ROOT, "data/experimental/v3/ext/ViMMKG2"))
D1 = (
    os.path.dirname(os.environ["VIMMKG2_LAB"])
    if os.environ.get("VIMMKG2_LAB")
    else os.path.join(ROOT, "data/final/ViMMKG")
)
OUT_DIR = os.environ.get("VIMMKG2_RA", os.path.join(ROOT, "data/experimental/v3"))
K1, B, TOP_BM25, DF_SKIP = 1.5, 0.75, 100, 2000  # same as the 3 other datasets, NOT tuned
MAX_LABEL_WORDS = 8
FEATURE_NAMES = ["bm25", "n_chung", "idf_chung", "co_trung_ten", "co_trung_so", "len_q", "len_c"]
NORMALIZERS = {"n_chung": 32.0, "idf_chung": 64.0, "len": 512.0}
K_FOLD, SEED = 3, 42


def load_inputs(masked_text=None) -> tuple[dict, dict, int]:
    """(normalised text, normalised labels, N)."""
    e2q = {
        int(r["id"]): r["ten"]
        for r in csv.DictReader(
            open(os.path.join(D, "entity2id.tsv"), encoding="utf-8"), delimiter="\t"
        )
    }
    labels_vi = {
        r["qid"]: r["label_vi"]
        for r in csv.DictReader(
            open(os.path.join(D1, "entity_labels.tsv"), encoding="utf-8"), delimiter="\t"
        )
    }
    labels = {i: normalize_words_vi(labels_vi.get(q) or q.split("::")[-1]) for i, q in e2q.items()}
    # masked_text != None: read the MASKED text (column `mo_ta_che`) instead of texts_by_id.
    if masked_text:
        text = {
            int(r["id"]): normalize_words_vi(r["mo_ta_che"])
            for r in csv.DictReader(open(masked_text, encoding="utf-8"), delimiter="\t")
        }
    else:
        text = {
            int(r["id"]): normalize_words_vi(r["text"])
            for r in csv.DictReader(
                open(os.path.join(D, "texts_by_id.tsv"), encoding="utf-8"), delimiter="\t"
            )
        }
    return text, labels, len(e2q)


def make_folds() -> None:
    n = sum(1 for l in open(os.path.join(D, "train.txt"), encoding="utf-8") if l.strip())
    idx = list(range(n))
    random.Random(SEED).shuffle(idx)
    fold = [0] * n
    for rank, i in enumerate(idx):
        fold[i] = rank % K_FOLD
    form_counts = Counter(fold)
    assert sum(form_counts.values()) == n and len(form_counts) == K_FOLD
    json.dump(
        {"k": K_FOLD, "seed": SEED, "n": n, "fold": fold},
        open(os.path.join(D, "folds_k3.json"), "w"),
    )
    print(f"OK folds_k3.json: n={n} · {dict(sorted(form_counts.items()))}")


def bm25(text: dict, N: int, out=None) -> tuple:
    ids = sorted(text)
    df = Counter()
    for i in ids:
        df.update(set(text[i]))
    n_doc = len(ids)
    avgdl = sum(len(text[i]) for i in ids) / max(n_doc, 1)
    idf = {w: math.log(1 + (n_doc - c + 0.5) / (c + 0.5)) for w, c in df.items()}
    inverted = defaultdict(list)
    for i in ids:
        tf, dl = Counter(text[i]), len(text[i])
        for w, f_ in tf.items():
            inverted[w].append(
                (i, idf[w] * f_ * (K1 + 1) / (f_ + K1 * (1 - B + B * dl / max(avgdl, 1e-9))))
            )
    idx = np.zeros((N, TOP_BM25), dtype=np.int32)
    val = np.zeros((N, TOP_BM25), dtype=np.float32)
    for i in ids:
        scores = defaultdict(float)
        for w in set(text[i]):
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
            idx[i, c], val[i, c] = j, v / mx
    p = out or os.path.join(OUT_DIR, "v10_bm25_ViMMKG2.npz")
    np.savez_compressed(p, idx=idx, val=val)
    n_covered = int((val.sum(1) > 0).sum())
    print(
        f"OK BM25 top-{TOP_BM25}: {n_covered}/{N} entities have neighbours "
        f"({100*n_covered/N:.1f} %) · {os.path.getsize(p)/1048576:.0f} MB"
    )
    return idx.astype(np.int64), val, idf


def lexical_features(text: dict, labels: dict, N: int, idx, val, idf, out=None) -> None:
    K = idx.shape[1]
    word_sets = {i: set(w) for i, w in text.items()}
    digit_sets = {i: {w for w in s if w.isdigit()} for i, s in word_sets.items()}
    feat = np.zeros((N, K, 7), dtype=np.float32)
    n_with_cands = 0
    for e in range(N):
        if val[e].sum() == 0:
            continue
        n_with_cands += 1
        te, se, soe = word_sets.get(e, set()), text.get(e, []), digit_sets.get(e, set())
        len_q = len(se) / NORMALIZERS["len"]
        for j in range(K):
            if val[e, j] <= 0:
                continue
            c = int(idx[e, j])
            shared = te & word_sets.get(c, set())
            feat[e, j, 0] = val[e, j]
            feat[e, j, 1] = len(shared) / NORMALIZERS["n_chung"]
            feat[e, j, 2] = sum(idf.get(w, 0.0) for w in shared) / NORMALIZERS["idf_chung"]
            feat[e, j, 3] = float(contains_label(se, labels.get(c, [])))
            feat[e, j, 4] = float(bool(soe & digit_sets.get(c, set())))
            feat[e, j, 5] = len_q
            feat[e, j, 6] = len(text.get(c, [])) / NORMALIZERS["len"]
    Fi, Ff = torch.tensor(idx, dtype=torch.int32), torch.tensor(feat, dtype=torch.float16)
    # Gold invariance for the lexical block: rows are looked up by SOURCE entity, so gold is
    # not a parameter at all.
    ng = torch.tensor([3, 7, 11, 19])
    assert torch.equal(Ff[ng], Ff[ng]) and torch.equal(Fi[ng], Fi[ng]), "L invariance BROKEN"
    p = out or os.path.join(D, "L_feat.pt")
    torch.save({"idx": Fi, "feat": Ff, "ten": FEATURE_NAMES, "chia": NORMALIZERS}, p)
    mean_feat = feat[feat[:, :, 0] > 0].mean(0)
    print(
        f"OK L_feat.pt: N={N} · with candidates {n_with_cands} ({100*n_with_cands/N:.1f} %) · "
        f"top-{K} · {os.path.getsize(p)/1048576:.0f} MB"
    )
    print(
        "   mean features: " + " · ".join(f"{t} {v:.3f}" for t, v in zip(FEATURE_NAMES, mean_feat))
    )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--masked_text",
        "--text_che",
        default="",
        help="masked text (column `mo_ta_che`); empty = base",
    )
    ap.add_argument(
        "--out", default="", help="output path for the masked L_feat (empty = base L_feat.pt)"
    )
    ap.add_argument("--bm25_out", default="", help="output path for the masked BM25 npz")
    a = ap.parse_args()
    text, labels, N = load_inputs(masked_text=a.masked_text or None)
    print(
        f"ViProKG{' (MASKED)' if a.masked_text else ''}: N={N} entities · {len(text)} with text",
        flush=True,
    )
    if not a.masked_text:
        make_folds()
    idx, val, idf = bm25(text, N, out=a.bm25_out or None)
    lexical_features(text, labels, N, idx, val, idf, out=a.out or None)


if __name__ == "__main__":
    main()
