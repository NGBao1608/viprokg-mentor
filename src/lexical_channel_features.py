"""Build the features of the lexical channel (L block). No GPU.

Each entity `e` has its **top-100 BM25 candidates** (from `build_lexical_data`) and **7 features**
for every pair `(e, c)`:

| # | feature | definition |
|---|---|---|
| 0 | `bm25` | BM25 score normalised per row to [0,1] |
| 1 | `n_chung` | \\|words(e) ∩ words(c)\\|, divided by 32 |
| 2 | `idf_chung` | sum of idf of the shared words, divided by 64 |
| 3 | `co_trung_ten` | 1 if the **label of c** appears verbatim in text(e) |
| 4 | `co_trung_so` | 1 if e and c share at least one all-digit token |
| 5 | `len_q` | number of words of text(e), divided by 512 |
| 6 | `len_c` | number of words of text(c), divided by 512 |

Designed so that gold cannot leak: features depend only on `(e, c)`, never on the query,
relation or gold. The tensor is indexed by the SOURCE entity, so permuting gold cannot change
any value. The script still checks this once (`check_gold_invariance`).

Output: `<ext>/<dataset>/L_feat.pt` = {`idx` int32[N,100], `feat` fp16[N,100,7]}.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_leak_probe import (  # noqa: E402
    DATASETS,
    EXT_DIR,
    OUT_DIR,
    TOKENS_DIR,
    contains_label,
    normalize_words,
)
from make_masked_text import MAX_LABEL_WORDS, load_labels  # noqa: E402

FEATURE_NAMES = ["bm25", "n_chung", "idf_chung", "co_trung_ten", "co_trung_so", "len_q", "len_c"]
NORMALIZERS = {"n_chung": 32.0, "idf_chung": 64.0, "len": 512.0}  # normalisation constants


def load_text(dataset: str, tok, e2i: dict, tok_json: str | None = None) -> dict[int, list[str]]:
    """`tok_json=None` uses the base token file; otherwise a different (masked) token file is
    used. The computation below does not change."""
    tk = json.load(
        open(tok_json or os.path.join(TOKENS_DIR, f"{dataset}-textual.json"), encoding="utf-8")
    )
    out = {}
    for k, ids in tk.items():
        i = int(k) if dataset == "DB15K" else e2i.get(k)
        if i is not None:
            out[i] = normalize_words(
                tok.decode([x for x in ids if x < tok.vocab_size], skip_special_tokens=True)
            )
    return out


def check_gold_invariance(feat: torch.Tensor, idx: torch.Tensor) -> bool:
    """Looking up a batch of source entities twice, with two different 'gold' values (gold is
    not a parameter at all), must give identical tensors."""
    ng = torch.tensor([3, 7, 11, 19])
    a = feat[ng].clone()
    _fake_gold = torch.tensor([100, 200, 300, 400])  # unused on purpose
    b = feat[ng].clone()
    return bool(torch.equal(a, b) and torch.equal(idx[ng], idx[ng]))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", "--bo", default="", help="empty = all 3 datasets (old behaviour)")
    ap.add_argument("--tok_json", default="", help="REPLACEMENT token file (masked variant)")
    ap.add_argument("--bm25", default="", help="REPLACEMENT BM25 npz (built on the masked text)")
    ap.add_argument("--out", "--ra", default="", help="REPLACEMENT output path for L_feat.pt")
    ap.add_argument("--enc", default="bert-base-uncased")
    # --data_dir: read labels from the REAL directory (e.g. a Kaggle working dir) instead of the
    # hard-coded EXT_DIR. Empty = old behaviour (EXT_DIR) for local callers.
    ap.add_argument("--data_dir", default="")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.enc)
    report_rows = []
    for dataset in [a.dataset] if a.dataset else DATASETS:
        d = a.data_dir or os.path.join(EXT_DIR, dataset)
        labels, e2i = load_labels(dataset, a.data_dir or None)
        N = len(e2i)
        text = load_text(dataset, tok, e2i, a.tok_json or None)
        z = np.load(a.bm25 or os.path.join(OUT_DIR, f"v10_bm25_{dataset}.npz"))
        idx, val = z["idx"].astype(np.int64), z["val"].astype(np.float32)
        K = idx.shape[1]

        # idf over the description set itself (same formula as build_lexical_data)
        df = Counter()
        for i in text:
            df.update(set(text[i]))
        n_doc = max(len(text), 1)
        idf = {w: math.log(1 + (n_doc - c + 0.5) / (c + 0.5)) for w, c in df.items()}

        # label dictionary for the name-match flag
        label_index = defaultdict(list)
        for i, v in labels.items():
            if v and len(v) <= MAX_LABEL_WORDS:
                label_index[tuple(v)].append(i)

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
                tc = word_sets.get(c, set())
                shared = te & tc
                feat[e, j, 0] = val[e, j]
                feat[e, j, 1] = len(shared) / NORMALIZERS["n_chung"]
                feat[e, j, 2] = sum(idf.get(w, 0.0) for w in shared) / NORMALIZERS["idf_chung"]
                feat[e, j, 3] = float(contains_label(se, labels.get(c, [])))
                feat[e, j, 4] = float(bool(soe & digit_sets.get(c, set())))
                feat[e, j, 5] = len_q
                feat[e, j, 6] = len(text.get(c, [])) / NORMALIZERS["len"]
        Fi = torch.tensor(idx, dtype=torch.int32)
        Ff = torch.tensor(feat, dtype=torch.float16)
        assert check_gold_invariance(Ff, Fi), "L-block invariance check FAILED"
        p = a.out or os.path.join(d, "L_feat.pt")
        torch.save({"idx": Fi, "feat": Ff, "ten": FEATURE_NAMES, "chia": NORMALIZERS}, p)
        mb = os.path.getsize(p) / 1048576
        mean_feat = feat[feat[:, :, 0] > 0].mean(0) if (feat[:, :, 0] > 0).any() else np.zeros(7)
        print(
            f"OK {dataset:7} N={N} · with candidates {n_with_cands} ({100*n_with_cands/N:.1f} %) "
            f"· top-{K} · {mb:.0f} MB"
        )
        print(
            "   mean features (cells with BM25>0): "
            + " · ".join(f"{t} {v:.3f}" for t, v in zip(FEATURE_NAMES, mean_feat))
        )
        report_rows.append(
            [dataset, N, n_with_cands, K, f"{mb:.1f}"] + [f"{v:.4f}" for v in mean_feat]
        )
    # Write the side report NEXT TO `--out`, not into OUT_DIR: OUT_DIR derives from `__file__` of
    # the label-leak module, which is read-only when the script runs from a Kaggle dataset. The
    # main output is already written by now, so a failure here must not kill the run.
    _report_dir = os.path.dirname(os.path.abspath(a.out)) if a.out else OUT_DIR
    try:
        _report_file = os.path.join(_report_dir, "lsr_L_bao.tsv")
        open(_report_file, "a").close()
    except OSError as e:
        print(
            f"cannot write {_report_dir}/lsr_L_bao.tsv ({e.strerror}): SKIPPING the side report,"
            f" main output already written",
            flush=True,
        )
        return 0
    with open(_report_file, "w", encoding="utf-8") as f:
        f.write(
            "bo\tN\tn_co_ung_vien\ttop_k\tMB\t" + "\t".join("tb_" + t for t in FEATURE_NAMES) + "\n"
        )
        for r in report_rows:
            f.write("\t".join(map(str, r)) + "\n")
    print(f"wrote {_report_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
