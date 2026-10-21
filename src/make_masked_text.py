"""Produce answer-name-masked descriptions, used to build `T_ctx_mask.pt`.

Every verbatim occurrence of the label of ANY entity (including the entity itself) in a
description is replaced by `[ENT]`. Greedy longest-first matching, so a long label is not cut up
by a shorter one.

Uses exactly the normalisation of `label_leak_probe.py`, i.e. the same channel as the leak
measurement. After masking, the leak rate is measured again on the masked text; it must drop to
about 0, otherwise the masking is broken.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_leak_probe import (  # noqa: E402
    DATASETS,
    EXT_DIR,
    OUT_DIR,
    TOKENS_DIR,
    contains_label,
    label_from_key,
    normalize_words,
)

MAX_LABEL_WORDS = 8  # longest label considered (in words); caps the scan
MASK_TOKEN = "[ENT]"


def load_labels(
    dataset: str, data_dir: str | None = None
) -> tuple[dict[int, list[str]], dict[str, int]]:
    # data_dir: read entity2id / ent_links from a real directory (e.g. on Kaggle) instead of the
    # fixed EXT_DIR/<dataset> path. The default None keeps the original behaviour for local callers.
    d = data_dir if data_dir else os.path.join(EXT_DIR, dataset)
    e2i = {
        r.split("\t")[0]: int(r.split("\t")[1])
        for r in open(os.path.join(d, "entity2id.tsv"), encoding="utf-8").read().splitlines()[1:]
    }
    links = {}
    if dataset == "MKG-W":
        for r in open(os.path.join(d, "ent_links"), encoding="utf-8").read().splitlines():
            db, qid = r.split("\t")
            links[qid] = db
    return {i: normalize_words(label_from_key(dataset, k, links)) for k, i in e2i.items()}, e2i


def mask_spans(
    words: list[str], vocab: set[tuple[str, ...]], max_len: int
) -> tuple[list[str], int]:
    """Scan left to right, trying the longest n-gram first at each position; replace by MASK_TOKEN."""
    out, i, n_masked = [], 0, 0
    L = len(words)
    while i < L:
        match = 0
        for n in range(min(max_len, L - i), 0, -1):
            if tuple(words[i : i + n]) in vocab:
                match = n
                break
        if match:
            out.append(MASK_TOKEN)
            i += match
            n_masked += 1
        else:
            out.append(words[i])
            i += 1
    return out, n_masked


def main() -> None:
    ap = argparse.ArgumentParser(description="Produce answer-name-masked token files.")
    ap.add_argument(
        "--text_encoder",
        default="bert-base-uncased",
        help="tokenizer: a LOCAL PATH on machines without internet, or a model name (needs internet or a cache)",
    )
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.text_encoder)
    report = []
    for dataset in DATASETS:
        labels, e2i = load_labels(dataset)
        vocab = {tuple(v) for v in labels.values() if v and len(v) <= MAX_LABEL_WORDS}
        max_len = max((len(v) for v in vocab), default=1)
        token_ids = json.load(
            open(os.path.join(TOKENS_DIR, f"{dataset}-textual.json"), encoding="utf-8")
        )

        out, n_words, n_masked_total, n_docs = {}, 0, 0, 0
        orig_words: dict[int, list[str]] = {}
        for k, ids in token_ids.items():
            words = normalize_words(
                tok.decode([x for x in ids if x < tok.vocab_size], skip_special_tokens=True)
            )
            i = int(k) if dataset == "DB15K" else e2i.get(k)
            if i is not None:
                orig_words[i] = words
            new_words, c = mask_spans(words, vocab, max_len)
            out[k] = " ".join(new_words)
            n_words += len(words)
            n_masked_total += c
            n_docs += 1
        p = os.path.join(OUT_DIR, f"{dataset}-textual-mask.json")
        json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False)
        print(
            f"[mask] {dataset}: {n_docs} descriptions - {len(vocab)} labels (longest {max_len} words) - "
            f"masked {n_masked_total} spans / {n_words} words "
            f"({100 * n_masked_total / max(n_words, 1):.2f} %) -> {p}",
            flush=True,
        )

        # check: the leak channel must drop to ~0 on the masked text
        masked_words = {
            i: normalize_words(out[k])
            for k, i in ((k, int(k) if dataset == "DB15K" else e2i.get(k)) for k in out)
            if i is not None
        }
        split_file = os.path.join(
            EXT_DIR, dataset, "valid_sach.txt" if dataset == "DB15K" else "valid.txt"
        )
        triples = [
            tuple(int(x) for x in r.split("\t"))
            for r in open(split_file, encoding="utf-8").read().splitlines()
            if r.strip()
        ]
        g = sum(
            1
            for h, _, t in triples
            if h in orig_words and contains_label(orig_words[h], labels.get(t, []))
        )
        m = sum(
            1
            for h, _, t in triples
            if h in masked_words and contains_label(masked_words[h], labels.get(t, []))
        )
        print(
            f"   valid check: t in text(h) original {100 * g / len(triples):.2f} % "
            f"-> masked {100 * m / len(triples):.2f} %",
            flush=True,
        )
        report.append(
            [
                dataset,
                n_docs,
                len(vocab),
                n_masked_total,
                n_words,
                f"{100 * n_masked_total / max(n_words, 1):.2f}",
                f"{100 * g / len(triples):.2f}",
                f"{100 * m / len(triples):.2f}",
            ]
        )

    with open(os.path.join(OUT_DIR, "v9a_mask_bao.tsv"), "w", encoding="utf-8") as f:
        f.write("bo\tn_mo_ta\tn_nhan\tn_cum_che\tn_tu\tpc_tu_bi_che\tro_goc_pc\tro_che_pc\n")
        for r in report:
            f.write("\t".join(str(x) for x in r) + "\n")
    print("wrote v9a_mask_bao.tsv")


if __name__ == "__main__":
    sys.exit(main())
