#!/usr/bin/env python3
"""Build answer-name-masked descriptions under rule v1 or v2. No GPU, runs locally.

Why: the only masked file available (`<dataset>-textual-mask.json`) was built under rule v1,
which masks EVERY verbatim label with none of the v2 guards (zipf threshold, uniqueness,
character-loss, context). The LLM probe is measured on the mention-free group under rule v2, so
using the v1 file would mean measuring with two different instruments.

Both paths are copied from code that already ran, the rule is not rewritten:
  `--rule v1` : calls `make_masked_text.mask_spans` + `label_leak_probe.normalize_words` +
                `make_masked_text.load_labels`; it must reproduce `<dataset>-textual-mask.json`
                bit-for-bit. This is the script's gate: if the v1 path cannot be reproduced, the
                v2 path (same skeleton) is not trustworthy.
  `--rule v2` : the form dictionary comes from `label_normalization.usable_variants(...,
                form_counts=...)`, the same function `rebuild_mention_edges_v2.py` uses to build the v2
                edges, plus the context guard (guard 4) checked AT THE MATCH POSITION, as that
                script does.

A missing zipf key makes `label_normalization._folded_zipf` raise; this script does not catch
that exception (falling back to the `wordfreq` library would reopen the door that was closed).
The two text normalisations (`label_leak_probe.normalize_words` for v1,
`label_normalization.split_description` for v2) were measured to agree on English samples; the
script prints that comparison on every run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import label_leak_probe as RT  # noqa: E402
import label_normalization as CHN  # noqa: E402,F401
import make_masked_text as TM  # noqa: E402
from label_normalization import (  # noqa: E402
    GUARD4_WINDOW,
    MAX_LABEL_WORDS,
    ZIPF_THRESHOLD,
    count_form_owners,
    load_proper_name_whitelist,
    load_zipf_table,
    split_description,
    usable_variants,
)
from rebuild_mention_edges_v2 import load_entity_labels  # noqa: E402

MASK_TOKEN = TM.MASK_TOKEN  # "[ENT]", taken from the v1 path, not retyped


def normalization_samples() -> list[tuple[str, bool]]:
    """Compare `normalize_words` (v1) with `split_description` (v2): they must agree, otherwise
    the v2 dictionary and the v1 text are two different instruments."""
    samples = [
        "Ho Chi Minh City (Viet Nam)",
        "U.S. Route 66",
        "ANTI-flag",
        "New_York",
        "Male",
        "co-founder of X",
        "Route 66, California",
        "The Beatles' album",
    ]
    return [(s, RT.normalize_words(s) == split_description(s)) for s in samples]


def mask_v2(words: list[str], lookup: dict, max_len: int) -> tuple[list[str], int]:
    """Mask left to right, trying the longest n-gram first, with the context guard at the match.

    Guard 4 is copied from `rebuild_mention_edges_v2.py`: the window `+-GUARD4_WINDOW` around the
    match must intersect the `ctx` set that the form requires. One accepting entity is enough.
    """
    out, i, n_masked = [], 0, 0
    L = len(words)
    while i < L:
        match = 0
        for n in range(min(max_len, L - i), 0, -1):
            g = lookup.get(tuple(words[i : i + n]))
            if not g:
                continue
            window = set(words[max(0, i - GUARD4_WINDOW) : i + n + GUARD4_WINDOW])
            if any((not ctx) or (ctx & window) for _e, ctx in g):
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


def mask_v2_positions(words: "list[str]", lookup: dict, max_len: int) -> "list[bool]":
    """Like mask_v2 but returns a per-token boolean array (True = masked), without merging [ENT].

    Used by the random-masking control: it needs to know exactly which tokens are names (to
    exclude them) and how many tokens each description masks (a span can be longer than 1 token).
    """
    mask_spans = [False] * len(words)
    i, L = 0, len(words)
    while i < L:
        match = 0
        for n in range(min(max_len, L - i), 0, -1):
            g = lookup.get(tuple(words[i : i + n]))
            if not g:
                continue
            window = set(words[max(0, i - GUARD4_WINDOW) : i + n + GUARD4_WINDOW])
            if any((not ctx) or (ctx & window) for _e, ctx in g):
                match = n
                break
        if match:
            for j in range(i, i + match):
                mask_spans[j] = True
            i += match
        else:
            i += 1
    return mask_spans


def _random_rng(dataset: str, seed: int, key: str) -> "random.Random":  # noqa: F821
    """RNG with a stable hash: blake2b of (dataset|seed|key), NOT str.__hash__."""
    import hashlib as _h
    import random as _r

    return _r.Random(
        int(_h.blake2b(f"{dataset}|{seed}|{key}".encode(), digest_size=8).hexdigest(), 16)
    )


def mask_random_positions(
    name_mask: "list[bool]", dataset: str, seed: int, key: str
) -> "list[bool]":
    """Mask exactly sum(name_mask) tokens at random, chosen among NON-name positions (seeded)."""
    k = sum(name_mask)
    cand = [j for j in range(len(name_mask)) if not name_mask[j]]
    if k == 0 or not cand:
        return [False] * len(name_mask)
    pick = set(_random_rng(dataset, seed, key).sample(cand, min(k, len(cand))))
    return [j in pick for j in range(len(name_mask))]


def _merge_runs(words: "list[str]", mask_spans: "list[bool]") -> "tuple[list[str], int]":
    """Consecutive True spans -> ONE [ENT]. Returns (tokens, number of masked tokens)."""
    out, k, n = [], 0, sum(mask_spans)
    while k < len(words):
        if mask_spans[k]:
            out.append(MASK_TOKEN)
            while k < len(words) and mask_spans[k]:
                k += 1
        else:
            out.append(words[k])
            k += 1
    return out, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", "--luat", required=True, choices=["v1", "v2"])
    ap.add_argument("--dataset", "--bo", default="MKG-W")
    ap.add_argument(
        "--data", default=RT.EXT_DIR, help="directory containing <dataset>/entity2id.tsv"
    )
    ap.add_argument(
        "--textual", default=RT.TOKENS_DIR, help="directory containing <dataset>-textual.json"
    )
    ap.add_argument(
        "--encoder",
        required=True,
        help="PATH to the tokenizer (never pass a model name: Kaggle has no internet)",
    )
    ap.add_argument(
        "--zipf_table",
        "--bang_zipf",
        default="",
        help="required with --rule v2 (frozen zipf table)",
    )
    ap.add_argument(
        "--whitelist", default="", help="required with --rule v2 (proper-name whitelist)"
    )
    ap.add_argument("--out", "--ra", required=True, help="TSV: id \\t key \\t masked description")
    ap.add_argument(
        "--out_json",
        "--ra_json",
        default="",
        help="also write JSON {key: text}, to compare bit-for-bit",
    )
    ap.add_argument(
        "--reference",
        "--doi_chieu",
        default="",
        help="JSON file to reproduce bit-for-bit (use with --rule v1)",
    )
    ap.add_argument("--cases", "--soat", type=int, default=20, help="print N before/after cases")
    ap.add_argument(
        "--random_control",
        "--ngau_nhien",
        action="store_true",
        help="control: mask the same number of tokens at RANDOM (only --rule v2)",
    )
    ap.add_argument(
        "--seed", type=int, default=42, help="masking seed (= training seed); with --random_control"
    )
    a = ap.parse_args()
    print(
        f"PATHS: data={Path(a.data).resolve()} - textual={Path(a.textual).resolve()} - "
        f"encoder={a.encoder}"
    )

    print("== v1 vs v2 normalisation ==")
    bad = [s for s, ok in normalization_samples() if not ok]
    for s, ok in normalization_samples():
        print(f"   {'OK' if ok else 'FAIL'} {s!r}")
    if bad:
        raise SystemExit(
            f"`normalize_words` (v1) != `split_description` (v2) on {bad}: two different "
            f"instruments, the v2 masked text cannot be compared with the v2 edge group. Stop."
        )

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.encoder)  # a PATH, not a name
    dataset = a.dataset
    d = Path(a.data) / dataset
    token_ids = json.loads(
        (Path(a.textual) / f"{dataset}-textual.json").read_text(encoding="utf-8")
    )
    e2i = {
        r.split("\t")[0]: int(r.split("\t")[1])
        for r in (d / "entity2id.tsv").read_text(encoding="utf-8").splitlines()[1:]
    }

    if a.rule == "v1":
        # Copy of `make_masked_text.main`: same `load_labels`, same `normalize_words`, same mask.
        labels, _e2i = TM.load_labels(dataset)
        lookup_v1 = {tuple(v) for v in labels.values() if v and len(v) <= TM.MAX_LABEL_WORDS}
        max_len = max((len(v) for v in lookup_v1), default=1)
        print(f"[v1] dictionary {len(lookup_v1)} forms - longest {max_len} words")
    else:
        if not (a.zipf_table and a.whitelist):
            raise SystemExit("--rule v2 requires --zipf_table and --whitelist")
        print(f"[zipf] {load_zipf_table(a.zipf_table)} keys - threshold {ZIPF_THRESHOLD}")
        print(
            f"[whitelist] {dataset}: {load_proper_name_whitelist(a.whitelist, dataset)} proper-name forms"
        )
        names = load_entity_labels(d)
        form_counts = count_form_owners(names)  # uniqueness over ALL labels, before any acceptance
        lookup: dict[tuple, list] = defaultdict(list)
        max_len, n_kept, n_dropped = 1, 0, 0
        for i, label in names.items():
            reasons: list = []
            for v, ctx in usable_variants(label, form_counts=form_counts, reasons=reasons):
                if len(v) <= MAX_LABEL_WORDS:
                    lookup[tuple(v)].append((i, ctx))
                    max_len = max(max_len, len(v))
            n_kept += sum(1 for _v, r in reasons if r.startswith("GIỮ"))
            n_dropped += sum(1 for _v, r in reasons if r.startswith("bỏ"))
        print(
            f"[v2] dictionary {len(lookup)} forms - longest {max_len} words - "
            f"variants KEPT {n_kept} - DROPPED {n_dropped} "
            f"({n_dropped / max(1, n_kept + n_dropped):.1%} removed by the guards)"
        )

    out_json, lines, n_changed, n_words, n_masked_total, cases = {}, [], 0, 0, 0, []
    for k, ids in token_ids.items():
        raw = tok.decode([x for x in ids if x < tok.vocab_size], skip_special_tokens=True)
        if a.rule == "v1":
            words = RT.normalize_words(raw)
            new_words, c = TM.mask_spans(words, lookup_v1, max_len)
        elif a.random_control:
            words = split_description(raw)
            name_mask = mask_v2_positions(words, lookup, max_len)
            random_mask = mask_random_positions(name_mask, dataset, a.seed, k)
            assert not any(
                random_mask[j] and name_mask[j] for j in range(len(words))
            ), f"random masking touched a NAME token ({dataset}/{k})"
            new_words, c = _merge_runs(words, random_mask)
        else:
            words = split_description(raw)
            new_words, c = mask_v2(words, lookup, max_len)
        out_json[k] = " ".join(new_words)
        i = int(k) if dataset == "DB15K" else e2i.get(k)
        if i is not None:
            lines.append((i, k, out_json[k]))
        n_words += len(words)
        n_masked_total += c
        if c:
            n_changed += 1
            if len(cases) < a.cases:
                cases.append((k, " ".join(words)[:150], out_json[k][:150], c))

    n_docs = len(token_ids)
    print(f"\n== {dataset} rule {a.rule} ==")
    print(f"  descriptions     {n_docs}")
    print(f"  CHANGED          {n_changed}  ({n_changed / max(1, n_docs):.2%})")
    print(
        f"  masked spans     {n_masked_total} - words {n_words} => "
        f"{n_masked_total / max(1, n_words):.2%} units"
    )
    print(f"\n== {len(cases)} before/after cases ==")
    for k, t, s, c in cases:
        print(f"  [{k}] masked {c}\n    BEFORE: {t}\n    AFTER : {s}")

    with open(a.out, "w", encoding="utf-8") as f:
        f.write("id\tkhoa\tmo_ta_che\n")
        for i, k, t in sorted(lines):
            f.write(f"{i}\t{k}\t{t}\n")
    print(f"\n-> {a.out} ({len(lines)} lines)")
    if a.out_json:
        # written exactly as `make_masked_text` writes it, so a bit-for-bit comparison is meaningful
        json.dump(out_json, open(a.out_json, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"-> {a.out_json}")

    if a.reference:
        ref_path = Path(a.reference)
        if not ref_path.exists():
            raise SystemExit(f"MISSING reference file: {ref_path.resolve()}")
        ref_bytes = ref_path.read_bytes()
        our_bytes = json.dumps(out_json, ensure_ascii=False).encode("utf-8")
        if our_bytes == ref_bytes:
            print(f"BIT-IDENTICAL with {ref_path} ({len(ref_bytes)} B)")
            return 0
        ref_dict = json.loads(ref_bytes)
        differing = [k for k in set(ref_dict) | set(out_json) if ref_dict.get(k) != out_json.get(k)]
        print(
            f"NOT bit-identical: {len(our_bytes)} B vs {len(ref_bytes)} B - "
            f"{len(differing)}/{len(ref_dict)} keys differ"
        )
        for k in differing[:3]:
            print(
                f"   [{k}]\n     reference: {str(ref_dict.get(k))[:120]}\n     ours     : "
                f"{str(out_json.get(k))[:120]}"
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
