#!/usr/bin/env python3
"""Build the whole answer-name-masking chain for the TEST pairs of one dataset.

Three products, **one masking rule**:

    1. `<dataset>-textual-che-test.json`  wordpiece tokens of the MASKED descriptions
       (shared by the masked BM25, the masked L_feat, and the masked-text baselines)
    2. `v10_bm25_<dataset>_che.npz`       BM25 top-K on the masked text, same K1/B/TOP
    3. `L_feat_che_test.pt`               built by exactly `lexical_channel_features.py`,
                                          only the input differs

Why BM25 must be rebuilt as well: `L_feat` holds `idx` (the top-K candidate set) and `feat`;
masking changes both. Zeroing only the name-match column would not mask `n_chung`/`idf_chung`.

Assertion: the name-match count on the TEST pairs is 0 after masking. Also prints the fraction
of top-K candidates that changed versus the original.

Command (example MKG-W):
    python3 build_masked_chain.py --dataset MKG-W --data_dir <d> \
        --textual repos/MyGO-main/tokens/MKG-W-textual.json \
        --encoder <bert-base-uncased> --out_dir <out> --rule v2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

SCR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCR))


def find_sibling_script(name: str) -> Path:
    """Locate a sibling script (`lexical_channel_features.py`, ...) via PYTHONPATH, not only the
    script's own directory.

    When this file is embedded into a runner's working directory, `Path(__file__).parent` points
    at the embedding directory, which does not hold the sibling scripts (they live in the
    dataset's script directory).
    """
    search = (
        [SCR]
        + [Path(x) for x in os.environ.get("PYTHONPATH", "").split(os.pathsep) if x]
        + [Path(x) for x in sys.path if x]
    )
    for d in search:
        p = d / name
        if p.is_file():
            return p
    raise SystemExit(
        f"cannot find {name}. Searched in:\n  "
        + "\n  ".join(str(x) for x in dict.fromkeys(search))
        + "\n  -> attach the dataset holding the scripts, or embed this file as well."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", "--bo", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--textual", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--out_dir", "--ra_dir", required=True)
    # REQUIRED: the masking dictionary must name its rule explicitly, no default.
    ap.add_argument("--rule", "--luat", required=True, choices=["v1", "v2"])
    ap.add_argument(
        "--tokens_only",
        "--chi_token",
        action="store_true",
        help="ONLY build the masked tokens + assert the name-match count is 0, then STOP. For "
        "consumers that do not need L_feat. Skips BM25, masked L_feat and the comparison with "
        "`L_feat.pt` (the last step requires `L_feat.pt` in --data_dir).",
    )
    ap.add_argument("--whitelist", default="", help="whitelist_1tu.tsv (only with --rule v2)")
    ap.add_argument(
        "--zipf_table", "--bang_zipf", default="", help="zipf_1tu.json (only with --rule v2)"
    )
    ap.add_argument(
        "--allow_names",
        "--cho_phep_ten",
        action="store_true",
        help="SKIP the assertion 'TEST pairs still naming the answer = 0'; for the RANDOM-mask "
        "control (names are kept on purpose, other tokens are masked). Only prints the count.",
    )
    ap.add_argument(
        "--premasked_text",
        "--text_che_san",
        default="",
        help="TSV of ALREADY masked text (columns `id`/`khoa`/`mo_ta_che`): the masking rule is "
        "NOT applied again, only tokenise + BM25 + L_feat. For a masked variant whose rule "
        "needs `zipf_1tu.json` + `whitelist_1tu.tsv` that no dataset has.",
    )
    a = ap.parse_args()
    _int_key = a.dataset in ("DB15K", "ViMMKG2")  # these datasets key entities by INT, like DB15K

    import numpy as np  # noqa: F401
    import torch
    from transformers import AutoTokenizer

    import build_text_context_masked as MK
    import label_normalization as LN
    from label_leak_probe import label_from_key, normalize_words
    from make_masked_text import MAX_LABEL_WORDS, load_labels

    os.makedirs(a.out_dir, exist_ok=True)
    d = Path(a.data_dir)

    # Declare the inputs up front rather than discovering a missing one after minutes of work.
    _required = ["test.txt", "entity2id.tsv"] + ([] if a.tokens_only else ["L_feat.pt"])
    _missing = [f_ for f_ in _required if not (d / f_).exists()]
    if _missing:
        raise SystemExit(
            f"`--data_dir` {d} is MISSING {_missing}.\n"
            f"   - `L_feat.pt` is only needed for the last step (comparing top-K candidates). If\n"
            f"     the consumer does NOT use L_feat, pass **`--tokens_only`** to stop after the\n"
            f"     masked tokens."
        )
    print(
        f"[inputs] {d} has {_required}"
        + ("  (--tokens_only: L_feat.pt NOT needed)" if a.tokens_only else ""),
        flush=True,
    )
    tok = AutoTokenizer.from_pretrained(a.encoder)
    labels, e2i = load_labels(a.dataset, a.data_dir)
    # The masking dictionary must use the SAME rule that defines the mention groups. A v1
    # dictionary has no zipf guard, no diacritic fold, no whitelist and no uniqueness guard, so it
    # would mask `film`/`music`/`world` everywhere and miss `dusseldorf` (the folded form).
    # `MK.tach_tu` is identical to `LN.split_description`, so v2 forms can be fed to
    # `remove_partner_names`.
    if a.rule == "v2":
        if a.zipf_table:
            print(f"[zipf] frozen table {LN.load_zipf_table(a.zipf_table)} keys", flush=True)
        if a.whitelist:
            print(
                f"[whitelist] {a.dataset}: "
                f"{LN.load_proper_name_whitelist(a.whitelist, a.dataset)} proper-name forms",
                flush=True,
            )
        links = {}
        if a.dataset == "MKG-W":
            for r in (Path(a.data_dir) / "ent_links").read_text(encoding="utf-8").splitlines():
                if "\t" in r:
                    db, qid = r.split("\t")[:2]
                    links[qid] = db
        raw_labels = {i: label_from_key(a.dataset, k, links) for k, i in e2i.items()}
        form_counts = LN.count_form_owners(raw_labels)
        label_index: dict[tuple, list[int]] = defaultdict(list)
        n_forms = 0
        for i, nh in raw_labels.items():
            for v, _c in LN.usable_variants(nh, form_counts=form_counts):
                if v and len(v) <= MAX_LABEL_WORDS:
                    label_index[tuple(v)].append(i)
                    n_forms += 1
        print(
            f"[mask] rule v2: {len(label_index)} distinct forms from {n_forms} forms "
            f"(v1 gives {sum(1 for v in labels.values() if v and len(v) <= MAX_LABEL_WORDS)})",
            flush=True,
        )
    else:
        label_index = defaultdict(list)
        for i, v in labels.items():
            if v and len(v) <= MAX_LABEL_WORDS:
                label_index[tuple(v)].append(i)
        print(f"[mask] rule v1: {len(label_index)} forms", flush=True)
    max_len = max((len(k) for k in label_index), default=1)

    # partner = the ANSWER of the TEST triple itself
    test = [
        tuple(int(x) for x in l.split("\t"))
        for l in (d / "test.txt").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    answers: dict[int, set[int]] = defaultdict(set)
    for h, _r, t in test:
        answers[h].add(t)
        answers[t].add(h)

    # --- 1. masked token json ---
    tk = json.load(open(a.textual, encoding="utf-8"))
    masked, n_edited = {}, 0
    if a.premasked_text:
        # Text ALREADY masked: do NOT apply the rule again, only tokenise. Match by `id` (DB15K)
        # or `khoa` (MKG-W/Y), reading by column NAME; an entity without a row keeps its original
        # text and is counted. The TEST-pair assertion below stays, so it checks the mask itself.
        import csv as _csv

        with open(a.premasked_text, encoding="utf-8") as _f:
            _r = list(_csv.DictReader(_f, delimiter="\t"))
        if not _r or "mo_ta_che" not in _r[0]:
            raise SystemExit(f"invalid {a.premasked_text}: missing column `mo_ta_che`")
        _by_id = {str(int(x["id"])): x["mo_ta_che"] for x in _r if x.get("id", "").strip()}
        _by_key = {x["khoa"]: x["mo_ta_che"] for x in _r if x.get("khoa")}
        print(
            f"[mask] PRE-MASKED TEXT: {a.premasked_text} · {len(_r)} rows "
            f"(rule `--rule {a.rule}` NOT applied, the text was produced elsewhere)",
            flush=True,
        )
        n_missing = 0
        for k, ids in tk.items():
            i = int(k) if _int_key else e2i.get(k)
            _txt = _by_key.get(k) or (_by_id.get(str(i)) if i is not None else None)
            if _txt is None:
                masked[k] = ids
                n_missing += 1
                continue
            masked[k] = tok(_txt[:2000], add_special_tokens=True)["input_ids"]
            n_edited += 1
        print(
            f"[mask] matched {n_edited}/{len(tk)} entities · WITHOUT masked text: {n_missing} "
            f"(original kept)",
            flush=True,
        )
        if n_edited == 0:
            raise SystemExit("0 entities matched -> wrong key mapping, STOP (do not mask nothing)")
    else:
        for k, ids in tk.items():
            i = int(k) if _int_key else e2i.get(k)
            raw = tok.decode([x for x in ids if x < tok.vocab_size], skip_special_tokens=True)
            if i is None:
                masked[k] = ids
                continue
            masked_text, changed = MK.remove_partner_names(
                raw[:2000], answers.get(i, set()), label_index, max_len
            )
            if changed:
                n_edited += 1
            masked[k] = tok(masked_text, add_special_tokens=True)["input_ids"]
    _suffix = "-v2san" if a.premasked_text else ""
    p_tok = os.path.join(a.out_dir, f"{a.dataset}-textual-che-test{_suffix}.json")
    json.dump(masked, open(p_tok, "w"), ensure_ascii=False)
    print(
        f"[{a.dataset}] masked tokens: {n_edited}/{len(tk)} entities edited -> {p_tok}", flush=True
    )

    # --- assert: name-match count on the TEST pairs = 0 ---
    n_left = 0
    for h, _r, t in test:
        k = next(
            (kk for kk, ids in tk.items() if (int(kk) if _int_key else e2i.get(kk)) == h), None
        )
        if k is None:
            continue
        words = normalize_words(
            tok.decode([x for x in masked[k] if x < tok.vocab_size], skip_special_tokens=True)
        )
        if any(
            label_index.get(tuple(words[p : p + n])) and t in label_index[tuple(words[p : p + n])]
            for p in range(len(words))
            for n in range(1, min(max_len, len(words) - p) + 1)
        ):
            n_left += 1
    print(f"[{a.dataset}] TEST pairs still naming the answer: {n_left}")
    if n_left and not a.allow_names:
        raise SystemExit(f"ASSERT FAILED: {n_left} test pairs still name the answer after masking")
    if n_left and a.allow_names:
        print(
            f"[{a.dataset}] RANDOM mask (names kept on purpose): {n_left} pairs still name the "
            "answer, as designed, not stopping (--allow_names)",
            flush=True,
        )

    if a.tokens_only:
        print(f"OK {a.dataset} done (--tokens_only): {p_tok}", flush=True)
        print("   SKIPPED by flag: BM25 / masked L_feat / comparison with `L_feat.pt`.", flush=True)
        return 0

    # --- 2. BM25 on the masked text, same K1/B/TOP ---
    import build_lexical_data as V10

    p_bm = os.path.join(a.out_dir, f"v10_bm25_{a.dataset}_che{_suffix}.npz")
    V10.build_bm25_topk(a.dataset, tok, e2i, tok_json=p_tok, out=p_bm)
    print(f"[{a.dataset}] masked BM25 -> {p_bm} (K1={V10.K1} B={V10.B} TOP={V10.TOP_BM25})")

    # --- 3. masked L_feat, by exactly lexical_channel_features.py ---
    p_lf = os.path.join(a.out_dir, f"L_feat_che_test{_suffix}.pt")
    cmd = [
        sys.executable,
        str(find_sibling_script("lexical_channel_features.py")),
        "--dataset",
        a.dataset,
        "--tok_json",
        p_tok,
        "--bm25",
        p_bm,
        "--out",
        p_lf,
        "--enc",
        a.encoder,
        "--data_dir",
        str(d),
    ]
    print("  command: " + " ".join(cmd), flush=True)
    r = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        # pass the FULL PYTHONPATH: `str(SCR)` alone would lose the path to the dataset's scripts
        env=dict(
            os.environ,
            PYTHONPATH=os.pathsep.join(
                [str(SCR)] + [x for x in os.environ.get("PYTHONPATH", "").split(os.pathsep) if x]
            ),
        ),
    )
    # Print BOTH streams: `stdout or stderr` would hide stderr whenever stdout has content, which
    # once masked an `OSError: Read-only file system` for three runs.
    print(r.stdout[-3000:])
    if r.stderr.strip():
        print("----- stderr -----\n" + r.stderr[-3000:])
    if r.returncode:
        raise SystemExit(f"lexical_channel_features rc={r.returncode}")

    # --- print the fraction of top-K candidates that CHANGED ---
    old = torch.load(d / "L_feat.pt", map_location="cpu", weights_only=False)
    new = torch.load(p_lf, map_location="cpu", weights_only=False)
    a_idx, b_idx = old["idx"].long(), new["idx"].long()
    n = min(a_idx.shape[0], b_idx.shape[0])
    n_changed = sum(1 for i in range(n) if set(a_idx[i].tolist()) != set(b_idx[i].tolist()))
    n_old = (old["feat"][:, :, 3].float() > 0).sum().item()
    n_new = (new["feat"][:, :, 3].float() > 0).sum().item()
    print(
        f"[{a.dataset}] top-K candidates CHANGED for {n_changed}/{n} entities ({n_changed/max(1,n):.4f})"
    )
    print(f"[{a.dataset}] name-match cells: {n_old} -> {n_new}")
    print(f"OK {a.dataset} done: {p_tok} · {p_bm} · {p_lf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
