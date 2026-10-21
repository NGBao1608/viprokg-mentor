"""Label-leak measurement: does label(t) occur verbatim in text(h)?

Every rate comes with a shuffled control (h paired with a random other t') to give the chance
floor; the real gain is real rate minus shuffled rate. Without the control the numbers are
meaningless, since the part predictable by construction must be removed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import urllib.parse

from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXT_DIR = os.path.join(ROOT, "data", "experimental", "v3", "ext")
TOKENS_DIR = os.path.join(ROOT, "repos", "MyGO-main", "tokens")
OUT_DIR = os.path.join(ROOT, "data", "experimental", "v3")
DATASETS = ["MKG-W", "MKG-Y", "DB15K"]
SHUFFLE_SEED = 42
N_SHUFFLES = 3  # number of shuffles, averaged


def normalize_words(s: str) -> list[str]:
    """Lower-cased word list without punctuation; the same function is used for labels and text."""
    s = urllib.parse.unquote(s)
    s = re.sub(r"[^0-9a-zA-ZÀ-ɏ]+", " ", s).lower()
    return s.split()


# `normalize_words` only keeps characters up to U+024F, so it drops most Vietnamese diacritic
# letters (Latin Extended Additional, U+1EA0-U+1EF9): 'Hà Nội' -> ['hà', 'n', 'i']. For
# Vietnamese data that would shred labels and text into one-character pieces and fill the mention
# edges with false positives while the output still looks valid. A separate function is provided
# for Vietnamese. Do NOT change `normalize_words`: it fixes the mention-edge files already
# reported.
def normalize_words_vi(s: str) -> list[str]:
    """Like `normalize_words` but keeps every Unicode word character.

    Uses `\\w` with the UNICODE flag instead of listing code ranges.
    `normalize_words_vi('Hà Nội') == ['hà', 'nội']`.
    """
    s = urllib.parse.unquote(s)
    s = re.sub(r"[^\w]+", " ", s, flags=re.UNICODE).replace("_", " ").lower()
    return s.split()


def label_from_key(dataset: str, key: str, links: dict[str, str]) -> str:
    """Derive the text label of an entity from its entity2id key (each dataset has its own form)."""
    if dataset == "MKG-Y":
        return key
    if dataset == "DB15K":
        return key.strip("<>").rsplit("/", 1)[-1]
    return links.get(key, "").rsplit("/", 1)[-1]  # MKG-W: QID -> DBpedia


def contains_label(words: list[str], labels: list[str]) -> bool:
    """The label occurs verbatim (contiguous, on word boundaries) in the sentence."""
    if not labels or not words:
        return False
    n = len(labels)
    return any(words[i : i + n] == labels for i in range(len(words) - n + 1))


def load_dataset_bundle(dataset: str, tok) -> dict:
    d = os.path.join(EXT_DIR, dataset)
    e2i = {
        r.split("\t")[0]: int(r.split("\t")[1])
        for r in open(os.path.join(d, "entity2id.tsv"), encoding="utf-8").read().splitlines()[1:]
    }
    links = {}
    if dataset == "MKG-W":
        for r in open(os.path.join(d, "ent_links"), encoding="utf-8").read().splitlines():
            db, qid = r.split("\t")
            links[qid] = db
    labels = {i: normalize_words(label_from_key(dataset, k, links)) for k, i in e2i.items()}

    token_ids = json.load(
        open(os.path.join(TOKENS_DIR, f"{dataset}-textual.json"), encoding="utf-8")
    )
    text: dict[int, list[str]] = {}
    for k, ids in token_ids.items():
        i = int(k) if dataset == "DB15K" else e2i.get(k)
        if i is None:
            continue
        text[i] = normalize_words(
            tok.decode([x for x in ids if x < tok.vocab_size], skip_special_tokens=True)
        )
    splits = {}
    for split_name in ["train", "valid", "test"]:
        f = os.path.join(
            d,
            (
                "valid_sach.txt"
                if (dataset == "DB15K" and split_name == "valid")
                else f"{split_name}.txt"
            ),
        )
        splits[split_name] = [
            tuple(int(x) for x in r.split("\t"))
            for r in open(f, encoding="utf-8").read().splitlines()
            if r.strip()
        ]
    return {"labels": labels, "text": text, "splits": splits, "n_ent": len(e2i)}


def count_matches(
    bundle: dict, triples: list[tuple[int, int, int]], rng: random.Random | None
) -> dict:
    """Count matches on one split. rng not None -> shuffle tail/head (random control)."""
    labels, text = bundle["labels"], bundle["text"]
    ids = sorted(text)
    n_t, n_h, n = 0, 0, 0
    for h, _, t in triples:
        hh, tt = h, t
        if rng is not None:  # shuffle: swap partners, keep the label distribution
            tt = rng.choice(ids)
            hh = rng.choice(ids)
        observed, tt_ = text.get(h), text.get(tt)  # noqa: F841
        if observed is not None:
            n_t += contains_label(observed, labels.get(tt, []))
        tx = text.get(t)
        if tx is not None:
            n_h += contains_label(tx, labels.get(hh, []))
        n += 1
    return {"n": n, "t_trong_h": n_t, "h_trong_t": n_h}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Measure how often the label of one entity occurs verbatim in the text of the other."
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
        bundle = load_dataset_bundle(dataset, tok)
        print(
            f"-- {dataset}: {bundle['n_ent']} entities - {len(bundle['text'])} with text "
            f"({100 * len(bundle['text']) / bundle['n_ent']:.1f} %)",
            flush=True,
        )
        for split_name, triples in bundle["splits"].items():
            observed = count_matches(bundle, triples, None)
            shuffled = [
                count_matches(bundle, triples, random.Random(SHUFFLE_SEED + k))
                for k in range(N_SHUFFLES)
            ]
            x_t = sum(x["t_trong_h"] for x in shuffled) / N_SHUFFLES
            x_h = sum(x["h_trong_t"] for x in shuffled) / N_SHUFFLES
            n = observed["n"]
            rows.append(
                [
                    dataset,
                    split_name,
                    n,
                    f"{100 * observed['t_trong_h'] / n:.2f}",
                    f"{100 * x_t / n:.2f}",
                    f"{100 * (observed['t_trong_h'] - x_t) / n:+.2f}",
                    f"{100 * observed['h_trong_t'] / n:.2f}",
                    f"{100 * x_h / n:.2f}",
                    f"{100 * (observed['h_trong_t'] - x_h) / n:+.2f}",
                ]
            )
            print(
                f"   {split_name:5} n={n:6}  t in text(h) {rows[-1][3]:>6} % "
                f"(shuffled {rows[-1][4]:>5} % -> {rows[-1][5]:>6})   "
                f"h in text(t) {rows[-1][6]:>6} % (shuffled {rows[-1][7]:>5} % -> {rows[-1][8]:>6})",
                flush=True,
            )
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, "v9a_ro_nhan.tsv")
    with open(p, "w", encoding="utf-8") as f:
        f.write("bo\tlat\tn\tt_trong_h\txao_t\tdo_loi_t\th_trong_t\txao_h\tdo_loi_h\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"wrote {p} ({len(rows)} rows)")


if __name__ == "__main__":
    sys.exit(main())
