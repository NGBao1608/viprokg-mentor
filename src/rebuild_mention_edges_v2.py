#!/usr/bin/env python3
"""Rebuild the mention-edge files under rule E v2 and report what moved. No GPU.

Writes `v10_canh_nhac_ten_<dataset>_v2.tsv` NEXT TO the v1 file; v1 is never overwritten, so
every published number stays reproducible and the two can be diffed.

Prints:
  * number of edges v1 vs v2, per dataset
  * number of keys that move mention-free -> mention, per split (train / valid / test)
  * random NEW edges with the matched surface form, for manual review
  * the mention-free key counts per split (= 2 x triples, self-loops excluded)

Self-loops are excluded from the edge list exactly as v1 did (`b != a`), and additionally
excluded from the group counts: all of them sit in the mention-free group while being
answerable by identity.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_normalization as _LN  # noqa: E402
from label_normalization import (  # noqa: E402
    CHAR_LOSS_GUARD,
    GUARD4_WINDOW,
    MAX_LABEL_WORDS,
    ZIPF_THRESHOLD,
    count_form_owners,
    fold_diacritics,
    forms_v1,
    load_proper_name_whitelist,
    load_zipf_table,
    split_description,
    usable_variants,
)

ALL_DATASETS = ["MKG-W", "MKG-Y", "DB15K"]


def _v1_edge_path(v1_dir: str, dataset: str) -> Path:
    """Path of the v1 edge file; fail early and clearly when it is missing.

    Called at the start of `main()` for every dataset, so a wrong `--v1_dir` is reported before
    the tokenizer is loaded and the variant dictionary built.
    """
    p = Path(v1_dir) / f"v10_canh_nhac_ten_{dataset}.tsv"
    if p.is_file():
        return p
    found = sorted(x.name for x in Path(v1_dir).glob("*.tsv")) if Path(v1_dir).is_dir() else []
    raise SystemExit(
        f"missing v1 edge file: {p}\n"
        f"   `--v1_dir` = {v1_dir!r} "
        f"(directory {'exists' if Path(v1_dir).is_dir() else 'does NOT exist'})\n"
        f"   .tsv files there: {found[:8] or 'none'}\n"
        f"   `--v1_dir` must be the DIRECTORY holding `v10_canh_nhac_ten_<dataset>.tsv`."
    )


def load_entity_labels(d: Path) -> dict[int, str]:
    """Readable label per entity id. MKG-W ids are QIDs, so go through `ent_links`."""
    import re

    q2t = {}
    if (d / "ent_links").exists():
        for l in (d / "ent_links").read_text(encoding="utf-8").splitlines():
            if "\t" in l:
                db, q = l.split("\t")[:2]
                q2t[q.strip()] = db.strip().rstrip(">").rsplit("/", 1)[-1].lstrip("<")
    out = {}
    for l in (d / "entity2id.tsv").read_text(encoding="utf-8").splitlines()[1:]:
        c = l.split("\t")
        raw_id = c[0].strip()
        short_id = raw_id.rstrip(">").rsplit("/", 1)[-1].lstrip("<")
        out[int(c[1])] = q2t.get(raw_id, short_id) if re.fullmatch(r"Q\d+", short_id) else short_id
    return out


def load_triples_text(p: Path):
    return [
        tuple(int(x) for x in l.split())
        for l in p.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--textual", required=True)
    ap.add_argument("--out_dir", "--ra_dir", required=True)
    ap.add_argument("--v1_dir", required=True, help="dir holding v10_canh_nhac_ten_<dataset>.tsv")
    ap.add_argument("--dataset", "--bo", nargs="*", default=ALL_DATASETS)
    ap.add_argument("--n_review", "--n_soat", type=int, default=10)  # new edges sampled per dataset
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--vi", action="store_true", help="ViProKG: Vietnamese variant, NO diacritic folding"
    )
    # Encoder is a PATH: GPU runners may have no internet, and a model NAME would try to download.
    ap.add_argument(
        "--encoder",
        default="bert-base-uncased",
        help="PATH to a downloaded tokenizer (e.g. /kaggle/input/...); a model name only works "
        "on a machine with internet/cache",
    )
    ap.add_argument(
        "--whitelist",
        default="",
        help="whitelist_1tu.tsv; empty = no whitelist (the zipf guard drops proper names)",
    )
    ap.add_argument(
        "--zipf_table",
        "--bang_zipf",
        default="",
        help="json built by the zipf-table generator; empty = use the installed wordfreq",
    )
    # default None, NOT False: passing False explicitly would override the locked setting
    # `label_normalization.STRIP_DISAMBIGUATION`. None = "use the locked rule".
    ap.add_argument(
        "--diacritics_only",
        "--chi_dau",
        action="store_const",
        const=True,
        default=None,
        help="force diacritic folding ONLY. Empty = use the locked rule.",
    )
    ap.add_argument(
        "--strip",
        "--cat",
        action="store_const",
        const=False,
        default=None,
        dest="diacritics_only",
        help="force stripping `(...)`/after-comma; only to reproduce older measurements.",
    )
    a = ap.parse_args()
    if a.zipf_table:
        _n = load_zipf_table(a.zipf_table)
        print(
            f"[zipf] frozen table {_n} keys "
            f"(built {_LN._ZIPF_TABLE_META.get('ngay','?')}, "
            f"wordfreq {_LN._ZIPF_TABLE_META.get('wordfreq','?')})",
            flush=True,
        )
    # Check every input per dataset up front: fail in 0 seconds, not after building the dictionary.
    for _b in a.dataset:
        _v1_edge_path(a.v1_dir, _b)
        for _f in (f"{_b}-textual.json",):
            if not (Path(a.textual) / _f).is_file():
                raise SystemExit(f"missing {Path(a.textual) / _f}")
        for _f in ("entity2id.tsv", "train.txt", "valid.txt", "test.txt"):
            if not (Path(a.data) / _b / _f).is_file():
                raise SystemExit(f"missing {Path(a.data) / _b / _f}")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.encoder)
    from label_normalization import STRIP_DISAMBIGUATION

    _strip = (not a.diacritics_only) if a.diacritics_only is not None else STRIP_DISAMBIGUATION
    level_desc = (
        "STRIP `(...)`/after-comma (REPRODUCES OLDER MEASUREMENT)"
        if _strip
        else "level (a): DIACRITIC FOLD ONLY (locked)"
    ) + f": v1 always kept (zipf < {ZIPF_THRESHOLD} if one word) · v2 variants must be UNIQUE"
    print("=" * 104)
    print(f"REBUILD MENTION EDGES under rule E v2 · {level_desc} · no GPU")
    print("=" * 104)

    canon = {}
    for dataset in a.dataset:
        # Load the whitelist INSIDE the loop: it is scoped per dataset (`nice` is a proper name
        # in MKG-Y but an adjective elsewhere); loading once and using `a.dataset[0]` for every
        # dataset would be silently wrong.
        if a.whitelist:
            _nw = load_proper_name_whitelist(a.whitelist, dataset)
            print(
                f"[whitelist] {dataset}: {_nw} proper-name forms "
                f"(over {_LN._PROPER_NAME_WHITELIST_META.get('n_dong')} rows)",
                flush=True,
            )
        d = Path(a.data) / dataset
        labels = load_entity_labels(d)
        token_ids = json.loads(
            (Path(a.textual) / f"{dataset}-textual.json").read_text(encoding="utf-8")
        )
        i2e = {}
        for l in (d / "entity2id.tsv").read_text(encoding="utf-8").splitlines()[1:]:
            c = l.split("\t")
            i2e[int(c[1])] = c[0].strip()

        # Uniqueness is dataset-wide and spans v1 UNION v2 of every entity, so it has to be
        # counted BEFORE any variant is accepted.
        form_counts = count_form_owners(labels, vi=a.vi, diacritics_only=a.diacritics_only)
        # form -> [(entity, context words guard 4 requires)]. The requirement travels WITH the
        # form, because the same form can come from one label as a plain name (no requirement)
        # and from another as a stripped variant (requirement non-empty).
        form_owners: dict[tuple, list[tuple[int, set]]] = defaultdict(list)
        max_len = 1
        n_v2_kept = n_v2_dropped = 0
        for i, label in labels.items():
            why: list = []
            for v, ctx in usable_variants(
                label,
                vi=a.vi,
                form_counts=form_counts,
                diacritics_only=a.diacritics_only,
                reasons=why,
            ):
                if len(v) <= MAX_LABEL_WORDS:
                    form_owners[tuple(v)].append((i, ctx))
                    max_len = max(max_len, len(v))
            n_v2_kept += sum(1 for _v, r in why if r.startswith("GIỮ"))
            n_v2_dropped += sum(1 for _v, r in why if r.startswith("bỏ"))
        print(
            f"  v2 variants: KEPT {n_v2_kept} · DROPPED {n_v2_dropped} "
            f"({n_v2_dropped / max(1, n_v2_kept + n_v2_dropped):.1%} removed by the ambiguity/zipf guards)"
        )

        # How many entities share each surface form? A form that resolves to many entities
        # carries no identifying power (`central district` maps to every Iranian county seat),
        # so an edge it creates is a false positive.
        n_owners = {k: len({i for i, _c in v}) for k, v in form_owners.items()}
        edges_v2: set[tuple[int, int]] = set()
        new_examples: list[tuple] = []
        for k, ids in token_ids.items():
            e = int(k) if dataset == "DB15K" else None
            if e is None:
                # non-DB15K: json is keyed by the raw entity string
                e = next((i for i, s in i2e.items() if s == k), None)
                if e is None:
                    continue
            # No truncation: v1 decodes the WHOLE description; truncating would lose edges and
            # break the v2-superset-of-v1 property.
            raw = tok.decode([x for x in ids if x < tok.vocab_size], skip_special_tokens=True)
            words = split_description(raw, vi=a.vi)
            for i0 in range(len(words)):
                for n in range(1, min(max_len, len(words) - i0) + 1):
                    g = form_owners.get(tuple(words[i0 : i0 + n]))
                    if g:
                        # guard 4 is positional, so it is checked HERE, at the match site
                        around = set(words[max(0, i0 - GUARD4_WINDOW) : i0 + n + GUARD4_WINDOW])
                        for b, ctx in g:
                            if ctx and not (ctx & around):
                                continue
                            if b != e:
                                edges_v2.add((e, b))
                                new_examples.append(
                                    (
                                        e,
                                        b,
                                        " ".join(words[i0 : i0 + n]),
                                        " ".join(words[max(0, i0 - 10) : i0 + n + 10]),
                                        n_owners.get(tuple(words[i0 : i0 + n]), 1),
                                    )
                                )

        p1 = _v1_edge_path(a.v1_dir, dataset)
        edges_v1 = {
            tuple(int(x) for x in l.split("\t"))
            for l in p1.read_text(encoding="utf-8").splitlines()[1:]
            if l
        }
        p2 = out / f"v10_canh_nhac_ten_{dataset}_v2.tsv"
        with p2.open("w", encoding="utf-8") as f:
            f.write("a\tb\n")
            for x, y in sorted(edges_v2):
                f.write(f"{x}\t{y}\n")

        print(f"\n### {dataset}")
        print(
            f"  edges v1 {len(edges_v1):7} · v2 {len(edges_v2):7} · "
            f"added {len(edges_v2 - edges_v1):6} · lost {len(edges_v1 - edges_v2):6}"
        )
        # The superset check is a REPORT, but only for losses the zipf / character guards
        # explain. A loss from any other source is a hard stop.
        lost = edges_v1 - edges_v2
        if lost:
            lost_zipf, lost_char, unexplained = {}, {}, []
            for x, y in lost:
                # Record each guard separately: the character guard fixes a normalisation bug
                # (`8½` -> `8`), the zipf guard is a post-hoc threshold. If the zipf threshold
                # is loosened later, the character-guard part must NOT change.
                label = labels.get(y, "")
                v1 = forms_v1(label, vi=a.vi)
                if len(v1) == 1:
                    from label_normalization import (
                        _PROPER_NAME_WHITELIST,
                        _folded_zipf,
                        _loses_chars,
                    )

                    if CHAR_LOSS_GUARD and _loses_chars(label):
                        lost_char[label] = lost_char.get(label, 0) + 1
                        continue
                    if (
                        _PROPER_NAME_WHITELIST is not None
                        and fold_diacritics(v1[0]).lower() in _PROPER_NAME_WHITELIST
                    ):
                        unexplained.append((x, y, label))  # whitelisted yet lost: must be reported
                        continue
                    z = _folded_zipf(v1[0], a.vi)
                    if z is not None and z >= ZIPF_THRESHOLD:
                        lost_zipf.setdefault(f"{label} ({z:.2f})", 0)
                        lost_zipf[f"{label} ({z:.2f})"] += 1
                        continue
                unexplained.append((x, y, label))
            print(
                f"  REPORT: v2 lost {len(lost)} edges vs v1 · "
                f"{sum(lost_zipf.values())} due to the ZIPF guard · {sum(lost_char.values())} due "
                f"to the CHARACTER guard · {len(unexplained)} unexplained"
            )
            for k, v in sorted(lost_char.items(), key=lambda kv: -kv[1])[:8]:
                print(f"     character-guard drop {k:26} -> lost {v} edges")
            for k, v in sorted(lost_zipf.items(), key=lambda kv: -kv[1])[:14]:
                print(f"     zipf-guard drop      {k:26} -> lost {v} edges")
            if unexplained:
                print(
                    f"  BLOCKED: {len(unexplained)} edges lost NOT due to the zipf/character guards"
                )
                print(f"     Only those two guards may lose edges. Examples: {unexplained[:5]}")
                raise SystemExit("edges lost for an unknown reason: stop, do not build further")
            print(f"  all {len(lost)} lost edges trace back to the zipf or character guard")
        else:
            print(f"  v2 contains v1: 0 edges lost")  # noqa: F541
        print(f"  wrote {p2}")

        pairs_v1 = {frozenset(e) for e in edges_v1}
        pairs_v2 = {frozenset(e) for e in edges_v2}
        canon[dataset] = {}
        for split in ("train", "valid", "test"):
            triples = load_triples_text(d / f"{split}.txt")
            n_free1 = n_free2 = n_self = 0
            for h, _r, t in triples:
                if h == t:
                    n_self += 1
                    continue  # self-loops excluded from group counts
                n_free1 += frozenset((h, t)) not in pairs_v1
                n_free2 += frozenset((h, t)) not in pairs_v2
            print(
                f"  {split:6} triples {len(triples):6} · self-loops {n_self:4} (excluded) · "
                f"mention-free v1 {n_free1:6} -> v2 {n_free2:6} · moved to mention "
                f"{n_free1 - n_free2:5} ({(n_free1 - n_free2) / max(1, n_free1):.2%}) · "
                f"v2 KEYS (x2 directions) {2 * n_free2:6}"
            )
            canon[dataset][split] = 2 * n_free2

        # Random NEW edges for manual review. The difference set must be computed ONCE: building
        # it inside a list comprehension is O(|new_examples| * |edges_v2|) and did not finish in
        # 37 minutes on the largest dataset.
        only_new = edges_v2 - edges_v1
        new_only_examples = [x for x in new_examples if (x[0], x[1]) in only_new]
        sample = random.Random(a.seed).sample(
            new_only_examples, min(a.n_review, len(new_only_examples))
        )
        p_review = out / f"soat_canh_moi_{dataset}.tsv"
        with p_review.open("w", encoding="utf-8") as f:
            f.write("a_id\ta_ten\tb_id\tb_ten\tkhop\tn_entity_cung_dang\tngu_canh\n")
            for e, b, matched, ctx, n_own in sample:
                f.write(
                    f"{e}\t{labels.get(e,'?')}\t{b}\t{labels.get(b,'?')}\t{matched}\t{n_own}\t{ctx}\n"
                )
        print(f"  wrote {p_review} ({len(sample)} random new edges for manual review)")
        for e, b, matched, ctx, n_own in sample[:4]:
            print(
                f"     {labels.get(e,'?')[:26]:26} -> {labels.get(b,'?')[:22]:22} "
                f"matched {matched!r} (x{n_own} entities) | ...{ctx[:52]}..."
            )
        # ambiguity proxy over ALL new edges, not just the printed sample
        all_new = new_only_examples  # same set, not rebuilt
        n_amb = sum(1 for x in all_new if x[4] > 1)
        print(
            f"  AMBIGUOUS matched forms (>=2 entities share the form): {n_amb}/{len(all_new)} "
            f"= {n_amb / max(1, len(all_new)):.1%} of new edges"
        )

    print("\n" + "=" * 104)
    # NOT the reference counts of the mention-free split: those are counted on the dump KEYS and
    # KEEP self-loops, while the numbers below are counted on the triple FILE and DROP them.
    print("MENTION-FREE KEY COUNTS ON THE TRIPLE FILE (= 2 x triples, self-loops EXCLUDED)")
    print("NOT the reference counts of the mention-free screen; use the reference-count script")
    for split in ("valid", "test"):
        print(
            f"  {split:6} " + " · ".join(f"{b} {canon[b][split]}" for b in a.dataset if b in canon)
        )
    print("\n  Do not paste these into the mention-free screen; use the reference-count script.")
    print("  Do not replace the existing reference counts by hand: they back published reports.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
