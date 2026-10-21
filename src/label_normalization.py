#!/usr/bin/env python3
"""Single label-normalisation rule used to build the mention-overlap edges (rule E v2).

A mention edge (a, b) exists when the label of entity b appears verbatim in the description of
entity a. The text has already lost some "furniture" that the label keeps, so the rule folds
the label towards the text:

  | gap                     | label becomes (v1)                  | text says        |
  |-------------------------|-------------------------------------|------------------|
  | disambiguator `(...)`   | `robert taylor actor` (3 words!)    | `robert taylor`  |
  | text after a comma      | `oxford ohio`                       | `oxford`         |
  | diacritics              | `peña`                              | `pena`           |

`bert-base-uncased` strips accents, so the diacritic gap is structural. The disambiguator and
comma cuts are implemented but locked OFF (`STRIP_DISAMBIGUATION = False`): they delete the
token that identifies the entity and produced mostly wrong edges in a manual audit.

Scope: this changes a grouping / evaluation rule, not a model input. The English and Vietnamese
variants are two implementations of one rule: unit = word for EN, syllable for VI; the VI
variant keeps diacritics.
"""
from __future__ import annotations

import re
import unicodedata
import urllib.parse

# Longest label considered, in units (bounds the n-gram scan cost).
MAX_LABEL_WORDS = 8

# v1 character class, kept for the "raw" variant so v1 behaviour is reproducible from here.
_EN_NON_WORD = re.compile(r"[^0-9a-zA-ZÀ-ɏ]+")


def fold_diacritics(s: str) -> str:
    """Fold combining marks: `Peña` -> `Pena`. Matches what `bert-base-uncased` does to text."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def strip_disambiguator(s: str) -> str:
    """Drop KB furniture that is not part of the name.

    `Robert Taylor (actor)` -> `Robert Taylor` · `Oz (TV series)` -> `Oz`
    `Oxford, Ohio` -> `Oxford`
    """
    s = re.sub(r"\s*\([^)]*\)", " ", s)
    s = s.split(",")[0]
    return s.strip()


# One-word forms are dropped when their zipf frequency is >= this threshold. The value was
# chosen after inspecting MKG-Y (post-hoc): with 5.0 the label `Malé` (folded `male`, zipf 4.89)
# slipped through and produced the single largest hub of false edges. 4.8 drops it; MKG-W is
# unchanged over [4.5, 5.0]. Frequency cannot separate "common word" from "famous entity", so
# no threshold is clean.
ZIPF_THRESHOLD = 4.8
GUARD4_WINDOW = 20  # context window of guard 4 (guard 4 is OFF by default)
# Locked: the rule does NOT strip `(...)` or text after a comma. Those two cuts produced most
# of the new edges and nearly all sampled examples were wrong (`2001 (Dr. Dre album)` -> `2001`
# matched a year). `--diacritics_only` / `--strip` on the rebuilder exist only to reproduce
# the measurements that led here.
STRIP_DISAMBIGUATION = False
# Guard 4 (context words of a dropped disambiguator must appear near the match) is dropped:
# the words a disambiguator removes are type words (`series`, `song`, `film`), which are
# nearly free to satisfy for a same-type false positive.
GUARD4_ENABLED = False

# Function words of a disambiguator carry no evidence, so they cannot satisfy guard 4.
FUNCTION_WORDS = {
    "the",
    "a",
    "an",
    "of",
    "and",
    "or",
    "in",
    "on",
    "at",
    "to",
    "for",
    "by",
    "with",
    "from",
    "as",
    "is",
    "was",
    "were",
    "be",
    "it",
    "its",
    "s",
}


def _split_units(s: str, vi: bool) -> list[str]:
    if vi:
        return re.sub(r"[^\w]+", " ", s, flags=re.UNICODE).replace("_", " ").lower().split()
    return _EN_NON_WORD.sub(" ", s).lower().split()


def forms_v1(label: str, *, vi: bool = False) -> list[str]:
    """The v1 unit-list. Always kept: no length guard.

    A guard `>= 2 units OR >= 4 chars` would delete real 3-letter acronyms (`MTV`, `NBC`,
    `EMI`, ...), which are names and occur in descriptions. Length never made an edge wrong;
    ambiguity did.
    """
    return _split_units(urllib.parse.unquote(label).replace("_", " "), vi)


def forms_v2(
    label: str, *, vi: bool = False, diacritics_only: "bool | None" = None
) -> "list[tuple[list[str], set]]":
    """The v2 ADDITIONS only (v1 form excluded). Order is not meaningful.

    `diacritics_only=True` keeps only the diacritic fold and drops the disambiguator/comma
    variants (stripping `(film)` from `New York (film)` deletes the token that told it apart
    from the state and the city). Diacritic folding has no such failure mode.
    """
    if diacritics_only is None:
        diacritics_only = not STRIP_DISAMBIGUATION
    label = urllib.parse.unquote(label).replace("_", " ")
    base = _split_units(label, vi)
    out: "list[tuple[list[str], set]]" = []

    def add(x: str, ctx: set) -> None:
        t = _split_units(x, vi)
        if t and t != base and all(t != v for v, _c in out):
            out.append((t, ctx))

    # Guard 4: a variant produced by DROPPING something may only be used when a content word of
    # the dropped part sits near the match. Diacritic folding drops nothing, so its requirement
    # is empty.
    stripped = strip_disambiguator(label)
    dropped = set(_split_units(label, vi)) - set(_split_units(stripped, vi))
    dropped = {w for w in dropped if w not in FUNCTION_WORDS and len(w) > 1}
    if not diacritics_only:
        add(stripped, dropped)
    if not vi:  # diacritics: EN only; Vietnamese text keeps its marks
        add(fold_diacritics(label), set())
        if not diacritics_only:
            add(fold_diacritics(stripped), dropped)
    return out


def all_forms_v2(label: str, *, vi: bool = False) -> list[list[str]]:
    """All forms, v1 first. Kept for inspection; matchers use `usable_variants`."""
    return [forms_v1(label, vi=vi)] + forms_v2(label, vi=vi)


def count_form_owners(
    labels: "dict[int, str]", *, vi: bool = False, diacritics_only: "bool | None" = None
) -> "dict[tuple, int]":
    """How many DISTINCT entities produce each form, over v1 UNION v2 of the whole dataset.

    A v2 variant is usable only when its form resolves to exactly one entity. The count spans
    v1 and v2 of every entity, otherwise a v2 form that collides with someone else's v1 name
    would pass as unique.
    """
    from collections import defaultdict

    owners: dict[tuple, set] = defaultdict(set)
    for i, nh in labels.items():
        for v in [forms_v1(nh, vi=vi)] + [
            x for x, _c in forms_v2(nh, vi=vi, diacritics_only=diacritics_only)
        ]:
            if v:
                owners[tuple(v)].add(i)
    return {k: len(v) for k, v in owners.items()}


# Frozen zipf table. The rule depends on the installed `wordfreq` version, and a GPU runner
# without internet may not have it at all (the guard would then silently drop every one-word
# label). The three datasets have only ~3.3k distinct one-word forms, so the whole table is
# frozen into a small file. Once a table is loaded, a form missing from it is an ERROR; falling
# back to wordfreq would reopen the door the table closes.
_ZIPF_TABLE: "dict[str, float] | None" = None
_ZIPF_TABLE_META: dict = {}


def load_zipf_table(duong: str) -> int:
    """Load the frozen zipf table. Returns the number of entries. Call before `usable_variants`."""
    global _ZIPF_TABLE, _ZIPF_TABLE_META
    import json

    with open(duong, encoding="utf-8") as f:
        o = json.load(f)
    _ZIPF_TABLE = {k: float(v) for k, v in o["zipf"].items()}
    _ZIPF_TABLE_META = o.get("meta", {})
    return len(_ZIPF_TABLE)


# Proper-name whitelist, scoped PER DATASET (`nice` is a club name in MKG-Y but a common
# adjective elsewhere). The zipf guard cannot tell a common noun from a famous proper name, so
# `England` (5.12) and `London` (5.27) would be dropped like `film` (5.20). The text is
# lowercase in every dataset, so capitalisation cannot be used instead.
#
# Character guard: a ONE-word form is dropped when the original label (after folding) holds a
# character outside [A-Za-z0-9] and whitespace, i.e. normalisation lost a meaningful character
# and the remaining form no longer identifies the entity (`8½` -> `8`, `ANTI-`, `Going!`).
# It runs BEFORE the whitelist and is not bypassed by it.
CHAR_LOSS_GUARD = True


def _loses_chars(label: str) -> bool:
    """Does the original label hold a non-alphanumeric character (besides spaces) that normalisation drops?"""
    return any(not c.isalnum() and not c.isspace() for c in fold_diacritics(label))


_PROPER_NAME_WHITELIST: "set[str] | None" = None
_PROPER_NAME_WHITELIST_META: dict = {}


def load_proper_name_whitelist(duong: str, bo: str) -> int:
    """Load the proper-name whitelist of dataset `bo` only. Returns the number of forms."""
    global _PROPER_NAME_WHITELIST, _PROPER_NAME_WHITELIST_META
    import csv as _csv

    with open(duong, encoding="utf-8") as f:
        lines = [l for l in f if not l.startswith("#")]
    r = list(_csv.DictReader(lines, delimiter="\t"))
    _PROPER_NAME_WHITELIST = {
        fold_diacritics(x["dang_khop"]).lower()
        for x in r
        if x["bo"] == bo and x["nhan_tay"] == "TEN-RIENG"
    }
    _PROPER_NAME_WHITELIST_META = {"duong": duong, "bo": bo, "n_dong": len(r)}
    return len(_PROPER_NAME_WHITELIST)


def _folded_zipf(word: str, vi: bool) -> "float | None":
    """Zipf of the form that will ACTUALLY be matched, i.e. after folding.

    The datasets' descriptions go through `bert-base-uncased`, which strips combining marks, so
    a label like `Malé` can only match the text as `male`. Measuring the raw label
    (`zipf("malé") = 1.66`) let it through; the folded form has `zipf("male") = 4.89`.

    If `wordfreq` is unavailable and no frozen table is loaded, returns None and the caller
    drops the form; no document-frequency proxy is substituted.
    """
    ng = "vi" if vi else "en"
    if _ZIPF_TABLE is not None:
        # The frozen table serves both languages; check its language so an English table is
        # never used for Vietnamese (vi keeps diacritics, key = lower; en folds them).
        _lang = _ZIPF_TABLE_META.get("ngon_ngu", "en")
        if (_lang == "vi") != vi:
            raise SystemExit(
                f"zipf table language {_lang!r} but _folded_zipf called with vi={vi}: wrong "
                f"table (vi keeps diacritics / lower-case keys, en folds them). Load the right one."
            )
        k = word.lower() if vi else fold_diacritics(word).lower()
        if k in _ZIPF_TABLE:
            return _ZIPF_TABLE[k]
        raise SystemExit(
            f"one-word form {word!r} (key {k!r}) is NOT in the frozen zipf table "
            f"({len(_ZIPF_TABLE)} entries, language {_lang}, built {_ZIPF_TABLE_META.get('ngay','?')}); "
            f"rebuild the table. Not falling back to wordfreq."
        )
    try:
        from wordfreq import zipf_frequency
    except ImportError:
        return None
    # Vietnamese is NOT folded: diacritics are part of the spelling (`má` != `ma`).
    if vi:
        return zipf_frequency(word, ng)
    folded = fold_diacritics(word)
    # Take the MAX: the matched form is `folded`, but if the accented spelling is more frequent
    # it can also match in datasets that keep accents, so guard on the worst case.
    return max(zipf_frequency(folded, ng), zipf_frequency(word, ng))


def _zipf_both(word: str, vi: bool) -> "tuple[float | None, float | None]":
    """(zipf of the raw label, zipf of the folded form). For reporting only."""
    try:
        from wordfreq import zipf_frequency
    except ImportError:
        return None, None
    ng = "vi" if vi else "en"
    return zipf_frequency(word, ng), zipf_frequency(fold_diacritics(word), ng)


def usable_variants(
    label: str,
    *,
    vi: bool = False,
    form_counts: "dict[tuple, int] | None" = None,
    zipf_threshold: float = ZIPF_THRESHOLD,
    diacritics_only: "bool | None" = None,
    reasons: "list | None" = None,
) -> "list[tuple[list[str], set]]":
    """Forms a matcher may use, under three guards.

      1. the v1 form is ALWAYS kept (no length guard)
      2. a v2 form is kept only if it is UNIQUE in the dataset (needs `form_counts`)
      3. a ONE-WORD form is additionally dropped when `zipf >= zipf_threshold`

    Guard 2 is what a `>= 2 words` rule fails to do: `central district` (26 entities) and
    `new york` are two words and still ambiguous. Guard 3 catches the residue that is unique in
    this dataset but still a common word.

    `form_counts=None` means uniqueness cannot be checked, so v2 forms are NOT used at all.

    `reasons` collects `(form, verdict)` for auditing. Verdicts start with "GIỮ" (keep) or
    "bỏ" (drop); callers count them with `startswith`, so the prefixes are a protocol.
    """
    v1 = forms_v1(label, vi=vi)
    out: "list[tuple[list[str], set]]" = []
    # Guard 3 applies to every one-word form, the v1 form included: it separates real acronyms
    # (`MTV` 3.87, `EMI` 3.11) from an entity literally named `11` (zipf 5.32).
    if v1:
        if len(v1) == 1 and CHAR_LOSS_GUARD and _loses_chars(label):
            # character guard: runs before both the whitelist and zipf
            if reasons is not None:
                reasons.append(
                    (
                        v1,
                        f"bỏ: 1 từ, nhãn gốc {label!r} có ký tự phi-chữ-số bị "
                        f"chuẩn hoá làm mất (rào ký tự)",
                    )
                )
        elif (
            len(v1) == 1
            and _PROPER_NAME_WHITELIST is not None
            and fold_diacritics(v1[0]).lower() in _PROPER_NAME_WHITELIST
        ):
            # the whitelist bypasses the zipf guard (not the uniqueness guard of the v2 branch)
            out.append((v1, set()))
            if reasons is not None:
                reasons.append((v1, "GIỮ v1: TÊN-RIÊNG trong whitelist, bỏ qua rào zipf"))
        elif len(v1) == 1:
            z1 = _folded_zipf(v1[0], vi)
            if z1 is None:
                if reasons is not None:
                    reasons.append((v1, "bỏ: v1 một từ và THIẾU wordfreq ⇒ không kiểm được zipf"))
            elif z1 >= zipf_threshold:
                if reasons is not None:
                    reasons.append((v1, f"bỏ: v1 một từ, zipf {z1:.2f} >= {zipf_threshold}"))
            else:
                out.append((v1, set()))
                if reasons is not None:
                    reasons.append((v1, f"GIỮ v1: một từ, zipf {z1:.2f} < {zipf_threshold}"))
        else:
            out.append((v1, set()))
    for v, ctx in forms_v2(label, vi=vi, diacritics_only=diacritics_only):
        if not GUARD4_ENABLED:
            ctx = set()  # guard 4 off
        if form_counts is None:
            if reasons is not None:
                reasons.append((v, "bỏ: không có bảng đếm ⇒ không kiểm được tính duy nhất"))
            continue
        n = form_counts.get(tuple(v), 0)
        if n != 1:
            if reasons is not None:
                reasons.append(
                    (
                        v,
                        (
                            "bỏ: dạng này KHÔNG có trong bộ"
                            if n == 0
                            else f"bỏ: MƠ HỒ, {n} entity cùng dạng"
                        ),
                    )
                )
            continue
        if len(v) == 1 and CHAR_LOSS_GUARD and _loses_chars(label):
            if reasons is not None:
                reasons.append(
                    (
                        v,
                        f"bỏ: 1 từ, nhãn gốc {label!r} có ký tự phi-chữ-số bị "
                        f"chuẩn hoá làm mất (rào ký tự)",
                    )
                )
            continue
        if (
            len(v) == 1
            and _PROPER_NAME_WHITELIST is not None
            and fold_diacritics(v[0]).lower() in _PROPER_NAME_WHITELIST
        ):
            if reasons is not None:
                reasons.append((v, "GIỮ: duy nhất, TÊN-RIÊNG trong whitelist, bỏ qua rào zipf"))
            out.append((v, ctx))
            continue
        if len(v) == 1:
            z = _folded_zipf(v[0], vi)
            if z is None:
                if reasons is not None:
                    reasons.append((v, "bỏ: 1 từ và THIẾU wordfreq ⇒ không kiểm được zipf"))
                continue
            if z >= zipf_threshold:
                if reasons is not None:
                    reasons.append((v, f"bỏ: 1 từ, zipf {z:.2f} >= {zipf_threshold}"))
                continue
            if reasons is not None:
                reasons.append((v, f"GIỮ: duy nhất, 1 từ, zipf {z:.2f} < {zipf_threshold}"))
            out.append((v, ctx))
            continue
        if reasons is not None:
            reasons.append(
                (
                    v,
                    f"GIỮ: duy nhất ({len(v)} từ)"
                    + (f", cần ngữ cảnh {sorted(ctx)}" if ctx else ""),
                )
            )
        out.append((v, ctx))
    return out


def first_match(
    variants: "list[tuple[list[str], set]]", text_words: list[str], window: int = GUARD4_WINDOW
) -> "tuple[int, int, list[str]] | None":
    """First position where a variant matches AND satisfies guard 4. None if nothing matches.

    Unit-run, never substring: `India` inside `Indian` is not a mention.
    """
    for v, ctx in variants:
        n = len(v)
        if not n:
            continue
        for i in range(len(text_words) - n + 1):
            if text_words[i : i + n] != v:
                continue
            if ctx:
                around = text_words[max(0, i - window) : i + n + window]
                if not (ctx & set(around)):
                    continue  # guard 4 refuses this position
            return i, n, v
    return None


def has_match(variants, text_words, window: int = GUARD4_WINDOW) -> bool:
    """Boolean wrapper over `first_match`."""
    return first_match(variants, text_words, window) is not None


def split_description(s: str, *, vi: bool = False) -> list[str]:
    """Unit-split for DESCRIPTIONS. Must stay identical to the label splitter, or the two sides
    are measured with different instruments."""
    s = urllib.parse.unquote(s)
    if vi:
        return re.sub(r"[^\w]+", " ", s, flags=re.UNICODE).replace("_", " ").lower().split()
    return _EN_NON_WORD.sub(" ", s).lower().split()
