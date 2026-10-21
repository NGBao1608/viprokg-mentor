"""Check that the public tables in ``tables/`` carry exactly the numbers of the frozen tables.

Both files are reduced to the sequence of numeric tokens (digits with their decimal part and an attached
sign: ``+``, ``-``/``$-$``/``−``, ``±``/``$\\pm$``); everything else (words, punctuation, LaTeX) is dropped.
``{,}`` and ``,`` inside a number are read as the decimal point.

Check A (all numbers): LaTeX comment lines and internal report labels (TIN_/BC<n>/section signs/footnote ids/
hashes/the old label "A3"/refs/labels) are removed from BOTH files first, then the two token lists must be equal.
Check B (table body, stricter): the contents of ``tabular`` must give equal token lists with only the label "A3"
and the footnote marker "[F8]" removed, so no label rule can hide a change in a table cell.
Exit code 0 means 0 mismatches for every table.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN, PUBLIC = ROOT / "tables_frozen", ROOT / "tables"
NAMES = [
    "bang_chinh",
    "bang_vimmkg2",
    "bang_phuluc_db15k",
    "bang_trung_tam",
    "bang_canh_v2",
    "bang_e_che",
    "bang_lat_prior",
    "bang_anh",
    "bang_e_muc",
    "bang_prior_k3",
    "bang_tran_thong_tin",
    "bang_ablation",
    "bang_delta_L",
]
LABELS = [
    r"TIN\\?_\d+(?: [A-Z]\d*)?",
    r"BC\d+[a-z]?(?:-\d+)?(?: Table \d+)?(?:\s?/\s?(?:BC\d+\s)?§\d+[a-z]?(?:\.\d+)?|\s§\d+[a-z]?(?:\.\d+)?)*",
    r"KHOA\\?_CHU(?: §\d+)?",
    r"§\s?\d+[a-zA-Z]?(?:\.\d+)?",
    r"❓[A-Z]+\d*",
    r"\b[0-9a-f]{16}\b",
    r"\[F8\]",
    r"ext G1",
    r"note \\#\d+[a-z]?",
    r"\\ref\{[^}]*\}",
    r"\\label\{[^}]*\}",
]
A3 = r"\bA3\b"
FROZEN_NAME = re.compile(r"ViMMKG(?:~|\\ | )2\.0")
NUM = re.compile(r"(\$-\$|\$\\pm\$|\\pm|−|±|\+|(?<![\w-])-(?=\d))?(\d+(?:\.\d+)*)")


def tokens(text: str, labels: list[str]) -> list[str]:
    text = "\n".join(l for l in text.split("\n") if not l.lstrip().startswith("%"))
    for pat in labels:
        text = re.sub(pat, " ", text)
    text = text.replace("{,}", ".").replace("\\,", "")
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    out = []
    for sign, num in NUM.findall(text):
        sign = {"$-$": "-", "−": "-", "$\\pm$": "±", "\\pm": "±"}.get(sign, sign)
        out.append(sign + num)
    return out


def body(text: str) -> str:
    m = re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}", text, re.S)
    return m.group(0) if m else ""


def first_diff(a: list[str], b: list[str]) -> str:
    i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
    return f"token {i}: frozen {a[max(0, i - 3):i + 4]} vs public {b[max(0, i - 3):i + 4]} (lengths {len(a)}/{len(b)})"


def main() -> int:
    bad = 0
    for n in NAMES:
        old = (FROZEN / f"{n}.tex").read_text(encoding="utf-8")
        path = PUBLIC / f"{n}.tex"
        if not path.exists():
            print(f"MISSING {n}")
            bad += 1
            continue
        new = path.read_text(encoding="utf-8")
        # v1.3.0: the public tables name the dataset ViProKG; the frozen ones say "ViMMKG 2.0".
        # Map the frozen NAME (not a number) before tokenising, so "2.0" of the old name is not read as data.
        old = FROZEN_NAME.sub("ViProKG", old)
        ta, tb = tokens(old, LABELS + [A3]), tokens(new, LABELS + [A3])
        ba, bb = tokens(body(old), [A3, r"\[F8\]"]), tokens(body(new), [A3, r"\[F8\]"])
        ok_a, ok_b = ta == tb, ba == bb
        print(
            f"{'OK  ' if ok_a and ok_b else 'DIFF'} {n}  A(all numbers, {len(ta)} tokens)={'ok' if ok_a else 'DIFF'}  B(table body, {len(ba)} tokens)={'ok' if ok_b else 'DIFF'}"
        )
        if not ok_a:
            print("     A:", first_diff(ta, tb))
        if not ok_b:
            print("     B:", first_diff(ba, bb))
        bad += not (ok_a and ok_b)
    print(f"MISMATCHES: {bad} of {len(NAMES)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
