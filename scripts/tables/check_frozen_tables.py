"""Regression gate: compare the freshly generated tables with the frozen headline anchors.

Confirming a freeze without checking the headline numbers once let a regression of the MENTOR
W/Y rows (42.46/45.87) into the PDF. This script holds the frozen headline cells and checks that
the generated `.md` tables contain them, printing EVERY mismatching cell. Run before each freeze;
exit code 0 means no mismatch.

Only cells that must stay invariant are checked (headline W/Y, v2 A3'*, baseline*, prior). Cells
that were changed on purpose (added TSAM row, prior dagger) are not anchored.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper_b"

# ANCHOR = (file, substring of the row key, [expected values in order of appearance]).
ANCHOR = [
    (
        "bang_chinh.md",
        "MENTOR",
        ["42.50", "35.89", "45.26", "54.89", "44.80", "40.07", "47.00", "53.27"],
    ),
    ("bang_vimmkg2.md", "MENTOR** (A3′+L", ["41.20", "33.66", "53.93"]),  # overall MRR / H@1 / H@10
    (
        "bang_vimmkg2.md",
        "MENTOR − L (ablation)",
        ["40.89", "45.12", "33.83"],
    ),  # overall / regular / slice MRR
    ("bang_vimmkg2.md", "prior tần suất (lát)", ["33.55", "22.19", "42.15", "50.00"]),
]


def main() -> int:
    mismatches = []
    for fn, key, vals in ANCHOR:
        p = PAPER_DIR / fn
        if not p.exists():
            mismatches.append(f"{fn}: file MISSING")
            continue
        all_lines = p.read_text(encoding="utf-8").splitlines()
        rows = []
        for i, line in enumerate(all_lines):
            if key in line:
                rows.append(line)
                # collect continuation rows (multi-group model: empty name cell)
                j = i + 1
                while j < len(all_lines) and re.match(r"\|\s*\|", all_lines[j]):
                    rows.append(all_lines[j])
                    j += 1
        if not rows:
            mismatches.append(f"{fn}: no row containing '{key}'")
            continue
        text = " ".join(rows)
        numbers = re.findall(r"\d+\.\d+", text)
        for v in vals:
            if v not in numbers:
                mismatches.append(
                    f"{fn} [{key}]: MISSING expected value {v} (frozen anchor). "
                    f"Numbers found: {numbers[:12]}"
                )
    if mismatches:
        print("FROZEN-ANCHOR MISMATCH (" + str(len(mismatches)) + "):")
        for x in mismatches:
            print("  -", x)
        return 1
    print("0 mismatches: every headline cell matches the frozen anchor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
