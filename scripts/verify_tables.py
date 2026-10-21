"""Regenerate every generated table from the shipped inputs and compare it bit-for-bit
with the frozen copy in ``tables_frozen/`` (sha256). Then run the regression gate.

Exit code 0 means: zero mismatches and ``check_frozen_tables.py`` returned 0.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "scripts" / "tables"
FROZEN = ROOT / "tables_frozen"
OUT = ROOT / "paper_b"

GENERATORS = [
    "build_main_tables.py",
    "build_mention_split_table.py",
    "build_central_ablation_table.py",
    "build_mini_tables.py",
]
GENERATED = [
    "bang_chinh",
    "bang_vimmkg2",
    "bang_phuluc_db15k",
    "bang_trung_tam",
    "bang_canh_v2",
    "mini_bang_chinh",
    "mini_bang_vimmkg2",
    "mini_bang_db15k",
    "mini_bang_ablation",
    "mini_bang_ceiling",
]
CANON_NUMBERS = {
    "data/experimental/v3/hits_bang.tsv": "68ac108bf1440bf5",
    "data/experimental/v3/cham_canh_v2_lambda8.json": "45a7d6c0c6056400",
}


DATE = re.compile(rb"sinh \d{4}-\d{2}-\d{2}")


def sha16(path: Path) -> str:
    """First 16 hex chars of the sha256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def sha16_nodate(path: Path) -> str:
    """Same, after replacing the generation date written by the generators ("sinh YYYY-MM-DD")."""
    return hashlib.sha256(DATE.sub(b"sinh <date>", path.read_bytes())).hexdigest()[:16]


HAND_INPUTS = [
    "bang_ablation",
    "bang_tran_thong_tin",
]  # hand-assembled tables read by build_mini_tables.py


def main() -> int:
    bad: list[str] = []
    OUT.mkdir(exist_ok=True)
    for name in HAND_INPUTS:
        for ext in ("tex", "md"):
            src = FROZEN / f"{name}.{ext}"
            if src.exists() and not (OUT / src.name).exists():
                shutil.copy(src, OUT / src.name)
    for gen in GENERATORS:
        rc = subprocess.run([sys.executable, str(TABLES / gen)], capture_output=True).returncode
        print(f"run {gen}: rc={rc}")
        if rc:
            bad.append(f"{gen} rc={rc}")
    for name in GENERATED:
        for ext in ("tex", "md"):
            new, old = OUT / f"{name}.{ext}", FROZEN / f"{name}.{ext}"
            if not old.exists():
                continue
            ok = new.exists() and sha16_nodate(new) == sha16_nodate(old)
            same = new.exists() and sha16(new) == sha16(old)
            note = "" if same else " (identical except generation date)" if ok else ""
            print(
                f"{'OK  ' if ok else 'DIFF'} {name}.{ext} {sha16(new) if new.exists() else '-'}{note}"
            )
            if not ok:
                bad.append(f"{name}.{ext}")
    for rel, want in CANON_NUMBERS.items():
        got = sha16(ROOT / rel)
        print(f"{'OK  ' if got == want else 'DIFF'} {rel} {got}")
        if got != want:
            bad.append(rel)
    rc = subprocess.run([sys.executable, str(TABLES / "check_frozen_tables.py")]).returncode
    print(f"check_frozen_tables rc={rc}")
    if rc:
        bad.append("check_frozen_tables")
    for script in ("build_public_tables.py", "diff_numbers_only.py"):
        rc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script)],
            capture_output=script == "build_public_tables.py",
        ).returncode
        print(f"{script} rc={rc}")
        if rc:
            bad.append(script)
    print("MISMATCHES:", len(bad), bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
