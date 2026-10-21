#!/usr/bin/env python3
"""Reference counts of the mention-free-pair group for edge rules v1 / v2, measured with the
screening script's own `measure_run` (same instrument as the check that consumes them). No GPU.

Why this script exists instead of reading the number printed by the edge-rebuild script: the two
counts differ and the difference is not an error.
  - the edge-rebuild script counts on the TRIPLE FILE `valid.txt` and drops self-loops;
  - `mention_free_screen.py` counts on the KEYS of the dump `sweep_gL.tsv` and keeps self-loops
    (`frozenset((h, h))` has one element, so it is never in the edge set).
MKG-W validation: 6 028 vs 6 048, i.e. exactly 2 x 10 self-loops. DB15K differs more (14 048 vs
5 870) because its dump has only 7 922 keys rather than 2 x 9 904.
Cross-check: with the v1 rule this script must reproduce the registered `N_MENTION_FREE_V1`
exactly; if not, do not trust the v2 numbers it prints.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mention_free_screen as MFS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results_dir",
        "--ket_qua",
        default="data/from_kaggle/v3_18c",
        help="directory holding the reference run A3p_<dataset>_s42 (with sweep_gL.tsv)",
    )
    ap.add_argument("--data", default="data/experimental/v3/ext")
    ap.add_argument(
        "--edges_v1", "--canh_v1", default="data/experimental/v3/v10_canh_nhac_ten_{bo}.tsv"
    )
    ap.add_argument(
        "--edges_v2",
        "--canh_v2",
        default="data/experimental/v3/canh_v2/v10_canh_nhac_ten_{bo}_v2.tsv",
    )
    ap.add_argument("--baseline_run", "--moc", default="A3p_{bo}_s42")
    ap.add_argument("--bo", nargs="*", default=["MKG-W", "MKG-Y", "DB15K"])
    # `valid` is the default so the old code path is bit-identical; `both` prints the table
    # 2 rules x 2 splits x 3 datasets.
    ap.add_argument("--split", choices=["valid", "test", "both"], default="valid")
    # Write a TSV so reports are generated from the file, never typed by hand.
    ap.add_argument("--out_tsv", "--ra_tsv", default="")
    a = ap.parse_args()

    # v1 reference per split: `valid` is the registered canon; `test` was measured earlier and
    # matched exactly through an independent path. The v1 reference is the cross-check: if the
    # v1 rule does not reproduce it, the v2 numbers come from a different instrument.
    REF_V1 = {
        "valid": MFS.N_MENTION_FREE_V1,
        "test": {"MKG-W": 5080, "MKG-Y": 3598, "DB15K": 14186},
    }
    SWEEP = {"valid": "sweep_gL.tsv", "test": "sweep_gL_test.tsv"}
    splits = ["valid", "test"] if a.split == "both" else [a.split]

    print(
        f"T12 reference - measured with `mention_free_screen.measure_run` on {Path(a.results_dir).resolve()}"
    )
    print(f"split: {', '.join(splits)}")
    print("=" * 100)
    print(
        f"{'dataset':8}{'split':7}{'n keys':>9}{'n_free v1':>12}{'ref v1':>9}{'':3}"
        f"{'n_free v2':>12}{'d v2-v1':>9}"
    )
    v2_new, errors, result_rows = {}, [], []
    for bo in a.bo:
        for split_name in splits:
            run = Path(a.results_dir) / a.baseline_run.format(bo=bo)
            if not (run / SWEEP[split_name]).exists():
                print(f"{bo:8}{split_name:7} missing {run}/{SWEEP[split_name]} - skipped")
                errors.append(f"{bo}/{split_name}")
                continue
            d = Path(a.data) / bo
            e2i = {
                r.split("\t")[0]: int(r.split("\t")[1])
                for r in (d / "entity2id.tsv").read_text(encoding="utf-8").splitlines()[1:]
            }
            m1 = MFS.measure_run(
                run,
                MFS.load_mention_edges(Path(a.edges_v1.format(bo=bo))),
                e2i,
                SWEEP[split_name],
            )
            m2 = MFS.measure_run(
                run,
                MFS.load_mention_edges(Path(a.edges_v2.format(bo=bo))),
                e2i,
                SWEEP[split_name],
            )
            expected = REF_V1[split_name].get(bo)
            ok = expected is not None and m1["n_khong"] == expected
            print(
                f"{bo:8}{split_name:7}{m1['n']:>9}{m1['n_khong']:>12}{expected if expected else '-':>9}"
                f"{'  OK' if ok else '  FAIL'}{m2['n_khong']:>12}"
                f"{m2['n_khong'] - m1['n_khong']:>+9}"
            )
            if not ok:
                errors.append(
                    f"{bo}/{split_name}: v1 does NOT reproduce the reference ({m1['n_khong']} != {expected})"
                )
            v2_new[(bo, split_name)] = m2["n_khong"]
            result_rows.append((bo, split_name, m1["n"], m1["n_khong"], expected, m2["n_khong"]))
    print("=" * 100)
    if errors:
        print("FAILED:", "; ".join(map(str, errors)))
        print("   Do not use the v2 numbers above. Wrong instrument or wrong edge file.")
        return 1
    print(
        "OK: rule v1 reproduces the canon on every dataset, so the v2 numbers share the instrument."
    )
    if a.out_tsv:
        out = Path(a.out_tsv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            f.write("bo\tsplit\tn_khoa\tn_khong_v1\tmoc_v1\tn_khong_v2\n")
            for r in result_rows:
                f.write("\t".join(str(x) for x in r) + "\n")
        print(f"-> {out} ({len(result_rows)} rows)")
    print("\nPaste into `src/mention_free_screen.py`:")
    for split_name in splits:
        name = "N_MENTION_FREE_V2" if split_name == "valid" else "N_MENTION_FREE_V2_TEST"
        pairs = [(b, v) for (b, s2), v in v2_new.items() if s2 == split_name]
        print(
            f'{name}: "dict[str, int | None]" = {{'
            + ", ".join(f'"{b}": {v}' for b, v in pairs)
            + "}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
