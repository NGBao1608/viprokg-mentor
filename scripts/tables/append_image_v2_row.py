"""Append the ViProKG image-permutation rows to the image-ablation table (`bang_anh`).

Reads the scorer output `data/experimental/v3/anh_v2_score.json` (written by `scripts/score_anh_v2.py`) and replaces the
single placeholder row of ViProKG in `bang_anh.tex` / `bang_anh.md` by three rows (overall / mention-overlap /
mention-free) with the MRR of the real and permuted branches and delta(real - permuted) +- SD across seeds. The
off-branch cells stay `--`: the existing on/off control of v2 was measured at lambda_align 0, not at lambda 8.

Idempotent: a table that already carries the rows is left untouched. A backup `<file>.bak-<date>` is written next to the
file before any change. The headline file `hits_bang.tsv` is never touched.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MARK = "ViMMKG 2.0$^{\\dagger}$"
MARK_MD = "**ViMMKG 2.0** †"


def fmt(x: float) -> str:
    """Signed number for TeX: a real minus sign as `$-$`."""
    return f"{x:+.2f}".replace("-", "$-$")


def fmt_md(x: float) -> str:
    """Signed number for markdown, Vietnamese decimal comma."""
    return f"{x:+.2f}".replace("-", "−").replace(".", ",")


def rows(score: dict) -> list[tuple[str, float, float, float, float]]:
    out = []
    for cell in ("overall|all", "overall|overlap", "overall|free"):
        c = score["cells"][cell]
        out.append((cell, c["mrr_real"], c["mrr_perm"], c["delta"], st.stdev(c["delta_per_seed"])))
    return out


def patch_tex(text: str, score: dict) -> str:
    r = rows(score)
    labels = ("tổng", "CÓ-cạnh", "mention-free")
    lines = [
        f"\\multirow{{3}}{{*}}{{{MARK}}} & {labels[0]} & -- & {r[0][1]:.2f} & {r[0][2]:.2f} & -- & "
        f"{fmt(r[0][3])}\\,$\\pm$" + f"{r[0][4]:.2f} \\\\"
    ] + [
        f" & {labels[i]} & -- & {r[i][1]:.2f} & {r[i][2]:.2f} & -- & {fmt(r[i][3])}\\,$\\pm$"
        f"{r[i][4]:.2f} \\\\"
        for i in (1, 2)
    ]
    old_row = next(
        ln for ln in text.split("\n") if ln.startswith("ViMMKG 2.0 & tổng & \\multicolumn{5}")
    )
    assert text.count(old_row) == 1
    text = text.replace(old_row, "\n".join(lines))
    cap_old = "mention-free $-$0.67 (triệt tiêu ở tổng) --- khớp phát hiện xuyên suốt.}"
    assert text.count(cap_old) == 1
    text = text.replace(
        cap_old,
        "mention-free $-$0.67 (triệt tiêu ở tổng) --- khớp phát hiện xuyên suốt. "
        "v2: đối chứng hoán vị cùng run, 3 seed ($\\lambda$=8).}",
    )
    c = score["cells"]["overall|all"]
    foot_old = "087ff9effc6a8d72); v2 từ ❓AZ (BC74 §1b)."
    assert text.count(foot_old) == 1
    foot_new = (
        "087ff9effc6a8d72).\n"
        f"$^{{\\dagger}}$ ViMMKG~2.0 ($\\lambda$=8, 3 seed, cùng run): hoán vị ảnh chỉ giữa 7{{,}}790 thực thể có "
        "ảnh, mặt nạ giữ nguyên; $\\pm$ = SD qua seed. $\\Delta$(thật$-$hoán vị) tổng "
        f"{fmt(c['delta'])} (KTC qua seed $t$, df\\,=\\,2 [{fmt(c['ci_seed'][0])}; {fmt(c['ci_seed'][1])}]; bootstrap "
        f"theo khoá [{fmt(c['ci_boot'][0])}; {fmt(c['ci_boot'][1])}]): \\emph{{không phát hiện}} tín hiệu từ nội dung "
        "ảnh (KTC qua seed rộng hơn ngưỡng 0.65, nên không đọc là ``tương đương''). Ô ``tắt'' của v2 không "
        "chạy ở $\\lambda$=8; đối chứng bật/tắt có sẵn đo ở $\\lambda_{\\text{align}}$=0: $\\Delta$(thật$-$tắt) tổng "
        "$-$0.02 [$-$0.57; +0.52]."
    )
    return text.replace(foot_old, foot_new)


def patch_md(text: str, score: dict) -> str:
    r = rows(score)
    c = score["cells"]["overall|all"]
    names = ("tổng", "CÓ-cạnh", "mention-free")

    def vi(x: str) -> str:
        return x.replace(".", ",")

    new_rows = [
        f"| {MARK_MD if i == 0 else ''} | {names[i]} | — | {vi(f'{r[i][1]:.2f}')} | {vi(f'{r[i][2]:.2f}')} | — | "
        f"{vi(fmt_md(r[i][3]))} ±{vi(f'{r[i][4]:.2f}')} |"
        for i in range(3)
    ]
    old_row = next(ln for ln in text.split("\n") if ln.startswith("| **ViMMKG 2.0** | tổng |"))
    text = text.replace(old_row, "\n".join(new_rows))
    old_note = next(ln for ln in text.split("\n") if ln.startswith("- v2: ảnh trung tính"))
    note = (
        "- † v2 (λ=8, 3 seed, cùng run): hoán vị chỉ giữa 7.790 thực thể có ảnh, mặt nạ giữ nguyên. "
        f"Δ(thật−hoán vị) tổng {fmt_md(c['delta'])} (KTC seed [{fmt_md(c['ci_seed'][0])}; "
        f"{fmt_md(c['ci_seed'][1])}], bootstrap khoá [{fmt_md(c['ci_boot'][0])}; {fmt_md(c['ci_boot'][1])}]): "
        "không phát hiện tín hiệu từ nội dung ảnh. Ô “tắt” v2 không chạy ở λ=8 (bật/tắt có sẵn đo ở "
        "λ_align=0: Δ tổng −0,02 [−0,57; +0,52]). ± = SD qua seed."
    )
    return text.replace(old_note, note)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score", default=str(ROOT / "data/experimental/v3/anh_v2_score.json"))
    ap.add_argument("--tex", required=True, help="bang_anh.tex to patch")
    ap.add_argument("--md", default="", help="bang_anh.md to patch (optional)")
    a = ap.parse_args()
    score = json.load(open(a.score, encoding="utf-8"))
    for path, fn, marker in ((a.tex, patch_tex, MARK), (a.md, patch_md, MARK_MD)):
        if not path:
            continue
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if marker in text:
            print(f"{p.name}: rows already present, unchanged")
            continue
        bak = p.with_name(f"{p.name}.bak-{dt.date.today():%Y%m%d}")
        if not bak.exists():
            shutil.copy2(p, bak)
        p.write_text(fn(text, score), encoding="utf-8")
        print(f"{p.name}: v2 rows added (backup {bak.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
