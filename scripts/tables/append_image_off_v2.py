"""Fill the "off" column of the ViProKG rows of the image-ablation table (`bang_anh`).

Run after `append_image_v2_row.py`. Reads the scorer output of the off branch (`score_anh_v2.py --tag_b off`), puts the
MRR of the images-off branch and delta(real - off) +- SD across seeds into the three v2 rows and replaces the footnote
sentence that said the cell was not measured at lambda 8. Idempotent (a filled table is left untouched); a backup
`<file>.bak-<date>` is written before any change.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CELLS = (("tổng", "overall|all"), ("CÓ-cạnh", "overall|overlap"), ("mention-free", "overall|free"))


def tex_num(x: float) -> str:
    return f"{x:+.2f}".replace("-", "$-$")


def md_num(x: float) -> str:
    return f"{x:+.2f}".replace("-", "−").replace(".", ",")


def patch_tex(text: str, off: dict) -> str:
    start = text.index("ViMMKG 2.0$^{\\dagger}$")
    head, tail = text[:start], text[start:]
    for label, cell in CELLS:
        c = off["cells"][cell]
        sd = st.stdev(c["delta_per_seed"])
        pat = re.compile(rf"(& {re.escape(label)} & )-- (& [\d.]+ & [\d.]+ & )-- (& )")
        assert len(pat.findall(tail)) == 1, f"placeholder row not found for {label}"
        tail = pat.sub(
            rf"\g<1>{c['mrr_b']:.2f} \g<2>{tex_num(c['delta'])}\\,$\\pm$" + f"{sd:.2f} \\g<3>", tail
        )
    c = off["cells"]["overall|all"]
    verdict = "ĐẠT" if off["criterion"]["pass"] else "SAI"
    old = re.compile(
        r"Ô ``tắt'' của v2 không chạy ở \$\\lambda\$=8; đối chứng bật/tắt có sẵn đo ở \$\\lambda_\{\\text\{align\}\}\$=0: \$\\Delta\$\(thật\$-\$tắt\) tổng \$-\$0\.02 \[\$-\$0\.57; \+0\.52\]\."
    )
    assert len(old.findall(tail)) == 1, "footnote sentence not found"
    new = (
        "Ô ``tắt'' của v2: 3 run \\texttt{use\\_image}=0 ở $\\lambda$=8 (cùng cấu hình); $\\Delta$(thật$-$tắt) tổng "
        f"{tex_num(c['delta'])} (KTC qua seed [{tex_num(c['ci_seed'][0])}; {tex_num(c['ci_seed'][1])}]; bootstrap theo "
        f"khoá [{tex_num(c['ci_boot'][0])}; {tex_num(c['ci_boot'][1])}]), tiêu chí trung tính $|\\Delta|\\le$0.65: {verdict}. "
        "Đối chứng bật/tắt có sẵn đo ở $\\lambda_{\\text{align}}$=0 cho $\\Delta$ tổng $-$0.02 [$-$0.57; +0.52]."
    )
    tail = old.sub(lambda m: new, tail)
    out = head + tail
    cap = "v2: đối chứng hoán vị cùng run, 3 seed ($\\lambda$=8).}"
    assert out.count(cap) == 1
    return out.replace(cap, "v2: đối chứng hoán vị cùng run và tắt ảnh, 3 seed ($\\lambda$=8).}")


def patch_md(text: str, off: dict) -> str:
    lines = text.split("\n")
    names = [n for n, _ in CELLS]
    first = next(i for i, ln in enumerate(lines) if ln.startswith("| **ViMMKG 2.0** †"))
    for k, (label, cell) in enumerate(CELLS):
        c = off["cells"][cell]
        sd = st.stdev(c["delta_per_seed"])
        cols = lines[first + k].split("|")
        assert (
            cols[2].strip() == names[k] and cols[3].strip() == "—" and cols[6].strip() == "—"
        ), "md row not a placeholder"
        cols[3] = f" {c['mrr_b']:.2f} ".replace(".", ",")
        cols[6] = f" {md_num(c['delta'])} ±{sd:.2f} ".replace(".", ",")
        lines[first + k] = "|".join(cols)
    c = off["cells"]["overall|all"]
    verdict = "ĐẠT" if off["criterion"]["pass"] else "SAI"
    idx = next(i for i, ln in enumerate(lines) if ln.startswith("- † v2"))
    pat = re.compile(
        r"Ô “tắt” v2 không chạy ở λ=8 \(bật/tắt có sẵn đo ở λ_align=0: Δ tổng −0,02 \[−0,57; \+0,52\]\)\."
    )
    assert pat.search(lines[idx]), "md footnote sentence not found"
    new = (
        f"Ô “tắt” v2: 3 run use_image 0 ở λ=8; Δ(thật−tắt) tổng {md_num(c['delta'])} (KTC seed "
        f"[{md_num(c['ci_seed'][0])}; {md_num(c['ci_seed'][1])}], bootstrap khoá [{md_num(c['ci_boot'][0])}; "
        f"{md_num(c['ci_boot'][1])}]), tiêu chí trung tính |Δ|≤0,65: {verdict} (bật/tắt có sẵn ở λ_align=0: "
        "Δ tổng −0,02 [−0,57; +0,52])."
    )
    lines[idx] = pat.sub(lambda m: new, lines[idx])
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score_off", default=str(ROOT / "data/experimental/v3/anh_v2_score_off.json"))
    ap.add_argument("--tex", required=True)
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    off = json.load(open(a.score_off, encoding="utf-8"))
    assert off["tag_b"] == "off", "score file is not an off-branch score"
    for path, fn in ((a.tex, patch_tex), (a.md, patch_md)):
        if not path:
            continue
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        marker = "3 run \\texttt{use\\_image}=0" if path == a.tex else "3 run use_image 0"
        if marker in text:
            print(f"{p.name}: off cells already filled, unchanged")
            continue
        bak = p.with_name(f"{p.name}.bak-{dt.date.today():%Y%m%d}-off")
        if not bak.exists():
            shutil.copy2(p, bak)
        p.write_text(fn(text, off), encoding="utf-8")
        print(f"{p.name}: off cells filled (backup {bak.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
