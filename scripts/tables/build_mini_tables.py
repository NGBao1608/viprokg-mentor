#!/usr/bin/env python3
"""Generate the Vietnamese tables for the Vietnamese mini-paper from the canonical .md files.

The canonical .tex files (English version) are not touched. Reads
`paper_b/bang_{chinh,phuluc_db15k,vimmkg2}.md` (canonical output, numbers already generated), then:
  - drops rows that are not yet peer-reviewed (double dagger / peer=N) from the main and DB15K
    tables (preprints are not allowed in the main table);
  - uses Vietnamese captions and right-aligned numeric columns (2-decimal numbers aligned);
  - keeps ViProKG as a SEPARATE table (different instrument, has slices, no cross-comparison),
    placed next to the main table in the mini-paper with a note.
Numbers are read from the canonical .md, never typed by hand.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper_b"


def read_md_table(p: Path) -> tuple[list[str], list[list[str]]]:
    """Return (header, rows) of the first markdown table in the file."""
    lines = [
        line for line in p.read_text(encoding="utf-8").splitlines() if line.strip().startswith("|")
    ]
    table_rows = [[c.strip() for c in line.strip().strip("|").split("|")] for line in lines]
    header = table_rows[0]
    body = [r for r in table_rows[2:] if any(c for c in r)]  # skip the '---' separator row
    return header, body


def cell_to_tex(c: str, empty: str = "--") -> str:
    """Markdown cell -> LaTeX cell: **x** -> bold (also inline), *x* -> italic, dash -> empty, +- -> $\\pm$."""
    c = c.strip()
    if c in ("", "–", "-", "--"):
        return empty
    c = re.sub(r"\*\*(.+?)\*\*", lambda m: rf"\textbf{{{m.group(1)}}}", c)
    c = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", lambda m: rf"\emph{{{m.group(1)}}}", c)
    c = c.replace("±", r"\,$\pm$\,")
    return c


def is_unreviewed(r0: str) -> bool:
    return "‡" in r0


def is_block_title(row: list[str]) -> bool:
    # block-title row: first cell is **...**, the other cells are empty
    return row[0].startswith("**") and all(not c for c in row[1:])


def build_standard_table(md: Path, row_label: str, caption_vi: str, label: str) -> str:
    header, body = read_md_table(md)
    n_num = len(header) - 2  # drop the Model + Peer columns
    body = [
        r for r in body if not is_unreviewed(r[0])
    ]  # drop rows with a double dagger (not peer-reviewed)
    # drop block-title rows left empty after filtering (block with only its title)
    kept = []
    for k, r in enumerate(body):
        if is_block_title(r):
            rest = body[k + 1 :]
            if not any(not is_block_title(x) for x in rest[: _next_block_start(rest)]):
                continue
        kept.append(r)
    col = "l " + " ".join(["r"] * n_num)
    L = [
        f"% SINH bởi scripts/v3/sinh_mini_vi.py từ {md.name} — bảng VI cho mini-paper. XeLaTeX.",
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        rf"\caption{{{caption_vi}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col}}}",
        r"\toprule",
    ]
    # two-level group header: group by dataset prefix ("MKG-W MRR" -> dataset "MKG-W", metric "MRR")
    dataset_seq, metric_seq = [], []
    for h in header[2:]:
        dataset, _, metric = h.rpartition(" ")
        dataset_seq.append(dataset or h)
        metric_seq.append(metric or h)
    # merge consecutive columns of the same dataset
    groups, i = [], 0
    while i < len(dataset_seq):
        j = i
        while j < len(dataset_seq) and dataset_seq[j] == dataset_seq[i]:
            j += 1
        groups.append((dataset_seq[i], j - i))
        i = j
    top_row = r"\multirow{2}{*}{Mô hình}"
    cm, c0 = [], 2
    for dataset, w in groups:
        top_row += rf" & \multicolumn{{{w}}}{{c}}{{{dataset}}}"
        cm.append(rf"\cmidrule(lr){{{c0}-{c0 + w - 1}}}")
        c0 += w
    L.append(top_row + r" \\")
    L.append("".join(cm))
    L.append(" & " + " & ".join(metric_seq) + r" \\")
    L.append(r"\midrule")
    for r in kept:
        if is_block_title(r):
            name = cell_to_tex(r[0])
            L.append(rf"\multicolumn{{{n_num + 1}}}{{l}}{{\itshape {_unbold(name)}}} \\")
            continue
        name = cell_to_tex(r[0])
        cells = [cell_to_tex(c) for c in r[2:]]  # drop the Peer column
        if "MENTOR" in r[0]:
            L.append(r"\midrule")
        L.append(f"{name} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def _next_block_start(rest: list[list[str]]) -> int:
    for k, x in enumerate(rest):
        if is_block_title(x):
            return k
    return len(rest)


def _unbold(s: str) -> str:
    return s.replace(r"\textbf{", "").rstrip("}") if s.startswith(r"\textbf{") else s


def build_v2_table(md: Path, label: str) -> str:
    header, body = read_md_table(md)
    n_bold = sum(c.strip().startswith("**") for row in body for c in row[3:])
    cap = (
        "Dự đoán liên kết trên \\textbf{ViMMKG 2.0} (của chúng tôi), 2{,}637 triple kiểm thử "
        "= 5{,}274 truy vấn; 3 dòng mỗi mô hình: \\emph{tổng\\,/\\,thường\\,/\\,lát}. Đơn vị \\%. "
        "\\textbf{In đậm} = vượt mốc-ngoài-mạnh-nhất từng ô (gồm TSAM) với KTC 95\\% qua seed trên 0 "
        f"({n_bold}/12 ô). \\textbf{{KHÔNG so chéo với Bảng~\\ref{{tab:main}}}}: ViMMKG~2.0 dùng kênh ảnh, "
        "37{,}43\\% triple kiểm thử là lát thuộc tính (ba benchmark chuẩn không có), và giao thức "
        "cấm so chéo cột không-cạnh."
    )
    col = "l l " + " ".join(["r"] * (len(header) - 3))
    L = [
        f"% SINH bởi scripts/v3/sinh_mini_vi.py từ {md.name} — bảng VI cho mini-paper. XeLaTeX.",
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4.5pt}",
        rf"\caption{{{cap}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col}}}",
        r"\toprule",
        "Mô hình & Nhóm & " + " & ".join(header[3:]) + r" \\",
        r"\midrule",
    ]
    first = True
    for r in body:
        if is_block_title(r):
            continue
        new_model = bool(r[0].strip())  # row that starts a new model (non-empty model cell)
        model = cell_to_tex(r[0], empty="") if new_model else ""
        groups = r[1]
        if new_model and not first:
            L.append(r"\midrule")
        first = False
        cells = [cell_to_tex(c) for c in r[3:]]
        L.append(f"{model} & {groups} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def replace_caption(src: Path, vi: str) -> str:
    """Copy a canonical .tex, replacing \\caption{...} (balanced braces) with the Vietnamese caption; numbers unchanged."""
    s = src.read_text(encoding="utf-8")
    k = s.index(r"\caption{")
    i = k + len(r"\caption{")
    depth = 1
    while depth:
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
        i += 1
    return (
        f"% caption VI (mini-paper) — thân bảng chép từ {src.name}, KHÔNG đổi số.\n"
        + s[:k]
        + rf"\caption{{{vi}}}"
        + s[i:]
    )


def main() -> None:
    cap_main = (
        "Dự đoán liên kết trên hai benchmark chuẩn (MKG-W, MKG-Y). Đơn vị \\%; "
        "chỉ liệt kê các công trình \\emph{đã phản biện}; \\textbf{in đậm} = tốt nhất mỗi "
        "cột (gồm MENTOR). Các bản tiền ấn (arXiv) không đưa vào bảng chính. "
        "Số chép từ bảng gốc mỗi công trình (xuất xứ: \\texttt{moc\\_ngoai\\_2026.tsv})."
    )
    (PAPER_DIR / "mini_bang_chinh.tex").write_text(
        build_standard_table(PAPER_DIR / "bang_chinh.md", "MENTOR", cap_main, "tab:main"),
        encoding="utf-8",
    )
    cap_db = (
        "Phụ lục --- DB15K. Đơn vị \\%; chỉ công trình đã phản biện. Mọi cấu hình của chúng "
        "tôi dừng ở $\\sim$40{,}2 MRR, dưới ngưỡng dừng tiền đăng ký 42{,}0, nên DB15K không "
        "dùng để chống đỡ claim nào."
    )
    (PAPER_DIR / "mini_bang_db15k.tex").write_text(
        build_standard_table(PAPER_DIR / "bang_phuluc_db15k.md", "MENTOR", cap_db, "tab:db15k"),
        encoding="utf-8",
    )
    (PAPER_DIR / "mini_bang_vimmkg2.tex").write_text(
        build_v2_table(PAPER_DIR / "bang_vimmkg2.md", "tab:vimmkg2"), encoding="utf-8"
    )
    # ablation + information ceiling: keep the canonical .tex body, only replace the caption
    (PAPER_DIR / "mini_bang_ablation.tex").write_text(
        replace_caption(
            PAPER_DIR / "bang_ablation.tex",
            "Ablation thành phần (MRR tổng trên test, \\%, 3 seed, [KTC 95\\%, $t$, df\\,=\\,2]). "
            "Bước lớn nhất là +L (kênh từ vựng nhắc-tên). Huấn luyện GPU không tất định bit "
            "(nhiễu run-qua-run $\\sim$0{,}3--0{,}65 điểm); hiệu dưới mức này không được claim.",
        ),
        encoding="utf-8",
    )
    (PAPER_DIR / "mini_bang_ceiling.tex").write_text(
        replace_caption(
            PAPER_DIR / "bang_tran_thong_tin.tex",
            "Thăm dò trần thông tin (Analysis, \\%). MKG-W, test, 3 seed nền. Prior tham số là thăm dò "
            "\\textsc{khám phá} trên nền A3$'$ ($\\lambda{=}0{,}1$), không phải mục của bảng chính.",
        ),
        encoding="utf-8",
    )
    print(
        "wrote paper_b/mini_bang_{chinh,db15k,vimmkg2,ablation,ceiling}.tex (Vietnamese, no unreviewed rows, from canon)"
    )


if __name__ == "__main__":
    main()
