"""Central ablation table: overall MRR per step plus the with / without mention-edge gain of the
+L step.

Chain: TuckER (structure-only) -> base (TuckER + text/image + align) -> +L (A3') -> +gate (OOF gate).
The overall column of each step comes from ablation_chain.json (headline numbers). The two right
columns are the gain of the +L step split by mention-edge group (Delta_L = A3' - base, per-key
paired, 3 seeds): with edge (T_overlap) and mention-free (T_free), from the delta-L scores runs
(v2 from the lambda-8 mention-edge scoring). Overall and Delta_L use the same score runs and are
not mixed with the g0.15 grid. For v2 the seed CI covers 0 and the difference is <= ~0.3.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ABLATION_CHAIN = json.load(open(ROOT / "data/inputs/ablation_chain.json", encoding="utf-8"))

# Delta_L (A3' - base) by mention-edge group, matching bang_delta_L.md. (with, without, DiD).
DELTA_L = {
    "MKG-W": (14.06, -0.51, 14.57),
    "MKG-Y": (17.13, -1.10, 18.23),
    "DB15K": (6.81, -0.85, 7.66),
    "ViMMKG2": (0.94, 0.09, 0.85),
}
LABELS_TEX = {
    "MKG-W": "MKG-W ($\\lambda$=4)",
    "MKG-Y": "MKG-Y ($\\lambda$=0.1)",
    "DB15K": r"DB15K ($\lambda$=2)$^\ddagger$",
    "ViMMKG2": r"ViMMKG 2.0 ($\lambda$=8)",
}
LABELS_MD = {
    "MKG-W": "MKG-W (λ4)",
    "MKG-Y": "MKG-Y (λ0,1)",
    "DB15K": "DB15K (λ2) ‡",
    "ViMMKG2": "ViMMKG 2.0 (λ8)",
}


def point_only(s: str) -> str:
    """'42.50 [42.02;42.98]' -> '42.50' (point value only, for a compact table)."""
    return s.split(" ")[0]


def main() -> int:
    datasets = ("MKG-W", "MKG-Y", "DB15K", "ViMMKG2")
    L = [
        r"% SINH bởi scripts/v3/sinh_bang_trung_tam.py (TIN_278 B7) — ablation + Δ_L theo nhóm cạnh.",
        r"% Nguồn: ablation_chain.json (tổng, BC121) + bang_delta_L (Δ_L scores run). CẦN booktabs.",
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Bảng ablation trung tâm: MRR tổng (\%) theo từng bước hợp nhất "
        r"(TuckER $\to$ base $\to$ +L $\to$ +gate), và \textbf{độ lợi của bước +L tách theo nhóm cạnh} "
        r"($\Delta_L$ = A3$'-$nền, ghép cặp per-key, 3 seed): $\Delta_L$ dồn vào nhóm \emph{có cạnh "
        r"nhắc-tên} (T\_overlap), gần 0 hoặc âm ở \emph{mention-free} (T\_free). DiD$_L$ = "
        r"$\Delta_L^{\text{overlap}}-\Delta_L^{\text{free}}$. KTC qua seed ($t$, df=2).}",
        r"\label{tab:trung_tam}",
        r"\begin{tabular}{l r r r r r r r}",
        r"\toprule",
        r"Bộ & TuckER & base & +L & +gate & $\Delta_L$ T\_overlap & $\Delta_L$ T\_free & DiD$_L$ \\",
        r"\midrule",
    ]
    M = [
        "<!-- SINH bởi scripts/v3/sinh_bang_trung_tam.py (TIN_278 B7) -->",
        "",
        "# Bảng ablation trung tâm — tổng/bước + Δ_L (A3′−nền) theo nhóm cạnh (MRR %, 3 seed)",
        "",
        "| bộ | TuckER | base | +L | +gate | Δ_L CÓ-cạnh | Δ_L KHÔNG-cạnh | DiD_L |",
        "|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for b in datasets:
        a = ABLATION_CHAIN[b]
        tucker = point_only(a["tucker"]) if a["tucker"] != "–" else "--"
        tucker_md = point_only(a["tucker"]) if a["tucker"] != "–" else "–"
        base, plus_l, gate = point_only(a["nen"]), point_only(a["a3"]), point_only(a["cong"])
        overlap, free, did = DELTA_L[b]
        overlap_s = f"+{overlap:.2f}" if overlap >= 0 else f"$-${abs(overlap):.2f}"
        free_s = f"+{free:.2f}" if free >= 0 else f"$-${abs(free):.2f}"
        L.append(
            f"{LABELS_TEX[b]} & {tucker} & {base} & {plus_l} & {gate} & \\textbf{{{overlap_s}}} & {free_s} & +{did:.2f} \\\\"
        )
        overlap_md = f"+{overlap:.2f}" if overlap >= 0 else f"−{abs(overlap):.2f}"
        free_md = f"+{free:.2f}" if free >= 0 else f"−{abs(free):.2f}"
        M.append(
            f"| {LABELS_MD[b]} | {tucker_md} | {base} | {plus_l} | {gate} | **{overlap_md}** | {free_md} | +{did:.2f} |"
        )
    L += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\begin{minipage}{\linewidth}\footnotesize\raggedright",
        r"base = TuckER + \textbf{text-only} (W/Y/DB15K, use\_image 0) hoặc text+ảnh (v2); +align gộp trong base. Bước lớn nhất = +L (kênh lexical mention). $\Delta_L$ dồn vào T\_overlap (+6.8 đến +17.1) "
        r"nhưng $\approx$0/âm ở T\_free trên W/Y/DB. \textbf{ViMMKG 2.0:} $\Delta_L$ yếu "
        r"(+0.94/+0.09, DiD +0.85); KTC qua seed \textbf{ôm 0}, bootstrap khoá ở biên, hiệu "
        r"$\le\sim$0.3 --- không claim kênh L chuyển sang v2. +gate thêm nhỏ (W +0.35/Y +0.74/DB "
        r"+0.61); v2 +gate $<$ A3$'$ nên headline v2 = A3$'$* (không gate). $^\ddagger$DB15K = "
        r"clean-valid; $\Delta$ trong-đường hợp lệ, không so phụ lục split gốc (cùng dụng cụ). "
        r"TuckER DB15K ở khối uni-modal (không chạy lại).",
        r"\end{minipage}",
        r"\end{table}",
    ]
    (ROOT / "paper_b/bang_trung_tam.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    (ROOT / "paper_b/bang_trung_tam.md").write_text("\n".join(M) + "\n", encoding="utf-8")
    print("wrote paper_b/bang_trung_tam.{tex,md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
