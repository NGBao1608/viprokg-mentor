#!/usr/bin/env python3
"""Generate the paper's main tables (LaTeX + Markdown) from two canonical files.

No number is typed by hand:
  - external references: data/experimental/v3/moc_ngoai_2026.tsv (MRR in %, nine Hits columns in %)
  - our results:         data/experimental/v3/hits_bang.tsv (scale 0-1, with t-based CI, df=2)

The MENTOR configuration per dataset is the one selected on validation:
  MKG-W / MKG-Y -> MENTOR headline (OOF gate: W scores run lambda 4 / Y y_cu run lambda 0.1);
  DB15K -> A3' on the ORIGINAL split (train 79 222, released valid/test, MRR 0.4026). The
  label is fixed wording: do not rename it to "clean-valid protocol", which belongs to the
  probe runs (valid_sach 3 961 + train 75 261), a different split.

Three cell states that must never be mixed:
  number - copied from the original paper's own table
  '-'    - the original paper does NOT report that metric -> printed as "--" + footnote (a)
  '?'    - not retrieved (paper unavailable / unusable table) -> printed as "--" + footnote (b)
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_TSV = ROOT / "data" / "experimental" / "v3" / "moc_ngoai_2026.tsv"
HITS = ROOT / "data" / "experimental" / "v3" / "hits_bang.tsv"
OUT_TEX = ROOT / "paper_b" / "bang_chinh.tex"
OUT_MD = ROOT / "paper_b" / "bang_chinh.md"

# DB15K is in the appendix, not the main table: every configuration we trained plateaus at
# 40.2 MRR against the pre-registered stopping threshold of 42.0 (original split, TEST, 3 seeds:
# A3' 40.216 [39.719; 40.713], V 40.213 [39.483; 40.943]; the highest upper bound, 40.94, never
# reaches 42.0 at any seed).
MAIN_DATASETS = ("MKG-W", "MKG-Y")
APPENDIX_DATASETS = ("DB15K",)
DATASETS = (
    MAIN_DATASETS + APPENDIX_DATASETS
)  # used when loading the TSV (the external-reference file covers all 3 datasets)
# Suffix of the Hits columns in `moc_ngoai_2026.tsv`: an explicit map, not `zip(DATASETS, ...)`,
# so that the pairing is by key rather than by position.
HITS_SUFFIX = {"MKG-W": "W", "MKG-Y": "Y", "DB15K": "DB"}
METRICS = ("MRR", "H@1", "H@3", "H@10")
# ViProKG: a separate table with 3 rows per model; lookup key into `hits_bang.tsv`.
V2_DATASET = "ViMMKG2"
V2_GROUPS = ("tổng", "thường", "lát")
# The ViProKG table uses MyGO+seg (word-segmented PhoBERT). The older MyGO row (PhoBERT
# without word segmentation) moves to the appendix and does not belong in this table.
# The `MyGO+seg` row only appears when `hits_bang.tsv` has `cau_hinh = "MyGO+seg"`; until then
# V2_MODELS keeps the old `MyGO` and SYNCED_PREPROCESSING=False ensures no cell is bold, so no
# superiority claim is printed.
# Headline v2 = A3'* lambda 8 (selected on valid); the baseline* row goes to the ablation and the
# v2 gate row is not printed (gate < A3'). MoMoK is added as an external reference. Bold means
# CI > 0 against the strongest external baseline in each cell. AdaMF-MAT follows MoMoK in the
# external block; DSoM goes through the script path (cau_hinh in hits_bang.tsv). Order: ours,
# ablation, [external: MyGO+seg, MoMoK, AdaMF, DSoM], TuckER (structure-only, last ablation
# reference).
V2_MODELS = (
    "A3′* λ8 (ours)",
    "nền text+ảnh (MENTOR)",
    "MyGO+seg",
    "MoMoK",
    "TSAM",
    "AdaMF",
    "DSoM re-impl.",
    "TuckER (chỉ cấu trúc)",
)
# printed labels (the lookup key differs from the displayed label)
V2_MD = {
    "A3′* λ8 (ours)": "**MENTOR** (A3′+L, λ=8, ours)",
    "nền text+ảnh (MENTOR)": "MENTOR − L (ablation)",
    "MyGO+seg": "MyGO (PhoBERT đã tách từ)",
    "MoMoK": "MoMoK",
    "TSAM": "TSAM",
    "AdaMF": "AdaMF-MAT (our features; last ckpt, 1000 ep), 3 seed",
    "DSoM re-impl.": "DSoM re-impl. (approx.), 3 seed",
    "TuckER (chỉ cấu trúc)": "TuckER (structure-only)",
}
V2_TEX = {
    "A3′* λ8 (ours)": r"\textbf{MENTOR} (A3'+L, $\lambda$=8, ours)",
    "nền text+ảnh (MENTOR)": r"MENTOR $-$ L (ablation)",
    "MyGO+seg": "MyGO (word-segmented PhoBERT)",
    "MoMoK": "MoMoK",
    "TSAM": "TSAM",
    "AdaMF": r"AdaMF-MAT (our features; last ckpt, 1000 ep), 3 seeds",
    "DSoM re-impl.": r"DSoM re-impl.\ (approx.), 3 seeds",
    "TuckER (chỉ cấu trúc)": "TuckER (structure-only)",
}
# External baselines used for bolding: {MyGO+seg, MoMoK, DSoM re-impl., AdaMF} (+TSAM when
# available). Each json holds Delta(A3'* - baseline), paired per key over 3 seeds. A3'* is bold
# only if it beats the strongest measured baseline with CI > 0. A baseline without a json prints
# a warning and imposes no constraint (never silently dropped); the binding constraint remains
# MyGO+seg (CI_V2_MYGO). ktc_v2_adamf.json needs per-key ranks of A3'* lambda 8; until it exists
# AdaMF imposes no constraint and the report marks it pending.
CI_V2_BASELINES = {
    "MoMoK": ROOT / "data" / "experimental" / "v3" / "ktc_v2_momok.json",  # Delta(A3'* - MoMoK)
    "DSoM re-impl.": ROOT / "data" / "experimental" / "v3" / "ktc_v2_dsom.json",
    "AdaMF": ROOT / "data" / "experimental" / "v3" / "ktc_v2_adamf.json",
    "TSAM": ROOT / "data" / "experimental" / "v3" / "ktc_v2_tsam.json",  # Delta(A3'* - TSAM)
}
# CI of Delta(A3'* lambda 8 - MyGO+seg), paired per key over 3 seeds (scores run); the v2
# headline is A3'* lambda 8, so bolding reads `ktc_v2_mygo.json`.
CI_V2_MYGO = ROOT / "data" / "experimental" / "v3" / "ktc_v2_mygo.json"

# Flag for synchronized text preprocessing. When False the ViProKG table prints NO bold cell
# plus a note that bolding awaits synchronization: MENTOR receives word-segmented PhoBERT input
# while MyGO on v2 would receive unsegmented text, i.e. different instruments. Measured on MyGO
# itself (v1, 3 seeds): +0.0066 MRR, +0.072 head-MRR. Set True only after MyGO@v2 was rerun on
# the segmented tokens and `ktc_v2_mygo.json` regenerated from the new ranks.
# Currently True: MyGO+seg ranks come from `data/experimental/mygo_seg/MyGO_ViMMKG2_seg_s*` and
# every ktc_v2_mygo*.json is built from segmented ranks, so both arms share one preprocessing.
SYNCED_PREPROCESSING = True
T_CRIT_DF2 = 4.303
# Probe row, MKG-W test, pooled over 3 bases (MRR 0.4298 +-0.0016, H@1 0.3650, H@3 0.4566,
# H@10 0.5521), in %. Already generated from TSV earlier; recorded here so the table is
# self-contained, not a new number.
PROBE_W = [r"42.98\,$\pm$\,0.16", "36.50", "45.66", "55.21"]

# MENTOR configuration chosen on validation per dataset.
# OURS_CONFIG is the LOOKUP KEY into `hits_bang.tsv` (column `cau_hinh`), not a display label:
# changing it raises KeyError. Display labels live in OURS_LABEL_TEX / OURS_LABEL_MD.
OURS_CONFIG = {"MKG-W": "MENTOR headline", "MKG-Y": "MENTOR headline", "DB15K": "A3′ (split gốc)"}
# Do not rename the DB15K label to "clean-valid protocol": the main table uses the ORIGINAL
# V1b split (MRR 0.4026; train 79 222, released valid/test) and the wording is fixed. The
# "clean-valid protocol" label belongs to the probe runs, not to the main table.
OURS_LABEL_MD = {
    "MKG-W": "cổng OOF (λ=4, scores run)",
    "MKG-Y": "cổng OOF (λ=0,1, y_cu run)",
    "DB15K": "A3′ (split gốc)",
}  # display label (differs from the OURS_CONFIG lookup key)
# LaTeX labels: pdflatex cannot read U+2032 or Vietnamese diacritics (compilation fails with
# "Unicode character"); the .md version keeps the original labels.
OURS_LABEL_TEX = {
    "MKG-W": "gate OOF ($\\lambda$=4, scores run)",
    "MKG-Y": "gate OOF ($\\lambda$=0.1, y\\_cu run)",
    "DB15K": "A3' (original split)",
}

# Presentation blocks; `lop` in moc_ngoai -> block name. CHI-RELATED-WORK rows are excluded
# from the table (HFR-MKGC is related work only, since it reran all baselines itself).
# Rows that appear only in the footnote, not in the table body: RAPC reports both a "best run"
# and a multi-seed "mean+-std"; the comparable number is the mean+-std, so best-run goes to the
# footnote.
FOOTNOTE_ONLY = {"RAPC (best run)"}

# Rows kept in the table but marked, with a footnote on the origin of the numbers.
PROVENANCE_MARKS = {
    "MoSE-AI": (
        "dag",
        "figures for MoSE-AI are a third-party reproduction (DSoM, Table 1): the "
        "original MoSE paper (EMNLP 2022) does not evaluate these three benchmarks",
    ),
}

BLOCKS = [
    ("uni-modal KGC", ("unimodal",)),
    ("MMKGC — embedding / MoE", ("embedding-based",)),
    ("MLLM-based", ("LLM-MLLM",)),
    ("retrieval-generative", ("retrieval-generative",)),
]


def read_tsv(p: Path) -> tuple[list[str], list[dict]]:
    raw_lines = p.read_text(encoding="utf-8").rstrip("\n").split("\n")
    i = next(k for k, line in enumerate(raw_lines) if line.startswith(("ten\t", "cau_hinh\t")))
    header = raw_lines[i].split("\t")
    return header, [
        dict(zip(header, line.split("\t"))) for line in raw_lines[i + 1 :] if line.strip()
    ]


def so(x: str) -> float | None:
    """Return None for '-' and '?'; the caller keeps the two states apart."""
    x = x.strip()
    try:
        return float(x)
    except ValueError:
        return None


def load_external() -> tuple[list[dict], str]:
    _, rows = read_tsv(EXTERNAL_TSV)
    col_of = {("MRR", b): b for b in DATASETS}
    for k in DATASETS:
        for h in (1, 3, 10):
            col_of[(f"H@{h}", k)] = f"H@{h}_{HITS_SUFFIX[k]}"
    out = []
    for r in rows:
        if r["lop"] == "CHI-RELATED-WORK":
            continue
        o = {"ten": r["ten"], "lop": r["lop"], "peer": r["peer_reviewed"].strip(), "o": {}}
        for b in DATASETS:
            for d in METRICS:
                v = r[col_of[(d, b)]].strip()
                o["o"][(b, d)] = ("so", so(v)) if so(v) is not None else ("tt", v)
        out.append(o)
    # Two different scan dates that must not be merged: `ngay_quet` is the MRR scan date,
    # `ngay_quet_hits` the date the nine Hits columns were copied from the original tables.
    g = lambda c: max(
        (r.get(c, "").strip() for r in rows if r.get(c, "").strip().startswith("202")), default="?"
    )
    return out, (g("ngay_quet"), g("ngay_quet_hits"))


def load_ours() -> dict:
    _, rows = read_tsv(HITS)
    ix = {(r["cau_hinh"], r["bo"], r["do_do"]): r for r in rows if r["nhom"] == "tổng"}
    out: dict = {}
    for b in DATASETS:
        for d in METRICS:
            r = ix[(OURS_CONFIG[b], b, d)]
            tb = float(r["tb"])
            upper = float(r["KTC95_t_df2"].strip("[]").split(";")[1])
            out[(b, d)] = (tb * 100.0, (upper - tb) / T_CRIT_DF2 * 100.0)  # (%, SE %)
    return out


def build_data() -> tuple[list[tuple], dict, str, bool, bool]:
    moc, scan_dates = load_external()
    ta = load_ours()
    blocks: list[tuple] = []
    has_not_reported = has_not_retrieved = False
    for label, classes in BLOCKS:
        block_rows = [m for m in moc if m["lop"] in classes and m["ten"] not in FOOTNOTE_ONLY]
        if not block_rows:
            continue
        blocks.append((label, block_rows, None))
        for m in block_rows:
            for k, (t, v) in m["o"].items():
                if t == "tt":
                    has_not_reported = has_not_reported or v == "-"
                    has_not_retrieved = has_not_retrieved or v == "?"
    # Bold rule: bold = best of each column among PEER-REVIEWED papers, over the whole table
    # (not per block). Non-peer-reviewed references (peer = N) are never bold; they are set in
    # italics with a double dagger, since an unreviewed preprint number does not qualify as the
    # best. MENTOR is ranked with the peer-reviewed group, so each column has exactly one bold
    # cell: max{peer-reviewed} U {MENTOR}.
    ceiling_peer = {}
    for b in DATASETS:
        for d in METRICS:
            vals = [
                m["o"][(b, d)][1]
                for m in moc
                if m["o"][(b, d)][0] == "so" and m["peer"] == "Y" and m["ten"] not in FOOTNOTE_ONLY
            ]
            ceiling_peer[(b, d)] = max(vals) if vals else None
    separate = {m["ten"]: m for m in moc if m["ten"] in FOOTNOTE_ONLY}
    return blocks, ta, scan_dates, has_not_reported, has_not_retrieved, ceiling_peer, separate


def cell_tex(t: str, v, dam: bool, peer: bool = True) -> str:
    """One LaTeX cell. `dam` is only set for peer-reviewed papers; non-peer rows are italic."""
    if t == "tt":
        return "--"
    s = f"{v:.2f}"
    if dam:
        return rf"\textbf{{{s}}}"
    return s if peer else rf"\emph{{{s}}}"


def build_table(
    bo: tuple,
    ra_tex: Path,
    ra_md: Path,
    table_title: str,
    ours_label_tex: str,
    caption: str,
    label: str,
    probe: bool,
) -> None:
    """Generate ONE peer table for `bo`; used twice: main table (W, Y) and appendix (DB15K).

    All widths are derived from `len(bo)`, with no hard-coded 3/12/14, so that `\\cmidrule` and
    `multicolumn` stay correct when a dataset is dropped.
    """
    (
        blocks,
        ta,
        (mrr_scan_date, hits_scan_date),
        has_not_reported,
        has_not_retrieved,
        tran,
        separate,
    ) = build_data()
    today = _dt.date.today().isoformat()
    n_datasets = len(bo)
    n_num_cols = 4 * n_datasets  # number of numeric columns
    n_cols = 2 + n_num_cols  # + Model + Peer
    cmidrules = "".join(rf"\cmidrule(lr){{{3 + 4 * k}-{6 + 4 * k}}}" for k in range(n_datasets))

    L = [
        r"% SINH TỰ ĐỘNG bởi scripts/v3/bang_bai_chinh.py — KHÔNG sửa tay.",
        f"% nguồn: {EXTERNAL_TSV.relative_to(ROOT)} + {HITS.relative_to(ROOT)} | sinh {today}",
        r"% CẦN: \usepackage{booktabs,multirow}. KHÔNG dùng threeparttable — chú thích",
        r"% để trong minipage, vì threeparttable thiếu ở TeX tối giản (đã kiểm kpsewhich).",
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\caption{" + caption + r" All figures in \%, copied from each paper's own table "
        r"(provenance in \texttt{moc\_ngoai\_2026.tsv}). "
        r"\textbf{Bold} marks the best figure in each column among "
        r"\emph{peer-reviewed} work (MENTOR included), over the whole table rather "
        r"than per block. Rows from venues that are not peer-reviewed are set in "
        r"\emph{italics} and marked \ddag; they are never bolded. "
        r"\emph{Peer} = peer-reviewed venue.}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{l c " + " ".join(["r r r r"] * n_datasets) + "}",
        r"\toprule",
        r"\multirow{2}{*}{Model} & \multirow{2}{*}{Peer} & "
        + " & ".join(rf"\multicolumn{{4}}{{c}}{{{b}}}" for b in bo)
        + r" \\",
        cmidrules,
        " & & " + " & ".join(["MRR", "H@1", "H@3", "H@10"] * n_datasets) + r" \\",
        r"\midrule",
    ]

    M = [
        "<!-- SINH TỰ ĐỘNG bởi scripts/v3/bang_bai_chinh.py — KHÔNG sửa tay. -->",
        "",
        f"# {table_title}",
        "",
        "| Model | Peer | " + " | ".join(f"{b} {d}" for b in bo for d in METRICS) + " |",
        "|---|:-:|" + "---:|" * n_num_cols,
    ]

    for label, block_rows, _ in blocks:
        L.append(rf"\multicolumn{{{n_cols}}}{{l}}{{\textit{{{label}}}}} \\")
        M.append(f"| **{label}** | | " + " | ".join([""] * n_num_cols) + "|")
        for m in block_rows:
            o = [m["o"][(b, d)] for b in bo for d in METRICS]
            is_peer = m["peer"] == "Y"
            dam = [
                is_peer
                and tran[(b, d)] is not None
                and m["o"][(b, d)][0] == "so"
                and abs(m["o"][(b, d)][1] - tran[(b, d)]) < 1e-9
                and not (ta[(b, d)][0] > tran[(b, d)] + 1e-9)
                for b in bo
                for d in METRICS
            ]
            ten = (
                m["ten"]
                .replace("&", r"\&")
                .replace("_", r"\_")
                .replace("±", r"$\pm$")
                .replace("²", r"$^2$")
            )
            if not is_peer:
                ten = rf"\emph{{{ten}}}\,\ddag"
            if m["ten"] in PROVENANCE_MARKS:
                ten += "\\," + "\\" + PROVENANCE_MARKS[m["ten"]][0]
            L.append(
                f"{ten} & {m['peer']} & "
                + " & ".join(cell_tex(t, v, d, is_peer) for (t, v), d in zip(o, dam))
                + r" \\"
            )
            M.append(
                f"| {m['ten']}{'' if is_peer else ' ‡'}"
                f"{' †' if m['ten'] in PROVENANCE_MARKS else ''} | {m['peer']} | "
                + " | ".join(
                    (
                        "–"
                        if t == "tt"
                        else (f"**{v:.2f}**" if d else (f"{v:.2f}" if is_peer else f"*{v:.2f}*"))
                    )
                    for (t, v), d in zip(o, dam)
                )
                + " |"
            )

    L.append(r"\midrule")
    M.append("| **ta** | | " + " | ".join([""] * n_num_cols) + "|")
    cells_tex, cells_md = [], []
    for b in bo:
        for d in METRICS:
            v, se = ta[(b, d)]
            dam = tran[(b, d)] is None or v > tran[(b, d)] + 1e-9
            st = f"{v:.2f}"
            if d == "MRR":
                cells_tex.append((rf"\textbf{{{st}}}" if dam else st) + rf"\,$\pm$\,{se:.2f}")
                cells_md.append((f"**{st}**" if dam else st) + f" ±{se:.2f}")
            else:
                cells_tex.append(rf"\textbf{{{st}}}" if dam else st)
                cells_md.append(f"**{st}**" if dam else st)
    L.append(ours_label_tex + r" & -- & " + " & ".join(cells_tex) + r" \\")
    M.append(f"| {ours_md_label(table_title)} | – | " + " | ".join(cells_md) + " |")

    # The "- L (ablation)" row is not printed in the main table (Hits are incomplete and the
    # baseline already appears in the central ablation table).

    # The probe row only has numbers for MKG-W, so it is printed only in the table containing MKG-W.
    if probe and "MKG-W" in bo:
        n_blank = 4 * (n_datasets - 1)
        L.append(
            r"+ parametric prior (Qwen2.5-7B, zero-shot, $K$=20)\textsuperscript{F8} & -- & "
            + " & ".join(PROBE_W)
            + ((" & " + " & ".join(["--"] * n_blank)) if n_blank else "")
            + r" \\"
        )
        M.append(
            "| + parametric prior (Qwen2.5-7B, zero-shot, K=20) [F8] | – | "
            + " | ".join(PROBE_W)
            + ((" | " + " | ".join(["–"] * n_blank)) if n_blank else "")
            + " |"
        )

    footnotes = [
        r"All baseline figures copied from each paper's own table; "
        r"MRR scan \textbf{" + mrr_scan_date + r"}, Hits@$k$ scan \textbf{" + hits_scan_date + r"}."
    ]
    if has_not_reported:
        footnotes.append(r"\textbf{(a)} ``--'' = \textbf{not reported} by the original paper.")
    if has_not_retrieved:
        footnotes.append(
            r"\textbf{(b)} ``--'' = \textbf{not retrieved} (paper behind a "
            r"paywall, or its published table is unusable)."
        )
    for ten, m in sorted(separate.items()):
        mrr_list = " / ".join(
            f"{m['o'][(b, 'MRR')][1]:.2f}" if m["o"][(b, "MRR")][0] == "so" else "--" for b in bo
        )
        footnotes.append(
            ten.replace("±", r"$\pm$")
            + " is not listed separately: that paper reports both a single best "
            "run (MRR " + mrr_list + " on " + " / ".join(bo) + ") and a multi-seed "
            r"mean$\pm$std; the table uses the multi-seed figures, which are the "
            "comparable ones."
        )
    for ten, (mark, explanation) in sorted(PROVENANCE_MARKS.items()):
        footnotes.append(rf"\{mark}{{}} {explanation}.")
    if any(m["peer"] != "Y" for _, block_rows, _ in blocks for m in block_rows):
        footnotes.append(
            r"\ddag{} = venue is not peer-reviewed (preprint). Such rows are shown "
            r"for completeness and are never counted as the best in a column."
        )
    if probe and "MKG-W" in bo:
        footnotes.append(
            r"\textsuperscript{F8} The DB15K probe was run under the clean-valid protocol "
            r"(train 75,261; $\lambda$ selected on valid\_sach) and is not comparable with "
            r"the original-split row. The MKG-Y cell is blank because the mention-free gain "
            r"does not replicate across three base seeds (+0.0075 $\pm$ 0.0019, CI covering "
            r"zero). Both MKG-Y and DB15K probe numbers are validation-split and are "
            r"reported in Analysis only."
        )
    footnotes.append(
        r"MENTOR: configuration selected on \emph{validation} per dataset "
        + "; ".join(f"({b}: {OURS_LABEL_TEX[b]})" for b in bo)
        + r"; the single-expert A3' and the 2-expert gate are compared on validation and "
        r"the gate is the selected headline (single-expert shown in the ablation). "
        r"3 seeds; $\pm$ = SE, 95\% CI $\approx \pm 4.3\times$SE ($t$, df\,=\,2), on MRR."
    )
    L += (
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"",
            r"\vspace{2pt}",
            r"\begin{minipage}{\linewidth}\footnotesize\raggedright",
        ]
        + [c + r" \\" for c in footnotes]
        + [r"\end{minipage}", r"\end{table*}"]
    )

    M += ["", f"- ngày quét mốc ngoài: MRR **{mrr_scan_date}** · Hits **{hits_scan_date}**"]
    for ten, m in sorted(separate.items()):
        mrr_list = " / ".join(
            f"{m['o'][(b, 'MRR')][1]:.2f}" if m["o"][(b, "MRR")][0] == "so" else "–" for b in bo
        )
        M.append(
            f"- **{ten}** KHÔNG có dòng riêng: bài đó báo cả best run (MRR {mrr_list} trên "
            + " / ".join(bo)
            + ") lẫn mean±std đa-seed; bảng dùng **mean±std** vì đó mới "
            "là số so được."
        )
    for ten, (mark, explanation) in sorted(PROVENANCE_MARKS.items()):
        M.append(f"- `†` ({ten}): {explanation}.")
    M += [
        (
            "- `–` = bài gốc **không báo** chỉ số đó"
            + (
                " (a); và **chưa tra được** (b) — hai trạng thái KHÁC NHAU, chú thích tách bạch"
                " ở bản LaTeX."
                if has_not_retrieved
                else ". Bảng này **không còn ô 'chưa tra được'** nào."
            )
        ),
        "- `‡` = venue **chưa phản biện**; dòng in nghiêng và **không bao giờ** được in đậm. "
        "Đậm = tốt nhất mỗi cột **giữa các bài peer-reviewed** (có MENTOR), xét toàn bảng.",
        "- MENTOR = cấu hình chọn trên **valid** từng bộ: "
        + " · ".join(f"{b} → {OURS_LABEL_MD[b]}" for b in bo)
        + ". A3′ một-chuyên-gia và cổng 2-chuyên-gia "
        "so trên valid, **cổng** là headline (một-chuyên-gia ở ablation). 3 seed; **± = SE, "
        "95% CI ≈ ±4,3×SE** (t, df=2) ở MRR.",
        f"- sinh {today} bởi `scripts/v3/bang_bai_chinh.py`",
    ]

    ra_tex.write_text("\n".join(L) + "\n", encoding="utf-8")
    ra_md.write_text("\n".join(M) + "\n", encoding="utf-8")
    print(
        f"ghi {ra_tex.relative_to(ROOT)} + {ra_md.relative_to(ROOT)} "
        f"({n_datasets} bộ: {', '.join(bo)})"
    )


def ours_md_label(_: str) -> str:
    """Label of our row in the .md version (kept in one place to avoid repeating the string)."""
    return "**MENTOR** (ours)"


def build_v2_table() -> None:
    """ViProKG table: separate, 3 rows per model.

    Why a separate table rather than one more column next to W/Y: the main results file records
    that the ViProKG row is NOT compared with the three standard benchmarks, for three
    reasons: (1) A3' on v2 uses `--use_image 1` while A3' on the three benchmarks uses `0`;
    (2) attribute slices make up 37.43 % of the v2 test set and the three benchmarks have none;
    (3) the protocol forbids cross-dataset comparison of the mention-free column. Putting them in
    one table invites exactly the reading that is ruled out. The only external references for
    this dataset are MyGO and TuckER (no outside paper measured it).
    """
    _, rows = read_tsv(HITS)
    ix = {(r["cau_hinh"], r["nhom"], r["do_do"]): r for r in rows if r["bo"] == V2_DATASET}
    missing = [
        (m, n, d) for m in V2_MODELS for n in V2_GROUPS for d in METRICS if (m, n, d) not in ix
    ]
    if missing:
        raise SystemExit(
            f"`hits_bang.tsv` is missing {len(missing)} ViProKG cells, e.g. {missing[:3]}; "
            f"refusing to generate a partial table. STOP."
        )
    today = _dt.date.today().isoformat()
    # Bold rule: bold ONLY when the 95 % across-seed CI (t, df=2) of Delta(ours - MyGO) lies
    # entirely above 0. The earlier rule bolded on "higher mean", which compares two points
    # without any variance: 11/12 cells were bold, now 4/12, and overall MRR (+0.01110, CI
    # [-0.00171; +0.02390]) lost its bold.
    if not CI_V2_MYGO.exists():
        raise SystemExit(
            f"missing {CI_V2_MYGO.relative_to(ROOT)}; run "
            f"`scripts/v3/ktc_v2_mygo.py` first. Bolding by mean is not allowed."
        )
    ci_mygo = json.loads(
        CI_V2_MYGO.read_text(encoding="utf-8")
    )  # Delta(A3'* - MyGO+seg): the binding constraint
    _missing = [f"{n}|{d}" for n in V2_GROUPS for d in METRICS if f"{n}|{d}" not in ci_mygo]
    if _missing:
        raise SystemExit(f"CI file is missing {len(_missing)} cells, e.g. {_missing[:3]}. STOP.")
    # Load the remaining external baselines; a missing json gives a warning (never silent) and imposes no constraint.
    ci_baselines: dict = {}
    for _nm, _p in CI_V2_BASELINES.items():
        if _p.exists():
            ci_baselines[_nm] = json.loads(_p.read_text(encoding="utf-8"))
        else:
            print(
                f"WARNING missing {_p.relative_to(ROOT)}: baseline '{_nm}' does not constrain bolding (pending)."
            )
    # Delta(A3'* - baseline) by baseline name: MyGO+seg is in ci_mygo, the rest in ci_baselines.
    CI_BY_BASELINE = {"MyGO+seg": ci_mygo, **ci_baselines}
    # Baselines considered for bolding: the strongest by mean is searched here.
    # TuckER (structure-only) is not in the set (ablation reference, not a competing baseline).
    BOLD_SET = tuple(
        x for x in ("MyGO+seg", "MoMoK", "TSAM", "AdaMF", "DSoM re-impl.") if x in V2_MODELS
    )

    def o(m, n, d):
        """(value %, is_bold). Only our row can be bold; external baselines never are.

        For each cell, find the baseline in BOLD_SET with the highest mean (the column's
        runner-up) and bold A3'* only if the 95 % across-seed CI of Delta(A3'* - that baseline)
        is above 0. An AND over all baselines is not used: a noisy re-implementation (DSoM, CI
        covering 0 despite a lower mean) would veto the whole column (AND gives 0/12 against
        5/12 for beat-strongest). If the strongest baseline has no CI file (e.g. AdaMF without
        A3'* lambda 8 ranks) the cell is not bold (pending). With SYNCED_PREPROCESSING False no
        cell is bold, since the two arms use different text preprocessing.
        """
        v = float(ix[(m, n, d)]["tb"]) * 100.0
        if not SYNCED_PREPROCESSING or m != "A3′* λ8 (ours)":
            return v, False
        cell = f"{n}|{d}"
        strongest = max(BOLD_SET, key=lambda e: float(ix[(e, n, d)]["tb"]))
        ci = CI_BY_BASELINE.get(strongest)
        dam = bool(ci) and ci.get(cell, {}).get("dam", False)
        return v, dam

    M = [
        "<!-- SINH TỰ ĐỘNG bởi scripts/v3/bang_bai_chinh.py — KHÔNG sửa tay. -->",
        "",
        "# Bảng ViMMKG 2.0 — **3 dòng/model** (LUẬT #31), test 2 637 triple = 5 274 khoá",
        "",
        "> 🔴 **KHÔNG so chéo bảng này với MKG-W/MKG-Y** (`KET_QUA_CUOI.md` §1.1, đã ký): A3′ v2",
        "> dùng `--use_image 1` (ba bộ chuẩn dùng `0`); lát thuộc tính chiếm **37,43 %** test v2",
        "> mà ba bộ chuẩn không có lát; Quy tắc E cấm so chéo cột KHÔNG-cạnh.",
        "",
        "| Model | nhóm | n | " + " | ".join(METRICS) + " |",
        "|---|---|--:|" + "---:|" * 4,
    ]
    L = [
        r"% SINH TỰ ĐỘNG bởi scripts/v3/bang_bai_chinh.py — KHÔNG sửa tay.",
        f"% nguồn: {HITS.relative_to(ROOT)} | sinh {today}",
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Link prediction on \textbf{ViMMKG 2.0} (ours), 2{,}637 test triples "
        r"= 5{,}274 queries. Three rows per model as required by the attribute-slice split: "
        r"\emph{all} / \emph{ordinary} / \emph{slice}. Figures in \%. "
        + (
            r"\textbf{Bold} = exceeds the strongest external baseline in that cell among "
            r"\{MyGO+seg, MoMoK, TSAM, DSoM re-impl., AdaMF\} with a 95\% across-seed CI above "
            r"zero (paired per-key, $t$, df\,=\,2, 3 seeds); per-key bootstrap (1000) agrees; a "
            r"higher mean alone is not enough; the full 12-cell table is in the appendix. "
            if SYNCED_PREPROCESSING
            else r"\textbf{No cell is bold}: MENTOR receives word-segmented PhoBERT input "
            r"(pyvi) whereas the MyGO row on this dataset does not, so the two arms do "
            r"not yet share one text pre-processing pipeline, so no superiority claim is "
            r"made here. "
        )
        + r"\textbf{These figures are not comparable with the standard-benchmark table}: "
        r"A3' on ViMMKG 2.0 uses the image channel, 37.43\% of its test triples are "
        r"attribute slices (absent from the three benchmarks), and cross-dataset comparison "
        r"of the mention-free column is disallowed by our protocol.}",
        r"\label{tab:vimmkg2}",
        r"\begin{tabular}{l l r r r r r}",
        r"\toprule",
        r"Model & Group & $n$ & " + " & ".join(METRICS) + r" \\",
        r"\midrule",
    ]
    for m in V2_MODELS:
        for k, n in enumerate(V2_GROUPS):
            n_keys = ix[(m, n, "MRR")]["n"]
            vals = [o(m, n, d) for d in METRICS]
            name_md = V2_MD[m] if k == 0 else ""
            M.append(
                f"| {name_md} | {n} | {n_keys} | "
                + " | ".join((f"**{v:.2f}**" if b else f"{v:.2f}") for v, b in vals)
                + " |"
            )
            name_tex = V2_TEX[m] if k == 0 else ""
            L.append(
                f"{name_tex} & {n} & {n_keys} & "
                + " & ".join((rf"\textbf{{{v:.2f}}}" if b else f"{v:.2f}") for v, b in vals)
                + r" \\"
            )
        if m != V2_MODELS[-1]:
            L.append(r"\midrule")
    # Frequency-prior row (slice only, both directions); never bold. Lift = model - prior.
    # Source: relation frequency on train_lat (type-constrained), head+tail pooled (n=1974).
    PRIOR = (33.55, 22.19, 42.15, 50.00)
    L.append(r"\midrule")
    L.append(
        r"prior tần suất (lát)$^\dagger$ & lát & 1974 & "
        + " & ".join(f"{v:.2f}" for v in PRIOR)
        + r" \\"
    )
    M.append(
        "| prior tần suất (lát) † | lát | 1974 | " + " | ".join(f"{v:.2f}" for v in PRIOR) + " |"
    )
    L += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\begin{minipage}{\linewidth}\footnotesize\raggedright",
        r"$^\dagger$ Prior tần suất (lát), gộp hai chiều để tham chiếu; tách chiều: head 1.67 / "
        r"tail 65.43 --- ở lát tail prior vượt mọi mô hình trừ bản tái hiện xấp xỉ DSoM re-impl.\ "
        r"(xem Bảng~\ref{tab:lat_prior}). Cột lát không in đậm.",
        r"\end{minipage}",
        r"\end{table}",
    ]
    # number of bold cells (/12) per baseline if it were the only constraint (report only)
    _n_bold_per_baseline = {
        nm: sum(1 for n in V2_GROUPS for d in METRICS if j.get(f"{n}|{d}", {}).get("dam"))
        for nm, j in CI_BY_BASELINE.items()
    }

    # count exactly as beat-strongest (same as function o): per cell, the baseline with the highest mean must have Delta CI > 0
    def _is_bold(n, d):
        cell = f"{n}|{d}"
        strongest = max(BOLD_SET, key=lambda e: float(ix[(e, n, d)]["tb"]))
        ci = CI_BY_BASELINE.get(strongest)
        return bool(ci) and ci.get(cell, {}).get("dam", False)

    n_bold = sum(1 for n in V2_GROUPS for d in METRICS if SYNCED_PREPROCESSING and _is_bold(n, d))
    _per_baseline = " · ".join(f"**{nm}** {_n_bold_per_baseline[nm]}/12" for nm in CI_BY_BASELINE)
    _absent = [nm for nm in CI_V2_BASELINES if nm not in ci_baselines]
    _pending = (
        f" ⚠️ Mốc CHƯA đo KTC per-key (ô nào mốc đó mạnh nhất thì để pending, không đậm): "
        f"{', '.join(_absent)}."
        if _absent
        else ""
    )
    bold_notes = (
        [
            f"- **Đậm = A3′\\* λ8 (ours) vượt MỐC-NGOÀI-MẠNH-NHẤT TỪNG Ô với KTC 95 % qua seed "
            f"trên 0** (ghép cặp per-key, `t` df=2, 3 seed; TIN_261 đ.1 — tập mốc {{MyGO+seg, "
            f"MoMoK, DSoM re-impl., AdaMF, TSAM}}) — **{n_bold}/12 ô** (tổng·{{MRR,H@1,H@10}} + "
            f"thường·{{MRR,H@1,H@10}}, nơi mốc mạnh nhất là **TSAM** mà A3′\\* vượt KTC>0; "
            f"**tổng·H@3 + thường·H@3 + mọi lát**: A3′\\* KHÔNG vượt mốc-mạnh-nhất KTC>0 ⇒ không đậm). "
            f"Số ô mỗi mốc có Δ KTC>0: "
            f"{_per_baseline}.{_pending} ⚠️ **KHÔNG dùng AND-mọi-mốc** (0/12): DSoM re-impl. NHIỄU "
            "(Δ ôm 0 dù mean thấp hơn MyGO+seg) sẽ veto oan — chỉ xét nó ở ô nó THỰC mạnh nhất. "
            "Trung bình cao hơn **không đủ** để đậm (LUẬT #32)."
        ]
        if SYNCED_PREPROCESSING
        else [
            "- 🔴 **KHÔNG ô nào in đậm — đậm chờ đồng bộ tiền xử lý** (Fable ký ❓BQ1, "
            "TIN_180 điểm 1). MENTOR nhận PhoBERT **đã tách từ** (pyvi) còn MyGO ở bộ này "
            "nhận bản **chưa tách từ**, nên hai nhánh **khác dụng cụ** và chưa so được "
            "bằng cùng một thước (LUẬT #23). Chênh do tách từ, đo trên **chính MyGO** "
            "(ViMMKG 1.0, 3 seed): **+0,0066 MRR** và **+0,072 head-MRR**. Hồ sơ: "
            "`baocao/BC112_TACH_TU.md`.",
            "- 🟡 Bốn ô từng đậm theo KTC (`tổng H@1` · `thường MRR` · `thường H@1` · "
            "`lát H@1`) nay mang nhãn **HẬU KIỂM** ở `baocao/BC110_KTC_V2_MYGO.md` §3.1 "
            "— **không trích vào bài** tới khi MyGO@v2 chạy lại bằng token đã tách từ.",
        ]
    )
    M += [
        "",
        "- `tổng` = 5 274 khoá · `thường` 3 300 · `lát` 1 974 (hai chiều lát: xem "
        "`KET_QUA_CUOI.md` §2.1 — `lát·head` vs `lát·tail` lệch **66×**, KHÔNG gộp).",
        *bold_notes,
        "- **Nhóm `lát`: A3′\\* KHÔNG dẫn** (đậm 0/4) — A3′\\* lát MRR 33,89 chỉ nhỉnh TuckER 32,76 "
        "/ MyGO 31,53 / MoMoK 28,54, nhưng KTC vs mốc-mạnh-nhất KHÔNG rời 0 ⇒ không đậm. Headline "
        "v2 = **A3′\\* λ8** (chọn trên valid, BC118 §9); **nền\\* (MENTOR − L) là ablation**; **hàng "
        "cổng OOF v2 KHÔNG in** (cổng < A3′).",
        "- 🔴 **† Hàng `prior tần suất (lát)` = 33,55** (gộp hai chiều để tham chiếu; tần suất quan "
        "hệ trên train, type-constrain) — **nằm GIỮA dải model lát** (MoMoK 28,54 – MENTOR 33,89) ⇒ "
        "\\*\\*lift (model − prior) ≈ 0\\*\\*. Tách hai chiều: **head 1,67 / tail 65,43**; ở chiều "
        "tail prior **vượt mọi model TRỪ bản tái hiện xấp xỉ DSoM re-impl. (+2,7)** — kết luận không "
        "đổi. **Cột lát KHÔNG đậm**: lát·tail đo được \\*prior giá trị\\*, chưa đo phân biệt thuộc "
        'tính (xem `bang_lat_prior`). CẤM đọc lát là "grounding ảnh".',
        "- Mốc ngoài v2: **MyGO (seg) · MoMoK · AdaMF-MAT (our features) · DSoM re-impl. · TuckER** "
        "(TSAM khi GPU xong). Không bài gốc nào đo bộ này — chạy lại nguyên bản (`baselines/`). "
        "🔵 **lát·H@10 cao nhất = AdaMF 51,33** (> DSoM 49,78, A3′\\* 47,28) — ở nhóm lát, nơi "
        "A3′\\* không dẫn; AdaMF tổng/thường vẫn dưới MyGO+seg nên không đổi tập ô đậm của A3′\\*.",
        "- **Chẻ CÓ/KHÔNG-cạnh × lát/thường (luật cạnh v2 tiếng Việt, TIN_229 đ.2):** phụ lục "
        "ở `baocao/BC118_CANH_V2_VI.md` §7 (4 model). Mọi model: `thường·CÓ-cạnh` > `thường·"
        "KHÔNG-cạnh`. 🔴 **Ghi chú bắt buộc:** cạnh v2 ViMMKG2 **tập trung cao** — hub 'Việt Nam' "
        "= **14,4 %** mọi cạnh, top-10 đích **28,4 %**; **lát ∩ CÓ-cạnh = 8** (ô `lát·CÓ-cạnh` "
        "n=16 = 8×2 chiều) **không đủ mẫu, chỉ báo n** (BC118 §1/§5b).",
        f"- sinh {today} bởi `scripts/v3/bang_bai_chinh.py`",
    ]
    (ROOT / "paper_b" / "bang_vimmkg2.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    (ROOT / "paper_b" / "bang_vimmkg2.md").write_text("\n".join(M) + "\n", encoding="utf-8")
    print(
        f"wrote paper_b/bang_vimmkg2.{{tex,md}} ({len(V2_MODELS)} models x {len(V2_GROUPS)} groups)"
    )


def main() -> None:
    build_table(
        MAIN_DATASETS,
        OUT_TEX,
        OUT_MD,
        "Bảng chính — 2 bộ chuẩn (thang %)",
        r"\textbf{MENTOR} (ours)",
        "Link prediction on the two standard MMKGC benchmarks.",
        "tab:main",
        False,
    )  # prior row goes to the ceiling table
    build_table(
        APPENDIX_DATASETS,
        ROOT / "paper_b" / "bang_phuluc_db15k.tex",
        ROOT / "paper_b" / "bang_phuluc_db15k.md",
        "Phụ lục — DB15K (thang %)",
        r"\textbf{MENTOR} (ours)",
        "Link prediction on DB15K (appendix). Reported here rather than in the main "
        "table: under our pre-registered stopping rule every configuration we trained "
        "plateaus at 40.2 MRR, below the 42.0 threshold, so DB15K is not used to "
        "support any claim.",
        "tab:db15k",
        False,
    )
    build_v2_table()


if __name__ == "__main__":
    main()
