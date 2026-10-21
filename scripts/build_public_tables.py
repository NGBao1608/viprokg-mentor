"""Build the public, English-only LaTeX tables in ``tables/`` from the frozen tables.

Each frozen table (``tables_frozen/*.tex``) is copied with: LaTeX comment lines removed (they carry internal
provenance), Vietnamese captions/labels replaced by English text following the project glossary, internal
report labels and local paths dropped. Numbers are never edited; ``scripts/diff_numbers_only.py`` checks that
the digit stream is unchanged.

Line edits are keyed by 1-based line number of the frozen file and guarded by an anchor substring, so the script
fails loudly if a frozen table changes.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tables_frozen"
OUT = ROOT / "tables"

# name -> list of (first_line, last_line, anchor, replacement lines); applied bottom-up.
EDITS: dict[str, list[tuple[int, int, str, list[str]]]] = {
    "bang_vimmkg2": [
        (
            49,
            49,
            "Prior tần suất",
            [
                r"$^\dagger$ Frequency prior (attribute triples), both directions pooled for reference; split by "
                r"direction: head 1.67 / tail 65.43 --- on the attribute tail the prior exceeds every model except "
                r"the approximate re-implementation DSoM re-impl.\ (see Table~\ref{tab:lat_prior}). The attribute "
                r"column is not bolded."
            ],
        ),
    ],
    "bang_trung_tam": [
        (
            21,
            21,
            "base = TuckER",
            [
                r"base = TuckER + \textbf{text-only} (W/Y/DB15K, use\_image 0) or text+image (v2); the alignment "
                r"loss is included in base. The largest step is +L (the lexical mention channel). $\Delta_L$ "
                r"concentrates on T\_overlap (+6.8 to +17.1) but is $\approx$0 or negative on T\_free for W/Y/DB. "
                r"\textbf{ViProKG:} $\Delta_L$ is weak (+0.94/+0.09, DiD +0.85); the across-seed CI "
                r"\textbf{covers 0} and the key bootstrap is at the boundary, differences $\le\sim$0.3 --- we do "
                r"not claim that the L channel transfers to v2. +gate adds little (W +0.35/Y +0.74/DB +0.61); on "
                r"v2 +gate $<$ A3$'$, so the v2 headline is A3$'$* (no gate). $^\ddagger$DB15K = clean-valid; the "
                r"in-path $\Delta$ is valid and is not compared with the original-split appendix (same "
                r"instrument). TuckER for DB15K is in the uni-modal block (not re-run)."
            ],
        ),
        (
            11,
            11,
            "Bộ & TuckER",
            [
                r"Dataset & TuckER & base & +L & +gate & $\Delta_L$ T\_overlap & $\Delta_L$ T\_free & DiD$_L$ \\"
            ],
        ),
        (
            7,
            7,
            r"\caption{Bảng ablation",
            [
                r"\caption{Central ablation: overall MRR (\%) per fusion step (TuckER $\to$ base $\to$ +L $\to$ "
                r"+gate), and \textbf{the gain of the +L step split by mention-overlap group} ($\Delta_L$ = "
                r"MENTOR\_base+L$-$MENTOR\_base, paired per key, 3 seeds): $\Delta_L$ concentrates on the \emph{mention-overlap} "
                r"split (T\_overlap) and is near 0 or negative on the \emph{mention-free} split (T\_free). "
                r"DiD$_L$ = $\Delta_L^{\text{overlap}}-\Delta_L^{\text{free}}$. Across-seed CI ($t$, df=2).}"
            ],
        ),
    ],
    "bang_canh_v2": [
        (
            25,
            25,
            "Mức CÓ/KHÔNG gộp",
            [
                r"Overlap/free levels pool both directions (attribute + relational triples). "
                r"\textbf{MENTOR\_base+L} has the largest gap$-$gap$_{\text{TuckER}}$ (+4.76); most of this gap is "
                r"already present in \emph{MENTOR\_base*} (+3.92, text/image without L) and in TSAM (+3.96), so "
                r"the +L step adds only $\sim$+0.84 (consistent with the weak v2 $\Delta_L$, "
                r"Table~\ref{tab:trung_tam}). The weak baselines MyGO/MoMoK/AdaMF/DSoM are $\approx$ or $\le$ "
                r"TuckER. The mention-free column spans $\sim$9.4 points. Mention edges v2 (file hash "
                r"\texttt{cefc05167b7123d3})."
            ],
        ),
        (
            14,
            14,
            "nền*",
            [r"MENTOR\_base* ($\lambda$=8, ablation) & 49.65 & 37.84 & +11.81 & +3.92 \\"],
        ),
        (
            13,
            13,
            r"\textbf{MENTOR}",
            [
                r"\textbf{MENTOR\_base+L*} ($\lambda$=8, headline) & 50.59 & 37.93 & +12.65 & \textbf{+4.76} \\"
            ],
        ),
        (
            11,
            11,
            "Mô hình",
            [r"Model & mention-overlap & mention-free & gap & gap$-$gap$_{\text{TuckER}}$ \\"],
        ),
        (
            7,
            7,
            r"\caption{Mức MRR",
            [
                r"\caption{MRR level (\%, 2 directions = keys) by \textbf{mention-overlap / mention-free} "
                r"split on ViProKG, 8 models. \emph{gap} = overlap $-$ free (the model's own mention gap); "
                r"\emph{gap$-$gap$_{\text{TuckER}}$} = the part beyond the \textbf{hub difficulty of the structural "
                r"base} (TuckER gap = +7.89). \textbf{MENTOR\_base+L} has the largest DiD (+4.76); strong text "
                r"models (TSAM, MENTOR\_base*) also exceed TuckER ($\sim$+3.9), while MyGO/MoMoK/AdaMF/DSoM are "
                r"$\approx$ or $\le$ TuckER. The mention-free column (T\_free) spans $\sim$9.4 points across "
                r"models.}"
            ],
        ),
    ],
    "bang_e_che": [
        (
            28,
            35,
            "Nguồn: BC123b-2",
            [
                r"Source: answer-name masking runs (W/Y/DB + ViProKG round 2, correct instrument, "
                r"bit-equal to \texttt{che2\_kq.json}).",
                r"$^{\S}$The \emph{random} column is the control of \textbf{RANDOM masking} with the same "
                r"\%tokens per description that \emph{does not touch name tokens} (3 seeds). Result: random drop "
                r"$<$ name-mask drop on the three standard sets, and the \textbf{random overlap/free ratio "
                r"$\approx$1} (W 1.28 / Y 0.99 / DB 0.93) --- random masking does \emph{not} concentrate on "
                r"overlap as name masking does (6--15$\times$), so the concentration on overlap belongs to the "
                r"\emph{name channel}.",
                r"On ViProKG (\%token 23.2) the random drop 23.22 $>$ name-mask 10.78 and random mention-free "
                r"22.96 $\geq$ 50\% of 8.46, so the ratio 2.06$\times$ is read as \emph{context loss} (pct\_L "
                r"99.35, L correct).",
                r"(i)--(iii) are three pre-specified checks (recorded BEFORE running); results: (i) random drop "
                r"$<$ name-mask \textbf{3/4} (fails on v2) · (ii) random ratio $\in$[0.7;1.5] \textbf{3/3} · "
                r"(iii) v2 random mention-free $\geq$50\% \textbf{holds}. AdaMF-MAT ``\,--\,'' = not applicable: "
                r"its text is a learned parameter and the checkpoint restores it on load. ``\,--\,'' does "
                r"\emph{not} mean ``no drop''.",
            ],
        ),
        (
            16,
            16,
            "Bộ & che tên",
            [
                r"Dataset & name mask & \emph{random}$^{\S}$ & overlap & free & name mask & \emph{random} & "
                r"\%masked tokens & threshold & pass \\"
            ],
        ),
        (
            14,
            14,
            "rớt tổng",
            [
                r" & \multicolumn{2}{c}{overall drop} & & & \multicolumn{2}{c}{overlap/free ratio} & & & \\"
            ],
        ),
        (
            6,
            10,
            r"\caption{Rớt MRR",
            [
                r"\caption{MRR drop (points \%) under \textbf{answer-name masking} on the headline MENTOR "
                r"configuration (3 seeds, TEST). Masking concentrates on the \emph{mention-overlap} split on the "
                r"three standard sets (ratio 6{,}3--14{,}9$\times$, meeting the pre-specified threshold); on "
                r"ViProKG masking still concentrates on overlap (17{,}43 $>$ 8{,}46) but the ratio "
                r"2{,}06$\times$ is below the threshold, because the high share of masked tokens (23{,}2) also "
                r"removes context. AdaMF-MAT is not applicable (its text is a learned parameter, overwritten by "
                r"the checkpoint).}"
            ],
        ),
    ],
    "bang_lat_prior": [
        (
            32,
            34,
            "Khai trước",
            [
                r"Pre-specified (recorded before measuring): prior tail $\geq$ 45 $\Rightarrow$ pass (65.43).",
                r"All rows are filled from 3-seed ranks (no pending cells). Prior from",
                r"\texttt{train\_lat.txt}; models from \texttt{ranks\_test.tsv} filtered by "
                r"\texttt{test\_lat.txt}, 3 seeds.",
            ],
        ),
        (
            7,
            12,
            r"\caption{Lát thuộc tính",
            [
                r"\caption{ViProKG attribute triples (987 attribute triples, 2 directions): "
                r"\textbf{frequency prior} (ranking by relation frequency on train, type-constrained) versus "
                r"models; \emph{lift} = model $-$ prior (MRR, points). \textbf{On the attribute tail, every model "
                r"EXCEPT the approximate DSoM re-implementation fails to beat the prior} (lift $-$1.6 to $-$9.5); "
                r"the only exception is DSoM re-impl.\ (+2.71, an approximate version that does not match the "
                r"published figures). The ``high'' attribute-tail score is due to a frequency prior that already "
                r"solves most of it (48 values, 9 relations), not to the models discriminating. Attribute head "
                r"is close to random for both models and prior.}"
            ],
        ),
    ],
    "bang_anh": [
        (
            33,
            37,
            "Headline W/Y/DB15K",
            [
                r"Headline W/Y/DB15K is \textbf{text-only} (\texttt{use\_image}=0); the ViProKG headline uses",
                r"text+image. $\pm$ = SD across seeds (run-to-run noise $\sim$0.3--0.65). Image permutation: rows",
                r"of the image matrix are shuffled across entities with \texttt{blake2b} (seed per train).",
                r"$^{\dagger}$ ViProKG ($\lambda$=8, 3 seeds, same run): images are permuted only among the "
                r"7{,}790 entities that have an image, the mask is unchanged; $\pm$ = SD across seeds. "
                r"$\Delta$(real$-$permuted) overall +0.17 (across-seed CI $t$, df\,=\,2 [$-$0.92; +1.26]; key "
                r"bootstrap [$-$0.12; +0.46]): \emph{no signal detected} from the image content (the "
                r"across-seed CI is wider than the 0.65 threshold, so this is not read as ``equivalent''). The "
                r"``off'' cell of v2: 3 runs with \texttt{use\_image}=0 at $\lambda$=8 (same configuration); "
                r"$\Delta$(real$-$off) overall +0.10 (across-seed CI [$-$0.86; +1.05]; key bootstrap "
                r"[$-$0.18; +0.37]), neutrality criterion $|\Delta|\le$0.65: PASS. The existing on/off control "
                r"was measured at $\lambda_{\text{align}}$=0 and gave $\Delta$ overall $-$0.02 [$-$0.57; +0.52].",
            ],
        ),
        (
            15,
            15,
            "Bộ & Nhóm",
            [
                r"Dataset & Group & off & real & permuted & $\Delta$(real$-$off) & "
                r"$\Delta$(real$-$permuted) \\"
            ],
        ),
        (
            7,
            11,
            r"\caption{Ablation kênh ảnh",
            [
                r"\caption{Image-channel ablation (MRR \%, 3 seeds; $\pm$ = SD across seeds): off "
                r"(\texttt{use\_image}=0) / real images / \emph{permuted} images (images shuffled across "
                r"entities, stable hash) $\times$ \{overall, mention-overlap, mention-free\}. "
                r"\textbf{The image channel is neutral on the headline} ($|\Delta$ overall$|\le$0.65 on both "
                r"sets) and \textbf{image content adds no signal} (real $\approx$ permuted). On MKG-W real images "
                r"raise mention-overlap by +1.41 but lower mention-free by $-$0.67 (cancelling in the total) --- "
                r"consistent with the main finding. v2: same-run permutation control and images-off runs, 3 seeds ($\lambda$=8).}"
            ],
        ),
    ],
    "bang_e_muc": [
        (
            23,
            27,
            "Giá trị thêm",
            [
                r"Additional values (mention-free \%): TSAM W~38.21 / Y~45.76 (1 seed) / v2~41.47 (3 seeds, "
                r"relational$\cdot$free); DSoM re-impl.\ W~34.68 / Y~43.72 (1 seed) / v2~38.91 (3 seeds, "
                r"relational$\cdot$free); AdaMF v2~34.22",
                r"(3 seeds). v2 uses relational$\cdot$free consistently; every value lies in [32.45; 41.85], so "
                r"the v2 min--max is unchanged, but TSAM~41.47 is close to ours~41.85 (we lead v2 mention-free "
                r"only narrowly). MKG-W DSoM has a degenerate policy (read with caveat). Cross-pipeline, no "
                r"common-key CI yet. DB15K: common evaluation (original test), training split differs between "
                r"models (labelled per cell).",
            ],
        ),
        (
            12,
            12,
            "Bộ & min",
            [
                r"Dataset & min--max mention-free (model) & ours (MENTOR) & range & within 1{,}5 points? \\"
            ],
        ),
        (
            6,
            8,
            r"\caption{(e) MỨC",
            [
                r"\caption{(e) LEVEL --- min--max mention-free MRR (\%) across \{ours, MyGO, AdaMF, MoMoK, TSAM,",
                r"DSoM re-impl.\} and our position. The ``level'' column reports only the \emph{sign}: we do NOT "
                r"lead mention-free on",
                r"MKG-W/MKG-Y (below AdaMF/TSAM) and lead on DB15K and ViProKG.}",
            ],
        ),
    ],
    "bang_prior_k3": [
        (
            24,
            27,
            "Nguồn:",
            [
                r"Source: archived OOF-gate prior run (\texttt{kq\_prior.json}).",
                r"$^\dagger$ MKG-Y gate-2 46.90 is a new run within this experiment and differs from the headline "
                r"44.80 (different instrument); used only for the in-run $\Delta$.",
                r"\textbf{DO NOT} extend the claim ``the prior rescues the mention-free column'' beyond DB15K "
                r"clean-valid.",
            ],
        ),
        (
            14,
            14,
            "Bộ & $(\\alpha",
            [
                r"Dataset & $(\alpha,\mu)^*$ & gate-2 & gate-3 & $\Delta$ overall [CI] & $\Delta$ mention-free "
                r"[CI] & pick \\"
            ],
        ),
        (
            6,
            10,
            r"\caption{\textbf{[KHÁM PHÁ",
            [
                r"\caption{\textbf{[EXPLORATORY · Analysis only]} OOF gate with three experts "
                r"\{MENTOR\_base (S), MENTOR\_base+L (L), relation prior\} versus the two-expert gate "
                r"\{MENTOR\_base, MENTOR\_base+L\} (test, 3 seeds, $t$ df\,=\,2). "
                r"prior~=~$\text{base}(S) + \mu\log P(e\mid r,\text{dir})$; $(\alpha,\mu)$ chosen on "
                r"valid. Read $\Delta$ \emph{only within the same run}. DB15K = clean-valid (train 75\,261). "
                r"Only DB15K has a positive $\Delta$ with CI above 0; MKG-W/Y cover 0.}"
            ],
        ),
    ],
    "bang_delta_L": [
        (
            41,
            45,
            "DiD(L) dương",
            [
                r"\textbf{DiD(L) is positive on 4/4 datasets} (overlap $>$ free consistently), but the magnitude "
                r"on W/Y/DB (+7.7 to +18.2)",
                r"is much larger than on v2 (+0.85). Reference \emph{hub difficulty}: structure-only TuckER has "
                r"overlap$-$free = \textbf{+7.89}",
                r"(overlap 42.65 $-$ free 34.76, 3 seeds); A3$'$* $\lambda$8 has +12.65 $\Rightarrow$ the text+L "
                r"channel adds",
                r"\textbf{+4.76} points to the overlap/mention-free gap beyond the structural base level. The "
                r"key bootstrap is",
                r"tighter than the across-seed CI and gives the same conclusion --- the two measures (seed, key) "
                r"are not mixed.",
            ],
        ),
        (
            35,
            35,
            "DiD(L)",
            [
                r" & DiD(L) & \textbf{+0.85} & \multicolumn{2}{l}{overlap $>$ free (weak L on v2)} \\"
            ],
        ),
        (
            20,
            20,
            "DiD(L)",
            [
                r" & DiD(L) & \textbf{+14.57} & \multicolumn{2}{l}{overlap $\gg$ free, CIs separate} \\"
            ],
        ),
        (
            15,
            15,
            "Bộ & Nhóm",
            [
                r"Dataset & Group & $\Delta$(MENTOR\_base+L$-$MENTOR\_base) & across-seed CI & key bootstrap \\"
            ],
        ),
        (
            7,
            11,
            r"\caption{Độ lợi kênh L",
            [
                r"\caption{Gain of the L channel measured as a paired $\Delta$ \emph{within the same run}: "
                r"$\Delta$(MENTOR\_base+L $-$ MENTOR\_base) per key (MRR, points \%), 3 seeds. Across-seed CI ($t$, df\,=\,2) "
                r"\emph{and} paired key bootstrap (1000 resamples). \textbf{The L channel gives +6.8 to +17.1 "
                r"points on the mention-overlap split but $-$0.5 to $-$1.1 on mention-free} (MKG-Y/DB15K "
                r"bootstrap entirely below 0): the gain concentrates on the mention split, "
                r"DiD(L)~=~$\Delta_{\text{overlap}}-\Delta_{\text{free}}$ = +7.7 to +18.2.}"
            ],
        ),
    ],
}

# substring replacements (applied to every table after the line edits), (old, new)
SUBS = [
    ("Bộ &", "Dataset &"),
    (r"CÓ $\gg$ KHÔNG, KTC tách", r"overlap $\gg$ free, CIs separate"),
    (r"CÓ $\gg$ KHÔNG", r"overlap $\gg$ free"),
    (" & tổng &", " & overall &"),
    (" & CÓ-cạnh &", " & mention-overlap &"),
    (" & KHÔNG-cạnh &", " & mention-free &"),
    (r"\textbf{MENTOR} (A3'+L, $\lambda$=8, ours)", r"\textbf{MENTOR\_base+L} ($\lambda$=8, ours)"),
    (r"MENTOR (A3$'$* $\lambda$8)", r"MENTOR\_base+L* ($\lambda$8)"),
    (r"PRIOR tần suất", r"FREQUENCY prior"),
    (r"prior tần suất (lát)$^\dagger$", r"frequency prior (attribute)$^\dagger$"),
    ("TuckER (cấu trúc)", "TuckER (structure-only)"),
    ("(giữa)", "(middle)"),
    ("(gần đáy)", "(near bottom)"),
    ("(cao nhất)", "(highest)"),
    ("(ta)", "(ours)"),
    ("ta (MENTOR)", "ours (MENTOR)"),
    (r"\emph{ng.nhiên}", r"\emph{random}"),
    (" (BC125 / note \\#26d)", ""),
    (r"\,[F8]", r"$^\dagger$"),
    (r"\textbf{[F8]}", r"$^\dagger$"),
    (" (ext G1, MKG-W valid)", " (MKG-W valid)"),
    (" (BC121 §6 / BC120 §4)", ""),
    (", EN not locked (KHOA\\_CHU §14 / BC108 §3.2)", ""),
    (r"\texttt{V\_hoanvi} only rules out", r"The permuted-caption control only rules out"),
    (
        "Sources: BC92 Table 1 (base + prior (N), 3 base seeds); BC121 §12 (prior label).",
        "Sources: archived probe runs (base + prior (N), 3 base seeds).",
    ),
    ("A3$'$*", r"MENTOR\_base+L*"),
    ("A3$'$", r"MENTOR\_base+L"),
    ("A3'", r"MENTOR\_base+L"),
]
# vimmkg2 uses the table's own wording (all / ordinary / slice)
VIMMKG2_SUBS = [
    (" & overall &", " & all &"),
    (" & thường &", " & ordinary &"),
    (" & lát &", " & slice &"),
    (" & tổng &", " & all &"),
]
PUBLIC_NAME = re.compile(r"ViMMKG(?:~|\\ | )2\.0")
RE_WORDS = [
    (re.compile(r"\bA3\b"), ""),
]


def build(name: str) -> str:
    lines = (SRC / f"{name}.tex").read_text(encoding="utf-8").split("\n")
    for first, last, anchor, new in sorted(EDITS.get(name, []), key=lambda e: -e[0]):
        assert (
            anchor in lines[first - 1]
        ), f"{name}:{first} anchor {anchor!r} not found in {lines[first - 1][:80]!r}"
        lines[first - 1 : last] = new
    text = "\n".join(l for l in lines if not l.lstrip().startswith("%"))
    if name == "bang_vimmkg2":
        text = text.replace(" & tổng &", " & all &")
        for old, new in VIMMKG2_SUBS:
            text = text.replace(old, new)
    for old, new in SUBS:
        text = text.replace(old, new)
    # public dataset name (v1.3.0): display name only; tables_frozen/ and data paths keep ViMMKG2
    text = PUBLIC_NAME.sub("ViProKG", text).replace(r"\label{tab:vimmkg2}", r"\label{tab:viprokg}")
    return text


def main() -> int:
    OUT.mkdir(exist_ok=True)
    names = sorted(EDITS) + [
        n
        for n in ("bang_chinh", "bang_phuluc_db15k", "bang_ablation", "bang_tran_thong_tin")
        if n not in EDITS
    ]
    for name in names:
        (OUT / f"{name}.tex").write_text(build(name), encoding="utf-8")
        print("wrote tables/%s.tex" % name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
