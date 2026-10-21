# ViProKG and MENTOR — code and data release

This repository contains the code, data and frozen result tables for the paper
*"Where Do Multimodal Gains Come From? Auditing Lexical Dependence in Multimodal Knowledge Graph
Completion with ViProKG and MENTOR"*.

What the release supports:

* **ViProKG** — Vietnamese multimodal KG (Wikidata triples + Wikimedia Commons image metadata +
  Vietnamese text), with an evaluation protocol that separates *mention-overlap* from *mention-free*
  queries, relational from attribute triples, and the two directions of attribute queries.
  ViProKG contributes a benchmark with provenance-tracked images and the evaluation protocol above.
* **MENTOR** — TuckER-style model with text/image channels, a lexical mention channel (S_lex) and an
  out-of-fold gate. Variants: `MENTOR_base` (no lexical channel), `MENTOR_base+L`, `MENTOR` (full).
* **Finding reproduced by the tables** — the multimodal gain concentrates on the mention-overlap split;
  the mention-free split barely moves. We read this as *lexical dependence*, not generalization.

## Layout

```
scripts/tables/      table generators + regression gate (CPU only)
scripts/verify_tables.py   regenerate all tables and compare sha256 with tables_frozen/
src/                 model, training, evaluation, mention-edge rule v2, masking, OOF gate (GPU for training)
baselines/harness/   adapters for MoMoK, TSAM and our approximate DSoM re-implementation
baselines/{momok,tsam,dsom}/   per-seed rank files and run configs
data/experimental/v3/   canonical numbers (hits_bang.tsv, ktc_*.json), ranks_v2/, canh_v2/, ext/ViMMKG2/
data/final/ViMMKG2_core/manifest.tsv   per-image license log (8,029 rows; 239 excluded images)
tables_frozen/       the frozen .tex/.md tables (reference for sha256 comparison)
tables/              public English-only version of the 13 tables (built by scripts/build_public_tables.py)
notebooks/           executed Kaggle notebooks: 01 reproduction, 02 one training run, 03 image-permutation control on ViProKG
                     04 images-off branch on ViProKG at lambda 8 (completes the image table)
baselines/patches/   diffs applied to MoMoK and TSAM (upstream code is not redistributed)
data/wikipedia_attribution.tsv   Vietnamese Wikipedia page, URL and revision of every entity (CC BY-SA attribution)
MANIFEST.md          source file -> release file, role, tables
```

Note: internal file names keep the development name ViMMKG2; the public name of the dataset is ViProKG.

Large per-key rank files and tensors are kept out of git (see `.gitignore`) and distributed in the Zenodo record
of this release; `scripts/verify_tables.py` needs them, so run it on the full release archive.

Images are **not** shipped. `manifest.tsv` lists, per entity, the Commons file name, source page,
original URL, license short name, artist and sha256; excluded images carry a non-empty `ly_do_loai`
(license 195, content/spam 25, lost image 14, license+spam 5 = 239).

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy torch transformers wordfreq
```

Table generation needs only the Python standard library and NumPy.

## Reproduce the tables (CPU, no GPU)

```bash
python3 scripts/verify_tables.py
```

This runs the four generators, compares every generated table with `tables_frozen/` by sha256,
checks the two canonical number files, and runs the regression gate
`scripts/tables/check_frozen_tables.py` (exit code 0 means zero mismatches against the signed
headline values). Expected last line: `MISMATCHES: 0 []`.

| Table | Generator | Canonical output |
|---|---|---|
| Main results, MKG-W / MKG-Y | `scripts/tables/build_main_tables.py` | `paper_b/bang_chinh.tex` |
| ViProKG | `scripts/tables/build_main_tables.py` | `paper_b/bang_vimmkg2.tex` |
| Central ablation | `scripts/tables/build_central_ablation_table.py` | `paper_b/bang_trung_tam.tex` |
| Mention-overlap / mention-free split (λ=8) | `scripts/tables/build_mention_split_table.py` | `paper_b/bang_canh_v2.tex` |
| Appendix: DB15K | `scripts/tables/build_main_tables.py` | `paper_b/bang_phuluc_db15k.tex` |
| Compact versions | `scripts/tables/build_mini_tables.py` | `paper_b/mini_bang_*.tex` |

The public English-only tables are in `tables/`. `python3 scripts/build_public_tables.py` rebuilds them from
`tables_frozen/` and `python3 scripts/diff_numbers_only.py` checks that every one of the 13 tables carries exactly the
numbers of its frozen version (token by token over the whole file, and again over the table body with only two
label tokens, `A3` and `[F8]`, removed).

The image-permutation control on ViProKG (real vs permuted images, 3 seeds, same run) is
`notebooks/03-image-permutation-v2.ipynb`; its scoring code is `scripts/score_anh_v2.py` (paired delta, across-seed
95% t-CI with df = 2, query-level paired bootstrap, B = 1000).
Result (3 seeds): delta(real - permuted) overall = +0.17 points, across-seed CI [-0.92, +1.26], key-bootstrap CI [-0.12, +0.46];
the pre-specified criterion |delta| <= 0.65 holds (no signal detected from the image content; not an equivalence claim).
Images switched off (notebook 04, 3 runs at lambda 8): delta(real - off) overall = +0.10, across-seed CI [-0.86, +1.05], key-bootstrap CI
[-0.18, +0.37]; the neutrality criterion |delta| <= 0.65 holds.

Tables without a generator (assembled from archived run outputs, shipped frozen with sha256 in
`MANIFEST.md`): answer-name masking and matched random masking, attribute-triple prior, image ablation,
mention-free level, prior K=3, information-ceiling probe.

`scripts/tables/build_hits_table.py` and the `append_*.py` scripts rebuild `hits_bang.tsv`. Because
`build_hits_table.py` **overwrites** it, always run the `append_*` scripts afterwards; the verification
above does not need either.

Headline values (frozen): MKG-W MRR 42.50, MKG-Y MRR 44.80, ViProKG (MENTOR_base+L, λ=8) total MRR 41.20.
The MKG-W/MKG-Y headline rows are stored as signed constants (`append_headline_wy_rows.py`); per-query
ranks for them are not part of this release.

## Training and evaluation (GPU)

`src/train.py` trains one configuration; `src/evaluate.py` produces filtered ranks. Scripts that tokenise text take
`--text_encoder <path>` (default `bert-base-uncased`, which needs internet or a local cache; pass a local path offline).
Legacy identifiers (e.g. module attributes such as `goc`, `dau`, `them_*`) are kept for checkpoint compatibility. The Kaggle notebooks
in `notebooks/` show the exact command lines. Kaggle GPU sessions run with the internet off, so pass
encoder paths (`--encoder`) rather than model names.

## Baselines

`baselines/harness/` contains our adapters and `baselines/patches/` the small patches applied to the upstream code
(repository, commit and how to apply: `baselines/patches/README.md`). Third-party code (MoMoK, TSAM) is not
redistributed; fetch it from the upstream repositories. Labels used in the paper: MoMoK and TSAM are original code run on our
data; **DSoM is an approximate re-implementation (approx.)** and does not reproduce the published numbers.

## License

| Part | License |
|---|---|
| Code (`src/`, `scripts/`, `notebooks/`, `baselines/harness/dsom/`) | MIT, see `LICENSE` |
| Data (`data/`, `tables_frozen/`, rank files and run configs) | CC BY-SA 4.0, see `LICENSE-DATA` |
| `baselines/patches/*.diff` | the added lines are MIT; the context belongs to the (unlicensed) upstream projects |
| `baselines/harness/{momok,tsam}/`, `baselines/{momok,tsam}/` | **not** MIT; adapters for third-party code that publishes no license; see `baselines/NOTICE` (source repositories, commits, local modifications) |

Attribution of Wikipedia text: `data/wikipedia_attribution.tsv` lists, for every entity QID, the Vietnamese Wikipedia
page title, URL and the id/timestamp of the page's latest revision on the retrieval date (17/10/2026); the stored text
was fetched earlier, so it may correspond to an older revision of the same page, and the page history lists the
authors. Of 8,255 nodes, 8,024 have a page title (7,594 recorded at crawl time, 430 from the Wikidata sitelink), 8,011 have a
revision id (13 pages recorded at crawl time no longer exist: title kept, revision empty); the others (attribute-value nodes
and entities without a Vietnamese page) have empty fields. The column `title_source` says where the title comes from. Reuse of the text must keep CC BY-SA 4.0 and this attribution.

Data provenance: entity descriptions are taken from Vietnamese Wikipedia (CC BY-SA 4.0); graph structure from
Wikidata (CC0). Images are **not** distributed; each image keeps its own Wikimedia Commons / Flickr license, logged per
image in `data/final/ViMMKG2_core/manifest.tsv` (license short name, artist, source page, URL, sha256).
`src/mygo_reference_eval.py` contains a verbatim excerpt of the MyGO repository (no upstream license; see `baselines/NOTICE`).
DSoM here is an **approximate re-implementation** of the published method, not the authors' code.

## Citation

If you use the ViProKG data, the evaluation protocol or MENTOR, please cite the paper (entry to be added with the
camera-ready version).

When you use the baseline numbers, also cite the original baseline papers (MoMoK, TSAM, DSoM) and MyGO; see `baselines/NOTICE`.
