# Release notes — v1.0.0 (17/10/2026)

First public release of the code and data for *"Where Do Multimodal Gains Come From? Auditing Lexical Dependence in
Multimodal Knowledge Graph Completion with ViProKG and MENTOR"*.

Contents: table generators and regression gate (`scripts/`), model/training/evaluation/mention-rule/masking/OOF-gate
code (`src/`), baseline adapters and rank files (`baselines/`), ViProKG data without images (`data/`), frozen
tables (`tables_frozen/`), and two executed Kaggle notebooks (`notebooks/`, outputs included).

Validation performed before tagging:
* `python3 scripts/verify_tables.py`: 0 mismatches, `check_frozen_tables.py` returns 0, sha256 of the 11 tables of the
  third freeze unchanged.
* Notebook 01 (Kaggle, CPU): 4/4 criteria pass (tables identical; ViProKG 41.20 / 40.89 from per-key ranks;
  headline rows 42.50 / 44.80 / 41.20; answer-name masking drop 10.78 / 17.43 / 8.46).
* Notebook 02 (Kaggle GPU, MKG-Y, seed 42, about 1.5 min): g* = 0.25, test MRR 44.83 against the reference 44.77
  (|delta| = 0.06 <= 0.65).

Licenses: code MIT (`LICENSE`); data CC BY-SA 4.0 (`LICENSE-DATA`); third-party baselines see `baselines/NOTICE`.

Known limitations: eight tables were assembled from archived run outputs and are shipped frozen without a generator;
the MKG-W/MKG-Y headline rows are stored constants; the two local patches applied to MoMoK and TSAM are recorded in
`baselines/NOTICE` but not shipped; `rebuild_mention_edges_v2.py` was not run end to end (the v2 edge file is shipped as data).

Suggested sentence for the paper (insert the final URL): "Code, data (without images), frozen tables and Kaggle
notebooks to reproduce Tables 2-8 are available at <URL> (v1.0.0)."

## v1.1.0 (17/10/2026)

* `tables/`: English-only versions of the 13 tables (no internal labels or local paths); `scripts/diff_numbers_only.py`
  checks that every table carries exactly the numbers of its frozen version.
* `baselines/patches/`: diffs applied to MoMoK and TSAM (upstream commits stated; applied files AST-identical to the files run).
* `data/wikipedia_attribution.tsv`: Vietnamese Wikipedia page, URL and revision of every entity (CC BY-SA attribution).
* `--text_encoder` for the four scripts that load a tokenizer; import order normalised (`ruff --select I`);
  CPU smoke of 12 training variants byte-identical to the original code.
* `.gitignore`, `CITATION.cff`, `.zenodo.json`; large rank files and tensors are distributed in the Zenodo record.
* Executed notebooks in `notebooks/` were re-run on Kaggle with this version (`train.py` sha256-16 `bed4445a713725d3`): notebook 01 PASS 4/4 (13/13 public tables identical in numbers); notebook 02 g* = 0.25, test MRR 44.83 vs 44.77 (delta +0.06, criterion 0.65) PASS.

## v1.2.0 (18/10/2026)

* `notebooks/03-image-permutation-v2.ipynb`, `scripts/score_anh_v2.py`: image-permutation control on ViProKG
  (MENTOR_base+L, lambda 8; real vs permuted images; 3 seeds; one GPU session, 6 runs).
  Delta(real - permuted) overall = +0.17 points, across-seed 95% CI [-0.92, +1.26], key-bootstrap CI [-0.12, +0.46];
  the pre-specified criterion |delta| <= 0.65 holds. The CI across seeds is wider than the threshold, so the result is
  "no signal detected", not equivalence. The real branch reproduces the signed ranks bit for bit (5,274/5,274 keys, 3 seeds).
* `tables_frozen/bang_anh` and `tables/bang_anh`: ViProKG rows added (the only changed table; `hits_bang.tsv` unchanged).
* per-key ranks of both branches in `data/experimental/v3/ranks_v2/anh_v2_*`.

## v1.2.1 (19/10/2026)

* `notebooks/04-image-off-v2.ipynb` (executed), `scripts/tables/append_image_off_v2.py`, `--tag_b off` in `scripts/score_anh_v2.py`:
  3 runs with `--use_image 0` at lambda 8 fill the "off" column of the ViProKG image table.
  Delta(real - off) overall = +0.10 points, across-seed 95% CI [-0.86, +1.05], key-bootstrap CI [-0.18, +0.37]; the
  neutrality criterion |delta| <= 0.65 holds (written before the run). Per-key ranks in `data/experimental/v3/ranks_v2/anh_v2_off_s*`.
* `tables_frozen/bang_anh` and `tables/bang_anh` now have all cells of the ViProKG rows (the only changed table).

## v1.3.0 (19/10/2026)

* Public name of the dataset is now **ViProKG** (Vietnamese Provenance-tracked multimodal Knowledge Graph) in the
  README, MANIFEST, CITATION.cff, .zenodo.json, LICENSE-DATA, the public tables (`tables/`, label `tab:viprokg`) and
  the markdown cells of notebooks 01-04. **No number changed** (`scripts/diff_numbers_only.py` 13/13,
  `scripts/verify_tables.py` 0 mismatches).
* Data folders, file names and lookup keys keep their identifiers (`data/experimental/v3/ext/ViMMKG2/`,
  `data/final/ViMMKG2_core/`, `ViMMKG2-textual-phobert-seg.json`, the `hits_bang.tsv` key `ViMMKG2`), so every
  path, hash and executed notebook output stays valid. In the executed notebooks only two display strings of each of
  notebooks 03 and 04 changed (the printed pre-registered criterion and an assertion message); outputs, data paths and
  lookup keys are untouched.
* `tables_frozen/` is unchanged (frozen record). `scripts/build_public_tables.py` applies the public name;
  `scripts/diff_numbers_only.py` maps the frozen dataset name to the public one before comparing numbers.
* The citation of the earlier conference paper was removed from README and CITATION.cff; this release stands alone.
* Docstrings, comments and printed/help messages in `src/`, `scripts/` and `baselines/harness/` use the public name (20
  replacements; every changed file is AST-equivalent to the previous one). Identifiers, dataset keys and data paths keep
  `ViMMKG2`.
* Suggested repository name: `viprokg-mentor`.

## v1.3.0 (re-packed 20/10/2026)

* Archive rebuilt after a final clean-up of comments and the cross-reference lines in `scripts/` (no internal ticket
  references; the `build_public_tables.py` source strings now use the public name directly). No behaviour, number or
  data path changed: `scripts/verify_tables.py` 0 mismatches, `scripts/tables/check_frozen_tables.py` rc 0,
  `scripts/diff_numbers_only.py` 13/13, regenerated `tables/` byte-identical.
* The Zenodo archive is a superset of the git repository (rank files and baseline outputs).
