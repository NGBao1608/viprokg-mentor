# MANIFEST — ViProKG / MENTOR release (TIN_281)

Every file under `scripts/`, `src/` and `baselines/harness/` is a **copy** of a repository file that was
renamed and cleaned (Vietnamese comments/messages -> English, internal labels, emoji, dead code and hard-coded
local paths removed, black -l 100 / ruff F,E9 clean). Logic is unchanged: order of operations, seeds, default
hyper-parameters, data-reading paths, tie-breaks and filters were not touched. "Source" paths are relative to the
original working repository (not shipped).

Verification evidence is in section 6. "Tables" use the paper numbering: T1 dataset, T2 main (MKG-W/Y),
T3 ViProKG, T4 central ablation, T5 answer-name masking, T6 mention-overlap/free split, T7 attribute-triple prior,
T8 image ablation.

Licenses: code `LICENSE` (MIT), data `LICENSE-DATA` (CC BY-SA 4.0), third-party baselines `baselines/NOTICE`.

## 1. Table generators and gate (`scripts/`)
| source | release file | role | tables | lines before -> after |
|---|---|---|---|---|
| scripts/v3/bang_bai_chinh.py | scripts/tables/build_main_tables.py | main / ViProKG / DB15K tables | T2,T3,App.DB15K | 605 -> 776 |
| scripts/v3/sinh_cham_canh_v2.py | scripts/tables/build_mention_split_table.py | mention-overlap/free split + cham_canh_v2_lambda8.json | T6 | 206 -> 254 |
| scripts/v3/sinh_bang_trung_tam.py | scripts/tables/build_central_ablation_table.py | central ablation table | T4 | 84 -> 112 |
| scripts/v3/sinh_mini_vi.py | scripts/tables/build_mini_tables.py | compact table variants | T2,T3,App. | 198 -> 236 |
| scripts/v3/them_hang_v2.py | scripts/tables/append_v2_rows.py | append 5 ViProKG rows to hits_bang.tsv | T3 | 153 -> 179 |
| scripts/v3/them_hang_mygo_seg.py | scripts/tables/append_mygo_seg_row.py | append MyGO+seg row | T3 | 65 -> 77 |
| scripts/v3/them_hang_headline_wy.py | scripts/tables/append_headline_wy_rows.py | append signed MKG-W/MKG-Y headline rows | T2 | 57 -> 73 |
| scripts/v3/bang_hits.py | scripts/tables/build_hits_table.py | build base hits_bang.tsv (OVERWRITES it; not run by verify) | T2 | 352 -> 388 |
| scripts/v3/doi_chieu_dong_bang.py | scripts/tables/check_frozen_tables.py | regression gate vs signed headline values | all | 62 -> 75 |
| (new) | scripts/verify_tables.py | regenerate tables, compare sha256 with `tables_frozen/`, run the gate | all | - -> 91 |
| (new, from a notebook cell) | scripts/score_masking_drop.py | answer-name masking MRR drop on overlap/free split | T5 | - -> 82 |
| (new, from a notebook cell) | scripts/score_a3p_run.py | select g* on validation, test MRR at g* | notebook 02 | - -> 55 |

These generators have no cross-imports except `append_*` and `build_mention_split_table` importing `build_hits_table`, and `build_hits_table` reading `verify/mygo_eval_reference.py`.

## 2. Model, training, evaluation, mention rule, masking, OOF gate (`src/`)
| source | release file | role | tables | lines before -> after |
|---|---|---|---|---|
| vipronet/train.py | src/train.py | train a MENTOR variant (one run, writes dumps) | T2,T3,T4,App.DB15K | 1359 -> 2099 |
| vipronet/huong.py | src/scan_variants.py | model variants incl. TuckER+text, lexical channel L, heads, losses | T2-T4 | 926 -> 1127 |
| vipronet/evaluate.py | src/evaluate.py | filtered two-direction evaluation, rank/score dumps | all | 413 -> 497 |
| vipronet/model.py | src/model.py | base multimodal TuckER model | T2-T4 | 271 -> 318 |
| vipronet/config.py | src/config.py | constants and defaults | all | 70 -> 74 |
| vipronet/data.py | src/data.py | graph + metadata loader | all | 140 -> 147 |
| vipronet/eval_v2.py | src/eval_attribute_slice.py | attribute-triple (slice) evaluation | T3,T7 | 96 -> 122 |
| vipronet/mygo_ref.py | src/mygo_reference_eval.py | verbatim MyGO ranking reference used for cross-checks | gate | 59 -> 57 |
| vipronet/run_grid.py | src/run_grid.py | grid driver (Kaggle) | - | 105 -> 159 |
| vipronet/v16_cong.py | src/oof_gate.py | out-of-fold gate over {S, L} scores | T2,T4 | 85 -> 101 |
| scripts/v3/lsr_dac_trung_L.py | src/lexical_channel_features.py | features of the lexical mention channel (S_lex) | T2-T4 | 164 -> 179 |
| scripts/v3/chuan_hoa_nhan.py | src/label_normalization.py | the single label-normalisation rule (mention rule v2) | T6,T5 | 436 -> 443 |
| scripts/v3/v16_chia_fold.py | src/make_oof_folds.py | 3 triple-level folds for the OOF gate | T4 | 44 -> 48 |
| scripts/v3/dung_lai_canh_v2.py | src/rebuild_mention_edges_v2.py | rebuild mention-overlap edge files under rule v2 | T6 | 326 -> 398 |
| scripts/v3/v2_tien_xu_ly.py | src/preprocess_vimmkg2.py | ViProKG 0-GPU preprocessing: folds, BM25, L features | T3,T4 | 167 -> 207 |
| scripts/v3/sinh_L_che.py | src/build_masked_chain.py | answer-name masking chain for one dataset | T5 | 277 -> 334 |
| scripts/v3/v10_sinh_du_lieu.py | src/build_lexical_data.py | BM25 top-K candidates for the lexical channel | T4 | 170 -> 209 |
| scripts/v3/v9a_baseline_ten.py | src/mention_only_baseline.py | mention-only baseline | App. | 172 -> 211 |
| scripts/v3/v9a_ro_text.py | src/label_leak_probe.py | label-in-text leak probe | App. | 149 -> 181 |
| scripts/v3/v9a_tao_mask.py | src/make_masked_text.py | masked text for the masking tensors | T5 | 111 -> 159 |
| scripts/v3/moc_t12_v2.py | src/mention_split_reference_counts.py | reference counts of the mention-free split | T6 | 108 -> 137 |
| scripts/v3/sinh_mo_ta_che.py | src/build_masked_descriptions.py | masked descriptions under rule v1/v2 | T5 | 266 -> 342 |
| scripts/v3/sinh_T_ctx.py | src/build_text_context.py | single source of truth for T_ctx.pt | T2-T5 | 90 -> 103 |
| scripts/v3/sinh_T_ctx_cli.py | src/build_text_context_cli.py | CLI wrapper of build_text_context | T2-T5 | 112 -> 131 |
| scripts/v3/sinh_T_ctx_mask.py | src/build_text_context_masked.py | masked T_ctx tensor | T5 | 224 -> 235 |
| scripts/v3/sinh_T_tok.py | src/build_text_tokens.py | frozen token embeddings (reader) | - | 112 -> 135 |
| scripts/v3/bang_egen.py | src/mention_free_screen.py | mention-free screening table | T6 | 277 -> 355 |
| scripts/v3/mo_xe_ext.py | src/inspect_validation_dump.py | dissection of validation dumps | App. | 386 -> 509 |
| scripts/v3/doc_dump_scores.py | src/read_score_dump.py | read score dump, top-K, gate checks | T4,T5 | 162 -> 228 |
| scripts/v3/cong_oof_prior_k3.py | src/oof_gate_prior_k3.py | OOF gate with K=3 experts {base, L, prior} | App. prior K=3 | 211 -> 256 |
| scripts/v3/prior_quan_he.py | src/relation_prior.py | type-constrained relation frequency prior | T7,App. | 295 -> 383 |
| scripts/v3/kiem_ke_3_bo.py | src/dataset_profile.py | dataset profile incl. degeneracy (high-cardinality) profile | T1 | 248 -> 332 |

Total src: 8031 -> 10216 lines (black expands wrapped lines; no logic added).

## 3. Baselines (`baselines/`)
| source | release file | role | lines before -> after |
|---|---|---|---|
| baselines/MoMoK_harness/make_vimmkg2_momok.py | baselines/harness/momok/make_vimmkg2_momok.py | build the ViProKG layout for MoMoK | 111 -> 120 |
| baselines/MoMoK_harness/train_eval_dump.py | baselines/harness/momok/train_eval_dump.py | run MoMoK (original code) and dump per-key ranks | 346 -> 423 |
| baselines/TSAM_harness/circle_gate_v2.py | baselines/harness/tsam/circle_gate_v2.py | TSAM row gate | 60 -> 67 |
| baselines/TSAM_harness/make_vimmkg2_tsam.py | baselines/harness/tsam/make_vimmkg2_tsam.py | build the ViProKG layout for TSAM | 83 -> 87 |
| baselines/TSAM_harness/train_eval_dump_tsam.py | baselines/harness/tsam/train_eval_dump_tsam.py | run TSAM (original code) and dump per-key ranks | 234 -> 358 |
| baselines/DSoM_harness/dsom.py | baselines/harness/dsom/dsom.py | **approximate re-implementation (approx.)** of DSoM; does not reproduce published numbers | 367 -> 516 |
| baselines/kiem_nhan_hang.py | baselines/harness/check_baseline_row.py | gate before a baseline row enters a table | 168 -> 207 |

Rank files and run configs: `baselines/{momok,tsam,dsom}/<model>_<dataset>_s<seed>/` (copied unchanged). MoMoK and TSAM are original third-party code run on our data; the third-party repositories are **not** redistributed (upstream publishes no license; sources, commits and local modifications in `baselines/NOTICE`; the MIT license does not cover `harness/{momok,tsam}`); fetch upstream. DSoM is **approx.** and MIT.

## 4. Data
| path | content | note |
|---|---|---|
| data/experimental/v3/ext/ViMMKG2/ | entities, relations, id maps, train/valid/test (+ relational/attribute splits), image_index, text (PhoBERT word-segmented), folds_k3.json, L_feat.pt, V.pt | 2,637 test triples |
| data/experimental/v3/canh_v2/ | mention-overlap edges under rule v2 (`v10_canh_nhac_ten_*_v2.tsv`), whitelist, frozen zipf tables | ViProKG: 44,770 edges (header excluded) |
| data/experimental/v3/ranks_v2/ | per-key test ranks, seeds 42/7/2024: A3pLC (MENTOR_base+L), nen_l8 (MENTOR_base), momok, dsom, adamf | 5.9 MB |
| data/experimental/v3/che_v2_dumps/ | per-key dumps unmasked / answer-name-masked, 3 seeds, + `che2_kq.json` | used by notebook 01 |
| data/experimental/v3/hits_bang.tsv, ktc_v2_*.json, cham_canh_v2_lambda8.json, moc_ngoai_2026.tsv | canonical numbers and CIs | sha in section 6 |
| data/experimental/mygo_seg/, baselines/tsam/, data/from_kaggle/v2_03/result/D4_tucker_s*/ | per-key ranks for MyGO+seg, TSAM, TuckER | inputs of `build_mention_split_table.py` |
| data/final/ViMMKG2_core/manifest.tsv | per-image license log: 8,029 rows; 239 excluded (license 195, content/spam 25, lost image 14, license+spam 5) | **no images shipped** |
| data/inputs/ablation_chain.json | input of the central ablation table | |

## 5. Frozen tables (`tables_frozen/`)
Generated tables are regenerated and compared by `scripts/verify_tables.py`. Tables below have **no generator script**; they were assembled from archived run outputs and are shipped frozen (label: *no generator*).

| table | file | sha256-16 | numeric source |
|---|---|---|---|
| T5 masking + matched random masking | tables_frozen/bang_e_che.tex | d0c52df6cd03cc44 | che2_kq.json (ViProKG row recomputed by scripts/score_masking_drop.py); random-masking runs: archived outputs |
| T7 attribute-triple prior | tables_frozen/bang_lat_prior.tex | 18b2c5833d545972 | ranks_v2 + frequency prior |
| T8 image ablation | tables_frozen/bang_anh.tex | ad1a7f94cf36067f | archived run outputs (W/Y x off/real/permuted) |
| App. mention-free level | tables_frozen/bang_e_muc.tex | 3370303bc3ad1751 | per-model min-max of the mention-free split |
| App. prior K=3 (exploratory) | tables_frozen/bang_prior_k3.tex | 6efc7baf7f25aed4 | cong_oof_prior_out `kq_prior.json` |
| App. information ceiling (exploratory) | tables_frozen/bang_tran_thong_tin.tex | de483d0e1716533c | probe, MKG-W test 3 seeds |
| delta_L (earlier body table) | tables_frozen/bang_delta_L.tex | c541491927d30e3d | hits_bang + ranks |
| ablation (input of build_mini_tables) | tables_frozen/bang_ablation.tex | 2bbdaf8ec092f73f | hits_bang.tsv |

## 6. Verification evidence (CPU, 16/10/2026)
* `python3 scripts/verify_tables.py` -> every generated table identical to the frozen copy (sha256, generation date ignored), `hits_bang.tsv` = 68ac108bf1440bf5, `cham_canh_v2_lambda8.json` = 45a7d6c0c6056400, `check_frozen_tables.py` rc=0, `MISMATCHES: 0`.
* AST equivalence of each cleaned file against its original (docstrings removed, renames applied, message strings masked): see BC142 for the per-file result and the listed exceptions (dead imports removed, alias strings added, message lists).
* Dynamic check (CPU, 400 training / 80 valid / 80 test triples of MKG-Y, 2 epochs, seed 42, `PYTHONHASHSEED=0`): original vs cleaned `train.py` give byte-identical `dump_valid.tsv`, `sweep_gL.tsv`, `sweep_gL_test.tsv` for the LSR combination used in notebook 02, and identical `dump_valid.tsv` for variants T1, T2, S1, S2, L1, L2, L3, H1, H2, X1, P1. M1 and X2 fail identically in both (missing inputs); G1, G2, H2T were not run.
* Cross-module name resolution: every `from m import n` / `m.n` inside `src/` resolves.
* Not run here: GPU paths (device placement), `rebuild_mention_edges_v2.py` end to end (the shipped v2 edge file is data), baseline harness runs.

## 7. Notebooks (`notebooks/`)
| file | hardware | pass criterion (pre-registered in the first cell) |
|---|---|---|
| 01-reproduce-tables-and-masking.ipynb | CPU | tables identical; ViProKG 41.20 / 40.89 from ranks; headline rows 42.50 / 44.80 / 41.20; masking drop 10.78 / 17.43 / 8.46 |
| 02-train-mkgy-mentor-base-plus-l-seed42.ipynb | 1 GPU, ~2 min | single run MKG-Y seed 42: `|test MRR(g*) - 44.77| <= 0.65` |


## 8. Added after v1.0.0 (TIN_283)

| item | file | note |
|---|---|---|
| public English-only tables | `tables/*.tex` (13 files) | built from `tables_frozen/` by `scripts/build_public_tables.py` (460 lines); `scripts/diff_numbers_only.py` (103 lines) checks 13/13 tables for identical numbers; compiled with XeLaTeX |
| baseline patches | `baselines/patches/{momok.diff,tsam/merge_tokens.diff,README.md}` | apply cleanly to the stated upstream commits; patched files AST-identical to the files that were run |
| Wikipedia attribution | `data/wikipedia_attribution.tsv` (8255 rows) | built by `scripts/fetch_wikipedia_attribution.py`; revision = latest on 17/10/2026 |
| tokenizer path | `--text_encoder` in `build_lexical_data.py`, `mention_only_baseline.py`, `make_masked_text.py`, `label_leak_probe.py` | default `bert-base-uncased` unchanged |
| import order | `ruff --select I --fix` on `src/ scripts/ baselines/harness/` | 17 fixes; CPU smoke of 12 training variants byte-identical to the original |
| repository metadata | `.gitignore`, `CITATION.cff`, `.zenodo.json` | large rank files and `*.pt` are excluded from git (Zenodo) |

sha256-16 of the public tables (`tables/`):

| file | sha256-16 |
|---|---|
| tables/bang_ablation.tex | 1aaae0393fc44cbb |
| tables/bang_anh.tex | 48da52b2f53bfbc3 |
| tables/bang_canh_v2.tex | 2b3580b325b7cb1c |
| tables/bang_chinh.tex | ef3bd64a84bbf234 |
| tables/bang_delta_L.tex | 2a1332c94aabe48e |
| tables/bang_e_che.tex | 266b34b56caf7b8e |
| tables/bang_e_muc.tex | b4b01aff9681e2da |
| tables/bang_lat_prior.tex | fe7f02e932bc5f31 |
| tables/bang_phuluc_db15k.tex | 3df0fa28d6ecfcb3 |
| tables/bang_prior_k3.tex | f522ef03287cdfbf |
| tables/bang_tran_thong_tin.tex | 4b31ed45c444c0aa |
| tables/bang_trung_tam.tex | 81081884f68dd756 |
| tables/bang_vimmkg2.tex | bbd572d780d02273 |

## 9. Image-permutation control on ViProKG (TIN_286, v1.2.0)

| file | role |
|---|---|
| `notebooks/03-image-permutation-v2.ipynb` | 6 runs (3 seeds x real/permuted images), builds `T_ctx.pt` with `src/build_text_context.py`, pre-specified criterion in the first cells |
| `scripts/score_anh_v2.py` | paired delta real - permuted by group, across-seed CI and key bootstrap, criterion verdict |
| `scripts/tables/append_image_v2_row.py` | adds the ViProKG rows to `bang_anh` (idempotent, backup before writing); `hits_bang.tsv` is not touched |
| `data/experimental/v3/ranks_v2/anh_v2_{real,perm}_s{42,7,2024}/` | per-key test ranks of both branches (`dump_test.tsv`) and the run configs |
| `data/experimental/v3/anh_v2_score.json`, `anh_v2_permutation_log.json` | scorer output; sha256-16 and fixed points of the three permutations |

Result (Kaggle, 1 GPU, 6 runs): delta(real - permuted) overall = +0.17 MRR points; across-seed 95% t-CI [-0.92; +1.26], key
bootstrap [-0.12; +0.46]; pre-specified criterion |delta| <= 0.65: PASS. The real branch reproduces the signed MENTOR_base+L
ranks exactly (5,274 of 5,274 keys at every seed). `tables_frozen/bang_anh.{tex,md}` changed (v2 rows, caption, footnote):
sha256-16 `ad1a7f94cf36067f` -> `43fe10c71fe6133e` (v1.2.0: `c6131b3b41f92246`, before the off cells); the other 12 tables are unchanged.

## 10. Images-off branch on ViProKG (executed, v1.2.1)

| file | role |
|---|---|
| `notebooks/04-image-off-v2.ipynb` | 3 runs with `--use_image 0` (seeds 42, 7, 2024); the real branch comes from `ranks_v2/anh_v2_real_s*`; criterion |delta overall (real - off)| <= 0.65 written before the run |
| `scripts/tables/append_image_off_v2.py` | fills the off cells and delta(real - off) of the v2 rows of `bang_anh` (idempotent, backup) |

Result: delta(real - off) overall = +0.10 (across-seed CI [-0.86; +1.05], key bootstrap [-0.18; +0.37]); criterion |delta| <= 0.65: PASS. Per-key ranks: `data/experimental/v3/ranks_v2/anh_v2_off_s*`; scorer output `data/experimental/v3/anh_v2_score_off.json`.
