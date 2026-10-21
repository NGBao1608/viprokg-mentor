# Appendix tables (repository only)

Tables referred to in the paper as "repository appendix" (TIN_301 Part A). They are copies of the public tables in
`../` (numbers identical to `tables_frozen/`, checked by `scripts/diff_numbers_only.py`):

| file | content |
|---|---|
| `bang_lat_prior.tex` | attribute slice: type-constrained frequency prior vs models, per direction (head / tail) |
| `bang_trung_tam.tex`, `bang_ablation.tex` | fusion ladder TuckER → MENTOR_base → +L → +gate on every dataset, incl. DB15K, with CIs |
| `bang_phuluc_db15k.tex` | DB15K link prediction (original split) |
| `bang_delta_L.tex` | paired lexical gain Δ_L with across-seed and query-level bootstrap CIs (incl. ViProKG) |
| `bang_e_muc.tex` | mention-free MRR level across models (MKG-W / MKG-Y / DB15K / ViProKG) |
| `bang_prior_k3.tex` | relation prior (K = 3) probe |
| `bang_tran_thong_tin.tex` | information-ceiling probe |

Not yet built: the 12-cell CI table behind the bold marks of the ViProKG table (`data/experimental/v3/ktc_v2_*.json`).
