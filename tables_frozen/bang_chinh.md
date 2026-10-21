<!-- SINH TỰ ĐỘNG bởi scripts/v3/bang_bai_chinh.py — KHÔNG sửa tay. -->

# Bảng chính — 2 bộ chuẩn (thang %)

| Model | Peer | MKG-W MRR | MKG-W H@1 | MKG-W H@3 | MKG-W H@10 | MKG-Y MRR | MKG-Y H@1 | MKG-Y H@3 | MKG-Y H@10 |
|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|
| **uni-modal KGC** | |  |  |  |  |  |  |  | |
| TransE | Y | 29.19 | 21.06 | 33.20 | 44.23 | 30.73 | 23.45 | 35.18 | 43.37 |
| RotatE | Y | 33.67 | 26.80 | 36.68 | 46.73 | 34.95 | 29.10 | 38.35 | 45.30 |
| TuckER | Y | 30.39 | 24.44 | 32.91 | 41.25 | 37.05 | 34.59 | 38.43 | 41.45 |
| **MMKGC — embedding / MoE** | |  |  |  |  |  |  |  | |
| IKRL | Y | 32.36 | 26.11 | 34.75 | 44.07 | 33.22 | 30.37 | 34.28 | 38.26 |
| RSME | Y | 29.23 | 23.36 | 31.97 | 40.43 | 34.44 | 31.78 | 36.07 | 39.09 |
| OTKGE | Y | 34.36 | 28.85 | 36.25 | 44.88 | 35.51 | 31.97 | 37.18 | 41.38 |
| MoSE-AI † | Y | 35.99 | 29.33 | 38.38 | 48.81 | 34.96 | 31.24 | 36.26 | 41.46 |
| IMF | Y | 34.50 | 28.77 | 36.62 | 45.44 | 35.79 | 32.95 | 37.14 | 40.63 |
| AdaMF-MAT | Y | 35.85 | 29.04 | 39.01 | 48.42 | 38.57 | 34.34 | 40.59 | 45.76 |
| MyGO | Y | 36.10 | 29.78 | 38.54 | 47.75 | 38.44 | 35.01 | 39.84 | 44.19 |
| MoMoK | Y | 35.89 | 30.38 | – | – | 37.91 | 35.09 | – | – |
| MoCME ‡ | N | *37.79* | *30.81* | – | – | *40.37* | *36.21* | – | – |
| M-Hyper | Y | 37.02 | 31.24 | 39.16 | 48.84 | 39.46 | 36.02 | 40.92 | 45.22 |
| M2GDT ‡ | N | *37.77* | *32.18* | *39.83* | *48.15* | *39.25* | *36.12* | *40.96* | *45.10* |
| RAPC (mean±std) | Y | 38.70 | 33.59 | 40.10 | 48.72 | 38.60 | 35.83 | 39.57 | 44.57 |
| DSoM | Y | 40.90 | 33.56 | 43.58 | **55.28** | 40.02 | 35.67 | 41.70 | 47.71 |
| TSAM | Y | 40.07 | 33.29 | 42.53 | 52.72 | 39.80 | 35.28 | 41.29 | 46.44 |
| MDBGF | Y | 40.10 | 33.42 | 42.83 | 52.35 | 39.13 | 35.15 | 40.78 | 46.06 |
| DyME-KGC | Y | 36.35 | 31.17 | – | – | 38.68 | 35.93 | – | – |
| **MLLM-based** | |  |  |  |  |  |  |  | |
| KGT ‡ | N | *43.27* | *36.02* | – | – | *43.62* | *37.68* | – | – |
| MgHiSal | Y | 41.15 | 34.12 | 43.17 | 53.25 | – | – | – | – |
| **retrieval-generative** | |  |  |  |  |  |  |  | |
| HERGC | Y | 39.12 | 33.65 | 41.67 | 48.12 | 39.82 | 36.73 | 41.42 | 44.84 |
| RADD ‡ | N | *45.45* | *39.88* | *47.58* | *55.83* | *39.40* | *35.75* | *40.93* | *45.87* |
| **ta** | |  |  |  |  |  |  |  | |
| **MENTOR** (ours) | – | **42.50** ±0.11 | **35.89** | **45.26** | 54.89 | **44.80** ±0.29 | **40.07** | **47.00** | **53.27** |

- ngày quét mốc ngoài: MRR **2026-10-02** · Hits **2026-10-04**
- **RAPC (best run)** KHÔNG có dòng riêng: bài đó báo cả best run (MRR 39.12 / 38.89 trên MKG-W / MKG-Y) lẫn mean±std đa-seed; bảng dùng **mean±std** vì đó mới là số so được.
- `†` (MoSE-AI): figures for MoSE-AI are a third-party reproduction (DSoM, Table 1): the original MoSE paper (EMNLP 2022) does not evaluate these three benchmarks.
- `–` = bài gốc **không báo** chỉ số đó. Bảng này **không còn ô 'chưa tra được'** nào.
- `‡` = venue **chưa phản biện**; dòng in nghiêng và **không bao giờ** được in đậm. Đậm = tốt nhất mỗi cột **giữa các bài peer-reviewed** (có MENTOR), xét toàn bảng.
- MENTOR = cấu hình chọn trên **valid** từng bộ: MKG-W → cổng OOF (λ=4, scores run) · MKG-Y → cổng OOF (λ=0,1, y_cu run). A3′ một-chuyên-gia và cổng 2-chuyên-gia so trên valid, **cổng** là headline (một-chuyên-gia ở ablation). 3 seed; **± = SE, 95% CI ≈ ±4,3×SE** (t, df=2) ở MRR.
- sinh 2026-10-16 bởi `scripts/v3/bang_bai_chinh.py`
