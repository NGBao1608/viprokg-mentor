<!-- SINH TỰ ĐỘNG bởi scripts/v3/bang_bai_chinh.py — KHÔNG sửa tay. -->

# Phụ lục — DB15K (thang %)

| Model | Peer | DB15K MRR | DB15K H@1 | DB15K H@3 | DB15K H@10 |
|---|:-:|---:|---:|---:|---:|
| **uni-modal KGC** | |  |  |  | |
| TransE | Y | 24.86 | 12.78 | 31.48 | 47.07 |
| RotatE | Y | 29.28 | 17.87 | 36.12 | 49.66 |
| TuckER | Y | 33.86 | 25.33 | 37.91 | 50.38 |
| **MMKGC — embedding / MoE** | |  |  |  | |
| IKRL | Y | 26.82 | 14.09 | 34.93 | 49.09 |
| RSME | Y | 29.76 | 24.15 | 32.12 | 40.29 |
| OTKGE | Y | 23.86 | 18.45 | 25.89 | 34.23 |
| MoSE-AI † | Y | 37.64 | 28.14 | 39.60 | 51.63 |
| IMF | Y | 32.25 | 24.20 | 36.00 | 48.19 |
| AdaMF-MAT | Y | 35.14 | 25.30 | 41.11 | 52.92 |
| MyGO | Y | 37.72 | 30.08 | 41.26 | 52.21 |
| MoMoK | Y | 39.57 | 32.38 | 43.45 | 54.14 |
| MoCME ‡ | N | *39.62* | *29.71* | – | *55.36* |
| M-Hyper | Y | 41.25 | 33.64 | 45.01 | 56.09 |
| M2GDT ‡ | N | *40.54* | *32.60* | *44.49* | *55.48* |
| RAPC (mean±std) | Y | 41.80 | **35.49** | 44.74 | 53.96 |
| DSoM | Y | **42.68** | 34.45 | **46.62** | **58.12** |
| TSAM | Y | 40.50 | 32.60 | 44.36 | 55.44 |
| MDBGF | Y | 40.58 | 32.77 | 44.23 | 55.53 |
| DyME-KGC | Y | 40.72 | 33.30 | 44.42 | 55.02 |
| **MLLM-based** | |  |  |  | |
| KGT ‡ | N | *42.16* | *34.06* | *46.03* | *57.69* |
| MgHiSal | Y | 42.66 | 35.27 | 46.32 | 57.22 |
| **retrieval-generative** | |  |  |  | |
| HERGC | Y | 40.95 | 33.47 | 45.66 | 55.12 |
| RADD ‡ | N | *45.02* | *37.16* | *48.94* | *60.26* |
| **ta** | |  |  |  | |
| **MENTOR** (ours) | – | 40.26 ±0.07 | 32.47 | 43.80 | 55.47 |

- ngày quét mốc ngoài: MRR **2026-10-02** · Hits **2026-10-04**
- **RAPC (best run)** KHÔNG có dòng riêng: bài đó báo cả best run (MRR 42.21 trên DB15K) lẫn mean±std đa-seed; bảng dùng **mean±std** vì đó mới là số so được.
- `†` (MoSE-AI): figures for MoSE-AI are a third-party reproduction (DSoM, Table 1): the original MoSE paper (EMNLP 2022) does not evaluate these three benchmarks.
- `–` = bài gốc **không báo** chỉ số đó. Bảng này **không còn ô 'chưa tra được'** nào.
- `‡` = venue **chưa phản biện**; dòng in nghiêng và **không bao giờ** được in đậm. Đậm = tốt nhất mỗi cột **giữa các bài peer-reviewed** (có MENTOR), xét toàn bảng.
- MENTOR = cấu hình chọn trên **valid** từng bộ: DB15K → A3′ (split gốc). A3′ một-chuyên-gia và cổng 2-chuyên-gia so trên valid, **cổng** là headline (một-chuyên-gia ở ablation). 3 seed; **± = SE, 95% CI ≈ ±4,3×SE** (t, df=2) ở MRR.
- sinh 2026-10-16 bởi `scripts/v3/bang_bai_chinh.py`
