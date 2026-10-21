<!-- SINH bởi TAY-M (TIN_244 đ.3) — ablation chuỗi, test tổng MRR, 3 seed, KTC t df=2. -->

# Bảng ablation — chuỗi thành phần (test tổng MRR %, 3 seed, [KTC 95% t df=2])

| bộ | TuckER (structure-only) | + text+ảnh (nền) | + L (A3′) | + cổng OOF (headline) |
|---|---|---|---|---|
| MKG-W (λ=4) | 30.39 | 38.17 [37.33; 39.00] | 42.15 [41.57; 42.74] | **42.50** [42.02; 42.98] |
| MKG-Y (λ=0,1) | 37.05 | 39.12 [38.91; 39.34] | 44.06 [43.82; 44.31] | **44.80** [43.56; 46.04] |
| DB15K (λ=2) ‡ | – | 39.15 [38.86; 39.43] | 40.43 [40.32; 40.54] | **41.04** [40.86; 41.23] |
| ViMMKG 2.0 (λ=8) | 36.80 | 40.89 [40.73; 41.05] | 41.20 [40.66; 41.75] | 41.08 [40.47; 41.70] (< A3′) |

- **Bước lớn nhất = + L (kênh lexical mention):** W +3,98 · Y +4,94 · DB +1,28 · v2 +0,31. + cổng OOF
  thêm nhỏ (W +0,35 · Y +0,74 · DB +0,61); **v2: cổng < A3′ ⇒ headline v2 = A3′ (không cổng).**
- **Định nghĩa "nền" = TuckER + text (+ ảnh ở v2) + λ_align** (λ theo bộ: W 4 · Y 0,1 · DB 2 · v2 8, chọn trên valid). **+ text và + λ_align gộp trong "nền"** (chưa có cấu hình text-không-align riêng để tách hai bước);
  ghi gộp, chú thích. **+ CoRe-FT: CHƯA ĐO.**
- `±`/KTC: `t` df=2 qua 3 seed; **nhiễu run-qua-run GPU ~0,3–0,65 điểm** — khác biệt dưới mức này không claim.
- TuckER = structure-only (bang_chinh/bang_vimmkg2); DB15K "–" (TuckER DB ở uni-modal block, không chạy lại).
- **‡ DB15K = clean-valid (train 75 261);** Δ trong-đường (nền→+L→+cổng) hợp lệ, **KHÔNG so với phụ lục 40,26 (split gốc, train 79 222)** — hai dụng cụ khác train split (BC125/#26d).
