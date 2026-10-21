<!-- SINH bởi TAY-M (TIN_278 B10) — ablation kênh ảnh. Nguồn: BC136 (anh_wy 3 seed) + ❓AZ v2. -->

# Bảng ablation kênh ẢNH — W/Y/v2 × {tắt, thật, hoán vị} × {tổng, CÓ, mention-free} (MRR %, 3 seed)

| bộ | nhóm | tắt | thật | hoán vị | Δ(thật−tắt) | Δ(thật−hoán vị) |
|---|---|--:|--:|--:|--:|--:|
| **MKG-W** | tổng | 42,15 | 42,13 | 42,23 | **−0,03** ±0,29 | **−0,10** ±0,26 |
| | CÓ-cạnh | 53,70 | 55,11 | 53,68 | +1,41 ±0,39 | +1,43 ±0,62 |
| | mention-free | 37,00 | 36,33 | 37,12 | −0,67 ±0,58 | −0,79 ±0,64 |
| **MKG-Y** | tổng | 44,06 | 44,27 | 44,06 | **+0,21** ±0,19 | **+0,21** ±0,34 |
| | CÓ-cạnh | 43,61 | 43,52 | 43,12 | −0,10 ±1,33 | +0,39 ±0,69 |
| | mention-free | 44,29 | 44,65 | 44,53 | +0,36 ±0,54 | +0,12 ±0,26 |
| **ViMMKG 2.0** † | tổng | 41,11 | 41,20 | 41,03 | +0,10 ±0,38 | +0,17 ±0,44 |
|  | CÓ-cạnh | 50,54 | 50,59 | 50,40 | +0,04 ±0,42 | +0,19 ±0,38 |
|  | mention-free | 37,82 | 37,93 | 37,77 | +0,12 ±0,38 | +0,16 ±0,46 |

- Headline W/Y/DB15K = **text-only** (`use_image 0`); v2 headline = text+ảnh (nhất quán §ảnh).
- **Kênh ảnh trung tính** trên headline (B7: |Δ tổng| ≤ 0,65) và **nội dung ảnh không thêm tín hiệu**
  (B14: thật ≈ hoán vị). MKG-W: ảnh thật nâng CÓ-cạnh +1,41 / hạ mention-free −0,67 ⇒ triệt tiêu tổng.
- † v2 (λ=8, 3 seed, cùng run): hoán vị chỉ giữa 7.790 thực thể có ảnh, mặt nạ giữ nguyên. Δ(thật−hoán vị) tổng +0,17 (KTC seed [−0,92; +1,26], bootstrap khoá [−0,12; +0,46]): không phát hiện tín hiệu từ nội dung ảnh. Ô “tắt” v2: 3 run use_image 0 ở λ=8; Δ(thật−tắt) tổng +0,10 (KTC seed [−0,86; +1,05], bootstrap khoá [−0,18; +0,37]), tiêu chí trung tính |Δ|≤0,65: ĐẠT (bật/tắt có sẵn ở λ_align=0: Δ tổng −0,02 [−0,57; +0,52]). ± = SD qua seed.
- Nguồn: BC136 (`anh_wy`, sha 087ff9effc6a8d72); v2 ❓AZ (BC74 §1b).
