<!-- SINH bởi TAY-M (TIN_240 đ.1 / TIN_244 / TIN_261 đ.2) — (e) MỨC: min–max KHÔNG-cạnh + vị trí ta. -->

# Bảng (e) MỨC — KHÔNG-cạnh MRR (%): min–max {ta, MyGO, AdaMF, MoMoK, TSAM, DSoM re-impl.} + vị trí ta

> TIN_256 B.2: thêm **TSAM** (1 seed, W/Y/v2) + **DSoM re-impl. (approx.)** (W/Y 1 seed · v2 **3 seed**)
> vào min–max; biên độ tính lại; nhãn seed từng ô. DB15K: TSAM/DSoM **không chạy** ⇒ giữ nguyên.

| bộ | min–max KHÔNG-cạnh (model) | ta (MENTOR) | biên độ | trong 1,5 điểm? |
|---|---|--:|--:|:--:|
| MKG-W | 34,68 (DSoM re-impl.) – 38,21 (TSAM) | 35,52 (giữa) | **3,53** | ❌ |
| MKG-Y | 41,87 (MyGO) – 45,76 (TSAM) | 42,21 (gần đáy) | 3,89 | ❌ |
| DB15K | 34,08 (AdaMF) – 35,32 (ta) | 35,32 (cao nhất) | **1,24** | ✅ |
| ViMMKG 2.0 | 32,45 (MoMoK) – 41,85 (ta) | 41,85 (cao nhất) | 9,40 | ❌ |

- **Giá trị thêm (KHÔNG-cạnh %) + nhãn seed:** TSAM W **38,21** · Y **45,76** (1 seed) · v2 **41,47**
  (**3 seed, thường·KHÔNG**). DSoM re-impl. W **34,68** · Y **43,72** (1 seed) · v2 **38,91** (**3 seed,
  thường·KHÔNG**). 🔴 **TIN_278 B9: v2 dùng thường·KHÔNG NHẤT QUÁN** cho mọi model (trước đây DSoM/TSAM
  v2 ghi theo KHÔNG-gộp lát+thường 36,89/36,43 — SAI định nghĩa; nay 38,91/41,47). **AdaMF v2
  thường·KHÔNG 34,22** (3 seed). Mọi giá trị v2 trong [32,45; 41,85] ⇒ **min–max v2 KHÔNG đổi** (MoMoK
  32,45 – ta 41,85, biên độ 9,40), nhưng **TSAM 41,47 sát ta 41,85** ⇒ ta **chỉ nhỉnh** mention-free v2. ta/MyGO/AdaMF(W/Y/DB)/MoMoK giữ seed như BC95 bản 3 (1 seed,
  trừ ta/MENTOR 3 seed; AdaMF v2 nay **3 seed**). 🔴 **MKG-W DSoM policy degenerate** (max action
  0,97, nhãn BC_TAY2_DSoM); đọc số W DSoM với caveat đó.
- **"KHÔNG-cạnh cụm trong 1,5 điểm": nay SAI W/Y/v2** (TSAM kéo trần W lên 38,21 · Y 45,76; v2 vẫn trải
  rộng); **chỉ DB15K ✅** (không có TSAM/DSoM). ta KHÔNG dẫn KHÔNG-cạnh ở W/Y (dưới AdaMF, TSAM).
- 🔵 **ta KHÔNG dẫn KHÔNG-cạnh ở W/Y** (dưới AdaMF) ⇒ *"L giúp CÓ-cạnh, không giúp khái quát hoá"*.
- MoMoK không có DB15K; v2 dùng thường·KHÔNG; cross-pipeline (chưa KTC chung-khoá). TSAM thêm khi có.
- 🔴 **DB15K — khác TRAIN split giữa các ô (TIN_250 đ.2, đọc config):** EVAL chung (test gốc, n KHÔNG-cạnh = **14 292** = T12, mọi model). TRAIN split lệch: **ta 35,32 = clean-valid (train 75 261)** (pipeline ta, mọi run DB15K = clean-valid, #26d/BC125; nguồn A3′ ← v3_18d/v16_hang) · **MyGO 34,53** (`train_mygo_fgc.py --data DB15K`, split chuẩn MyGO ~79 222; mygo_che_A) · **AdaMF 34,08** (`run_adamf_mat.py -dataset=DB15K`, split chuẩn AdaMF; gop_b). ⇒ **so LEVEL trên cùng test**, nhưng ta train ÍT dữ liệu hơn (nhãn từng ô, KHÔNG bỏ hàng). W/Y/v2: mọi model cùng eval; ta dùng cùng pipeline mỗi bộ.
