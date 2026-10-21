<!-- SINH bởi TAY-M (TIN_273 A9) — prior tần suất lát + lift model−prior. Nguồn: train_lat/test_lat + ranks. -->

# Bảng lát · prior tần suất vs model — ViMMKG 2.0 (987 lát test, 2 chiều)

> **Prior** = tần suất quan hệ trên **train** (type-constrain): tail = xếp 48 giá trị theo
> P(giá trị | r); head = xếp thực thể theo P(thực thể | r). **Lift = model − prior** (MRR, điểm).
> Khai trước (ghi trước khi chạy): **prior tail ≥ 45** ⇒ ✅ (65,43). Nguồn số:
> `ext/ViMMKG2/{train,test}_lat.txt` (prior) · ranks_test 3 seed (model).

| | head MRR | head H@1 | tail MRR | tail H@1 | lift head | lift tail |
|---|--:|--:|--:|--:|--:|--:|
| **PRIOR tần suất** | **1,67** | 0,00 | **65,43** | 44,38 | — | — |
| MENTOR (A3′* λ8) | 2,78 | — | 64,99 | — | +1,11 | **−0,44** |
| AdaMF-MAT | 2,22 | — | 63,47 | — | +0,55 | **−1,96** |
| MyGO+seg | 1,41 | — | 61,65 | — | −0,26 | **−3,78** |
| MoMoK | 1,14 | — | 55,94 | — | −0,53 | **−9,49** |
| TSAM | 2,28 | — | 60,05 | — | +0,61 | **−5,38** |
| TuckER (cấu trúc) | 1,67 | — | 63,84 | — | −0,00 | **−1,59** |
| DSoM re-impl. | 1,42 | — | **68,14** | — | −0,25 | **+2,71** |

- 🔴 **Phát hiện chính:** trên **lát tail**, **mọi model TRỪ bản tái hiện xấp xỉ DSoM đều KHÔNG vượt
  prior tần suất** (lift −1,6…−9,5): MENTOR 64,99 · TuckER 63,84 · AdaMF 63,47 · MyGO 61,65 · TSAM
  60,05 · MoMoK 55,94 đều **≤** prior 65,43. **Ngoại lệ duy nhất: DSoM re-impl. 68,14 (lift +2,71)**
  — bản **re-impl. approx., không khớp mốc gốc** (G2 policy degenerate ở MKG-W, CHAM_V15). ⇒ **Không
  model peer-reviewed/của-ta nào vượt prior**; điểm tail "cao" (~0,62) của lát **không phải** do model
  phân biệt mà vì **một prior tần suất type-constrain đã giải gần hết** (48 giá trị, 9 quan hệ lát).
  **Kết luận KHÔNG đổi:** lát không là tác vụ phân biệt có ý nghĩa (DSoM re-impl. +2,7 nhỏ, không tin cậy).
- **lát head** gần ngẫu nhiên ở mọi model (MRR 1,1–2,8) **và** prior (1,67) — đoán thực thể từ giá
  trị dùng chung là bất khả (nhiều thực thể/giá trị).
- ✅ **Hết ô ⟳.** MoMoK/DSoM/TuckER/TSAM điền từ ranks 3 seed (`v2_03/D4_tucker`, `baselines/tsam`,
  scratchpad momok/dsom); method tái lập **khớp bit** MENTOR 2,78/64,99 + MyGO 1,41/61,65.
- Nguồn: `scripts/v3/*` (prior tính từ train_lat); model từ `ranks_test.tsv` (lọc test_lat, 2 chiều).
