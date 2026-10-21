<!-- SINH TỰ ĐỘNG bởi scripts/v3/bang_bai_chinh.py — KHÔNG sửa tay. -->

# Bảng ViMMKG 2.0 — **3 dòng/model** (LUẬT #31), test 2 637 triple = 5 274 khoá

> 🔴 **KHÔNG so chéo bảng này với MKG-W/MKG-Y** (`KET_QUA_CUOI.md` §1.1, đã ký): A3′ v2
> dùng `--use_image 1` (ba bộ chuẩn dùng `0`); lát thuộc tính chiếm **37,43 %** test v2
> mà ba bộ chuẩn không có lát; Quy tắc E cấm so chéo cột KHÔNG-cạnh.

| Model | nhóm | n | MRR | H@1 | H@3 | H@10 |
|---|---|--:|---:|---:|---:|---:|
| **MENTOR** (A3′+L, λ=8, ours) | tổng | 5274 | **41.20** | **33.66** | 45.48 | **53.93** |
|  | thường | 3300 | **45.58** | **39.05** | 48.15 | **57.91** |
|  | lát | 1974 | 33.89 | 24.65 | 41.00 | 47.28 |
| MENTOR − L (ablation) | tổng | 5274 | 40.89 | 33.31 | 45.44 | 53.38 |
|  | thường | 3300 | 45.12 | 38.43 | 48.02 | 57.12 |
|  | lát | 1974 | 33.83 | 24.74 | 41.13 | 47.13 |
| MyGO (PhoBERT đã tách từ) | tổng | 5274 | 37.76 | 29.60 | 42.48 | 51.77 |
|  | thường | 3300 | 41.48 | 35.07 | 43.85 | 53.42 |
|  | lát | 1974 | 31.53 | 20.47 | 40.19 | 49.00 |
| MoMoK | tổng | 5274 | 32.56 | 26.07 | 35.40 | 45.05 |
|  | thường | 3300 | 34.96 | 29.75 | 36.59 | 45.02 |
|  | lát | 1974 | 28.54 | 19.93 | 33.42 | 45.09 |
| TSAM | tổng | 5274 | 39.42 | 31.98 | 43.17 | 52.45 |
|  | thường | 3300 | 44.35 | 37.82 | 46.86 | 56.40 |
|  | lát | 1974 | 31.17 | 22.22 | 37.02 | 45.85 |
| AdaMF-MAT (our features; last ckpt, 1000 ep), 3 seed | tổng | 5274 | 35.57 | 26.80 | 41.32 | 50.53 |
|  | thường | 3300 | 37.20 | 30.32 | 40.21 | 50.05 |
|  | lát | 1974 | 32.84 | 20.91 | 43.16 | 51.33 |
| DSoM re-impl. (approx.), 3 seed | tổng | 5274 | 38.59 | 31.01 | 43.28 | 51.10 |
|  | thường | 3300 | 40.87 | 34.91 | 42.91 | 51.89 |
|  | lát | 1974 | 34.78 | 24.48 | 43.90 | 49.78 |
| TuckER (structure-only) | tổng | 5274 | 36.80 | 29.67 | 40.87 | 49.11 |
|  | thường | 3300 | 39.22 | 33.63 | 41.11 | 49.60 |
|  | lát | 1974 | 32.76 | 23.05 | 40.48 | 48.29 |
| prior tần suất (lát) † | lát | 1974 | 33.55 | 22.19 | 42.15 | 50.00 |

- `tổng` = 5 274 khoá · `thường` 3 300 · `lát` 1 974 (hai chiều lát: xem `KET_QUA_CUOI.md` §2.1 — `lát·head` vs `lát·tail` lệch **66×**, KHÔNG gộp).
- **Đậm = A3′\* λ8 (ours) vượt MỐC-NGOÀI-MẠNH-NHẤT TỪNG Ô với KTC 95 % qua seed trên 0** (ghép cặp per-key, `t` df=2, 3 seed; TIN_261 đ.1 — tập mốc {MyGO+seg, MoMoK, DSoM re-impl., AdaMF, TSAM}) — **6/12 ô** (tổng·{MRR,H@1,H@10} + thường·{MRR,H@1,H@10}, nơi mốc mạnh nhất là **TSAM** mà A3′\* vượt KTC>0; **tổng·H@3 + thường·H@3 + mọi lát**: A3′\* KHÔNG vượt mốc-mạnh-nhất KTC>0 ⇒ không đậm). Số ô mỗi mốc có Δ KTC>0: **MyGO+seg** 10/12 · **MoMoK** 10/12 · **DSoM re-impl.** 0/12 · **TSAM** 9/12. ⚠️ Mốc CHƯA đo KTC per-key (ô nào mốc đó mạnh nhất thì để pending, không đậm): AdaMF. ⚠️ **KHÔNG dùng AND-mọi-mốc** (0/12): DSoM re-impl. NHIỄU (Δ ôm 0 dù mean thấp hơn MyGO+seg) sẽ veto oan — chỉ xét nó ở ô nó THỰC mạnh nhất. Trung bình cao hơn **không đủ** để đậm (LUẬT #32).
- **Nhóm `lát`: A3′\* KHÔNG dẫn** (đậm 0/4) — A3′\* lát MRR 33,89 chỉ nhỉnh TuckER 32,76 / MyGO 31,53 / MoMoK 28,54, nhưng KTC vs mốc-mạnh-nhất KHÔNG rời 0 ⇒ không đậm. Headline v2 = **A3′\* λ8** (chọn trên valid, BC118 §9); **nền\* (MENTOR − L) là ablation**; **hàng cổng OOF v2 KHÔNG in** (cổng < A3′).
- 🔴 **† Hàng `prior tần suất (lát)` = 33,55** (gộp hai chiều để tham chiếu; tần suất quan hệ trên train, type-constrain) — **nằm GIỮA dải model lát** (MoMoK 28,54 – MENTOR 33,89) ⇒ \*\*lift (model − prior) ≈ 0\*\*. Tách hai chiều: **head 1,67 / tail 65,43**; ở chiều tail prior **vượt mọi model TRỪ bản tái hiện xấp xỉ DSoM re-impl. (+2,7)** — kết luận không đổi. **Cột lát KHÔNG đậm**: lát·tail đo được \*prior giá trị\*, chưa đo phân biệt thuộc tính (xem `bang_lat_prior`). CẤM đọc lát là "grounding ảnh".
- Mốc ngoài v2: **MyGO (seg) · MoMoK · AdaMF-MAT (our features) · DSoM re-impl. · TuckER** (TSAM khi GPU xong). Không bài gốc nào đo bộ này — chạy lại nguyên bản (`baselines/`). 🔵 **lát·H@10 cao nhất = AdaMF 51,33** (> DSoM 49,78, A3′\* 47,28) — ở nhóm lát, nơi A3′\* không dẫn; AdaMF tổng/thường vẫn dưới MyGO+seg nên không đổi tập ô đậm của A3′\*.
- **Chẻ CÓ/KHÔNG-cạnh × lát/thường (luật cạnh v2 tiếng Việt, TIN_229 đ.2):** phụ lục ở `baocao/BC118_CANH_V2_VI.md` §7 (4 model). Mọi model: `thường·CÓ-cạnh` > `thường·KHÔNG-cạnh`. 🔴 **Ghi chú bắt buộc:** cạnh v2 ViMMKG2 **tập trung cao** — hub 'Việt Nam' = **14,4 %** mọi cạnh, top-10 đích **28,4 %**; **lát ∩ CÓ-cạnh = 8** (ô `lát·CÓ-cạnh` n=16 = 8×2 chiều) **không đủ mẫu, chỉ báo n** (BC118 §1/§5b).
- sinh 2026-10-16 bởi `scripts/v3/bang_bai_chinh.py`
