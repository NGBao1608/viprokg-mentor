<!-- TAY-M (TIN_248 đ.2, sửa TIN_249 đ.3). Bảng trần-thông-tin — Analysis, MKG-W test 3 base seed. -->
<!-- 2 hàng (nền A3′ λ0,1 · +prior). λ_align và cánh V = 2 câu ghi chú dưới bảng (số/nhãn giữ nguyên). -->
<!-- Nguồn: BC92 Bảng 1 (MKG-W, (N), 3 base seed) / BC121 §12. -->

# Bảng trần-thông-tin (information-ceiling probe) — Analysis (thang %)

Probe thử nâng **trần** cột KHÔNG-cạnh / tổng; **không** phải dòng bảng chính. **MKG-W, test, 3 base seed.** Nhãn **KHÁM PHÁ**.

| | MRR | H@1 | H@3 | H@10 |
|---|--:|--:|--:|--:|
| nền A3′ (λ=0,1) | 40.99 ±0.08 | 34.27 | 43.70 | 53.75 |
| + parametric prior (Qwen2.5-7B, K=20) [F8] | 42.98 ±0.16 | 36.50 | 45.66 | 55.21 |

- Prior = biến thể (N), **on A3′ λ=0,1 base**; bắt buộc ghi kèm *"recognition of description text cannot be excluded"*. `[F8]` = footnote F8 (KHOA_CHU §2, nguyên văn): *The DB15K probe was run under the clean-valid protocol (train 75,261; λ selected on valid_sach) and is not comparable with the original-split row. The MKG-Y cell is blank because the mention-free gain does not replicate across three base seeds (+0.0075 ± 0.0019, CI covering zero). Both MKG-Y and DB15K probe numbers are validation-split and are reported in Analysis only.*
- **Ghi chú — λ_align (ext G1, MKG-W valid):** quét λ_align không nâng trần — tổ hợp mới tốt nhất **42,29 ≤ vòng-1 42,31** (BC121 §6 / BC120 §4).
- **Ghi chú — cánh V (caption VLM 7B thay toàn bộ text, DB15K clean-valid, 3 seed):** không cho độ lợi — Δ tổng **−0,0032** [−0,0072; +0,0009], Δ KHÔNG-cạnh **−0,0072** [−0,0151; +0,0007]; **3 seed · clean-valid · KTC ôm 0 · dấu âm lặp 3/3 · EN chưa khoá** (KHOA_CHU §14 / BC108 §3.2). `V_hoanvi` chỉ loại giả thuyết caption-là-nhiễu. **CẤM** *"tương đương"* / *"ảnh có hại"* / *"ổn định"*.
- Nguồn số bảng: BC92 Bảng 1 (nền + prior (N), 3 base seed) · BC121 §12 (nhãn prior).
