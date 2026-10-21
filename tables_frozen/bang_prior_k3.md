<!-- SINH bởi TAY-M (TIN_262 A) — prior K=3. Nguồn: BC122 §1. Nhãn: KHÁM PHÁ · Analysis · DB15K clean-valid. -->

# Bảng prior K=3 — cổng OOF {nền(S), A3′(L), prior} vs cổng-2 {nền, A3′} (test, 3 seed)

> **Nhãn: KHÁM PHÁ · chỉ Analysis · DB15K = clean-valid (train 75 261).** Nguồn:
> `cong_oof_prior_out.zip → result_prior/kq_prior.json`; `baocao/BC122_PRIOR_K3.md` §1.
> prior = chuyên gia thứ 3 = `nền(S) + μ·log P(e|r,dir)`; cổng OOF chọn (α,μ) trên **valid**,
> test 1 lần, 3 seed (t, df=2). CHỈ đọc Δ **trong-đường** (cùng run), KHÔNG so tuyệt đối headline.

| bộ | (α,μ)* | cổng-2 tổng | cổng-3 tổng | Δ tổng [KTC 95%] | Δ KHÔNG-cạnh [KTC 95%] | pick KHÔNG-cạnh |
|---|---|--:|--:|---|---|---|
| MKG-W | (1; 0,25) | 42,50 | 42,57 | +0,068 [−0,11; +0,25] ôm 0 | +0,098 [−0,17; +0,36] ôm 0 | prior 2/3 · a3 1/3 |
| MKG-Y | (1; 0,5) | 46,90* | 46,93 | +0,021 [−0,13; +0,17] ôm 0 | +0,031 [−0,20; +0,26] ôm 0 | prior 3/3 |
| DB15K | (1; 0,5) | 41,04 | 41,50 | **+0,456 [+0,32; +0,59]** 🟢 | **+0,632 [+0,45; +0,82]** 🟢 | **prior 3/3** |

- 🟢 **prior (kênh phi-text) nâng DB15K cột KHÔNG-cạnh +0,63 (KTC>0)** — cổng OOF chọn prior cho
  nhóm KHÔNG-cạnh **3/3 seed**, tức prior vượt cả nền(S) lẫn A3′(L) ở nhóm mention-free của DB15K.
  Đây là cột đo khái quát hoá (phát hiện xuyên suốt): một kênh ngoài text/L cải thiện nhóm mà L
  không giúp. Tổng DB15K cũng +0,46 (KTC>0).
- **W/Y: prior KHÔNG thêm** (Δ ôm 0). ⚠️ **CẤM** viết *"prior cứu cột khái quát hoá"* dạng tổng quát
  — chỉ đúng **DB15K clean-valid**, 1 vòng, (α,μ) chọn valid.
- \* MKG-Y cổng-2 46,90 là **train mới trong chính run này** (≠ headline 44,80, run/dụng-cụ khác —
  BC132 §B); chỉ dùng cho Δ trong-đường, KHÔNG thay headline.
- Nguồn đầy đủ: `cong_oof_prior_out.zip → result_prior/{kq_prior,<bộ>/k3}.json` (notebook sha 43977bd0f07845ee).
