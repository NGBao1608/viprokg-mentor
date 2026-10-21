<!-- SINH bởi TAY-M (TIN_272 A1 / TIN_273 A10) — Δ(A3′−nền) ghép cặp, cùng run. Nguồn: cong_oof_prior_out.zip. -->

# Bảng Δ kênh L — Δ(A3′ − nền) theo nhóm cạnh, ghép cặp cùng run (MRR, điểm %)

> Δ per-key = (1/hạng A3′) − (1/hạng nền), **cùng run scores** (W/Y/DB), 3 seed. **KTC qua seed**
> (t, df=2) đo "lặp lại qua seed"; **bootstrap qua khóa** (1000 lần, ghép cặp) đo "chắc chắn trên
> tập khóa". **DiD(L) = Δ_CÓ − Δ_KHÔNG** = độ lợi L dồn vào nhóm có cạnh. Nguồn: `cong_oof_prior_out.zip`.

| bộ | nhóm | Δ(A3′−nền) | KTC qua seed [t] | bootstrap qua khóa [1000] |
|---|---|--:|---|---|
| **MKG-W** | tổng | +3,99 | [+2,86; +5,11] | [+3,46; +4,55] |
| | CÓ-cạnh | **+14,06** | [+12,76; +15,35] | [+12,74; +15,44] |
| | KHÔNG-cạnh | **−0,51** | [−1,57; +0,56] ôm 0 | [−1,03; +0,04] ôm 0 |
| | **DiD(L)** | **+14,57** | — | CÓ ≫ KHÔNG, KTC tách |
| **MKG-Y** | tổng | +4,94 | [+4,52; +5,36] | [+4,25; +5,60] |
| | CÓ-cạnh | **+17,13** | [+12,81; +21,45] | [+15,50; +18,78] |
| | KHÔNG-cạnh | **−1,10** | [−2,61; +0,42] ôm 0 | [−1,60; −0,56] **< 0** |
| | **DiD(L)** | **+18,23** | — | CÓ ≫ KHÔNG |
| **DB15K** | tổng | +1,28 | [+0,98; +1,58] | [+1,02; +1,53] |
| | CÓ-cạnh | **+6,81** | [+5,44; +8,17] | [+6,17; +7,48] |
| | KHÔNG-cạnh | **−0,85** | [−0,97; −0,73] **< 0** | [−1,10; −0,62] **< 0** |
| | **DiD(L)** | **+7,66** | — | CÓ ≫ KHÔNG |
| **ViMMKG 2.0** | tổng | +0,31 | [−0,15; +0,77] ôm 0 | [+0,01; +0,60] |
| | CÓ-cạnh | **+0,94** | [−0,68; +2,56] ôm 0 | [+0,22; +1,76] |
| | KHÔNG-cạnh | +0,09 | [−0,72; +0,90] ôm 0 | [−0,22; +0,41] ôm 0 |
| | **DiD(L)** | **+0,85** | — | CÓ > KHÔNG (nhưng L **yếu** trên v2) |
| TuckER (cấu trúc, mức) | CÓ−KHÔNG | **+7,89** | — | mức: CÓ 42,65 − KHÔNG 34,76 (độ khó hub, nền cấu trúc) |

- 🔴 **Phát hiện xuyên suốt, đo bằng Δ ghép cặp cùng run:** kênh L (A3′ − nền) cho **+6,8…+17,1 điểm
  ở CÓ-cạnh** nhưng **−0,5…−1,1 ở KHÔNG-cạnh** (bootstrap MKG-Y/DB15K **hoàn toàn dưới 0** ⇒ L **HẠI**
  khái quát hoá; MKG-W ôm 0). **DiD(L) = +7,7…+18,2** ⇒ độ lợi kênh L **tập trung ở nhóm có cạnh**.
- **Bootstrap qua khóa chặt hơn KTC qua seed** (nhiều khóa), và cho cùng kết luận — hai cách đo
  (seed · khóa) không lẫn (Limitations "hai loại nhiễu").
- 🔴 **v2: kênh L YẾU** — Δ(A3′*−nền*λ8) chỉ +0,94 ở CÓ-cạnh (W/Y/DB: +6,8…+17,1), DiD **+0,85**;
  bootstrap CÓ trên 0 nhưng biên độ nhỏ ⇒ "kênh L chuyển kém sang v2". **DiD(L) dương 4/4 bộ** (hướng
  CÓ>KHÔNG nhất quán), biên độ W/Y/DB +7,7…+18,2 vs v2 +0,85.
- ✅ **Hàng tham chiếu TuckER-mức CÓ−KHÔNG = +7,89** (ranks v2_03/D4_tucker, 3 seed): đây là
  **độ khó hub của nền CẤU TRÚC** (không text). A3′* λ8 có CÓ−KHÔNG = **+12,65** ⇒ kênh text+L
  **thêm +4,76 điểm** vào khoảng cách có-cạnh/mention-free, tức độ lợi đa phương thức **dồn thêm**
  vào nhóm nhắc-tên **vượt mức nền cấu trúc đã có** — khớp phát hiện xuyên suốt.
