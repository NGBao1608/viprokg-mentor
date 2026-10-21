<!-- SINH bởi TAY-M (TIN_262 A) — (e) CHE: rớt do che tên đáp án, 4 bộ. Nguồn: BC123b-2 §3. -->

# Bảng (e) CHE — rớt MRR (điểm %) khi che tên đáp án trên cấu hình headline (3 seed, TEST)

> Nguồn số: `baocao/BC123b2_CHE_V2_LAN2.md` §3 (W/Y/DB + v2 lần 2, dụng cụ ĐÚNG, khớp bit
> `che2_kq.json`). Cột AdaMF-MAT = `--` (xem footnote). Dự đoán khai trước: che dồn CÓ-cạnh.

| bộ | rớt tổng che-tên | rớt tổng ng.nhiên § | CÓ-cạnh | KHÔNG-cạnh | tỷ che-tên | tỷ ng.nhiên | %token che | ngưỡng | ĐẠT |
|---|--:|--:|--:|--:|--:|--:|--:|:--:|:--:|
| MKG-W | 10,64 | 3,67 | 29,06 | 2,43 | **11,98×** | 1,28 | 4,6 | ≥5× | ✅ |
| MKG-Y | 10,30 | 2,90 | 27,39 | 1,84 | **14,90×** | 0,99 | 3,0 | ≥5× | ✅ |
| DB15K | 7,52 | 3,80 | 19,15 | 3,04 | **6,31×** | 0,93 | 8,7 | ≥5× | ✅ |
| **ViMMKG 2.0** (lần 2) | 10,78 | 23,22 | 17,43 | **8,46** | **2,06×** | 1,04 | **23,2** | ≥3× | 🔴 |

> **(i)–(iii) khai trước, kết quả: 3/4 · 3/3 · ĐÚNG** — (i) rớt ngẫu < che-tên 3/4 (SAI ở v2);
> (ii) tỷ ngẫu ∈[0,7;1,5] 3/3; (iii) v2 KHÔNG-cạnh ngẫu ≥50% ĐÚNG (pct_L 99,35, L fix TIN_270).

§ **che ngẫu nhiên** = đối chứng (3 seed, BC134): che cùng %token mỗi mô tả nhưng KHÔNG chạm token
tên. **Tỷ CÓ/KHÔNG ngẫu ≈1** (W 1,28 / Y 0,99 / DB 0,93) — KHÔNG dồn CÓ-cạnh như che-tên (6–15×)
⇒ tập trung CÓ-cạnh là của **kênh TÊN**. v2: rớt ngẫu 23,22 > che-tên 10,78 và KHÔNG-cạnh ngẫu 22,96 ≫ 8,46 ⇒ tỷ 2,06× là **"mất ngữ cảnh"** (pct_L 99,35).

- **Đọc:** 3 bộ chuẩn — che tên **dồn mạnh vào CÓ-cạnh** (tỷ 6,3–14,9×, đạt ngưỡng), bằng chứng
  model dựa shortcut nhắc-tên. **ViMMKG 2.0**: che vẫn dồn CÓ-cạnh (17,43 > 8,46) nhưng tỷ **2,06×
  DƯỚI ngưỡng ≥3×** — vì **%token che v2 = 23,2** (gấp 2,7–7,7× ba bộ: che cắt cả context, không chỉ
  tên) và kênh L/nhắc-tên **chuyển kém sang v2** (tiếng Việt, 37 % lát). ⚠️ CẤM viết *"v2 đạt khai
  trước che"* hoặc *"che không dồn CÓ-cạnh ở v2"*.
- **AdaMF-MAT: cột che = `--` (KHÔNG ÁP DỤNG).** Kiến trúc AdaMF nạp text là **tham số HỌC**
  (`nn.Embedding.from_pretrained(...).requires_grad_(True)`), checkpoint **ghi đè** tensor text khi
  load ⇒ đổi text lúc chấm cho điểm **TRÙNG BIT** (KHOA_CHU §13 / BC129). Che-lúc-test **không thi
  hành được** trên AdaMF — `--` KHÔNG phải "không rớt".
- v2 lần 1 (che_headline, L trống, tỷ 1,48×) chỉ giữ hồ sơ "dụng cụ hỏng" (BC123 §3); bài dùng **lần 2**.
- Nguồn: `baocao/BC123b2_CHE_V2_LAN2.md` (sha ghi trong báo cáo) · `BC123.md` §3 (W/Y/DB) · AdaMF: `BC129`.
