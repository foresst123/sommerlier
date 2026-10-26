# Thử nghiệm điểm đặt bộ lọc nhiễu (14 trường hợp)

Bổ sung cho [so_sanh_loc_nhieu.md](so_sanh_loc_nhieu.md). Câu hỏi: cùng một bộ lọc, đặt ở đâu trong pipeline thì cho hội thoại tốt nhất?

## 1. Ba điểm đặt

| Điểm | Cờ main.py (khoá profile `pipeline.*`) | Chạy ở đâu | Thư mục kết quả |
|---|---|---|---|
| P1 `pre_diarization` | `--denoise` (`denoise`) | sau tách nhạc, trước diarization | `01b_denoise/` |
| P2 `pre_separation` | `--denoise_pre_separation` | sau diarization, trước separation. Diarization vẫn nghe âm thanh của P1 | `02b_denoise_pre_separation/` |
| P3 `pre_asr` | `--denoise_pre_asr` | sau separation, trước ASR. Lọc cả waveform (phần không tách và phần đệm ngữ cảnh) lẫn từng đoạn đã tách | `04b_denoise_pre_asr/` |

Mặc định cả ba là `none`: pipeline chạy y như cũ. Mỗi điểm nhận chuỗi như `--denoise` (`a>b>c`), lọc xong phải đúng độ dài mẫu. Mỗi `denoise.json` có khoá `point`. Điểm P3 không ghi wav từng đoạn, chỉ ghi `after_denoise.wav` của waveform và số liệu đếm trong json (`arrays`, `filtered`, `skipped_short`, `segments_with_audio`).

Lỗi: một mảng nào lọc hỏng thì cả điểm đó giữ nguyên âm thanh đầu vào và ghi lỗi vào `errors` (không dừng pipeline). Mảng ngắn hơn 0,5 giây ở P3 không được lọc (backend không xử lý nổi), được đếm ở `skipped_short`, không tính là lỗi.

Lưu ý: checkpoint của separation, ASR... không chứa id chuỗi lọc. Mỗi biến thể phải có `--job_id` và `--save_path` riêng (runner làm sẵn); đừng đổi chuỗi lọc trong cùng một thư mục kết quả.

## 2. 14 trường hợp

L = `rf_crowd>rf_vocals_resurrection>rf_dereverb_anvuew` (luôn đủ ba bước, đúng thứ tự). M = `mpsenet_dns`. Trong cùng một điểm, L chạy trước M. File: `podcast-pipeline/tools/variants_placement.json`.

| id | P1 | P2 | P3 | nhóm |
|---|---|---|---|---|
| `raw` | - | - | - | control: tắt `steps.music_analysis` và `steps.music_removal`, không lọc ("audio gốc không xử lý gì") |
| `none` | - | - | - | control, baseline: pipeline hiện tại, có bước nhạc |
| `l_p1` / `l_p2` / `l_p3` | L | | | light (L ở P1 / P2 / P3) |
| `m_p1` / `m_p2` / `m_p3` | M | | | mpsenet |
| `lm_p1` / `lm_p2` / `lm_p3` | L>M tại điểm tương ứng | | | same_point |
| `l_p1_m_p2` | L | M | | light_earlier |
| `l_p1_m_p3` | L | | M | light_earlier |
| `l_p2_m_p3` | | L | M | light_earlier |

Nhãn mỗi biến thể là luồng đầy đủ, ví dụ `audio gốc → tiền xử lý → rf_crowd → rf_vocals_resurrection → rf_dereverb_anvuew → diarization → separation → mpsenet_dns → asr`.

## 3. Lệnh chạy trên DGX (một thư mục nhiều file)

```bash
cd podcast-pipeline
python tools/run_variants.py --audio-dir <thư_mục_audio> \
    --variants tools/variants_placement.json --out-root <thư_mục_kết_quả> \
    --env a100 --stop-after none --gpus 0,1 --parallel 2 --resume
```

- Phải dùng `--stop-after none` (runner mặc định `asr`, dừng trước bước tạo hội thoại). `main.py` hiểu `none` là chạy hết mọi bước.
- Xem kế hoạch trước: thêm `--list` (hoặc `--dry-run` để in lệnh). Thêm `--only-available` để bỏ qua (có ghi lý do) biến thể mà backend chưa dùng được trên máy; `--groups light,mpsenet` để chạy theo nhóm; `--only-ids l_p2,l_p3` để chạy lại vài biến thể.
- 14 biến thể x số file lần chạy pipeline đầy đủ. `--resume` bỏ qua biến thể đã có `.done`.

## 4. Verdict và "degraded"

Runner đọc `denoise.json` của từng điểm có chuỗi khác rỗng, cho từng file:

- `ok`: chạy xong, không lỗi.
- `fallback`: có lỗi, điểm đó đã giữ nguyên âm thanh vào.
- `missing`: có chuỗi nhưng không có `denoise.json` (hoặc hỏng, hoặc thiếu thư mục file).
- `suspect_noop`: không lỗi nhưng đầu ra giống hệt đầu vào (chỉ cảnh báo).

Biến thể là `degraded` nếu BẤT KỲ điểm nào ở `fallback`/`missing`: không ghi `.done` (nên `--resume` chạy lại; `--accept-degraded` để giữ), có trong bảng DEGRADED cuối lệnh, `--strict` thoát mã 3. Lý do ghi rõ điểm lỗi, ví dụ `pre_asr: mpsenet_dns: out of memory`. Một biến thể degraded không nói gì về tác dụng của bộ lọc. Chi tiết từng điểm nằm ở `denoise_report[file].points` trong `variants_manifest.json` (có thêm `chain_pre_separation`, `chain_pre_asr`).

## 5. Đọc kết quả

```bash
python tools/build_compare_data.py --manifest <out>/variants_manifest.json --out <out>/compare_data.json
python tools/make_compare_page.py <out>/compare_data.json -o <out>/compare.html
```

Trang so sánh đo thật: chỉ số âm thanh so với bản gốc, **WER so với baseline** (và so với bản gõ tay nếu có `--reference-dir`), **DER so với baseline** (khác biệt phân đoạn người nói giữa biến thể và baseline), số từ/số đoạn của ASR, verdict từng điểm. Cần nghe: WER/DER so với baseline chỉ cho biết khác bao nhiêu, không nói tốt hơn hay kém hơn; chỉ WER so với bản gõ tay mới là sai số thật.

Trang so sánh **không đếm số hội thoại**. Số hội thoại phải tự đếm trong từng biến thể: `<out>/<id>/<file>/07d_conversation_selection/selection.json` (vùng được chọn) và `09_conversation_exports/exports.json` (bản đã xuất). Cũng chưa có DER so với nhãn thật.

## Trang xem từng variant theo speaker

Mỗi file audio một trang: mỗi variant một khối, mỗi speaker một dòng sóng (đoạn của speaker đó tô màu), text ASR nằm ngay dưới.

```bash
python tools/make_variant_lanes_page.py --manifest <thư_mục_kết_quả>/variants_manifest.json
# -> <thư_mục_kết_quả>/lanes/<tên_file>_lanes.html
```

- `--transcript latest` để xem text sau refinement / word alignment thay vì `05_asr`.
- Audio được tham chiếu bằng đường dẫn tương đối: giữ trang trong `<thư_mục_kết_quả>/lanes/`. Nếu trình duyệt chặn
  phát file cục bộ: `cd <thư_mục_kết_quả> && python -m http.server 8001` rồi mở `http://localhost:8001/lanes/...`.
- Nhãn speaker là của từng variant (diarization chạy riêng), `SPEAKER_00` ở hai variant không nhất thiết là cùng người.
