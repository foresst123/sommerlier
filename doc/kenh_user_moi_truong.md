# Kênh user là môi trường thật, kênh assistant sạch

Tài liệu này ghi lại thiết kế hai kênh của bản export full-duplex, các số đo đi kèm, và các thí nghiệm còn phải làm.
Mã nằm ở `podcast-pipeline/utils/ambience_bed.py` (hợp đồng và tham số) và
`podcast-pipeline/services/conversation_export_service.py` (`_clean_assistant`, `_verify`).

Trạng thái: hợp đồng và cấu hình đã có; thân các hàm trong `ambience_bed.py` còn `NotImplementedError`.
Khi đó `_clean_assistant` ghi lỗi vào `assistant_clean.postprocess.error` và vẫn ghi file đã lọc.
Profile hiện tại trong `config.json` đã đặt `conversation_exports.user_ambience.mode = "raw_mask"`,
nên chạy thật trước khi thân hàm xong sẽ chỉ ra lỗi đó, không ra kênh đã xử lý.

## 1. Vì sao đổi

Trước đây, mỗi track giọng tách được đặt lên một track số 0. Kênh user vì thế im hẳn giữa các câu,
còn kênh assistant có số 0 giữa các câu sau khi lọc. Hai kênh đều không giống một micro thật.

## 2. Thiết kế

### Kênh user: môi trường thật, liên tục

- Vùng an toàn: dùng nguyên bản ghi (`raw`). An toàn nghĩa là không có người kia nói, không có bên thứ ba,
  không có đoạn tách hỏng, và cách tiếng người kia đủ xa theo đuôi vang đo được (bị chặn ở 400 ms).
- Vùng không an toàn (người kia nói, hoặc tách hỏng): giọng tách của user đặt lên **nền dựng**.
  Không bao giờ dùng "hỗn hợp trừ giọng kia": nó để lại bóng của giọng kia.
- Nền dựng lấy từ **khoảng không ai nói của chính bản ghi**, ưu tiên khoảng cách ±60 s quanh đoạn xuất.
  Nền không được cộng vào `raw` (sẽ thành hai phòng). Chỗ nối raw và nền là crossfade công suất đều 30 ms.
- Nền bám theo mức sàn của phòng (cửa sổ 2 s), để không có khúc nền phẳng lộ ra.

### Kênh assistant: sạch

- Không nền. Giữa các câu là số 0 thật.
- Sau chuỗi lọc: mặt nạ mềm mở quanh mỗi câu (đệm 30 ms, fade 10 ms), phần ngoài bằng 0.
- Expander: mẫu dưới −62 dBFS về đúng 0, độ dốc mượt, không rung.
- Khớp mức: RMS mỗi câu sau lọc nằm trong ±6 dB so với trước lọc.
- Khôi phục backchannel: câu ngắn hơn 1 s mà lọc làm mất (RMS sau < 0,25 × RMS trước) lấy lại từ track trước lọc.

### Nền dự phòng (fallback), theo thứ tự

1. `local`: khoảng im cục bộ trong ±60 s.
2. `file`: khoảng im bất kỳ của bản ghi.
3. `synthetic`: nhiễu tạo theo phổ của các khung yên nhất (minimum statistics).
4. `pink`: nhiễu hồng ở −65 dBFS, khi không đo được gì.

## 3. Cấu hình

`config.json` → `environments.<env>.pipeline.conversation_exports`:

```json
"conversation_exports": {
  "assistant_clean_chain": "rf_vocals_resurrection>mpsenet_dns",
  "user_ambience": { "mode": "raw_mask" }
}
```

`mode` nhận `none` (mặc định, hành vi cũ, không gọi module nào), `raw_mask` (M1), `bed_only` (M2).
Giá trị khác bị từ chối khi dựng service, không phải lúc từng đoạn.

Các tham số khác trong `AmbienceParams` (đặt trong cùng khối `user_ambience`, khóa không biết sẽ bị bỏ qua):

| Nhóm | Khóa | Mặc định |
|---|---|---|
| Khoảng im | `gap_margin_after_s` / `gap_margin_before_s` | 0,25 / 0,10 s |
| | `gap_min_s` | 0,40 s |
| | `prefer_radius_s` | 60 s |
| | `energy_reject_db` | 12 dB |
| Nền | `xfade_ms` | 300 ms |
| | `floor_window_s` | 2 s |
| | `pink_dbfs` | −65 dBFS |
| | `seed` | 0 |
| Mặt nạ an toàn | `safe_dilate_ms` | 150 ms |
| | `reverb_tail_max_ms` | 400 ms |
| | `other_margin_ms` | 100 ms |
| | `splice_fade_ms` | 30 ms |
| | `min_overlap_s` | 0,10 s |
| Assistant | `island_pad_ms` / `island_fade_ms` | 30 / 10 ms |
| | `expander_floor_dbfs` | −62 dBFS |
| | `level_match_db` | 6 dB |
| | `restore_max_s` / `restore_ratio` | 1,0 s / 0,25 |

## 4. Các số trong conversation.json

- `assistant_clean` (chỉ có khi chuỗi lọc chạy):
  - `chain`, `chain_id`, `steps`, `errors`, `written`, `reason` như trước.
  - `postprocess` (chỉ có khi `mode` không phải `none`):
    - khi chạy được: `islands` (số câu), `restored` (số backchannel khôi phục), `level_adjusted` (số câu được khớp mức), `zeroed_fraction` (tỉ lệ mẫu bị expander đưa về 0);
    - khi lỗi: `{"error": "<loại lỗi>: <thông báo>"}`. Khi đó file `assistant_clean.wav` là đầu ra lọc chưa xử lý.
- `verification.quality` (luôn có khi có cả `user` và `assistant`; không ảnh hưởng `verification.ok`):
  - `longest_zero_run_s`: đoạn số 0 liên tục dài nhất của file user, tính trên cả file (chưa tách theo lời nói). Mục tiêu 0 khi kênh user không có khoảng im kỹ thuật.
  - `assistant_floor_dbfs`: phân vị 10% mức khung 20 ms của assistant, trên cả file. Khung số 0 nằm ở mức chặn −120 dBFS, nên khi expander đã đưa khoảng im về 0 giá trị này đọc ra −120; đó là dấu hiệu, không phải lỗi.
  - `assistant_zero_frame_share`: tỉ lệ khung số 0 của assistant, để giải thích số trên.
- `user_ambience`: **hiện code chưa ghi khóa này vào conversation.json.** Khi thêm, nên ghi các `AmbienceParams` đã dùng để một bản ghi tái lập được.

`quality_report` trong `ambience_bed.py` là số đo có span (đo đúng hơn, cần span lời nói). Nó chưa được gọi từ service.

## 5. Thí nghiệm A/B còn phải làm

- **E1: shortcut.** Nếu AUC của bộ phân loại "shortcut" dưới 0,6 thì chọn M1 (`raw_mask`); nếu không, chọn M2 (`bed_only`). Ngưỡng 0,6 là quy ước đã thống nhất, chưa kiểm chứng.
- **E2: bộ lọc khoảng hở (gap).** So sánh các giá trị `gap_margin_*`, `gap_min_s`, `energy_reject_db`: số khoảng im dùng được và tỉ lệ rơi vào tầng `local`.
- **E3: ngưỡng vá lỗ.** Chọn `xfade_ms` và `min_overlap_s`; đo số lần dùng lại một đoạn mẫu (`repeats`, mục tiêu 0) và độ gắn nối nghe được.
- **E4: chuỗi lọc assistant.** Lọc cả track hay lọc theo cụm câu; so `zeroed_fraction` và `restored` với nghe thử.

## 6. Chạy lại Lhotse

Sau khi xuất lại, manifest Lhotse cần chạy lại với ngưỡng phủ đã nới cho kênh có nền:

```bash
cd podcast-pipeline
python scripts/export_lhotse_manifest.py --root <thư mục export> --out <manifest.jsonl.gz> \
  --coverage-max-uncovered 3.0 --coverage-min-frames 5 --validate
```

`--coverage-max-uncovered` là số giây lời nói chưa có nhãn được chấp nhận trên mỗi kênh; `--coverage-min-frames`
là đoạn lời ngắn nhất (theo khung 20 ms) được tính là lời nói. Ngưỡng mặc định là 0,30 s và 3 khung.
