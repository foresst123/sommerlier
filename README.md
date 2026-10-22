# training

Branch chuyên xử lý phần **huấn luyện**, tách khỏi pipeline xử lý audio. Nhánh này
không chia sẻ lịch sử với các nhánh khác (orphan) và chỉ chứa những file cần cho huấn
luyện; đường dẫn giữ nguyên như trong repo chính để dễ chép qua lại.

## Nội dung

| Thư mục | Vai trò |
|---|---|
| `doc/` | Thiết kế và kế hoạch: hệ thống huấn luyện full-duplex, kế hoạch triển khai, mở rộng dữ liệu, kế hoạch thử nghiệm bộ tách giọng theo vòng, thử nghiệm MossFormer2, hướng cải thiện tách giọng |
| `podcast-pipeline/tools/` | Tạo dữ liệu huấn luyện bộ tách giọng: đo overlap thật, lấy mẫu overlap, dựng mixture, kiểm định |
| `podcast-pipeline/schemas/` | Các lớp dữ liệu cần để đọc cache pickle của pipeline (`collect_overlap_samples.py` unpickle `SpeechSegment`) |
| `podcast-pipeline/utils/lhotse_validate.py` | Kiểm manifest Lhotse full-duplex theo spec 2.2.0 (không phụ thuộc gì ngoài thư viện chuẩn) |
| `podcast-pipeline/tests/` | Test của các thành phần trên; `tests/data/spec_sample.jsonl` là sample 3 cut của bên spec |

## Chạy

```bash
cd podcast-pipeline
pip install numpy soundfile pytest          # lhotse chỉ cần nếu muốn kiểm load bằng lhotse
python -m pytest tests -q

# dữ liệu huấn luyện bộ tách giọng (xem docstring từng tool)
python tools/measure_real_overlap_stats.py --help
python tools/render_mix_dataset.py --help
python tools/validate_mix_dataset.py --help

# kiểm manifest full-duplex
python -m utils.lhotse_validate path/to/manifest.jsonl --max-print 0 --stats
```

Các tool đọc cache của pipeline (`--cache`), nên cần chạy ở nơi có cache hoặc trỏ tới nó.

## Ngoài phạm vi

Mọi thứ còn lại của pipeline (diarization, tách giọng thật, ASR, word alignment, chọn
hội thoại, xuất) ở các nhánh khác. Khi cần một file từ pipeline, chép nó vào đây rõ ràng
chứ không merge cả repo.
