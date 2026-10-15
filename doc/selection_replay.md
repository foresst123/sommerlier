# Nới điều kiện chọn conversation mà không gọi lại model

Bước chọn conversation gửi từng cửa sổ transcript cho model rồi để code kiểm tra các vùng model giữ lại (độ dài, số lượt, số người nói, điểm, tách giọng, cờ review...). Báo cáo trong checkpoint `conversation_selection` lưu **nguyên văn câu trả lời của model cho từng cửa sổ** (`report.replies[].raw`, kèm `first_index`/`last_index` của cửa sổ). Vì thế khi chỉ đổi điều kiện kiểm tra trong code, có thể chấm lại đúng các câu trả lời đó thay vì hỏi model lại.

## Bật

Trong profile (`models.conversation_selection` của `config.json`):

```json
"replay_old_replies": true
```

Mặc định `false`: hành vi như cũ (đổi cấu hình hay prompt thì hỏi model lại). Khóa này **không** nằm trong mã băm cấu hình của checkpoint nên không làm đổi tên checkpoint nào.

## Cách hoạt động

- Mỗi file có thư mục checkpoint theo namespace (một thư mục cho mỗi bộ cấu hình/prompt). Khi bật, nếu namespace hiện tại chưa có kết quả, pipeline đọc các namespace **khác** của cùng file (mới nhất trước) và gom câu trả lời của từng cửa sổ.
- Cửa sổ nào có câu trả lời đọc được và không bị cắt khi suy nghĩ, và có **cùng cặp (first_index, last_index)** với cửa sổ hiện tại, thì dùng lại câu đó (log: `[selection] replaying N earlier window answer(s)`). Cửa sổ còn lại vẫn hỏi model như bình thường.
- Nếu mọi cửa sổ đều có câu trả lời cũ thì không gọi model, không nạp vLLM, không tốn token API.
- Mọi vùng vẫn đi qua bộ kiểm tra hiện tại, nên cờ review và lỗi tách giọng vẫn được áp dụng.

## Điều kiện nên nới (theo số vùng bị loại đã đo ở 78 file)

Số vùng model giữ là 3.158, chỉ 483 qua kiểm tra. Lý do loại và khóa tương ứng trong `models.conversation_selection` (profile `a100` hiện tại):

| Lý do (số vùng) | Khóa | Giá trị hiện tại | Hướng nới |
|---|---|---|---|
| `duration` (691) | `min_seconds`, `max_seconds` | 30, 120 | ví dụ 20 và 150 |
| `too_few_turns` (398) | `min_turns_total`, `min_turns_each` | 3, 1 | `min_turns_total` 2 |
| `low_score` (263) | `min_score` | 70 | 55 (mặc định của code) |
| `not_self_contained` (238) | `min_semantic` | 4 | 3 (chấp nhận vùng hơi cụt) |
| `not_two_speakers` (377), `bad_separation` (281), cờ review (176), `lasting_noise` (40), `seam` (18) | | | **nên giữ**: đây là các bộ lọc chất lượng |

Nới các điều kiện chất lượng (`min_score`, `min_semantic`) làm dữ liệu kém sạch hơn; nới độ dài và số lượt ít ảnh hưởng chất lượng hơn.

## Chạy lại cho các file đã xong

1. Sửa `config.json` (các khóa trên và `replay_old_replies: true`).
2. Bỏ dấu `done` trong ledger của các file cần chấm lại (file đã `done` bị bỏ qua khi chạy lại).
3. Chạy pipeline như thường. Các bước trước dùng lại checkpoint; chọn conversation dùng câu trả lời cũ; căn từ và xuất tính lại cho các vùng mới; manifest Lhotse được dựng lại với dung sai `outputs.lhotse`.

## Giới hạn

- Chỉ file còn checkpoint chọn conversation (namespace cũ) mới có câu trả lời để dùng lại; file nào mất checkpoint thì hỏi model lại.
- Nếu cửa sổ đổi (transcript, cấu hình cửa sổ hoặc số token đổi) thì cặp chỉ số không khớp và cửa sổ đó hỏi lại.
- Câu trả lời cũ được tạo bởi prompt và model lúc đó (Qwen hoặc API); không có gì đảm bảo prompt mới sẽ trả lời giống.
