# Gọi API thay cho Qwen ở các bước LLM

Các bước LLM dùng chung một mô hình cục bộ (Qwen): **refinement, relabel, transcript_review, persona_profile, conversation_selection**. Có thể cho tất cả gọi một API tương thích OpenAI thay cho Qwen. Khi chọn API thì **không còn nạp Qwen/vLLM cục bộ** (không worker, không trọng số), nên GPU rảnh cho các bước khác.

## Chi phí: nên chọn bước nào gọi API

Refinement gửi **một yêu cầu cho mỗi đoạn** (mỗi yêu cầu lặp lại prompt hệ thống khoảng 1.500 token), nên chiếm gần hết token: khoảng 575 nghìn token vào cho một file 283 đoạn. Review và chọn conversation chỉ gửi vài cửa sổ mỗi file. Để giảm chi phí mà không đổi chất lượng refinement, để refinement chạy Qwen cục bộ và chỉ cho các bước sau gọi API:

```bash
python main.py ... --llm_backend api --api_model <id model> --api_stages review,selection
```

`--api_stages` nhận danh sách trong `refinement, relabel, review, persona, selection` (mặc định: tất cả). Cũng đặt được trong cấu hình bằng `"stages": ["review", "selection"]` trong `models.api_llm`. Bước nào không nằm trong danh sách vẫn dùng Qwen cục bộ; nếu refinement không dùng API thì Qwen vẫn cần nạp cho riêng bước đó.

## Dùng thế nào

```bash
export OPENAI_API_KEY=...            # chỉ trong cửa sổ chạy, không ghi vào file hay repo
python main.py --env a100 --audio_dir <thư mục> --by_stage \
  --llm_backend api --api_model <id model>
```

Hoặc đặt cố định trong profile của `config.json`:

```json
"models": {
  "llm_backend": "api",
  "api_llm": {
    "model": "<id model>",
    "base_url": "https://api.openai.com/v1",
    "api_key_env": "OPENAI_API_KEY",
    "concurrency": 8,
    "timeout_seconds": 120,
    "max_retries": 3,
    "reasoning_effort": "low",
    "completion_floor": 1024,
    "stages": ["review", "selection"]
  }
}
```

Chỉ `model` là bắt buộc. `reasoning_effort` chỉ gửi đi khi được đặt (bỏ nếu model không hỗ trợ). `completion_floor` là số token trả lời tối thiểu được xin: model suy luận tiêu một phần `max_completion_tokens` để suy nghĩ nên một giới hạn nhỏ có thể không còn chỗ cho câu trả lời; mặc định là 1024 khi có `reasoning_effort`, 0 nếu không. `base_url` đổi được sang cổng tương thích OpenAI khác. Cờ dòng lệnh thắng cấu hình; mặc định là `local`, nên không có gì đổi nếu không bật.

## Không làm lại việc đã xong

- Refinement không đặt tên checkpoint theo model, nên file đã có checkpoint `refinement` được nạp lại như cũ, không gọi API. Các bước sau thì:
- Mỗi bước tính tên checkpoint từ `llm.model_name`. `ApiLLM.model_name` được đặt **bằng tên model cục bộ** (`models.api_llm.checkpoint_model_name`, mặc định là tên model của refinement) nên checkpoint cũ của review, persona, relabel và selection vẫn được nhận. File đã có kết quả thì nạp từ checkpoint, **không gọi API**; chỉ file còn dở mới được gửi đi.
- Prompt, cách chia cửa sổ, cách đọc kết quả, định dạng checkpoint và thư mục xuất đều giữ nguyên. Refinement giữ nguyên cách tách đoạn, kiểm tra trung thực (`_accept`) và điều kiện báo lỗi cả file khi quá nhiều đoạn thất bại; chỉ phần sinh chữ chuyển sang API, và bộ nhớ đệm tiền tố cục bộ (`use_prefix`) không dùng.
- Model thật được gọi nằm trong từng dòng của `llm_trace/` (`"backend": "api"`).
- Hệ quả cần biết: kết quả do API tạo sẽ nằm chung tên checkpoint với kết quả do Qwen tạo. Muốn tách riêng thì đặt `checkpoint_model_name` khác (khi đó các file đã làm sẽ bị tính lại).
- Bước chọn conversation đã đổi version prompt (`v14-concise`) nên vẫn tính lại một lần cho file chưa xuất, độc lập với việc đổi backend.

## An toàn

- Khóa đọc từ biến môi trường, không ghi log, và được che khỏi thông báo lỗi.
- Thiếu khóa, thiếu `model` hay sai tên cấu hình thì **báo lỗi ngay lúc khởi động**, không để từng file bị bỏ qua lặng lẽ.
- Tổng số yêu cầu đang bay cùng lúc không vượt `concurrency`, kể cả khi refinement xếp nhiều đoạn một lúc.
- Mỗi yêu cầu thử lại khi gặp 429/5xx/timeout (theo `Retry-After` nếu có). Yêu cầu không trả lời được thì bước đó coi như "không đổi gì" như khi Qwen không trả lời.
- Chi phí: số token mỗi lần chạy được cộng dồn và ghi log mỗi 25 lần gọi (`[api-llm] ... call(s) ... prompt + completion tokens`).

## Chưa kiểm chứng

- Chưa gọi API thật từ DGX (cần khóa và mạng). Nên thử 5–10 file trước, so thời gian, chi phí và chất lượng với Qwen.
- Tham số gửi đi là `max_completion_tokens` (và `reasoning_effort` nếu đặt). Model hay cổng không hỗ trợ tham số nào thì báo lỗi `HTTP 400` trong log.
- Nội dung transcript được gửi ra dịch vụ bên ngoài: kiểm tra điều khoản dữ liệu và bản quyền nguồn trước khi chạy hàng loạt.
