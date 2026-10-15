# Giới hạn thời gian suy nghĩ của Qwen (vLLM)

Bước chọn conversation chạy Qwen ở chế độ suy nghĩ (`thinking: true`, `max_new_tokens: 16384`). Đo từ log trace: mỗi cửa sổ sinh trung bình khoảng 10.400 token (lớn nhất 16.384, tức bị cắt), trong khi đầu ra thật chỉ khoảng 1.500–2.000 token. Tốc độ một chuỗi khoảng 66 token/giây, gần trần băng thông bộ nhớ của model 9B bf16 trên A100, nên thời gian chủ yếu do số token suy nghĩ.

## Cách bật

Đặt trong profile (`models.refinement` của `config.json`):

```json
"refinement": { "...": "...", "thinking_budget": 4000 }
```

- `thinking_budget` là số token tối đa cho phần suy nghĩ của một yêu cầu. `0` (mặc định) là không giới hạn, hành vi như cũ.
- Chỉ áp dụng cho backend `vllm` và cho các yêu cầu bật `thinking` (chủ yếu bước chọn conversation).

## Cách hoạt động

1. Lượt đầu sinh tối đa `thinking_budget` token.
2. Câu trả lời đã kết thúc tự nhiên thì giữ nguyên.
3. Câu bị cắt khi đang suy nghĩ: chèn đoạn đóng suy nghĩ của Qwen ("Considering the limited time ... directly now.</think>") rồi cho model trả lời tiếp với số token còn lại (tối thiểu 512).
4. Câu bị cắt khi đang trả lời: sinh tiếp, không chèn gì.

Chỉ các câu bị cắt mới đi lượt thứ hai, nhờ bộ nhớ đệm tiền tố của vLLM nên phần prompt không tính lại.

## Lưu ý

- Chất lượng: câu trả lời sau khi bị buộc đóng suy nghĩ có thể kém hơn câu trả lời sau suy nghĩ đầy đủ, nhưng tốt hơn bị cắt ở 16.384 token (mất cả cửa sổ). Nên so sánh trên vài file trước khi dùng rộng (có thể dùng script so sánh số token, thời gian, tỷ lệ vùng hợp lệ trong log `llm_trace`, trường `thinking` và `completion_tokens`).
- Giá trị gợi ý để thử: 3.000–6.000. Thấp quá thì model ít suy nghĩ (nhanh hơn nhưng dễ sai thời lượng), cao quá thì ít tiết kiệm.
- Đổi `thinking_budget` không đổi tên checkpoint của bước chọn: file đã chọn xong vẫn được nạp lại, file còn dở dùng ngân sách mới.
- Cách này chỉ giảm số token sinh. Để tăng tổng thông lượng còn cần chạy nhiều chuỗi cùng lúc (ví dụ nhiều file song song).
