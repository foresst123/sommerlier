# ASR: lane, worker PhoWhisper và log tiến độ

## Lane và khởi động
Mỗi model ASR (Qwen3, Whisper vLLM, PhoWhisper) là một *lane* có hàng đợi riêng, mọi file đang chạy cùng đẩy clip vào (`services/asr_scheduler.py`). Lane xong việc của file này thì lấy tiếp clip của file khác, không chờ lane chậm hơn.

Khi bật `stages.asr.cross_file`, các worker của stage ASR được spawn **cùng lúc** rồi lane nào có worker sẵn sàng thì chạy ngay (`LaneWorker.ready`). Lane chưa lên hiện `loading Ns` trong log; clip của nó xếp hàng chờ. Worker không lên được (thiếu GPU, thiếu model) làm **file cần nó thất bại kèm lý do**, không đi tiếp với transcript rỗng.

## PhoWhisper thành worker riêng
`stages.asr.phowhisper_workers` (0 đến 4, mặc định 0):
- `0`: PhoWhisper chạy trong tiến trình chính như trước.
- `N > 0`: N tiến trình `phowhisper_worker.py`, phân đều qua các GPU, bắt đầu từ `phowhisper_gpu`. Cả N worker lấy chung một hàng đợi của lane, nên lane chậm nhất có thể tăng tốc bằng cách tăng N.

Giao thức là line-JSON qua stdio như các worker khác; một batch đi thành một file `.npy` gồm các clip nối tiếp cùng danh sách độ dài. Bản sao động (`replica`) của PhoWhisper cũng là một worker mới trên GPU vừa rảnh.

## Log tiến độ
Toàn bộ tiến độ do một bộ ghi duy nhất (`utils/asr_progress.py`) in ra thành dòng log thường, không dùng `\r`, không có thanh tqdm:

```
[ASR] files 3/3 queued | waiting for lanes 1 | voting 1 | voted 1/3
[ASR] qwen3 164/222 (73%, 9.1/s, ~6s) | whisper 58/222+ (26% so far) | phowhisper loading 40s (0/222)
```

- `done/total`: `total` là số clip đã xếp hàng. Dấu `+` nghĩa là còn file chưa tới, tổng còn tăng.
- Số chỉ tăng; dòng chỉ in lại khi có gì đổi (trừ lane đang nạp, để giây chạy tiếp).
- Mỗi file xong in `[ASR] <tên file>: N transcripts voted in Xs`.
- Đường chạy từng file (không cross-file) dùng cùng bộ ghi, dòng có tên file.
- Log của worker vLLM/PhoWhisper chỉ ghi dòng lỗi; đặt `VLLM_LOGGING_LEVEL=DEBUG` khi cần điều tra treo.
