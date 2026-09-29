# MiniCPM-o 4.5: Vietnamese audio to streaming text

Thư mục này là trainer thử nghiệm độc lập cho mục tiêu hiện tại:

```text
audio tiếng Việt -> MiniCPM-o 4.5 -> text trả lời -> TTS bên ngoài
```

Nó không train vision, TTS hoặc full-duplex. Trainer giữ nguyên độ chính xác BF16, không
quantize/QLoRA, đóng băng vision/TTS và phần lớn Whisper/Qwen3. Phần được cập nhật mặc định:

- `audio_projection_layer` đầy đủ;
- LoRA trên `q/k/v/o` attention và `gate/up/down` MLP của riêng `llm.model.layers`;
- audio encoder được đóng băng; chỉ mở các lớp trên cùng bằng cấu hình sau khi đã có baseline.

Checkpoint Hugging Face được pin theo revision trong YAML để remote code không đổi giữa hai lần chạy.

Đây là implementation riêng vì LLaMA-Factory và SWIFT hiện chưa có đường train audio input
được xác nhận cho MiniCPM-o 4.5. Nó gọi trực tiếp processor và forward của checkpoint chính thức.

## 1. Tạo environment riêng

Không cài các dependency này vào environment chạy pipeline. Trên máy A100:

```bash
cd training/minicpmo45_audio_text
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

`torch>=2.8,<2.9` là ràng buộc bắt buộc; lệnh trên không hạ Torch xuống dưới 2.8. Nếu máy dùng
wheel CUDA riêng, hãy cài `torch==2.8.0` và `torchaudio==2.8.0` từ index PyTorch phù hợp trước,
rồi chạy `pip install -e ".[dev]"`.

Kiểm tra unit test CPU:

```bash
pytest
```

## 2. Dựng manifest từ kết quả pipeline

Đầu vào là thư mục chứa các `conversation.json`, `speaker_A.wav` và `speaker_B.wav` do stage
conversation export tạo ra:

```bash
minicpmo-prepare \
  --exports-root /data/sommelier/results/10_conversation_export \
  --output-dir data/minicpmo45 \
  --tasks asr dialogue \
  --dev-ratio 0.05
```

Mặc định chỉ nhận `channels_2ch.method=strict_separation_tracks`; dữ liệu `time_gated` bị loại để
không học audio chồng/lẫn. Không có audio nào được copy: manifest lưu đường dẫn và khoảng thời gian
cần đọc. Split được hash theo `source_group`, nên các lượt từ cùng nguồn không lọt sang cả train và
dev.

Các mẫu được tạo:

- `asr`: audio một lượt nói -> transcript của lượt đó;
- `dialogue`: audio lượt người A -> text lượt trả lời kế tiếp của người B;
- `text`: replay reasoning/instruction tùy chọn để giảm suy giảm năng lực Qwen3.

Text replay là JSONL dạng:

```json
{"id":"vi-0001","source_group":"text-vi","messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

Thêm vào manifest bằng `--text-replay /data/text_replay.jsonl`. Chỉ câu trả lời cuối cùng nhận loss;
system prompt, lịch sử và audio placeholder đều bị mask.

Sampler mặc định dùng tỷ lệ mục tiêu `25% ASR / 50% dialogue / 25% text replay`. Nếu manifest chưa
có text replay, tỷ lệ được chuẩn hóa trên ASR và dialogue; trainer không tạo dữ liệu text giả.

## 3. Kiểm tra trước khi dùng GPU

Sửa ba đường dẫn trong `configs/a100_2x.yaml`, rồi chạy:

```bash
minicpmo-check --config configs/a100_2x.yaml --check-audio 100
```

Lệnh này kiểm tra schema, task count, file audio và giới hạn timestamp mà không tải model.

## 4. Smoke test một GPU

Chạy một optimizer step trước. Model tải với `init_vision=False`, `init_audio=True`,
`init_tts=False`; collator luôn cấp `vision_hidden_states=[]` để remote forward không khởi tạo
dummy vision graph trong lúc train.

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 \
  -m minicpmo_train.train \
  --config configs/a100_2x.yaml \
  --max-steps 1
```

Dùng `output_dir` mới cho mỗi smoke run. Trainer cố ý từ chối ghi vào thư mục không rỗng.

## 5. Train trên hai A100

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 \
  -m minicpmo_train.train \
  --config configs/a100_2x.yaml
```

Cấu hình mặc định là microbatch 1 mỗi GPU, gradient accumulation 16, effective batch 32. Đây là
DDP: mỗi A100 giữ một bản model BF16; không có trao đổi model giữa CPU/GPU mỗi batch. Dataloader
đọc đúng đoạn audio, resample một lần nếu cần, processor tạo mel và chuyển thẳng sang GPU.

Checkpoint có cấu trúc:

```text
runs/minicpmo45_vi_lora/checkpoint-00000100/
  adapter/       # LoRA + audio projector, safetensors
  state/         # optimizer, scheduler, RNG, distributed state
  metadata.json  # config, hash manifest, versions, số tham số
  COMPLETE
```

Resume:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 \
  -m minicpmo_train.train \
  --config configs/a100_2x.yaml \
  --resume runs/minicpmo45_vi_lora/checkpoint-00000100
```

Thử adapter bằng một file audio và nhận text streaming:

```bash
minicpmo-infer \
  --adapter runs/minicpmo45_vi_lora/checkpoint-00000100/adapter \
  --audio /data/test_vi.wav \
  --stream
```

## Giới hạn cần biết

- Đây là code thử nghiệm trên checkpoint remote-code; phải chạy smoke forward/backward trên đúng
  máy A100 trước một run dài.
- Mỗi audio target tối đa 30 giây, đúng boundary processor chính thức. Mẫu dài hơn bị loại thay vì
  cắt âm thanh hoặc hạ chất lượng.
- Không bật thinking cho SFT audio→text: chain-of-thought không được dùng làm target. Khả năng suy
  luận được bảo vệ bằng text replay có câu trả lời cuối, không train nội dung `<think>`.
- Chưa coi checkpoint là đạt chỉ vì loss giảm. Cần so WER tiếng Việt, relevance hội thoại và
  regression text-only trên dev độc lập trước khi dùng.
