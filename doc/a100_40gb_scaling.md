# Chạy 100 giờ audio trên một A100 40GB

Ghi lại để thực hiện sau. Mọi con số đo được đều ghi rõ nguồn; con số ước
lượng cũng ghi rõ là ước lượng.

---

## 1. Xuất phát điểm: đo trên 2×T4

Từ log ngày 2026-08-22, 10 phút audio (`thu_that_thach_10m.mp3`), 200 segment:

| Bước | Thời gian | xRT | Ghi chú |
|---|---|---|---|
| Diarization | 3.7 p | 0.37× | DiariZen, 914 chunk |
| Separation (TSE) | 0.1 p | 0.01× | 2 job |
| Music removal | 0.4 p | 0.04× | PANNs + Demucs |
| ASR ×3 | 4.0 p | 0.40× | Whisper + PhoWhisper + Qwen3 |
| **LLM refinement** | **12.2 p** | **1.22×** | **49% tổng thời gian** |
| Export | 0.4 p | 0.04× | |
| **Tổng** | **20.8 p** | **2.08×** | |

Ngoài ra ~3.8 phút nạp model mỗi file (trong đó 2.3 phút là tải PANNS
checkpoint lần đầu, các lần sau có cache).

**100 giờ audio trên 2×T4 ≈ 208 giờ GPU ≈ 8.7 ngày liên tục.**

---

## 2. VRAM: tổng kiểm kê

| Model | Tham số | dtype | VRAM | Nguồn |
|---|---|---|---|---|
| DiariZen worker (WavLM-large + Conformer + wespeaker) | ~370M | fp32 | 5.15 | đo từ log |
| Qwen3-ASR worker (Qwen3-ASR-1.7B) | 1.7B | fp16 | 4.69 | đo từ log |
| Whisper large-v3 | 1.55B | fp16 | 3.10 | đo: model.bin 3.09GB |
| PhoWhisper-large | 1.55B | fp16 | 3.10 | đo: model.bin 3.09GB |
| Sidon worker (DialogueSidon) | ~250M | fp32 | 2.59 | đo từ log |
| Demucs (htdemucs) | ~42M | fp32 | 1.00 | ước lượng |
| Pyannote Embedder | ~20M | fp32 | 0.50 | ước lượng |
| PANNS Cnn14 | ~81M | fp32 | 0.35 | tính từ ckpt 312MB |
| ECAPA-TDNN | ~22M | fp32 | 0.30 | ước lượng |
| Silero VAD (ONNX) | ~1.8M | fp32 | 0.10 | ước lượng |
| **Cộng (trừ LLM)** | | | **20.88** | |

Bốn dòng ước lượng cộng lại 1.9 GB — sai số ở đó không đổi kết luận.

---

## 3. Phương án A: giữ toàn bộ model thường trú

**Đã chọn.** Không giải phóng VRAM giữa các bước, kể cả 3 worker subprocess.

Hệ quả: LLM refinement chỉ còn `40 − 20.88 = 19.1 GB`, và phải trừ tiếp dự
phòng phân mảnh. Model lớn nhất vừa được là **Qwen3-4B**.

### Vì sao không phải 7B/8B

Tính với prefix cache, prompt hệ thống 1200 token dùng chung:

| LLM | Trọng số bf16 | Tổng ở batch 32 | +13% dự phòng | 40GB? |
|---|---|---|---|---|
| **Qwen3-4B** | 8.0 GB | 34.5 GB | 39.0 GB | sát mép |
| Qwen2.5-7B-Instruct | 15.2 GB | 41.6 GB | 47.0 GB | tràn |
| Qwen3-8B | 16.4 GB | 44.8 GB | 50.6 GB | tràn |

Nếu sau này chấp nhận cho 3 worker thoát trước refinement (chúng đã xong việc,
không được gọi lại trong file đó), sẽ giải phóng 12.43 GB và **Qwen3-8B batch
32 chỉ dùng 32.4 GB**. Cái giá là ~40 giây khởi động lại worker cho file kế
tiếp. Ghi lại ở đây như phương án B, chưa chọn.

### Không có bản Qwen 6B

Các bậc kích thước:

- Qwen2.5: 0.5B → 1.5B → **3B** → **7B** → 14B → 32B → 72B
- Qwen3: 0.6B → 1.7B → **4B** → **8B** → 14B → 32B (+ MoE)

Qwen3-4B là bậc gần nhất dưới 7B.

> Tên model trên HuggingFace **chưa xác minh** (máy soạn tài liệu không có
> mạng). Xác nhận trước khi đưa vào config:
> ```bash
> python3 -c "
> from huggingface_hub import model_info
> for m in ['Qwen/Qwen3-4B','Qwen/Qwen3-8B']:
>     try: model_info(m); print(m,'OK')
>     except Exception as e: print(m,type(e).__name__)
> "
> ```

---

## 4. Bảng batch size đề xuất

Profile `a100` mới, cạnh `kaggle` và `notebook`.

| Model | kaggle (2×T4) | **a100 (40GB)** | Lý do |
|---|---|---|---|
| `refinement.batch_size` | 2 | **24** | bước chiếm 49% thời gian; 24 an toàn hơn 32 |
| `refinement.model_name` | Qwen2.5-3B-Instruct | **Qwen3-4B** | lớn nhất còn vừa ở phương án A |
| `diarizen.batch_size` | 12 | **64** | T4 bị giới hạn bởi 14.5GB, A100 không |
| `whisper.batch_size` | 16 | **48** | CT2, batch lớn giảm overhead mỗi lần gọi |
| `phowhisper.batch_size` | 16 | **48** | cùng CT2 |
| `qwen3.batch_size` | 6 | **24** | |
| `qwen3.use_flash_attention` | false | **true** | flash-attn 2 chạy tốt từ Ampere |
| `demucs.segment` | 10 | **30** | ít lần gọi hơn |
| `allow_tf32` | true | **true** | mặc định bật, no-op trên Turing |

### dtype

Tất cả để **bf16** trên A100:

- `whisper.compute_type: "bfloat16"`
- `phowhisper.compute_type: "bfloat16"`
- `qwen3.torch_dtype: "bfloat16"`
- LLM refinement: hiện hardcode `torch.bfloat16` ở
  `services/diarization_refinement_service.py:86` — nên đưa vào config.

T4 (Turing) **không có** TensorCore bf16, nên các lần chạy Kaggle đang giả lập
và chậm hơn fp16. Trên A100 (Ampere) bf16 chạy native.

### Chi tiết VRAM cho Qwen3-4B

Giữ toàn bộ 20.88 GB model khác:

| batch | trọng số | KV cache | activation | LLM | **Tổng** | +13% |
|---|---|---|---|---|---|---|
| 8 | 8.0 | 0.71 | 0.81 | 9.6 | 30.4 | 34.4 |
| 16 | 8.0 | 1.25 | 1.63 | 10.9 | 31.8 | 35.9 |
| **24** | 8.0 | 1.79 | 2.44 | 12.3 | **33.2** | **37.5** |
| 32 | 8.0 | 2.33 | 3.26 | 13.6 | 34.5 | 39.0 |
| 48 | 8.0 | 3.40 | 4.88 | 16.3 | 37.2 | 42.0 (tràn) |

**Chọn batch 24**: 37.5 GB có dự phòng, batch 32 ở 39.0 GB quá sát 40 GB.

Độ chính xác: trọng số và KV cache tính theo công thức (chính xác);
activation là ước lượng thô, có thể lệch ±40% tuỳ implementation.

---

## 5. Việc cần làm, theo thứ tự

### Đã xong

- [x] PhoWhisper đọc `batch_size` từ config của chính nó — trước đây nó lấy
      `models.qwen3.batch_size` qua `ASRService`, nên trên kaggle chạy batch 6
      thay vì 16
- [x] TF32 bật qua `allow_tf32` trong profile (mặc định true)
- [x] Ba worker được giải phóng đúng lúc (`_release_worker`)
- [x] `--keep_models` giữ model in-process giữa các file

### Còn lại

1. **Thêm profile `a100`** vào `config.json` theo bảng trên
2. **Đưa `torch_dtype` của refinement vào config** thay vì hardcode
3. **Prefix caching cho LLM** — prompt hệ thống ~1200 token giống nhau mọi
   lượt; tái dùng KV của nó tiết kiệm ~70% KV cache và phần lớn thời gian
   prefill. Cần vLLM hoặc `past_key_values` thủ công.
4. **Chỉ refine khi ROVER bất đồng** — nếu 3 model ASR nhất trí thì text đã
   đúng. Ước tính ~35% segment cần sửa thật → tiết kiệm ~65% bước nặng nhất.
   **Cần đo trước**, đừng cắt mù.
5. **`segmentation_step` về 0.10** — xem mục 6.

---

## 6. `segmentation_step`: bằng chứng ngược với trực giác

Hai lần chạy thật:

| step | hop | chunks/600s | overlap phát hiện |
|---|---|---|---|
| 0.05 | 0.80s | 750 | 0.98% |
| 0.04 | 0.64s | 914 | **0.43%** |

Hop **mịn hơn** cho overlap **thấp hơn**. Giả thuyết "giảm hop để bắt
backchannel tốt hơn" bị bác bỏ bằng số liệu. Nó chỉ chậm thêm 22%.

Đề xuất: về `0.10` (375 chunk) — nhanh hơn 60% bước diarization. Chất lượng
overlap phải giải quyết bằng cách khác (mục 7).

---

## 7. Vấn đề chất lượng còn treo

Không liên quan tới tốc độ, nhưng ảnh hưởng tới giá trị của 100 giờ dữ liệu.

**Overlap 0.43–0.98%** — podcast đối thoại thật thường 5–15%. DiariZen gán
turn liền mạch không có khoảng nghỉ, nên backchannel bị chôn trong turn người
kia và tạo overlap giả. Đây là gốc rễ của lỗi separation đã sửa ở
`07f92a3`.

Hướng chưa làm: chạy **pyannote overlapped-speech-detection** song song với
DiariZen rồi hợp nhất. Repo đã có `PyannoteEmbedder` và token HF nên hạ tầng
sẵn.

---

## 8. Quy trình triển khai

**Không chạy thẳng 100 giờ.** Chạy thử **2 giờ audio** trước để:

1. Đo xRT thật trên A100 — thay các hệ số ước lượng bằng số đo
2. Xác nhận `--keep_models` không OOM ở batch 24
3. Kiểm tra chất lượng refinement với Qwen3-4B so với Qwen2.5-3B
4. Đọc `manifest.json` và `stats.json` xem cảnh báo nào bắn

Rồi ngoại suy. Phát hiện vấn đề sau 2 giờ rẻ hơn nhiều so với sau 20 giờ.

Lệnh:

```bash
python main.py --env a100 --audio_dir /path/to/audio \
    --keep_models --max_hours 2 \
    --tse --vad --panns --ASRMoE --llm_refinement
```

Sau khi xác nhận, tăng `--max_hours` lên 20 và chạy nhiều lượt.

---

## 9. Ước lượng thời gian sau tối ưu

Hệ số A100/T4 lấy **bảo thủ** (2–3.5× thay vì 4.8× lý thuyết).

| Giai đoạn | 100h audio |
|---|---|
| 2×T4 hiện tại | 208 giờ (8.7 ngày) |
| A100, chỉ đổi phần cứng | ~72 giờ (3 ngày) |
| A100 + batch + TF32 + bf16 native | ~40 giờ |
| A100 + thêm prefix cache + refine chọn lọc | **~24 giờ (1 ngày)** |

Chưa tính khoản tiết kiệm nạp model: với ~600 file, `--keep_models` tránh
được phần lớn 3.8 phút/file, tức khoảng **38 giờ**.

**Con số 24 giờ có thể lệch ±30%.** Chạy thử 2 giờ sẽ cho số thật.
