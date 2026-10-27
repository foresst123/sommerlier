# Chạy pipeline bằng ba job GPU

Ba script trong `podcast-pipeline/tools/` khởi động cả dây chuyền, mỗi script cho một job Slurm có 2 GPU.
Cả ba dùng chung `OUT` và `CACHE` trên ổ chia sẻ. Thứ tự khởi động không quan trọng: mỗi process tự chờ tín hiệu
(`.ready_<stage>`) của process trước.

## Sơ đồ

```
JOB 1 (tools/job_front.sh, 2 GPU)
  denoise ×2 ──► diarization ×2 ──► separation_plan (CPU) ──► separation ×2 ──┐
                                                                             │ signal separation
JOB 2 (tools/job_asr.sh, 2 GPU)                                              ▼
  asr (dùng cả 2 GPU) ◄───────────────────────────────────────────────────────┘
        │ signal asr
        ▼
JOB 3 (tools/job_llm.sh, 2 GPU)
  conversation_selection (refinement, speaker_relabel, Qwen3.5-9B vLLM trên 2 GPU)
        │ signal conversation_selection
        ▼
JOB 1 (cùng job, process cuối)
  all (word_alignment, conversation_exports, lọc kênh assistant)
```

`×2` nghĩa là hai bản sao của cùng stage, mỗi bản gắn một GPU của job. Hai bản sao chia file với nhau bằng
`--claim` (mỗi file có marker `.claim_<stage>.<stem>`, tạo bằng `O_EXCL`).

## Cách chạy

Chạy trong `podcast-pipeline/`, mỗi script trong một shell có quyền vào job Slurm tương ứng:

```bash
# xem bảng process -> GPU -> stage, không chạy gì
DRY=1 AUDIO=/data/vi_podcasts_20 OUT=/data/run1 GPUS="0 1" bash tools/job_front.sh

# chạy thật (mỗi lệnh một job)
AUDIO=/data/vi_podcasts_20 OUT=/data/run1 GPUS="0 1" bash tools/job_front.sh
AUDIO=/data/vi_podcasts_20 OUT=/data/run1 GPUS="2 3" bash tools/job_asr.sh
AUDIO=/data/vi_podcasts_20 OUT=/data/run1 GPUS="4 5" bash tools/job_llm.sh
```

Các script **không mở tmux**: bạn tự mở terminal (tmux của bạn) rồi chạy script, mỗi process của job được bật nền bằng
`nohup` (mỗi process một nhóm tiến trình riêng, không phụ thuộc terminal). Mặc định chạy thẳng trên máy này, **không cần
JOBID, không srun**: bạn chỉ chọn GPU bằng `GPUS`. Nếu muốn chạy trong một job Slurm cụ thể thì thêm `JOBID=<n>`, khi đó
mỗi process chạy `srun --jobid=<n> --overlap bash <script>`.

- PID từng process ghi ở `<OUT>/_resident/logs/<front|asr|llm>.pids`, log ở `<OUT>/_resident/logs/<job>_<số>_<tên>.log`
  (`tail -f` để xem).
- Dừng: `bash tools/job_stop.sh front <OUT>` (hoặc `asr`, `llm`). Chạy lại script sau đó, `--resume` làm tiếp.
- Chạy script lần nữa khi process cũ còn sống sẽ bị từ chối; `FORCE=1` dừng các process cũ rồi chạy lại.

Biến dùng chung: `AUDIO`, `OUT` (bắt buộc), `GPUS="0 1"` (số card vật lý), `JOBID` (tuỳ chọn), `CACHE` (mặc định `OUT/_resident/cache`),
`ENV_NAME=a100`, `CONDA_ENV=som`, `VARIANTS=tools/variants_production.json`, `FORCE=1` (dừng process cũ của job rồi chạy lại).

Biến riêng của `job_front.sh`:

| biến | mặc định | nghĩa |
|---|---|---|
| `REPLICAS_PER_GPU` | 1 | số bản sao mỗi stage trên mỗi GPU (tăng khi VRAM còn dư) |
| `PLAN_REPLICAS` | 1 | số bản sao của `separation_plan` (CPU); lớn hơn 1 thì dùng `--claim` |
| `SIDON_WORKERS` | 2 | `performance.stages.separation.workers_per_gpu` |

Diarization chạy `workers=1` trong mỗi bản sao, vì mỗi bản sao đã có một GPU riêng.

## GPU của từng process

`GPUS` là **số card vật lý** bạn chọn: `GPUS="2 3"` chạy trên card 2 và card 3 (mỗi process được đặt
`CUDA_VISIBLE_DEVICES` bằng đúng card đó, không phụ thuộc biến môi trường của shell đang đứng). Chỉ khi có `JOBID`
thì `GPUS` mới là vị trí trong danh sách card của job Slurm đó.

| job | process | bản sao | GPU (vị trí) | stage |
|---|---|---|---|---|
| JOB 1 | denoise | 2 | 0, 1 | `--stop-after denoise` (rf_crowd, blend50 theo variant) |
| JOB 1 | diarization | 2 | 0, 1 | `--stop-after diarization` (Nemotron, workers=1 mỗi bản) |
| JOB 1 | separation_plan | 1 | không (CPU) | `--stop-after separation_plan` |
| JOB 1 | separation | 2 | 0, 1 | `--stop-after separation` (Sidon) |
| JOB 1 | all | 2 | 0, 1 | `--stop-after all` (word_alignment, exports, lọc kênh assistant) |
| JOB 2 | asr | 1 | 0 và 1 | `--stop-after asr` (Qwen3 trên gpu_1, Whisper và PhoWhisper trên gpu_2) |
| JOB 3 | conversation_selection | 1 | 0 và 1 | `--stop-after conversation_selection` (refinement pipelined, speaker_relabel) |

Chuỗi chờ: denoise → diarization → separation_plan → separation → (asr chờ separation) → (conversation_selection
chờ asr) → all chờ conversation_selection.

## Kiểm tra

```bash
python tools/line_status.py /data/run1          # từng variant ở điểm nào, dòng cuối log mỗi process
tail -f /data/run1/_resident/logs/front_*.log    # log từng process (tên: <job>_<thứ tự>_<process>.log)
```

Script của từng process nằm ở `OUT/_resident/launch_<job>/<thứ tự>_<process>.sh`. Chạy tay một script để khởi động lại
đúng một process (ví dụ sau khi nó chết). Với `--resume` process bỏ qua variant đã xong.

## Khi một job chết

1. Xem log của process đó trong `OUT/_resident/logs/`.
2. Nếu process chết giữa chừng một file, marker `.claim_<stage>.<stem>` của file đó còn lại và bản sao khác sẽ
   không nhận file. Xoá marker đó, hoặc dùng `--reclaim-after MINUTES` (cờ của `run_variants_resident.py`, do
   nhánh H2 thêm; nhánh này chưa có cờ đó) để bản sao còn sống nhận lại marker cũ.
3. Chạy lại script của process đó trong job còn sống (hoặc job mới với `JOBID` khác).

## Lưu ý

- Hai bản sao cùng stage chạy trên hai card khác nhau. Stage không có cờ `--claim` (job 2, job 3) dùng cả hai card.
- Chưa có trên nhánh này: stage `separation_plan`, `all`, cờ `--claim` và `--reclaim-after`. Ba script viết đúng
  tên trong thiết kế; `run_variants_resident.py` phải có các stage và cờ đó trước khi chạy thật.
- Tín hiệu của stage nhiều bản sao do cờ `--signal` của từng bản ghi; bản sao nào xong trước sẽ ghi tín hiệu trước.
