# Chạy nhiều variant trong một process (model nằm sẵn trong VRAM)

`podcast-pipeline/run_variants_resident.py` import `main.py` một lần, dựng service một lần với `--keep_models`, rồi chạy lần lượt từng variant
qua `run_batch_by_stage` thật (mỗi variant là một lượt stage-by-stage đủ mọi file). Diarizer, Sidon, Roformer, Qwen3 chỉ nạp một lần.
Đường cũ `tools/run_variants.py` không đổi.

## Khi nào dùng cái nào

| | `run_variants_resident.py` | `tools/run_variants.py` |
|---|---|---|
| Nạp model | một lần cho cả đợt | mỗi variant một process, nạp lại |
| GPU | thấy cả hai card; Qwen3 một card riêng (`--layout split`) | mỗi process một card (`--gpus`) |
| Chạy song song variant | không (`--mode sequential`) | có (`--parallel`) |
| Variant có `models.*`, `outputs.*`, `music_separator`, `extra_args` | không chạy, ghi `skipped` + lý do | chạy được |

Dùng bản resident khi các variant chỉ khác nhau ở: ba điểm lọc nhiễu (`chain`, `chain_pre_separation`, `chain_pre_asr`), `steps.*`, và `pipeline.*`
mà main không đọc lúc dựng service. Còn lại dùng `tools/run_variants.py`.

## Lệnh

Chạy trong `podcast-pipeline/`:

```bash
# xem kế hoạch, argv truyền cho main và các giá trị args từng variant; không chạy gì, không import torch
python run_variants_resident.py --audio-dir /data/vi_podcasts_20 --variants tools/variants_placement.json \
    --out-root /data/denoise_cmp --env a100 --dry-run

# chạy (variant lỗi hoặc skipped không chặn variant sau)
python run_variants_resident.py --audio-dir /data/vi_podcasts_20 --variants tools/variants_placement.json \
    --out-root /data/denoise_cmp --env a100 --stop-after asr --only-available --resume

# chạy lại: variant có .done khớp hash thì bỏ qua; không còn gì để chạy thì không nạp model
python run_variants_resident.py ...cùng tham số... --resume

# trước hết dừng ở separation, sau đó chạy tiếp tới asr (checkpoint nằm trong cache theo job_id từng variant)
python run_variants_resident.py ... --stop-after separation
python run_variants_resident.py ... --stop-after asr
```

Các tuỳ chọn khác: `--only-ids a,b`, `--groups x,y`, `--cache-dir`, `--gpu-1 0 --gpu-2 1`, `--set KEY=VALUE` (lặp được, chuyển cho `main.py`,
đứng sau preset layout nên ghi đè được nó), `--accept-degraded`, `--strict`, `--skip-bootstrap`.

Ghi chú về `--resume`: file này tự giữ `OUT/<id>/.done` (hash = cài đặt hiệu lực của variant + các file đầu vào). Mỗi lần chạy tạo thư mục
input symlink mới `OUT/_resident/input_<thời gian>` nên ledger của main không bao giờ báo "xong hết". Sửa `--set` hoặc `--stop-after` thì hash đổi và variant chạy lại.
Variant `degraded` (bộ lọc rơi về audio gốc) không có `.done`, giống bản cũ.

## `--layout`

- `split` (mặc định): các model ASR trên `gpu_2`; music, denoise, diarization, separation trên `gpu_1`.
- `split_strict`: như `split` và tắt luôn worker gán speaker (`assignment_workers_per_gpu=0`), để `gpu_2` chỉ còn model ASR. Đổi lại việc gán speaker chạy trong process chính nên stage separation chậm hơn.
- `default`: không đổi gì, profile quyết định.

Preset là dict `LAYOUT_PRESETS` ở đầu `run_variants_resident.py`, mỗi khoá là một đường dẫn `--set` vào `performance.stages.*`
(xem `utils/performance_config.py`). Test `tests/test_run_variants_resident.py` kiểm preset với validator thật nên khoá sai sẽ làm test đỏ.

Giới hạn đã biết của `split`:
- Backend lọc nhiễu chạy trong process chính với `device: auto` nên dùng `cuda:0`; giữ `--gpu-1 0`.
- Ở `split`, các worker gán speaker (`assignment_workers_per_gpu`) vẫn được tạo trên cả hai card (main.py khởi động chúng ở `gpu_1` và `gpu_2`). Các stage chạy lần lượt nên chúng không tranh tính toán với ASR, chỉ chiếm một ít VRAM của `gpu_2`; muốn bỏ hẳn dùng `split_strict`.
- Nếu `--stop-after` vượt qua `asr`, LLM refinement dùng cả hai card (profile `pipeline_devices`), chung card với Qwen3 đang giữ.
- Backend lọc nhiễu tự `unload()` sau mỗi lần gọi, nên `--keep_models` không giữ chúng: chúng được nạp lại cho mỗi file.

Các giới hạn trên được in thành dòng `WARNING (--layout ...)` ngay khi bắt đầu chạy.

## Khi một file lỗi

Scheduler tự xoá checkpoint và output của file đó **trong variant đó** (`cache/exp-<id>_<tên file>` và `OUT/<id>/<tên file>`), như main làm cho lần chạy riêng,
để `--resume` tính lại từ đầu file chứ không nạp lại các stage đã xong của lần lỗi. Variant khác không bị đụng tới.

## Không hỗ trợ

- Variant cần `models.*`, `outputs.*`, `performance.*`, `music_separator`, `extra_args`, hay `pipeline.*` dùng lúc dựng service (`bss`, `asr_models`, `ASRMoE`, ...):
  `status: skipped`, `reason: set key X needs its own process (use tools/run_variants.py)`. Bật một step build-time (`steps.diarization`, `separation`, `asr`, ...)
  mà profile đang tắt cũng bị skipped.
- `--mode overlap` (chạy chồng nhiều variant bằng thread): thoát với thông báo "not implemented: the pipeline shares per-stage state".
  `PipelineService` giữ `timeline`, `noise_track`, `separation_svc.music_map`, scope giải phóng model theo stage... dùng chung nên hai variant chạy đồng thời sẽ đạp lên nhau.
  Giữa hai variant tuần tự, scheduler xoá `_prefetch_cache` của separation, `_pending_*`, `noise_track`, `timeline` để không rò sang variant sau.

## Dựng trang so sánh sau đó

Cùng định dạng `variants_manifest.json` và thư mục `OUT/<id>/<stem>/...`, nên lệnh cũ dùng nguyên:

```bash
python tools/build_compare_data.py ...
python tools/make_compare_page.py ...
python tools/make_variant_lanes_page.py ...
```
(đối số giống khi chạy bằng `tools/run_variants.py`; `OUT/_resident/` là thư mục nội bộ, chỉ chứa input symlink.)
