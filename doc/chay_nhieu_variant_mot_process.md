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

## Dây chuyền nhiều process

Muốn các GPU rảnh không ngồi chờ: một (hoặc vài) process "producer" chạy giai đoạn 1 (music, denoise, diarization, separation) cho variant này sang variant khác,
còn process "consumer" chạy ASR cho những variant mà separation đã xong. Không dùng thread; mỗi process có model riêng và thấy một GPU qua `CUDA_VISIBLE_DEVICES`
nên dùng `--layout default`. Ba lệnh (mỗi lệnh một cửa sổ tmux; thứ tự khởi động tuỳ ý):

```bash
# tmux 1: producer trên GPU 0
CUDA_VISIBLE_DEVICES=0 python run_variants_resident.py --audio-dir DIR --variants tools/variants_placement.json \
    --out-root OUT --layout default --stop-after separation --signal separation --resume

# tmux 2: consumer trên GPU 1
CUDA_VISIBLE_DEVICES=1 python run_variants_resident.py --audio-dir DIR --variants tools/variants_placement.json \
    --out-root OUT --layout default --stop-after asr --wait-signal separation --resume --poll-seconds 15

# tmux 3 (tuỳ chọn): thêm producer, chia kế hoạch bằng --shard (cùng hoặc khác GPU)
#   producer 1: ... --shard 1/2      producer 2: ... --shard 2/2
```

Các tham số giữ nguyên giữa các process (cùng `--audio-dir/--variants/--out-root/--env/--set/--only-ids/--groups`), vì hash dùng để nhận tín hiệu tính từ chúng.

Cách bàn giao dựa vào checkpoint: producer chạy `--stop_after separation`, ghi checkpoint vào `cache/exp-<variant>_<stem>` rồi tạo `OUT/<id>/.ready_separation`
(`{hash, stop_after, time}`); nếu lỗi tạo `.failed_separation` (`{reason, hash}`). Consumer chạy `--stop_after asr` với cùng job id nên đọc lại checkpoint đó và chỉ làm phần ASR.
Consumer không chạy theo thứ tự kế hoạch: nó lấy variant sẵn sàng đầu tiên (có `.ready_` với hash khớp hash tính lại), gặp `.failed_` thì ghi `failed` ("producer failed: ...") mà không chạy,
không có gì sẵn sàng thì ngủ `--poll-seconds` (mặc định 15) rồi xem lại. `--wait-timeout PHÚT` (0 = chờ mãi, tính từ lần hoạt động cuối) quá hạn thì các variant còn chờ bị ghi `failed` ("producer never signalled").
`.ready_` có hash cũ (đổi `--set`, file đầu vào, env) bị bỏ qua.

- `--shard K/N` (K từ 1): process chỉ sở hữu các mục kế hoạch có `chỉ số % N == K-1` (chỉ số tính sau khi bỏ các variant skipped, `none` tính vào). Các shard rời nhau, hợp lại là cả kế hoạch. Mọi process phải thấy cùng danh sách skipped (cùng máy/cùng `--only-available`).
- `--signal STAGE` phải bằng `--stop-after`; `--wait-signal STAGE` phải là stage sớm hơn `--stop-after`. Thứ tự stage là `STAGE_ORDER` (bản sao `PipelineService._STOP_ORDER`, có test so khớp).
- Resume: process có `--signal` ghi `.done.<stop_after>` (marker riêng), các process khác dùng `.done`, nên marker của producer không làm consumer tưởng đã xong.
- Đánh giá denoise theo điểm dừng: producer dừng ở separation chỉ xét các điểm lọc đã chạy (pre_diarization, pre_separation), không xét `chain_pre_asr`, nên không bị coi là `degraded`.
- `variants_manifest.json` do mọi process ghi: đọc-sửa-ghi dưới `flock` trên `OUT/.manifest.lock`, thay thế nguyên tử. Mỗi process chỉ ghi mục của variant nó sở hữu (và các mục skipped), giữ nguyên mục của process khác, giữ thứ tự kế hoạch; cùng một variant thì mục của stop_after muộn hơn thắng. Mỗi mục có thêm khoá `stop_after` (các tool so sánh bỏ qua khoá thừa).
- Thư mục input mỗi process: `OUT/_resident/input_<thời gian>_<pid>`.

Giới hạn:
- Checkpoint `input_audio_quality` (cổng DNSMOS) gắn với đường dẫn tuyệt đối của file đầu vào; consumer dùng thư mục input khác producer nên nếu cổng DNSMOS bật thì nó tính lại. Profile `a100` để `audio_quality.enabled = false` nên không ảnh hưởng ở đó.
- VRAM: mỗi process nạp model riêng. Consumer nạp Qwen3 vLLM; `a100` đặt `gpu_memory_utilization: 0.5` và `kv_cache_memory_bytes` 5 GiB (Qwen3-ASR-1.7B). Ước tính (chưa đo): với `gpu_memory_utilization` là tỉ lệ tổng VRAM thì chỉ vừa 1 consumer trên card 40 GB (khoảng 20 GB mỗi cái, hai cái là chạm trần);
  nếu vLLM dùng `kv_cache_memory_bytes` thay cho tỉ lệ, mỗi consumer khoảng 10 đến 12 GB (trọng số bf16 cỡ 3,5 GB + 5 GiB KV + dư) nên có thể 2 đến 3 cái. Hãy đo bằng `nvidia-smi` trước khi chạy nhiều consumer trên một card.
- Nhiều producer trên cùng một card chia nhau tính toán của card đó (không nhanh hơn tỉ lệ thuận), và mỗi producer giữ một bản Sidon/diarizer/Roformer riêng.
- `--dry-run` in vai trò (producer/consumer/all), shard, signal/wait và danh sách variant process đó sở hữu.

## Một lệnh mở cả 6 process

```bash
cd ~/sommerlier/podcast-pipeline && git pull
bash tools/launch_line.sh <thư_mục_audio> <thư_mục_kết_quả_MỚI>
tmux attach -t line        # Ctrl+B rồi n/p đổi cửa sổ, Ctrl+B rồi D để thoát mà vẫn chạy
```
Script ghi 6 file lệnh vào `<OUT>/_resident/launch/`, mỗi process một cửa sổ `tmux`, log ở `<OUT>/_resident/logs/`.
Mỗi process tự `conda activate som`, `source tools/denoise_env.sh`, đặt `CUDA_VISIBLE_DEVICES` rồi chạy đúng tham số của nó.
Dừng: `tmux kill-session -t line`; chạy lại đúng lệnh cũ thì `--resume` làm tiếp.

**Trong Slurm:** nếu shell của bạn nằm ngoài job (`nvidia-smi` báo `No devices were found`) thì phải đặt `JOBID`, mỗi cửa sổ khi đó chạy `srun --jobid=<JOBID> --overlap --pty bash <file lệnh>`:
```bash
JOBID=18790 bash tools/launch_line.sh <thư_mục_audio> <thư_mục_kết_quả_MỚI>
```
Trên máy có Slurm, thiếu `JOBID` (và không có `SLURM_JOB_ID` trong shell) thì script từ chối chạy: một shell trên nút tính toán vẫn thấy mọi GPU của máy, kể cả GPU của job khác, nên "thấy GPU" không chứng minh gì; cửa sổ `tmux` cũng không thừa hưởng môi trường của shell đã khởi động nó nếu server `tmux` được bật ở nơi khác, vì vậy mỗi cửa sổ luôn tự `srun` vào job. `NO_SRUN=1` để bỏ qua.

Biến môi trường chỉnh được: `GPUS="0 0 1 1 0 1"` (vị trí trong **danh sách GPU của job** cho process 1..6, không phải số GPU vật lý: trong `srun` Slurm đặt `CUDA_VISIBLE_DEVICES` theo GPU của job, script chọn phần tử thứ n của danh sách đó và in dòng `[gpu] ...` ở đầu log; mặc định chia đều theo ước lượng VRAM, **chưa đo**),
`SIDON_WORKERS=2` (profile đặt 3; ít worker hơn thì dùng ít VRAM hơn), `SESSION`, `CONDA_ENV`, `VARIANTS`, `ENVNAME`,
`DRY=1` (chỉ ghi file lệnh, không chạy), `FORCE=1`. Đối số thêm sau `OUT` được chuyển cho cả 6 process, ví dụ
`--wait-timeout 240` hoặc `--accept-degraded`.

## `--music` (mặc định `off`)

Mặc định không variant nào quét hay bỏ nhạc (`steps.music_analysis=false`, `steps.music_removal=false` được ép cho mọi variant),
nên các bộ lọc thấy đúng bản ghi gốc. `--music profile` trả về hành vi của profile (a100 bỏ nhạc cả file bằng BS-RoFormer).
Hệ quả khi `off`:
- variant chỉ tắt bước nhạc (`raw`) trùng với `none`, nên bị bỏ qua và ghi lý do;
- cache mặc định đổi sang `cache_nomusic`, vì checkpoint diarization/separation không ghi lại audio nó được tính trên;
- `launch_line.sh` từ chối một thư mục kết quả đã có `variants_manifest.json` mà không tạo bằng cùng chế độ nhạc. Dùng thư mục mới.

## Lỗi Sidon `There is no item named 'version' in the archive`

Trong PyTorch 2.8 `torch.export.load` nuốt mọi `RuntimeError` của định dạng mới rồi thử định dạng cũ, và chính lần thử cũ mới báo câu trên.
Nên câu đó chỉ là lỗi phụ; lỗi gốc (thường là hết VRAM khi nạp lên GPU, hoặc file `.pt2` xuất bằng bản torch khác) bị che.
Để xem lỗi gốc: `~/sommerlier/sidon_env/bin/python -c "import torch; from torch.export.pt2_archive._package import load_pt2; load_pt2('<file .pt2 trong cache HF>')"`.
Nếu nạp trên CPU được mà trên GPU không được thì đó là thiếu VRAM: đổi `GPUS` để Sidon không chung card với nhiều process, hoặc giảm `SIDON_WORKERS`.

## Xem dây chuyền đang ở đâu

```bash
python tools/line_status.py <thư_mục_kết_quả>
```
In một bảng: mỗi variant một dòng, mỗi cột là một lần bàn giao (`.ready_<bước>`): `ok 3m` là đã bàn giao cách đây 3 phút, `FAIL` là bước đó lỗi (lý do ở dưới bảng), `.` là chưa tới. Bên dưới là dòng log cuối của từng process,
nên process nào đang chờ thì thấy nó chờ gì.

Không có `--per-file` (`PER_FILE=0`) thì bàn giao theo **cả variant**: một process chỉ báo `.ready` khi xong **tất cả file** của variant đó ở bước của nó, nên file dài nhất (ví dụ 47 phút) quyết định lúc process kế tiếp bắt đầu.
Với `--per-file` (mặc định của `launch_line.sh`) ô bảng là `ok` khi mọi file đã có dấu, `2/3` khi mới có một phần, `FAIL` nếu có file lỗi (lý do liệt kê dưới bảng, kèm tên file); ô `-` là bước không có việc cho variant đó (xem dưới).
Mã băm trong `.ready` bỏ qua các `--set performance.*` (đổi số worker không làm kết quả khác đi), nên mỗi process có thể có số worker riêng.

## Xoá cache của đúng các file này

```bash
python tools/clean_variant_cache.py --audio-dir <thư_mục_audio> --variants tools/variants_placement.json --out-root <OUT>          # chỉ liệt kê
python tools/clean_variant_cache.py --audio-dir <thư_mục_audio> --variants tools/variants_placement.json --out-root <OUT> --delete  # xoá thật
```
Chỉ xoá các thư mục `exp-<variant>_<tên file>` trong `cache/` và `cache_nomusic/` (tên dựng từ file variants và các file trong `--audio-dir`, nên cache của bộ dữ liệu khác nằm cùng thư mục không bị đụng).
Với `--out-root` còn xoá các dấu `.done*`, `.ready_*`, `.failed_*`, thư mục kết quả của các file đó và `variants_manifest.json`; thiếu bước này `--resume` thấy dấu cũ và bỏ qua variant.
Không đụng tới cache từng bước lọc nhiễu (`denoise/step_cache`, khoá theo nội dung audio) và các thư mục cache khác.

## Kiểm tra đã chạy đủ chưa

```bash
python tools/verify_line.py --audio-dir <thư_mục_audio> --out-root <OUT>
```
Với từng cặp (variant, file) đọc thẳng trên đĩa: có `02_diarization`, `03_separation`, `05_asr` và mỗi file có segment không; mỗi điểm lọc nhiễu mà variant dùng có `denoise.json` và **không có lỗi**
(lỗi nghĩa là bộ lọc đã rơi về audio gốc, khi đó variant đó không nói gì về bộ lọc); trạng thái trong `variants_manifest.json` là `ok`.
Trả về 0 chỉ khi mọi thứ đủ; ngược lại liệt kê từng chỗ thiếu. Variant bị bỏ qua có chủ đích (`raw` khi `--music off`) được báo là skipped, không tính là thiếu.
Bộ lọc không đổi gì (đầu ra giống hệt đầu vào) chỉ là cảnh báo.

## Bỏ qua bước không có việc làm

Dây chuyền: denoise (điểm 1 + nhạc) -> diarization -> denoise_pre_separation (điểm 2) -> separation -> denoise_pre_asr (điểm 3) -> asr.
Ba bước denoise là no-op với variant không có chuỗi lọc ở điểm đó: `denoise` khi variant không có `chain` **và** `--music off` (với `--music profile` bước này bỏ nhạc nên là việc thật,
trừ variant tự tắt cả hai `steps.music_*`), `denoise_pre_separation` khi `chain_pre_separation` rỗng, `denoise_pre_asr` khi `chain_pre_asr` rỗng. Các bước khác luôn là việc thật.

- **Phía chờ:** process có `--wait-signal S` đợi bước thật gần nhất ở S hoặc trước S (đi lùi dọc dây chuyền) và dùng đúng bước đó cho cả tên dấu `.ready_<bước>` lẫn mã băm. Nếu đi lùi tới đầu mà toàn no-op thì variant sẵn sàng ngay, không cần dấu.
  Ví dụ `l_p2` (không có chuỗi điểm 1, nhạc tắt) vào diarization ngay; `none` đi từ diarization thẳng vào separation, không chờ process denoise 2.
- **Phía sản xuất:** process mà bước của chính nó (`--stop-after`) là no-op với variant **không gọi run_batch** (chỉ tốn công giải mã lại audio). Nó chờ như consumer bước thật phía trước, rồi ghi `.ready_<bước>` của mình,
  `.done.<bước>` và mục manifest `ok` với lý do "nothing to do at this point", không chạy gì. Dấu vẫn đúng sự thật nên `line_status.py` vẫn thấy cả chuỗi. Các process sau bỏ qua dấu này vì nhìn thẳng bước thật trước đó.
- `--dry-run` in, cho từng variant process sở hữu, các bước no-op và bước nó chờ.

## `--per-file`: bàn giao từng file

Mặc định tắt trong `run_variants_resident.py`; `tools/launch_line.sh` bật sẵn, `PER_FILE=0` để tắt.

- Dấu theo (variant, file): `.ready_<bước>.<stem>` và `.failed_<bước>.<stem>` (cùng nội dung JSON: `hash, stop_after, time` / `reason, hash`). Không ghi dấu mức variant trong chế độ này. Mã băm là `variant_hash` như cũ.
- Một process chạy **một file mỗi lần**: `run_batch(pipeline, vargs, config, [file])`, trong cùng một variant theo thứ tự **kích thước tăng dần** (file nhỏ đi trước để tới bước sau sớm).
  Producer ghi `.ready_<bước>.<stem>` ngay khi file đó xong không lỗi, ngược lại ghi `.failed_<bước>.<stem>`; việc dọn checkpoint của file lỗi (đúng variant đó) vẫn làm theo từng file.
- Consumer lấy (variant, file) đầu tiên theo thứ tự kế hoạch rồi tới thứ tự file mà dấu của bước thật phía trước đã sẵn sàng. Dấu `.failed_` của một file chỉ đánh dấu file đó lỗi ("producer failed: ...") mà không chạy; các file còn lại của variant vẫn chạy.
  Variant được chốt (trạng thái, `.done*`, gộp manifest) đúng một lần khi mọi file đã chạy hoặc lỗi. `--wait-timeout` vẫn là thời gian rảnh liên tục, hết hạn thì các file chưa có tín hiệu bị ghi lỗi "producer never signalled".
  Process no-op chuyển tiếp từng file: dấu của file nào xuất hiện ở phía trước thì dấu của nó được ghi ngay.
- main có thể gọi scheduler nhiều lần (mỗi nhóm theo giờ); danh sách file đã thấy và lỗi của từng variant được giữ giữa các lần gọi.
- **Resume (chọn cho đơn giản):** `.done*` hợp lệ của bước đó thì bỏ qua cả variant (các dấu theo file giữ nguyên, và được ghi lại nếu thiếu). Không có `.done*` thì **chạy lại tất cả file** của variant; trước mỗi file, dấu `.ready/.failed` cũ của file đó bị xoá.
  Nhờ checkpoint nên chạy lại rẻ, nhưng dấu của file đã xong sẽ biến mất rồi hiện lại, và process phía sau có thể chạy lại file đó nếu nó vẫn đang chạy.
- Variant `degraded` (bộ lọc rơi về audio gốc) và không có `--accept-degraded`: mọi dấu `.ready` theo file của nó bị thay bằng `.failed`.
- **Đánh đổi:** chạy từng file làm mất song song giữa các file trong một lượt (ví dụ hàng đợi ASR cross-file `files_in_flight` chỉ còn một file). Muốn song song thì dùng `--shard K/N` trên process đó: song song theo variant thay vì theo file.
- `clean_variant_cache.py` đã xoá các dấu theo file (glob `.ready_*`, `.failed_*`); `verify_line.py` đọc kết quả trên đĩa nên không cần xử lý dấu.

## Worker con chạy đúng card của process cha

Mỗi process của dây chuyền chỉ thấy **một** card (`CUDA_VISIBLE_DEVICES=<card>`), nên với nó đó là `cuda:0`. Các dịch vụ worker của pipeline (Sidon, gán speaker, vLLM Qwen3, Nemotron...) khi sinh process con
luôn đặt `CUDA_VISIBLE_DEVICES=<chỉ số>` tuyệt đối, tức `0`: đó là card 0 **của job**, không phải card của process cha. Kết quả đo trên máy: ba process dặn dùng card 1 vẫn đặt Sidon, worker gán speaker và `VLLM::EngineCore` trên card 0,
card 0 đầy 98% còn card 1 trống.

`run_variants_resident.py` sửa bằng cách bọc `subprocess.Popen`: một `CUDA_VISIBLE_DEVICES` là một chữ số `i` trong môi trường của process con được đổi thành phần tử thứ `i` trong danh sách card của chính process cha
(luật CUDA bình thường: chỉ số của con tính theo danh sách của cha). Dòng `[gpu] this process owns CUDA_VISIBLE_DEVICES=1: a worker that asks for cuda index i runs on 0->1` ở đầu log xác nhận nó đang bật.
Kiểm tra: `nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader | sort` phải thấy Sidon và `VLLM::EngineCore` ở card 1 nếu process 4 và 6 được đặt ở card 1.
Process không có `CUDA_VISIBLE_DEVICES` riêng thì không bị đổi gì.


## Xem trang kết quả và nghe audio file nặng

`python -m http.server` gửi nguyên cả file và bỏ qua `Range`, nên WAV 47 phút (~135 MB) phải tải xong mới tua được. Dùng server riêng, chỉ dùng thư viện chuẩn, có `Range`:

```bash
python tools/serve_results.py --root <OUT> --audio-dir <thư_mục_audio_gốc>      # cổng 8001
# laptop: ssh -L 8001:localhost:8001 lamkd2@<dgx>     rồi mở http://localhost:8001/lanes/  và  http://localhost:8001/compare.html
```
- Trình duyệt chỉ xin đoạn đang nghe (`206 Partial Content`), nên bấm phát là nghe ngay và tua tới đâu tải tới đó; phát cần khoảng 0,4 Mbit/s.
- `--audio-dir` gắn thư mục audio gốc ở `/_audio/`. Variant không có bộ lọc nào và tắt nhạc (ví dụ `none`) không có file wav riêng,
  nên trang lanes phát bản gốc từ `/_audio/<tên file>`. Thư mục audio gốc cần để phẳng (file ngay trong thư mục, không lồng).
- Mặc định chỉ nghe trên `127.0.0.1`; `--host 0.0.0.0` mở thư mục cho bất kỳ ai tới được cổng đó.
- Không dùng WebSocket: thẻ `<audio>` của trình duyệt không đọc được WebSocket, và `Range` đã cho cùng kết quả (tải dần, tua) bằng bộ phát gốc.
