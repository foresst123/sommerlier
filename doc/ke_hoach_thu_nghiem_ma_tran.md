# Thử nghiệm ma trận trên nhánh `polish-experiment`

## Context

Pipeline có nhiều ứng viên model ở mỗi stage nhưng **không có bằng chứng** cái nào tốt hơn cho podcast tiếng Việt hai người. Mọi quyết định gần đây dựa vào benchmark của người khác (AMI, CALLHOME, AliMeeting) hoặc số đo lẻ trên một file.

Mục tiêu: chạy tích chéo các lựa chọn, đo bằng bốn thước đo không cần tham chiếu, xuất **một file HTML tự chứa** để tải về — xem bảng số và nghe trực tiếp từng tổ hợp.

Kết quả mong đợi: biết tổ hợp nào thắng, và quan trọng hơn — **đóng góp biên của từng model**, vì tích chéo cho thấy cả tương tác.

## Bốn điều mã nguồn nói khác dự tính ban đầu

Đọc kỹ code trước khi lập kế hoạch làm lộ ra bốn điều, cái đầu nghiêm trọng nhất.

**1. Một lỗi tạo ra ma trận rỗng mà vẫn báo thành công.** `main.py:536` dựng `ProgressLedger(args.audio_dir)` — ghi `_sommelier_progress.json` **vào thư mục audio đầu vào**. Cả 192 tổ hợp đọc cùng thư mục đó, nên tổ hợp 1 đánh dấu mọi file là done, rồi tổ hợp 2…192 thấy `ledger.pending()` rỗng ở `main.py:548`, `break` ngay, và log `"Corpus complete: 3 done"` sau khi **không tính gì cả**. Đây là việc phải sửa trước tiên, nếu không toàn bộ ma trận là no-op tự báo thành công.

**2. Đánh bóng là hậu xử lý, không phải stage.** Track đã tách nằm sẵn trên đĩa dưới dạng `{tag}_mix.wav / _trackA.wav / _trackB.wav` (`separation_service.py:429-452`, gọi ở `:641`). Trục F không cần vào trong pipeline — nó là một lượt quét trên các file dump. Cây rút từ 304 lượt stage xuống **8 + 16 + 32 = 56 lượt stage mỗi file**, cộng các lượt đánh bóng rẻ trên file dump. Nếu sau này nhận đánh bóng vào production thì chỗ nối là `separation_service.py:682-684`.

**3. `--audio` đang hỏng trên nhánh này.** `exts` chỉ được gán trong nhánh `if args.audio_dir:` (`main.py:512`) nhưng được dùng vô điều kiện ở `:547`, và `ProgressLedger(None)` ném lỗi ở `:536`. Không đáng sửa cho việc này — runner luôn truyền `--audio_dir`.

**4. Ghép cặp yếu hơn tưởng.** Trục B, C, D, E đều đổi **danh mục cửa sổ** (waveform khác → VAD/diarization khác → `WindowPlanner.build` cho cửa sổ khác). Nên **so sánh theo cặp từng cửa sổ chỉ tồn tại trên trục F**, trong một tiền tố A–E cố định. Với A–E chỉ có phân bố tổng hợp theo file, không có delta theo cặp. Báo cáo **phải nói rõ điều này**, không thì người đọc sẽ hiểu chênh lệch trung vị giữa hai giá trị D là "cải thiện từng cửa sổ" trong khi hai tập cửa sổ vốn khác nhau.

## Ma trận: 7 trục × 192 tổ hợp

| Trục | Giá trị 0 | Giá trị 1 | Giá trị 2 |
|---|---|---|---|
| **A** tách nhạc | `model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt`<br>vocals SDR **10.54** (đang dùng) | `model_bs_roformer_ep_368_sdr_12.9628.ckpt`<br>vocals SDR **12.10** | — |
| **B** khử nhiễu | không | `denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt`<br>masking, không sinh tín hiệu | — |
| **C** cắt hát + nhiễu | tắt | bật | — |
| **D** diarization | `BUT-FIT/diarizen-wavlm-large-s80-md-v2`<br>≤4 người, VBx+PLDA | `BUT-FIT/diarizen-wavlm-large-s80-mlc`<br>tiếng Việt DER **12.69**, AHC, ghim 2 người | — |
| **E** enrollment memory | tắt | bật | — |
| **F** đánh bóng | không (đối chứng — **bắt buộc**) | `resemble-enhance` (pip, MIT) | `audiosr==0.0.7`, `model_name=speech` |
| **G** quét lớp hai sau SSLAM | không | `inaSpeechSegmenter` engine `smn` (MIT) | — |

### Cố định, để cô lập biến

| Thành phần | Checkpoint |
|---|---|
| Tagger stage 1 | `ta012/SSLAM_AS2M_Finetuned`, cửa sổ 1.0s hop 0.5s (fps 2) |
| Separator | `sarulab-speech/DialogueSidon`, `num_steps=100`, chunk 20s |
| Gán người nói | `speechbrain/spkrec-ecapa-voxceleb` |
| Cửa sổ tách | `utils/separation_window.py`, `POLICY_VERSION="connected-balanced-15s-v1"` |
| Chấm điểm | `models/dnsmos.py` + `sig_bak_ovr.onnx` (**chưa có trong repo**) |

**Không chạy**: ASR, captioning, refinement, export. Bốn thước đo không cần chúng, và `model_loader.py:149-154` đã từ chối nạp model ASR khi `stop_after` là `separation`. Theo `doc/a100_40gb_scaling.md`, bốn stage đó chiếm **16.6 trong 20.8 phút** mỗi file — tức ~80% chi phí.

### Trục G — vì sao là inaSpeechSegmenter

Đã tra: **không có tagger AudioSet nào mạnh hơn SSLAM.** Dòng self-supervised (BEATs, EAT, SSLAM) đẩy mAP từ 0.485 lên 0.502, và 0.502 chính là SSLAM. **Cũng không có model phát hiện giọng hát nào được phát hành** — HuggingFace trả rỗng cho `singing voice detection`, `vocal detection`, `music speech segmentation`; SVD có survey và dataset công khai (Jamendo, MIR-1K, RWC) nhưng không ai phát trọng số.

[inaSpeechSegmenter](https://github.com/ina-foss/inaSpeechSegmenter) (Viện Lưu trữ Nghe nhìn Pháp, **MIT**, **thắng MIREX 2018** hạng mục phát hiện tiếng nói): engine `smn` chia thẳng **speech / music / noise**. Chuyên gia ba lớp, không phải tagger tổng quát 527 nhãn — và lớp `noise` phục vụ luôn trục B/C. Chạy **nối tiếp sau SSLAM**: SSLAM dựng bản đồ, ina quét lại tìm đoạn bị bỏ sót, hai bản đồ hợp nhất rồi mới cắt. Phụ thuộc TensorFlow nên cần venv riêng.

### Một tín hiệu miễn phí, không thành trục

Khi bản đồ nhạc báo có nhạc **và** stem vocal có năng lượng dạng tiếng nói cùng lúc đó — tức có người phát âm trên nền nhạc, nghĩa là hát. Không tốn model nào: phép tách đã chạy, chỉ đọc lại stem. Ghi thành một cột chẩn đoán.

## Sơ đồ pipeline chéo

```
                 3 file: thu_that 10′ · lm8 22′ · vimeanh 28′
                                    │
╔═══════════════════════════════════╧════════════════════════════════════════╗
║ STAGE 1 — music            --stop_after music                              ║
║   audio → [B: khử nhiễu?] → SSLAM → bản đồ nhạc + NoiseTrack               ║
║                                 │                                          ║
║                                 ├─ [G: ina quét lại → hợp nhất bản đồ]     ║
║                                 ├─ [A: lột nhạc, 2 checkpoint]             ║
║                                 └─ [C: cắt hát+nhiễu?]                     ║
║   job_id = s1-h(A,B,C,G)                                    →  16 nhánh    ║
╚═══════════════════════════════════╤════════════════════════════════════════╝
                                    │ 16        cache_parents = —
╔═══════════════════════════════════╧════════════════════════════════════════╗
║ STAGE 2 — diarization      --stop_after diarization                        ║
║   [D: diarizen-md-v2  |  diarizen-mlc]                                     ║
║   job_id = s3-h(A,B,C,G,D)                                  →  32 nhánh    ║
╚═══════════════════════════════════╤════════════════════════════════════════╝
                                    │ 32        cache_parents = s1-h(...)
╔═══════════════════════════════════╧════════════════════════════════════════╗
║ STAGE 3 — separation       --stop_after separation                         ║
║   Sidon + ECAPA (cố định) · [E: enrollment memory?]                        ║
║   job_id = s4-h(A,B,C,G,D,E)                                →  64 nhánh    ║
║   ghi {tag}_mix/_trackA/_trackB.wav + {tag}.json + enroll   ← nguyên liệu  ║
╚═══════════════════════════════════╤════════════════════════════════════════╝
                                    │ 64 lượt stage — HẾT phần cần GPU pipeline
┌───────────────────────────────────┴────────────────────────────────────────┐
│ HẬU XỬ LÝ — không phải stage, chỉ quét file dump                           │
│   [F: không | resemble-enhance | audiosr]  → 192 tổ hợp                    │
└───────────────────────────────────┬────────────────────────────────────────┘
                                    │
┌───────────────────────────────────┴────────────────────────────────────────┐
│ ĐO — không cần GPU pipeline, chạy lại không tốn gì                         │
│   DNSMOS · tilt/floor/gated · lệch F0 (cents) · ECAPA ×3 · overlap%        │
└───────────────────────────────────┬────────────────────────────────────────┘
                                    │
                  matrix_report.html  ←  tải về, xem bảng + nghe + tick
```

**56 lượt stage mỗi file** (8 + 16 + 32) sinh ra 192 tổ hợp, vì mọi nhánh chia sẻ phần phía trên, và F không tốn lượt stage nào.

## Mã checkpoint — cụ thể

Cơ chế đã có: `CheckpointManager(cache_dir, job_id)` lưu vào `cache_dir/job_id/<stage>[/<namespace>]/result.{pkl,json}`. Hôm nay `job_id = f"{args.job_id}_{stem}"` với `args.job_id` mặc định `"default"` — **mọi tổ hợp dùng chung một cây**.

Không dùng một `job_id` dài cho mọi stage (khi đó stage 1 phải tính lại cho từng giá trị D). Thay vào đó: **job_id theo tiền tố + chuỗi cha đọc ngược lên**.

```python
# tools/matrix_axes.py  — nguồn chân lý duy nhất, runner/metrics/report cùng import
AXES = {
    "A": ["mel3005", "bsr368"],          "B": ["nodenoise", "aufr33"],
    "C": ["nocut", "cut"],               "G": ["nog", "ina"],
    "D": ["mdv2", "mlc"],                "E": ["nomem", "mem"],
    "F": ["raw", "resemble", "audiosr"],
}
OWNS = {                       # trục nào đã được quyết khi stage này chạy
    "music":       ("A", "B", "C", "G"),
    "diarization": ("A", "B", "C", "G", "D"),
    "separation":  ("A", "B", "C", "G", "D", "E"),
}
PREFIX = {"music": "s1", "diarization": "s3", "separation": "s4"}

def job_id(cell, stage):
    """Hash đúng những trục ảnh hưởng tới stage này — không hơn, không kém."""
    key = "-".join(f"{k}={cell[k]}" for k in OWNS[stage])
    return f"{PREFIX[stage]}-{hashlib.sha1(key.encode()).hexdigest()[:10]}"

def parents(cell, stage):
    """Cây cha để ĐỌC checkpoint thượng nguồn; ghi thì chỉ vào job_id của mình."""
    order = ["music", "diarization", "separation"]
    return [job_id(cell, s) for s in order[:order.index(stage)]][::-1]
```

**Sửa `utils/checkpoint.py`** (~15 dòng): thêm `parents: list[str] = ()`; `save()` luôn ghi vào `self.job_dir`; `exists()`/`load()` thử `self.job_dir` rồi lần lượt từng cha (vẫn tôn trọng `self.namespaces`). Nối ở `pipeline_service.py:229-232` qua cờ `--cache_parents`.

Cách này hơn hardlink thư mục: tường minh, test được, và miễn nhiễm với `_discard_partial` (`main.py:287-304`) xoá thư mục dùng chung.

Ba lớp gác, vì một lần đọc lẫn checkpoint cho ra ô sai mà **không báo gì**:

1. **`job_id` là hàm của đúng những trục ảnh hưởng stage đó** — không tồn tại cấu hình nào chạm được checkpoint nó không được chạm.
2. **`axes.json` cạnh mỗi cây cache**, ghi trọn bộ trục và những stage nó đã tính. Trước khi spawn, runner khẳng định `axes.json` của từng cha khớp tiền tố mong đợi; lệch thì dừng ô đó và báo to.
3. **Quét log tìm `"Music settings have changed since this file was checkpointed"`** (`pipeline_service.py:353-358`). Cảnh báo đó nghĩa là timeline đến từ cấu hình nhạc khác — lẽ ra không thể xảy ra, nên coi là lỗi cứng.

Giữ nguyên namespace `POLICY_VERSION` (`pipeline_service.py:235-239`) — nó đã bảo vệ `separation` khỏi việc đổi chính sách cửa sổ giữa thí nghiệm.

## Việc chặn — theo đúng thứ tự thực hiện

**B0. Ledger cho từng ô** — `main.py:536`. Thêm `--ledger` (mặc định `_sommelier_progress.json`), truyền vào `name=`. `os.path.join(dir, "/abs/path")` trả về đường tuyệt đối nên runner đưa path dưới thư mục ma trận. **Không sửa `utils/progress.py`.** Đây là việc số một: không sửa thì mọi thứ phía sau báo thành công mà không tính gì.

**B1. Chuỗi cha cho checkpoint** — như mục trên. Test theo khuôn `tests/test_prefix_cache.py`: con đọc được stage của cha và chỉ ghi vào của mình.

**B2. Sidecar cho dump — bổ sung giá trị cao nhất.** Mở rộng `_dump_tracks` (`separation_service.py:429`) ghi thêm, cạnh ba wav: `{tag}.json` chứa `{sr, job:[lo,hi], core:[lo,hi], speakers, probes:{spk:[[a,b]…]}, sims, accepted, rejected}`, và `{tag}_enrollA.wav/_enrollB.wav` (mảng `enroll_a/enroll_b` ở `:578-579`). Mọi thứ đã có sẵn tại chỗ gọi (`:559-566`, `:599-620`).
Không có sidecar thì **ba trong bốn thước đo hoặc bất khả thi** (ECAPA so với enrollment) **hoặc sai** (F0 và lệch phổ tính trên vùng overlap, nơi mixture chứa hai giọng nên không phải tham chiếu cho ai cả).

**B3. Trục D — checkpoint DiariZen** — `diarizen_worker.py:170,172`. Giải theo thứ tự `os.environ.get("DIARIZEN_MODEL") or diar_cfg.get("model") or "…md-v2"`, và **đọc config TRƯỚC `from_pretrained`** (hiện `_load_diarizen_config` gọi ở `:174`, tức sau). In id đã giải vào dòng `{"status":"loading"}`.
Runner sinh một file config riêng mỗi tổ hợp; với `mlc` thì bỏ `clustering_method` và đặt `min_speakers: 2, max_speakers: 2`. **Runner phải khẳng định log không có `config_ignored` nào nêu `clustering_method` hay `max_speakers`** — đó là điều phân biệt "trục D đã đổi" với "trục D được yêu cầu rồi bị bỏ".

**B4. Trục B — cái bẫy khiến nó thành no-op im lặng.** `models/bs_roformer.py:218` chọn stem đầu ra bằng `"vocal" in os.path.basename(p).lower()`. Checkpoint denoise **không sinh stem tên "Vocals"**, nên `_run` trả `None`, `separate_segment` trả nguyên đầu vào (`:257`), và **trục B thành no-op chỉ log một warning**.
Sửa: thêm tham số `stem: str = "vocals"` và đường dự phòng "lấy stem *không* khớp `noise`", cộng một dòng INFO nêu stem thực sự lấy. **Tên nhãn thật phải xác nhận bằng một lượt chạy rồi đọc tên file** — không được đoán.
Vị trí: đặt bước khử nhiễu **trước** lượt quét nhạc ở `:273`, để tagger thấy audio đã khử. Đặt sau thì bản đồ nhạc dùng chung cho cả hai giá trị B, số tiền tố stage 1 rút từ 16 xuống 8 — rẻ hơn, nhưng B không còn ảnh hưởng thứ tagger thấy, mất nửa ý nghĩa.

**B5. Trục C — cắt hát + nhiễu.** `utils/music_map.py` cần lại `SINGING` với `EXCISED` đổi được qua env (`MUSIC_MAP_EXCISE_SINGING`); `models/audioset.py` cần `SINGING_LABELS` trở lại `group_scores` **thành nhóm riêng, không gộp vào `MUSIC_LABELS`** — ghi chú đo được ở `audioset.py:83-98` là đúng. `build_maps` cần cổng `is_singing = loud_singing & (speech < SPEECH_PRESENT)` soi gương cổng SONG, để 39.5s dương tính giả có lời nói trên `vimeanh` không bị cắt.
Nửa nhiễu: thêm `NoiseTrack.excise_spans(threshold, min_span, merge_gap, pad)` gated bằng `NOISE_EXCISE`, hợp nhất vào `cuts` ở `pipeline_service.py:314`.
Giữ `CUT_SHARE_LIMIT` (`pipeline_service.py:13`) — nó từ chối cắt quá 60%; **runner phải quét cảnh báo đó và đánh dấu ô "trục C không có hiệu lực"**, không thì một no-op bị bình quân vào cột C=bật.
Đây là **đảo ngược có ý thức** chính sách "đánh dấu, không xoá" trong `utils/noise_map.py:1-20` và `doc/audio-cleanliness.md`. Thí nghiệm là cách đúng để xem lại, nhưng sau đó phải cập nhật tài liệu.

**B6. Trục F — hai môi trường tách biệt.** `tools/polish_worker.py` nhận `--backend {resemble,audiosr}`, in `{"status":"ready"}`, rồi mỗi dòng một JSON `{"in": "...npy", "out": "...npy"}` — đúng khuôn `sidon_worker.py`. Interpreter giải bằng `resolve_worker_python("resemble"/"audiosr", …)`, venv `resemble_env/` và `audiosr_env/`.

**B7. Trọng số DNSMOS.** `models/dnsmos.py:32` cần `sig_bak_ovr.onnx` — **không có trong repo** và không có trong `download_offline_weights.py`. Thêm vào đó, giải lúc chạy bằng `resolve_checkpoint(..., config_key="dnsmos")`.
**Không đánh thức `services/verification_service.py`** — nó tự ghi `NOT CALIBRATED` (`:30`), và nối nó vào sẽ đổi chính cái audio mà thí nghiệm đang đo. DNSMOS vào với vai trò **chỉ để đo**.

## Runner

`tools/run_matrix.py`, định nghĩa trục ở `tools/matrix_axes.py` để runner/metrics/report cùng một nguồn.

**Vì sao một tiến trình cho mỗi tổ hợp, không phải sửa config:** các tham số bị chốt lúc import module — `MUSIC_MAP_*` (`music_map.py:40-91`), `BSS_*` (`separation_service.py:19-43`), `BSS_MEMORY` (`enrollment_memory.py:49`), `NOISE_NOTICEABLE` (`noise_map.py:49`), `SSLAM_*` (`sslam.py:33-51`). Không thể đổi sau khi `import main`.

| Trục | Truyền bằng |
|---|---|
| A | `--music_separator <ckpt>` (CLI thắng profile ở `model_loader.py:134`) |
| B | `DENOISE_MODEL=<ckpt>` + `--steps noise_removal=on\|off` |
| C | `MUSIC_MAP_EXCISE_SINGING=0\|1`, `NOISE_EXCISE=0\|1` |
| D | `--config <config sinh riêng>` |
| E | `BSS_MEMORY=0\|1` (`main.py:262-263` chỉ đặt khi env còn trống, nên giá trị runner thắng) |
| G | `--steps ina_scan=on\|off` |

Luôn có: `--audio_dir`, `--save_path <matrix>/out/<combo>`, `--job_id`, `--cache_dir <matrix>/cache`, `--cache_parents`, `--ledger`, `--stop_after`, `--steps asr=off,captioning=off,refinement=off,export=off`, `--bss --music --vad --env a100`, `--no_review_page`.
**Không bao giờ đặt**: `--no_stage_output` (bộ đo đọc `03_separation/`), `SOMMELIER_AUDIO_CACHE=0` (cache giải mã ở `audio_service.py:25-34` khoá theo `(abspath, sr)` và được cả 56×N lượt dùng chung — tiền miễn phí).

**Resumability**: `<matrix>/state.json`, một dòng mỗi (tổ hợp, stage, file) với `pending|running|done|failed`, đường log, thời lượng. Resume = bỏ `done`, chạy lại `running` (tiến trình bị kill) và `failed` tối đa hai lần. Đây là ledger **riêng** với `ProgressLedger`: cái kia theo file trong một lượt, cái này theo ô. Giới hạn 12 giờ mỗi session Kaggle làm nó thành thiết yếu, không phải tiện nghi.

**Cô lập lỗi**: một tổ hợp hỏng không được dừng ma trận. Ghi lỗi + 50 dòng log cuối vào `state.json`, đi tiếp. Ô chưa xong hiện thành **ô trống có tooltip lý do**, không phải bỏ dòng — dòng thiếu đọc thành "không đáng chú ý", ô trống đọc thành "chưa đo".

## Bốn thước đo

Package `tools/metrics/` — `dnsmos_scorer.py`, `spectral.py`, `f0.py`, `ecapa.py`, `run_metrics.py`. Kết quả vào **một `<matrix>/metrics.jsonl`** append-only, khoá `(combo, stem, tag, track, polish)`, để chấm lại được và dựng lại báo cáo không cần chấm lại. Hai pha, hai tiến trình: chấm, rồi tổng hợp.

| Thước đo | Cách tính | Ngưỡng / ghi chú |
|---|---|---|
| **DNSMOS** | `ComputeScore` trên `mix`, `trackA`, `trackB` và từng biến thể đánh bóng; báo giá trị thô **và** `Δ vs mix` | BAK là đại diện không-tham-chiếu cho SIR (`verification_service.py:12-16` nói đúng thế). **Hai cảnh báo phải in ra, không được vùi**: DNSMOS resample về 16 kHz (`dnsmos.py:121-126`) nên **dải 48 kHz của AudioSR vô hình với nó theo thiết kế**; và nó huấn luyện trên tiếng nói một người nên BAK trên hỗn hợp hai giọng là ngoài phân bố — dùng **thứ tự**, không dùng giá trị tuyệt đối |
| **Lệch phổ** | `tilt_db` (lệch trung bình phổ dài hạn 1/3 octave, 100 Hz–8 kHz, sau khi bỏ gain), `floor_db` (p5 RMS khung so đỉnh), `gated_pct` (% khung dưới −50 dB) | Theo đúng tiền lệ `doc/thu_nghiem_mossformer2.md` để số so được với việc MossFormer2. `floor_db` nhắm giá trị **của mixture**, không phải thấp hơn. Hai số cuối là thứ bắt được "câm tuyệt đối" −75.4 dB / 48.5% của Sidon. Chỉ tính trên span `probes` của người đích |
| **Lệch F0** | `librosa.pyin`, hop 10 ms, `fmin=70 fmax=400`, **chỉ trên span đơn giọng/probe** | `median_abs_cents`, `pct_over_50_cents` (tương phản thanh điệu tiếng Việt nằm ở ~50–200 cents nên 50 là mốc báo động), `voiced_agreement` (bắt việc model sinh làm mất âm cuối mang thanh). **Đây là thước đo nhắm đúng lỗi tiếng Việt**, và cũng chậm nhất — chạy song song theo core |
| **ECAPA** | `speechbrain/spkrec-ecapa-voxceleb` nạp độc lập, `savedir=$BSS_PATH/ecapa` | Ba số: `sim_enroll` (so enrollment đã dump — cùng đại lượng cổng QC dùng, nên so được trực tiếp với `sim_percentiles` trong `report.json`), `sim_self` (đánh bóng vs chưa — trả lời "đánh bóng có đổi giọng không"), `sim_cross` (trackA vs trackB — cao nghĩa là hai track cùng một người, tức tách thất bại theo cách `qc_sim` có thể không bắt). **Không dựng `BssSeparator`** cho việc này: nó kéo theo backend Sidon và Silero VAD không cần |

Thêm `stage_output_service.segment_stats` — đã tính `overlap{seconds, pct_of_audio}`, đúng con số backchannel đang là nút thắt (0.45%, ngưỡng cảnh báo 3%).

So F0 và phổ dùng **envelope/đường cong, không phải mẫu thô**: `scripts/measure_sidon_lag.py` đã ghi lý do — VAE của Sidon không giữ pha, lệch 2 ms đủ kéo tương quan mẫu thô từ 1.0 xuống −0.64.

**Kiểm chứng bộ đo trước khi tốn một giây GPU nào.** Trên đĩa đã có **24 window triple thật** ở `kaggle 2/working/vi_audio/_final/-tse-True-…/hoahau/02_separation/audio/raw/separated/` (22 cửa sổ) và `…/thu_that_thach_10m/…` (2). Chú ý số thư mục cũ là `02_separation` chứ không phải `03_` — bộ đo nhận glob hoặc cả hai. Dump đó không có sidecar nên thước đo 3 và 4 chạy ở chế độ suy giảm, nhưng 1 và 2 kiểm được đầy đủ, **miễn phí**.

## Trục F — cơ chế

`tools/run_polish.py` quét `<matrix>/out/*/*/03_separation/audio/raw/separated/*_track{A,B}.wav`, đưa qua worker, ghi `{tag}_trackA.re.wav` / `.sr.wav` cạnh bản gốc. Idempotent theo sự tồn tại của file đích.

**Chọn mẫu cửa sổ**: giới hạn `--windows 4` mỗi tiền tố A–E, chọn tất định là bốn cửa sổ có core-overlap dài nhất (lấy từ sidecar), để phép nghe rơi vào ca khó nhất **và cùng một tập cửa sổ cho cả ba giá trị F** — đó là điều giữ cho F thật sự ghép cặp được, trục duy nhất có được điều đó.

Giữ trọn cửa sổ ~11 s, không chỉ core: DNSMOS lặp lại đầu vào ngắn hơn 9.01 s (`dnsmos.py:131-132`), nên chấm một core 2 s cho ra MOS lệch và nhiễu.

AudioSR xuất 48 kHz — giữ file 48 kHz để nghe, resample 24 kHz cho các thước đo so được với pipeline, **ghi cả hai** để báo cáo chỉ ra được chỗ lợi ích biến mất.

## Báo cáo HTML

**Không mở rộng `build_review_page`.** `make_review_page.py:31-44` đòi một transcript JSON mà thí nghiệm này cố tình không tạo (export tắt), và các dòng/bộ lọc/nhãn của nó có hình dạng ASR. Viết `tools/make_matrix_page.py`, chỉ mượn lại phần thật sự dùng được: `_data_uri` + khuôn `budget` dạng list-một-phần-tử (`:66-76`), và vòng Save/Export-JSON (`:866-893`) — cơ chế clone DOM, ghi lại `#payload`, tải lại trang chính là thứ giữ được ô tick của người duyệt, đáng copy nguyên văn.

Ba tầng:

1. **Đóng góp biên, ở trên cùng.** Với mỗi trục: trung vị thước đo khi tắt vs bật, bình quân hoá trên sáu trục còn lại, kèm n. **Đây mới là câu trả lời**; 192 dòng số thô thì không. Kèm cảnh báo không-ghép-cặp (Phát hiện 4) cạnh trục B–E.
2. **Bảng ma trận.** Mỗi dòng: bảy giá trị trục, rồi ΔBAK, ΔSIG, OVRL, `tilt_db`, `gated_pct`, F0 median |Δ| cents, `pct_over_50_cents`, `sim_enroll` p50, và số cửa sổ thử/ghép/thất-bại-theo-lý-do (từ `report.json`). Header sắp xếp được, bảy dropdown lọc, thang màu mỗi cột.
3. **Bảng nghe mỗi tổ hợp**, bung ra khi bấm vào dòng: mỗi cửa sổ một dòng với nút phát mix / trackA / trackB / trackA-đánh-bóng / trackB-đánh-bóng, thước đo của chính cửa sổ đó, và ô tick chuyển mục đích cho thí nghiệm này — đề xuất: *artefact*, *sai người*, *hỏng thanh điệu*, *thích hơn mix* — cộng ô ghi chú, phím `j/k/space`, Save + Export-JSON.

**Kích thước.** 192 × 5 clip × 4 cửa sổ × 540 KB PCM 24 kHz = **1.0 GB** — quá lớn. Hai thay đổi giải quyết:
- Chuyển **mp3 mono 48 kbps** (11 s ≈ 66 KB) bằng pydub, đúng dependency `export_service.py:79-95` đang dùng. 192 × 5 × 4 × 66 KB ≈ **127 MB**. Chọn mp3 chứ không Opus để Safari phát được.
- **Chi ngân sách byte theo chiều rộng, không theo chiều sâu.** `_data_uri` hiện trừ dần một ngân sách theo thứ tự dòng (`:79-114`), nên các tổ hợp đầu ăn hết và năm mươi tổ hợp cuối im lặng. Lặp vòng tròn — mỗi lượt một clip cho mỗi tổ hợp — để suy giảm đều. Mặc định `--max-mb 200`; dòng nào không vừa vẫn hiện đường dẫn tương đối để mở từ đĩa.

## Thứ tự thực hiện

1. **B0** ledger — không có nó thì mọi thứ sau báo thành công mà không làm gì.
2. **B2** sidecar + dump enrollment.
3. **Bốn thước đo** + `metrics.jsonl`, kiểm trên 24 dump có sẵn trong `kaggle 2/`. **Không tốn GPU**, và bắt được vấn đề trọng số DNSMOS (B7) cùng mọi lỗi đơn vị trước khi chạy đắt.
4. **B1** chuỗi cha, kèm test theo khuôn `tests/test_prefix_cache.py`.
5. **B3** trục D. Chạy một file cả hai đường, diff số speaker trong `02_diarization/stats.json`; xác nhận không có `config_ignored` cho `clustering_method`.
6. **B4** trục B, kèm bước xác nhận tên stem là bước tường minh.
7. **B5** trục C, kèm kiểm `CUT_SHARE_LIMIT` không phủ quyết.
8. **Trục G thử trước khi vào ma trận**: chạy ina một lượt trên `lm8` (nhạc thật 21%) và so bản đồ với SSLAM. **Nếu nó không tìm thêm được gì thì bỏ trục này**, tiết kiệm một nửa số tổ hợp trước khi tốn hàng chục giờ.
9. **Runner** — trước hết ma trận khói 4 tổ hợp × 1 file để kiểm chia sẻ tiền tố: khẳng định 4 lượt sinh ra **1** thư mục stage 1, **2** diarization, **4** separation.
10. **B6** worker đánh bóng + `run_polish.py`.
11. **Báo cáo HTML.**
12. Ma trận đầy đủ.

Bước 5, 6, 7 độc lập với nhau và với 10 — ranh giới tốt để chạy song song.

## Chi phí — kèm giả định

Giả định, vì đáp số đổi 3× nếu một trong số này sai: **3 bản ghi** (thu_that 10′, lm8 22′, vimeanh 28′ — trung bình ~20′); **một A100 40 GB**, profile `a100`; `num_steps: 100` như đang cấu hình; **~20 cửa sổ tách mỗi file**; chi phí cố định mỗi lượt ~25 s import torch + 40 s khởi động worker; lượt denoise mel-band ở **5× realtime** — **số này chưa đo, và là số cần đo nhất trước khi cam kết**.

| Việc | Số lượt | Mỗi lượt | Tổng |
|---|---|---|---|
| Stage 1, B=tắt | 8 | ~1.8 phút | 14 phút |
| Stage 1, B=bật (khử nhiễu toàn file chiếm phần lớn) | 8 | ~5.2 phút | 42 phút |
| Diarization (+40 s worker, ~108 s chạy) | 16 | ~3.3 phút | 53 phút |
| Separation (+40 s worker, 20 cửa sổ × ~9 s) | 32 | ~5.2 phút | 166 phút |
| **Pipeline, mỗi file** | 56 | | **~4.6 giờ** |
| Đánh bóng Resemble (4 cửa sổ × 32 tiền tố) | | 0.4× RT | ~0.3 giờ |
| Đánh bóng AudioSR | | ~3× RT | ~2.4 giờ |
| Đo (pyin chiếm phần lớn, 8 core) | | | ~0.3 giờ |
| **Mỗi file** | | | **~7.6 giờ** |

**Ba file ≈ 23 giờ trên một A100** — tôi báo là "khoảng một ngày, ±50%".

Trên 2×T4 (profile `kaggle`, diarization 0.37× RT đo trong `doc/a100_40gb_scaling.md`) nhân thêm 2.5–3× → **2–3 ngày qua ~5 session Kaggle**. Đây chính là lý do ledger theo ô là thiết yếu.

**Các cần điều khiển, lớn trước:**
- `num_steps: 100 → 25` rút khối separation 166 phút/file xuống ~60 → **tiết kiệm ~5 giờ trong 23**. Nó đổi thứ đang được đo, nên nếu kéo cần này thì kéo cho **mọi** ô và nói rõ.
- Bỏ AudioSR: **~7 giờ**.
- `--windows 2` thay vì 4: nửa chi phí đánh bóng.

## Điều tôi sẽ cắt, và vì sao

**Bỏ tích chéo đầy đủ cho trục F.** Đánh bóng là hậu xử lý trên một track; việc Resemble Enhance làm gì với track đó gần như không phụ thuộc thượng nguồn lột nhạc bằng mel-band hay bs-roformer. Tương tác F×(A–E) là bậc hai và chiếm hai phần ba ngân sách đánh bóng. Đo F trên **4 trong 32 tiền tố**, chọn để trải cả hai giá trị A, B, E → **44 ô đo thay vì 192**, báo F như đóng góp biên, và dồn thời gian tiết kiệm được vào **thêm bản ghi** — nơi phương sai thật nằm. Nếu 12 ô đó cho thấy F tương tác với gì thì mở rộng sau.

**AudioSR là trục yếu nhất, tôi sẽ cân nhắc bỏ hẳn.** Toàn bộ giá trị của nó là mở rộng dải trên ~12 kHz. Pipeline chạy 24 kHz (Nyquist 12 kHz), ASR 16 kHz, DNSMOS resample về 16 kHz — nên **ba trong bốn thước đo không thể thấy** thứ AudioSR thêm vào, theo thiết kế, còn thước đo thứ tư (lệch phổ so mixture 24 kHz) sẽ tính nó là **lệch**, tức là hỏng. Nó cũng đắt nhất trong ma trận. Tài liệu của chính bạn đã đo: năng lượng trên 8 kHz chỉ **0.0075%**, cắt ở 13.5 kHz. Nếu giữ, lời tuyên bố trung thực duy nhất là "nghe file 48 kHz" — nên thu hẹp nó vào vài clip để nghe và **để ngoài bảng số**, chứ không in ra những con số không thể động.

**Trục D sẽ không cho kết quả ghép cặp, và có thể không cho kết quả hợp lý.** `mlc` ghim `min=max=2` với `AgglomerativeClustering` không PLDA. Podcast thường có giọng thứ ba (đồng dẫn, clip, người gọi vào). Ép 2 người phân bổ lại audio đó lên hai cụm nó cho phép → đổi tập segment → đổi overlap nào tồn tại → đổi danh mục cửa sổ. Nên **không gì phía dưới D so được theo từng cửa sổ**. Giữ trục (một diarizer tinh chỉnh tiếng Việt đáng biết) nhưng báo nó thành **bảng riêng theo file**: số người tìm được, số segment, tổng giây overlap, cửa sổ thử, cửa sổ ghép. Xét nó bằng trung vị DNSMOS trên hai tập cửa sổ khác nhau là sai loại.

**Không nối `verification_service.py`.** Nó sẽ đổi chính cái audio thí nghiệm đang đo, bằng ngưỡng mà source của nó tự đánh dấu `NOT CALIBRATED`. Hiệu chuẩn nó là dự án tiếp theo tốt, và `doc/huong-cai-thien-tach-giong.md` §4a đã mô tả cách đúng (hỗn hợp tổng hợp từ `clean_segments` + `mine_enrollments` ở SNR biết trước, cho SI-SDR thật để hồi quy ECAPA-sim). Đó là một ngày dùng tốt hơn 192 ô — nhưng là **thí nghiệm khác**, còn bạn đang yêu cầu cái này.

## File sẽ sửa / thêm

**Sửa**: `main.py` (`--ledger`, `--cache_parents`), `utils/checkpoint.py` (chuỗi cha), `services/separation_service.py` (sidecar + enroll dump), `diarizen_worker.py` (model id từ config, đọc trước `from_pretrained`), `models/bs_roformer.py` (chọn stem theo tham số, không theo chuỗi `"vocal"`), `utils/music_map.py` (`SINGING` có ngưỡng riêng), `models/audioset.py` (`SINGING_LABELS` thành nhóm riêng), `utils/noise_map.py` (`excise_spans`), `services/music_service.py` (bước khử nhiễu), `services/pipeline_service.py` (nối khử nhiễu trước tagger; hợp nhất cut nhiễu; truyền `cache_parents`), `config.json` (`models.diarizen.model`, `models.denoise`, `models.dnsmos`, steps `noise_removal`/`ina_scan`), `download_offline_weights.py` (`sig_bak_ovr.onnx`).

**Thêm**: `tools/matrix_axes.py`, `tools/run_matrix.py`, `tools/run_polish.py`, `tools/polish_worker.py`, `tools/make_matrix_page.py`, `tools/metrics/{dnsmos_scorer,spectral,f0,ecapa,run_metrics}.py`, `services/ina_worker_service.py` + `ina_worker.py`, `tests/test_matrix_isolation.py`.

**Venv mới**: `resemble_env`, `audiosr_env`, `ina_env` — theo khuôn `utils/worker_env.py::resolve_worker_python` đang dùng cho `diarizen_env`, `qwen3_env`, `sidon_env`.

**Không chạm**: `utils/separation_window.py` (`WindowPlanner`, `POLICY_VERSION` — thiết kế của bạn, ma trận chỉ tiêu thụ), `services/verification_service.py`, `utils/progress.py`.

## Kiểm chứng

1. **Chia sẻ checkpoint** — ma trận khói 4 tổ hợp × 1 file phải sinh đúng 1 thư mục stage 1, 2 diarization, 4 separation; và hai ô khác nhau đúng một trục phải cho kết quả **khác nhau**. Đây là phép thử bắt lỗi tệ nhất trong danh sách: 192 ô giống hệt nhau mà không báo gì.
2. **Ledger** — chạy hai tổ hợp liên tiếp trên cùng corpus, khẳng định tổ hợp thứ hai **thực sự tính**, không log `"Corpus complete"` rồi thoát.
3. **Mỗi trục có phép khẳng định đã-thật-sự-đổi**: D — không `config_ignored`; B — log nêu tên stem lấy được và nó không phải đầu vào nguyên bản; C — `CUT_SHARE_LIMIT` không phủ quyết; G — bản đồ hợp nhất khác bản đồ SSLAM.
4. **Bộ đo** — chạy trên 24 dump có sẵn trước, không tốn GPU.
5. **Báo cáo** — dựng với dữ liệu giả 192 dòng, mở trong trình duyệt, không lỗi JS, dưới 200 MB.
6. **`python -m pytest tests/ -q`** — nền hiện tại 4 lỗi có sẵn (`test_batch` ×2, `test_config_driven`, `test_prefix_cache`), không được thêm lỗi mới.
