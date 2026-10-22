# Kế hoạch viết code và chạy train full-duplex — 2 A100

Ngày: 2026-09-28. **Chưa thực thi các bước bên dưới.**
Thiết kế/giả định/nguồn: [full_duplex_training_system_design.md](full_duplex_training_system_design.md).

## 1. Đầu ra hoàn chỉnh cần bàn giao

Không coi một script gọi `train.py` hoặc một file LoRA là hoàn thành. Bản v1 phải có:

1. Importer cho hai dạng export hiện tại, canonical schema, audit và quarantine có lý do.
2. Dataset builder chống trùng/leakage, xử lý timeline/seams/role, cache codec/text có resume.
3. Trainer hai GPU: BF16, LoRA/FSDP, loss đúng/masks đúng, accumulation, checkpoint đầy đủ và resume kiểm thử.
4. Báo cáo train gồm loss thành phần, quality theo nhóm, VRAM/GPU/CPU/IO, data-wait và exposure.
5. Eval offline + causal streaming + live; bộ gold tiếng Việt độc lập và bảng trước/sau.
6. Checkpoint nạp được trong server, runbook tái lập, model/data cards, giới hạn và giấy phép/provenance.

Không thay thế các file pipeline hiện tại; không bật cloud logging hay upload dữ liệu mặc định. Không thêm RL, tool use hoặc cloning giọng tùy ý vào v1.

## 2. Vị trí và cấu trúc code dự kiến

Thêm một package độc lập ở `training/` do dependency train khác pipeline. Không chuyển `podcast-pipeline/` sang layout mới.

```text
training/
  pyproject.toml                 # env/dependency boundary riêng, CLI
  requirements.lock             # hoặc lock tương đương; chọn một cơ chế
  README.md
  configs/
    data_vi.yaml
    moshi_a100_40gb.yaml
    moshi_a100_80gb.yaml
    eval_vi.yaml
  src/sommelier_train/
    cli.py                      # audit/build/cache/benchmark/evaluate/export
    config.py                   # strict typed config; lỗi nếu key lạ
    schema.py                   # manifest/sample/window/run contracts
    importers/
      clean_two_channel.py
      conversation_export.py
    data/
      audit.py                  # audio/text/track QC, quarantine reasons
      provenance.py             # source UID, export revision, rights status
      split.py                  # groups + dedup clusters + frozen split
      windows.py                # samples, seams, role views, valid masks
      cache.py                  # codec/text cache + atomic shards
      dataset.py                # reader, rank sharding, bucket sampler
    backends/
      moshi.py                  # thin adapter cho pinned upstream, không rewrite LM
    train.py                    # distributed entrypoint/loop
    checkpoint.py               # save/resume/RNG/sampler/config hashes
    metrics.py                  # structured logs + resource profiler
    eval/
      offline.py
      streaming.py
      scoring.py
      report.py
  tests/
    fixtures/                   # tiny synthetic audio/metadata, không corpus thật
    test_importers.py
    test_timeline.py
    test_quality_masks.py
    test_split.py
    test_cache_equivalence.py
    test_moshi_contract.py
    test_training_resume.py
    test_streaming_causality.py
    test_cli.py
```

Đây là các module theo trách nhiệm; có thể gộp module nhỏ khi triển khai. Không dựng framework nhiều backend trước khi Moshi chạy được. Lychee chỉ thêm khi có quyết định qua gate.

Artifact lớn đặt ngoài Git: manifests/splits, codec shards, dataset reports, run logs, checkpoints, eval recordings. Chỉ đưa schema/config/tests/docs và dependency patch cần tái lập vào Git. Giữ nguyên file hiện có `:memory:.ses`, không gộp vào thay đổi training.

## 3. P0 — Preflight dữ liệu, phần cứng và mục tiêu

Mục tiêu: biết đang train với gì, trên máy nào, chi phí và chất lượng đo ra sao.

Công việc:

- Đọc metadata của hai export kinds, đối chiếu source registry; lấy một tập mẫu có overlap, không overlap, failure spans và seams.
- Chạy kiểm kê nguồn/file/hours, không nạp toàn corpus vào RAM.
- Trên máy A100 ghi `nvidia-smi`, `nvidia-smi topo -m`, CPU/RAM/free NVMe, driver/CUDA/Torch thực tế; kiểm tra tiến trình đang dùng GPU bằng PID và chủ sở hữu.
- Chốt output mode: human–human adaptation, assistant SFT hoặc cả hai; tiêu chí nghe và hành vi.
- Kiểm tra quyền train/phân phối, consent khi dùng giọng; nguồn thiếu quyền được quarantine cho đến khi xác minh.
- Ghi baseline release/commit của trainer, codec, model; chọn gold dev/test không lấy lại từ train.

Đầu ra: `dataset_inventory.json`, `hardware.json`, `baseline_decision.md`, source/rights inventory. Đây là artifact dự kiến sinh, chưa tồn tại.

Gate: đường dẫn data thực đọc được, biết hai GPU dung lượng bao nhiêu, không còn nhầm giờ raw với giờ clean; không có job lạ bị dừng tự động. Nếu chưa có truy cập máy A100 vẫn làm P1/P2 CPU nhưng không xác nhận VRAM/train throughput.

## 4. P1 — Importer, schema và audit

Các file: `schema.py`, `config.py`, `importers/`, `data/audit.py`, `provenance.py`, CLI audit, tests tương ứng.

Thực hiện:

1. Định nghĩa normalized schema với discriminator `export_kind`; map A/B và SP1/SP2 về local channel IDs mà không mất original speaker ID.
2. Với clean export, đọc `audio.method`; với conversation export đọc `channels_2ch.method`. Không đoán strict chỉ từ tên `clean.wav`.
3. Map clip timestamps/original spans; dùng sample bounds làm chuẩn media. Log sai lệch rounding, reject lệch vượt dung sai. JSON malformed/thiếu fields phải có error ID rõ ràng.
4. Ghi text/words và provenance; nếu refinement không còn khớp spoken text, flag cần sửa/alignment lại, không tự tạo gold.
5. Thống kê duration, sample rate, clipping, silence ratio, channel correlation, turns, failure/seam coverage và missing alignment. Correlation cao chỉ là flag, không phải auto kết tội có leakage.
6. Quarantine `time_gated`, window chứa zero giả, source chưa rõ hoặc corrupted audio. Không sửa audio gốc và không xóa file nguồn.
7. Report HTML nghe song song mixture/A/B và timeline cho mẫu audit; summary JSON máy đọc được.

Importer chỉ nhận source/export đã hoàn tất và cùng revision audio/text/alignment; phải có fixture source đang `building`, stale alignment và export bị ngắt giữa chừng. Không quét staging như dữ liệu train hợp lệ.

Tests: cả hai schema; invalid/missing metadata; mono/stereo mismatch; out-of-range/NaN timestamps; overlap giữa hai speaker được chấp nhận; synthetic silence thật không bị coi là failure; `same_speaker` không bị gán nhầm `zeroed_in_strict_track=true`; thiếu failure metadata không tự coi là sạch.

Gate: fixture import deterministic; mỗi accepted/rejected record truy ngược được; 100–200 clip audit tay phân tầng được ghi nhận, mức lỗi đủ chấp nhận trước mở rộng. Không tự kết luận tất cả corpus sạch từ fixture tests.

## 5. P2 — Dataset snapshot và split

Các file: `data/split.py`, `windows.py`, `dataset.py`, config data, tests split/timeline.

Thực hiện:

- Định danh bằng source provenance và fingerprint thay vì ID số thứ tự của exporter. Union near-duplicate/source groups trước split.
- Freeze train/dev/test manifest + split hash. Không tách các window cùng episode/re-upload/role-swapped sang split khác.
- Chọn representative cho vùng trùng giữa hai exporter; export-kind là provenance, không phải nhãn chất lượng.
- Window mặc định khoảng 100 giây, boundary có context hợp lệ, không cắt qua seam/failure; phần tail xử lý valid mask. Không nối hai recording để lấp batch nếu chưa có attention/reset mask đúng.
- Không cắt ngắn lượt nói để giả thành nhiều lượt ngắn. Short-turn subset xác định theo lượt gốc, không theo boundary window.
- Tạo role views sau split; A/B dùng cùng cache audio, hoán đổi đúng text target. Người target cố định trong một window.
- Ghi accepted hours, unique hours, target speaking hours, padding fraction, số lượt, overlap/backchannel, accent/topic/source mix. Sampling theo strata/source có trần lặp, không chỉ top score.
- Snapshot bất biến khi đang train; dữ liệu pipeline mới đi vào version sau. Resume không tự nhập thêm clip mới.

Gate: 0 overlap source/dedup clusters giữa các split; kiểm thử role swap đúng kênh/text; seam có reset hoặc split; tổng accounting tái lập cùng seed. Nếu chưa biết cross-recording speaker identity, report là chưa biết.

## 6. P3 — Environment và baseline compatibility

Các file: `training/pyproject.toml`, lock, pinned upstream patch/integration và `backends/moshi.py`.

Trình tự:

1. Tạo env riêng với Torch phiên bản không thấp hơn baseline hiện dùng. Pin CUDA build/torchaudio tương thích; không sửa env pipeline.
2. Resolve có kiểm soát xung đột upstream `torch==2.6`. Cập nhật dependency constraint trong integration patch sau kiểm thử, không bỏ dependency resolution bằng `--no-deps`.
3. Pin Moshi/trainer/model/tokenizer/codec revisions; archive config resolved và license notices.
4. Load checkpoint BF16, count total/trainable parameters thật; kiểm tra codec sample rate/frame rate/codebooks và special IDs từ config.
5. Smoke một GPU rồi FSDP hai GPU, finite loss/backward, only intended trainable params đổi, base frozen không đổi, checkpoint nạp lại sinh audio.
6. Không sử dụng `overwrite_run_dir` kiểu xóa thư mục để resume. Run mới có ID mới; resume là thao tác riêng.

Gate: import/forward/backward/save/load qua trên Torch yêu cầu và A100 thực. Dependency compatibility là blocker độc lập, không đánh tráo với chuyện “tạo env riêng là xong”.

## 7. P4 — Gate tiếng Việt và cache codec

Các file: `data/cache.py`, text interleaver adapter, tests cache/backend, eval codec.

Thực hiện:

- Đánh giá codec reconstruction trên gold Việt đa accent; model codec được freeze, giữ cấu hình checkpoint.
- Kiểm tra tokenizer round-trip, unknown token, token rate/timing và số token thực được giữ sau interleave. **Không cho phép silent token loss**; log overflow/backlog cuối window thành lỗi có thể tái hiện.
- Viết golden tiny example: hai người overlap, một pause, một backchannel, một tail padded; xác nhận tensor layouts, target/input channel order và model delay masks bằng upstream.
- Cache per-window/per-channel đúng context/reset policy. Hai GPU encode hai shard partitions độc lập, không có model training đồng thời.
- Atomic shard write, checksum, locking/ownership, resume bỏ qua shard hoàn tất đúng version. Giới hạn CPU workers và RAM/prefetch.
- So token IDs/valid frames/loss giữa online và cached path; nếu kernel codec không bitwise deterministic, điều tra và cố định đường encode trước khi cho cache làm nguồn chuẩn. Không tự nới tolerance để che channel/timing lỗi.

Gate: codec nghe chấp nhận được; không rơi text Việt; cache bất biến và tương đương baseline trong contract đã chọn. Nếu tokenizer/frame budget không xử lý được tiếng Việt, dừng full train và mở quyết định kiến trúc; không tiếp tục trên nhãn thiếu chữ.

## 8. P5 — Trainer dùng thực tế

Các file: `train.py`, `checkpoint.py`, `metrics.py`, `backends/moshi.py`, profiles, tests training.

### 8.1 Training loop

- Khởi tạo `torchrun` world size 2, rank/device mapping rõ; FSDP theo integration upstream đã pin.
- LoRA rank 128/scaling theo checkpoint recipe; freeze Mimi/base; inventory module names trainable và số tham số. Không dùng tên PEFT target_modules của Qwen cho Moshi.
- BF16 forward theo model; optimizer/reduction FP32 phù hợp, gradient clipping; normalization theo valid target counts khi padding/length khác nhau.
- Accumulation đúng: phân biệt microbatch steps/optimizer steps, đồng bộ gradient và LR/scheduler chỉ bước khi optimizer thực cập nhật. Test single-large-batch so với accumulated nhỏ trong dung sai numerical.
- Causal model masks/delay đúng; text/audio losses tách riêng, masks ngoài dữ liệu không được thành target silence.
- Bắt đầu từ learning rate `2e-6`, weight decay `0.1`, text padding weight `0.5`, semantic-codebook multiplier `100` theo [example Moshi](https://github.com/kyutai-labs/moshi-finetune/blob/main/example/moshi_7B.yaml); xem là baseline cần validation, không hyperparameter tối ưu tiếng Việt đã biết.
- Dev evaluation thật được bật; cấu hình upstream dùng `do_eval`, không dựa vào tên `no_eval` ở một phần README.
- Không encode Mimi lại mỗi train step khi cache được chọn. Không gọi `.cpu()`/`.item()` mỗi frame chỉ để logging; batch log có giới hạn, nhưng giữ kiểm tra NaN cần thiết.

### 8.2 Dữ liệu phân tán và lỗi

- Rank sampler không trùng item ngoài sampling có chủ ý; có equal step count hoặc padding sentinel đúng để tránh deadlock FSDP.
- Corrupted cache trong train: fail đồng bộ với record ID, không mỗi rank skip tùy ý rồi treo collective.
- Eval aggregate sum/count trên toàn ranks, không trung bình các mean khác sample count.
- Padding batch không làm thiên loss về clip ngắn; source/role weighting được log.
- SIGTERM/scheduled wall-time: safe checkpoint ở optimizer boundary; không báo checkpoint hoàn tất trước khi đủ các shard.

### 8.3 Checkpoint

Tách `resume/` với model/LoRA + optimizer + scheduler + RNG các rank + sampler/cursor + step/exposure/config/data hashes, và `export/` cho adapter/model inference.

Ghi staging → validate → atomic completion marker. Không ghi đè checkpoint tốt khi save lỗi. Resume mismatch base/config/split phải fail hoặc tạo run mới explicit; không silently “resume weights” nhưng reset optimizer/sampling.

Gate bắt buộc: run N+M steps liên tục so với run N → save → resume M có cùng sample sequence/LR và numerical agreement phù hợp; trainer 2 GPU chạy ổn, checkpoint export được model server đọc. Cần test cả state trên hai rank, không chỉ rank 0.

### 8.4 Report hiệu năng

Mỗi interval: step time, valid audio-sec/sec, optimizer-step exposure, data wait, H2D, forward/backward/optimizer/NCCL, eval/save time; GPU util và memory allocated/reserved + device total peak, CPU util/RSS/IO. Phân rã GPU bằng sampling/profiler ở đoạn ngắn, không synchronize mọi operation trong run chính.

Gate: 100–200 steady steps cùng stress longest batch và checkpoint, không OOM/NaN; mức headroom tối thiểu theo thiết kế. Tăng workers/microbatch chỉ khi benchmark cùng precision/context/data/quality contract.

## 9. P6 — Eval harness trước pilot dài

Các file: `eval/`, config eval, fixture scenarios, tests causality và report.

Thực hiện:

- Baseline checkpoint nguyên bản chạy cùng input/protocol với checkpoint fine-tune; giữ decoding config/seed khi so sánh.
- Replay user audio causal, model tự sinh target; không teacher-force target ở bài behavioral evaluation.
- Test causality: hai input giống hệt đến t nhưng khác sau t phải cho output prefix đến t giống nhau ở chế độ deterministic, sau tính đúng buffering/codec delay.
- Record waveform hai phía và monotonic event timestamps; phân biệt source clock, model-compute clock, transport clock.
- Áp dụng protocol FDB v1/v1.5; thêm 200 dev scenarios Việt. Live test có ngắt lời/sửa ý và nghe lâu; không dùng score tự đánh giá duy nhất.
- ASR metrics, blind listening, response relevance, silence/repetition, turn errors, p50/p95 latency và failure rates theo scenario/accent/noise/length.
- LLM judge nếu dùng: version/prompt cố định, ground truth chỉ cho judge, so với gold tay; chạy sau train hoặc ở máy riêng, không chiếm VRAM training.

Gate: report baseline được tái lập; một model luôn im lặng hoặc luôn nói không thể đạt nhờ latency/loss; test data không lộ vào train/generator prompts.

## 10. P7 — Lộ trình chạy thí nghiệm

Các lượng dữ liệu/step dưới đây là **ngân sách thử đề xuất**, không dự báo chất lượng hay giờ GPU.

| Run | Dữ liệu / thời lượng | Mục đích và điều kiện qua |
|---|---|---|
| R0 | 8–16 clip sạch, khoảng 10–50 smoke steps | Contract tensor/loss/gradient/save/load, không kết luận chất lượng |
| R1 | Cùng tiny set, 100–300 steps nếu cần | Overfit target để bắt bug; model không im lặng do mask sai |
| R2 | 5–20 giờ đa dạng; 1 exposure pass đầu, tối đa 3 trước review | Xác minh tiếng Việt/generalization, eval mỗi ~0.25 exposure |
| R3 | Cùng pilot và exposure budget | So short-turn subset với balanced subset; không test quá nhiều knobs cùng lúc |
| R4 | 50–100 giờ accepted nếu corpus có đủ | Scale gate; kiểm tra data/throughput/chất lượng, không train 2.000 steps máy móc |
| R5 | Toàn bộ snapshot accepted | Adaptation 1–3 passes có early stopping dev, giữ best checkpoint |
| R6 | Corpus assistant đã duyệt + replay mix | Chỉ chạy nếu cần trợ lý và có data phù hợp, gate mới cho instruction/voice |

Nếu chỉ có ít dữ liệu, bỏ các mốc số giờ không thể đáp ứng và báo giới hạn. Không lặp corpus nhỏ cho đủ “giờ train” rồi ghi như có thêm dữ liệu độc lập. `max_steps` tính từ accepted frames/exposure và early stopping, không sao chép số steps của paper khác batch.

Thử LoRA-only trước. `ft_embed` chỉ là ablation khi tokenizer giữ đủ text nhưng ngôn ngữ học kém; cần đo memory và forgetting. Không dùng đổi embedding để chữa codec không tái tạo tiếng Việt hoặc làm mất token trong interleaver.

Checkpoint selection: ưu tiên đạt gate hiểu lời/âm thanh trước, sau đó turn-taking/latency; không gộp hết thành một scalar có thể che regression. Tối thiểu hai seed cho finalist nếu ngân sách cho phép; nếu một seed, nêu hạn chế. Test khóa kín chạy ở mốc release, không tune liên tục theo test.

## 11. P8 — Export và serving

Đầu ra: adapter/checkpoint hợp lệ, resolved model config, pinned base/tokenizer/codec refs, checksum, model/data cards, evaluation report và runbook.

- Dùng runtime Moshi streaming tương thích checkpoint, không mặc định đem model vào vLLM của pipeline.
- Endpoint WebSocket/audio session: input liên tục, output liên tục kể cả lúc nghe; reset state riêng theo session, bound queues/backpressure, reconnect/cancel rõ ràng.
- Đánh giá một session trên A100 trước; không hứa throughput nhiều user khi chưa đo. GPU thứ hai có thể phục vụ session replica sau khi train xong, không đồng nghĩa model luôn cần 2 GPU để inference.
- Nếu cần web microphone thực, kiểm tra echo cancellation/playback buffering và acoustic echo. Microphone thu lại giọng model là một test khác với replay clean dual tracks.
- Voice/role behavior theo khả năng checkpoint, không tuyên bố voice-cloning/prompt control nếu chưa có.
- Checkpoint quay lui: giữ baseline + best previous, manifest tương ứng; inference nạp thử và so trace trước đổi bản phục vụ.

## 12. CLI dự kiến sau khi implement

**Các lệnh dưới đây là giao diện cần xây, hiện chưa chạy được trong repo.** Chạy trong env của `training/`; đường dẫn `/data/...` là ví dụ phải thay theo máy.

```bash
python -m sommelier_train.cli audit --config configs/data_vi.yaml
python -m sommelier_train.cli build --config configs/data_vi.yaml --snapshot vi-v1
python -m sommelier_train.cli cache --snapshot /data/train/vi-v1 --gpus 0,1
python -m sommelier_train.cli benchmark --config configs/moshi_a100_40gb.yaml
torchrun --standalone --nproc_per_node=2 -m sommelier_train.train --config configs/moshi_a100_40gb.yaml
python -m sommelier_train.cli evaluate --run /data/runs/vi-r2 --suite configs/eval_vi.yaml
python -m sommelier_train.cli export --run /data/runs/vi-r2 --checkpoint best
```

`--resume` phải là tham số riêng dẫn tới complete checkpoint; `--dry-run` validate/config/inventory không tự download weights. Tên/interface cuối có thể thay nhưng semantics audit/build/cache/train/eval/export phải giữ rõ.

Verification commands cần có sau triển khai:

```bash
python -m pytest tests/test_importers.py tests/test_timeline.py tests/test_split.py -q
python -m pytest tests/test_quality_masks.py tests/test_cli.py -q
python -m pytest tests/test_cache_equivalence.py tests/test_moshi_contract.py -m gpu -q
torchrun --standalone --nproc_per_node=2 -m sommelier_train.train --config configs/test_resume.yaml
python -m pytest tests/test_streaming_causality.py -q
```

Đánh dấu CPU/GPU/network tests riêng. CPU CI không download model; GPU integration chạy trên A100 với fixture nhỏ và artifact report. Mỗi pha chạy focused tests trước, rồi suite training; nếu sửa pipeline contracts mới cần regression suite pipeline.

## 13. Thứ tự thực hiện và tiêu chí dừng

```text
P0 inventory → P1 importer/audit → P2 split/windows
                                ↓
P3 env/compatibility → P4 codec/tiếng Việt/cache
                                ↓
                       P5 trainer + P6 eval
                                ↓
                       P7 pilot → full train
                                ↓
                       P8 export/serving
```

P1/P2 CPU có thể làm trước khi được cấp GPU. Không bỏ P6 để chạy full sớm. Khi gặp blocker, lưu report và dừng nhánh bị ảnh hưởng:

| Blocker | Hành động |
|---|---|
| Phần lớn audio là time-gated/zero giả/text sai | Sửa/re-export/relabel dữ liệu, không tăng số epoch |
| Torch yêu cầu không tương thích | Compatibility patch/tests hoặc đổi backend sau duyệt; không downgrade |
| Codec/tokenizer không giữ tiếng Việt | Dừng scale Moshi; đánh giá alternate, không bỏ dấu/cắt text |
| OOM context chuẩn trên 2×40GB | Đo allocations/activation/sharding; nếu không giải được giữ chất lượng, báo không khả thi |
| Train loss tốt nhưng live im lặng/ngắt sai | Audit silence/text masks, turn mix, decoding và dữ liệu hành vi; không chọn bằng loss |
| Chưa có assistant corpus | Bàn giao conversation-adapted model với giới hạn, chưa gọi là assistant hoàn chỉnh |
| Không biết quyền nguồn/giọng | Quarantine nguồn tương ứng, tiếp tục phần đã xác minh |

Ước lượng công sức kỹ thuật để lập lịch, không phải cam kết: khoảng 8–15 ngày kỹ sư cho data/compatibility/trainer/eval v1 nếu dependency và tiếng Việt qua gate; khoảng 3–5 ngày thêm cho serving/hardening cơ bản. Annotation, thu âm và thời gian GPU tính riêng sau P0/benchmark. Nếu phải sửa tokenizer/codec hoặc port model khác, cần re-plan thay vì giữ timeline này.

## 14. Việc nên bắt đầu ngay sau khi duyệt

Làm **P0–P2 trước**: inventory → importer/QC → snapshot/split và báo bao nhiêu giờ dữ liệu thực sự dùng được. Đồng thời chuẩn bị compatibility tests nhưng chưa khởi chạy train dài.

Sau đó P3–P6 dựng một đường hoàn chỉnh trên tiny dataset; chỉ khi có baseline evaluation và bằng chứng tiếng Việt mới chạy R2 pilot. Đây là bước quyết định tiếp tục đầu tư vào Moshi hay chuyển hướng, trước khi tốn thời gian train toàn corpus.
