# Kế hoạch tối ưu hiệu năng toàn pipeline

Ngày: 2026-09-20. Nhánh khảo sát: `solid-architecture`.

Trạng thái: **bản thiết kế để duyệt; chưa triển khai và chưa benchmark GPU**.

## 1. Mục tiêu và phạm vi

Tăng số giờ audio xử lý thành công trong mỗi giờ chạy máy, giữ chất lượng và khả năng truy xuất kết quả. GPU, CPU, RAM và ổ đĩa được điều phối cùng nhau. GPU có VRAM trống là cơ hội nhận việc nếu việc đó giúp kết thúc sớm hơn; không phải lý do để luôn nạp thêm model.

Phạm vi bao gồm: đọc audio, nhận diện nhạc/noise, tách nhạc, diarization, dựng window, separation, ASR ba model, ROVER, LLM refinement, caption nếu bật, checkpoint và export/review. Chạy một file và nhiều file đều được hỗ trợ; máy một GPU có đường chạy riêng.

Các yêu cầu giữ từ thảo luận:

- Refinement: một tiến trình suy luận, một bản Qwen chia trên hai GPU khi có hai card phù hợp; không tạo hai bản Qwen refinement.
- Diarization: một worker quản lý pipeline, clustering chung cả file. Trước tiên phân bố segmentation và embedding trên hai card; tăng mức song song sau khi xác minh được tính tương đương.
- Separation: tối đa hai worker Sidon, mỗi card một worker.
- ASR: cho phép tạo thêm bản sao của model còn nhiều việc trên GPU vừa rảnh, kể cả đang xử lý cùng một file.
- Nhạc: số worker phụ thuộc số job và bộ nhớ thực tế. Noise hiện là nhận diện/chấm điểm, chưa có denoiser.
- Các tham số vận hành đưa vào config; không đổi model, thuật toán cắt, ngưỡng QC hay quy tắc transcript chỉ để đạt tốc độ.

“Chạy max” được đo bằng công việc hữu ích hoàn thành. RAM đầy gây swap, VRAM sát trần gây OOM và CPU mở quá nhiều thread đều có thể làm máy chậm hơn.

## 2. Những điểm nghẽn đã xác minh

| Hiện trạng trong code | Hệ quả | Thay đổi cần có |
| --- | --- | --- |
| `_run_whisper_batch` lặp từng segment; mỗi lần truyền một VAD span | Batch config lớn không tạo batch nhiều segment độc lập | Batch ở cấp feature/input của nhiều segment |
| PhoWhisper `transcribe_batch` tính batch size nhưng vẫn gọi từng `transcribe`, bên trong đặt batch 1 | Chưa dùng batching thật giữa các segment | Adapter batch có ánh xạ kết quả theo ID |
| Qwen3-ASR gửi từng audio và chờ từng phản hồi | GPU nhận nhiều lượt suy luận nhỏ; thiếu giao thức batch/job | RPC theo job và batch, worker nhận việc động |
| ASR giao toàn bộ danh sách cho ba future rồi chờ đủ ba model | Model xong sớm không giúp model còn lại; không resume từng voter | Hàng đợi riêng theo model, registry worker và checkpoint từng kết quả |
| Profile `a100` đặt Whisper/PhoWhisper/Qwen-ASR ở float32 và batch 256 | Có cơ hội giảm bộ nhớ, nhưng con số 256 chưa phản ánh batch thực tế | Đo dtype được backend hỗ trợ và batch thực, kiểm tra chất lượng trước đổi |
| Refinement dùng `device_map="auto"`, không có ngân sách VRAM công khai | Khó kiểm soát layer placement và chỗ dành cho KV cache | Phân bổ có ngân sách, ghi `hf_device_map`, batch theo token |
| Refinement gọi `unload()` cuối mỗi file khi `keep_models=false` | Trong batch theo stage vẫn có thể nạp lại LLM cho từng file | Quản lý vòng đời refinement theo scope, rồi chuyển sang registry chung |
| `WindowBuildPool.build_all` submit tất cả và giữ danh sách future | Kết quả audio đã dựng có thể tích trong RAM khi GPU xử lý chậm | Cửa sổ submit có giới hạn theo số job và byte |
| Window song song mặc định energy, nhánh tuần tự có thể dùng Silero | Đổi lịch chạy có thể đổi điểm cắt và output | Chung nguồn quyết định biên hoặc policy được ghi rõ, đo tương đương |
| Main/worker/thread pool có nhiều bộ thread riêng, CT2 wrapper còn mặc định 4 thread | Tổng số thread không bám sát số CPU được cấp và stage đang chạy | Ngân sách CPU toàn chương trình |
| Audio đã có mmap cache và window đã có shared memory | Có nền tảng tốt, nhưng vẫn copy khi tạo shared memory và pickle output | Tái sử dụng buffer, truyền descriptor và giới hạn số file đang giữ |
| ASR chuẩn bị toàn bộ audio segment trước suy luận | GPU phải chờ prep; RAM tăng theo file | Producer chuẩn bị theo lô, cấp dữ liệu có giới hạn |
| RoFormer wrapper chưa chuyển GPU index vào lúc dựng `Separator` | Chỉ đổi `device` chưa bảo đảm chọn đúng card | Worker có phạm vi GPU rõ ràng |
| Sidon worker có cả CUDA mask và đối số chỉ số card bên ngoài | Worker thứ hai có nguy cơ gọi GPU index không tồn tại trong mask | Thống nhất UUID/index host và index cục bộ |
| ONNX WeSpeaker chưa đặt CUDA provider `device_id` | Phần ONNX có thể vẫn dồn về card mặc định | Provider được cấu hình và xác minh theo worker |
| Checkpoint stage ghi trực tiếp, `exists` chỉ kiểm tra file tồn tại | Bị ngắt lúc ghi hoặc có nhiều producer có thể tạo checkpoint lỗi | Atomic artifact + journal commit, một writer sở hữu kết quả |
| Export MP3 tuần tự, review có thể chứa lượng audio lớn | CPU/I/O có thể thành đoạn cuối kéo dài khi GPU đã xong | Export queue giới hạn, tái sử dụng artifact và encode đã có |

Nguồn local đã đọc: [ASR service](/Users/lam/Documents/sommelier/sommerlier/podcast-pipeline/services/asr_service.py), [PhoWhisper](/Users/lam/Documents/sommelier/sommerlier/podcast-pipeline/models/phowhisper.py), [Whisper pipeline](/Users/lam/Documents/sommelier/sommerlier/podcast-pipeline/models/whisper.py), [Qwen client](/Users/lam/Documents/sommelier/sommerlier/podcast-pipeline/models/qwen3_asr.py), [window pool](/Users/lam/Documents/sommelier/sommerlier/podcast-pipeline/utils/window_pool.py), [CPU plan](/Users/lam/Documents/sommelier/sommerlier/podcast-pipeline/utils/cpu_plan.py), [checkpoint](/Users/lam/Documents/sommelier/sommerlier/podcast-pipeline/utils/checkpoint.py), [pipeline](/Users/lam/Documents/sommelier/sommerlier/podcast-pipeline/services/pipeline_service.py).

Một chi tiết cần khóa baseline: wrapper Whisper thực tế cắt input theo VAD span trước feature extraction. Khi viết batching mới phải đối chiếu tensor thật, không chỉ dựa vào comment về context padding trong ASR service; sửa semantics context là thay đổi chất lượng riêng.

## 3. Kiến trúc điều phối đề xuất

Giữ các model wrapper và môi trường Python hiện có, thêm một coordinator quản lý tài nguyên và công việc. Không cần đưa ngay Ray, Kubernetes hay một inference server mới vào hệ thống.

| Thành phần | Trách nhiệm |
| --- | --- |
| Resource monitor | Quan sát GPU, RAM/cgroup, CPU quota, queue và I/O |
| Scheduler | Chọn job đã đủ phụ thuộc; cấp ngân sách và quyết định mở/đóng worker |
| Worker registry | Theo dõi model, thiết bị, interpreter, trạng thái, thời gian load, tốc độ và bộ nhớ |
| Job journal | Theo dõi công việc, attempt, lease và kết quả đã commit |
| Audio store | Mmap/shared buffer, timeline, sample rate, provenance và vòng đời dữ liệu |
| Result coordinator | Gán kết quả theo ID, chạy hậu xử lý đúng thứ tự và ghi checkpoint |
| CPU/I/O pools | Chuẩn bị âm thanh, feature, window, encode và ghi artifact với quota |

Luồng phụ thuộc trong một file:

```text
decode/cache -> SSLAM -> music/noise maps -> music patches -> timeline
  -> diarization -> overlap windows -> Sidon outputs -> identity/QC/splice
  -> ASR input -> Whisper + PhoWhisper + Qwen3 -> ROVER
  -> refinement -> export/review
```

Caption, khi được bật, nhận đúng audio/transcript theo hợp đồng hiện tại và phải hoàn thành trước các consumer cần nó. Noise score đi cùng provenance tới export. Các file khác nhau có thể chạy gối khi ngân sách cho phép.

Coordinator giữ metadata và quyền sở hữu, không giữ thêm một bản đầy đủ waveform cho mỗi hàng đợi. Không chia sẻ cùng một `PipelineService` mutable giữa các thread xử lý file: music map, noise track, timeline, stats và dump directory hiện là trạng thái riêng của file.

## 4. Đo phần cứng và đặt ngân sách trước khi chạy

Nhận diện GPU bằng thiết bị được runtime cho phép thấy, lưu UUID và ánh xạ index; hỗ trợ mask như `CUDA_VISIBLE_DEVICES=2,3`, một GPU và hai GPU khác dung lượng. Không suy luận rằng `gpu_2=0` trong profile A100 nghĩa là máy chỉ có một card. Chế độ auto và danh sách thiết bị do người dùng chỉ định phải có thứ tự ưu tiên rõ ràng.

Đọc CPU affinity, cgroup quota, allocation SLURM và RAM được cấp; không chỉ dùng tổng tài nguyên host. Kiểm tra dung lượng local scratch và `/dev/shm`. Nếu có NUMA, đo lợi ích pin CPU/memory gần GPU trước khi bật.

Ngân sách một worker mới phải bao gồm:

```text
GPU: trọng số + peak batch + KV/activation + workspace + CUDA context + dự phòng
RAM: peak load checkpoint + input/output + preprocessing + runtime + dự phòng
Disk: cache/download cần thiết + exchange buffers + output tạm + fail artifacts
```

Đặt reservation trước khi nạp model để hai quyết định cùng lúc không tiêu cùng phần VRAM. Đối chiếu cả GPU memory toàn thiết bị và PyTorch allocated/reserved; CT2 và ONNX có allocator riêng. `empty_cache()` không chứng minh model đã giải phóng bộ nhớ.

Giá trị khởi đầu để benchmark, không phải cam kết tối ưu: VRAM dự phòng `max(2 GiB, 15% dung lượng card)`; RAM mức mềm 75%, mức dừng producer 85% giới hạn được cấp. Các mức này có thể điều chỉnh theo workload và peak đã đo.

## 5. Scheduler quyết định dựa trên công việc còn lại

Worker có các trạng thái `loading -> warming -> ready -> busy -> draining -> stopped/failed`. Job có `pending -> leased -> running -> output_ready -> committed`, hoặc `retry_wait/failed`. Worker đang load không được tính như đã xử lý được việc.

Mỗi job mang: `run_id`, `file_id`, `stage`, `segment/window_id`, `model_revision`, input/config fingerprint, attempt, lease generation và dependency IDs. Batch là danh sách job; không dùng thứ tự response làm danh tính.

Thứ tự ưu tiên:

1. Hoàn tất hoặc phục hồi công việc đang chặn file gần hoàn thành.
2. Cấp batch tiếp theo cho model đã ở GPU và có việc sẵn.
3. Dùng card rảnh để hỗ trợ model tạo phần đuôi dài của stage.
4. Chạy stage/file kế tiếp khi phụ thuộc đã xong và không chặn stage cần cả hai card.
5. Prefetch nhẹ hoặc để card nghỉ nếu chi phí mở thêm model lớn hơn lợi ích.

Queue trống không đồng nghĩa đã hết việc; queue đầy không có nghĩa CPU nên tiếp tục sản xuất. Job đang phụ thuộc đầu ra chưa có không được dùng làm lý do nạp replica ngay.

Scheduler sử dụng cả event job hoàn thành và telemetry lấy mẫu khoảng 1 giây. Khi chỉ đổi VRAM mà không tăng throughput, không tăng worker tiếp. Có cooldown và ngưỡng lợi ích tối thiểu để tránh liên tục load/unload model.

## 6. ASR: GPU rảnh hỗ trợ model chưa xong

### 6.1 Hàng đợi và batching thật

Ba hàng đợi ứng với ba model. Mỗi segment cần một kết quả cho từng voter được cấu hình. Worker chỉ nhận một microbatch vừa ngân sách, không giữ toàn bộ danh sách segment của file.

Whisper và PhoWhisper cần adapter tạo batch nhiều input/feature độc lập bằng API backend phù hợp. Không nối các segment thành một câu/audio dài. Giữ language, task, decode options, VAD core, timestamp offset và normalization riêng từng job. Whisper thường pad feature theo khung cố định; dự toán memory dựa trên tensor sau pad, không chỉ tổng số giây nguồn.

Qwen3-ASR cần request chứa nhiều job, processor xử lý audio/text theo batch và trả một result cho mỗi ID. Phân nhóm theo độ dài để giảm pad; xác minh attention mask, token sinh, parser và quan hệ input/output với phiên bản đang cài. `models.qwen3.batch_size` hiện chưa khiến worker tự batch, nên cần sửa đủ client, worker và service.

Batch builder có các giới hạn: số item, frame sau pad, audio seconds, prompt/output token và thời gian chờ gom batch. Khi ít việc còn lại, chạy batch nhỏ ngay. Input không tương thích language/task/decoding không gom chung.

### 6.2 Ví dụ chuyển GPU rảnh thành Qwen worker thứ hai

| Thời điểm | GPU 0 | GPU 1 | Coordinator |
| --- | --- | --- | --- |
| Bắt đầu ASR | Whisper | PhoWhisper + Qwen nếu đã đo là cùng chạy hiệu quả và vừa VRAM | Ba queue microbatch |
| Whisper hoàn tất | Drain và dừng Whisper worker | Tiếp tục job đang chạy | Đo lại free VRAM và lượng việc mỗi queue |
| Qwen còn nhiều việc | Nạp Qwen replica từ cache rồi warm-up | Qwen cũ tiếp tục | Chưa chuyển job đang chạy |
| Replica ready | Qwen nhận các job chưa cấp | Qwen cũ nhận từ cùng queue | Lease bảo đảm mỗi job chỉ có một chủ hợp lệ |
| ASR hoàn tất | Dừng model theo lịch tiếp theo | Dừng model theo lịch tiếp theo | Commit đủ voter, cấp cả hai GPU cho refinement |

GPU 1 có hai model chỉ là một cách đặt ban đầu để đo, không phải cấu hình luôn thắng. Nếu hai model tranh compute làm tổng thời gian tăng, scheduler chạy từng đợt hoặc đổi vị trí theo profile đã hiệu chuẩn.

Khi PhoWhisper xong mà Qwen vẫn chạy trên GPU 1, card đó chưa rảnh hoàn toàn. Bộ điều phối phải biết mọi model đang thường trú trên card, không chỉ worker vừa kết thúc.

### 6.3 Quyết định có nên tạo replica

Ước lượng công việc bằng audio/frame/token và lịch sử thời gian, không chỉ số segment. Dùng tốc độ có tính đến model khác cùng card.

Với công việc đủ đồng đều, công thức khởi đầu:

```text
W = lượng việc còn lại, gồm pending và phần đang chạy ước lượng
r1 = tốc độ worker hiện tại; r2 = tốc độ dự kiến replica
L = chờ tài nguyên + giải phóng + load + warm-up
T_old = W / r1
T_new = L + max(0, W - r1 * L) / (r1 + r2)
gain = T_old - T_new - tổn thất do tranh CPU/I/O - chi phí cơ hội stage khác
```

Chỉ mở replica nếu có job pending đủ chia, ngân sách đủ và gain vượt biên dự phòng. Công thức là mô hình dự báo, không thay thế đo thực tế và có sai số ở input không đồng đều.

Ví dụ minh họa: 600 job tương đương, mỗi worker 5 job/s, load thêm 20s. Một worker cần 120s; hai worker dự kiến hoàn thành sau 70s kể từ quyết định. Nếu chỉ còn 40 job, worker cũ cần 8s: mở thêm model không giúp.

Job đang generate tiếp tục ở worker cũ. Không di chuyển KV cache hay hủy một lượt đang chạy chỉ để cân tải. Khi replica warm-up xong mà việc đã ít, dùng nó cho queue tương thích của file kế tiếp hoặc dừng theo chính sách warm residency.

### 6.4 Kết quả và resume

Kết quả khóa theo `(file_id, segment_id, voter, semantic_fingerprint)`. Hai replica của Qwen vẫn chỉ tạo **một phiếu Qwen** cho mỗi segment. Retry có thể tính lại, nhưng commit chỉ nhận một kết quả hợp lệ; response của lease cũ bị bỏ.

ROVER chạy khi đủ trạng thái terminal của các voter cần thiết. Transcript rỗng hợp lệ, lỗi suy luận, timeout và voter bị tắt phải là bốn trạng thái khác nhau. Không biến lỗi worker thành phiếu rỗng rồi báo thành công.

Checkpoint từng voter/segment để resume đúng phần thiếu. Ghép output theo index/timeline nguồn, không theo thứ tự job xong. Có thể batch nhiều file nếu giữ ranh giới prompt và ID; ưu tiên tuổi job để file ngắn không bị chờ vô hạn.

### 6.5 Tách model thành worker có vòng đời kiểm soát được

Whisper/PhoWhisper hiện ở tiến trình chính. Chuyển dần sang worker riêng theo cùng giao thức để drain, dừng và xác nhận trả VRAM trước khi nạp model khác. Tiến trình Qwen giữ interpreter riêng. Không gọi đồng thời nhiều luồng vào một pipe stdin/stdout của client cũ.

CTranslate2 có API batching và thực thi song song; đây là nền tảng để dùng lại thay vì viết decoder. Adapter phải kiểm tra bản CT2 4.5.0 đang pin: tài liệu online hiện mô tả bản mới hơn, không giả định mọi API đều có. [API Whisper](https://opennmt.net/CTranslate2/python/ctranslate2.models.Whisper.html), [parallelism](https://opennmt.net/CTranslate2/parallel.html).

## 7. LLM refinement: một model dùng hai GPU

Một worker duy nhất giữ toàn bộ Qwen được chia thành các nhóm layer liên tiếp; báo device map thực tế. `auto` có thể đã phân bố nhiều GPU, nên không coi “đổi auto sang balanced” là toàn bộ phần nâng cấp. Chọn map theo budget đã đo và chừa bộ nhớ cho `generate`/KV cache. Đây là inference hợp nhất transcript, không phải huấn luyện fine-tune. [Accelerate device map](https://huggingface.co/docs/accelerate/concept_guides/big_model_inference#designing-a-device-map).

Thiết kế cụ thể:

- Coordinator cấp đồng thời hai GPU cho worker; không để một replica ASR khác nạp vào VRAM đã dành cho refinement.
- Batch theo tổng prompt/token dự kiến và độ dài lớn nhất, không chỉ số segment. Tách batch câu cực dài khỏi câu ngắn.
- Giữ model qua nhiều file trong một scope refinery. Giữ tokenizer và tokenized prompt chung; prefix KV cache chỉ bật sau kiểm tra trên device map phân tán và phiên bản cache cụ thể.
- Không `.to(cuda:0)` toàn bộ model sau dispatch. Input tới thiết bị entry, cache của từng layer nằm đúng device.
- Theo dõi OOM theo từng card; giảm batch/token budget, retry phần chưa commit. Không âm thầm CPU-offload khi chính sách không cho phép.
- Giữ prompt, guard, giới hạn sinh và logic bỏ qua backchannel của baseline. Không tự bỏ refinement chỉ vì ba ASR giống nhau trong đợt tối ưu lịch chạy.
- Khi chỉ có một GPU, chọn batch/map vừa card. Nếu checkpoint không vừa, báo rõ; không tự đổi model hay quantization.

Chia layer thông thường giúp phân bố VRAM nhưng hai GPU có thể thực thi luân phiên. Không hứa utilization 100% hoặc tốc độ gấp đôi. Tensor parallel/engine khác là hướng thử nghiệm riêng, chỉ xem xét sau khi đo ra đây là nút nghẽn; không tự thay ràng buộc một tiến trình bằng engine nhiều subprocess.

Để model hai GPU không bị bỏ đói, scheduler dùng một đợt ASR có kích thước giới hạn, rồi drain worker và chạy refinement trên các file sẵn sàng. GPU phục vụ layer trong một worker hai card không được xem là “rảnh cho thuê” chỉ vì một mẫu telemetry thấy utilization bằng 0.

## 8. Diarization: một worker, clustering chung

Mức 1: segmentation trên GPU 0, embedding trên GPU 1, mỗi thành phần một bản; batch hai phần được điều chỉnh riêng. Đây chủ yếu là phân bổ bộ nhớ, chưa làm các pha chạy đồng thời vì pipeline hiện gọi segmentation xong mới tới embedding. [Luồng upstream](https://github.com/BUTSpeechFIT/DiariZen/blob/main/diarizen/pipelines/inference.py).

Mức 2: giữ một worker điều phối, cho segmentation sản xuất từng lô window và embedding nhận các lô đã có mask cuối cùng. Hai pha chạy gối trong giới hạn queue. Chỉ triển khai nếu API backend cho phép tách các pha và tensor được chuyển đúng thiết bị.

Điều kiện bắt buộc của mức 2:

- Giữ lưới window gốc, hop, context, warm-up và local speaker slots; không chia waveform thành hai nửa rồi nối nhãn.
- Median filter phải có đủ ngữ cảnh tại biên xử lý; xét đúng trục mà phiên bản đang cài dùng, không làm filter lại trên batch bị thiếu biên.
- Embedding dùng đúng overlap exclusion/mask của segmentation và cùng quy tắc xử lý speaker không hoạt động.
- Gom kết quả theo `(window_id, local_speaker_id)` trước clustering; clustering/VBx nhìn toàn bộ embedding của file một lần.
- Giữ frame-level speaker count, reconstruct và timestamp như baseline; so kết quả sau căn hoán vị nhãn speaker.
- Lưu segmentation/embedding trung gian đủ để retry pha lỗi mà không phải làm lại file, nếu dung lượng cache cho phép.

Mức 1 phải ghi rõ “distributed placement”; chỉ công bố “overlapped execution” khi profiler xác nhận mức 2 chạy gối. Nếu không thể tách backend an toàn, dừng ở mức 1 và báo giới hạn; không tuyên bố đã dùng đồng thời hai GPU. Không nhân đôi DiariZen thành hai worker trong phạm vi thiết kế đã chốt.

## 9. Separation: CPU chuẩn bị, hai GPU tách, hậu xử lý có thứ tự

Tách đường xử lý thành ba pha:

```text
CPU window queue -> Sidon GPU 0 / Sidon GPU 1 -> raw tracks
  -> speaker assignment + continuity + QC + gain -> splice/commit
```

Sidon hiện tách mù, enrollment được dùng ở bước gán speaker sau đó. Vì vậy có thể chạy trước phần suy luận Sidon của các window độc lập mà chưa cập nhật enrollment từ job trước. Không song song hóa nguyên `separate_two_speakers` nếu trong đó còn trạng thái gán nhãn dùng chung.

Các sửa đổi:

- Window queue có giới hạn; trả descriptor hoặc output buffer có quản lý vòng đời thay vì giữ vô hạn audio trong future.
- Mỗi Sidon worker có process, pipe, temp directory, seed theo job/attempt và GPU mask riêng. Chỉ số trong worker một GPU là `cuda:0`.
- Raw track gắn sample rate thật; giữ bước đổi từ output Sidon về sample rate pipeline và kiểm tra số sample.
- Hậu xử lý commit theo thứ tự baseline trong từng file trước; mở rộng song song qua các file vì state tách biệt. Chưa đảo thứ tự các cặp speaker nếu enrollment memory có thể tạo phụ thuộc.
- Kiểm tra coverage ngay trước splice, tránh kết quả job cũ ghi đè retry đã thành công. Retry nhận job identity mới nhưng cùng mục tiêu và có giới hạn.
- Worker thứ hai chỉ được tạo khi có đủ việc và vừa VRAM; một window đơn lẻ không được nhân đôi để chạy cho đầy card.
- QC, similarity, continuity và gain vẫn được kiểm tra đầy đủ; không bỏ các bước đó để giảm thời gian.

Thư mục fail phải giữ overlap, window, tracks nếu có, quyết định biên, pad/context, GPU UUID, worker ID, attempt, seed, exception, thời gian và memory peak. Debug audio thành công có thể cấu hình riêng; metadata lỗi bắt buộc không bị tắt bởi chế độ tối ưu I/O.

### Giữ nhất quán VAD khi dựng window

Không chấp nhận việc bật CPU pool tự đổi Silero thành energy mà không khai báo. Phương án ưu tiên là chuẩn bị/caching bằng chứng acoustic để worker dùng chung với cùng policy. Cần kiểm tra theo đúng input interval: kết quả VAD toàn file không mặc nhiên tương đương với việc chạy VAD từng đoạn.

Nếu phải tạo session CPU Silero theo worker, tải artifact một lần có lock trước khi spawn, khống chế ORT threads và reuse session ở mỗi process. Nếu cần sửa semantics finder, tách version và đánh giá chất lượng trước; không nhập chung như một tối ưu thuần hiệu năng.

## 10. Nhạc và noise

SSLAM tạo cả music map và noise curves trong một lượt. Noise hiện phục vụ đánh giá/lọc; không thêm denoiser trong kế hoạch hiệu năng.

SSLAM: batch thực trên một GPU, CPU chuẩn bị mel trước một vài batch. Giữ normalization toàn file và lưới window, không tự normalize từng batch. Cache score theo input/model/preprocess fingerprint để thay ngưỡng không cần chạy lại model khi score cũ đủ thông tin.

RoFormer: nhiều vùng nhạc độc lập hoặc nhiều file có thể phân cho hai worker, mỗi card một bản. Nếu chỉ một vùng ngắn, dùng một worker. Với một vùng dài, giữ chunking/overlap-add thư viện trước; phân phối window nội bộ qua hai GPU là đợt riêng có kiểm tra seam và normalization.

Worker RoFormer chạy trong GPU mask riêng. Wrapper hiện dùng file tạm cố định `in.wav` trong work directory, nên không gọi cùng instance đồng thời. Mỗi worker có directory riêng; coordinator ghép patch theo sample index sau khi kiểm tra length và source reference.

Scheduler chọn một trong hai cách theo backlog: SSLAM file B chạy cùng RoFormer file A, hoặc hai RoFormer chạy các vùng độc lập sau khi scan đã đủ. Không cộng cả hai phương án khiến ba model tự tranh hai GPU.

Tách nhạc hoàn tất và patch/timeline được commit trước khi diarization file đó bắt đầu. Đo riêng decode 44.1 kHz, suy luận và encode; nguồn hi-res được giữ theo chính sách hiện tại. Không giảm overlap hoặc đổi sample rate chỉ để đẹp tốc độ.

## 11. CPU: cấp thread theo việc đang chạy

Từ ngân sách CPU thực tế, ưu tiên đường cấp dữ liệu cho GPU: decode/resample, mel/tokenize, window preparation và đọc batch. Export nhận phần dư nhưng có mức tối thiểu để không bị dồn mãi về cuối.

| Tác vụ | Phương án |
| --- | --- |
| NumPy/scipy/feature native | Thử thread pool nhỏ hoặc process tùy GIL và số liệu đo |
| Window logic Python nặng | Process pool bền theo stage; số process nằm trong quota |
| GPU worker | Ít CPU thread đủ feed batch, không tự lấy toàn bộ core |
| Tokenizer/ORT/CT2/BLAS | Ngân sách riêng, tính chung tổng thread runnable |
| FFmpeg/export | Hạn số tiến trình và thread mỗi encode, tránh nhiều FFmpeg cùng chiếm core |
| Scheduler/checkpoint | Giữ năng lực xử lý event, heartbeat và commit |

Không tạo một pool full-core trong mỗi model. Thiết lập OMP/MKL/OpenBLAS/ORT/CT2/tokenizer trước import hoặc lúc tạo session. Không kỳ vọng đổi env sau import làm mọi native pool tự đổi; điều chỉnh concurrency hoặc tái tạo worker khi cần.

Trên máy hai core có thể tắt window process pool và chỉ prefetch nhẹ. Trên máy nhiều core tăng dần theo GPU starvation và RAM, không mở `cores - 2` bất kể đã có bao nhiêu worker khác.

Tối ưu hotspot CPU sau profiling:

- Dựng chỉ mục interval theo speaker/timestamp một lần mỗi file để tra support, overlap và SP3; giữ nguyên tập candidate và thứ tự tie-break.
- Cache kết quả acoustic analysis theo audio version, interval, sample rate và policy; không tính lại cùng RMS/VAD cho từng job khi có thể tái sử dụng chính xác.
- Giữ metadata đã chuẩn bị theo file trong LRU của CPU worker, có giới hạn byte; tránh unpickle/dựng lại cấu trúc lớn ở mỗi window.
- Xử lý RMS/normalization theo block để tránh tạo nhiều mảng float64 lớn cho file dài; kiểm tra sai số và quyết định ngưỡng.
- Chuẩn bị feature/tokenizer một lần khi các consumer thực sự dùng cùng cấu hình; không bỏ tìm điểm cắt hay giảm candidate để đổi lấy thời gian.

Worker GPU sử dụng subprocess mới hoặc `spawn`; tránh fork từ tiến trình đã khởi tạo CUDA. CPU pool cũng cần entrypoint không chạy lại CLI/model loading khi import. Đây là yêu cầu của runtime PyTorch, không chỉ quy ước code. [PyTorch 2.8 multiprocessing](https://docs.pytorch.org/docs/2.8/notes/multiprocessing.html).

## 12. RAM, shared memory và audio cache

Một giờ mono float32 chiếm khoảng 330 MiB ở 24 kHz, 220 MiB ở 16 kHz; stereo float32 44.1 kHz khoảng 1.18 GiB. Nhiều bản waveform, feature, output và peak load model có thể lớn hơn trọng số trên GPU.

Đề xuất một audio store tái sử dụng mmap cache hiện có:

- Audio gốc đã chuẩn hóa là read-only; music patches, waveform sau cắt và separated tracks là các phiên bản có provenance riêng.
- Job gửi path/offset/shape/dtype/sample rate, không pickle cả waveform qua từng pipe.
- Chỉ buffer sắp chạy mới vào shared/pinned memory. Pinned memory có trần riêng vì không được swap; không pin cả file dài.
- Refcount/lease dữ liệu theo consumer, cả worker và export. Coordinator là chủ cleanup; worker không unlink buffer dùng chung.
- Mmap vẫn tiêu RAM qua page cache, shared memory vẫn tiêu RAM hệ thống. Tính theo cgroup và PSS/USS khi có, không cộng RSS các process để đếm trùng trang chia sẻ.
- Giới hạn số file đang xử lý, byte queue, số batch đã chuẩn bị, output đang chờ ghi và model đang load cùng lúc.
- Khi RAM vượt mức mềm: giảm prefetch, flush output/cache không cần, tạm dừng mở replica. Vượt mức cứng: dừng producer, tiếp tục consumer/commit để giải phóng bộ nhớ.
- Có fallback mmap khi `/dev/shm` nhỏ; không để thiếu shared memory làm chết cả job.

Cache 16 kHz có thể giảm resample lặp, nhưng phải giữ thứ tự dựng core/context, DC removal và normalization. Resample cả file rồi cắt không mặc nhiên giống resample từng đoạn; chỉ thay sau kiểm tra sample/timestamp và chất lượng. Không chia sẻ mel giữa hai ASR nếu feature extractor hoặc số mel khác nhau.

## 13. Prefetch, truyền GPU và I/O

Producer chuẩn bị một số batch có giới hạn trong lúc GPU chạy batch hiện tại. Dùng pinned buffers và chuyển bất đồng bộ khi backend hỗ trợ và phép đo cho thấy có lợi; không mặc định Python async đồng nghĩa copy và compute đã chồng lấp. [PyTorch CUDA](https://docs.pytorch.org/docs/2.8/notes/cuda.html).

Model weights/tokenizer/config được tải trước vào local cache với lock; worker động load từ cache. Cold download là thời gian riêng trong báo cáo và không được giả định miễn phí khi tính lợi ích replica. Giới hạn simultaneous model loads để tránh peak RAM/disk contention.

Qwen/Sidon có thể chuyển từ `.npy` mỗi job sang container mmap hoặc shared buffers sau khi có giao thức ownership. Giai đoạn đầu giữ file exchange nhưng batch nhiều job và directory riêng để đơn giản hóa kiểm chứng.

Không đổi codec artifact phục vụ đánh giá mà không khai báo. Encode MP3 cho review/export được cache theo nội dung và thông số; khi được yêu cầu cả trang portable có audio nhúng thì giữ hỗ trợ, nhưng tránh giữ đồng thời WAV, MP3, base64 và HTML toàn bộ trong nhiều bản RAM.

## 14. Checkpoint, lỗi và khôi phục

Một coordinator sở hữu journal trên local scratch; dùng SQLite transaction cho trạng thái job nếu cần khả năng resume nhỏ hơn stage. Worker ghi artifact tạm riêng, đóng/flush rồi trả descriptor; coordinator kiểm tra, atomic rename và commit. Thứ tự artifact trước, journal sau; sau crash đối soát orphan và record thiếu artifact. Không đặt SQLite WAL trên filesystem không hỗ trợ phù hợp.

Fingerprint gồm input identity, timeline/audio version, stage policy, model revision, preprocess, decoding/precision và các tham số ảnh hưởng kết quả. GPU ID và số worker thuần scheduling không tự làm mất cache. Nếu thay batch/backend làm thay semantics thì version tương ứng phải được phản ánh.

Các nguyên tắc:

- Không tin chỉ `exists(result.pkl)` là stage hoàn tất; có manifest/schema/checksum/commit marker.
- Ghi lại stage checkpoint hiện có từ các job đã commit để giữ khả năng đọc của các bước sau; có migration/adapter, không xóa cache cũ.
- Khi worker chết, thu hồi lease, chờ timeout hợp lý và retry chưa commit. Heartbeat/progress reader riêng để một generate dài không bị coi là process chết chỉ vì stdout chưa ra transcript.
- OOM giảm batch; batch hỏng có thể tách để xác định input lỗi. CUDA fatal cần restart worker; giới hạn retry và chống lặp vô hạn.
- Không tự skip segment, giảm model, giảm steps separation hoặc trả transcript rỗng để vượt qua lỗi.
- Kết quả GPU tới sau retry phải kiểm tra attempt generation trước commit. Mục tiêu là commit một lần, không hứa computation luôn chỉ chạy đúng một lần.
- Khi dừng chương trình: ngừng nhận job, drain có timeout, lưu phần đã commit và giải phóng buffer sau consumer cuối cùng.

## 15. Lịch nhiều file và stage cần hai GPU

Đợt đầu giữ `by_stage` để hạn chế phạm vi thay đổi, nhưng cho các queue nhỏ trong mỗi stage tự cân tải. Sau khi ổn mới chạy gối nhiều stage/file bằng dependency graph.

Stage entrypoint chỉ đọc dependency thật cần: refinement chỉ cần transcript thì không mở lại waveform, apply music patch và đọc mọi checkpoint âm thanh trước đó. Tránh xuất lại artifact các stage đã commit. Khi có client cũ còn trong service, chỉ rebind/start worker nếu stage tương ứng thực sự sắp chạy, không khởi động chỉ vì gặp một bước ở đầu `run()`.

Stage dùng hai card như refinement cần reservation theo nhóm: cấp cả hai cùng lúc hoặc chưa cấp; không giữ một GPU rồi chờ GPU kia vô hạn. Có giới hạn thời gian chờ/tuổi job, để ASR replica và file mới không liên tục đẩy refinement ra sau.

Chạy theo các đợt vừa đủ để khấu hao load model. Nếu LLM đã warm và có nhiều transcript chờ, ưu tiên hoàn tất đợt refinement; nếu ASR còn backlog lớn, replica có thể đáng giữ. Quyết định dựa trên chi phí toàn đường tới output, không chỉ thời gian của một stage.

CPU export có thể chạy song song refinement hoặc GPU stage file kế tiếp khi dữ liệu đã bất biến. Giới hạn lượng output chưa export để RAM/disk không tăng vô hạn. Không chạy hai full pipeline một cách mù trên cùng bộ GPU.

Caption server ngoài tiến trình chỉ tham gia scheduler nếu có API quản lý và đo tài nguyên rõ ràng; nếu dùng chung GPU, phải tính phần bộ nhớ nó chiếm. Không dừng server hay process GPU bên ngoài mà chương trình không sở hữu. Model phụ như embedder cũng phải có consumer thực tế mới được load.

Các preset hiệu chuẩn cần bàn giao:

| Phần cứng/workload | Chính sách bắt đầu |
| --- | --- |
| Một GPU | Một thiết bị, không replica đa card; batch theo VRAM, CPU/export chạy gối |
| Hai T4 hoặc hai card VRAM nhỏ | Hạn resident models; kiểm tra khả năng đồng trú trước khi chạy PhoWhisper cùng Qwen; replica chỉ sau khi trả VRAM |
| Hai A100 hoặc hai card VRAM lớn | Batch envelope lớn hơn sau đo; vẫn dành chỗ activation, không mặc định batch 256 là tối ưu |
| Hai GPU khác nhau | Batch và chia backlog theo tốc độ/VRAM từng card; không chia 50/50 cứng |
| CPU ít hoặc RAM nhỏ | Prefetch ít, ít file đang hoạt động, tránh nhiều worker làm GPU càng chờ input |
| Nhiều file ngắn | Ưu tiên giữ model warm và gom batch qua file có input tương thích |
| Một file dài | Ưu tiên microbatch, checkpoint nhỏ và hàng đợi có trần; không giữ mọi output trung gian trong RAM |

## 16. Precision và tối ưu kernel

Precision hiện có là baseline. Profile A100 float32 có cơ hội thử fp16/bf16 tùy backend, nhưng phải xem dtype thực tế, hỗ trợ phần cứng và kiểm tra output. T4 cần kiểm tra đường fp16/bf16 thực tế; không áp một dtype cho mọi model và mọi card.

Thứ tự thử: batching thật -> feed dữ liệu -> phân công GPU -> precision phù hợp -> attention/kernel -> compile. Không bật quantization/FP8/TF32 mới, đổi beam, số diffusion steps, VAD hop hoặc ngưỡng lọc cùng lúc với thay scheduler.

`torch.compile`, FlashAttention và CUDA graphs chỉ bật theo backend/shape đã xác minh; tính thời gian compile/warm-up vào cold run. File ngắn có thể không hoàn vốn. Prefix cache có kiểm thử độc lập về mask, vị trí token và tính cách ly giữa request.

Một model load được trên GPU không có nghĩa batch dài nhất chạy được. Cache batch envelope theo model revision, dtype, backend version, GPU và shape bucket; khi model cùng card thay đổi phải tính lại giới hạn.

## 17. Config đề xuất

Các khóa sau **chưa tồn tại trong chương trình**; đây là schema dự kiến. Giá trị số là điểm khởi đầu để hiệu chuẩn, không thay config hiện tại ở bước lập kế hoạch.

```yaml
performance:
  enabled: false                 # bật sau các cổng nghiệm thu
  scheduler: adaptive
  gpu_ids: auto                  # tối đa 2 GPU trong đợt đầu
  max_gpus: 2
  gpu_reserve_gib: 2.0
  gpu_reserve_fraction: 0.15     # dùng mức dự phòng lớn hơn
  allow_cpu_offload: false
  cpu_budget: auto
  cpu_reserve_for_coordinator: 1
  ram_soft_fraction: 0.75
  ram_hard_fraction: 0.85
  max_active_files: 2
  max_concurrent_model_loads: 1
  prefetch_batches_per_worker: 2
  max_pending_jobs: 32
  max_queue_bytes: auto          # chia từ RAM envelope
  pinned_memory_budget: auto
  scratch_budget: auto
  telemetry_interval_seconds: 1
  replica_min_gain_seconds: 15
  replica_cooldown_seconds: 60
  retry_limit: 2
  warm_model_policy: cost_based
  stages:
    asr:
      true_batching: true
      dynamic_replicas: true
      max_replicas_per_model: 2
      batch_limits: auto         # per-model items/frames/tokens
      batch_wait_ms: 50
      checkpoint_per_voter: true
    refinement:
      workers: 1
      placement: sharded
      batch_limits: auto
      prefix_cache: false
    diarization:
      workers: 1
      placement: split_components
      overlap_segmentation_embedding: false  # cổng nâng cấp riêng
      segmentation_batch: auto
      embedding_batch: auto
    separation:
      max_workers: 2
      window_cpu_workers: auto
      ordered_postprocess: true
    music:
      tagger_workers: 1
      max_separator_workers: 2
      cross_file_overlap: false   # bật sau scheduler nhiều stage
    export:
      workers: auto
      max_pending_bytes: auto
```

`cpu_reserve_for_coordinator` là ngân sách lịch chạy, không bắt buộc pin độc quyền một core trên máy chỉ có một hoặc hai core. Auto RAM/queue/pinned/scratch phải resolve ra con số trong log trước khi dispatch; không dùng `auto` như giá trị vô hạn.

`max_replicas_per_model=2` là trần, không phải luôn mở sáu worker ASR. Tổng worker còn chịu budget cả hai GPU, CPU, RAM và lợi ích dự báo. GPU explicit trong cấu hình người dùng được ưu tiên hơn tự khám phá.

Config performance mới có schema validation; chỉ các field runtime thực sự hỗ trợ mới nhận. Log effective config và lý do fallback. Mỗi chế độ có cờ rollback; policy `keep_models` cần được làm rõ: compatibility mode giữ nghĩa cũ, adaptive mode sử dụng warm policy được người dùng chọn, không âm thầm bỏ qua cờ.

## 18. Chỉ số và báo cáo bắt buộc

| Nhóm | Các số cần lưu |
| --- | --- |
| Toàn chương trình | Wall time, audio-hours/hour, số file/segment hợp lệ, cold/warm thời gian |
| GPU | UUID/model đang ở card, load/warm/infer/wait, peak memory, utilization lấy mẫu, power/throttle nếu đọc được |
| CPU | Affinity/quota, worker/thread budget, thời gian feature/window/tokenize/encode, run queue/context switch khi có |
| RAM | Cgroup current/limit, available, peak, swap/page faults, queue bytes, shared/pinned buffers |
| I/O | Decode/resample/read/write/encode time, bytes, scratch còn lại, cache hit/miss |
| Job | Pending/running/committed/retry/failed theo model/stage, input shape, latency p50/p95 |
| Scheduler | Vì sao mở/không mở replica, gain dự báo, gain thực tế ước lượng, cooldown, budget từ chối |
| Chất lượng | Coverage, missing/duplicate ID, WER/CER, diarization và separation QC, fail reasons |

Artifact dự kiến: `performance_summary.json`, `scheduler_events.jsonl`, `resource_samples.jsonl`, bảng per-stage và per-model. Rotating/chunked log tránh file event vô hạn. Đồng hồ đo GPU phải phản ánh hoàn thành thực tế; không gọi synchronize toàn GPU ở mọi job chỉ để đo, vì có thể phá tính chồng lấp.

Phân biệt GPU không có việc đủ điều kiện, đang chờ CPU, chờ I/O, chờ phụ thuộc, đang load model và đang đợi layer của GPU khác. Utilization lấy mẫu thấp chưa đủ để kết luận được quyền mở worker mới.

## 19. Kiểm thử và tiêu chí nghiệm thu

### 19.1 Bộ dữ liệu và baseline

Khóa commit/config/model revision/interpreter/dependency/hardware. Chọn file ngắn, podcast dài, nhiều overlap/backchannel, nhiều nhạc, ít nhạc, nhiều speaker và batch nhiều file. Dùng các kết quả review đã có nhãn để đo ASR; không coi output cũ là ground truth.

Baseline gồm code hiện tại, code mới một GPU và code mới hai GPU trên cùng input. Đo cold load và warm load riêng; benchmark ngắn lặp tối thiểu ba lần, soak test dài tối thiểu một lượt đủ vượt vòng đời nhiều file. Ghi overhead instrumentation và thời gian tạo artifact yêu cầu.

### 19.2 Kiểm thử điều phối bằng model giả

- Một worker nhanh xong, replica model chậm được nạp và chỉ lấy job chưa cấp; trường hợp backlog nhỏ không mở replica.
- Worker crash/load fail/OOM/timeout/response muộn; không mất job, không commit trùng, không bỏ phần đang chạy.
- Chạy một GPU, hai GPU, mask đảo thứ tự, GPU khác VRAM, hết RAM hoặc scratch; fallback ghi rõ.
- Job hai GPU không deadlock, không bị starvation khi ASR tiếp tục sinh việc.
- Restart coordinator khi artifact đã ghi mà journal chưa commit và chiều ngược lại; đối soát đúng.
- Queue/backpressure và buffer lease vẫn đúng khi consumer chậm hoặc export lỗi.

### 19.3 Đối chiếu output

- Tất cả ID phải đủ, không trùng; không lệch ánh xạ file/speaker/sample rate; timeline và số sample qua assemble phải đúng.
- ASR batch so với single có kiểm tra core/context thực tế, language/task, timestamp và cả ba voter. Sai khác floating-point được đo WER/CER, không giả định bitwise equality.
- Diarization căn hoán vị nhãn trước so DER/JER, overlap và độ lệch biên; không chấp nhận tăng tốc bằng đổi hop/segmentation.
- Separation cố định seed theo job khi backend hỗ trợ, so coverage/QC và nghe vùng khó; không hứa waveform bitwise khi kernel/batching khác.
- Nhạc kiểm tra độ dài, mức âm, seam và đúng vocal stem; noise map vẫn cùng timeline gốc.
- Refinement giữ guard và tỷ lệ hỏng/từ thêm/xóa; xuất đúng audio version. Export hiện có đường cắt mixture cho MP3, cần giữ provenance rõ và không đánh đồng với separated stem.

Cổng chất lượng: không có lỗi cấu trúc; mọi thay đổi chất lượng có ý nghĩa phải được đánh giá trên dữ liệu có nhãn trước bật mặc định. Sai khác chỉ do precision được báo riêng để người dùng biết trade-off, không hòa vào kết quả scheduler.

Cổng hiệu năng: nhanh hơn baseline vượt nhiễu đo trên workload được chọn, không OOM không xử lý, RAM không tăng vô hạn, worker không churn. Có thể đặt mục tiêu ban đầu giảm 10-15% thời gian toàn pipeline để đáng triển khai rộng, nhưng đây là mục tiêu cần chứng minh, không dự báo đã đạt. Chưa có cơ sở hứa tăng tốc 2x/5x.

## 20. Lộ trình triển khai có thể nghiệm thu từng đợt

| Đợt | Công việc | Điều kiện hoàn thành |
| --- | --- | --- |
| P0 | Instrument baseline, inventory model/device/CPU/RAM; khóa phiên bản benchmark | Có báo cáo nút nghẽn và cold/warm load thật |
| P1 | Sửa GPU index/mask/provider; model lifecycle refinement; quota CPU; queue window có trần; atomic checkpoint | Vòng đời, resume, memory ổn trên đường chạy hiện tại |
| P2 | Giao thức job/batch, journal và audio descriptors; batching ASR thật trên một GPU | Output mapping và chất lượng đạt, batch có nhiều item thực |
| P3 | Scheduler hai GPU, leases, reservation, replica ASR động | Minh chứng GPU rảnh giúp model chậm; backlog nhỏ không mở replica |
| P4 | Refinement một model sharded, token batch, scope nhiều file | Device map đúng, memory đủ, output guards đạt |
| P5 | Hai Sidon worker, raw inference song song, hậu xử lý đúng thứ tự | Retry/coverage/continuity và enrollment không hồi quy |
| P6 | RoFormer workers, SSLAM prefetch, cache/encode nhạc-noise | Patch/timeline đúng và có lợi trên file nhiều nhạc |
| P7 | DiariZen split components, tiếp đến thử segmentation/embedding chạy gối | Pha 1 placement đúng; pha 2 chỉ đạt khi có profiler và quality parity |
| P8 | Lịch nhiều stage/file, fair scheduling stage hai GPU, CPU export chồng lấp | End-to-end throughput tăng, không starvation/memory leak |
| P9 | Precision/kernel/compile theo backend và phần cứng đã đo | Có A/B chất lượng và cold/warm performance riêng |
| P10 | Soak, resume, tài liệu vận hành và rollout | Có preset phần cứng, fallback/rollback và artifact kiểm chứng |

Mỗi đợt có flag và phần sửa giới hạn; đánh dấu hoàn thành bằng bằng chứng test/benchmark. P2-P3 là phần ưu tiên cao vì trực tiếp đáp ứng yêu cầu ASR tận dụng card rảnh. Không chờ refactor toàn bộ chương trình mới được đo lợi ích.

## 21. Các vùng code cần thay đổi khi được duyệt

| Nhóm | Điểm tích hợp |
| --- | --- |
| Scheduling và config | `main.py`, `config.json`, `utils/batch.py`, `services/pipeline_service.py`; thêm module scheduler/resource/registry theo trách nhiệm |
| Worker runtime | `services/base_worker_service.py`, các worker service và script; adapter protocol dùng lại thay vì tạo framework mới |
| ASR | `services/asr_service.py`, `models/whisper.py`, `models/whisper_wrapper.py`, `models/phowhisper.py`, `models/qwen3_asr.py`, `qwen3_worker.py` |
| Refinement | `services/diarization_refinement_service.py`, lifecycle trong pipeline, worker entrypoint nếu cô lập process |
| Diarization | `diarizen_worker.py`, client/service; adapter backend cho các pha với version pin |
| Separation | `services/separation_service.py`, `models/bss_model.py`, `models/separation_backends.py`, Sidon worker/service, WeSpeaker provider |
| Nhạc/noise | `models/sslam.py`, `models/bs_roformer.py`, `services/music_service.py`; giữ noise semantics |
| CPU/RAM | `utils/cpu_plan.py`, `utils/window_pool.py`, `services/audio_service.py`, descriptor/lifetime storage |
| Durability/output | `utils/checkpoint.py`, progress/batch resume, `services/export_service.py`, stage output và review generation |
| Kiểm chứng | Test scheduler/batch/retry bằng model giả và benchmark runner GPU có manifest |

Trước mỗi đợt đọc lại diff hiện tại để giữ thay đổi của người dùng. Không tự cập nhật toàn bộ dependency; main, DiariZen, Qwen3 và Sidon có môi trường riêng, cần compatibility matrix/version capture theo worker.

## 22. Phạm vi duyệt và kết quả bàn giao

Đề nghị duyệt kiến trúc scheduler chung, batching thật và replica ASR động; giữ các ràng buộc số bản model của refinement/diarization đã nêu. Lịch nhiều stage và tối ưu precision là các đợt có cổng kiểm chứng, không bật đồng loạt ngay lần đầu.

Kết quả bàn giao sau triển khai gồm code, config schema/preset, báo cáo benchmark, log quyết định replica, báo cáo chất lượng, cách resume và cờ quay về chế độ cũ. Nếu chưa có máy hai GPU để đo, chỉ được đánh dấu phần code và test giả hoàn thành; mục performance/GPU validation phải để chưa hoàn thành.

Checklist hiện tại:

- [x] Đọc luồng và lập kế hoạch dựa trên code đang có.
- [x] Làm rõ worker rảnh, VRAM trống, model replica và model sharding là các khái niệm khác nhau.
- [x] Thiết kế ASR giúp nhau theo backlog, bộ nhớ và thời gian nạp.
- [ ] Người dùng duyệt kế hoạch/phạm vi triển khai.
- [ ] P0-P10 triển khai và nghiệm thu theo từng đợt.

Tài liệu này không thay runtime, không chạy GPU, không thay threshold/window policy và không tuyên bố đã tăng tốc.
