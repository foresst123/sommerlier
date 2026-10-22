# Thiết kế train full-duplex từ dữ liệu Sommelier trên 2 A100

Ngày đối chiếu: 2026-09-28. Trạng thái: **đề xuất để duyệt, chưa triển khai trainer, chưa train**.
Code pipeline được đối chiếu tại commit `995e063`. Đi cùng [kế hoạch triển khai](full_duplex_training_implementation_plan.md).

## 1. Kết luận và phạm vi

Đề xuất xây một nhánh training độc lập, nhận output pipeline hiện tại, thử nghiệm **Moshiko BF16 + LoRA + FSDP trên hai A100** trước. Đây là baseline có đường đi từ dữ liệu hai track đến trainer công khai rõ nhất, không phải kết luận Moshi là model tốt nhất cho tiếng Việt.

Điều kiện chọn model cuối cùng: dữ liệu đủ sạch; codec giữ được tiếng Việt; tokenizer không làm mất chữ/dấu; model học được tiếng Việt trong pilot; chất lượng hội thoại và tốc độ streaming đạt yêu cầu. Nếu không qua các điều kiện này, dừng mở rộng Moshi, đánh giá Lychee-FD bằng cùng bộ kiểm thử. Không giải quyết thất bại bằng cách cứ train thêm nhiều giờ.

Giả định sản phẩm: trước tiên học hội thoại tiếng Việt tự nhiên từ podcast, sau đó thích nghi thành trợ lý có thể nghe khi nói, nhận ngắt lời, phản hồi ngắn và tiếp tục đúng ngữ cảnh. Nếu chỉ cần mô phỏng hai người nói chuyện thì bỏ giai đoạn trợ lý, không gọi checkpoint đó là trợ lý hoàn chỉnh.

Ràng buộc giữ nguyên:

- Chỉ dùng hai A100; chưa xác nhận 40GB hay 80GB, nên thiết kế có hai profile.
- Không hạ Torch hiện tại, không dùng INT4/INT8/FP8 để ép vừa VRAM; giữ checkpoint BF16 và đầy đủ codebook của checkpoint.
- Không tự động giảm độ dài hội thoại, bỏ overlap, xóa khoảng im lặng hoặc giảm chất lượng audio để đạt tốc độ.
- LoRA là chiến lược thích nghi tham số, **không có bảo đảm bằng hoặc hơn full fine-tuning**. Chỉ nhận checkpoint nếu qua đánh giá chất lượng; nếu không đạt thì báo giới hạn tài nguyên/dữ liệu.
- Không train foundation model từ đầu. Không tự tạo tokenizer/codec/backbone mới ở phiên bản đầu.
- Không sửa thứ tự stage pipeline trong hạng mục này. Output cuối đã relabel/align là đầu vào; trainer không phụ thuộc việc stage được chạy bằng thứ tự nào.
- Lượt hiện tại chỉ tạo kế hoạch, không tải bộ weights lớn, cài môi trường, chạy GPU, commit hay push.

## 2. Tài liệu đã dùng và cách lựa chọn

Đã rà danh mục [Awesome-Full-Duplex-SDM](https://github.com/Ruiqi-Yan/Awesome-Full-Duplex-SDM), đọc các phần kiến trúc/training/evaluation liên quan trong paper Moshi, Sommelier, Lychee-FD và đối chiếu repo/model card/trainer. Không coi danh mục là bằng chứng rằng mọi mục đều có code train, và không tuyên bố đã đọc trọn mọi paper trong danh mục.

| Hướng | Bằng chứng đã kiểm tra | Quyết định cho dự án |
|---|---|---|
| Moshi/Moshiko | Có trainer LoRA chính thức, reader audio hai kênh và FSDP | Baseline triển khai đầu, phải qua kiểm tra tiếng Việt |
| Lychee-FD | Có `Training_package`, loader/control-token/backchannel; paper khoảng 10B, thí nghiệm 8 H20 | Phương án đối chứng; chưa chứng minh train recipe gốc vừa 2 A100 |
| PersonaPlex | Có code/model phục vụ hội thoại có role/voice conditioning | Đối chứng inference; chưa chọn làm trainer mặc định |
| BayLing-Duplex, Realtime-Venus | Các repo đã xem có kiến trúc/checkpoint/inference | Không suy từ “code + weights” thành trainer có thể chạy ngay; cần spike riêng |
| DuplexGen | Có pipeline tạo dữ liệu human–AI | Tham khảo bổ sung dữ liệu trợ lý, không thay thế hội thoại thật |
| Full-Duplex-Bench | Có kiểm thử offline overlap và tương tác nhiều lượt | Tái sử dụng protocol, thêm test tiếng Việt riêng |

Nguồn trực tiếp: [Moshi trainer](https://github.com/kyutai-labs/moshi-finetune), [Lychee trainer](https://github.com/HITsz-TMG/Lychee-FD/tree/main/Training_package), [Lychee paper §4.3](https://arxiv.org/html/2607.06540v1), [PersonaPlex](https://github.com/NVIDIA/personaplex), [BayLing](https://github.com/BayLing-Models/BayLing-Duplex), [Realtime-Venus](https://github.com/inclusionAI/Realtime-Venus).

Một kết quả đặc biệt liên quan: paper Sommelier đã fine-tune Moshiko bằng khoảng **83 giờ**, LoRA rank 128, trên **8 A100**. Tác giả báo cáo rủi ro model trở nên không phản hồi khi một speaker giữ lượt quá lâu; thử nghiệm chọn vùng có ít nhất ba lượt liên tiếp, mỗi lượt không quá 10 giây. Đây là căn cứ tạo một subset tái lập, **không phải quy tắc xóa mọi lượt dài khỏi dữ liệu của mình**, cũng không chứng minh hiệu quả tiếng Việt. Nguồn: [Sommelier §3.1 và Appendix D](https://arxiv.org/pdf/2603.25750).

Một mâu thuẫn release cần ghi rõ: [DuplexGen README](https://github.com/duplexgen/duplexgen-code) dẫn sang trainer PersonaPlex, nhưng [repo trainer được dẫn](https://github.com/duplexgen/personaplex-finetune) tại thời điểm kiểm tra vẫn ghi code sắp phát hành. Không đưa dependency chưa xác minh đó vào critical path.

## 3. Dữ liệu pipeline hiện tại: có gì, còn thiếu gì

### 3.1 Hai định dạng phải hỗ trợ

**Clean two-channel dataset**: cấu hình A100 trỏ tới `/home/lamkd2/output_fullduplex_data`. Mỗi item có `sp1_clean.wav`, `sp2_clean.wav`, `audio_2ch.wav`, `metadata.json`; metadata có source/item ID, thời gian, `orig_spans`, mapping speaker/kênh, transcript/word timestamps và separation failure spans.

**Conversation exports**: mỗi item có `mixture.wav`, `speaker_A.wav`, `speaker_B.wav`, `stereo_2ch.wav`, `conversation.json`, `transcript.txt`; có score/tier, đánh giá semantic và mô tả phương pháp tạo hai kênh.

Hai nơi không mặc nhiên chứa cùng tập conversation: clean exporter lấy candidate theo logic chọn của nó; không mặc nhiên đã qua cùng quyết định semantic LLM của conversation exporter. Nếu cùng một đoạn gốc xuất ở cả hai nơi, chỉ tính một nguồn dữ liệu và dùng bản đạt QC tốt hơn, giữ liên kết provenance.

Bằng chứng code:

- `podcast-pipeline/services/clean_two_channel_dataset_service.py`: `_reserve_source`, `_conversation`, `_failure_rows`, phần ghi metadata/audio.
- `podcast-pipeline/services/conversation_export_service.py`: `_speaker_tracks`, `_metadata`, `channels_2ch`.
- `scripts/build_training_conversations.py`: script legacy đọc JSONL từng lượt; không phải adapter trực tiếp cho hai định dạng trên, cũng chưa phải trainer full-duplex.

Đã kiểm tra contract bằng code; **chưa có corpus audio export thực tế trong workspace để nghe, thống kê số giờ hoặc đo lỗi**. Số giờ train hợp lệ phải đợi audit trên máy lưu dữ liệu; không lấy số giờ input pipeline làm số giờ train.

### 3.2 Bốn rủi ro bắt buộc xử lý

1. `channels_2ch.method == time_gated`: đây là mixture mở theo thời gian speaker nói; khi overlap, hai track có thể cùng chứa hai giọng. Không đưa vào tập strict full-duplex. Giữ trong quarantine để sửa/re-export, không xóa bản gốc.
2. `strict_separation_tracks` vẫn có thể chứa vùng separation thất bại bị thay bằng zero. `failed_separation_spans[].zeroed_in_strict_track` phân biệt trường hợp này. Phiên bản đầu loại **window giao vùng hỏng** khỏi tập train chính, kể cả chỉ user track hỏng: masking target không sửa được input conditioning sai. Không coi zero giả là im lặng thật.
3. Refinement có thể sửa câu thành văn viết, mất “ừ/ờ”, lặp từ hoặc thêm ý. Cần kiểm chứng lời nói–text, giữ text verbatim riêng. Không dùng transcript đã được LLM viết lại nhưng không đúng audio làm gold; không đưa `<think>`/lý do LLM vào target.
4. `orig_spans` có thể cho thấy timeline đã cắt nối. Không dạy model rằng hai đoạn bị cắt là hội thoại liên tục. Split/reset ở seam, hoặc phục hồi khoảng gốc nếu có nguồn; phải lưu provenance và re-align nếu audio/text đổi.

Ngoài ra: IDs speaker là local theo recording, không phải định danh một người trên toàn corpus; source ID số thứ tự phụ thuộc registry/path, không đủ để chống trùng giữa hai máy hay hai lần export.

## 4. Kiến trúc data → training

```text
Hai loại export (read-only)
       ↓
Import + canonical manifest + xác minh provenance
       ↓
QC / quarantine / deduplicate / chia source-group train-dev-test
       ↓
Window liền mạch + role view (assistant/user) + masks
       ↓
Codec/text cache có version và checksum
       ↓
Train FSDP trên GPU 0 + GPU 1
       ↓
Dev offline + streaming → chọn checkpoint → test khóa kín
       ↓
Artifact model + resume state + data/model cards + demo realtime
```

### 4.1 Canonical manifest v1

Mỗi record phải có những nhóm trường sau, schema typed và validate ngay lúc import:

| Nhóm | Trường bắt buộc hoặc trạng thái thiếu rõ ràng |
|---|---|
| Identity | `schema_version`, `conversation_id`, `source_uid`, `source_group_id`, `export_kind`, `source_metadata_path`, checksums |
| Provenance | source original path/URL nếu có, pipeline commit/config hash, export revision, rights/consent status |
| Audio | 2 channel paths/index, sample rate, sample count, mapping local speaker, track method |
| Timeline | clip-relative times, source range, `orig_spans`, seams; không tự cộng offset hai lần |
| Speech | turns, words, verbatim text, refined text nếu có, alignment/text provenance và confidence |
| QC | separation failure spans, noise/music, missing fields, accept/quarantine reasons |
| Split | train/dev/test, group/dedup cluster, split version |
| Train view | target speaker, input speaker, window sample bounds, conditioning/target masks, role source |

Thiếu provenance không được điền giá trị bịa. `unknown` không đồng nghĩa `passed`. Media đọc trực tiếp hoặc tạo cache ở thư mục mới; không ghi ngược lên export.

### 4.2 QC và chống leakage

- Kiểm tra file tồn tại/đọc được, hữu hạn, đúng 2 kênh hoặc 2 mono tương ứng, cùng clock, đúng sample count; stereo khớp mono. `duration = num_samples / sample_rate` là chuẩn, cho phép sai số rounding đã định nghĩa ở metadata.
- Kiểm tra timestamps trong clip, words theo thứ tự **trong từng track**, không cấm overlap giữa hai người. Timestamp rounded 3 chữ số không có độ chính xác thật tới một sample; log dung sai thay vì sửa âm thầm.
- Kiểm tra speaker leakage bằng tổ hợp nghe mẫu, ASR trên từng track, embedding/confidence, energy overlap. Không dùng similarity separation hiện tại làm chứng nhận “sạch tuyệt đối”. Không dùng SI-SDR/PESQ với mixture làm ground truth khi không có clean reference.
- Nhận diện duplicated windows bằng source UID + khoảng gốc; với các bản re-upload dùng fingerprint/near-duplicate cluster. Chọn representative bằng QC, giữ audit log.
- Chia nhóm **trước khi crop, swap role, augment**; tạm dùng 90/5/5 theo nhóm nguồn, điều chỉnh để dev/test đủ đa dạng. Tách recording/episode và near-duplicate là bắt buộc; nếu khả thi tách show/host/speaker, báo riêng speaker-seen/unseen. Không được khẳng định speaker-disjoint nếu chưa có bằng chứng.
- Bản đầu: tập chính chỉ lấy strict tracks, không giao failure/seam, đủ alignment, lời nói sạch và verbatim đã kiểm chứng mức phù hợp. Tập nghi ngờ giữ riêng, không silently drop.
- Chỉ import export đã hoàn tất: kiểm tra source registry/manifest completion và checksum, bỏ qua staging/source đang `building`. Snapshot phải gắn cùng revision audio–relabel–word alignment; không ghép words của bản cũ với text/speaker đã sửa.
- Tạo subset QC theo accent, noise, overlap, turn length, topic; chống dữ liệu một chương trình lấn át.
- Khởi đầu nghe kiểm tra 100–200 clip phân tầng, cộng các trường hợp bị flag; đây là budget audit đề xuất, không bảo đảm một mẫu đủ đại diện toàn corpus.

### 4.3 Cách biểu diễn full-duplex

Chọn một người làm target/assistant, người còn lại làm user; giữ cả hai track trên **cùng một trục thời gian**. Khi người target không nói, target audio là im lặng thật ở thời điểm tương ứng. Không nối các câu target lại với nhau, không trộn hai người thành mono, không đổi thành QA luân phiên.

Moshi dùng Mimi 24kHz, nhịp 12.5 frame/giây (80ms); lấy codebook count/token IDs/delay pattern từ checkpoint, không hard-code cho mọi backend. Text target là lời target nói, không phải toàn transcript của hai người. Học text/audio theo mask/delay chính thức; không tự dịch target một frame cho “có vẻ đúng”. Nguồn kiến trúc: [Moshi §3](https://arxiv.org/html/2410.00037v2).

Với hội thoại người–người, có thể sample A→target hoặc B→target theo epoch; là hai view của cùng record, **không nhân đôi số giờ dữ liệu độc lập**. Với human–assistant SFT thì khóa đúng vai; không swap ngẫu nhiên khiến model học bắt chước người dùng.

Training prediction giữ tính causal theo delay pattern. Teacher-forced target audio quá khứ là hợp lệ; target tương lai hoặc full transcript làm prompt conditioning là leakage. Việc ground truth dùng để tính loss không cho phép cung cấp toàn bộ nó cho inference.

Loss v1 kế thừa audio/text CE của trainer chính thức, phân biệt: audio silence thật, text padding, padding ngoài window, vùng không có nhãn. Không thêm loss “phạt mọi overlap” hoặc bắt nhường lời theo VAD cứng. Xem [training loop](https://github.com/kyutai-labs/moshi-finetune/blob/main/train.py) và [interleaver](https://github.com/kyutai-labs/moshi-finetune/blob/main/finetune/data/interleaver.py).

### 4.4 Cache để GPU không chờ encode lại

Mimi frozen: encode offline một lần cho mỗi window/channel, lưu token integer, valid lengths và text alignment. Cùng track+window có thể dùng lại khi đổi vai. Cache key phải có waveform hash, crop samples, codec/checkpoint hash, codebook count, dtype/mode, resampler version, context/reset policy; text cache còn có tokenizer/alignment revision.

**Không giả định** encode cả file rồi cắt token bằng encode từng crop: codec có context và trạng thái. Bản đầu cache đúng từng window với cùng context/reset policy như baseline; test offline/streaming boundary và cached/on-the-fly trước khi tối ưu thêm. Nếu augment audio, cache của bản augment là artifact khác.

Ưu tiên sharded files/mmap + index nhỏ trên NVMe; tránh mở hàng triệu file WAV nhỏ mỗi step. Một writer cho mỗi shard, ghi atomic/checksum; cache hỏng fail rõ ràng. Không tự giảm codebook hay bitrate để tiết kiệm dung lượng.

## 5. Gate tiếng Việt trước khi train dài

[Moshiko model card](https://huggingface.co/kyutai/moshiko-pytorch-bf16) công bố English; [Lychee model card](https://huggingface.co/HIT-TMG/Lychee-FD) khai báo Chinese/English. Không có cơ sở từ hai card này để hứa chất lượng tiếng Việt.

Pilot phải tách ba vấn đề:

1. **Codec**: encode/decode audio Việt có dấu thanh, vùng miền, tên riêng, nói nhanh, overlap; nghe mù original/reconstruction và đo ASR CER hỗ trợ. Nếu codec làm hỏng phát âm rõ rệt, fine-tune LM đơn thuần không phải cách chữa đáng tin.
2. **Tokenizer/alignment**: kiểm tra round-trip Unicode NFC, `<unk>`, số token mỗi từ/giây, token bị interleaver thay thế/tràn cuối window, và độ trễ text so với audio. Ở nhịp cố định, tiếng Việt phân mảnh nhiều có thể không đủ slot text. Không xóa dấu/chữ hay âm thầm bỏ token. Điều chỉnh interleaver chỉ sau test; nếu cần đổi vocab/kiến trúc thì đó là nhánh R&D riêng phải duyệt lại.
3. **Khả năng học ngôn ngữ**: overfit 8–16 ví dụ sạch để chứng minh gradient/target đúng; tiếp theo pilot 5–20 giờ đa dạng, có dev chưa thấy. Overfit không chứng minh generalization. Chỉ scale khi cải thiện dev lẫn nghe live; có thể cần LoRA + train embedding có kiểm soát, hoặc cần chọn base khác.

Không thay text tiếng Việt thành bản dịch tiếng Anh trong khi audio vẫn tiếng Việt. Không đổi base language model bên trong Moshi thành Qwen3.5 rồi coi là tương thích checkpoint.

## 6. Train gì từ corpus đang có

### Giai đoạn A — domain/conversation adaptation

Mục tiêu: tiếng Việt, phân bố timing, backchannel, overlap thật, phản ứng qua lại. Tạo hai view dataset để ablation:

- `interaction_short_turns`: vùng liền mạch ≥3 lượt, mỗi lượt ≤10 giây, nhằm tái lập ý tưởng Sommelier.
- `interaction_balanced`: dữ liệu sạch rộng hơn, gồm short turn, long listening, pause tự nhiên, overlap và các lượt dài hợp lệ; sampling cân bằng có thống kê.

Không hard-delete lượt dài. So sánh khả năng nói/giữ im lặng/nghe lâu giữa hai recipe. Không chỉ lấy conversation có score cao hoặc nhiều overlap nhất: model còn phải biết không ngắt lời.

### Giai đoạn B — assistant SFT

Podcast chưa đủ ground truth cho instruction following, nhường lượt khi bị yêu cầu dừng, không phản hồi tiếng nền, sửa câu trả lời khi nghe thêm ý, kiến thức đúng và giọng nhất quán.

Đề xuất bổ sung pilot 5–20 giờ human–assistant có quyền sử dụng: hỏi đáp nói tự nhiên, yêu cầu sửa, ngắt lời đổi ý, người dùng “ừ” không giành lượt, im lặng suy nghĩ, tiếng nền/side conversation. Ưu tiên ghi hai mic/channel độc lập. Nếu dùng synthetic, phải kiểm tra tiếng Việt của TTS, role, timing và quyền giọng; không giả định ghép overlap tùy tiện tạo ra phản ứng causal đúng.

Khởi điểm thử mix theo **audio-seconds hữu dụng**: 70% human–human sạch + 30% assistant, đối chứng 50/50; đây là giả thuyết cần dev, không tỷ lệ đã được paper chứng minh. Nếu chưa có dữ liệu assistant, chỉ hoàn tất A và ghi đúng giới hạn checkpoint. Giọng trợ lý riêng cần dữ liệu/conditioning phù hợp, không mặc nhiên có sau role swapping podcast.

Tham khảo phương pháp tạo dữ liệu và calibration theo tình huống của [DuplexGen](https://github.com/duplexgen/duplexgen-code); không nhập toàn bộ corpus chưa kiểm tra ngôn ngữ/giấy phép. RL/preference tuning để sau khi có baseline ổn định và preference/rollout data thực, không nằm trên critical path v1.

## 7. Hai GPU hoạt động như thế nào

Không chạy vLLM/ASR/separation cạnh training trên cùng hai GPU trong run chuẩn. Chờ job pipeline đến điểm checkpoint an toàn, xác minh GPU trống, không tự kill tiến trình chưa biết chủ sở hữu.

| Chặng | GPU 0 | GPU 1 | CPU/NVMe |
|---|---|---|---|
| Audit | Có thể không cần GPU | Có thể không cần GPU | Import, validate, dedup, split |
| Codec cache | Mimi worker cho shard chẵn | Mimi worker cho shard lẻ | Decode/prefetch/atomic shard writes |
| Train | Rank 0: shard model và microbatch riêng | Rank 1: shard model và microbatch riêng | Prefetch token cache, metrics/checkpoint |
| Dev loss | Hai rank tham gia đồng bộ | Hai rank tham gia đồng bộ | Tổng hợp theo valid token count |
| Streaming eval | Sau khi checkpoint, inference model | Có thể chạy scorer riêng nếu vừa bộ nhớ | Audio replay/live transport, timestamps |

Trong train, hai GPU **hợp tác train một model** qua FSDP, không phải GPU 0 chỉ nghe và GPU 1 chỉ nói, cũng không phải hai vLLM replicas như pipeline xử lý dữ liệu. Trainer upstream dùng FULL_SHARD khi nhiều GPU: [wrapped_model.py](https://github.com/kyutai-labs/moshi-finetune/blob/main/finetune/wrapped_model.py).

Tối ưu không đổi dữ liệu/precision: cache frozen codec, bucket theo frame length, pinned-memory prefetch có giới hạn, giữ model trên GPU giữa step, checkpoint theo block, kernels attention được kiểm thử, giảm logging đồng bộ. Không tăng worker vô hạn: thử loader 2/4/8 **mỗi rank**, OMP/MKL khoảng 1–2 thread/worker ban đầu; chọn bằng data-wait/CPU/IO measurements. Không cố nhét nhiều training processes độc lập vào một GPU.

### 7.1 Khởi điểm cấu hình — cần benchmark, chưa phải kết quả đo

| Tham số | 2×A100 40GB | 2×A100 80GB |
|---|---|---|
| Base | Moshiko BF16 | Như bên trái |
| Phân tán | FSDP 2 rank | FSDP 2 rank; so sánh DDP chỉ nếu thực sự vừa và đã có implementation |
| Trainable | LoRA rank 128; frozen base/codec | Giữ cùng recipe để so chất lượng/tốc độ |
| Microbatch | Bắt đầu 1 window/rank | Bắt đầu 1, thử 2 rồi 4 nếu đủ headroom |
| Context | Mục tiêu 100 giây/window, giữ phần tail hợp lệ | 100 giây; thử dài hơn sau baseline nếu mục tiêu cần |
| Accumulation | Bắt đầu 8, chốt theo valid audio seconds | Điều chỉnh để cùng effective batch khi so sánh |
| Precision | BF16 forward, optimizer/reductions ổn định FP32 theo recipe được kiểm thử | Như bên trái |
| Bộ nhớ | Peak toàn thiết bị <90% bộ nhớ khả dụng trong stress test | Như bên trái |

Window 10–20 giây chỉ dùng smoke/debug nhanh, **không** tự động biến thành context train production khi OOM. Thứ tự xử lý OOM: giải phóng model ngoài scope → microbatch nhỏ → activation checkpointing/FSDP wrap/prefetch → kiểm tra allocator/initialization/checkpoint peaks. Nếu vẫn không vừa context/quality đã chọn, báo không khả thi với profile đó; CPU offload chỉ thử tùy chọn sau đo, không bật mặc định vì tăng truyền RAM↔GPU.

Ước lượng chỉ để lập ngân sách: N tham số BF16 cần khoảng 2N byte weights; full Adam mixed-precision thường cỡ 16N byte gồm weights, grads, master weights và moments trước activations/buffers. Với N=7B đã khoảng 112GB toàn hệ thống, còn Moshi có thêm thành phần ngoài backbone nên phải đếm checkpoint thật. Hai A100 40GB không phù hợp full fine-tuning recipe thông thường này; hai 80GB cũng không có bảo đảm đủ activations. LoRA bỏ phần lớn optimizer/grads của frozen base, vì vậy là lựa chọn đầu. Không suy VRAM train từ `gpu_memory_utilization` của vLLM.

## 8. Môi trường và reproducibility

Tạo env training riêng, không cài vào env pipeline. Giữ Torch 2.8.0/torchaudio tương ứng làm baseline theo repo; nếu máy thực tế đang cao hơn thì không tự hạ. [Dependency upstream Moshi](https://github.com/kyutai-labs/moshi-finetune/blob/main/pyproject.toml) ghim `torch==2.6`, nên `pip install`/`uv run` nguyên README có thể hạ bản: **không chạy nguyên trạng**.

Pha compatibility phải pin commit của Moshi, trainer và model revision; tạo patch dependency có kiểm thử cho Torch hiện dùng, khóa CUDA/wheels và lưu patch hash. Không dùng `--no-deps` để che xung đột. Gate: import, codec reconstruction, forward/backward BF16, FSDP 2 GPU, optimizer step và checkpoint round-trip đều qua. Nếu không tương thích thì báo blocker, không tự nới yêu cầu Torch.

Env riêng cho backend khác chỉ khi quyết định thử backend đó. vLLM trong pipeline vẫn phục vụ khâu tạo/đánh giá dữ liệu nếu cần; không phải engine train Moshi. Không đưa NeMo, diarization, Whisper toàn bộ vào training runtime nếu reader chỉ cần token cache.

Lưu mỗi run: git commit + dirty patch, dependency lock, hardware/topology/driver, seeds, resolved config, base/codec/tokenizer revisions, manifest/split/cache hashes, trainable parameter inventory, log metrics, checkpoint lineage và eval version. Mặc định log local; không tự upload audio/transcript lên dịch vụ ngoài.

## 9. Đánh giá và định nghĩa thành công

[Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench) cung cấp protocol pause, backchannel, turn-taking, interruption; v1.5 thêm overlap distractors, v2 thêm hội thoại live nhiều lượt. Dùng bộ gốc để đối chiếu nhưng không coi điểm tiếng Anh là chất lượng tiếng Việt; tạo bộ Việt độc lập, không dịch rồi gắn nhãn là benchmark gốc.

Ba tầng đánh giá:

1. **Offline teacher forcing**: loss tách text/semantic/acoustic, speech/silence, accent, short/long turn. Hữu ích để debug, không đủ chọn model cuối.
2. **Causal replay, tự sinh target**: chỉ cung cấp user audio đến hiện tại và lịch sử được phép. Đo lời đáp, độ trễ, repetition, silence collapse; log reference target chỉ cho evaluator, không cho model.
3. **Closed-loop/live**: người dùng điều chỉnh theo câu model vừa nói; kiểm tra stop/resume, sửa ý, long context và feedback. Replay podcast không thể tạo phản ứng mới của con người theo output model.

Đề xuất bộ dev ban đầu 200 tình huống Việt cân bằng: chuyển lượt thường, pause/hesitation, user backchannel, assistant backchannel, interruption, background speech, sửa ý, nghe lâu. Bộ test riêng khóa từ đầu, chỉ dùng ở mốc chọn release; chia theo source/scenario/speaker tránh reuse cùng giọng câu mẫu cả train và test.

Metrics bắt buộc:

- CER tiếng Việt với quy tắc giữ dấu và báo chuẩn hóa; WER chỉ khi thống nhất segmentation. ASR là chỉ báo, nghe mù vẫn bắt buộc. Nếu dùng cùng họ ASR tạo label, thêm gold sửa tay và scorer khác để giảm tự xác nhận.
- Tính đúng/liên quan của câu trả lời, repetition, silence collapse, completion rate; không chỉ chấm nói trôi chảy.
- Turn takeover sai trong pause; stop latency từ onset interruption; response latency từ user offset; backchannel giành lượt sai. Báo success rate và timeout cùng p50/p95, không chỉ latency của các ca thành công.
- Streaming compute RTF, frame deadline misses, first-audio latency, input/output queue delay, transport/buffer latency. Phân biệt compute nhanh với model chọn chờ vì ngữ nghĩa.
- Blind listening về phát âm, thanh điệu, tự nhiên, speaker leakage/voice drift. Với dialogue tự do không dùng waveform-reference metric như thể chỉ có một câu trả lời đúng.

Gate vận hành đề xuất: không có leakage/schema lỗi trong tập accepted; không OOM/NaN trong stress run; resume đúng; RTF <1 cho một session trên cấu hình serving đã chọn; không silence collapse trong bài long listening. Gate chất lượng: cải thiện mục tiêu turn-taking/tiếng Việt trên dev mà không có regression đáng kể về hiểu lời/âm thanh; thống nhất biên non-inferiority và số mẫu trước chạy full, không tự đặt sau nhìn kết quả. Với model nền không nói được tiếng Việt, cần thêm mức tối thiểu do người Việt đánh giá, không chỉ “hơn baseline”.

## 10. Ngân sách và các điều còn phải xác minh

Đếm `unique_source_hours`, `accepted_conversation_hours`, `target_speech_hours`, `valid_frame_seconds`, `sampled_exposure_hours` riêng; hai track cùng dài một giờ vẫn là một giờ conversation.

Với world_size W, microbatch B/rank, accumulation A, window L giây:

`audio_seconds_per_optimizer_step = W × B × A × L × valid_fraction`.

Ví dụ W=2, B=1, A=8, L=100: tối đa 1.600 giây conversation/step. Một exposure pass trên 100 giờ hữu dụng khoảng 225 optimizer steps khi không padding/oversampling. Đây là phép tính workload, **không phải ước lượng seconds/step**; step count theo paper không thể bê nguyên khi batch đã đổi.

Đo 100–200 steps ổn định sau warmup, cả eval/checkpoint; `wall_hours ≈ steps × measured_seconds_per_step / 3600 + cache/eval/checkpoint_time`. Hai GPU cho `GPU-hours ≈ 2 × training_wall_hours`, báo riêng preprocessing. Không công bố “train x giờ là xong” khi chưa biết corpus/VRAM/thông lượng.

PCM16 stereo 24kHz khoảng 0,346GB/giờ theo hệ thập phân, chưa tính các bản mono/mixture và backups. Token cache hai track, 8 codebook, 12.5Hz, uint16 khoảng 1,44MB/giờ trước text/index và các window trùng context; đây chỉ là kích thước integer codes, không gồm features float/weights/checkpoints. Không nhân bản WAV chỉ để swap role.

Thông tin cần trước khi chạy GPU: hai A100 40/80GB, NVLink/PCIe, CPU/RAM/NVMe, đường dẫn corpus thật và số giờ accepted, quyền sử dụng nguồn/giọng, mục tiêu mô phỏng hay assistant, ngân sách GPU và chuẩn chất lượng mong muốn. Các thiếu sót này không cản viết importer/tests, nhưng chặn cam kết thời gian và run train dài.

## 11. Decision ledger

| Quyết định | Lý do | Điều kiện phải xem lại |
|---|---|---|
| Moshi là baseline đầu, chưa khóa model sản phẩm | Trainer hai track công khai + liên hệ trực tiếp với Sommelier | Gate tiếng Việt/compatibility/quality thất bại |
| Training env độc lập, không downgrade Torch | Ràng buộc người dùng + xung đột pin upstream | Chỉ đổi khi người dùng đổi yêu cầu |
| Strict clean subset trước | Tránh học mixture leakage và silence giả | QC chứng minh bản re-export đã sửa |
| Split theo nguồn trước augmentation | Ngăn leakage do clip/window/role swap | Chỉ đổi version split, không làm nhiễm test cũ |
| Cached frozen codec, FSDP 2 rank | Giảm repeated encode và VRAM train | Cache không tương đương hoặc bottleneck đo được khác |
| Assistant data là pha riêng | Podcast không đồng nghĩa instruction data | Người dùng chỉ cần mô phỏng human–human |
| Chỉ mở rộng sau pilot có eval live | Loss không chứng minh full-duplex tốt | Không bỏ gate chỉ vì train loss đẹp |
