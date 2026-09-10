# Độ sạch của audio: quyết định, cái đã làm, cái để lại

Ngày 2026-09-06. Ghi lại để lần sau không phải tranh luận lại từ đầu.

---

## 0. Quyết định nền: không dùng enhancement

**Pipeline này không thêm denoiser, speech enhancement, hay BSS chạy toàn file.**

Lý do: mọi mô hình loại đó **biến đổi âm thanh đầu vào**. Cái nó tạo ra không
phải cái micro đã ghi được — nó là cái mô hình *đoán* rằng micro lẽ ra ghi được.
Với một corpus hội thoại full-duplex, thứ đó trở thành dữ liệu huấn luyện cho
một cuộc trò chuyện chưa từng xảy ra. Artifact của denoiser sẽ được model học
như đặc trưng của tiếng Việt nói.

Nguyên tắc: **thà ít dữ liệu mà thật, còn hơn nhiều dữ liệu mà bịa.**

Cách xử lý thay thế là **loại trừ**, không phải sửa chữa: tìm ra đoạn bẩn, đánh
dấu, rồi để nó ra ngoài corpus.

Điều kiện để xem lại quyết định này: **khi audio được ghi bằng 2 mic riêng cho
2 người nói.** Lúc đó việc tách nguồn không còn là suy đoán — mỗi kênh đã là một
quan sát thật của một người. Trước đó thì không.

### Cái vẫn được giữ, và vì sao không mâu thuẫn

- **BS-RoFormer** (tách nhạc nền): masking trên phổ, chỉ chạy trên span PANNs
  gán nhãn `music`. Nó bỏ đi cái không thuộc về giọng nói, không sinh tín hiệu mới.
- **TSE** (tách người nói khi chồng tiếng): cũng là masking, và chỉ chạy trên
  đoạn thật sự có chồng tiếng.

Ranh giới: **bỏ đi cái đã có** thì được; **thêm vào cái chưa từng có** thì không.

---

## 1. Đã làm — Truy vết (giai đoạn 1)

### Vấn đề

Pipeline cắt các đoạn hát / nhạc thuần rồi crossfade phần còn lại thành một
waveform liền. Từ diarization trở đi mọi thứ chạy trong **timeline đã cắt**.
`TimelineMap` (`utils/excise.py`) được viết để dịch ngược — nhưng
`to_original()` **chỉ được gọi trong test**. `self.timeline` được gán ở
`pipeline_service.py` rồi không ai đọc. Transcript xuất ra mang timestamp mà
không cách nào trỏ về file gốc.

### Đã thêm

`TimelineMap.spans_to_original(start, end)` — phân rã **khoảng**, không map điểm.
`TimelineMap.crosses_cut()`, `TimelineMap.cut_between()`.

`utils/provenance.py` gắn ba dấu lên mỗi `TranscriptSegment`, gọi ở export:

| Dấu | Nghĩa |
|---|---|
| `orig_spans` | Các khoảng gốc mà audio của segment thật sự gồm. **Là list** — segment vắt qua mối nối là hai mảnh dán lại. |
| `crosses_cut` | Segment đó có phải hai mảnh dán không. |
| `gap_before` | Khoảng nghỉ trước segment, hoặc `None` khi có vết cắt nằm trong đó. |

### `gap_before` là dấu quan trọng nhất

Nó là thứ dễ bỏ sót nhất và là thứ bảo vệ mục tiêu cuối. Ví dụ thật từ `hoahau`:

| | A kết thúc | B bắt đầu | khoảng nghỉ |
|---|---|---|---|
| Timeline đã cắt | 26.90 | 26.95 | **0.05s** — trông như đối đáp nhanh |
| Timeline gốc | 26.90 | 37.74 | **10.84s** — nhưng là nhạc đã xoá |

Cả hai con số đều không phải nhịp hội thoại. `None` nghĩa là **không biết được**,
không phải bằng 0. Bất cứ thứ gì học turn-taking phải bỏ qua các gap này.

Giá trị **âm** thì giữ nguyên: đó là ngắt lời, đúng thứ corpus full-duplex cần.

`metadata.timeline` và `metadata.provenance` cũng được ghi vào JSON xuất ra.

---

## 2. Đã làm — Nhìn thấy cái bẩn (giai đoạn 2)

### Vấn đề

`models/panns.py` đọc **3 nhóm nhãn trên 527**: `Speech`, `Singing`, `Music`.
Cnn14 tính đủ 527 nhãn mỗi forward pass — 524 nhãn còn lại bị tính rồi vứt đi.
Một segment ghi cạnh xe máy và một segment ghi trong phòng tiêu âm là như nhau
với pipeline.

### Đã thêm

Ba nhóm nhãn nhiễu trong `models/panns.py`, **mọi tên đã đối chiếu với
`class_labels_indices.csv` thật của AudioSet** (test `test_every_noise_label_exists_in_audioset` giữ điều đó):

- `NOISE_SPEECH_LABELS` — Chatter, Crowd, Hubbub, Television, Radio…
  Nguy hiểm nhất: phá diarization và đưa vào transcript chữ không ai trong cuộc
  trò chuyện nói.
- `NOISE_ENV_LABELS` — Motorcycle, Traffic noise, Vehicle horn, Wind, Rain…
- `NOISE_ROOM_LABELS` — Typing, Air conditioning, Mechanical fan, Clatter, Hum…

**Cố ý KHÔNG tính là nhiễu**: `Breathing`, `Cough`, `Throat clearing`, `Sneeze`,
`Laughter`, `Sigh`. Chúng phát ra từ chính người nói — đó là hiện tượng corpus
cần thu, không phải thứ ô nhiễm cần lọc.

`utils/noise_map.py` — `NoiseTrack`, giữ đường cong framewise trong **timeline gốc**.

Ba lựa chọn thiết kế, mỗi cái có test giữ:

- **Không tốn thêm một lần chạy model.** `build(scores, fps)` nhận kết quả của
  lần sweep mà music map đã trả tiền. `music_map.build_maps()` trả về cả hai.
- **Điểm là percentile 90, không phải mean hay max.** Mean cho phép một turn dài
  che một giây tiếng xe; max để một frame tiếng đóng cửa kết án cả turn.
- **Giữ ở dạng đường cong, không phải span.** Nhạc là quyết định định tuyến nên
  phải thành khoảng. Nhiễu là *ngưỡng mỗi consumer tự chọn* — thu thành span
  bây giờ là đóng đinh một ngưỡng vào mọi thứ phía sau trước khi ai kịp nhìn
  phân bố trên audio thật.

`score_spans()` nhận **orig_spans** từ giai đoạn 1 — segment dán từ hai mảnh
được chấm trên đúng hai mảnh đó, không phải trên khoảng giữa chúng.

`None` nghĩa là chưa đo, không phải sạch.

---

## 3. Chưa làm — Dùng dấu để dựng hội thoại (giai đoạn 3)

Đây là phần để lại. Nền móng đã xong; đây là cách tiêu thụ nó.

### 3.1 Luật dựng hội thoại

1. **Không nối hai lượt qua vết cắt.** `gap_before is None` → hai lượt này không
   liền nhau trong thực tế. Kết thúc đoạn hội thoại tại đó và bắt đầu đoạn mới.
2. **Loại segment `crosses_cut=True` khỏi dữ liệu học timing.** Audio của nó là
   hai mảnh dán; nội dung vẫn dùng được, nhịp thì không.
3. **Lọc theo `noise_score`.** Ngưỡng **chưa chọn** — phải nhìn phân bố thật trước.
   `noise_speech` nên có ngưỡng chặt hơn hai nhóm kia vì nó phá diarization.

### 3.2 Việc phải làm trước khi chọn ngưỡng

Chạy giai đoạn 1+2 trên `data/clip_selection.csv` — bộ này **đã phân tầng sẵn**
đúng những gì cần: `Clean 1-on-1 interview, no music` ×3 (tầng đối chứng),
`Outdoor / noisy environment` ×3 (tầng bẩn), `Có nhạc nền` ×3. Cột
`observed_failure` còn trống.

Nhìn phân bố `noise_score` trên ba tầng đó rồi mới đặt ngưỡng. Đặt ngưỡng trước
khi có phân bố là đoán.

### 3.3 Số liệu nền đã có (đo 2026-09-06)

```
file                              dur_s   music    song    cut   %cut  mảnh  seams
hoahau                             2978     3.1    10.8   10.8    0.4     2      1
thu_that_thach_10m                  600     0.9    10.5   10.5    1.8     2      1
lm8-vongtaynang-reaction-131131    1324    26.6   209.1  201.2   15.2    11     10
vimeanhphanchiatay-145413          1690    83.0    30.7   27.8    1.6    17     16
```

**Giả định cũ ghi trong code — "30s of music in 50 minutes" — đã sai.** `lm8`
bị cắt 15% recording. `vimeanh` bị băm thành 17 mảnh, mảnh ngắn nhất **0.4s**.

Với những mảnh vụn đó thì **cả nối lẫn không nối đều hỏng**: nối thì tạo turn
change giả, không nối thì mảnh quá ngắn để clustering làm gì. Đã chọn **nối +
đánh dấu**; `gap_before` là thứ ngăn hậu quả lan xuống dataset.

---

## 4. Đã làm thêm sau lần viết đầu (2026-09-07)

Chạy thật trên Kaggle sinh ra một loạt thay đổi, ghi lại ở đây vì phần 1–3 ở trên
không còn mô tả đúng hệ thống.

**Sidon bị xoá, chỉ còn USEF.** 13 file, ~280 dòng trong `separation_service`.
Cửa sổ mixture cũ (5s solo A + 5s solo B + overlap, 6–12s) tồn tại vì Sidon tách
mù. USEF được điều kiện hoá bằng enrollment 8s nên mixture chỉ cần **chính đoạn
overlap, nới tới 2s** — đúng cửa sổ graph ONNX nhận.

Mất theo: bộ chặn người-thứ-ba (lý do của nó là Sidon chỉ ra đúng 2 nguồn) và bộ
tool so sánh hai backend.

**Mối nối được xử lý ở cả ba tầng**, thay vì chỉ đánh dấu sau khi đã xảy ra:

```
excise         bỏ đảo <1s      →  hai vết cắt gộp làm một, bớt mối nối
diarization    cắt tại mối nối TRƯỚC VAD  →  mảnh vào chuỗi đã liền mạch
merge          không bắc qua mối nối      →  không dán lại được
```

`crosses_cut` vì thế chuyển từ cảnh báo thành bất biến được kiểm chứng.

**`SINGING_THRESHOLD` 0.35 → 0.12** và `MIN_SPAN` tách đôi: `0.32` khi tách nhạc
(hoàn tác được), `0.96` khi cắt bỏ (không hoàn tác được). Lần chạy thật đầu tiên
cho `8.4s singing` trên `lm8` — trước đó là 0.0 trên mọi file.

**Không còn định danh nào tên "enhance".** `EnhancedSegment` → `SpeechSegment`,
`enhanced_audio` → `audio`. Cái tên mô tả một bước không tồn tại và đã thật sự
làm người đọc hiểu nhầm là có. `models/separate_fast.py` xoá luôn — 289 dòng
không ai import, và là nơi cuối cùng còn nhánh `denoise`.

## 5. Nợ kỹ thuật còn lại

**Span chồng lên nhau — CHƯA SỬA.** `PAD_SECONDS=0.30` vẫn cộng vào từng loại
span độc lập, nên hai span kề nhau đè lên nhau đúng 0.60s. Đo trên dữ liệu thật:
69–100% span `music` chồng lên vùng sắp bị cắt. Hệ quả là BS-RoFormer tách nhạc
trên khúc rồi bị vứt. Cách sửa đúng là hoà giải nhãn ở mức frame trước khi tạo
span, thay vì pad ba lần rồi dọn.

**`MUSIC_THRESHOLD` vẫn phục vụ hai mục đích ngược chiều.** Ngưỡng lỏng 0.10 hợp
lý cho việc tránh lấy enrollment trên nền nhạc (sai thì mất một ứng viên trong
hàng trăm), nhưng cùng con số đó quyết định ghi đè waveform bằng separator (sai
thì hỏng audio). Nên tách làm hai hằng số.

**`SINGING_MARGIN = 0.15` giờ là ràng buộc chặn thật sự** và chưa từng được hiệu
chỉnh. Sau khi hạ ngưỡng xuống 0.12, chính margin mới là thứ giới hạn: trên `lm8`
ngưỡng cho 23.0s còn margin cắt xuống 10.2s.

**`qc_sim_threshold = 0.2`** — comment trong code tự ghi "NOT calibrated".

**Điểm nhiễu chưa gặp bản ghi ngoài trời.** Ngưỡng 0.10 đặt từ ba file trong nhà,
trần đo được là 0.28. Lần chạy thật cho `max=0.323` và `max=0.378` — **đã vượt
trần hiệu chỉnh**, nên có dữ liệu để đặt lại.

**14 test fail có sẵn** trong `test_steps` / `test_batch` / `test_config_driven` /
`test_prefix_cache`, từ commit `c33ce45` tắt các stage mà test vẫn kỳ vọng bật.
Đối chiếu với baseline `41d8c73`: đúng 14 cái đó, không có cái nào mới.

**`export_sdlm_dual_channel` chưa được gọi.** Hai track full-duplex — sản phẩm
chính — hiện không được ghi ra; đầu ra là mono. Hàm có sẵn, có test, chưa nối vào
pipeline. Và khi nối thì phải cắt track tại `seams()`, nếu không sẽ có những chỗ
nhịp hội thoại bịa ra.
