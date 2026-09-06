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

## 4. Nợ kỹ thuật phát hiện dọc đường

Chưa sửa, không nằm trong phạm vi đã làm.

- **Span chồng lên nhau.** `hoahau`: `music 26.58–27.5` nằm gọn trong
  `song 26.90–37.74`. `lm8`: `music 7.38–10.22` chồng `song 9.62–13.10`.
  Nguyên nhân: `PAD_SECONDS=0.30` cộng vào từng span độc lập. Hệ quả:
  BS-RoFormer chạy trên đoạn sắp bị xoá hoàn toàn — công vô ích. Và `remap()`
  phải xử lý span chồng nhau, chỗ chưa được kiểm chứng.

- **`singing_seconds = 0` trên cả 4 file / 79 span.** Nhánh `SINGING` có thể
  chưa từng kích hoạt trong thực tế; mọi thứ có nhạc đều rơi vào `SONG` hoặc
  `MUSIC`. Với một reaction video có 209s nhạc mà không giây hát nào thì
  `SINGING_MARGIN=0.15` đáng nghi.

- **`MUSIC_THRESHOLD` phục vụ hai mục đích ngược chiều nhau.** Comment biện minh
  ngưỡng thấp (0.10) bằng lý do *enrolment*: gán nhầm chỉ mất một ứng viên trong
  hàng trăm. Nhưng cùng map đó giờ điều khiển việc **ghi đè waveform** bằng
  BS-RoFormer. Hai consumer, hai bất đối xứng ngược nhau, chung một ngưỡng.
  Nên tách làm hai hằng số.

- **14 test fail có sẵn** trong `test_steps` / `test_batch` / `test_config_driven`
  / `test_prefix_cache` — hệ quả commit `c33ce45` tắt các stage. Chúng có từ
  trước và đang che tín hiệu của test thật.

- **Comment lệch** ở `services/model_loader.py`: nói BS-RoFormer chạy GPU 2,
  code truyền `device_1`.
