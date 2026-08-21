đo SI-SDR / SIR / SAR,
Asteroid

các vấn đề cần xử lý:
+ đo chất lượng đoạn sau khi tách
+ tăng cường chất lượng diarization cụ thể phần phân cách giữa các speaker 
+ hậu xử lý phục hồi âm thanh 
+ thử đẩy lên mô hình 9 tỉ tham số qwen 3.5 để cải thiện asr 
+ dữ liệu đầu ra đang lưu không đúng file+ không đúng thư mục cần sửa 
# Kiến Trúc Hệ Thống Lọc & Xây Dựng Conversation Dataset (Podcast Pipeline)

Tài liệu này mô tả chi tiết kiến trúc và quy trình (pipeline) để chuyển đổi kết quả ASR thô thành một tập dữ liệu hội thoại (Conversation Dataset) đạt chuẩn công nghiệp (Production-grade), phục vụ cho việc huấn luyện các mô hình AI giọng nói (Voice AI / Emotional TTS / Voice Cloning).

---

## 1. Chuẩn hóa Transcript (Normalization)
**Đầu vào (Input):** Danh sách các `segments[]` từ kết quả ASR tổng hợp.
Mỗi segment đang chứa các trường cơ bản: `start`, `end`, `speaker`, `text`.

**Hành động:**
- **Tính toán thời lượng:** Bổ sung trường `duration = end - start`.
- **Chuẩn hóa định danh Speaker:** Ánh xạ ID của các speaker gốc (ví dụ: `speaker 1`, `speaker 3`) về định dạng chuẩn của hội thoại hai người: `A` và `B`. (Hệ thống không cần quan tâm ID ban đầu của Diarization là số mấy).

---

## 2. Phân loại & Làm sạch Segment (Segment-level Filtering)
**Nguyên tắc:** **Soft-deletion** (Không xóa trực tiếp dữ liệu gốc, chỉ gán nhãn trạng thái để dễ dàng debug và truy xuất).

**Phân loại (Classification):**
Mỗi segment sẽ được gán vào 1 trong 3 trạng thái (State) sau:

1. **`NORMAL`**: Segment hội thoại bình thường. Chứa nội dung thực sự.
2. **`BACKCHANNEL`**: Các từ đệm. Điều kiện: `duration < 1.0s` VÀ `text` thuộc danh sách Blacklist (ví dụ: "ừ", "à", "ờ", "dạ", "vâng", "đúng rồi", "thế à"...). Backchannel được giữ lại nguyên vẹn nhưng **không tính là một lượt đổi mic (turn-switch)**.
3. **`INVALID`** (Bị loại trừ), đi kèm `reject_reason`:
   - `reason: "hallucination"`: Dùng Regex quét các cụm từ ảo giác kinh điển của Whisper (VD: *"[Nhạc]", "Cảm ơn các bạn đã theo dõi", "Hãy subscribe", "Bản quyền thuộc về"*...).
   - `reason: "too_short"`: `duration < 0.3s` VÀ không nằm trong danh sách Backchannel.
   - `reason: "no_speech"`: Text rỗng hoặc chỉ toàn dấu câu.

---

## 3. Thuật toán Gom Block (Block Aggregation Algorithm)
**Mục tiêu:** Duyệt qua danh sách segments bằng kỹ thuật "Cửa sổ trượt" (Sliding Window) để nhóm chúng lại thành các đoạn hội thoại có nghĩa (Conversation Blocks).

**Điều kiện bắt buộc (Hard Constraints):**
- **Sĩ số:** `distinct_speakers == 2` (Đúng 2 người tương tác trong suốt block).

**Điều kiện ngắt Block ngay lập tức (Cut-off Breaking Conditions):**
1. **Xuất hiện Speaker lạ:** Có `speaker_id` của người thứ 3 xen vào.
2. **Khoảng lặng quá lớn (Silence Gap):** `start` của segment hiện tại trừ `end` của segment liền trước **> 3.0s**. (Đánh dấu sự kết thúc của một chủ đề/nhịp điệu).
3. **Trôi dạt thời lượng (Duration Drift):** Tổng thời lượng của block hiện tại đã vượt quá **300s (5 phút)**.
4. **Độc thoại quá dài (Max Monologue):** Lượt nói của một người kéo dài liên tục **> 45s** mà không có sự phản hồi (`NORMAL`) từ đối phương (Backchannel không phá vỡ tính độc thoại).

---

## 4. Trích xuất Chỉ số (Metrics Calculation)
Mỗi block sẽ được tính toán các chỉ số toán học để định lượng độ tự nhiên:

| Chỉ số | Định nghĩa & Ý nghĩa |
| :--- | :--- |
| `duration` | Tổng thời lượng của Block (giây). |
| `turn_count` | Tổng số lượt nói thực sự (không cộng dồn `BACKCHANNEL`). |
| `speaker_count` | Số lượng speaker tham gia (bắt buộc = 2). |
| `speaker_balance` | Mức cân bằng thời lượng: `min(dur_A, dur_B) / max(dur_A, dur_B)`. |
| `switch_count` | Số lần luân phiên mic. |
| `max_monologue` | Lượt nói dài nhất trong block (giây). |
| `silence_ratio` | Tỷ lệ khoảng lặng / Tổng thời lượng. |
| `overlap_count` | Số lần xảy ra hiện tượng tranh lời (TSE Overlap). |
| `backchannel_count`| Tổng số lượng segment gán nhãn `BACKCHANNEL`. |
| `pacing_wpm` | **(MỚI)** Tốc độ nói (Words Per Minute). Chuẩn tự nhiên: 120 - 150 WPM. |
| `turn_latency_avg`| **(MỚI)** Độ trễ phản hồi trung bình giữa 2 người (Khoảng cách từ lúc A kết thúc đến lúc B bắt đầu). Dao động chuẩn: `-0.5s` đến `+1.0s`. |

---

## 5. Chấm điểm Conversation (Scoring System)
Hệ thống chấm điểm trên thang **100 điểm**, sử dụng công thức toán học thay vì cảm tính:

- **Turn-taking (20đ):** Chấm dựa trên `turn_latency_avg`. Nằm trong khoảng `[0.1s - 1.0s]` -> 20đ. Quá cao (phản hồi chậm) hoặc quá thấp (cướp lời liên tục) -> trừ điểm dần.
- **Duration (15đ):** Đỉnh Parabol ở mức 90s. (Ví dụ: `60s - 120s` = 15đ; `30s - 60s` = 10đ; Quá ngắn hoặc quá dài -> Trừ điểm nặng).
- **Speaker balance (15đ):** Công thức `15 * speaker_balance`. (Tỷ lệ 0.5 -> 7.5đ. Tỷ lệ 1.0 -> 15đ).
- **Interaction (15đ):** Điểm tương tác: `min(15, (switch_count * 2) + backchannel_count)`.
- **Audio continuity (10đ):** Liền mạch âm thanh, chấm dựa trên `silence_ratio` (Càng ít im lặng chết càng cao điểm).
- **Speaker correctness (20đ):** Độ sạch của Diarization (Phạt nặng nếu có TSE Failures).
- **Transcript quality (5đ):** Văn bản không có lỗi hiển nhiên.

---

## 6. Phân loại Cấp độ (Tier Classification)
Dựa vào tổng điểm, dữ liệu được phân chia vào các Tier phục vụ từng mục đích huấn luyện cụ thể:

- **Tier S — Excellent (85–100):** 
  - **Mục đích:** Dùng Fine-tune các mô hình TTS Cảm xúc (Emotional TTS), Voice Cloning sinh động, hội thoại Viral.
  - **Đặc điểm:** Tương tác A ↔ B rõ ràng, balance cực tốt (A, B nói ngang nhau), turn-latency tự nhiên, nhiều backchannel.
- **Tier A & B — Good/Usable (55–84):** 
  - **Mục đích:** Dùng Pre-train các mô hình ASR (Whisper) hoặc train LLM Text-only (Học phong cách nói).
  - **Đặc điểm:** Hội thoại tốt nhưng lệch pha thời lượng (A phỏng vấn, B trả lời dài), hoặc đôi chỗ im lặng.
- **Tier C — Weak (40–54):** 
  - **Mục đích:** Dữ liệu Monologue. Dùng train TTS Đọc sách (Audiobooks) hoặc Đọc tin tức.
  - **Đặc điểm:** 1 người nói độc thoại chiếm 90% thời lượng.
- **Reject (< 40):** 
  - Vi phạm Hard Constraints hoặc dính `INVALID` segments (Quảng cáo, âm thanh hỏng). Xóa bỏ.

---

## 7. Cắt Audio (Audio Cropping & Processing)
**Quy tắc cắt (Cropping):**
- Tìm điểm cắt (boundary) tự nhiên: Ưu tiên điểm kết thúc của A → Silence → Điểm bắt đầu của B. Tuyệt đối không cắt ngang câu.

**Kỹ thuật xử lý Audio (Tránh lỗi Click/Pop):**
1. **Padding (Đệm thời gian):** Cộng thêm `+0.2s` (200ms) ở đầu và `+0.2s` ở cuối block (để giữ trọn vẹn tiếng lấy hơi trước khi nói).
2. **Fade-in / Fade-out:** Áp dụng hiệu ứng Fade-in 50ms ở đầu và Fade-out 50ms ở cuối file Audio. Kỹ thuật này triệt tiêu hoàn toàn lỗi phần cứng "Click/Pop" (âm thanh bị khựng đột ngột).
3. **Zero-Crossing:** (Khuyến nghị) Lùi/tiến vài mili-giây để tìm điểm sóng âm giao trục 0 rồi mới cắt.

**Cấu trúc thư mục Output chuẩn:**
```text
dataset/
├── audio/
│   ├── conv_000001.wav
│   └── ...
└── metadata/
    ├── conv_000001.json
    └── ...
```

**Mẫu JSON Output (Metadata):**
```json
{
  "id": "conv_000001",
  "audio": "audio/conv_000001.wav",
  "start": 0.09,
  "end": 55.43,
  "duration": 55.34,
  "speakers": ["A", "B"],
  "turn_count": 5,
  "speaker_balance": 0.85,
  "switch_count": 4,
  "overlap_count": 0,
  "backchannel_count": 3,
  "pacing_wpm": 135,
  "turn_latency_avg": 0.45,
  "score": 92,
  "tier": "S",
  "conversation": [
    {
      "speaker": "A",
      "start": 0.09,
      "end": 5.85,
      "text": "...",
      "state": "NORMAL"
    },
    {
      "speaker": "B",
      "start": 6.65,
      "end": 7.00,
      "text": "Đúng rồi",
      "state": "BACKCHANNEL"
    }
  ]
}
```
