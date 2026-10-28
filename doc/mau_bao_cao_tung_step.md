# Mẫu báo cáo cho từng step của pipeline

Cách dùng: sao chép phần "Mẫu" bên dưới cho mỗi step, điền theo thứ tự. Bản đã điền cho step diarization là `doc/bao_cao_buoc_diarization.md`, dùng làm ví dụ.

Quy ước để báo cáo đọc được và kiểm được:
- Mỗi con số ghi nguồn (file doc, log chạy, bài báo) và ghi rõ **đã đo trên dữ liệu của dự án** hay **số công bố bên ngoài**.
- Điều chưa đo hoặc chưa kiểm ghi thẳng là "chưa đo", không suy ra từ cấu hình.
- Nhận định chỉ có ba mức: **Giữ**, **Cần đo**, **Cần sửa/thay**. Mỗi mức kèm một câu lý do.

---

## Mẫu

### Step N: <tên step>

**1. Mục đích và vị trí**
- Làm gì, vì sao cần cho mục tiêu dữ liệu hội thoại hai người sạch.
- Đầu vào: <file/dữ liệu lấy từ step nào>. Đầu ra: <file/dữ liệu đưa cho step nào>.

**2. Thành phần đang dùng**

| Thành phần | Model / thuật toán | Kiến trúc | Cấu hình chính (profile a100) |
|---|---|---|---|
| | | | |

**3. Luồng xử lý** (3 đến 6 gạch đầu dòng theo đúng thứ tự code chạy; ghi file:dòng nếu có)

**4. Đánh giá từng thành phần**

| Thành phần | Bằng chứng | Mức | Nhận định |
|---|---|---|---|
| | số đo, nguồn, hoặc "chưa đo" | Giữ / Cần đo / Cần sửa | một câu |

**5. Kết luận** (2 đến 4 ý: điểm yếu chính, điều chắc chắn, điều chưa chắc)

**6. Hướng cải thiện** (theo thứ tự lợi ích trên chi phí; mỗi hướng một dòng: làm gì, tốn gì, đo bằng gì)

**7. Rủi ro và việc chưa kiểm** (giấy phép, phụ thuộc môi trường, giả định chưa xác minh)

---

## Danh sách step và những gì đã có sẵn để điền

Cột "Đã có đánh giá" chỉ ghi những gì hai bên đã làm trong phiên này. Chỗ trống nghĩa là chưa có, phải đọc code hoặc đo thêm.

| # | Step | Thành phần (profile a100) | Đã có đánh giá |
|---|---|---|---|
| 1 | Khử nhiễu đám đông nhẹ | `rf_crowd` (Mel-Band RoFormer của aufr33/viperx) rồi `blend50`; cấu hình `denoise: rf_crowd>blend50` | Chưa đo ảnh hưởng lên diarization và ASR. Văn liệu cảnh báo khử nhiễu có thể làm DER tệ hơn nếu không chọn file theo SNR |
| 2 | VAD | Silero VAD, ngưỡng 0.35, nới biên 160 ms | Dùng để chọn điểm cắt cửa sổ diarization; không có thử nghiệm riêng |
| 3 | Diarization | `nvidia/Nemotron-3-Diarization` (tinh chỉnh tiếng Việt, AOSC) cộng WeSpeaker ResNet-293 để nối nhãn | Đầy đủ, xem `doc/bao_cao_buoc_diarization.md` |
| 4 | Separation (tách giọng chồng) | `sarulab-speech/DialogueSidon` | Chưa đánh giá trong phiên này; tài liệu liên quan: `doc/huong-cai-thien-tach-giong.md`, `doc/ke_hoach_thu_nghiem_tach_giong_theo_vong.md` |
| 5 | ASR | `Qwen/Qwen3-ASR-1.7B` (vLLM), `openai/whisper-large-v3` (vLLM), `kiendt/PhoWhisper-large-ct2`, bỏ phiếu giữa ba lane | Chưa đánh giá chất lượng; tài liệu liên quan: `doc/asr-lanes-va-tien-do.md`, `podcast-pipeline/tools/tinh_wer.py` |
| 6 | Refinement | `Qwen/Qwen3.5-9B` (vLLM, hai bản sao, một request mỗi đoạn) | Chưa đánh giá; `doc/thinking_budget.md` |
| 7 | Relabel | Hiện: `SpeakerRelabelService` chạy bằng Qwen3.5-9B, đổi nhãn theo từng đoạn. Dự định: `google/DiarizationLM-Gemma-4-E4B-v1` | Chưa đổi trong code. Chưa có bằng chứng DiarizationLM hỗ trợ tiếng Việt; làm theo từng chữ cần đưa forced alignment lên trước relabel |
| 8 | Chọn conversation | `Qwen/Qwen3.5-9B` | Chưa đánh giá; `doc/selection_replay.md` |

Các step còn lại trong profile a100 mà báo cáo cuối có thể cần nhắc: căn từng chữ (`nguyenvulebinh/wav2vec2-base-vietnamese-250h`), xuất hội thoại (`conversation_exports`, `export`). `transcript_review` và `persona_profile` đang tắt ở a100.
