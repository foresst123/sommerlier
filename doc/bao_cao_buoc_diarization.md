# Báo cáo nhanh: bước diarization và các thành phần đang dùng

Ngày 2026-10-28. Nguồn: `podcast-pipeline/config.json` (profile `a100`), code ở `services/diarization_service.py`, `nemotron_worker.py`, `utils/nemotron_two_pass.py`, số liệu trong `doc/nemotron_toi_uu_viyt_diar.md`, kết luận trong `doc/ke_hoach_toi_uu_diarization.md` và cuộc tranh luận hai agent về các hướng cải thiện. Chỗ nào chưa đo hoặc chưa kiểm thì ghi rõ.

## 1. Vị trí trong kiến trúc tổng thể

Khử nhiễu `rf_crowd>blend50` → Silero VAD → **Nemotron-3-Diarization** → DialogueSidon (tách giọng chồng) → ASR (Qwen3-ASR-1.7B, Whisper-large-v3, PhoWhisper-large-ct2) → refinement (Qwen3.5-9B) → relabel → chọn hội thoại (Qwen3.5-9B).

Mục tiêu cuối là dữ liệu hội thoại hai người sạch. Vì vậy chất lượng diarization được đo bằng độ sạch của nhãn người nói, không chỉ bằng DER.

## 2. Luồng bên trong bước diarization

1. **Cắt cửa sổ bằng Silero:** cửa sổ khoảng 25 s (20 đến 30 s), cắt tại khoảng lặng, ngữ cảnh 2 s mỗi bên.
2. **Nemotron** chạy từng cửa sổ, hai lượt (`two_pass`). Lượt 2 dùng kết quả lượt 1 làm gợi ý.
3. **Nối nhãn giữa các cửa sổ:** dựa trên phần chồng thời gian, nếu không đủ thì dùng cosine WeSpeaker (ngưỡng 0.6, cần tối thiểu 1.5 s tiếng sạch).
4. **Hậu xử lý ngưỡng** (a100): onset 0.53, offset 0.49, pad_onset 0.23, pad_offset 0.01, min_duration_on 0.42, min_duration_off 0.34.
5. **Hậu xử lý đoạn** (`diarize_postprocess`): bỏ đoạn dưới 200 ms, cắt tại mối nối nhạc, tách đoạn dài bằng VAD, gộp cùng người (`merge_gap` 0.5), nối đoạn bị người kia ngắt (`bridge_gap` 3.0 s), cắt đoạn dài hơn 20 s. Việc gộp "ghost speaker" đang **tắt** (`ENABLE_GHOST_MERGE = False`).

## 3. Đánh giá từng thành phần

| Thành phần | Vai trò | Bằng chứng đã có | Đánh giá |
|---|---|---|---|
| **Khử nhiễu `rf_crowd>blend50`** | Giảm tiếng đám đông trước diarization | **Chưa đo ảnh hưởng lên diarization.** `rf_crowd` được thiết kế để loại tiếng khán giả, giọng nằm ở luồng "non-crowd". Văn liệu cho thấy khử nhiễu có thể làm DER tệ hơn nếu không chọn file theo SNR (agent B, mức chắc vừa) | **Cần A/B** (âm gốc so với đã khử nhiễu) trước khi coi là có lợi |
| **Silero VAD** | Chọn điểm cắt cửa sổ; tách đoạn dài sau diarization | Không có thử nghiệm riêng. Ngưỡng hiện tại: 0.35, nới biên 160 ms | Hợp lý, rủi ro thấp. Cắt sai chỗ chỉ tạo thêm chỗ nối nhãn |
| **Nemotron-3-Diarization** (Sortformer, tinh chỉnh tiếng Việt, bộ nhớ người nói AOSC) | Phân đoạn và gán nhãn trong cửa sổ | ViYT-Diar, 20 clip khoảng 60 s: baseline DER@0ms 18,82%, bản tinh chỉnh tốt nhất 12,55 đến 13,24%; DER@0,25 s 6,01 đến 7,62%; đúng số người 85% (17/20). Ba file sai số người (4, 1, 1 người thay vì 2) kéo DER@0ms từ 9,16% lên 14,39% | **Mô hình trong cửa sổ khá tốt** (lỗi nhiều ở biên chứ ít nhầm người). Nhưng mọi số đo dùng bộ hậu xử lý "crisp" (pad 0.0); cấu hình production ở trên **chưa được đo** |
| **Refiner 10 ms (thử nghiệm 5)** | Tinh chỉnh biên | Không đạt mục tiêu nào, Conf tăng, đúng-số-người giảm | **Bỏ**, không dùng |
| **WeSpeaker ResNet-293 (VoxCeleb)** | Nối nhãn giữa cửa sổ | Đo được bốn cặp **khác người** có cosine 0,60 đến 0,67, tức cao hơn ngưỡng gộp 0,6 | **Điểm yếu nhất đã xác định.** Huấn luyện ngoài miền tiếng Việt; embedding hiện ghép cả phần chồng giọng |
| **Thuật toán nối nhãn** (`utils/nemotron_two_pass.py`) | Quyết định ai là ai xuyên cửa sổ | Đọc code: nối tham lam theo thứ tự, ngưỡng cố định không có margin, bộ nhớ đệm tham chiếu không làm mới, lỗi embedding bị nuốt và trả `-1.0`, `SPEAKER_UNKNOWN` được phát như một người nói thật | **Cần sửa trước tiên** (rẻ, chỉ cần CPU). Chưa có số đo lỗi gộp thừa hay tách thừa |
| **Chuỗi hậu xử lý đoạn** | Làm sạch đầu ra cho tách giọng | `bridge_gap` 3.0 s và `merge_gap` 0.5 đã tinh chỉnh cho tách giọng chồng. Nghi ngờ chưa kiểm: `min_duration_off 0.34` có thể kéo tiếng đệm của người kia vào; `min_duration_on 0.42` có thể mất "dạ/ừ" ngắn | **Cần quét tham số** theo rò kênh |
| **DiariZen** (WavLM-Large, gom cụm VBx) | Backend thay thế, mặc định ở profile kaggle | Số công bố (nguồn web, mức chắc vừa, chưa rõ collar): AMI 14,0 so với 22,4 của pyannote 3.1; DIHARD3 14,5 so với 21,7. Một bài báo tự báo 2,38% DER sau tinh chỉnh trên ViYT-Diar (chưa kiểm). Trọng số CC BY-NC | **Cần A/B với Nemotron** và làm rõ giấy phép trước khi dùng sản xuất |
| **Relabel: DiarizationLM-Gemma-4-E4B-v1** | Sửa nhãn người nói theo văn bản | **Trong code vẫn là relabel cũ** (`SpeakerRelabelService`, đổi theo từng đoạn, chạy bằng Qwen3.5-9B); chưa đổi. DiarizationLM huấn luyện trên Fisher, Callhome, ICSI, AMI; **chưa thấy bằng chứng hỗ trợ tiếng Việt** | **Phải đo trước khi thay** trên tập review Tốt/Xấu. Làm theo từng chữ thì cần đưa forced alignment lên trước relabel |

## 4. Kết luận từ các đánh giá

1. **Nút thắt nằm ở bước nối nhãn giữa các cửa sổ**, không phải ở mô hình trong cửa sổ. ViYT-Diar chỉ có clip 60 s nên **không đo được** chính chỗ này.
2. **Chưa có số đo nào trên dữ liệu thật của dự án** (hội thoại dài, cấu hình production). Việc đầu tiên là dựng bộ đo: tập ghép dài 10 đến 30 phút có nhãn thật, và 100 đến 150 hội thoại gán nhãn tay theo loại lỗi (tách thừa, gộp nhầm, có UNKNOWN, không phải hai người).
3. Hai profile (kaggle và a100) khác nhau cùng lúc ở backend, khử nhiễu và hậu xử lý, nên so sánh giữa chúng đang bị trộn biến.
4. Có một lỗi code thật ở `nemotron_worker.py:264` (bộ ngưỡng mặc định 0.62 là code chết, thực tế dùng 0.52), nhưng chỉ ảnh hưởng đường Kaggle khi chọn Nemotron. Profile a100 khai báo ngưỡng riêng nên **không bị ảnh hưởng**.

## 5. Hướng cải thiện, theo thứ tự ưu tiên

1. Dựng bộ đo dài và nhãn tay theo loại lỗi.
2. Đo cấu hình a100 thật so với bộ "crisp", rồi quét `min_duration_on/off`, pad.
3. Vá nối nhãn: embedding sạch (bỏ vùng chồng giọng), margin thay ngưỡng 0.6, đếm số lần trả `-1.0` và lượng UNKNOWN.
4. Nối nhãn bằng Hungarian cộng margin, sau đó gom cụm toàn cục có ràng buộc cannot-link nếu cần.
5. Cổng lọc sau diarization (chỉ lọc, không sửa), verifier dùng embedder khác họ WeSpeaker.
6. A/B Nemotron với DiariZen, và A/B khử nhiễu.
7. Cửa sổ dài hơn (45 đến 120 s), chỉ nhận khi đã đo trên tập dài.

**Đã làm:** arm F trong `tools/diar_arms_experiment.py` (commit `a069764`): Nemotron cửa sổ 30 s trượt bước 10 s, gom cụm có cannot-link, ghép xác suất kiểu DiariZen. Thuật toán đã có test với dữ liệu giả (46 test đạt; 3 test đỏ sẵn do thiếu `torch`). **Chưa chạy trên GPU thật**; chưa biết nó có tốt hơn arm A hay không.

## 6. Rủi ro và điều chưa kiểm

- Cách lấy xác suất theo frame của Nemotron (`include_tensor_outputs`) và độ dài frame (80 ms hay 10 ms) chưa kiểm trên NeMo thật.
- Số DIHARD của Nemotron-3 lấy từ báo chí, giấy phép mâu thuẫn giữa các nguồn (OpenMDW-1.1 hay evaluation), cần kiểm.
- Giấy phép DiariZen (CC BY-NC) và DiarizationLM chưa được xác minh cho dùng sản xuất.
- Chỉ có 20 clip test, nên chênh dưới khoảng 1 điểm DER giữa hai cấu hình là nhiễu.
