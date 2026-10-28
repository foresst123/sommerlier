# Kế hoạch tối ưu speaker diarization (kết luận tranh luận 3 vòng của hai Sonnet, có tra cứu mạng)

Nguồn chi tiết: các file `round1..3_A/B.md` và câu trả lời tra cứu nằm trong thư mục làm việc của phiên (không nằm trong repo). Bản này là tóm tắt để thực hiện. Mọi con số về mô hình lấy từ tra cứu mạng, mức chắc chắn ghi kèm; chưa có số nào được kiểm trên dữ liệu của dự án.

## Điều đã thống nhất
1. Mục tiêu là **dữ liệu sạch**: làm bộ kiểm chứng và cổng sau khi chọn hội thoại trước, đừng đầu tư mô hình mới trước khi sửa lỗi thuật toán và cấu hình.
2. **Mắt xích yếu nhất đọc được trong code là bước nối nhãn giữa các cửa sổ** (`utils/nemotron_two_pass.py` 270-402 nối tuần tự, 409-468 reconcile, 469-553 lượt 2; `nemotron_worker.py` 400-470):
   - nối theo chuỗi tuần tự, không gom cụm toàn cục, ngưỡng cosine cố định 0.6 không có margin;
   - bộ nhớ đệm embedding tham chiếu không bao giờ làm mới;
   - lượt 2 khoá nhãn thô xuyên cửa sổ (hiếm chạy);
   - `except` trả `-1.0` âm thầm (lỗi bị nuốt);
   - `SPEAKER_UNKNOWN` được phát ra như một người nói thật, không có bước sau xử lý.
3. Bốn cặp **khác người** đo được có cosine 0.60-0.67, tức là **cao hơn** ngưỡng gộp 0.6: một ngưỡng cố định không đáng tin cho các cặp giống nhau. Hướng lỗi (gộp thừa hay tách thừa) chưa biết, phải đo.
4. Bộ ViYT-Diar chỉ là kiểm tra hồi quy (clip khoảng 60 giây, không kiểm được bước nối).
5. **Cao độ (F0) không dùng để chọn hay gán lại**; chỉ là cột phụ sau khi chọn, giữ nếu cải thiện AUC từ 0.03 trở lên. Bằng chứng (mức chắc chắn thấp-vừa, thời GMM): thêm pitch vào MFCC EER 4.67 so với 4.65, gần như không đổi.
6. **Hình ảnh:** không vào sản xuất sớm. Chỉ thử pilot 20-30 tập, tải 360p cho khoảng đã chọn (khoảng 5%), có cổng. Số liệu ASD (mức chắc chắn vừa): AVA mAP TalkNet 92.0-92.3, Light-ASD 93.9-94.1, LR-ASD 94.5; ngoài miền (WASD) 88.8. Không tìm thấy số liệu ASD cho 360p hay phim lồng tiếng.
7. **Qwen-Omni** không dùng để gán thời gian diarization (AliMeeting DER khoảng 34.6, thấp hơn pyannote+Whisper khoảng 26.1); chỉ thử câu hỏi "cùng hay khác người" trên 50 nhãn tay, bỏ nếu AUC dưới 0.8.
8. Lọc theo **tầng cách ly** (hạ hạng, không xoá) để không mất các cặp nhân vật giống nhau.

## Lộ trình có thứ tự
- **Bước 0, bộ đo (1.5-2 ngày, dưới 1 giờ GPU):** phân loại nhãn Lỗi (tách thừa / gộp / có SPEAKER_UNKNOWN / không phải hai người); ROC của cosine cùng người (lượt chẵn so với lẻ trong hội thoại Tốt) và khác người ở các độ dài 1.5, 3, 5 giây; ghi `speaker_source`, số lần trả -1.0, thời lượng UNKNOWN; độ dài khoảng hội thoại; tập tổng hợp dài 10-30 phút để kiểm rò kênh; nhãn tay 100-150 hội thoại chọn theo bất đồng; "đạt" tính trên cận dưới khoảng tin cậy.
- **Bước 1, vá bước nối nhãn (1-2 ngày CPU):** UNKNOWN không phát như người nói; làm mới cache tham chiếu; khoá lượt 2; đếm -1.0; bằng chứng 1.5 → 3 giây và loại đoạn chồng khỏi embedding; dùng margin thay ngưỡng 0.6; thử cửa sổ 25 → 45 giây. **Đạt:** nhãn dư mỗi hội thoại giảm ít nhất 30% tương đối, UNKNOWN dưới 2%, ViYT không xấu hơn.
- **Bước 2, cổng chạy lại diarization cục bộ trên đúng khoảng hội thoại đã chọn:** một lượt Nemotron, chunk lớn (tới 250 giây), so với nhãn tập dài bằng ghép Hungarian, độc lập với WeSpeaker nối. Chi phí vài phút mỗi lô. **Đạt:** bắt ít nhất 70% hội thoại Lỗi, mất không quá 15% hội thoại Tốt. Phiên bản 1 chỉ lọc, chỉ sửa khi nhãn cục bộ đúng ít nhất 80% ca bất đồng.
- **Bước 3, verifier độc lập mô hình và tầng cách ly (1-2 ngày):** cross_dist, độ thuần mỗi vai (2-means), tỉ lệ lượt gần tâm vai kia, F0 cột phụ. **Đạt:** AUC (cận dưới) từ 0.70 trở lên, độ chính xác của phần "Tốt" giữ lại từ 95%, mất không quá 15%.
- **Bước 4, có điều kiện:** gom cụm toàn cục mềm với K nhỏ, chỉ nếu sau Bước 1-2 nhãn dư vẫn quá 10% hội thoại.
- **Bước 5, tín hiệu phụ:** embedder thứ hai (giữ nếu AUC tăng từ 0.05), DiariZen (nếu lỗi ít tương quan), Qwen-Omni (bỏ nếu AUC dưới 0.8).
- **Bước 6, ưu tiên thấp:** pilot hình ảnh 20-30 tập và hồ sơ theo phim, có cổng (độ phủ từ 50%, đồng thuận từ 90%).

## Bất đồng còn lại (nhỏ), và thí nghiệm quyết
| Điểm | Quan điểm A | Quan điểm B |
|---|---|---|
| Gom cụm toàn cục mềm | làm sớm bản đơn giản | chỉ khi nhãn dư vẫn trên 10% |
| Embedder thứ hai | thử sớm | ưu tiên thấp, sau Bước 3 |
| Pilot hình | song song, ưu tiên thấp | sau, chỉ khi lọc còn mất từ 15% |
| Chạy lại cục bộ chỉ lọc hay sửa luôn | cân nhắc sửa | v1 chỉ lọc |

## Cảnh báo
- Chạy lại cục bộ chỉ bắt lỗi nối, **không bắt lỗi nhầm trong cửa sổ** vì cùng mô hình nên cùng nhầm. Cần thêm cổng độc lập (Bước 3).
- Số DIHARD của Nemotron-3 (1-4 người 9.13%, 5-9 người 27.58%) lấy từ báo chí, license mâu thuẫn (OpenMDW-1.1 hay evaluation), cần kiểm.
- yt-dlp trên IP datacenter dễ bị chặn; video 360p khoảng 375 MB mỗi giờ (từ blog).
