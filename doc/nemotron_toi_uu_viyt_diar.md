# Nemotron-3-Diarization: tối ưu trên ViYT-Diar (thử nghiệm 4 và 5)

Số liệu thô của hai lần chạy, để phân tích tiếp. Nối tiếp [nemotron_diarization_experiment.md](nemotron_diarization_experiment.md) (cách cài và chạy backend). Mọi con số dưới đây chép từ log chạy trên DGX ngày 2026-10-27; phần "Số liệu tổng hợp" là mình tính lại từ bảng theo từng file, phần "Quan sát" là suy ra từ số liệu và ghi rõ chỗ nào chưa chắc.

## 1. Thiết lập

- Dữ liệu: ViYT-Diar. Huấn luyện trên **Train N=80**, đánh giá trên **Test N=20** (hai tập tách rời). Mỗi file test dài khoảng 53–73 s, tổng 1235,9 s (20,6 phút), 2 người nói thật mỗi file (cột `Ref`).
- Mô hình: `nvidia/Nemotron-3-Diarization` (`SortformerEncLabelModel`), chạy streaming. Tham số streaming `spkcache_len 384`, `fifo_len 60`, `chunk_len 340`, `chunk_right_context 40`, `spkcache_update_period 200` đọc được từ script của lần chạy thứ ba (mục 4); log thử nghiệm 4 và 5 không in các giá trị này, nhưng log thử nghiệm 4 có cảnh báo `spkcache_update_period (200) < chunk_len (340)` nên chu kỳ cập nhật thực tế là 340.
- "AOSC" trong tên cấu hình: mình đoán là `install_deep_aosc_hooks` (`tools/bench_nemotron_viyt.py`), vì đó là hàm các script gọi; chưa đọc mã của nó.
- Môi trường: `/home/lamkd2/nemotron_env/bin/python`, `CUDA_VISIBLE_DEVICES=0`, `PYTHONPATH=.`.
- Chỉ số: `DER@0ms` / `DER@.25s` là DER với collar 0 và 0,25 s. `Miss%`, `FA%`, `Conf%` là thành phần của DER@0ms (cộng lại bằng DER@0ms, riêng cấu hình 4 của thử nghiệm 4 lệch 0,01 do làm tròn). `SpkExact` là tỉ lệ file có đúng số người nói (N=20 nên mỗi file là 5 điểm phần trăm). `SpkMAE` là sai số tuyệt đối trung bình của số người nói. `<0.5s` là số đoạn ngắn hơn 0,5 s trên tổng số đoạn dự đoán.

## 2. Thử nghiệm 4: Hungarian + L2-SP + differential LR + warmup

Script: thử nghiệm 4 (trọng số lưu ở `models/nemotron_vi_upgraded.pt`). Huấn luyện: căn chỉnh Hungarian trên 240 cửa sổ, batch 16, L2-SP 0,05, 12 epoch.

| epoch | Aligned BCE | LR scale |
|---|---|---|
| 1 | 0,0591 | 0,56 |
| 4 | 0,0483 | 0,90 |
| 8 | 0,0462 | 0,40 |
| 12 | 0,0457 | 0,10 |

Kết quả trên Test N=20:

| Cấu hình | DER@0ms | DER@.25s | Miss% | FA% | Conf% | SpkExact | SpkMAE | <0.5s | Time |
|---|---|---|---|---|---|---|---|---|---|
| 1_Baseline_Repo | 18,82% | 6,48% | 2,01% | 15,71% | 1,10% | 75,0% | 0,30 | 2/165 | 5,3 s |
| 2_OptLearn_Balanced+AOSC | 13,24% | 6,01% | 3,29% | 8,92% | 1,03% | 85,0% | 0,20 | 21/271 | 4,5 s |
| 3_OptLearn_Crisp+AOSC | 12,86% | 6,25% | 3,94% | 7,88% | 1,04% | 85,0% | 0,20 | 23/297 | 3,8 s |
| 4_OptLearn_Tight+AOSC | 12,55% | 7,62% | 5,52% | 5,98% | 1,04% | 85,0% | 0,20 | 35/435 | 4,0 s |

## 3. Thử nghiệm 5: 10 ms Acoustic-Skip Neural Refiner + Quorum pruning

Script: `tools/exp5_breakthrough_nemotron.py` (trọng số lưu ở `models/nemotron_vi_acoustic_refiner.pt`). Mục tiêu đặt ra trong script, trên Test N=20:

| Chỉ số | Mục tiêu |
|---|---|
| DER@0ms | < 6,0% |
| Miss% | < 1,0% |
| FA% | < 2,0% |
| SpkExact | ≥ 95,0% |

Trích đặc trưng 80 file Train và 20 file Test mất 24,2 s. Huấn luyện bộ tinh chỉnh 25 epoch:

| epoch | 1 | 5 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|---|
| Refiner loss | 0,4215 | 0,2548 | 0,2570 | 0,2358 | 0,2299 | 0,2285 |

Kết quả trên Test N=20:

| Cấu hình | DER@0ms | DER@.25s | Miss% | FA% | Conf% | SpkExact | SpkMAE | <0.5s |
|---|---|---|---|---|---|---|---|---|
| 1_Nemotron_Exp2+Quorum | 14,13% | 9,15% | 5,88% | 6,31% | 1,94% | 80,0% | 0,25 | 34/416 |
| 2_AcousticSkip_Balanced | 13,95% | 8,23% | 3,90% | 7,55% | 2,50% | 70,0% | 0,35 | 31/379 |
| 3_AcousticSkip_Crisp | 13,68% | 8,28% | 4,42% | 6,92% | 2,35% | 70,0% | 0,35 | 30/386 |
| 4_AcousticSkip_Ultra | 14,27% | 8,67% | 4,16% | 7,26% | 2,85% | 60,0% | 0,45 | 33/387 |
| 5_AcousticSkip_LowMiss | 14,10% | 8,12% | 3,27% | 8,14% | 2,69% | 70,0% | 0,35 | 36/375 |

Không cấu hình nào đạt mục tiêu nào trong bốn chỉ số.

## 4. Bảng theo từng file (lần chạy thứ ba)

Cấu hình đầu ra này không được ghi trong log: đoạn script chạy bằng `python - <<'PY'` bị dán lỗi ở giữa (phần ghi kết quả bị chen chữ), nên chỉ biết nó dùng `models/nemotron_vi_upgraded.pt`, Deep AOSC, hậu xử lý "crisp" (`onset 0.62, offset 0.52, pad_onset 0.0, pad_offset 0.02, min_duration_on 0.18, min_duration_off 0.12`) và chạy hai lượt (lượt dò rồi lượt "pre-primed", mô tả trong script là "exact 4_Exp2_OmiTight+AOSC setup"). Trung bình của bảng (DER@0ms 14,39%) gần nhưng không bằng dòng `1_Nemotron_Exp2+Quorum` ở thử nghiệm 5 (14,13%), nên chưa nên coi hai cái là một.

| ID | Dài (s) | Ref | Hyp | DER@0 | DER@.25 | Miss0 | FA0 | Miss.25 | FA.25 | Conf0 | Thời lượng từng người nói (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 000 | 61,5 | 2 | 2 | 2,40% | 2,17% | 2,30% | 0,09% | 2,17% | 0,00% | 0,00% | S0 4,99 · S1 46,85 |
| 001 | 61,8 | 2 | 2 | 8,12% | 0,72% | 0,40% | 7,72% | 0,00% | 0,72% | 0,00% | S0 45,4 · S1 10,89 |
| 002 | 73,3 | 2 | 2 | 4,29% | 3,98% | 4,23% | 0,06% | 3,98% | 0,00% | 0,00% | S0 13,15 · S1 55,66 |
| 003 | 60,3 | 2 | 2 | 0,92% | 0,54% | 0,82% | 0,10% | 0,54% | 0,00% | 0,00% | S0 55,97 · S1 2,09 |
| 004 | 61,9 | 2 | 2 | 17,20% | 0,35% | 0,71% | 16,50% | 0,04% | 0,32% | 0,00% | S0 3,57 · S1 50,56 |
| 005 | 58,1 | 2 | **4** | 71,63% | 68,92% | 3,76% | 65,00% | 0,00% | 68,92% | 2,87% | S0 13,95 · S1 8,43 · S2 17,9 · S3 13,76 |
| 006 | 60,5 | 2 | 2 | 10,95% | 2,01% | 0,02% | 10,93% | 0,00% | 2,01% | 0,00% | S0 19,26 · S1 37,87 |
| 007 | 66,0 | 2 | 2 | 8,81% | 6,75% | 8,46% | 0,35% | 6,38% | 0,37% | 0,00% | S0 53,9 · S1 1,58 |
| 008 | 59,3 | 2 | 2 | 3,56% | 2,01% | 3,22% | 0,34% | 2,01% | 0,00% | 0,00% | S0 7,52 · S1 47,37 |
| 009 | 62,1 | 2 | 2 | 9,82% | 0,00% | 0,43% | 9,39% | 0,00% | 0,00% | 0,00% | S0 27,77 · S1 27,48 |
| 010 | 59,0 | 2 | 2 | 7,90% | 0,00% | 0,17% | 7,74% | 0,00% | 0,00% | 0,00% | S0 2,04 · S1 55,95 |
| 011 | 60,0 | 2 | 2 | 6,28% | 0,00% | 0,51% | 5,77% | 0,00% | 0,00% | 0,00% | S0 9,32 · S1 45,98 |
| 012 | 60,0 | 2 | 2 | 8,24% | 0,00% | 0,56% | 7,68% | 0,00% | 0,00% | 0,00% | S0 9,63 · S1 41,83 |
| 013 | 60,0 | 2 | 2 | 11,35% | 0,00% | 1,29% | 10,06% | 0,00% | 0,00% | 0,00% | S0 22,69 · S1 26,93 |
| 014 | 61,3 | 2 | **1** | 40,01% | 24,82% | 1,30% | 10,20% | 0,14% | 0,24% | 28,50% | S0 36,83 |
| 015 | 60,0 | 2 | 2 | 12,04% | 0,00% | 0,03% | 12,02% | 0,00% | 0,00% | 0,00% | S0 22,16 · S1 21,18 |
| 016 | 67,3 | 2 | **1** | 20,29% | 15,64% | 15,41% | 4,88% | 15,48% | 0,16% | 0,00% | S0 49,53 |
| 017 | 70,7 | 2 | 2 | 14,96% | 5,03% | 14,96% | 0,00% | 5,03% | 0,00% | 0,00% | S0 34,95 · S1 11,79 |
| 018 | 60,0 | 2 | 2 | 11,28% | 1,26% | 2,20% | 9,04% | 1,22% | 0,03% | 0,04% | S0 35,3 · S1 17,65 |
| 019 | 52,8 | 2 | 2 | 17,68% | 13,77% | 17,28% | 0,40% | 13,77% | 0,00% | 0,00% | S0 6,23 · S1 31,1 |

Ba file sai số người nói (005, 014, 016) là 17/20 = 85,0% SpkExact, khớp với bảng thử nghiệm 4.

## 5. Số liệu tổng hợp (tính lại từ bảng theo từng file)

| Chỉ số | Trung bình | Trung bình theo độ dài | Trung vị |
|---|---|---|---|
| DER@0ms | 14,39% | 14,19% | 10,38% |
| DER@.25s | 7,40% | 7,24% | 1,63% |
| Miss@0ms | 3,90% | 4,00% | 1,29% |
| FA@0ms | 8,91% | 8,64% | 7,70% |
| Miss@.25s | 2,54% | 2,59% | 0,09% |
| FA@.25s | 3,64% | 3,43% | 0,00% |
| Conf@0ms | 1,57% | 1,55% | 0,00% |

Bỏ ba file sai số người nói (005, 014, 016): DER@0ms trung bình 9,16%, DER@.25s 2,27%, Conf 0,00%. Trong 20 file có 10 file DER@.25s ≤ 2% và 6 file đúng 0%.

## 6. Quan sát

Đây là những điều số liệu cho thấy, chưa phải kết luận về nguyên nhân.

1. **Thử nghiệm 4 giảm mạnh lỗi ở collar 0 nhưng ít ở collar 0,25 s.** So với baseline, DER@0ms giảm 5,6–6,3 điểm, phần lớn nhờ FA giảm từ 15,71% xuống 5,98–8,92%; Miss lại tăng từ 2,01% lên 3,29–5,52%. DER@.25s chỉ đổi từ 6,48% sang 6,01% (Balanced), 6,25% (Crisp), 7,62% (Tight, tệ hơn baseline). Tức là phần cải thiện chủ yếu nằm ở độ chính xác biên đoạn, và cấu hình Tight đổi FA lấy Miss (5,52%) nên thua ở collar rộng.
2. **Các cấu hình đã huấn luyện cắt vụn nhiều hơn.** Số đoạn dự đoán tăng từ 165 lên 271 / 297 / 435 (+64% / +80% / +164%), trong đó đoạn <0,5 s chiếm khoảng 7,7–8,1% so với 1,2% ở baseline. Đáng kiểm tra xem ASR và separation ở sau có chịu được các đoạn vụn này (mình chưa đo).
3. **SpkExact tăng từ 75% lên 85%** ở cả ba cấu hình đã huấn luyện (thêm 2 file đúng trên 20).
4. **Thử nghiệm 5 không đạt mục tiêu nào.** DER@0ms tốt nhất 13,68% (mục tiêu < 6%), Miss 3,27–4,42% (mục tiêu < 1%), FA 6,92–8,14% (mục tiêu < 2%), SpkExact 60–70% (mục tiêu ≥ 95%). So với dòng `1_Nemotron_Exp2+Quorum` của chính thử nghiệm 5, bộ tinh chỉnh 10 ms giảm DER@0ms tối đa 0,45 điểm nhưng Conf tăng (1,94% lên 2,35–2,85%) và SpkExact giảm (80% xuống 60–70%). Với cấu hình này nó chưa tốt hơn không dùng.
5. **Loss của bộ tinh chỉnh chững lại.** Từ epoch 20 đến 25 chỉ giảm 0,2299 xuống 0,2285, và epoch 10 (0,2570) cao hơn epoch 5 (0,2548). Log không có loss trên tập validation nên chưa biết có quá khớp hay không.
6. **Lỗi theo từng file tập trung vào ba loại.**
   - Sai số người nói: file 005 (4 người, DER 71,63%, gần như toàn FA 65,00%), 014 (1 người, Conf 28,50%), 016 (1 người, Miss 15,41%). Ba file này kéo DER@0ms trung bình từ 9,16% lên 14,39%.
   - FA do biên: 12 file có FA@0ms ≥ 5%, trong đó 10 file có FA@.25s ≤ 0,75%. Phần lớn false alarm ở collar 0 biến mất khi nới collar, giống hiệu ứng đệm biên (pad) hơn là phát hiện sai.
   - Miss thật: file 007, 016, 017, 019 vẫn còn Miss@.25s từ 5,03% đến 15,48%, tức bỏ sót tiếng nói chứ không phải lệch biên.
7. **Độ tin cậy:** tập test chỉ 20 file, mỗi file khoảng 60 s. Chênh lệch dưới khoảng 1 điểm DER giữa các cấu hình (ví dụ 13,68% so với 14,13%) không đủ để kết luận, và SpkExact nhảy 5 điểm mỗi file nên 85% so với 70% là chênh 3 file.

## 7. Việc cần làm tiếp để phân tích

- Biết chính xác cấu hình của bảng theo từng file (script bị dán lỗi), rồi chạy lại để lưu ra file (CSV hoặc JSON) thay vì chỉ in màn hình.
- Xem riêng file 005, 014, 016: đã có thời lượng từng người nói; cần xem phân bố xác suất người nói theo thời gian của ba file này để biết lỗi nằm ở mô hình (nhầm số người) hay ở hậu xử lý.
- Thử tách hai phần của lỗi biên: giữ nguyên mô hình, quét `pad_onset`, `pad_offset`, `onset`, `offset` (xem bảng tham số trong [nemotron_diarization_experiment.md](nemotron_diarization_experiment.md)) chỉ trên 17 file đúng số người để xem FA@0ms còn hạ thêm được bao nhiêu mà không làm Miss tăng.
- Với cấu hình có nhiều đoạn <0,5 s, đo ảnh hưởng tới các bước sau (tách giọng, ASR) trước khi chọn làm cấu hình mặc định; `merge_gap` và `min_duration` trong `models.nemotron` là hai núm có sẵn.
- Thêm loss validation cho bộ tinh chỉnh 10 ms trước khi tăng số epoch.

## 8. Ghi chú lúc chạy

- Lần chạy đầu của `exp5_breakthrough_nemotron.py` lỗi `SyntaxError: invalid syntax` ở dòng 247 (`active_ binary = P > offset`, chen khoảng trắng khi dán). Đã sửa bằng `sed -i 's/active_ binary/active_binary/g'` rồi chạy lại được.
- Khi chạy qua `cat << 'EOF'` nhiều lần, file được ghi đè nhiều lần với phần đầu giống nhau; các dòng lệnh dán lẫn vào trong docstring của file (ví dụ dòng `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. ...`). Nên kiểm tra lại file bằng `python -m py_compile` và đọc nhanh phần đầu trước khi tin kết quả.
