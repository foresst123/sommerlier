# Tổng hợp kiến trúc diarization SOTA và hướng tối ưu cho Sommelier

Mục đích: một tài liệu tra cứu duy nhất để lần sau tiếp tục tối ưu diarization mà không phải tìm lại. Nó gộp (1) kết quả tra cứu về các kiến trúc SOTA, (2) những gì **đã kiểm chứng trong mã của dự án** (DiariZen, Nemotron, arm J "nemo-dz"), (3) danh sách việc nên thử, có thứ tự.

Liên quan: `bao_cao_buoc_diarization.md` (báo cáo bước), `ke_hoach_toi_uu_diarization.md` (kết luận tranh luận trước), `nemotron_toi_uu_viyt_diar.md` (tinh chỉnh Nemotron), `loc_nhieu_hai_tang_va_tang_3.md` và `so_sanh_loc_nhieu.md` (khử nhiễu), mã trong `podcast-pipeline/utils/dz_pipeline.py`, `tools/windowed_vbx_stage.py`, `tools/diar_arms_experiment.py`.

## 0. Cách đọc mức chắc chắn

| Nhãn | Nghĩa |
|---|---|
| **[MÃ]** | Đã đối chiếu với mã nguồn DiariZen hoặc mã của dự án |
| **[ĐO]** | Đã đo trên 5 clip mẫu của dự án (chưa có nhãn vàng, nên chỉ là so sánh tương đối) |
| **[TRA]** | Lấy từ tra cứu web hoặc bài báo, chưa kiểm lại; con số có thể lệch |
| **[CHƯA]** | Suy luận hoặc ý tưởng, chưa có bằng chứng |

Bản tra cứu gốc do một mô hình khác viết, có dùng tìm kiếm. Con số benchmark và tham số trong đó cần kiểm lại trước khi trích dẫn ra ngoài (xem mục 9).

## 1. Bài toán và trạng thái hiện tại

Sommelier biến bản ghi dài tiếng Việt (phim, series YouTube) thành đoạn hội thoại hai người sạch. Chất lượng quan trọng nhất là **nhãn người nói đúng và sạch**, không chỉ DER.

Các nhánh đã có (trong `tools/diar_arms_experiment.py`):

| Arm | Mô tả |
|---|---|
| A | Nemotron-3-Diarization gốc: hai lượt, cửa sổ 25 s, nối nhãn bằng WeSpeaker ResNet-293 cosine 0.6, hậu xử lý a100 |
| B | DiariZen gốc (`diarizen-wavlm-large-s80-md-v2`) |
| F, G, I | Nemotron cửa sổ + gom cụm AHC / VBx (thử nghiệm, G sụp về 1 người) |
| **J** | **nemo-dz**: xác suất 8 khe của Nemotron trên cửa sổ 16 s hop 0.8 s, rồi đúng các bước embedding / AHC / VBx / dựng lại của DiariZen |

Kết quả đã biết:
- Người dùng nghe trực tiếp và thấy arm J "tốt hơn nhiều", kết hợp được ưu điểm của cả hai **[ĐO, nghe]**.
- Chưa có nhãn vàng. DiariZen (arm B) chỉ là tham chiếu, không phải sự thật. Vì vậy chưa biết người thứ ba mà DiariZen thấy ở nhiều clip là thật hay do tách thừa **[ĐO]**.
- Phân tích hai chuyên gia, hai vòng: cải thiện biên và độ phủ (sau Hungarian) không đổi danh tính; cải thiện danh tính cần nhãn vàng **[ĐO]**.
- Công cụ đánh nhãn tay: `tools/label_page.html` + `label_server.py` (chọn vùng bằng hai thanh kéo), so sánh bằng `tools/compare_arms.py --ref labels`.

## 2. Kiến trúc tham chiếu (hybrid EEND-VC)

Chuẩn hiện nay: phân đoạn cục bộ bằng mạng nơ-ron trên cửa sổ ngắn (xử lý được nói chồng), rồi gom cụm toàn cục các embedding để nối nhãn xuyên suốt file dài. Sơ đồ:

```
audio 16 kHz mono
 -> cửa sổ trượt 10-16 s, hop ~10% (DiariZen: 16 s, hop 1.6 s mặc định; cấu hình a100 của ta 0.8 s)
 -> phân đoạn cục bộ: DiariZen = WavLM-Large cắt tỉa 80% + 4 Conformer + powerset
                       Nemotron = Fast-Conformer + Transformer, 8 đầu ra sigmoid
 -> mặt nạ: giữ khung chỉ một người nói, bỏ khung chồng và đoạn quá ngắn
 -> embedding (WeSpeaker ResNet34-LM, 256 chiều) cho mỗi (cửa sổ, người nói cục bộ)
 -> AHC khởi tạo, rồi PLDA + VBx            (gán danh tính toàn cục)
 -> dựng lại: nhãn cục bộ -> nhãn toàn cục, cộng chồng cửa sổ, nhị phân hoá
 -> hậu xử lý: lấp khoảng hở, bỏ đoạn quá ngắn, giao với VAD, (tuỳ chọn) DOVER-Lap, DiarizationLM
```

Điểm kiến trúc đã kiểm chứng trong mã DiariZen **[MÃ]**:
- Powerset: tối đa 4 người mỗi cửa sổ, tối đa 2 người nói chồng (1 + 4 + 6 = 11 lớp).
- Embedding nhận **cả cửa sổ** kèm `weights=` là mặt nạ (khung chỉ một người nói; nếu quá ít thì dùng cả mặt nạ khe). Embedding đóng băng, không tinh chỉnh.
- Tập gom cụm: khe có ít nhất 10% số khung sạch của cửa sổ.
- AHC (centroid, euclid, ngưỡng 0.6) rồi VBx với PLDA của DiariZen: `lda_dim 128`, `Fa 0.07`, `Fb 0.8`, 20 vòng, `loopProb 0`.
- Tâm cụm = tổng embedding thô có trọng số q; mọi (cửa sổ, khe) được chấm `2 - khoảng cách cosine`, Hungarian từng cửa sổ.
- Số người mỗi khung = làm tròn trung bình số khe bật qua các cửa sổ phủ khung; mỗi khung giữ top-`count` cụm; nhị phân hoá ở 0.5. Không có onset/offset/pad/min_duration.

Khác biệt Nemotron so với DiariZen **[MÃ]**:
- Nemotron là 8 khe sigmoid độc lập, không powerset, huấn luyện trên phiên 105 s với streaming có bộ nhớ khe (AOSC). Cửa sổ 16 s ngắn hơn độ dài huấn luyện.
- **Khung đầu ra của Nemotron là 10 ms**, đã lấy từ thực nghiệm. Con số 80 ms trong nhiều tài liệu là khung của bộ mã hoá (giảm mẫu 8x) phía sau tham số streaming. Báo cáo và mã của ta đã sửa chỗ này.

## 3. Danh mục mô hình và pipeline (đối chiếu)

Số liệu DER là **[TRA]**, điều kiện chấm (collar, có tính nói chồng hay không) khác nhau giữa các nguồn; không so các cột với nhau.

| Hệ thống | Cốt lõi | Điểm đáng chú ý | Hợp với ta? |
|---|---|---|---|
| PyannoteAI precision-2/3 | EEND-VC thương mại | Độc quyền, không có trọng số, không tinh chỉnh được | Không (chỉ tham khảo) |
| DiariZen (WavLM-L s80 md v2) | WavLM cắt tỉa + Conformer powerset + ResNet34 + VBx | SOTA mã nguồn mở; tối đa 4 người mỗi cửa sổ | Đã dùng (arm B và làm khuôn cho J) |
| TS-VAD+, MIMO-TSVAD | Vòng hai: tinh chỉnh theo hồ sơ người nói | Thắng các cuộc thi học thuật; cần phân cụm vòng một; bỏ sót người nếu vòng một không thấy | Ý tưởng thử (mục 7) |
| Nemotron-3-Diarization, Sortformer v2.x | Fast-Conformer + Transformer, sắp xếp theo thời điểm xuất hiện, bộ nhớ AOSC | Nhanh, streaming, 8 người (Nemotron-3), 4 người (Sortformer v2.x); đã tinh chỉnh tiếng Việt trong dự án | Đã dùng (arm A, và phân đoạn của J) |
| pyannote community-1 (4.0) | SincNet + BiLSTM powerset (3 người, 10 s) + ResNet34 + PLDA/VBx | Nhẹ, nhanh, giấy phép mở; yếu hơn DiariZen khi nhiễu và vang | Có thể làm nhánh thứ ba cho DOVER-Lap |
| 3D-Speaker (CAM++, ERes2NetV2) | VAD + embedding đa tỷ lệ + spectral; có nhánh ngữ nghĩa/hình ảnh | Embedding nhỏ mà mạnh; không có phân đoạn chồng | Embedder ứng viên (mục 5) |
| EEND-EDA / GLA / TA | Thuần end-to-end với attractor | Không cần gom cụm; cần dữ liệu mô phỏng rất lớn | Không (cần huấn luyện lớn) |
| NeMo MSDD | Embedding đa tỷ lệ (0.5-3 s) + decoder trọng số động | Bắt biên tốt, không giới hạn số người | Ý tưởng đa tỷ lệ, chưa thử |
| pyannote 3.1, Argmax SpeakerKit | PyanNet + ResNet34 + AHC ngưỡng 0.7045 | Phổ biến, nhẹ; AHC nhạy ngưỡng | Không |
| BUT VBx | x-vector + PLDA + Bayes HMM | Nền toán học của cả DiariZen và pyannote 4.0 | Đã dùng qua DiariZen |

## 4. VBx: cách nó hoạt động và cách chỉnh

Cơ chế **[TRA, khớp với mã DiariZen]**: các embedding (sau LDA, PLDA) được mô hình hoá như chuỗi quan sát của một HMM với tối đa S trạng thái ẩn (ta đặt `max_speakers=20`). Variational Bayes cập nhật hậu nghiệm của trạng thái, của biến người nói và của trọng số `pi` để tối đa ELBO. Cụm yếu bị ép `pi -> 0`, nên số người nói tự hội tụ mà không cần ngưỡng khoảng cách cứng.

Hai tham số chính:
- **Fa** (hệ số co giãn bằng chứng âm học): giảm bằng chứng vì các đoạn trượt chồng nhau không độc lập.
- **Fb** (hệ số phạt số người): tăng Fb thì mở cụm mới "đắt" hơn, nên gộp nhiều hơn; giảm Fb thì tách nhiều hơn.
- `loopProb`: xác suất giữ trạng thái giữa hai đoạn liên tiếp.

**Cảnh báo về giá trị tham số** (mâu thuẫn trong nguồn):
- Bản tra cứu viết "Fa 0.3-0.4, Fb 11-17, P_loop 0.8-0.9". Đó là cấu hình của pipeline BUT VBx gốc, với x-vector cắt đoạn 1.5 s hop 0.25 s, nhiều embedding nối tiếp nhau.
- **DiariZen thực tế dùng `Fa 0.07, Fb 0.8, loopProb 0`** **[MÃ]** (từ `config.toml`), vì embedding của nó theo cửa sổ và khe, không phải chuỗi đoạn liên tiếp. Arm J dùng đúng bộ này.
- Hệ quả: **không được áp các khoảng 0.3-0.4 / 11-17 vào arm J.** Quy tắc chỉnh "Fb tăng thì gộp, Fb giảm thì tách" vẫn đúng, nhưng thang đo khác. Khi cần chỉnh, quét quanh giá trị DiariZen (ví dụ Fb 0.4, 0.8, 1.6) và đo bằng nhãn tay.
- Cũng không chỉnh `min/max speakers` trên VBx: DiariZen không dùng chúng trong VBx (chỉ dùng khi quay về AHC).

Biến thể mới: SphereVBx / SphereVBx-PF (arXiv 2606.24528, đang tra cứu): thay PLDA Gaussian bằng mô hình von Mises-Fisher trên mặt cầu; bản PF không cần backend huấn luyện sẵn, gần với cosine scoring. Trong bảng cascade của bài, DER trung bình 22.7 (VBx) / 22.1 (SphereVBx) / 22.2 (PF) trên 8 tập **[TRA]**, tức cải thiện nhỏ. Ý nghĩa với ta: PF không phụ thuộc PLDA, nên dễ đi cùng embedder khác. Ký hiệu "MS" và "[6][16]" trong "MS-SphereVBx-PF[6][16]" **chưa đọc được** vì phiên chưa truy cập arxiv.org.

## 5. Embedding người nói

Đang dùng: `pyannote/wespeaker-voxceleb-resnet34-LM` (256 chiều), đóng băng, **giống DiariZen**, vì PLDA của DiariZen gắn với nó **[MÃ]**. Bước nối nhãn của arm A/F dùng WeSpeaker ResNet-293 ONNX.

Ứng viên khác **[TRA]**: WeSpeaker ResNet152/221/293 (EER VoxCeleb1-O thấp hơn ResNet34), CAM++ (3D-Speaker, 7.2M tham số), ERes2NetV2, TitaNet-L (NeMo), ReDimNet, ECAPA2, embedding dựa trên WavLM.

Ràng buộc quan trọng:
1. **Đổi embedder làm PLDA của DiariZen hết khớp.** Phải gom cụm bằng AHC cosine hoặc huấn luyện lại PLDA, hoặc dùng SphereVBx-PF.
2. Embedding của ta đến từ các đoạn rất ngắn có mặt nạ; chênh lệch giữa các embedder trên benchmark câu dài (VoxCeleb) co lại ở độ dài này.
3. Hầu hết model huấn luyện trên VoxCeleb (đa số tiếng Anh) hoặc dữ liệu tiếng Trung. Nhận định "CAM++ rất mạnh với tiếng Việt" **không có bằng chứng**; đừng ghi vào báo cáo.
4. Không có nhãn vàng thì không biết embedder nào tốt hơn. Hướng có khả năng giải quyết lệch miền (tiếng Việt, đoạn ngắn, phim) nhất là **tinh chỉnh embedding trên ViYT-Diar** (100 clip có nhãn), nhưng tốn công.

## 6. Khử nhiễu: quy tắc bất đối xứng

Nguồn gốc là TS-VAD+ (khử nhiễu bằng DEMUCS) và MIMO-TSVAD **[TRA]**:

- **Không khử nhiễu toàn bộ audio trước nhánh phân đoạn.** Mô hình khử nhiễu hay xoá giọng người nói thứ hai ở vùng chồng và gây méo biên; WavLM đã có tính kháng nhiễu.
- **Chỉ khử nhiễu các đoạn dùng để tạo hồ sơ/embedding người nói** (nhánh profile), để chống "embedding leakage" (nhiễu hoặc giọng người khác ngấm vào vector đại diện).
- Số liệu của TS-VAD+ trên DIHARD-III: DER 19.53 -> 17.97; False Alarm 7.15 -> 4.11; Speaker Confusion 2.72 -> 2.49 (kèm lọc biên bằng VAD ngoài) **[TRA]**.
- MIMO-TSVAD: tinh lọc hồ sơ lặp nhiều giai đoạn (giữ khung đơn thoại tin cậy để tính lại hồ sơ), tuỳ chọn dùng chuyển động môi làm mỏ neo; huấn luyện có che/xáo trộn hồ sơ để chịu lỗi.

Ý nghĩa cho dự án (nhánh `exp/denoise-compare`): arm J cho phép thử đúng quy tắc này, vì phân đoạn (Nemotron) và embedding là hai bước rời. Thử: phân đoạn trên audio gốc, nhưng embedding trích từ audio đã khử nhiễu, rồi so sánh với nhãn tay. Xem `so_sanh_loc_nhieu.md` về cách đo khử nhiễu đã làm, và nhớ rằng một số bộ khử nhiễu làm lệch đặc trưng thanh quản. **[CHƯA]** về hiệu quả trên dữ liệu của ta.

Tiền xử lý khác trong nguồn tra cứu, chưa kiểm: lọc thông cao 60-80 Hz, WPE khử vang (chỉ khi vang nặng) **[CHƯA]**.

## 7. Vòng hai và hậu xử lý

**Vòng hai kiểu TS-VAD** **[TRA]**: lấy hồ sơ từng người từ kết quả gom cụm, ghép với đặc trưng khung (cross-attention), quét lại xác suất hoạt động từng người. Cải thiện biên và vùng chồng. Nhược điểm: cần vòng một chạy xong; bỏ sót người nói nếu vòng một không thấy họ. Cần huấn luyện model riêng, nên chỉ nên làm sau khi có nhãn và đã khai thác hết các cải tiến rẻ hơn.

Hậu xử lý chuẩn **[TRA]**, đối chiếu với pipeline hiện tại:

| Bước | Ý | Trong pipeline của ta |
|---|---|---|
| Overlap-add xác suất | Làm mượt nhãn qua cửa sổ chồng | Có trong dựng lại của arm J |
| Gán lại đoạn ngắn | Đoạn bị loại lúc gom cụm gán về cụm gần nhất theo cosine | Chưa có cho arm J (`active_unassigned` đang là 0) |
| Lấp khoảng hở / bỏ đoạn quá ngắn | Nối khoảng lặng nhỏ, bỏ đoạn lẻ | Có ở `utils/segment_utils` (`compare_arms --postprocess`) |
| Giao với VAD ngoài (Silero) | Cắt false alarm | Có Silero trong pipeline |
| DOVER-Lap | Bỏ phiếu nhiều hệ thống (ví dụ A + B + J) | Chưa có |
| DiarizationLM | LLM sửa nhãn biên theo ngữ cảnh | Đã có bước relabel |
| Gán nhãn mức từ | Giao thời gian với ASR mức từ | Đã có theo kiến trúc ASR |

## 8. Việc nên thử tiếp (có thứ tự)

Nguyên tắc: **việc nào cần nhãn vàng thì chờ nhãn.** Hai cải tiến biên/độ phủ không đổi được danh tính.

1. **Đánh nhãn tay 3-5 clip** bằng `label_page.html`, rồi chạy `compare_arms.py --ref labels --postprocess` cho A, B, J. Mọi mục dưới đây phụ thuộc bước này để biết có thật sự tốt hơn.
2. **J2**: hysteresis không pad (onset/offset + min_on/min_off) thay cho ngưỡng 0.5 đơn thuần; kích hoạt cụm mềm và đếm người mềm; lọc khe mơ hồ (14.9% khung mơ hồ trong `inspect_probs`). Không cần huấn luyện.
3. **Thử khử nhiễu bất đối xứng**: embedding từ audio khử nhiễu, phân đoạn từ audio gốc (mục 6).
4. **Quét Fa/Fb quanh cấu hình DiariZen** (không dùng thang 0.3/11-17) và ngưỡng `thr` 0.35-0.5, `--min-frames-ratio`.
5. **So embedder**: ResNet293 + AHC cosine so với ResNet34 + VBx. Thêm lựa chọn `--clustering` mới cho SphereVBx-PF nếu đọc xong bài báo.
6. **DOVER-Lap giữa A, B, J** (rẻ, có sẵn thư viện); nên thử khi đã có nhãn để biết có giúp không.
7. **Gán lại đoạn ngắn** về cụm gần nhất cho arm J.
8. **Dài hạn, tốn công**: tinh chỉnh embedding trên ViYT-Diar; vòng hai TS-VAD; embedding đa tỷ lệ kiểu MSDD.

**Đã viết mã, chưa kiểm chứng (arm K và quét Fb).** Arm K (`nemotron_dz_demucs`) là arm J với HT Demucs chỉ cho nhánh embedding: xác suất giữ audio gốc, embedding lấy từ `chunk_vocals.wav`. Chạy (`$AUDIO` là thư mục nguồn, `$OUT` là thư mục đầu ra):
`python tools/diar_arms_experiment.py --audio-dir $AUDIO --samples tools/diar_arms_samples.json --out $OUT/diar_arms --only-arms K --windowed-python $NEMOTRON_PY --vbx-python $DZ_PY --demucs-python $DEMUCS_PY --python $SOM_PY`
Quét Fb cho J và K (chỉ chạy stage 2, không chạy lại stage 1 hay Demucs; kết quả là `nemotron_dz_fb0.2`, `nemotron_dz_demucs_fb0.4`, ...):
`python tools/diar_arms_experiment.py --audio-dir $AUDIO --samples tools/diar_arms_samples.json --out $OUT/diar_arms --only-arms J,K --fb-sweep 0.2,0.4 --windowed-python $NEMOTRON_PY --vbx-python $DZ_PY --demucs-python $DEMUCS_PY --python $SOM_PY`
HT Demucs là bộ tách nguồn **âm nhạc** (stem giọng hát), không phải bộ khử nhiễu lời nói; nó chỉ thử trên nhánh embedding. Chưa kiểm chứng: Demucs chưa chạy trên GPU, các giá trị Fb 0.2 và 0.4 chỉ là điểm quét đầu tiên, và chưa có nhãn tay để so.

**Đã viết mã, chưa kiểm chứng (biến thể nhạy hơn với chồng lấn).** `--overlap-variants` thêm arm `nemotron_dz_ov_thr<thr>_on<min_on>` cho J và `nemotron_dz_demucs_ov_thr<thr>_on<min_on>` cho K (mặc định `_ov_thr0.35_on0.2`, tức `--ov-thr` 0.35 và `--ov-min-on-s` 0.2 s); J và K giữ nguyên làm mốc. Tên cũ `nemotron_dz_ov` chưa từng được chạy. Hai arm này chỉ chạy stage 2 trên xác suất và vocals đã có. J và K không có onset/offset của NeMo; chỗ tương ứng là `--thr` và bộ lọc trung vị. Stage 2 bỏ mọi đoạn ON của từng cụm ngắn hơn 0.2 s (đoạn đúng 0.2 s được giữ). `compare_arms.py` thêm hai cột cuối: `overlap_s` (giây có từ hai người nói trở lên) và `overlap_ratio` (chia cho giây có tiếng nói), tính trên segment không có collar, cho cả arm và tham chiếu. Ví dụ:
`python tools/diar_arms_experiment.py --audio-dir $AUDIO --samples tools/diar_arms_samples.json --out $OUT/diar_arms --only-arms J,K --overlap-variants --windowed-python $NEMOTRON_PY --vbx-python $DZ_PY --demucs-python $DEMUCS_PY --python $SOM_PY`
Giá trị 0.35 chỉ là điểm đoán đầu tiên, chưa kiểm chứng trên dữ liệu nào. Ngưỡng thấp hơn cũng làm tăng báo động giả (khoảng không có người nói bị gán là có tiếng), nên phải đọc `miss`/`FA` cùng với `ovl %`.

**Đã viết mã, chưa kiểm chứng (lưới tham số overlap và thử embedder với cosine AHC).** `--ov-thr` và `--ov-min-on-s` nhận danh sách cách nhau bằng dấu phẩy; mỗi tổ hợp là một arm cho J và K, ví dụ `--ov-thr 0.3,0.35 --ov-min-on-s 0,0.2` cho `nemotron_dz_ov_thr0.3_on0`, `nemotron_dz_demucs_ov_thr0.35_on0.2`, và các tổ hợp còn lại. Thr phải nằm trong (0, 1), min-on không âm, giá trị trùng sau định dạng bị từ chối, tối đa 24 tổ hợp cho mỗi arm. `--embedder-trials` chạy riêng stage 2 của J với `--clustering dz-ahc`: AHC liên kết trung bình trên khoảng cách cosine, cắt tại `--cos-thr` (mặc định 0.4, 0.5, 0.6). Không dùng VBx vì PLDA của DiariZen chỉ được huấn luyện cho ResNet34-LM. Ví dụ thử embedder:
`python tools/diar_arms_experiment.py --audio-dir $AUDIO --samples tools/diar_arms_samples.json --out $OUT/diar_arms --only-arms J --embedder-trials r34lm,wespeaker_r293,wespeaker_simam100 --cos-thr 0.4,0.5,0.6 --windowed-python $NEMOTRON_PY --vbx-python $DZ_PY --embedder-python wespeaker_r293=$SOM_PY --embedder-python wespeaker_simam100=$SOM_PY --r293-onnx $R293_ONNX --simam-onnx $SIMAM_ONNX --python $SOM_PY`
Âm thanh đưa vào embedder là phép nối các khung chỉ có một người nói (mặt nạ > 0.5), và cửa sổ có ít hơn 0.5 s như vậy không cho embedding (kết quả rỗng, cửa sổ đó bị bỏ qua). Adapter SimAM-ResNet100 (huấn luyện trước trên VoxBlink2, tinh chỉnh trên VoxCeleb2; CC BY-NC-SA 4.0, chỉ phi thương mại) và ESPnet-SPK được viết từ API công khai theo trí nhớ và **chưa chạy**: người dùng phải tự tải trọng số, nên `--espnet-tag`, `--simam-onnx` và `--simam-dir` chưa được kiểm chứng. Lần viết này không chạy mô hình nào (không có trọng số, không có GPU); test chỉ dùng fake. Điều đã thấy với fake: khi AHC chỉ còn một cụm, các slot không có cụm được đẩy sang id 0 theo cách của DiariZen, nên một ngưỡng cosine lớn chưa chắc cho ra một người.

Chưa giải thích được: G sụp về 1 người (nghi do 30 s và ít embedding; chưa tách nguyên nhân). Chưa xem kết quả chạy thr 0.35 và median 0.22.

**Đã định nghĩa biến thể, chưa chạy biến thể nào (`podcast-pipeline/tools/variants.py`).** Quy trình có hai pha rồi mới chạy:
- Pha 1 (xong): registry 17 biến thể gồm J, K, cặp overlap (thr 0.35, min-on 0.2), Fb 0.2 và 0.4 cho J và K, embedder bằng cosine AHC và bằng VBx (r34lm, wespeaker_r293, wespeaker_simam100, espnet_spk). Tham số chưa đo nằm trong `tools/variant_params.example.json`, mọi giá trị là null ("filled after measurement").
- Pha 2 (chưa làm): đo PLDA cho từng embedder bằng `tools/train_plda.py` (ghi `plda_meta.json` với `embedder_id`) và đo cos cut trên clip đã gán nhãn tay; điền kết quả vào một bản `params.json`.
- Liệt kê: `python tools/variants.py list`. Dry-run (không chạy gì): `python tools/variants.py commands --params params.json --samples tools/diar_arms_samples.json --out $OUT --only J_base,K_base`. Biến thể còn tham số null bị từ chối và in tên tham số; lệnh của J và K trùng với lệnh của `diar_arms_experiment.py` (test trong `tests/test_variants.py`).
- SC-pNA (nhánh `nemo_sc`) CHƯA được cài đặt: biến thể `sc_pna` có trạng thái `todo` và không chạy. VBx với embedder khác r34lm cũng `todo`, vì `windowed_vbx_stage.py` chưa nhận `--embedder-kind` trên đường VBx.
- Chạy song song trên hai GPU chưa viết: bước sau sẽ chạy từng lệnh trên một GPU (CUDA_VISIBLE_DEVICES), sau khi xác suất (và Demucs cho nhánh K) của mẫu đó đã có.
- Guard: `windowed_vbx_stage.py --plda-dir` từ chối PLDA có `embedder_id` khác embedder, hoặc thiếu `plda_meta.json` với embedder khác r34lm. Đường mặc định (r34lm, PLDA của DiariZen) không đổi.
- Chưa có biến thể nào được chạy và chưa có số liệu mới từ bước này.

## 9. Việc cần kiểm lại trước khi trích dẫn

Những ý trong bản tra cứu gốc mà ta **chưa kiểm** hoặc **đã thấy sai lệch**:

1. Khoảng Fa/Fb/P_loop (mục 4): sai cho DiariZen, đã thay bằng giá trị đọc từ mã.
2. Benchmark: nhiều ô ghi "~" là ước tính; điều kiện chấm khác nhau giữa các nguồn; một số nguồn trích là tổng hợp của chính mô hình tra cứu. Đặc biệt các con số "ETH Zurich đa ngôn ngữ", "RTF 214x", "1912x realtime", "14 s mỗi giờ trên H100" chưa xác minh.
3. Nhận định "CAM++/ERes2NetV2 đặc biệt mạnh cho tiếng Việt": không có bằng chứng (mục 5).
4. Nhận định "WavLM chỉ giảm 80% tham số mà DER hầu như không đổi", "tốc độ gấp gần 4 lần": số liệu của nhóm DiariZen, chưa đối chiếu.
5. Khung 80 ms của Nemotron: sai cho đầu ra (là 10 ms), đúng cho khung bộ mã hoá (mục 2).
6. Kích thước Tensor và số lớp của từng mạng (ví dụ 17 Fast-Conformer + 18 Transformer, hidden 192/512): lấy từ tra cứu, chưa so với cấu hình thật.
7. Mô tả chi tiết cơ chế AOSC "6 bước": một phần là diễn giải, đừng coi là đặc tả.
8. Số liệu TS-VAD+/MIMO-TSVAD: lấy từ tra cứu, bài báo gốc chưa đọc.
9. SphereVBx-PF: "MS" và "[6][16]" chưa rõ; bài báo chưa đọc được vì `arxiv.org` bị chặn trong phiên.
