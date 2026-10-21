# Hướng cải thiện tách giọng: đánh giá và thứ tự việc

Ngày 2026-09-10. Ghi lại vì đây là quyết định kiến trúc, và vì có một mâu thuẫn
trong repo cần xử lý trước khi bàn tiếp.

---

## 0. Câu hỏi xuất phát

Có bản audio clean làm tham chiếu, có bản đã qua xử lý / có nhiễu. Dùng bản
clean để khôi phục hoặc cải thiện bản kia được không?

Áp vào bài cụ thể của pipeline này: một khối audio gồm đoạn clean + đoạn chồng
tiếng, sau khi qua diarization/separation ra 2 stream. Dùng đoạn clean của
**cùng speaker** để cải thiện phần overlap đã tách.

---

## 1. Ba kịch bản dùng bản clean

| Kịch bản | Clean đóng vai gì | Áp được ở đây? |
|---|---|---|
| **A** — supervisory signal | Target để train model denoise | Không. Cần dataset lớn có cặp (noisy, clean); không làm được với vài cặp |
| **B** — reference lúc inference | Điều kiện hoá model (speaker embedding, TSE) | **Có, và đã code sẵn** — xem mục 4c |
| **C** — chỉ để đánh giá | Ground truth cho PESQ/STOI/SI-SDR | **Có, và đang thiếu nhất** — xem mục 4a |

Ba hướng triển khai đã cân nhắc:

1. **Thay blind separation bằng TSE** — VoiceFilter, SpEx+, TD-SpeakerBeam,
   WeSep, USEF-TSE. Reference là đoạn clean liền trước overlap.
2. **Cascade: separation → refinement điều kiện hoá bằng speaker** — giữ
   separator, thêm tầng tinh chỉnh (SGMSE+/StoRM, hoặc vocoder resynthesis từ
   HuBERT/WavLM + speaker embedding).
3. **Joint training với speaker consistency loss** — ép embedding vùng overlap
   khớp embedding vùng clean.

---

## 2. Mâu thuẫn phải xử lý trước

[audio-cleanliness.md](audio-cleanliness.md) là văn bản chính sách. Nó nói:

> "Pipeline này không thêm denoiser, speech enhancement..."
> "**bỏ đi cái đã có** thì được; **thêm vào cái chưa từng có** thì không."
> "**TSE** (tách người nói khi chồng tiếng): **cũng là masking**" (dòng 30)
> "**Sidon bị xoá, chỉ còn USEF.**" (dòng 170)

Code hiện tại ngược lại. `models/separation_backends.py`: USEF đã xoá, chỉ còn
Sidon — và Sidon **không phải masking**:

> "Sidon resynthesises through a diffusion head and a VAE decoder, so its output
> is audio the model produced, not audio the microphone recorded... That is a
> deliberate trade being made here, not an oversight — see doc/audio-cleanliness.md
> for the argument that ruled it out before."

Code biết nó đi ngược tài liệu và trỏ thẳng vào tài liệu. Tài liệu chưa được cập
nhật. **Ai đọc doc sẽ tin TSE là masking. Không phải.**

Hệ quả cho ba hướng ở trên:

- **Hướng 1 không phải ý tưởng mới** — nó là *quay lại* đúng thứ doc tưởng đang
  chạy. Vẫn hợp lý: lý do USEF ra đi (hằng số bị trôi khi để hai backend cùng
  tồn tại; cửa sổ 2s của ONNX graph bóp nghẹt context của Sidon) **tan biến nếu
  USEF là backend duy nhất**. Nhưng phải đọc docstring đó trước, không thì đi
  lại vòng cũ.
- **Hướng 2 sai phía ranh giới, và sai gấp đôi** — chồng một tầng sinh nữa lên
  một tầng vốn đã sinh. Với corpus huấn luyện, cái model bịa ra sẽ thành đặc
  trưng của "tiếng Việt nói" mà model sau học được.

**Việc cần làm: cập nhật audio-cleanliness.md cho khớp thực tế.** Không phải
việc giấy tờ — sắp ra quyết định kiến trúc dựa trên một văn bản mô tả sai hệ thống.

---

## 3. Học "phép biến đổi" H từ cặp (X, X') — không được

Ý tưởng: đo sự khác biệt giữa clean X và processed X', rồi áp cho audio khác.
Đây là bài toán System Identification, và câu trả lời phụ thuộc H thuộc loại nào:

| H là gì | Đo được không | Transfer được không |
|---|---|---|
| LTI (EQ, filter, reverb, mic, codec tuyến tính) | Impulse response qua Wiener deconv | Có, chỉ cần 1 cặp |
| Phi tuyến ổn định (compressor, distortion) | Volterra / Wiener-Hammerstein, hoặc CNN nhỏ | Cần nhiều cặp |
| Neural processing (denoiser, separator, VAE) | Chỉ đo được hiệu ứng | **Không** |

**Với Sidon thì còn tệ hơn ô thứ ba một bậc.** `sidon_infer.py:127`:

```python
latents = torch.randn((1, seq_len, latent_dim * 2), device=device, ...)
```

Không `manual_seed`, không `Generator`, không ở đâu trong file. Cùng một đầu vào
chạy hai lần ra hai kết quả khác nhau. H không chỉ phi tuyến và phụ thuộc nội
dung — **nó không phải một hàm**. Không system-identify được cái không phải hàm.

**Nhưng bộ công cụ chẩn đoán vẫn nên dùng, cho câu hỏi khác.** Đo coherence
γ²(f) giữa mixture và (trackA + trackB) cho một **con số** về đúng thứ mà chính
sách quan tâm: bao nhiêu phần đầu ra còn quan hệ tuyến tính với đầu vào, bao
nhiêu là bịa. Chạy được ngay trên file `_dump_tracks` đã ghi sẵn trong
`03_separation/audio/raw/`, không cần chạy lại model. Đây là cách biến tranh
luận nguyên tắc thành phép đo.

---

## 4. Ba thứ bị bỏ sót

### a. Đang thiếu thước đo, không thiếu model

`services/separation_service.py` tự khai:

```python
# Ngưỡng CHƯA HIỆU CHỈNH. Đối chiếu phân vị similarity trong log [TSE] và
# nghe các đoạn thất bại trước khi kết luận.
BSS_QC_SIM_THRESHOLD = 0.20
```

Trong khi `utils/enrollment_memory.py` đo được similarity **p50 = 0.58**. Ngưỡng
0.20 nằm xa dưới trung vị → **cổng QC gần như không chặn gì**. Cả corpus đang
được lọc bằng một con số chưa ai đo, và nhiều khả năng nó đang mở toang.

**Cách sửa: dựng tập đánh giá tổng hợp có ground truth.** Lấy hai đoạn solo sạch
của hai người trong cùng file (`clean_segments` + `mine_enrollments` đã làm sẵn
việc tìm), trộn số học ở SNR biết trước → có (mixture, nguồn A thật, nguồn B
thật). Chạy separation lên đó. Lúc này intrusive metrics **dùng được**: SI-SDR,
PESQ, STOI. Quan trọng nhất: vẽ ECAPA-sim theo SI-SDR thật để **hiệu chỉnh
ngưỡng** `BSS_QC_SIM_THRESHOLD` và `BSS_NOT_A_MARGIN`.

Rẻ, dùng toàn code sẵn có, trả lời đúng câu hỏi mà code ghi là đang bỏ ngỏ.

### b. "Phương án 0": không tách, chỉ loại

Doc đã chọn đúng cái này: *"Cách xử lý thay thế là **loại trừ**, không phải sửa
chữa"*. Pipeline đã cài: `bss_failed_spans`, strict export zero hoá, cảnh báo
khi >5%. Với corpus huấn luyện, bỏ 5% bẩn thường thắng resynthesize 5% đó.
Bất kỳ danh sách phương án nào cũng nên có nó ở vị trí số 0.

### c. Kịch bản B đã code xong nhưng đang tắt

`utils/enrollment_memory.py` chính là "dùng clean làm reference lúc inference" —
kiểu EvoTSE, lấy các span tách tốt bổ sung ngược vào enrollment. Docstring ghi:
trên file đã đo, 21% span vượt ngưỡng nhận, span đầu tiên đến ở vị trí 3/24.

```python
ENABLED = os.environ.get("BSS_MEMORY", "0")...   # mặc định TẮT
```

Đã viết, đã có test, chưa bao giờ bật. Thí nghiệm rẻ nhất để trả lời "clean
reference có giúp không" — một biến môi trường.

---

## 5. Vài điểm nhỏ

- **Noise profile subtraction (Y = X + N)** không áp được. Nhiễu ở đây là *giọng
  người khác* — phi dừng, phổ giống hệt tín hiệu cần giữ. Spectral subtraction
  trên speech-on-speech hỏng nặng.
- **Hướng 3 (joint training)** là dự án ở quy mô khác, và có vòng luẩn quẩn:
  huấn luyện trên corpus do chính một model sinh tạo ra.

---

## 6. Thứ tự việc

1. **Cập nhật `audio-cleanliness.md`** cho khớp code. (mục 2)
2. **Dựng harness eval tổng hợp** → hiệu chỉnh `BSS_QC_SIM_THRESHOLD` và
   `BSS_NOT_A_MARGIN`. (mục 4a)
3. **Bật `BSS_MEMORY=1`** chạy một file, so bằng harness ở bước 2. (mục 4c)
4. **Đo coherence** để biết Sidon bịa bao nhiêu → rồi mới quyết giữ Sidon hay
   quay lại masking. (mục 3)

**Hướng 1 và 2 chỉ bàn sau bước 2.** Không có thước đo thì đổi kiến trúc chỉ là
đổi cảm giác.

---

## 7. Việc còn treo từ đợt sửa 2026-09-10

Các sửa đổi ở commit trước **chưa chạy test lần nào** — máy phát triển lúc đó
không có Python chạy được. Cần chạy:

```
cd podcast-pipeline
python -m pytest tests/test_separation_logic.py -q
python scripts/measure_sidon_lag.py output/<audio>/03_separation/audio/raw/separated
```

Script thứ hai đo xem phần đệm 10 ms đã được gỡ đúng chưa. Đọc kết quả: trung vị
gần 0 là xong; còn lệch đều một giá trị khác 0 thì đó là quy ước khung nội bộ của
model, script in luôn số mẫu cần chỉnh vào `PAD_SAMPLES_IN`.
