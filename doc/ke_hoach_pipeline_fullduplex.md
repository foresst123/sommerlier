# Pipeline phục chế audio từng người từ 1 micro

> Tài liệu đang xây dựng. Mọi con số dưới đây đều đo trên `hoahau.mp3` và
> `thu_that_thach_10m.mp3` trong `kaggle 2/`, không phải ước lượng.

## Bối cảnh

Mục tiêu: từ file **mono 1 micro**, lấy ra audio riêng của từng người ở chất lượng
studio, làm dữ liệu full-duplex. Nhánh ASR là phụ.

Kiến trúc *diarization lấy mồi → tách thô → generative bù tần số* là **đúng**, hội
tụ với trường phái Iterative Refinement đang thắng các giải 2025–2026. Tài liệu này
giữ kiến trúc đó, hiệu chỉnh theo số đo, và chỉ ra model cụ thể kèm giấy phép.

---

## Phần 1 — Đặc tính dữ liệu

| | hoahau (50.1p) | thu_that_thach (10.0p) |
|---|---|---|
| Định dạng | MP3 320 kbps, 48 kHz | MP3 328 kbps, 48 kHz |
| Kênh | 2 — **giống nhau từng bit** (0/8.640.000 mẫu) | 2 — giống nhau |
| Đỉnh / RMS | −6.5 / −28.5 dBFS | −7.0 / −27.0 dBFS |
| Crest factor | 21.9 dB | 20.0 dB |
| Clipping / DC | 0 mẫu / −1e−05 | 0 mẫu / −6e−06 |
| **SNR** | **35.3 dB** | — |
| Ổn định mức | **±1 dB / 45 phút** | — |
| **RT60** | **≈ 90 ms** | **≈ 45 ms** |

- High-pass dốc **105–115 Hz**; trên 8 kHz chỉ **0.0075%** năng lượng; cắt ~13.5 kHz
- Có tiếng **79%**; nghỉ p50 **0.06 s**; **chỉ 20 lần nghỉ >1 s / 50 phút**
- Nhạc: 30 s / 50 phút (hoahau), 0 s (thu_that_thach)
- Tỉ lệ nói **79/21**; lượt nói 2.9/phút, **max 229 s**; segment p50 2.23 s, 22% dưới 1 s
- **Overlap ghi nhận 1.51%** thời lượng nói
- F0 **213 Hz** và **135 Hz** — cách nhau **0.66 quãng tám**; trôi dạt chỉ 7–28 Hz

### Hệ quả cốt lõi

**98.5% audio là đơn giọng.** Với full-duplex, phần đó không cần tách — chỉ cần
*định tuyến* vào đúng kênh. Mixture khi chỉ một người nói **chính là** audio sạch
của người đó (SNR 35 dB).

---

## Phần 2 — Cái gì hỏng, cái gì không

### Bốn thứ đo ra KHÔNG hỏng

| Khâu | Đo được |
|---|---|
| Chất lượng audio | SNR 35.3 dB, 0 clipping, ±1 dB, RT60 45–90 ms |
| Biên segment | 92% / 89% trong **±50 ms** |
| Nhãn speaker | **98%** đúng (10/444 sai, hầu hết sát ranh giới 175 Hz) |
| Serialise hại xuôi dòng | mảnh bị cắt WER **7.2%** vs thường **7.7%** |

### Bốn thứ hỏng

**1. Kênh im lặng là số 0 tuyệt đối** — `export_sdlm_dual_channel` dựng track bằng
`np.zeros`:

| | kênh 1 | kênh 2 |
|---|---|---|
| hoahau | **82.6% là số 0** | 34.2% |
| thu_that_thach | 48.7% | **71.9%** |

**2. Chồng tiếng bị bỏ sót — độ lớn CHƯA ĐO ĐƯỢC.** Trong 55 backchannel đánh dấu
tay, **28 khiến lượt người kia bị cắt đôi**, chỉ **5** được nhận là chồng tiếng.

Tôi viết hai bộ dò hai-cao-độ, cả hai hỏng: bộ đầu kêu vì hài bậc 2 của giọng trầm
(135×2 = 270 Hz) rơi vào dải giọng cao; bộ sau loại quan hệ hài thì kêu ở **98% mọi
khung**, không phân biệt được khung diarization đánh dấu chồng (98.9%) với khung một
người (98.1%). **Cần model đã huấn luyện, không thay bằng DSP tự chế được.**

**3. Nhiễm chéo gần chuyển lượt — +7.3 điểm.** Có đối chứng để trừ nhiễu phép đo:

```
segment giữa lượt (nền nhiễu) : 12.7%
segment sát chuyển lượt       : 20.0%
-> nhiễm chéo THẬT            : +7.3 điểm
```

**4. Sidon chạy trên vùng không cần.** Đo trong và ngoài vùng overlap:

| Vùng | % audio | Lệch phổ | Cắt câm |
|---|---|---|---|
| Trong overlap | 1.5% | 5.91 dB | 0.0% |
| **Ngoài overlap** | **98.5%** | **1.83 dB** | **50.5% khung** |

Ngoài overlap chỉ có một người nói, audio đã sạch — nhưng chạy generative lên nó
làm mất 1.83 dB âm sắc và đẩy **một nửa số khung** xuống câm.

> **Nguyên tắc: chỉ xử lý nơi vật lý thật sự hỏng.**

---

## Phần 3 — Đánh giá các phương án generative

### Ba "siêu năng lực" của Resemble Enhance / AudioSR

| Năng lực | Dữ liệu này | Kết luận |
|---|---|---|
| **Denoise** | SNR **35.3 dB**, 0 clipping | Không có gì để khử |
| **Dereverb** | **RT60 45–90 ms** | Khô hơn phòng thu chuẩn (200–300 ms) |
| **Bandwidth extension** | >8 kHz chỉ **0.0075%**, cắt 13.5 kHz | Sẽ **bịa** nội dung chưa từng có |

Trên 98.5% audio đơn giọng: cả ba không có việc chính đáng — nhưng model generative
thì luôn *thay đổi* tín hiệu. Mất mát thuần tuý.

Trên 1.5% vùng overlap sau khi tách thô: lệch phổ **5.91 dB**, mất dải âm xát thật.
**Đây mới là chỗ generative thuộc về.**

### Hai rủi ro riêng cho mục tiêu full-duplex

1. **Bandwidth extension làm dataset không còn là bản ghi thật.** Nếu model
   full-duplex chạy ở 16/24 kHz thì dải sinh thêm là vô ích; nếu học từ đó, nó học
   một phân bố tần số cao **không tồn tại trong podcast thật**.
2. **Ảo giác thanh điệu.** Tiếng Việt có 6 thanh mang nghĩa, tải bởi đường cong F0.
   Model huấn luyện trên tiếng Anh có thể làm mượt hoặc đổi đường cong đó — nghe vẫn
   tự nhiên nhưng **sai từ**.

### Đánh giá 4 giai đoạn "không thoả hiệp"

| Giai đoạn | Đề xuất | Kết luận |
|---|---|---|
| **1. Profiling** | DiariZen WavLM-Large | ✅ **đã có sẵn** — `BUT-FIT/diarizen-wavlm-large-s80-md-v2`. Chỉ thiếu: embedding bị **vứt** ở `diarizen_worker.py:200` |
| **2. Bóc tách** | TIGER / USEF-TSE | ✅ **giá trị cao nhất** — sửa lỗi kiến trúc hiện tại |
| **3. Tái tạo** | Voicebox full-steps | ❌ **Voicebox không có trọng số chính thức** |
| **4. Upsampling** | AudioSR lên 48 kHz | ⚠️ khả dụng (MIT) nhưng **phản tác dụng** với dữ liệu này |

**Giai đoạn 2 quan trọng nhất.** Lý lẽ "discriminative giữ nguyên từ ngữ, không ảo
giác chữ" đúng, và nó chỉ ra lỗi kiến trúc: pipeline đang dùng **Sidon — model
generative — làm bước tách đầu tiên**, tức cho phép ảo giác ngay từ đầu, không có
neo tất định nào.

**Giai đoạn 3: "chạy full 100 bước" không sửa được vấn đề đã đo.** Sidon **đang**
chạy `num_steps=100`, tản giữa các lần chạy vẫn **4.8 dB**. Nguyên nhân không phải
ít bước, mà là **khởi tạo ngẫu nhiên** — `sidon_infer.py:118` `torch.randn(...)`.
Nhiều bước làm *mượt* hơn, không làm *ổn định* hơn.

**Giai đoạn 4 phục vụ việc nghe, không phục vụ mục tiêu.** Nâng lên 48 kHz nghĩa là
bịa toàn bộ dải 13.5–24 kHz. Nếu mục tiêu là **nghe**, đáng làm. Nếu là **dữ liệu
huấn luyện**, nó là tài sản giả.

---

## Phần 4 — BSS hay TSE

Hai loại "tách tất định" khác hẳn nhau:

| | **BSS** (tách mù) | **TSE** (tách có mồi) |
|---|---|---|
| Đầu vào | mixture | mixture **+ mồi giọng** |
| Đầu ra | 2 track, **không biết của ai** | **1 track** của đúng người đó |
| Cần gán kênh | ✅ phải dùng ECAPA | ❌ không cần |
| Rủi ro tráo kênh | ✅ có | ❌ không |
| Model | TIGER, MossFormer2, **Sidon** | SpEx+, USEF-TSE |

**Với mục tiêu này nên chọn TSE.** Pipeline hiện tại là BSS (Sidon ra 2 nguồn) cộng
gán bằng ECAPA, và `sim` chỉ đạt **p10 0.456 / p50 0.582 / p90 0.687** — giọng tự
nhiên với ECAPA thường 0.70–0.90. Toàn bộ logic `_maybe_swap`, `anchor_self`,
`not_a_fail`, `qc_sim` trong `tse_model.py` tồn tại **chỉ để giải quyết việc gán
kênh**. TSE xoá hẳn cả lớp lỗi đó.

---

## Phần 5 — Model đã kiểm chứng tồn tại

| Model | Trọng số | Giấy phép | Loại | Ghi chú |
|---|---|---|---|---|
| **SpEx+** `alibabasglab/log_wsj0-2mix_speech_SpEx-plus_2spk` | ✅ `.pt` | **apache-2.0** | **TSE** | Ưu tiên 1 — cùng hệ ClearerVoice |
| **TIGER** `JusperLee/TIGER-speech` | ✅ `.safetensors` | **apache-2.0** | BSS | 2.953 lượt tải; vẫn phải gán kênh |
| **USEF-TSE** `ZBang/USEF-TSE` | ✅ `.pth.tar` | **không ghi** | TSE | Rủi ro pháp lý; WHAM!/WHAMR! |
| **SoloSpeech** `OpenSound/SoloSpeech-models` | ✅ | **cc-by-nc-sa-4.0** | TSE | ⚠️ **PHI THƯƠNG MẠI** |
| **MossFormer2** `alibabasglab/MossFormer2_SS_16K` | ✅ | apache-2.0 | BSS | Công cụ đo đã viết sẵn |
| Sidon `sarulab-speech/DialogueSidon` | ✅ đang dùng | — | BSS sinh | w2v-BERT 2.0 + diffusion + VAE |

**Đánh bóng generative:** `audiosr` v0.0.7 (**MIT**) + `haoheliu/audiosr_speech`;
`resemble-enhance` v0.0.1 + `ResembleAI/resemble-enhance`.

**Không có trọng số công khai:** TS-VAD, NSD-MS2S, Voicebox (Meta chưa công bố;
`lucasnewman/voicebox-small` là bản tái hiện, 0 lượt tải), FlowTSE, TIGER-Large.

**Dò chồng tiếng:** `pyannote/segmentation-3.0` (**6.0M** lượt tải, powerset xử lý
chồng tiếng, gated `auto`), `pyannote/overlapped-speech-detection` (**26.7k**).

Tất cả đều huấn luyện trên **tiếng Anh**. Với tiếng Việt có thanh điệu, phải đo
trước khi tin.

---

## Phần 6 — Vấn đề kết nối

**Bốn biểu diễn giọng nói tính độc lập, không chia sẻ:**

| Nơi tính | Model | Sau đó |
|---|---|---|
| DiariZen (worker) | WavLM-Large | **vứt** |
| DiariZen | embedding phân cụm | **vứt** — `diarize()` chỉ trả `segments` |
| `models/tse_model.py:86` | ECAPA-TDNN | dùng cho QC |
| `sidon_infer.py` | w2v-BERT 2.0 | **vứt** |

Trong khi SOTA thì mỗi tầng **mồi** cho tầng sau. Embedding cần cho bước tách **đã
được tính** trong DiariZen rồi bị bỏ — sửa `diarizen_worker.py:200` là xong.

Một chỗ nữa: gộp segment (`cut_by_speaker_label`) làm **theo từng speaker**, nên
lượt của A bị chẻ đôi quanh backchannel của B chỉ liền lại khi `merge_gap` vượt
≈0.71 s. Hiện là **0.3**:

| merge_gap | segment | chỗ overlap | % thời lượng nói |
|---|---|---|---|
| **0.3** (hiện tại) | 932 | 39 | **0.77%** |
| 0.8 | 307 | 48 | 1.20% |
| 1.5 | 240 | 50 | **1.49%** |

Khiêm tốn, và trả giá bằng segment dài hơn hẳn.

---

## Phần 7 — Đường tắt từ đặc thù podcast

**F0 hai người cách nhau 0.66 quãng tám.** Phân loại chỉ bằng F0, độ chính xác cân
bằng theo cửa sổ:

| Cửa sổ | spk1 | spk2 | Cân bằng | Ngưỡng |
|---|---|---|---|---|
| 0.04 s (khung) | 89.3% | 83.4% | 86.4% | 168 Hz |
| 0.20 s | 89.8% | 85.9% | 87.9% | 170 Hz |
| **0.50 s** | 92.1% | 95.2% | **93.6%** | 174 Hz |
| **1.00 s** | 100% | 98.6% | **99.3%** | 178 Hz |
| 2.00 s | 100% | 100% | 100% | 188 Hz |

Backchannel dài p50 **0.42 s** — nằm đúng vùng 0.5 s cho 93.6%.

Đây là bộ dò speaker mức khung **gần như miễn phí** (autocorrelation), đúng vai trò
TS-VAD — mà TS-VAD lại không có trọng số công khai.

**Trôi dạt F0 theo phần tư thời lượng:**

| | Q1 | Q2 | Q3 | Q4 | Trôi dạt |
|---|---|---|---|---|---|
| hoahau spk1 | 203 | 207 | 213 | 201 | **12 Hz** |
| hoahau spk2 | 127 | 135 | 143 | 127 | **16 Hz** |
| thu_that spk1 | 142 | 140 | 142 | 146 | 7 Hz |
| thu_that spk2 | 217 | 202 | 207 | 189 | 28 Hz |

Trôi dạt 7–28 Hz so với khoảng cách 70–80 Hz → **một ngưỡng cố định dùng được cho
cả file**, không cần Graph Diarization chống drift.

> **Điều kiện.** Chạy được vì hai người là **một nam một nữ**. Cặp cùng giới không
> tách được bằng F0 — phải **đo độ tách F0 mỗi file** và chỉ bật khi đủ tách bạch.

**Hai đặc thù khác chưa dùng:** độc thoại dài (max 229 s → blip 0.4 s giữa monologue
gần chắc là backchannel, không phải đổi lượt), và bất đối xứng 79/21 (người thiểu số
là host: câu hỏi ngắn + backchannel).

---

## Phần 8 — Pipeline đề xuất

```
Audio mono
  │
  ├─ [0] Đo đặc tính file       → F0 hai người có tách bạch không?
  │                                (quyết định bật/tắt đường tắt Phần 7)
  ├─ [1] DiariZen               → nhãn + timestamp + embedding mỗi người
  │                                (embedding hiện đang bị VỨT)
  ├─ [2] Mồi (prompt)           → dùng lại mine_enrollments đã có
  │
  ├─ [3] ĐỊNH TUYẾN theo cờ
  │        • 98.5% một người   → CHÉP THẲNG mixture vào kênh, KHÔNG xử lý
  │        • 1.5% chồng tiếng  → sang [4]
  │
  ├─ [4] TÁCH — TSE có mồi      → SpEx+ chạy 2 lần, mỗi lần 1 người
  │                                tất định · không ảo giác chữ · không gán kênh
  ├─ [5] ĐÁNH BÓNG — generative → chỉ trên 1.5% vùng overlap
  │                                Sidon / Resemble Enhance / AudioSR — chọn bằng đo
  ├─ [6] Ghép lại               → khớp mức theo hai biên (đã làm, p90 9.6 dB)
  │
  └─ [7] Hai bản: full-duplex (GIỮ NỀN PHÒNG) | ASR
```

Bốn chỗ nối hiện chưa có: **[0]→[4]** (đặc tính file chọn phương pháp),
**[1]→[2]** (embedding dùng lại thay ECAPA tính lại), **[3]** định tuyến là hoàn
toàn mới, **[4]→[5]** (nháp làm mồi cho generative).

### Về việc thay Sidon

**Resemble Enhance và AudioSR không thay thẳng được Sidon.** Kiểm chứng model card:
Resemble Enhance là *denoising + enhancement* của **một giọng**; AudioSR là
*super-resolution* của **một luồng**. Cả hai **không tách người nói**. Thay thẳng
thì không còn gì tách overlap — kết quả là mixture sạch hơn, vẫn hai giọng.

**Nhưng bỏ Sidon thì được**, bằng cách tách đôi hai việc nó đang làm cùng lúc:

```
HIỆN TẠI
  mixture → Sidon (tách + sinh trong MỘT bước) → 2 track
            tản 4.8 dB · cắt câm 46% · tilt 2.10 dB · tương quan sóng 0.054

ĐỀ XUẤT
  mixture → SpEx+ (tách, TẤT ĐỊNH, có mồi)  → 2 track thô
          → generative đánh bóng TỪNG track  → chỉ trên 1.5% overlap
```

Lợi thế:
- **Hết tản ngẫu nhiên** — SpEx+ tất định, cùng đầu vào cho cùng đầu ra
- **Mỗi model làm việc nó được thiết kế cho**
- **Giữ nguyên từ ngữ ở bước quyết định nội dung**

**Ràng buộc kỹ thuật đã kiểm:** Sidon chỉ công bố `ssl_encoder.pt2`,
`diffusion_head.pt2`, `vae_decoder.pt2` — **không có VAE encoder**. Nên không thể
mã hoá bản nháp thành latent để khởi tạo diffusion. Cách nối khả thi là đưa bản nháp
làm **đầu vào** của Sidon thay cho mixture thô (`ssl_encoder` nhận waveform).

---

## Phần 9 — Các bước, xếp theo đòn bẩy đã đo

### Bước 1 — Định tuyến: ngừng xử lý 98.5% audio

Đòn bẩy lớn nhất, không phụ thuộc model nào.

Phần đơn giọng **chép thẳng mixture** vào kênh, không qua generative. Hiện
`process_overlaps` gán `e.enhanced_audio = waveform[...]` rồi mới splice vùng
overlap — cấu trúc đã gần đúng, cần đảm bảo phần ngoài overlap **không** bị
`spectral_restore` hay chuẩn hoá đụng vào.

Sửa: `services/separation_service.py`, `services/pipeline_service.py`.

### Bước 2 — Nền phòng thay cho số 0

`export_sdlm_dual_channel` (`services/separation_service.py:1158`) — **chưa bao giờ
được gọi trong luồng thật** dù đã có 4 test phủ. Nối vào, thay `np.zeros` bằng nền
lấy từ chính bản ghi.

Ràng buộc đã đo: **chỉ 20 khoảng nghỉ >1 s trong 50 phút** — phải lấy từ khoảng
0.3–0.5 s, lọc theo ba tiêu chí đã kiểm chứng trong `utils/music_map.py` (mức, đỉnh
khung, độ ổn định).

### Bước 3 — Đo overlap thật

Không thay được bằng DSP (Phần 2 mục 2).

`tools/measure_overlap.py` (chỉ đọc): chạy `pyannote/segmentation-3.0`, so với
diarization hiện tại **và** với bộ phân loại F0 ở Phần 7.

Điều kiện đã kiểm: `pyannote.audio==4.0.7` **đã có trong `requirements.txt`**, chưa
cài trên máy dev. `segmentation-3.0` gated `auto` — cần token HF đã chấp nhận điều
khoản; `main.py:95` đọc từ env/Kaggle secrets.

**Điểm quyết định.** Overlap thật ≈1.5% → diarization đã đúng. Nếu 3–5% → phải sửa
diarization trước khi bàn tới bước tách.

### Bước 4 — Thay bước tách bằng TSE tất định

Ứng viên theo thứ tự: **SpEx+** (apache-2.0, TSE) → USEF-TSE (không rõ license) →
TIGER (BSS, phải giữ tầng gán kênh).

Lấy embedding ra khỏi DiariZen (`diarizen_worker.py:200`) để dùng lại thay vì ECAPA
tính từ đầu.

Đo bằng `tools/compare_separators.py`: `floor dB`, `gated`, `tilt dB`, `fric >4k`.
Thêm **WER trên vùng overlap** (hiện 22.1%) để kiểm lý lẽ "giữ nguyên từ ngữ".

### Bước 5 — Chọn model đánh bóng bằng đo

Chạy cả ba (Sidon / Resemble Enhance / AudioSR) trên **cùng** bản nháp từ Bước 4.
So bằng `tools/compare_separators.py` cộng hai phép mới:

- **Độ lệch đường cong F0** — bắt ảo giác thanh điệu
- **WER trên vùng overlap** — hiện 22.1%

**Bắt buộc: giữ nguyên một bản không qua generative.** Bản "thật" và bản "đánh bóng"
phục vụ hai mục đích khác nhau, và chỉ bản thật mới kiểm chứng được.

### Bước 6 — Nhánh ASR (tuỳ chọn)

**362 lỗi (35%)** lấy lại được chỉ bằng chọn đúng bản — oracle **5.60%** so với
hiện tại **8.56%**. Ba model bổ sung nhau: mỗi cái thắng độc nhất ở 11–19% segment;
fusion thua model tốt nhất ở **28%** segment.

Tín hiệu thiếu là confidence, bị vứt ở `models/whisper_wrapper.py:74`,
`models/phowhisper.py:59`, `qwen3_worker.py:114`.

Mọi bộ chọn đơn giản đã thử đều **tệ hơn** hiện tại (theo độ dài 10.08%, đa số
9.71%, luôn Qwen3 10.16%). Muốn hơn phải có **tín hiệu mới**, không phải luật mới.

---

## Kiểm chứng

```bash
cd podcast-pipeline
python3 -m pytest tests/ -q
```

Nền: **278 passed, 2 failed** — hai lỗi có sẵn từ `ba5849f` (`keep_models`,
`prefix_cache`), không liên quan.

**Mốc mục tiêu chính:**

```
Sidon trên vùng đơn giọng : 1.83 dB tilt, 50.5% cắt câm   -> mục tiêu: không chạy
số 0 tuyệt đối / kênh     : 82.6% và 34.2%                -> mục tiêu ~6% (nền phòng)
nhiễm chéo sát chuyển     : +7.3 điểm trên nền 12.7%
backchannel nhận là chồng : 5 / 55
tản giữa các lần chạy     : 4.8 dB (đối chứng mixture 2.7)
```

**Mốc mục tiêu phụ**, trên `podcast-pipeline/tools/hoahau_edited.json` (932 segment
sửa tay): WER hiện tại **8.56%**, oracle **5.60%**.

---

## Ghi chú

Hai thay đổi lớn so với hướng cũ:

**Thôi coi separation là trung tâm.** 98.5% chỉ cần định tuyến đúng, và bốn khâu
từng bị nghi ngờ (audio, biên, nhãn, serialise) đều đo ra **không hỏng**.

**Khai thác đặc thù podcast.** F0 tách bạch 0.66 quãng tám cho bộ phân loại speaker
gần như miễn phí — đúng vai trò TS-VAD, mà TS-VAD thì không có trọng số công khai.

Điều duy nhất chưa đo được là **tỷ lệ overlap thật**; nó cần một model đã huấn
luyện, và nó quyết định Bước 3–4 có đáng làm hay không.
