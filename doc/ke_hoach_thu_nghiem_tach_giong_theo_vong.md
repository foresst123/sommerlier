# Kế hoạch thử nghiệm bộ tách giọng theo vòng (fine-tune trên tiếng Việt)

Ngày 2026-10-20. **Chưa thực thi gì trong tài liệu này**: đây là kế hoạch và các
quyết định còn mở. Số liệu về mô hình lấy từ đợt khảo sát ngày 2026-10-19 (đọc code,
README, header checkpoint; **chưa chạy thật mô hình nào**). Mục "Chưa xác nhận" ở cuối
liệt kê những gì nên coi là giả định.

Liên quan: [thu_nghiem_mossformer2.md](thu_nghiem_mossformer2.md) (MossFormer2 so với
Sidon trên dữ liệu thật), [huong-cai-thien-tach-giong.md](huong-cai-thien-tach-giong.md).

## 1. Mục tiêu và ràng buộc

- Cần bộ tách **mù hai người nói, một kênh**, fine-tune trên vài chục giờ tiếng Việt
  hội thoại có nhiễu, để tách đoạn overlap trong podcast.
- Audio nguồn 24 kHz. Hầu hết checkpoint dùng được là 16 kHz, nên train ở 16 kHz;
  đầu ra mất dải 8–12 kHz (nếu cần 24 kHz phải thêm bước mở rộng băng thông).
- GPU 24–40 GB. Giấy phép phải cho phép dùng cả code lẫn trọng số.
- **Tiêu chí bắt buộc của một ứng viên:** có code chạy được, có checkpoint tải được,
  có đường fine-tune từ checkpoint đó, nêu rõ sample rate và định dạng dữ liệu train.

## 2. Các vòng

| Vòng | Việc | Mô hình | Điều kiện / lưu ý |
|---|---|---|---|
| 0 | Zero-shot trên ~100 clip overlap thật, không train | MossFormer2_SS_16K, TIGER-speech, SpeechBrain whamr16k (DialogueSidon HF Space chỉ để nghe) | Rẻ nhất, cho biết có cần fine-tune không. Đánh giá **không tham chiếu** vì clip không có ground truth |
| 1 | Fine-tune | **MossFormer2_SS_16K**, **TIGER-speech**, **SpeechBrain whamr16k** (đối chứng) | Chi tiết ở mục 4 |
| 2 | Chỉ khi vòng 1 kém trên clip thật có nhiễu | TF-Locoformer (WHAMR!), SEAL-small, TUSS (chỉ nếu chấp nhận AGPL-3.0) | TF-Locoformer: ESPnet tự thêm nhiễu/vang khi train nhưng môi trường kén và chất lượng ở 16 kHz chưa đo. SEAL: mới, rất nhỏ (0.59M) |
| 3 | Chỉ khi cần thêm | SPMamba, TISDiSS, TDANet | SPMamba: cần `mamba-ssm 1.x` biên dịch CUDA, trọng số không nạp được nguyên trạng. TISDiSS: checkpoint công khai chỉ là WSJ0 sạch. TDANet: batch 1 mỗi process |
| Chặn | Chưa dùng | SR-CorrNet-SS (trọng số **không có giấy phép**), SonicSim (code CC BY-SA, trọng số không giấy phép), DialogueSidon (trọng số CC-BY-NC, không có checkpoint huấn luyện) | Xem lại nếu giấy phép được làm rõ |

Vì sao không chạy cả 11 mô hình cùng lúc: mỗi mô hình cần manifest, bản vá và môi
trường riêng; vòng 0 tốn một buổi và cho biết nên đầu tư vào đâu.

**Tiêu chí chuyển vòng (đề xuất, chốt sau vòng 0):**
- Vào vòng 1 nếu zero-shot trên clip thật còn rò rỉ giọng hoặc méo rõ rệt.
- Vào vòng 2 nếu sau fine-tune, mô hình tốt nhất vẫn kém trên clip có nhiễu so với
  trên val synthetic.
- Dừng nếu fine-tune không cải thiện hơn zero-shot trên bộ chỉ số mục 6.

## 3. Sự thật về các mô hình vòng 1

| | MossFormer2_SS_16K | TIGER-speech | SpeechBrain sepformer-whamr16k |
|---|---|---|---|
| Sample rate | 16 kHz | 16 kHz (band split hỏng ở 8 kHz) | 16 kHz |
| Tham số | 55.7M | 0.82M (tiny 0.10M) | ~28M (suy ra) |
| Checkpoint | HF `alibabasglab/MossFormer2_SS_16K`, `last_best_checkpoint.pt` 670 MB, epoch 59 | HF `JusperLee/TIGER-speech` (+ `-small`, `-tiny`) | HF `speechbrain/sepformer-whamr16k`, kèm optimizer state |
| Dữ liệu train của checkpoint | "dữ liệu mở và riêng", **không công bố** | EchoSet ~34 giờ, tiếng Anh, có vang (maintainer, issue #32) | WSJ0 + WHAM! + vang tổng hợp, đích không vang |
| Đường fine-tune | **Có sẵn**: `train.sh` + `init_checkpoint_path`, lr `finetune_learning_rate` 5e-5; bỏ qua optimizer state | **Không có**; vá ~4 dòng: `from_pretrained` + `load_state_dict(strict=True)` (nạp strict đã thử trên CPU) | Thêm khối `pretrained_separator` vào yaml recipe (nhánh có sẵn trong `train.py`). Có tiền lệ fine-tune tiếng Quan Thoại: lr 1.5e-5, 10 epoch |
| Định dạng dữ liệu | Danh sách text 3 cột `mix s1 s2`, 16 kHz, một đoạn cắt ngẫu nhiên 2 giây mỗi file mỗi epoch | `mix.json`, `s1.json`, `s2.json` là danh sách `[đường dẫn tuyệt đối, số mẫu]`, đoạn 3 giây, **file ngắn hơn bị bỏ im lặng**; thư mục test = bản sao val | CSV `ID,duration,mix_wav,s1_wav,s2_wav,noise_wav`; đoạn 4–6 giây |
| Nhiễu / vang khi train | Không → render offline | Không → render offline (speed perturb phá nhiễu, tắt đi) | Có thể thêm RIR và nhiễu tại chỗ nếu tên thư mục chứa "whamr" và có file noise/RIR; nếu không → render offline |
| Loss với đích im lặng | PIT SI-SNR; đích toàn 0 → hằng số, gradient 0, không dạy xuất im lặng | SNR PIT: như vậy | PIT SI-SNR: như vậy |
| Giấy phép | Apache-2.0 (code và trọng số); dữ liệu train không rõ | MIT (file LICENSE) / Apache-2.0 (thẻ HF) | Apache-2.0; dữ liệu WSJ0 (LDC), WHAM! (CC BY-NC theo nguồn thứ cấp) |
| Việc phải vá | `solver.py:194-197` ghi đè val loss tốt nhất bằng giá trị tệ hơn (giảm lr và dừng sớm gần như không kích hoạt); val dùng đoạn cắt ngẫu nhiên; `torch.distributed.launch` → single GPU | `WANDB_MODE=offline`; `typeguard==2.13.3`; speechbrain 0.5.14 hoặc sửa dòng import; issue #30 (lỗi chỉ số decoder, maintainer hứa train lại) → **lưu bản sao trọng số hiện tại** | Chưa kiểm bản SpeechBrain hiện tại nạp được checkpoint 2021 không |

Cài đặt đề xuất để bắt đầu (chưa kiểm chứng bằng chạy thật):
- MossFormer2: lr 5e-5, đoạn 4 giây, 10–30 epoch, tập val cố định, batch 1 cộng gom gradient.
- TIGER: đoạn 3 giây, lr 1e-4 giảm dần về 3e-5, tắt bf16.
- SpeechBrain: lr 1.5e-5–5e-5, 10 epoch.

## 4. Hướng tạo dữ liệu cho vòng 1

### 4.1 Điểm xuất phát

`tools/make_mix_dataset.py` đã sinh mẫu từ cache: cặp đoạn sạch của hai người trong
cùng một audio, chồng một phần / chứa trong / trùng hoàn toàn, mức B so với A ngẫu nhiên
±10 dB, mixture đúng bằng `clean_A + clean_B`, 24 kHz, ~2–14 giây, tách train/val theo
recording. Còn thiếu so với yêu cầu của vòng 1: **chưa có nhiễu và vang, chưa resample
16 kHz, chưa cắt đoạn đúng độ dài, chưa có manifest theo định dạng từng mô hình**.

### 4.2 Hai quan sát định hình thiết kế

1. **Stem sạch lấy từ podcast thật không phải tiếng khô.** `clean_A` và `clean_B` đã
   mang phòng thu và micro của recording gốc. Cộng thêm vang tổng hợp sẽ chồng lên vang
   sẵn có. Vì vậy **nhiễu quan trọng hơn vang**; vang tổng hợp nên nhẹ và là tùy chọn
   (đây là suy luận của tôi, cần nghe thử để xác nhận).
2. **Mọi mô hình vòng 1 gặp vấn đề đích im lặng.** Stem của ta bằng 0 ngoài đoạn người
   đó nói. Sau khi cắt ngẫu nhiên, một cửa sổ có thể chứa stem toàn 0 → loss hằng số,
   gradient 0. Chốt chặn của ESPnet chỉ bắt đoạn *đúng bằng 0*; look2hear và SpeechBrain
   không có. Vì vậy **cắt đoạn trước theo quy tắc hoạt động là bắt buộc**.

### 4.3 Quy trình

1. **Kiểm kê kho đoạn sạch** (script mới): từ cache, với mỗi recording và speaker, đo
   tổng số giây sạch (không ai nói trong ±0.5 giây, không nhạc). Quyết định số giờ mixture
   khả thi: mỗi cặp đoạn sạch có thể trộn lại nhiều lần với offset, mức và nhiễu khác nhau,
   nhưng số giờ *nội dung sạch duy nhất* mới là giới hạn thật. Nếu quá ít thì tăng số
   recording (hiện chọn 30).
2. **Chia tập trước, theo recording**, rồi mới làm bất cứ gì khác. Giữ ~100 clip overlap
   thật làm tập test; recording của chúng bị loại khỏi train (`--exclude-index`).
3. **Tạo cặp và trộn ở 24 kHz** như `make_mix_dataset.py` (giữ ba kiểu chồng, mức lệch
   ±10 dB; bắt đầu hẹp hơn, ±5 dB, nếu mô hình nhỏ hội tụ khó).
4. **Resample `clean_A` và `clean_B` 24k→16k** (tỷ lệ 2:3, bộ lọc chất lượng cao), rồi
   mới tính lại `M = A + B` ở float32. Không resample mixture riêng. Kiểm
   `max|M − A − B| < 1e-6`. Cùng một hệ số gain cho cả ba; đỉnh dưới 0.95.
5. **Thêm nhiễu offline** (SNR 5–25 dB; kiểu podcast, không phải −6…3 dB của WHAMR!).
   Với mỗi mẫu lưu *một* hoặc nhiều biến thể: sạch; chỉ nhiễu; nhiễu + vang nhẹ. Sau bước
   này `mix ≠ s1 + s2`, nên lưu `mix`, `s1`, `s2` thành file riêng. Đích của biến thể có
   nhiễu là hai stem **không có nhiễu**.
6. **Cắt đoạn** (sau chia tập, sau resample): cửa sổ 4 giây cho MossFormer2 và
   SpeechBrain, 3 giây cho TIGER, dựng **quanh một overlap thật** (xem 4.7). Quy tắc hoạt động:
   mỗi stem phải có ít nhất **0.15 giây** tiếng nói (khung 10 ms trên −45 dBFS). Quy tắc cũ
   "≥ 40% khung hoạt động cho cả hai stem" **bị bỏ**: overlap thật chủ yếu là người thứ hai
   chen ngắn trong lượt dài, nên stem thứ hai chỉ chiếm ~10% cửa sổ và quy tắc cũ sẽ loại
   gần hết mẫu đúng kiểu thật. Vẫn loại stem im lặng toàn phần (loss hằng số, gradient 0).
7. **Ghi một bảng đoạn duy nhất** (jsonl/parquet) làm nguồn sự thật, rồi sinh manifest
   riêng cho từng mô hình từ đó:
   - MossFormer2: ba cột `mix s1 s2`, đường dẫn không có khoảng trắng; `tr_list`/`cv_list`.
   - TIGER: `mix.json`/`s1.json`/`s2.json` của `[đường dẫn tuyệt đối, số mẫu]`; test = bản sao val.
   - SpeechBrain: CSV với cột `noise_wav` (hoặc `use_speedperturb: false`).
8. **Kiểm định** (script mới): bất biến `M ≈ A + B` (khi chưa thêm nhiễu), không có NaN,
   tỷ lệ đoạn bị loại, phân bố độ dài / mức lệch / tỷ lệ overlap, không recording nào
   xuất hiện ở cả train lẫn val lẫn test. Nghe ngẫu nhiên ~20 mẫu.

### 4.4 Tham số đề xuất ban đầu (ĐÃ BỊ THAY bằng số đo thật ở 4.7; giữ lại để so)

| Tham số | Giá trị khởi điểm | Ghi chú |
|---|---|---|
| Kiểu chồng | partial 50% / contained 25% / full 25% | Checkpoint công khai thấy chủ yếu chồng hoàn toàn; dữ liệu thật của ta chủ yếu chồng một phần |
| Mức B so với A | ±5 dB, mở rộng dần tới ±10 dB | Chưa có bằng chứng giá trị tốt nhất |
| SNR nhiễu | 5–25 dB | Podcast ít nhiễu hơn WHAMR! |
| Vang tổng hợp | tùy chọn, RT60 0.2–0.8 s, mức nhẹ | Xem quan sát 1 ở mục 4.2 |
| Cửa sổ | 4 s (3 s cho TIGER) | Theo mặc định của từng trainer |
| Khối lượng | 10–30 giờ đoạn đã render | Không nguồn nào nêu mức tối thiểu; tham chiếu: EchoSet ~34 h, LRS2-2Mix ~11 h, Echo2Mix 57 h |

### 4.7 Số đo overlap thật và thiết kế dữ liệu theo đó (2026-10-20)

`tools/measure_real_overlap_stats.py` chạy trên 360 recording trong cache (351 giờ, 18.623
sự kiện overlap, 19.013 cặp; mỗi file tối đa 50 sự kiện khi gộp phân phối). Các khoảng
tin cậy 95% lấy mẫu lại theo recording.

| Số đo | Kết quả | Giả định ban đầu | Sửa |
|---|---|---|---|
| Độ dài sự kiện | trung vị **0.46 s** (CI 0.459–0.48), p10 0.2, p90 2.58 s | mỗi người 2–8 s, chồng dài | Overlap thật ngắn; lấy độ dài từ mẫu thật |
| Kiểu chồng | contained **61.2%**, partial **37.5%**, full **1.3%** | 25 / 50 / 25 | full gần như không có; đề xuất ~60 / 37 / 3 |
| Chênh mức (người sau − người trước) | trung vị −0.08 dB; \|chênh\| trung vị 1.75 dB; p90 \|chênh\| 4.8–5.6 dB | ±10 dB đều | Lấy mẫu từ phân phối đo; đuôi ±6–12 dB chỉ là ca khó |
| Speech-to-noise mỗi recording | trung vị **34 dB** (p10 27, p90 42) | 5–25 dB | Dữ liệu thật rất sạch; nhiễu chỉ nhẹ |
| Nhiễu nền | trung vị −57 dBFS (p10 −65, p90 −49) | | |
| Mật độ | 0.885 sự kiện/phút; overlap = 1.54% audio | | Hiếm |
| Overlap lân cận (±30 s) | trung vị 1, p90 4 | | Một cửa sổ 4 s thường chỉ có **một** overlap |

Phát hiện chính: overlap thật là **người thứ hai chen một tiếng ngắn vào lượt dài của người
thứ nhất**. Nhiễu là việc phụ; khó khăn thật là phát hiện và tách một giọng ngắn, thường cùng
mức, nằm trong giọng dài.

Hệ quả cho `tools/render_mix_dataset.py` (đã viết, đã test trên dữ liệu giả, **chưa chạy
trên cache thật**):
- Mỗi mẫu bắt đầu từ một **mẫu overlap thật** trong `pairs.csv`: độ dài lượt người đầu, độ
  dài lượt người sau, độ dài overlap, kiểu chồng, chênh mức. Mẫu được hiện thực hóa bằng
  đoạn sạch của cùng recording, đặt trên một trục thời gian, cắt thành cửa sổ chứa overlap.
- Nguồn mẫu: 70% từ chính recording đó, 20% từ cả split nhưng chia đều theo nhóm độ dài
  (để overlap dài 2–5 s và >5 s không bị bỏ đói), 10% ca khó (overlap ≥ 1 s, chênh mức gần
  0 hoặc rất rộng). Chỉnh bằng `--mix`.
- Nhiễu: nền của chính recording (các đoạn không có segment) cộng vào **mixture**, xác suất
  `--noise-prob` (mặc định 0.5), SNR phần lớn 25–45 dB, 20% thấp hơn (10–25 dB). Bản có
  nhiễu có `mix ≠ s1 + s2`.
- Không thêm vang tổng hợp (stem đã mang vang của recording).
- Đoạn sạch cho người chen ngắn lấy từ chính các đoạn ngắn thật không bị chồng (ngưỡng tối
  thiểu 0.25 s), cắt ở chỗ năng lượng thấp để không cắt giữa từ.
- Chia tập theo recording; tập ~100 clip thật bị loại bằng `--exclude-index`.
- Manifest cho ba mô hình vòng 1 sinh từ cùng một bảng `chunks.jsonl`. Định dạng lấy từ hồ
  sơ khảo sát, **chưa kiểm bằng cách chạy từng trainer**.
- Chạy hai lần: `--window 4` (MossFormer2, SpeechBrain) và `--window 3` (TIGER).

`tools/validate_mix_dataset.py` kiểm các bất biến (file, sample rate, độ dài, NaN/clip,
`mix = s1 + s2` cho chunk không nhiễu, stem có tiếng, không recording nào ở cả hai split,
không dùng recording test) và in phân phối đã render cạnh phân phối thật.

**Giới hạn của số đo**
- Chênh mức là số gần đúng: lấy từ mức mỗi người khi nói một mình gần nhất, không phải mức
  tức thời trong overlap; tiếng chen vào thường nhỏ hơn nên chênh thật có thể lớn hơn.
- Overlap chỉ lấy từ ranh giới segment của DiariZen. Overlap chiếm 1.54% audio, dưới ngưỡng
  3% mà `stage_output_service.py` dùng để cảnh báo "diarizer có thể bỏ sót backchannel", nên
  đây có thể là cận dưới của overlap thật.
- Nhiễu nền đo trên waveform đã cắt nhạc, ở các đoạn không có segment; có thể lẫn tiếng
  cười hoặc nhạc sót, và thấp hơn thực tế nếu pipeline có khử nhiễu ở bước trước.
- 360 trong 364 recording được đo.

### 4.5 Việc suy luận không thuộc dữ liệu train nhưng ảnh hưởng thiết kế

Pipeline chỉ cần tách **đoạn overlap**, nên đầu vào lúc suy luận hầu như luôn có hai
người. Điều này giảm áp lực phải dạy mô hình xử lý đoạn một người. Nhưng cả ba mô hình
vòng 1 không theo dõi hoán vị giữa các cửa sổ (MossFormer2 giải mã từng cửa sổ 2 giây,
bước 75%), nên cần viết bộ căn hoán vị theo tương quan ở vùng chồng cho đoạn overlap dài.

### 4.6 Script cần viết (chưa có)

- `tools/measure_real_overlap_stats.py`: **đã chạy** (mục 4.7).
- `tools/render_mix_dataset.py`: các bước 3–7. **Đã viết, chưa chạy trên cache thật.**
- `tools/validate_mix_dataset.py`: bước 8. **Đã viết, chưa chạy trên cache thật.**
- `tools/audit_clean_pool.py` (kiểm kê kho đoạn sạch, bước 1): chưa viết; `render_mix_dataset.py`
  hiện tự dựng kho đoạn sạch từng recording.

## 5. Các quyết định còn mở (cần bạn chốt)

1. **Nguồn nhiễu và giấy phép:** WHAM! (CC BY-NC theo nguồn thứ cấp), DEMAND, MUSAN,
   FSD50K, hay nhiễu tự ghi? Nếu trọng số sẽ dùng thương mại thì nên ưu tiên nhiễu có
   giấy phép rõ.
2. **Vang tổng hợp:** có hay không, và đích là stem như đã thu hay stem khử vang.
3. **Số recording nguồn** (hiện 30) và **số giờ mục tiêu** sau khi có kết quả kiểm kê.
4. **Có thêm bước mở rộng băng thông** để có stem 24 kHz sau khi tách không.

## 6. Đánh giá

- **Val synthetic** (có ground truth): SI-SDRi, chia theo kiểu chồng, mức lệch và độ
  dài overlap; không so điểm giữa các benchmark khác nhau.
- **Clip thật** (không có ground truth, ~100 clip): DNSMOS hoặc NISQA, WER/CER của ASR
  trên từng đầu ra, rò rỉ giọng (VAD chéo giữa hai đầu ra), nghe ngẫu nhiên. Không dùng
  SI-SDR với mixture làm tham chiếu.
- Mọi run dùng cùng val cố định và cùng quy trình trên clip thật.

## 7. Chưa xác nhận

1. **Chất lượng trên hội thoại tiếng Việt**: chưa ai thử; chỉ có một chương Springer 2026
   về MossFormer2 cho hỗn hợp Việt–Anh, chưa đọc được nội dung.
2. MossFormer2: chưa nạp `strict` file 670 MB vào class của repo, chưa chạy một bước train
   trên PyTorch hiện tại, chưa đo bộ nhớ GPU ở đoạn 4 giây.
3. TIGER: chưa chạy trainer đầu-cuối trên môi trường hiện đại; trọng số có thể lỗi thời
   nếu bản train lại (issue #30) được phát hành.
4. Giấy phép dữ liệu train (WSJ0 của LDC, WHAM!, LRS2) có thể ràng buộc trọng số: chưa phân tích.
5. Không mô hình nào ở đây được kiểm tra trên dữ liệu ngôn ngữ có thanh điệu.
6. Mọi tham số ở mục 4.4 là điểm khởi đầu, chưa có bằng chứng.
