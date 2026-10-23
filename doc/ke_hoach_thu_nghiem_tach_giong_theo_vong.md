# Kế hoạch thử nghiệm bộ tách giọng theo vòng (fine-tune trên tiếng Việt)

Ngày 2026-10-20, cập nhật 2026-10-23 (gộp nhánh TSE, rút gọn mục 4). **Chưa thực thi gì trong tài liệu này**: đây là kế hoạch và các
quyết định còn mở. Số liệu về mô hình lấy từ đợt khảo sát ngày 2026-10-19 (đọc code,
README, header checkpoint; **chưa chạy thật mô hình nào**). Mục "Chưa xác nhận" ở cuối
liệt kê những gì nên coi là giả định.

Dữ liệu huấn luyện (BSS và TSE): [cach_tao_du_lieu_tach_overlap.md](cach_tao_du_lieu_tach_overlap.md), bản đặc tả duy nhất.
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

Nhánh TSE chạy song song, với vòng 0 riêng (mục 4b) và cùng giao thức đánh giá (mục 6).

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
Độ dài đoạn train là **W theo từng mô hình** (4 s tiêu biểu; xem mục 7 của đặc tả dữ liệu). Phải đặt đúng W trong từng trainer, thay cho mặc định của trainer.
- MossFormer2: lr 5e-5, đoạn W (thay cho đoạn cắt ngẫu nhiên 2 s), giải mã cả cửa sổ W trong một lượt (không chia 2 s), 10–30 epoch, tập val cố định, batch 1 cộng gom gradient.
- TIGER: đoạn W (mặc định trainer 3 s), lr 1e-4 giảm dần về 3e-5, tắt bf16.
- SpeechBrain: đoạn W, lr 1.5e-5–5e-5, 10 epoch.

## 4. Dữ liệu cho vòng 1 (tóm tắt)

Đặc tả đầy đủ và duy nhất nằm ở [cach_tao_du_lieu_tach_overlap.md](cach_tao_du_lieu_tach_overlap.md). Các ý chính:

- Mỗi mẫu dựng từ **một overlap thật** trong `pairs.csv` (360 recording, 351 h, 18.623 sự kiện, trung vị 0.46 s, contained 61% / partial 38% / full 1%). Lõi được tổng hợp từ đoạn sạch của chính recording đó, nên `mix = s1 + s2 (+ nhiễu)` và đích chính xác.
- BSS: mẫu dài **đúng W giây, W theo từng mô hình** (4 s chỉ là giá trị tiêu biểu), mỗi W một bộ render. Thứ tự ưu tiên: lõi → biên 0.25 s → solo → ctx dư. Bố cục dùng chung với `WindowPlanner` qua `utils/separation_layout.py` (train = suy luận).
- TSE: mix không có solo, dài `W_tse`; enrollment ≤ 4 s lấy ngoài mix; 2 mẫu mỗi mix.
- Nhiễu nền của chính recording, xác suất 0.5, SNR chủ yếu 25–45 dB. Không thêm vang tổng hợp.
- Chia tập theo recording, val phân tầng; ~100 clip thật làm test, recording của chúng bị loại khỏi train.
- Chọn mô hình theo SI-SDRi **vùng lõi** và các chỉ số trên clip thật (mục 6).

Trạng thái: `measure_real_overlap_stats.py` đã chạy; `render_mix_dataset.py` và `validate_mix_dataset.py` đã viết nhưng **chưa chạy trên cache thật** và đang được sửa theo đặc tả; `audit_clean_pool.py` chưa viết. Lịch sử: `tools/make_mix_dataset.py` (24 kHz, ~2–14 s, mức ±10 dB) là bộ dựng đầu tiên. Các tham số ban đầu (25/50/25 kiểu chồng, SNR 5–25 dB, 30 recording) đã được thay bằng số đo thật.

Trong pipeline, suy luận chỉ chạy trên **đoạn overlap**, nên đầu vào hầu như luôn có hai người. Ba mô hình vòng 1 không theo dõi hoán vị giữa các cửa sổ, nên overlap dài hơn W (bị chia thành nhiều cửa sổ) cần bộ căn hoán vị theo tương quan ở vùng chồng.

## 4b. Nhánh TSE (tách theo người mục tiêu)

TSE dùng chung dữ liệu (mục 4) và giao thức đánh giá (mục 6) với BSS. Đầu vào là mix cộng một đoạn tham chiếu giọng (enrollment) của người cần lấy; đầu ra là một nguồn. Nhờ vậy không có vấn đề hoán vị giữa các cửa sổ, nhưng cần enrollment sạch.

Nhãn **[tóm tắt]**: lấy từ bản tóm tắt khảo sát TSE ngày 2026-10-22; đợt khảo sát đó chưa nằm trong repo và chưa được kiểm lại.

- **Ứng viên** [tóm tắt]: PS4, WeSep `bsrnn_ecapa_vox1`, REAL-TSE, TD-SpeakerBeam, USEF-TFGridNet, AD-FlowTSE.
- **Vòng 0 TSE**: chạy zero-shot cả 6 mô hình trên ~100 clip thật và trên synthetic (không train nên rẻ). Bước đầu tiên: tải thử ba checkpoint trên DGX (USEF-TFGridNet, WeSep `bsrnn_ecapa_vox1`, PS4), chỉ lấy header để kiểm.
- **Quyết định đề xuất**:
  - TSE dạng mặt nạ hợp chính sách audio-cleanliness, vì mặt nạ chỉ lọc tín hiệu có sẵn. Mô hình sinh (AD-FlowTSE) chỉ dùng để *đo*, không dùng đầu ra làm dữ liệu.
  - PS4 có huy hiệu Apache-2.0 nhưng không có file LICENSE: chấp nhận cho thử nghiệm nội bộ; không dùng trọng số cho đầu ra phát hành cho đến khi có LICENSE rõ.
- **Việc phía trainer**:
  - chọn trainer TSE để chốt schema manifest;
  - dataloader bốc lại enrollment trong 8 ứng viên mỗi epoch (WeSep có sẵn kiểu này);
  - độ dài đoạn train = `W_tse`.
- **Đánh giá thêm**: kết quả tách theo vai người mục tiêu (giữ sàn / chen).
- **Chưa xác nhận**: mọi thông tin TSE (mô hình, giấy phép, định dạng) dựa vào bản tóm tắt, chưa tải hay chạy. Chưa có bằng chứng chất lượng trên tiếng Việt.

## 5. Các quyết định còn mở (cần bạn chốt)

1. **Nguồn nhiễu và giấy phép:** WHAM! (CC BY-NC theo nguồn thứ cấp), DEMAND, MUSAN,
   FSD50K, hay nhiễu tự ghi? Nếu trọng số sẽ dùng thương mại thì nên ưu tiên nhiễu có
   giấy phép rõ.
2. **Vang tổng hợp:** có hay không, và đích là stem như đã thu hay stem khử vang.
3. **Số giờ render mục tiêu** sau khi có kết quả `audit_clean_pool.py` (nguồn hiện là 360 recording đã đo).
4. **W cuối cho từng mô hình** (BSS và `W_tse`).

Câu hỏi mở về dữ liệu nằm ở mục 12 của đặc tả dữ liệu.
5. **Có thêm bước mở rộng băng thông** để có stem 24 kHz sau khi tách không.

## 6. Đánh giá

- **Val synthetic** (có ground truth): SI-SDRi **vùng lõi** (đọc `layout`, lõi ± 0.1 s) là chỉ số chính, kèm
  SI-SDRi cả cửa sổ; chia theo kiểu chồng, mức lệch và độ dài overlap. SI-SDRi cả cửa sổ
  chủ yếu đo phần solo/ctx và tăng theo W, nên không dùng nó để so các mô hình có W khác nhau.
  Val của mọi W dựng từ cùng các lần bốc template. Không so điểm giữa các benchmark khác nhau.
- **TSE**: thêm SI-SDRi theo vai người mục tiêu (giữ sàn / chen).
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
6. Mọi tham số dữ liệu (W, ngưỡng, tỷ lệ nguồn) là điểm khởi đầu, chưa có bằng chứng từ train thật.
