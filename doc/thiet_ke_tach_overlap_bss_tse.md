# Thiết kế tách overlap ngắn: BSS và TSE

Ngày 2026-10-22. **Chưa thực thi gì trong tài liệu này.** Ba nguồn, đánh dấu riêng trong từng mục:

- **[repo]** đã có trong nhánh `training-separation` (số đo `doc/data/overlap_stats_recordings.csv`, `tools/*`, [ke_hoach_thu_nghiem_tach_giong_theo_vong.md](ke_hoach_thu_nghiem_tach_giong_theo_vong.md)).
- **[tóm tắt]** lấy từ bản tóm tắt thiết kế BSS/TSE ngày 2026-10-22 (đợt khảo sát TSE chưa nằm trong repo; chưa kiểm lại).
- **[đề xuất]** suy luận mới, chưa có số đo.

## 1. Hai họ mô hình

Overlap thật ngắn (trung vị **0.46 s**) [repo]. Hai cách tách:

| | BSS (tách mù) | TSE (tách theo người mục tiêu) |
|---|---|---|
| Đầu vào | mixture | mixture + đoạn tham chiếu giọng (enrollment) của người cần lấy |
| Đầu ra | hai nguồn, hoán vị tùy ý | một nguồn: giọng người mục tiêu |
| Trạng thái | kế hoạch vòng 1 (MossFormer2, TIGER, SpeechBrain) [repo] | vừa khảo sát: PS4, WeSep `bsrnn_ecapa_vox1`, REAL-TSE, TD-SpeakerBeam, USEF-TFGridNet, AD-FlowTSE [tóm tắt] |
| Điểm yếu | không biết hai giọng nào là ai; phải căn hoán vị giữa cửa sổ | cần enrollment sạch; chưa kiểm trên tiếng Việt |

Hai họ dùng chung dữ liệu dựng từ cache và chung giao thức đánh giá (mục 5); quyết định
bằng số đo vòng 0 trên ~100 clip thật.

## 2. Dữ liệu overlap được chọn thế nào, bằng công cụ gì [repo]

1. `tools/measure_real_overlap_stats.py` đo trên cache: **360 recording, 351 giờ, 18.623
   sự kiện overlap**, ghi `pairs.csv` (mỗi dòng là một mẫu overlap thật: độ dài lượt người
   giữ sàn, độ dài lượt người chen, độ dài overlap, kiểu chồng, chênh mức).
2. `tools/collect_overlap_samples.py` chọn ~100 clip overlap thật làm **tập test**, trải
   đều ngắn / vừa / dài và giữa các recording. Recording của chúng bị loại khỏi train
   (`--exclude-index`).
3. `tools/render_mix_dataset.py` dựng mixture: mỗi mẫu bắt đầu từ **một mẫu overlap thật**
   trong `pairs.csv`, hiện thực hóa bằng đoạn sạch của cùng recording (không ai nói trong
   ±0.2 s, không nhạc), đặt trên một trục thời gian rồi cắt cửa sổ `--window` giây chứa
   overlap. Mỗi stem phải có ít nhất 0.15 s tiếng nói.
4. `tools/validate_mix_dataset.py` kiểm bất biến và in phân phối đã dựng cạnh phân phối thật.

Nguồn mẫu mỗi chunk (`--mix`, mặc định 0.7:0.2:0.1): 70% từ chính recording đó; 20% từ cả
split, chia đều theo nhóm độ dài overlap (0–0.5, 0.5–1, 1–2, 2–5, >5 s) để overlap dài không
bị bỏ đói; 10% ca khó (overlap ≥ 1 s, chênh mức gần 0 hoặc rất rộng).

**Số lượng:** kế hoạch cũ nêu 10–30 giờ đoạn đã render và ví dụ `--n 20000`. Con số cuối
chưa chốt; giới hạn thật là số giờ *nội dung sạch duy nhất* (script kiểm kê
`audit_clean_pool.py` chưa viết). **Chưa chạy trên cache thật.**

## 3. Đoạn overlap và đoạn solo: có bằng nhau không?

**Không.** Độ dài lấy từ phân phối đo, không đặt tay [repo, mục 4.7]:

- Kiểu chồng: contained 61.2%, partial 37.5%, full 1.3%.
- Overlap: trung vị 0.46 s, p10 0.2 s, p90 2.58 s.
- Chênh mức giữa hai người: |chênh| trung vị 1.75 dB, p90 4.8–5.6 dB.
- Một cửa sổ 4 s thường chỉ chứa **một** overlap; người thứ hai chen ngắn vào lượt dài của
  người thứ nhất, nên stem thứ hai chỉ chiếm khoảng 10% cửa sổ.

Nên trong một chunk, solo của người giữ sàn dài, solo của người chen gần như không có. Đây
là chủ ý: quy tắc cũ "≥ 40% khung hoạt động cho cả hai stem" đã bị bỏ vì sẽ loại gần hết
mẫu đúng kiểu thật.

## 4. Một bảng chung, hai bộ riêng

Bảng nguồn sự thật là `chunks.jsonl`. Từ đó sinh manifest riêng cho từng mô hình
(MossFormer2, TIGER, SpeechBrain cho BSS) [repo]. Với TSE, **dựng thêm từ cùng chunk**
[đề xuất]:

| | BSS | TSE |
|---|---|---|
| Mẫu | `(mix, s1, s2)` | `(mix, enrollment, target)`, mỗi chunk BSS cho **hai** mẫu TSE (target = s1, rồi s2) |
| Enrollment | không cần | đoạn sạch của người mục tiêu **lấy từ nơi khác trong cùng recording**, nằm ngoài cửa sổ, không ai khác nói trong ±0.5 s; độ dài vài giây. `collect_overlap_samples.py` đã dựng `speakers/<SPK>/reference.wav` cùng kiểu |
| Chia tập | theo recording | theo recording, cùng split với BSS (không để enrollment và cửa sổ khác split) |
| Loss | PIT SI-SNR | SI-SNR trực tiếp, không PIT |

**Có cần cân bằng bằng cách thêm dữ liệu sạch không?** Hai ý khác nhau:

- *BSS:* dữ liệu thật mất cân bằng (một giọng chiếm gần cả cửa sổ). Thêm cửa sổ **solo hoàn
  toàn** làm mô hình quen xuất im lặng cho nguồn thứ hai; nhưng loss PIT với đích im lặng cho
  gradient 0 [repo, mục 4.2], nên chúng không dạy gì thêm về việc *nhận ra hai người*. Giá trị
  còn lại là cân bằng nhiễu. Đề xuất: không thêm solo thuần; thêm cửa sổ **có solo ≥ 1 s mỗi
  người** (khớp `WindowPlanner` ở inference) để mô hình thấy dạng đầu vào sẽ gặp.
- *TSE:* mất cân bằng ít quan trọng hơn chất lượng enrollment. Nên có mẫu mà người mục tiêu
  **không** nói trong overlap (đầu ra phải là im lặng) để mô hình không bịa tiếng.

## 5. Đánh giá chung

Giữ nguyên giao thức [repo, mục 6]: val synthetic có ground truth (SI-SDRi theo kiểu chồng,
mức lệch, độ dài); clip thật không ground truth (DNSMOS/NISQA, WER/CER, rò rỉ giọng bằng
VAD chéo, nghe ngẫu nhiên). Không dùng SI-SDR với mixture làm tham chiếu.

## 6. Bước tiếp theo

1. Tải thử ba checkpoint trên DGX, chỉ lấy tiêu đề để kiểm: USEF-TFGridNet, WeSep
   `bsrnn_ecapa_vox1`, PS4 [tóm tắt].
2. Dựng mẫu thử nhỏ (`--n 200 --files 5`) để kiểm định dạng `mix, s1, s2`; thêm enrollment cho TSE.
3. Viết bộ dựng cửa sổ có đoạn solo (theo `WindowPlanner`) và điểm cắt âm học.
4. Vòng 0: zero-shot trên ~100 clip thật và synthetic.
5. Quyết định fine-tune vòng 1 theo số đo vòng 0.

## 7. Các câu hỏi cần chốt, kèm đề xuất

| # | Câu hỏi | Đề xuất [đề xuất] |
|---|---|---|
| 1 | TSE dạng mặt nạ có hợp chính sách audio-cleanliness (loại mô hình sinh âm)? | Có: mặt nạ chỉ lọc tín hiệu có sẵn. Mô hình sinh (AD-FlowTSE) chỉ dùng để *đo*, không dùng đầu ra làm dữ liệu |
| 2 | Giấy phép PS4 (huy hiệu Apache-2.0, không có file LICENSE)? | Chấp nhận cho thử nghiệm nội bộ; chưa dùng trọng số cho đầu ra phát hành cho đến khi có LICENSE rõ |
| 3 | Vòng 0 chạy mô hình nào? | Cả 6 (PS4, WeSep, REAL-TSE, TD-SpeakerBeam, USEF, AD-FlowTSE); vòng 0 không train nên rẻ |
| 4 | Cửa sổ huấn luyện: 4 s có solo, hay 8–15 s như inference? | 4 s trước: khớp mặc định MossFormer2/SpeechBrain (TIGER 3 s) và một cửa sổ thường chỉ có một overlap. Chỉ thử 8 s nếu vòng 0 cho thấy hoán vị bị lệch giữa cửa sổ |
| 5 | Độ dài tối thiểu đoạn chen: 0.25 s hay 0.4 s? | 0.25 s (đang là mặc định trong code). Ước thô từ p10 0.2 s và trung vị 0.46 s: 0.4 s sẽ bỏ khoảng 40% overlap thật, 0.25 s chỉ bỏ hơn 10% |

## 8. Chưa xác nhận

- Mọi mục TSE (mô hình, license, định dạng dữ liệu) dựa vào bản tóm tắt; chưa tải hay chạy.
- `render_mix_dataset.py` và `validate_mix_dataset.py` chưa chạy trên cache thật.
- Số đo overlap lấy từ ranh giới segment của DiariZen, có thể là cận dưới của overlap thật.
- Chưa có bằng chứng về chất lượng trên hội thoại tiếng Việt cho bất kỳ mô hình nào ở đây.
