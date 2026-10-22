# Cách tạo dữ liệu huấn luyện tách overlap (BSS và TSE)

Ngày 2026-10-22. Nhánh `training-separation`. **Đây là thiết kế; code chưa sửa theo thiết kế này** (xem mục 11). Nguồn: doc thiết kế chung trong chat, số đo `doc/data/overlap_stats_recordings.csv`, `tools/measure_real_overlap_stats.py`, `tools/render_mix_dataset.py`.

Quy ước nhãn: **[đã có]** là điều code/số đo hiện tại đã làm; **[đề xuất]** là thiết kế mới, chưa chạy; **[mở]** là câu hỏi chưa chốt.

## 1. Mục tiêu và nguyên tắc

- Huấn luyện mô hình tách hai người nói chồng nhau trên hội thoại tiếng Việt. Hai họ mô hình dùng chung một nguồn dữ liệu:
  - **BSS** (tách mù): `(mix, s1, s2)`, loss PIT SI-SNR.
  - **TSE** (tách theo người mục tiêu): `(mix, enrollment, target)`, loss SI-SNR trực tiếp, không PIT.
- **Train và suy luận phải giống nhau.** Mẫu train có hình dạng giống đầu vào lúc suy luận. Hệ quả: mẫu BSS dài đúng 4 s, dựng theo thứ tự cố định, và bộ chia cửa sổ lúc suy luận (`WindowPlanner`, hiện nhắm 15 s) phải ép ra mẫu 4 s theo cùng thứ tự [đề xuất].
- Ground truth tự dựng: mix = tổng các stem sạch (cộng nhiễu nền), nên mỗi stem là đáp án chính xác. Không dùng đầu ra của mô hình sinh làm dữ liệu.

## 2. Luồng dữ liệu

```
cache diarization (360 recording, 351 h)
   │  measure_real_overlap_stats.py
   ▼
pairs.csv  ── mỗi dòng = một overlap thật (độ dài lượt, độ dài overlap, kiểu chồng, chênh mức)
   │  collect_overlap_samples.py: ~100 clip thật làm tập test, loại recording của chúng khỏi train
   ▼
render_mix_dataset.py  ── mỗi mẫu bắt đầu từ MỘT dòng pairs.csv,
   │                        dựng lại bằng đoạn sạch của cùng recording (hoặc recording khác)
   ▼
chunks.jsonl  (bảng nguồn sự thật)
   ├── manifest BSS: MossFormer2 / SpeechBrain / TIGER
   └── manifest TSE: (mix, enrollment, target), 2 mẫu mỗi mix
   │  validate_mix_dataset.py: kiểm bất biến, so phân phối dựng với phân phối thật
```

Nguồn mẫu mỗi chunk (`--mix`, mặc định 0.7:0.2:0.1) [đã có]:
- 70% từ chính recording đó;
- 20% từ cả split, chia đều theo nhóm độ dài overlap (0–0.5, 0.5–1, 1–2, 2–5, >5 s) để overlap dài không bị bỏ đói;
- 10% ca khó (overlap ≥ 1 s, chênh mức gần 0 hoặc rất rộng).

Chia train/val/test theo recording; hai mẫu TSE cùng mix, và enrollment cùng mẫu, luôn cùng split.

## 3. Định nghĩa

| Thuật ngữ | Định nghĩa | Tham số |
|---|---|---|
| Overlap | hai segment của hai người khác nhau chồng nhau | ≥ 0.1 s (`--min-overlap`) |
| Sự kiện overlap | các khoảng chồng cách nhau ít hơn 1.0 s gộp lại | `--merge-gap 1.0` |
| Đoạn sạch | segment của một người, không ai khác nói trong ±0.3 s, không nhạc | `--margin 0.3`, dài ≥ 0.25 s (`min_clip`, chưa có cờ CLI) |
| Stem | tín hiệu sạch của một người trong mẫu | cần ≥ 0.15 s tiếng nói (`MIN_ACTIVE`) |
| Lõi overlap (core) | phần cả hai cùng nói, dài L | L trong 0.3–4 s |
| Người giữ sàn (A) | người có lượt dài hơn, chứa hoặc mở đầu overlap | |
| Người chen (B) | người còn lại | |
| Ngữ cảnh (ctx) | tín hiệu liền mạch của chính hai người quanh lõi (lượt nói liền, không ghép) | |
| Solo | đoạn sạch chỉ một người nói, ghép từ nơi khác | |

Kiểu chồng (cung cấp bởi `classify` trong `measure_real_overlap_stats.py`):
- **full**: đầu và cuối hai lượt lệch ≤ 0.25 s;
- **contained**: người sau kết thúc trước người trước (người chen nằm trong lượt của người giữ sàn);
- **partial**: còn lại (người chen bắt đầu trong lượt A và kéo dài sang sau khi A dừng).

## 4. Thống kê overlap thật (làm cơ sở thiết kế)

18.623 sự kiện, 19.013 cặp, trên 351 giờ.

| Đại lượng | Giá trị |
|---|---|
| Độ dài overlap | trung vị 0.46 s, p10 0.2 s, p90 2.58 s |
| Kiểu chồng | contained 61.2%, partial 37.5%, full 1.3% |
| Mật độ | 0.885 sự kiện/phút; overlap chiếm 1.54% thời lượng |
| Chênh mức giữa hai người | \|chênh\| trung vị 1.75 dB, p90 4.8–5.6 dB |

Hệ quả: một cửa sổ 4 s thường chỉ chứa **một** overlap; stem người chen chỉ chiếm khoảng 10% cửa sổ. Quy tắc cũ "≥ 40% khung hoạt động cho cả hai stem" đã bỏ vì sẽ loại gần hết mẫu đúng kiểu thật. Số đo lấy từ ranh giới segment của DiariZen, có thể là cận dưới của overlap thật.

## 5. Mẫu BSS: đúng 4 s, dựng theo thứ tự ưu tiên [đề xuất]

Thứ tự ưu tiên khi dựng, mỗi bước chỉ dùng phần 4 s còn dư:

1. **Lõi overlap** (L giây), luôn được giữ nguyên, kể cả L = 4 s.
2. **Ngữ cảnh biên**: 0.25 s mỗi bên lõi, cho contained và partial (để overlap nằm hẳn bên trong, không dính mép).
3. **Solo sạch**: phần dư còn lại là `R' = 4 − L − 0.5`.
   - `R' ≥ 2`: mỗi người 1 s solo;
   - `0.5 ≤ R' < 2`: chia đều `R'/2` mỗi người;
   - `R' < 0.5`: không solo.
4. **Ngữ cảnh còn lại**: chỗ còn lại chia nửa mỗi bên.

Thứ tự thời gian trong mẫu: `[solo A][ctx trái][LÕI][ctx phải][solo B]`. Nếu `4 − L < 0.5` thì không giữ được biên 0.25 s, ngữ cảnh chia nửa mỗi bên phần còn lại.

```
L = 1 s (contained):
 A  ░░solo░░|▒ctx▒|██ LÕI ██|▒ctx▒|
 B                |██ LÕI ██|              ░░solo░░
            0.25      1.0      0.25
```

### 5.1 Sáu trường hợp lớn

| # | Trường hợp | Dựng thế nào | Dùng khi train? |
|---|---|---|---|
| 1 | **Contained** | lõi = toàn bộ lượt của B nằm trong A; ctx hai bên đều thuộc lượt A liền (lane A); solo theo công thức trên | có (L ≤ ~3.5 s để còn biên) |
| 2 | **Partial** | ctx trái thuộc A, ctx phải thuộc B; solo theo công thức | có |
| 3 | **Full** | hai lượt trùng đầu và cuối; mẫu gần như chỉ có lõi. Full dài hầu như không tồn tại trong dữ liệu thật (đầu/cuối lệch ≤ 0.25 s) | có, hiếm (1.3%) |
| 4 | **Nhiều overlap** (2–5) | lõi = khối từ đầu overlap thứ nhất đến cuối overlap cuối cùng (kí hiệu Lb); khoảng giữa các overlap là tiếng liền mạch, giữ nguyên; có thể xen kẽ người giữ sàn | có nếu Lb ≤ 4 s |
| 5 | **Thiếu đoạn sạch** (không dựng được lõi hoặc solo cần thiết) | thiếu đoạn sạch cho solo → bỏ phần solo; thiếu cho lõi → bỏ mẫu | loại khỏi train |
| 6 | **Ngoài phạm vi** (L < 0.3 s, hoặc L / Lb > 4 s) | không dựng mẫu | loại khỏi train |

Với trường hợp 5 và 6: **suy luận vẫn cố tách như bình thường**; chỉ dữ liệu train bỏ chúng.

### 5.2 Nguồn tín hiệu để dựng lõi

- Lõi tổng hợp từ hai đoạn sạch: người giữ sàn là đoạn sạch dài, người chen là đoạn sạch ngắn, độ dài theo lượt thật của dòng `pairs.csv` [đã có: `pull()` bốc đoạn sạch cho từng mẫu].
- Mỗi mẫu bốc **đoạn sạch khác nhau**, không lặp một đoạn cho cả file (cả lõi lẫn solo).
- Điểm cắt đoạn sạch: `AcousticBoundaryFinder` (VAD, chỗ lặng) giống lúc suy luận, thay cho `snap` ±80 ms hiện tại [đề xuất]. Cắt giữa từ làm mô hình thấy đầu/cuối cụt mà lúc suy luận không có.
- Ghép: fade 10–20 ms tại mối nối [đã có 10 ms].
- Mix = tổng stem + nhiễu nền (xác suất 0.5) [đã có].

### 5.3 Câu hỏi thường gặp

- **Đoạn cắt có cần đủ chữ không?** Không cần cho mô hình tách: loss tính trên tín hiệu, chữ không vào loss. Nhưng ngữ cảnh liên tục và điểm cắt ở chỗ lặng vẫn cần để giống lúc suy luận.
- **Có bắt buộc 1 s sạch mỗi người không?** Không. Solo chỉ có khi 4 s còn dư (xem bước 3); L = 3 s chỉ còn 0.5 s mỗi người, L = 4 s không có.
- **Có cần cân bằng bằng cửa sổ solo thuần không?** Không [đề xuất]: PIT với đích im lặng cho gradient 0, không dạy thêm việc nhận ra hai người. Giá trị còn lại là giống dạng đầu vào lúc suy luận, nên dùng cửa sổ có solo như trên.
- **Lấy L dài bao nhiêu?** 0.3–4 s, chia ba nhóm 0.3–0.5, 0.5–1, 1–4 s, lấy đều mỗi nhóm (tương ứng nhóm đã có trong `--mix` 20%). Bỏ L < 0.3 s (quá ngắn để học) và L > 4 s (không vừa mẫu).

## 6. Mẫu TSE: không dựng cửa sổ, thêm một đoạn sạch ngoài [đề xuất]

Một mẫu = `(mix, enrollment, target)`.

```
 Enrollment A (~3 s, sạch, ngoài mix) ─┐
                                        ├─► mô hình TSE ─► target A
 Mix (3–4 s, lõi overlap nguyên vẹn)  ─┘                    (tiếng sạch của A trong mix)
```

- **Mix**: dài 3–4 s tùy mẫu, **không ghép solo, không chừa ngữ cảnh biên**. Nó vẫn bao quát đủ bốn trường hợp như BSS (contained, partial, full, nhiều overlap): phần quanh lõi là lượt nói liền của hai người, không ghép thêm. Độ dài tương ứng L (hoặc Lb), tối đa 4 s. [mở: nếu muốn dùng chung dữ liệu với BSS thì dùng luôn mẫu 4 s của BSS làm mix; enrollment vẫn lấy ngoài.]
- **Enrollment**: một đoạn sạch của người mục tiêu, lấy ở **cùng recording**, **nằm ngoài mix**, không ai khác nói trong ±0.3 s, không nhạc. Đề xuất dài ~3 s nếu đủ; người chen ít đoạn sạch thì ghép các đoạn sạch của họ lại.
- **Target**: tiếng sạch (stem) của người mục tiêu trong mix, đúng độ dài mix.
- **Hai mẫu mỗi mix**: target A (enrollment A) rồi target B (enrollment B). Hai mẫu không trùng: enrollment khác, target khác; mô hình học cả hai phía từ một overlap.
- **Mỗi mẫu một enrollment khác**, không dùng chung một file cho cả recording; enrollment bốc lại mỗi epoch để tăng đa dạng (mặc định đề xuất, chưa chốt).
- **Thiếu đoạn sạch cho enrollment hoặc cho lõi**: bỏ mẫu khỏi train. Overlap < 0.3 s hoặc > 4 s: bỏ khỏi train; suy luận vẫn cố tách.
- Mẫu mà người mục tiêu **không** nói trong overlap (đầu ra phải là im lặng) nên có một tỉ lệ nhỏ để mô hình không bịa tiếng [mở: tỉ lệ chưa chốt].

## 7. So sánh BSS và TSE

| | BSS | TSE |
|---|---|---|
| Mẫu | `(mix, s1, s2)` | `(mix, enrollment, target)`, 2 mẫu mỗi mix |
| Độ dài mix | đúng 4 s | 3–4 s tùy mẫu |
| Cửa sổ | có (ctx + solo theo thứ tự) | không |
| Đoạn sạch ngoài | không | enrollment ~3 s, ngoài mix |
| Loss | PIT SI-SNR | SI-SNR trực tiếp |
| Chia tập | theo recording | theo recording, cùng split với BSS |

## 8. Định dạng theo từng mô hình (BSS)

| Mô hình | Mẫu | Định dạng | Lưu ý |
|---|---|---|---|
| MossFormer2 | 4 s | danh sách 3 cột `mix s1 s2`, 16 kHz | trainer mặc định cắt ngẫu nhiên 2 s mỗi file mỗi epoch: phải đặt độ dài đoạn 4 s, không thì overlap có thể bị cắt mất |
| SpeechBrain | 4 s | CSV: ID, duration, mix_wav, s1_wav, s2_wav, noise_wav | trainer mặc định dùng đoạn 4–6 s: đặt đúng 4 s |
| TIGER | 4 s | `mix.json`, `s1.json`, `s2.json`; mỗi dòng gồm đường dẫn và số mẫu | mặc định 3 s, file ngắn hơn bị bỏ im lặng: đặt 4 s |

Chưa kiểm bằng cách chạy từng trainer.

## 9. Số lượng

Kế hoạch cũ nêu 10–30 giờ đoạn đã render, ví dụ `--n 20000`. Con số cuối chưa chốt. Giới hạn thật là số giờ *nội dung sạch duy nhất* (script kiểm kê `audit_clean_pool.py` chưa viết). **Chưa chạy trên cache thật.** Mẫu thử nhỏ để kiểm định dạng: `--n 200 --files 5`.

## 10. Kiểm tra (bất biến)

`validate_mix_dataset.py` cần kiểm (một phần đã có, phần còn lại thêm theo thiết kế mới):

- mẫu BSS dài đúng 4 s; mix = s1 + s2 (+ nhiễu) trong sai số số học;
- mỗi stem ≥ 0.15 s tiếng nói;
- lõi L trong 0.3–4 s; thứ tự `[solo A][ctx trái][LÕI][ctx phải][solo B]` đúng;
- enrollment TSE không giao với mix, cùng recording, cùng split;
- phân phối dựng (kiểu chồng, độ dài overlap, chênh mức) khớp phân phối thật ở mục 4;
- recording trong tập test (`--exclude-index`) không xuất hiện ở train.

## 11. Việc cần làm để code khớp thiết kế

1. `render_mix_dataset.py`: dựng theo thứ tự lõi → ctx 0.25 s → solo → ctx dư; cắt bằng `AcousticBoundaryFinder` thay `snap`; thêm cờ `--min-clip` (0.3); TIGER đặt 4 s.
2. Thêm xuất manifest TSE (hai mẫu mỗi mix, enrollment ngoài mix).
3. `WindowPlanner` lúc suy luận: ép mẫu 4 s theo cùng thứ tự (hiện nhắm 15 s, ngữ cảnh 2 s, padding ≥ 1 s).
4. Viết `audit_clean_pool.py` để biết số giờ nội dung sạch duy nhất.
5. Chạy mẫu thử nhỏ và `validate_mix_dataset.py`.

## 12. Câu hỏi còn mở

- Khi lượt nói thật ngắn hơn ngữ cảnh được cấp: cắt thế nào.
- Overlap sát mép stem (không đủ chỗ cho ctx 0.25 s).
- Đảo vai A/B (người giữ sàn thay đổi giữa mẫu có nhiều overlap).
- Lb > 4 s: bỏ cả khối hay tách thành nhiều mẫu.
- Biên 0.25 s là đề xuất, gắn với dung sai 0.25 s của `full`.
- Chia solo đều khi dư lẻ là đề xuất.
- Tỉ lệ mẫu TSE "người mục tiêu không nói trong overlap".
- Có dùng chung mix 4 s của BSS cho TSE hay mix 3–4 s riêng.
