# Cách tạo dữ liệu huấn luyện tách overlap (BSS và TSE)

Ngày 2026-10-22. Nhánh `training-separation`. **Đây là thiết kế; code chưa sửa theo thiết kế này** (xem mục 11). Nguồn: số đo `doc/data/overlap_stats_recordings.csv`, `tools/measure_real_overlap_stats.py`, `tools/render_mix_dataset.py` (đường dẫn `tools/` và `utils/` tính từ `podcast-pipeline/`). Doc này **thay thế** `thiet_ke_tach_overlap_bss_tse.md` ở các điểm mâu thuẫn (margin đoạn sạch ±0.3 s thay vì ±0.2/±0.5 s, cửa sổ W giây có solo theo công thức mục 5, `min_clip`) và kế hoạch vòng ghi TIGER 3 s: nay W theo mô hình (mục 8). `WindowPlanner` và `AcousticBoundaryFinder` **không có trên nhánh này**, chỉ có ở `origin/solid-architecture` (`podcast-pipeline/utils/separation_window.py`, `utils/acoustic_boundary.py`).

**Độ dài mẫu W là tham số theo từng mô hình, không phải hằng số 4 s.** Mọi con số "4 s" trong doc là ví dụ tiêu biểu (W = 4 s); mỗi mô hình có W riêng (mục 8) và được render thành một bộ dữ liệu riêng với `--window W`. Công thức ở mục 5 viết theo W.

Quy ước nhãn: **[đã có]** là điều code/số đo hiện tại đã làm; **[đề xuất]** là thiết kế mới, chưa chạy; **[mở]** là câu hỏi chưa chốt.

## 1. Mục tiêu và nguyên tắc

- Huấn luyện mô hình tách hai người nói chồng nhau trên hội thoại tiếng Việt. Hai họ mô hình dùng chung một nguồn dữ liệu:
  - **BSS** (tách mù): `(mix, s1, s2)`, loss PIT SI-SNR.
  - **TSE** (tách theo người mục tiêu): `(mix, enrollment, target)`, loss SI-SNR trực tiếp, không PIT.
- **Train và suy luận phải giống nhau.** Mẫu train có hình dạng giống đầu vào lúc suy luận. Hệ quả: mẫu BSS dài đúng W giây (W theo mô hình, ví dụ 4 s), dựng theo cùng thứ tự ưu tiên, và bộ chia cửa sổ lúc suy luận (`WindowPlanner`, hiện nhắm 15 s) phải ép ra mẫu W giây theo cùng thứ tự (danh sách thay đổi ở mục 11) [đề xuất].
- Ground truth tự dựng: mix = tổng các stem sạch (cộng nhiễu nền), nên mỗi stem là đáp án chính xác. Không dùng đầu ra của mô hình sinh làm dữ liệu.

## 2. Luồng dữ liệu

```
cache diarization (360 recording, 351 h)
   │  measure_real_overlap_stats.py
   ▼
pairs.csv  ── mỗi dòng = một overlap thật (độ dài lượt, độ dài overlap, kiểu chồng, chênh mức)
   │  collect_overlap_samples.py: ~100 clip thật làm tập test, loại recording của chúng khỏi cả train và val
   ▼
render_mix_dataset.py  ── mỗi mẫu bắt đầu từ MỘT dòng pairs.csv,
   │                        dựng lại bằng đoạn sạch của chính recording đó (mẫu overlap có thể lấy từ recording khác theo `--mix`, nhưng âm thanh luôn từ recording đang dựng)
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

Chia train/val theo recording (`--val-share`, mặc định 0.05); test là ~100 clip thật của `collect_overlap_samples.py`, recording của chúng bị loại bằng `--exclude-index`; TIGER `test/` hiện là bản sao val. Hai mẫu TSE cùng mix luôn cùng split.

## 3. Định nghĩa

| Thuật ngữ | Định nghĩa | Tham số |
|---|---|---|
| Overlap | hai segment của hai người khác nhau chồng nhau | ≥ 0.1 s (`--min-overlap`) |
| Sự kiện overlap | các khoảng chồng cách nhau ≤ 1.0 s gộp lại | `--merge-gap 1.0` |
| Đoạn sạch | segment của một người, không ai khác nói trong ±0.3 s, không nhạc | `--margin 0.3`, dài ≥ `min_clip` (đề xuất 0.3 s; code hiện cứng 0.25 s trong `main()`, chưa có cờ CLI; 0.3 s làm nhỏ kho đoạn sạch của người chen, chưa đo mức giảm) |
| Stem | tín hiệu sạch của một người trong mẫu | cần ≥ 0.15 s tiếng nói (`MIN_ACTIVE`) |
| Lõi overlap (core) | phần cả hai cùng nói, dài L | L trong 0.3–W s |
| Người giữ sàn (A) | vai có lượt bắt đầu trước trong dòng `pairs.csv` (độ dài lượt = cột `first_len`); `pairs.csv` không có ID người nói, renderer gán vai này cho một speaker ngẫu nhiên của recording (`realize`) | |
| Người chen (B) | người còn lại | |
| Ngữ cảnh (ctx) | phần nối tiếp liền mạch của **chính đoạn sạch đã dùng cho lõi** (không ghép từ nơi khác); contained: đoạn sạch của A phải dài ≥ L + 2·ctx; partial: đoạn A ≥ ctx trái + L, đoạn B ≥ L + ctx phải | |
| Solo | đoạn sạch chỉ một người nói, ghép từ nơi khác | |

Kiểu chồng (cung cấp bởi `classify` trong `measure_real_overlap_stats.py`):
- **full**: đầu và cuối hai lượt lệch ≤ 0.25 s;
- **contained**: người sau kết thúc không muộn hơn người trước quá 0.25 s (người chen nằm trong lượt của người giữ sàn);
- **partial**: còn lại (người chen bắt đầu trong lượt A và kéo dài sang sau khi A dừng).

## 4. Thống kê overlap thật (làm cơ sở thiết kế)

18.623 sự kiện, 19.013 cặp, trên 351 giờ.

| Đại lượng | Giá trị |
|---|---|
| Độ dài *sự kiện* overlap (đã gộp khoảng cách ≤ 1 s, không phải độ dài L từng cặp mà thiết kế bốc) | trung vị 0.46 s, p10 0.2 s, p90 2.58 s |
| Kiểu chồng | contained 61.2%, partial 37.5%, full 1.3% |
| Mật độ | 0.885 sự kiện/phút; overlap chiếm 1.54% thời lượng |
| Chênh mức giữa hai người | \|chênh\| trung vị 1.75 dB; p90 trong khoảng tin cậy 95% 4.8–5.6 dB |

Hệ quả: một cửa sổ cỡ 4 s thường chỉ chứa **một** overlap. Với bộ dựng cũ, stem người chen chỉ chiếm ~10% cửa sổ; theo bố cục mới người chen có L + solo (≈ 37% ở L = 0.46 s với contained; ≈ 56% với partial vì có thêm ctx phải), vẫn lệch về người giữ sàn. Quy tắc cũ "≥ 40% khung hoạt động cho cả hai stem" đã bỏ vì sẽ loại gần hết mẫu đúng kiểu thật. Các số trong bảng lấy từ plan vòng (`ke_hoach_thu_nghiem_tach_giong_theo_vong.md` mục 4.7); `pairs.csv`/`summary.json` không nằm trong repo. Số đo lấy từ ranh giới segment của DiariZen, có thể là cận dưới của overlap thật.

## 5. Mẫu BSS: đúng W giây (ví dụ W = 4 s), dựng theo thứ tự ưu tiên [đề xuất]

W do mô hình quyết định (mục 8). Mỗi bước chỉ dùng phần W còn dư; các ngưỡng 0.25 / 0.3 / 0.6 / 1 / 2 s là tuyệt đối, không đổi theo W:

1. **Lõi overlap** (L giây), luôn giữ nguyên, kể cả L = W.
2. **Ngữ cảnh biên**: 0.25 s mỗi bên lõi (thay `MARGIN = 0.2` s hiện tại trong `render_mix_dataset.py`), để overlap nằm hẳn bên trong, không dính mép. Giữ được khi `W − L ≥ 0.5`.
3. **Solo sạch**: nếu `W − L ≥ 0.5` thì `R' = W − L − 0.5`, nếu không bỏ bước này (R' = 0). Tính bằng số mẫu nguyên (sr = 16000) để các ngưỡng không lệch do số thực.
   - `R' ≥ 2`: mỗi người 1 s solo;
   - `0.6 ≤ R' < 2`: chia đều `R'/2` (≥ 0.3 s, bằng `min_clip`) mỗi người;
   - `R' < 0.6`: không solo. Solo có khi và chỉ khi L ≤ W − 1.1 s (W = 4: L ≤ 2.9 s).
4. **Ngữ cảnh dư**: phần còn lại sau solo chia nửa cộng vào ctx hai bên (nếu `W − L < 0.5`: ctx = `(W − L)/2` mỗi bên và chấp nhận overlap sát mép; ctx tối thiểu = 0.25 s, hoặc (W − L)/2 khi W − L < 0.5; thiếu dưới mức này → bỏ mẫu (trường hợp 5); thiếu phần ctx dư → câu hỏi mở ở mục 5.2).

Thứ tự thời gian: `[solo X][ctx trái][LÕI][ctx phải][solo Y]`. Bên nào của solo là A hay B **chọn ngẫu nhiên** mỗi mẫu (tránh thiên lệch vị trí); với partial ctx trái thuộc người bắt đầu trước.

Ví dụ với **W = 4 s**, ctx mỗi bên / solo mỗi người theo L (W khác thì tính lại theo các bước trên; W lớn hơn thì thêm ctx và solo, W nhỏ hơn thì solo vắng sớm hơn):

| L (s) | ctx mỗi bên | solo mỗi người |
|---|---|---|
| 0.3 | 0.85 | 1 |
| 0.46 (trung vị sự kiện, minh họa) | 0.77 | 1 |
| 1 | 0.5 | 1 |
| 1.5 | 0.25 | 1 |
| 2 | 0.25 | 0.75 |
| 3 | 0.5 | 0 (R' = 0.5 < 0.6) |
| 3.2 | 0.4 | 0 |
| 3.5 | 0.25 | 0 |
| 4 | 0 | 0 |

```
L = 1 s (contained): solo 1 s + ctx 0.5 s + lõi 1 s + ctx 0.5 s + solo 1 s = 4 s (W = 4)
 A  ░░ solo ░░|▒▒ ctx ▒▒|██ LÕI ██|▒▒ ctx ▒▒|
 B                      |██ LÕI ██|              ░░ solo ░░
```

Độ dài mẫu phải đúng W·sr mẫu: cắt/đệm sau khi ghép (mối nối crossfade 20 ms như planner, thay fade 10 ms hiện tại).

### 5.1 Sáu trường hợp lớn

| # | Trường hợp | Dựng thế nào | Dùng khi train? |
|---|---|---|---|
| 1 | **Contained** | lõi = lượt của B nằm trong A; ctx hai bên đều là phần nối tiếp của đoạn A | có, L 0.3–W s (L lớn thì biên co lại theo bước 2 và 4) |
| 2 | **Partial** | ctx trái nối tiếp đoạn A, ctx phải nối tiếp đoạn B | có |
| 3 | **Full** | hai lượt trùng đầu/cuối (dung sai 0.25 s): không có tiếng liền quanh lõi. ctx để im lặng trên cả hai stem; nếu mẫu được cộng nhiễu, mix ở đó là nhiễu nền từ `noise_bank()` [đã có] | có, hiếm (1.3%) |
| 4 | **Nhiều overlap** (2–5) | lõi = khối từ đầu overlap đầu đến cuối overlap cuối (Lb), khoảng giữa là tiếng liền mạch; người giữ sàn có thể xen kẽ. **Renderer hiện chỉ dựng một cặp từ `pairs.csv`**, nên cần dựng nhóm cặp + nội dung đoạn nối; v1 có thể bỏ trường hợp này [mở] | có nếu Lb ≤ 4 s, cần code mới |
| 5 | **Thiếu đoạn sạch** | thiếu đoạn sạch cho solo của một người → chia phần solo đó vào ctx hai bên nếu đoạn sạch của lõi đủ dài, nếu không để im lặng ở chỗ solo (stem = 0), vẫn dùng mẫu; thiếu cho lõi hoặc cho ctx tối thiểu → bỏ mẫu | một phần |
| 6 | **Ngoài phạm vi** (L < 0.3 s, hoặc L > W, hoặc Lb > W) | không dựng | loại khỏi train |

Với các mẫu bị bỏ ở trường hợp 5 và 6: **suy luận vẫn cố tách như bình thường**; chỉ dữ liệu train bỏ chúng. Lọc template L < 0.3 hoặc > W *trước* khi bốc để nhóm strat/hard không phí lượt thử.

### 5.2 Nguồn tín hiệu để dựng lõi

- Lõi tổng hợp từ hai đoạn sạch (không dùng audio overlap thật); L = cột `overlap` của dòng `pairs.csv`. [đã có: `pull()` bốc đoạn sạch cho từng mẫu; độ dài hiện theo `first_len`/`second_len` rồi cắt theo cửa sổ.] Khi `first_len` < L + 2·ctx (contained) hoặc < ctx trái + L (partial): phần ctx tối thiểu thiếu thì bỏ mẫu (trường hợp 5); với phần ctx dư: [mở] bỏ qua `first_len` và lấy đoạn sạch dài hơn, hay giữ `first_len` và để im lặng phần thiếu.
- Trong một mẫu, lõi/ctx và solo của cùng người lấy từ các đoạn sạch không giao nhau về thời gian; giữa các mẫu được dùng lại đoạn sạch nhưng vị trí cắt bốc ngẫu nhiên (không dùng một đoạn cố định cho cả file). Ctx nối tiếp đúng đoạn sạch dùng cho lõi.
- Điểm cắt đoạn sạch: `AcousticBoundaryFinder` (VAD, chỗ lặng) giống lúc suy luận, thay cho `snap` ±80 ms hiện tại [đề xuất]; điểm cắt dời tối đa ±80 ms tới chỗ lặng, sai lệch độ dài dồn vào ctx dư, rồi cắt/đệm cuối cho đúng W·sr. Cắt giữa từ làm mô hình thấy đầu/cuối cụt mà lúc suy luận không có.
- Ghép: crossfade 20 ms như planner, chỉ giữa hai khúc liền nhau trên cùng một stem; mép khúc giáp im lặng thì fade 20 ms [hiện code chỉ fade 10 ms mỗi đoạn trên nền im lặng].
- Mix = tổng stem + nhiễu nền (xác suất 0.5) [đã có]; nhiễu hiện không lưu thành file nên SpeechBrain thiếu cột `noise_wav` (mục 11).

### 5.3 Câu hỏi thường gặp

- **Đoạn cắt có cần đủ chữ không?** Không cần cho mô hình tách: loss tính trên tín hiệu, chữ không vào loss. Nhưng ngữ cảnh liên tục và điểm cắt ở chỗ lặng vẫn cần để giống lúc suy luận.
- **Có bắt buộc 1 s sạch mỗi người không?** Không. Solo chỉ có khi L ≤ 2.9 s (bảng ở trên).
- **Có cần cân bằng bằng cửa sổ solo thuần không?** Không [đề xuất]: cửa sổ solo thuần có một stem bằng 0, SI-SNR không xác định (cần loss riêng, như TSE ở mục 12) và không dạy thêm việc tách hai người. Dùng cửa sổ có solo như trên để giống dạng đầu vào lúc suy luận.
- **Lấy L dài bao nhiêu?** 0.3–W s. Thay `LEN_BINS` hiện tại (0–0.5, 0.5–1, 1–2, 2–5, >5 s) bằng ba nhóm 0.3–0.5, 0.5–1, 1–W s (W = 4: 1–4 s); "lấy đều" chỉ áp cho phần 20% strat. Bỏ L < 0.3 s (quá ngắn để học) và L > W (không vừa mẫu).

## 6. Mẫu TSE: không dựng cửa sổ, thêm một đoạn sạch ngoài [đề xuất]

Một mẫu = `(mix, enrollment, target)`.

```
 Enrollment A (~3 s, sạch, ngoài mix) ─┐
                                        ├─► mô hình TSE ─► target A
 Mix (T s, lõi overlap nguyên vẹn)  ─┘                    (tiếng sạch của A trong mix)
```

- **Mix**: lõi overlap + phần nối tiếp liền mạch của hai đoạn sạch quanh nó, tổng độ dài T bốc đều trong [max(3, L), W] s; ctx mỗi bên = `(T − L)/2` (partial: trái từ A, phải từ B); đoạn sạch không đủ → bỏ mẫu. **Không ghép solo, không áp quy tắc biên 0.25 s.** Bao quát các trường hợp được dựng ở mục 5.1. [mở]: dùng luôn mix W giây của BSS (một renderer, khớp suy luận) thay cho mix 3–W s riêng.
- **Enrollment**: một đoạn sạch của người mục tiêu, lấy ở **cùng recording**, **nằm ngoài mix** (không giao về thời gian với đoạn nào dùng trong mix), không ai khác nói trong ±0.3 s, không nhạc. Đề xuất dài ~3 s nếu đủ, tối thiểu 1 s sau khi ghép (không thì bỏ mẫu); người chen ít đoạn sạch thì ghép các đoạn sạch của họ lại. Lúc test, `collect_overlap_samples.py` dựng tham chiếu dài tới 30 s (`--ref-seconds`): cần chốt độ dài enrollment lúc suy luận cho khớp [mở].
- **Target**: tiếng sạch (stem) của người mục tiêu trong mix, đúng độ dài mix.
- **Hai mẫu mỗi mix**: target A (enrollment A) rồi target B (enrollment B); mô hình học cả hai phía từ một overlap.
- **Mỗi mẫu một enrollment khác**, không dùng chung một file cho cả recording; enrollment bốc lại mỗi epoch để tăng đa dạng (mặc định đề xuất; cần manifest lưu một nhóm enrollment ứng viên, tối đa 8 đoạn, chứ không chỉ một đường dẫn).
- **Thiếu đoạn sạch cho enrollment hoặc cho lõi**: bỏ mẫu khỏi train. Áp quy tắc loại của mục 5.1.

## 7. So sánh BSS và TSE

| | BSS | TSE |
|---|---|---|
| Độ dài mix | đúng W (theo mô hình) | 3–W s tùy mẫu (hoặc dùng mix W giây của BSS) |
| Cửa sổ | có (ctx + solo theo thứ tự) | không (trừ khi dùng mix BSS) |
| Đoạn sạch ngoài | không | enrollment ~3 s, ngoài mix |
| Loss | PIT SI-SNR | SI-SNR trực tiếp |

## 8. W và định dạng theo từng mô hình (BSS)

Mỗi mô hình có độ dài đoạn huấn luyện riêng; render **một bộ dữ liệu riêng cho mỗi W** (`--window W`, manifest riêng). Cột W là giá trị gợi ý, chưa chốt; mọi công thức ở mục 5 tính lại theo W đó. Lúc suy luận, planner dùng cùng W với mô hình đang chạy.

| Mô hình | W gợi ý | Định dạng | Lưu ý |
|---|---|---|---|
| MossFormer2 | 4 s | danh sách 3 cột `mix s1 s2`, 16 kHz | trainer mặc định cắt ngẫu nhiên 2 s mỗi file mỗi epoch: phải đặt độ dài đoạn bằng W, không thì overlap có thể bị cắt mất |
| SpeechBrain | 4 s (trainer mặc định dùng 4–6 s, chọn W trong khoảng đó) | CSV: ID, duration, mix_wav, s1_wav, s2_wav (`noise_wav` cần thêm, hiện chưa lưu) | đặt đúng W |
| TIGER | 4 s (mặc định của trainer là 3 s) | `mix.json`, `s1.json`, `s2.json`; mỗi file là một mảng JSON các cặp `[đường dẫn, số mẫu]`, một cặp mỗi mẫu | file ngắn hơn độ dài cấu hình bị bỏ im lặng: đặt đúng W |

Chưa kiểm bằng cách chạy từng trainer.

## 9. Số lượng

Kế hoạch cũ nêu 10–30 giờ đoạn đã render, ví dụ `--n 20000`. Con số cuối chưa chốt. Giới hạn thật là số giờ *nội dung sạch duy nhất* (script kiểm kê `audit_clean_pool.py` chưa viết). **Chưa chạy trên cache thật.** Mẫu thử nhỏ để kiểm định dạng: `--n 200 --files 5`.

## 10. Kiểm tra (bất biến)

`validate_mix_dataset.py` cần kiểm (một phần đã có, phần còn lại thêm theo thiết kế mới):

- mẫu BSS dài đúng W·sr mẫu (W lấy từ cấu hình render của bộ đó); mix = s1 + s2 (+ nhiễu) trong sai số số học;
- mỗi stem ≥ 0.15 s tiếng nói;
- lõi L trong 0.3–W s; trường `layout` trong `chunks.jsonl` có các khúc liền nhau, không chồng, theo thứ tự [solo X]? [ctx trái] [LÕI] [ctx phải] [solo Y]? (solo có thể vắng; nếu có cả hai thì X ≠ Y), tổng độ dài = W·sr; độ dài ctx/solo khớp bảng mục 5 trừ mẫu có cờ `case5` (solo dồn vào ctx hoặc để im) hoặc kiểu full (ctx im lặng trên cả hai stem);
- enrollment TSE không giao với mix, cùng recording, cùng split;
- phân phối dựng so với thật ở mục 4, chấp nhận lệch có chủ ý: bỏ L < 0.3 và > W, L chia nhóm đều ở phần strat; validator hiện so `template_overlap` (cặp) với `event_duration_s` (sự kiện), cần sửa cho cùng đơn vị;
- recording trong tập test (`--exclude-index`) không xuất hiện ở train.

## 11. Việc cần làm để code khớp thiết kế

1. `render_mix_dataset.py`: dựng theo thứ tự lõi → ctx 0.25 s → solo → ctx dư (thay `MARGIN = 0.2`); ctx nối tiếp đúng đoạn sạch của lõi; vị trí solo A/B ngẫu nhiên; cắt bằng `AcousticBoundaryFinder` thay `snap` ±80 ms; crossfade 20 ms; đệm/cắt đúng W·sr mẫu; thay `LEN_BINS` bằng 3 nhóm (0.3–0.5, 0.5–1, 1–W s) và lọc template L < 0.3 hoặc > W; thêm cờ `--min-clip` (0.3, hiện cứng 0.25); ghi trường `layout` vào `chunks.jsonl` (danh sách {loại: solo|ctx|core, người: A|B, bắt đầu, kết thúc tính bằng sample trong mẫu, src_start, src_end tính bằng giây trên recording}) và cờ `case5`; port `utils/acoustic_boundary.py` từ `origin/solid-architecture` (có fallback năng lượng khi không có Silero); lưu `noise.wav` và cột `noise_wav` cho SpeechBrain; TIGER đặt W theo bảng mục 8 (cũng `--window` help và các dòng ghi TIGER 3 s trong `ke_hoach_thu_nghiem_tach_giong_theo_vong.md`, khoảng dòng 57, 100, 122, 162).
2. Dựng nhóm cặp cho trường hợp nhiều overlap (hoặc chốt bỏ ở v1).
3. Xuất manifest TSE (hai mẫu mỗi mix, nhóm enrollment ứng viên ngoài mix, mỗi ứng viên lưu src_start/src_end để validator kiểm không giao).
4. `WindowPlanner` (`utils/separation_window.py`, nhánh `solid-architecture`): mục tiêu 15 → W (theo mô hình, tham số hóa); `preferred_context` đổi từ "ctx trước, đệm sau" sang "biên 0.25 s → solo → ctx dư"; `padding_min_seconds` (1.0 s) và ngưỡng 1 s tiếng nói bắt buộc đổi theo quy tắc solo mới; giữ crossfade 20 ms; thử [solo A…solo B] rồi [solo B…solo A], lấy cách đầu đủ đoạn sạch; chỉnh `context_seconds` 2.0, `max_context_seconds` 2.2, dung sai 0.35 s trong `build()`; `self.target` đang cứng 15 s (`round(15.0*sr)`) cần thành tham số; `padding_min_seconds` cũng là độ dài tối thiểu mỗi đoạn (`self.minimum`), giảm còn ~0.3 s cho solo; quy tắc tách lõi dài (`_split_core_bounds`, `min_piece`) xét lại khi lõi > 4 s.
5. `validate_mix_dataset.py`: so `template_overlap` với quantile cột `overlap` của `pairs.csv` (qua `--stats`) thay vì `event_duration_s`; kiểm `mix = s1 + s2 + noise` cho chunk có nhiễu khi đã lưu `noise.wav` (hiện chỉ kiểm chunk không nhiễu).
6. Viết `audit_clean_pool.py` để biết số giờ nội dung sạch duy nhất.
7. Chạy mẫu thử nhỏ và `validate_mix_dataset.py`.

## 12. Câu hỏi còn mở

- Nhiều overlap (trường hợp 4): nguồn đoạn nối; Lb > W bỏ cả khối hay tách; có đưa vào v1 không.
- Đoạn sạch nối tiếp đủ ctx tối thiểu nhưng ngắn hơn ctx dư: lấy đoạn dài hơn hay để im lặng (mục 5.2).
- TSE: dùng mix W giây của BSS hay mix 3–W s riêng; độ dài enrollment lúc suy luận (train ~3 s, test hiện tới 30 s).
- TSE mẫu âm (người mục tiêu không nói trong mix): cần loss riêng (SI-SNR không xác định với đích 0); bỏ cho tới khi chốt loss.
