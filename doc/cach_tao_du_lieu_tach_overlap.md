# Cách tạo dữ liệu huấn luyện tách overlap (BSS và TSE)

Ngày 2026-10-23, nhánh `training-separation`. Đây là **bản đặc tả dữ liệu duy nhất** cho tách overlap. Kế hoạch mô hình, các vòng thử, đánh giá và nhánh TSE nằm ở [ke_hoach_thu_nghiem_tach_giong_theo_vong.md](ke_hoach_thu_nghiem_tach_giong_theo_vong.md). Số đo gốc nằm ở `doc/data/`. Đường dẫn `tools/` và `utils/` tính từ `podcast-pipeline/`. `WindowPlanner` (`utils/separation_window.py`) hiện chỉ có ở `origin/solid-architecture` và **chưa dùng bố cục này** (mục 11).

Nhãn trạng thái:
- **[đã có]**: code có trên nhánh này, đã test trên dữ liệu tổng hợp, chưa chạy trên cache thật.
- **[đề xuất]**: thiết kế đã chốt, chưa có code.
- **[mở]**: chưa chốt.
- **[DGX]**: chỉ kiểm được khi có cache thật hoặc GPU.

**W (độ dài mẫu) là tham số theo từng mô hình.** 4 s chỉ là giá trị tiêu biểu. Mỗi W được render thành một bộ dữ liệu riêng (`--window W`), và mọi công thức dưới đây viết theo W.

## 1. Nguyên tắc

- Hai họ mô hình dùng chung một nguồn dữ liệu:
  - **BSS**: `(mix, s1, s2)`, loss PIT SI-SNR.
  - **TSE**: `(mix, enrollment, target)`, loss SI-SNR trực tiếp, không PIT.
- **Train = suy luận.** Mẫu train phải có đúng hình dạng đầu vào lúc suy luận. Bố cục nằm trong một module thuần, `utils/separation_layout.py` (không đọc hay ghi audio) [đã có]. Renderer đã dùng nó. `WindowPlanner` phải import đúng module này trước khi train thật [đề xuất]; trước đó, mẫu train chưa giống đầu vào lúc suy luận.
- Ground truth tự dựng: mix = tổng các stem + nhiễu thêm vào, nên mỗi stem là đáp án chính xác.
  - Stem "sạch" nghĩa là không có người nói khác. Stem vẫn mang nền và phòng thu của chính recording (speech-to-noise ~34 dB), nên đầu ra mô hình cũng sẽ giữ nền đó.
  - Không dùng đầu ra của mô hình sinh làm dữ liệu.
- Không thêm vang tổng hợp, vì stem đã mang vang của recording [đã có].
- Loss PIT SI-SNR với stem toàn 0 là hằng số, gradient bằng 0. Vì vậy mỗi stem phải có ít nhất 0.15 s tiếng nói (`MIN_ACTIVE`) [đã có]. Quy tắc cũ "≥ 40% khung hoạt động cho cả hai stem" đã bỏ, vì nó loại gần hết mẫu đúng kiểu thật.

## 2. Luồng dữ liệu và chia tập

```
cache diarization (360 recording, 351 h)
   │  measure_real_overlap_stats.py            [đã có, đã chạy]
   ▼
pairs.csv, recordings.csv, summary.json  (mỗi dòng pairs.csv = một cặp overlap thật: t, overlap, kiểu chồng, first_len, second_len, chênh mức)
   │  collect_overlap_samples.py: ~100 clip thật làm tập test (--exclude-index)
   ▼
render_mix_dataset.py --window W   (mỗi mẫu bắt đầu từ MỘT dòng pairs.csv, âm thanh luôn từ recording đang dựng)
   ▼
chunks.jsonl (nguồn sự thật) + render_report.json
   ├── manifest BSS: MossFormer2 / SpeechBrain / TIGER
   └── manifest TSE (--tse-mix): 2 dòng mỗi mix
   │  validate_mix_dataset.py, eval_separation.py
```

- **Nguồn template** (`--mix`, mặc định 0.7:0.2:0.1) [đã có]:
  - 70% từ chính recording đang dựng;
  - 20% "strat" từ cả split, chia đều theo nhóm độ dài L;
  - 10% "hard" (L ≥ 1 s, chênh mức gần 0 hoặc rất rộng).
- **Bốc giống nhau cho mọi W** [đã có]:
  - Nhóm độ dài có cạnh cố định trong `separation_layout`, không phụ thuộc W.
  - Template được bốc từ danh sách đầy đủ, **rồi mới loại** nếu L < `--l-min` hoặc L > W.
  - Cùng seed thì mọi W thấy cùng chuỗi template và cùng đoạn sạch. Vì vậy so mô hình có W khác nhau là so trên cùng các overlap; mẫu bị loại vì L > W chỉ là các ô trống trong chuỗi.
- **Template thuộc sự kiện nhiều overlap** (trường hợp 4) bị loại, không sinh dòng nào; số lượng ghi trong `render_report.json` [đã có].
- **Chia tập theo recording** [đã có]:
  - val mặc định 8% số recording (`--val-share 0.08`), **phân tầng theo `events_per_min`** (lấy từ `recordings.csv`). Lý do: overlap rất tập trung, một nửa trong 18.623 sự kiện nằm ở 64/360 recording, còn recording p10 chỉ có ≤ 4 sự kiện. `render_report.json` cảnh báo khi val có < 1000 template;
  - `--group-regex`: recording cùng nhóm luôn cùng split. Áp dụng cả cho **loại tập test**: một recording thuộc nhóm có recording test thì cũng bị loại;
  - tên recording là ID YouTube nên regex không suy ra được show hay host. Cần `--group-map` (CSV `video_id,group`) từ metadata của cache [đề xuất, mở: metadata];
  - hai mẫu TSE cùng mix luôn cùng split; kho enrollment theo recording nên không vượt split. TIGER `test/` là bản sao val.

## 3. Định nghĩa và tham số

| Thuật ngữ | Định nghĩa | Tham số |
|---|---|---|
| Overlap | hai segment của hai người khác nhau chồng nhau | ≥ 0.1 s (`--min-overlap`) [đã có] |
| Sự kiện overlap | các khoảng chồng cách nhau ≤ 1.0 s gộp lại | `--merge-gap 1.0` [đã có] |
| Đoạn sạch | phần segment của một người, không ai khác nói trong ±0.3 s, không nhạc | `--margin 0.3`, dài ≥ `--min-clip` 0.25 s [đã có] |
| Lõi (core) | phần cả hai cùng nói, dài L = cột `overlap` của `pairs.csv`; trường `L` lưu L *thực dựng* | `--l-min` 0.25 ≤ L ≤ W [đã có] |
| Biên lõi | phần ngữ cảnh tối thiểu mỗi bên lõi | 0.25 s (khác với margin đoạn sạch ±0.3 s) |
| Solo | đoạn sạch chỉ một người nói, lấy từ nơi khác trong cùng recording | ≥ 0.3 s mỗi khúc (độc lập với `--min-clip`) |
| Ngữ cảnh (ctx) | phần nối tiếp liền mạch của **chính đoạn sạch dùng cho lõi** | ≤ 2.0 s mỗi bên |
| Người giữ sàn A / người chen B | A có lượt bắt đầu trước trong dòng `pairs.csv`. `pairs.csv` không có ID người nói, nên renderer gán A và B cho hai speaker ngẫu nhiên của recording | |
| Stem | tín hiệu của một người trong mẫu | ≥ 0.15 s tiếng nói (`MIN_ACTIVE`) |

Kiểu chồng (do hàm `classify` gán):
- **full**: đầu và cuối hai lượt lệch nhau ≤ 0.25 s.
- **contained**: B kết thúc không muộn hơn A quá 0.25 s.
- **partial**: các trường hợp còn lại; B bắt đầu trong lượt của A và kéo dài sang sau khi A dừng.

`first_len` và `second_len` chỉ được lưu lại, **không ràng buộc độ dài**. Template chỉ quyết định kiểu chồng, L và chênh mức; ctx do W quyết định [đã có].

## 4. Thống kê overlap thật [đã có]

`measure_real_overlap_stats.py` đo trên 360/364 recording (351 h). Khoảng tin cậy 95% lấy mẫu lại theo recording; mỗi file đóng góp tối đa 50 sự kiện vào phân phối gộp. Số theo từng recording nằm ở `doc/data/overlap_stats_recordings.csv`. `pairs.csv` và `summary.json` không có trong repo.

| Đại lượng | Giá trị |
|---|---|
| Số lượng | 18.623 sự kiện, 19.013 cặp → tối đa ~2.1% sự kiện có nhiều hơn một cặp |
| Độ dài *sự kiện* | trung vị 0.46 s (CI 0.459–0.48), p10 0.2 s, p90 2.58 s |
| Kiểu chồng | contained 61.2%, partial 37.5%, full 1.3% |
| Chênh mức (người sau − người trước) | trung vị −0.08 dB; \|chênh\| trung vị 1.75 dB; p90 \|chênh\| 4.8–5.6 dB (CI) |
| Mật độ | 0.885 sự kiện/phút; overlap chiếm 1.54% thời lượng; theo recording: trung vị 33.5 sự kiện, p10 4, p90 119 |
| Overlap lân cận (±30 s) | trung vị 1, p90 4 |
| Speech-to-noise mỗi recording | trung vị 34 dB (p10 27, p90 42); nhiễu nền trung vị −57 dBFS |

Hệ quả:
- Một mẫu vài giây thường chỉ chứa **một** overlap.
- Khó khăn thật là phát hiện và tách một giọng ngắn, thường cùng mức, nằm trong một giọng dài.
- L < 0.25 s chiếm phần đáng kể của overlap thật (ước thô: ngưỡng 0.3 s đã bỏ khoảng 20–25%). `validate_mix_dataset.py` in tỷ lệ chính xác khi có `pairs.csv` [DGX].

Giới hạn của số đo:
- Chênh mức lấy từ mức khi nói một mình gần nhất, không phải mức tức thời trong overlap.
- Overlap lấy từ ranh giới segment DiariZen. 1.54% thấp hơn ngưỡng cảnh báo 3% của `stage_output_service.py`, nên đây có thể là cận dưới: DiariZen có thể bỏ sót backchannel.
- Nhiễu nền có thể lẫn tiếng cười hoặc nhạc sót.

## 5. Mẫu BSS: đúng W giây, dựng theo thứ tự ưu tiên [đã có]

### 5.1 Ngân sách độ dài (`separation_layout.plan_budget`)

Mọi độ dài tính bằng số mẫu nguyên. `round(W·16000)` và `round(W·24000)` phải là số nguyên đúng; nếu không, renderer từ chối W đó. Các ngưỡng 0.25 / 0.3 / 0.6 / 1 / 2 s là tuyệt đối, không đổi theo W.

1. **Lõi**: L, luôn giữ nguyên, kể cả L = W.
2. **Biên lõi**: 0.25 s mỗi bên, nếu `W − L ≥ 0.5`. Nếu không, ctx = `(W − L)/2` mỗi bên, không có solo.
3. **Solo**: `R' = W − L − 0.5`.
   - `R' ≥ 2`: 1 s mỗi người;
   - `0.6 ≤ R' < 2`: `R'/2` mỗi người;
   - `R' < 0.6`: không solo.
   - Tức là có solo khi và chỉ khi `L ≤ W − 1.1 s`.
4. **Ctx dư**: phần còn lại chia đôi, cộng vào ctx hai bên, **trần 2.0 s mỗi bên** (bằng `context_seconds` của planner).
5. **Solo thêm**: phần vượt trần chia đều vào solo hai người.
6. **Im lặng** ở mép ngoài nếu vẫn còn dư.

Thứ tự thời gian: `[solo X][ctx trái][LÕI][ctx phải][solo Y]`.
- X ≠ Y, và chọn ngẫu nhiên mỗi mẫu (`solo_order`). Planner phải bốc theo seed của từng sự kiện, rồi thử thứ tự kia nếu thiếu đoạn sạch [đề xuất].
- Partial: ctx trái thuộc A, ctx phải thuộc B.
- Full: ctx là im lặng trên cả hai stem (nếu có nhiễu, mix ở đó là nhiễu nền).

Ví dụ (ctx mỗi bên / solo mỗi người, tính bằng giây):

| L | W = 4 | W = 6 | W = 8 |
|---|---|---|---|
| 0.3 | 0.85 / 1 | 1.85 / 1 | 2.0 / 1.85 |
| 0.46 | 0.77 / 1 | 1.77 / 1 | 2.0 / 1.77 |
| 1 | 0.5 / 1 | 1.5 / 1 | 2.0 / 1.5 |
| 2 | 0.25 / 0.75 | 1.0 / 1 | 2.0 / 1 |
| 3 | 0.5 / 0 | 0.5 / 1 | 1.5 / 1 |
| 3.5 | 0.25 / 0 | 0.25 / 1 | 1.25 / 1 |
| 4 | 0 / 0 | 0.25 / 0.75 | 1.0 / 1 |

```
W = 4, L = 1 s, contained: solo 1 + ctx 0.5 + lõi 1 + ctx 0.5 + solo 1
 A  ░░ solo ░░|▒▒ ctx ▒▒|██ LÕI ██|▒▒ ctx ▒▒|
 B                      |██ LÕI ██|              ░░ solo ░░
```

Hệ quả: khi cả hai solo đều đủ, vị trí lõi chỉ phụ thuộc L (ví dụ W = 4, L = 0.46 thì lõi luôn bắt đầu ở 1.77 s). Điều này chỉ an toàn nếu planner dựng đúng cùng bố cục. Validator in phân bố vị trí lõi; jitter là ablation [mở].

### 5.2 Khi thiếu đoạn sạch: một chuỗi dự phòng (`separation_layout.fallback`)

Áp lần lượt:

1. lấy đoạn sạch vừa ngân sách; nếu không có thì lấy đoạn dài nhất;
2. thiếu ctx → thu ctx về tối thiểu (0.25 s), phần thiếu dồn sang solo thêm;
3. một người thiếu solo → dồn vào ctx, nếu đoạn lõi còn đủ dài (trong trần 2 s);
4. vẫn còn dư → dồn vào solo của người kia;
5. vẫn còn dư → im lặng ở mép ngoài;
6. không đủ lõi hoặc ctx tối thiểu → **bỏ mẫu**.

Trường `fallback` là `null`, hoặc các lý do nối bằng `+`, ví dụ `ctx_short+solo_missing_B+solo_to_ctx`. Các tên lý do:

| Lý do | Nghĩa |
|---|---|
| `ctx_short` | ctx bị thu lại, phần thiếu dồn sang solo |
| `solo_missing_A`, `solo_missing_B` | người đó không có solo nào ≥ 0.3 s |
| `solo_short_A`, `solo_short_B` | solo của người đó ngắn hơn ngân sách |
| `solo_to_ctx` | thời gian solo bị thiếu chuyển sang ctx |
| `solo_other` | thời gian thiếu chuyển sang solo của người kia |
| `silence_pad` | im lặng ở mép ngoài |
| `tse_shrunk` | mix TSE co lại vì nguồn ngắn |

Lý do bỏ mẫu được đếm trong `render_report.json`. Planner phải áp đúng chuỗi này lúc suy luận [đề xuất].

### 5.3 Các trường hợp

| # | Trường hợp | Dựng | Train? |
|---|---|---|---|
| 1 | contained | lõi = lượt của B nằm trong A; ctx hai bên nối tiếp đoạn A | có |
| 2 | partial | ctx trái nối tiếp đoạn A, ctx phải nối tiếp đoạn B | có |
| 3 | full | ctx im lặng trên cả hai stem | có (1.3%) |
| 4 | nhiều overlap trong một sự kiện (cặp cách nhau ≤ `--merge-gap`) | template bị loại. Cặp/sự kiện ≈ 1.021, nên ~2.1% sự kiện; tỷ lệ thật ghi vào `render_report.json` (`multi_overlap`). Nếu > 5% thì đưa vào v1 [DGX] | không (v1) |
| 5 | không đủ đoạn sạch cho lõi hoặc ctx tối thiểu | không dựng | không |
| 6 | L < `--l-min` hoặc L > W | không dựng | không |

- Trường hợp 4–6 **không sinh dòng** nào trong `chunks.jsonl`.
- Lúc suy luận vẫn cố tách cả ba trường hợp này [đề xuất, planner]:
  - Lõi > W được chia ở chỗ lặng thành ⌈L/W⌉ khúc ≤ W. Mỗi khúc tương ứng với mẫu L = W đã có trong train. `_split_core_bounds` hiện tại (`min_piece` 2.5 s, chỉnh cho 15 s) phải đổi theo.
  - Khoảng gộp cặp của planner phải bằng `--merge-gap`.

### 5.4 Nguồn tín hiệu và ghép [đã có]

- **Lõi tổng hợp** từ hai đoạn sạch, không dùng audio overlap thật.
  - Contained: một đoạn sạch của A dài ≥ L + 2·ctx.
  - Partial: đoạn A ≥ ctx trái + L; đoạn B ≥ L + ctx phải.
  - Full: hai đoạn ≥ L.
- **Mép lõi tự nhiên.** Mép lõi là nơi lúc suy luận có onset/offset thật, nên không được là chỗ cắt cụt giữa từ.
  - Contained: nếu B có một đoạn sạch dài trong `[L, L + 0.3 s]` thì dùng nguyên đoạn đó làm lõi (giữ đầu và cuối tự nhiên); khi đó L = độ dài đoạn.
  - Không có thì cắt từ đoạn dài hơn và dời **cả hai mép lõi** tới chỗ lặng.
  - Partial: mép cuối của A (cuối lõi) và mép đầu của B (đầu lõi) được dời tới chỗ lặng.
  - Mép ngoài của mỗi run (giáp solo hoặc mép cửa sổ) dời ≤ 80 ms. Mép chạm lõi dời ≤ 20 ms khi L < 0.4 s.
  - Độ dài các khúc giữ đúng ngân sách: điểm cắt chỉ dời *vị trí lấy trên nguồn*.
  - L thực dựng được lưu (`L`) và ghi `whole_clip_core`. Khi sai lệch L đo trên DGX là nhỏ thì hạ `--l-min` xuống 0.2 [DGX].
- **Bộ tìm điểm cắt**: `AcousticBoundaryFinder` (`utils/acoustic_boundary.py`, chép nguyên từ `solid-architecture`), mặc định theo năng lượng; `--no-snap` để tắt. Số điểm cắt theo phương pháp ghi trong `cuts`. Silero VAD như lúc suy luận chưa nối vào [đề xuất, DGX].
- **Không giao nhau**: trong một mẫu, các run của cùng một người lấy từ các đoạn nguồn không giao nhau. Giữa các mẫu được dùng lại đoạn sạch, nhưng vị trí cắt bốc ngẫu nhiên.
- **Ghép**: mỗi run (một khúc nguồn liền mạch) được đọc kèm đệm, resample 24 → 16 kHz riêng, rồi đặt vào stem. Mối nối giữa hai run chồng ≤ 20 ms với ramp tuyến tính bù nhau, kể cả khi đổi người (solo B → ctx A). Ramp không bao giờ ăn vào lõi.
- **Mức**: mỗi người dùng **một gain**, tính trên run chứa lõi của người đó và áp cho mọi khúc của người đó. Giữ chênh lệch mức tự nhiên giữa solo và lõi như lúc suy luận. A về −23 dB; B về −23 dB + chênh mức của template (± jitter 1 dB; ca hard dùng chênh mức riêng của nó).
- **Mix**: `M = A + B (+ N)` ở float32. Không resample mix riêng. Nếu đỉnh > 0.95 thì A, B, M, N chia cùng một hệ số.
- **Nhiễu**:
  - nền của chính recording (khoảng trống ≥ 0.8 s giữa các segment), xác suất `--noise-prob` 0.5;
  - SNR 25–45 dB (80%) hoặc 10–25 dB (20%);
  - đích là stem không có nhiễu thêm;
  - lưu `<id>_noise.wav`, nên `mix = s1 + s2 + noise` kiểm được [đã có].
  - Cờ tùy chọn [đã có, mặc định tắt]: SNR lấy theo `snr_db` của chính recording (`recordings.csv`, ±3 dB), và lọc khoảng trống có tiếng nói sót khỏi kho nhiễu.
  - Nếu bật cùng `--noise-prob` cao (0.8–1.0), nền liên tục sẽ che các bước nhảy nền ở mối nối và ở lõi (nơi nền của hai stem cộng lại). Lúc này đây là ablation [mở].

### 5.5 Câu hỏi thường gặp

- **Đoạn cắt có cần đủ chữ không?** Không. Chữ không vào loss. Nhưng ngữ cảnh liên tục và điểm cắt ở chỗ lặng vẫn cần để giống lúc suy luận.
- **Có thêm cửa sổ chỉ có solo không?** Không. Một stem toàn 0 làm SI-SNR không xác định, và mẫu đó không dạy thêm việc tách hai người.

## 6. Mẫu TSE: không dựng cửa sổ solo, enrollment ngoài mix [đã có]

```
 Enrollment A (≤ 4 s, sạch, ngoài mix) ─┐
                                         ├─► mô hình TSE ─► target A (tiếng của A trong mix)
 Mix (T ≤ W_tse, lõi nguyên vẹn)  ───────┘
```

- **Cờ**:
  - `--tse-mix none` (mặc định, không dựng TSE);
  - `plain`: thiết kế chính, mix riêng không solo;
  - `bss`: ablation, dùng lại mix W giây của BSS.
  - `--tse-window` = `W_tse` (mặc định bằng `--window`).
- **Mix `plain`** (`separation_layout` với `solo=False`):
  - lõi + phần nối tiếp liền mạch của hai đoạn sạch; ctx mỗi bên = `(T − L)/2`, trần 2.0 s. Partial: trái từ A, phải từ B;
  - không solo, không biên 0.25 s bắt buộc, không đệm im lặng;
  - T = `W_tse` khi đủ nguồn. Trần ctx làm T < `W_tse` khi `W_tse − L > 4 s`;
  - nguồn ngắn thì T co lại (`tse_shrunk`), nhưng không dưới `max(2 s, L + 0.5 s)`; dưới mức đó thì bỏ mẫu;
  - lúc suy luận áp cùng quy tắc [đề xuất, planner].
  - Mix TSE ghi ở `tse/<split>/` và `tse_mixes.jsonl`.
- **Enrollment** (`separation_layout.pick_enrollment` / `default_enrollment`):
  - đoạn sạch của người mục tiêu, cùng recording, không chạm (±0.3 s) mọi khúc nguồn của mix, không ai khác nói trong ±0.3 s, không nhạc;
  - **một kho enrollment cho mỗi (recording, speaker)**: `enroll/<recording>/<speaker>.wav` + `.json`. Kho nhận cả đoạn ngắn hơn 1 s và ghép lại khi cần, để người chen ít đoạn dài vẫn có enrollment;
  - mỗi mẫu lưu tối đa 8 ứng viên **gần lõi nhất về thời gian** (`enroll_candidates`, kèm `src_start`/`src_end`) và một tổ hợp mặc định `enroll_default` (gần nhất trước, ghép tới 4 s, tối thiểu 1 s; thiếu thì bỏ mẫu);
  - train: bốc lại trong các ứng viên mỗi epoch (việc của dataloader phía trainer); suy luận: tổ hợp gần nhất;
  - `collect_overlap_samples.py` dùng cùng quy tắc cho tham chiếu của tập test TSE [đã có; kiểm lại trên cache: DGX].
- **Target**: stem của người mục tiêu, đúng độ dài mix.
- **Hai mẫu mỗi mix**: (enrollment A → A, `target_role: holder`) và (enrollment B → B, `interrupter`). Báo kết quả theo vai; vai chen là ca khó mà pipeline cần.
- **Mẫu âm** (người mục tiêu không nói): không dùng cho tới khi chọn loss riêng [mở].

| | BSS | TSE (`plain`) |
|---|---|---|
| Độ dài mix | đúng W | ≤ `W_tse`, sàn `max(2 s, L + 0.5 s)` |
| Solo trong mix | có (mục 5.1) | không |
| Đoạn sạch ngoài mix | không | enrollment ≤ 4 s |
| Loss | PIT SI-SNR | SI-SNR trực tiếp |

## 7. W và định dạng theo từng mô hình

Mỗi mô hình có một bộ render riêng (`--window W`). Lúc suy luận, planner dùng **cùng W** với mô hình đang chạy, và mô hình phải giải mã **cả cửa sổ W trong một lượt**. Riêng MossFormer2, mặc định giải mã từng đoạn 2 s với bước 75%; phải đặt cửa sổ giải mã bằng W. Cột W dưới đây là gợi ý, chưa chốt [mở].

| Mô hình | W gợi ý | Định dạng [đã có] | Phải đặt |
|---|---|---|---|
| MossFormer2 | 4 s | `manifests/mossformer2/{tr,cv}_list.txt`, 3 cột `mix s1 s2` | độ dài đoạn train = W (mặc định cắt ngẫu nhiên 2 s); cửa sổ giải mã = W |
| SpeechBrain | 4 s (recipe dùng 4–6 s) | `manifests/speechbrain/{train,valid}.csv` có `noise_wav`; chunk không nhiễu trỏ tới một file im lặng cùng độ dài | độ dài = W; kiểm recipe chạy với cột này [DGX] |
| TIGER | 4 s (mặc định trainer 3 s) | `manifests/tiger/{train,val,test}/{mix,s1,s2}.json`, mỗi phần tử là cặp `[đường dẫn tuyệt đối, số mẫu]` | độ dài = W (file ngắn hơn bị bỏ im lặng) |
| TSE | `W_tse` theo mô hình | `manifests/tse/{train,val}.jsonl`, 2 dòng mỗi mix | schema cuối theo trainer TSE được chọn [mở] |

Chưa kiểm bằng cách chạy từng trainer [DGX].

## 8. Bảng nguồn sự thật [đã có]

Mỗi dòng `chunks.jsonl` gồm:
- **Nhận diện**: `id`, `split`, `recording`, `kind` (real|strat|hard), `window`, `sr`, `samples`, `src_sr`, `paths` {mix, s1, s2, noise|null}, `target_order`.
- **Template**: `template_overlap`, `template_t`, `first_len`, `second_len`, `pattern`, `delta_db`, `noise_snr_db`.
- **Bố cục**:
  - `L` (L thực dựng, giây);
  - `core` [đầu, cuối] (mẫu);
  - `slots`: các ô danh nghĩa liền nhau {kind: pad|solo|ctx|core, who, start, end};
  - `layout`: các khúc nguồn {kind, who: A|B, run, start, end (mẫu trong sample), fade_in, fade_out (mẫu), src_start, src_end (giây trên recording)}. Lõi xuất hiện một lần cho mỗi người; các khúc cùng `run` liền nhau trên nguồn;
  - `solo_order`, `fallback`, `whole_clip_core`, `cuts`, `holder`, `interrupter`;
  - `layout_cfg`: tham số `LayoutConfig` đã dùng; validator cần trường này;
  - `fade_samples`.
- **Tương thích cũ**: `overlap_in_window` (= L) và `overlap_start_in_window`.

Mỗi dòng TSE (`manifests/tse/*.jsonl`) gồm: `id`, `mix_id`, `split`, `recording`, `mix`, `target`, `target_role`, `target_speaker`, `samples`, `sr`, `enroll_bank`, `enroll_seconds`, `enroll_candidates`, `enroll_default`.

`render_report.json` ghi:
- tham số render;
- số recording và template theo split, số template ngoài `[l_min, W]`;
- `multi_overlap` (cặp/sự kiện, tỷ lệ sự kiện nhiều cặp);
- theo split: số lần thử, số dòng, số mẫu bị bỏ theo lý do (`drop:*`), số dòng theo `fallback`;
- cảnh báo.

## 9. Kiểm tra và đánh giá

`validate_mix_dataset.py` [đã có], thoát với mã 1 nếu có lỗi:
- file tồn tại; sample rate; `samples == round(W·sr)`; NaN/clip;
- `mix = s1 + s2 (+ noise)`; mỗi stem có tiếng;
- `check_layout`:
  - thứ tự `[pad]?[solo]?[ctx][core][ctx][solo]?[pad]?`; `slots` liền nhau phủ `[0, W·sr)`; `pad` chỉ có khi `fallback` khác null;
  - hai solo khác người; chủ của ctx đúng theo kiểu chồng;
  - khúc của hai run khác nhau chồng ≤ 20 ms (trừ hai lõi); ramp không vào lõi;
  - run liền mạch trên nguồn; các nguồn của cùng một người không giao nhau;
  - độ dài khớp `plan_budget` khi `fallback` là null; `l_min ≤ L ≤ W`;
- TSE: ứng viên enrollment cùng recording, cùng split, không giao với nguồn của mix, ≥ 1 s; mỗi mix tối đa 2 dòng khác vai;
- không recording nào ở cả hai split, không dùng recording test;
- in so sánh với số đo thật:
  - quantile L so với cột `overlap` của `pairs.csv` (cùng đơn vị cặp, kèm tỷ lệ cặp nằm ngoài `[l_min, W]`);
  - tỷ lệ lõi trong cửa sổ, phân bố vị trí lõi, tỷ lệ `fallback`, tỷ lệ có solo của cả hai;
  - kiểu chồng, chênh mức, nhiễu.

`eval_separation.py` [đã có]: SI-SDRi **vùng lõi** (đọc `layout`, lõi ± 0.1 s) và SI-SDRi cả cửa sổ, chia theo kiểu chồng, nhóm L và vai.
- SI-SDRi cả cửa sổ chủ yếu đo phần solo/ctx (ở W = 4, L = 0.46, lõi chỉ chiếm ~12%) và tăng theo W. Vì vậy chọn mô hình theo SI-SDRi vùng lõi cùng các chỉ số trên clip thật (mục 6 của kế hoạch vòng).
- Nối vào bước val của trainer [đề xuất]. Loss phụ có trọng số vùng lõi chỉ là ablation [mở].
- Nghe ngẫu nhiên ~20 mẫu mỗi bộ [DGX].

## 10. Số lượng

- Kế hoạch cũ nêu 10–30 giờ đoạn đã render (ví dụ `--n 20000`, ≈ 22 h ở W = 4). Con số cuối chưa chốt.
- Giới hạn thật là số giờ **nội dung sạch duy nhất**: `audit_clean_pool.py` [đã có] đếm giờ sạch theo speaker và kho của người chen. Chạy trên cache: [DGX].
- Chưa chạy trên cache thật. Mẫu thử định dạng: `--n 200 --files 5` cho mỗi W, rồi `validate_mix_dataset.py` [DGX].

## 11. Việc còn lại (xếp theo giá trị / chi phí)

1. **Port `WindowPlanner` sang `separation_layout`** trên `solid-architecture`, **trước mọi lần train thật**:
   - mục tiêu W thay cho 15 s cứng (`self.target`);
   - cùng ngân sách và chuỗi dự phòng, thay cho `preferred_context`, `padding_min_seconds` / `minimum` (1 s) và dung sai 0.35 s;
   - thứ tự solo theo seed của sự kiện; khoảng gộp = `--merge-gap`;
   - lõi > W chia thành ⌈L/W⌉ khúc;
   - cửa sổ TSE không solo;
   - cùng quy tắc mép lõi tự nhiên;
   - `separation_layout.py` và `acoustic_boundary.py` phải giống hệt ở cả hai nhánh.
2. **Lọc đoạn "sạch" bị lẫn tiếng** (DiariZen bỏ sót backchannel: "ừ", "vâng" của người chen nằm trong đích của người giữ sàn sẽ dạy mô hình giữ backchannel ở sai stem). Cách làm: kiểm nhất quán embedding speaker theo cửa sổ trượt (ECAPA so với tâm speaker) và/hoặc mô hình phát hiện overlap; bỏ đoạn bị gắn cờ; báo tỷ lệ bỏ trong `audit_clean_pool` [DGX].
3. **`--group-map`** (`video_id,group` từ metadata show/kênh) để chia tập theo show hoặc host; báo kết quả cả hai cách chia [mở: metadata].
4. **Độ thật của tiếng chen**: lệch mức âm cho B ở contained (ví dụ U(−6, 0) dB cho 30% mẫu, ablation); ưu tiên đoạn B < 0.8 s giống backchannel; hiệu chỉnh bằng mức đo trong overlap của ~100 clip test [DGX].
5. **Silero VAD** cho bộ tìm điểm cắt (`cut_method` theo từng mẫu), giống lúc suy luận [DGX].
6. Chạy trên DGX: `audit_clean_pool.py`, mẫu thử cho từng W, validator, nghe thử, chạy thử từng trainer, nối `eval_separation.py` vào val.

## 12. Câu hỏi còn mở (danh sách duy nhất)

1. `--l-min` 0.25 → 0.2: chờ đo tỷ lệ L < 0.25 / 0.3 trên `pairs.csv` và sai lệch L thực dựng [DGX].
2. Trường hợp 4: nếu `multi_overlap` > 5% thì đưa vào v1 [DGX].
3. Metadata show/host cho `--group-map`.
4. W cuối cho từng mô hình và `W_tse`.
5. Schema manifest TSE theo trainer TSE được chọn (kế hoạch vòng, nhánh TSE).
6. Mẫu âm TSE: loss nào (ví dụ loss năng lượng cho đích 0); nguồn mẫu âm từ vùng diarization báo overlap sai.
7. Ablation: loss phụ vùng lõi; jitter vị trí lõi; cờ nhiễu theo recording + `--noise-prob` cao.
8. Số giờ render cuối, sau `audit_clean_pool.py`.
