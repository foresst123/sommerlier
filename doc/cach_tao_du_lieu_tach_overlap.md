# Cách tạo dữ liệu huấn luyện tách overlap (BSS và TSE)

Ngày 2026-10-23, nhánh `training-separation`. Đây là **bản đặc tả dữ liệu duy nhất** cho tách overlap. Kế hoạch mô hình, các vòng thử, đánh giá và nhánh TSE nằm ở [ke_hoach_thu_nghiem_tach_giong_theo_vong.md](ke_hoach_thu_nghiem_tach_giong_theo_vong.md). Số đo gốc nằm ở `doc/data/`. Đường dẫn `tools/` và `utils/` tính từ `podcast-pipeline/`. `WindowPlanner` (`utils/separation_window.py`) và `AcousticBoundaryFinder` (`utils/acoustic_boundary.py`) hiện chỉ có ở `origin/solid-architecture`.

Nhãn trạng thái:
- **[đã có]**: code hoặc số đo hiện tại đã làm.
- **[đề xuất]**: thiết kế đã chốt, code chưa khớp hoặc chưa kiểm.
- **[mở]**: chưa chốt.
- **[DGX]**: chỉ kiểm được khi có cache thật hoặc GPU.

**W (độ dài mẫu) là tham số theo từng mô hình.** 4 s chỉ là giá trị tiêu biểu. Mỗi W được render thành một bộ dữ liệu riêng (`--window W`), và mọi công thức dưới đây viết theo W.

## 1. Nguyên tắc

- Hai họ mô hình dùng chung một nguồn dữ liệu:
  - **BSS**: `(mix, s1, s2)`, loss PIT SI-SNR.
  - **TSE**: `(mix, enrollment, target)`, loss SI-SNR trực tiếp, không PIT.
- **Train = suy luận.** Mẫu train phải có đúng hình dạng đầu vào lúc suy luận. Renderer và `WindowPlanner` dùng **chung một module bố cục thuần**, `utils/separation_layout.py` (không đọc hay ghi audio), để hai bên giống nhau từ cấu tạo chứ không phải nhờ chép tay [đề xuất].
- Ground truth tự dựng: mix = tổng các stem sạch (+ nhiễu nền), nên mỗi stem là đáp án chính xác. Không dùng đầu ra của mô hình sinh làm dữ liệu.
- Stem lấy từ podcast thật đã mang phòng thu và micro của recording. Vì vậy **không thêm vang tổng hợp**; nhiễu là việc phụ [đã có].
- Loss PIT SI-SNR với stem toàn 0 là hằng số, gradient bằng 0. Vì vậy mỗi stem phải có ít nhất 0.15 s tiếng nói (`MIN_ACTIVE`) [đã có]. Quy tắc cũ "≥ 40% khung hoạt động cho cả hai stem" đã bỏ, vì nó loại gần hết mẫu đúng kiểu thật.

## 2. Luồng dữ liệu và chia tập

```
cache diarization (360 recording, 351 h)
   │  measure_real_overlap_stats.py            [đã có, đã chạy]
   ▼
pairs.csv, summary.json  (mỗi dòng = một cặp overlap thật: overlap, kiểu chồng, first_len, second_len, chênh mức)
   │  collect_overlap_samples.py: ~100 clip thật làm tập test; recording của chúng bị loại khỏi train/val (--exclude-index)
   ▼
render_mix_dataset.py --window W   (mỗi mẫu bắt đầu từ MỘT dòng pairs.csv, âm thanh luôn từ recording đang dựng)
   ▼
chunks.jsonl (nguồn sự thật) + render_report.json (số mẫu bị bỏ theo lý do)
   ├── manifest BSS: MossFormer2 / SpeechBrain / TIGER
   └── manifest TSE: 2 mẫu mỗi mix
   │  validate_mix_dataset.py
```

- **Nguồn template** (`--mix`, mặc định 0.7:0.2:0.1) [đã có]:
  - 70% từ chính recording đang dựng;
  - 20% "strat" từ cả split, chia đều theo nhóm độ dài L;
  - 10% "hard" (L ≥ 1 s, chênh mức gần 0 hoặc rất rộng).
- **Nhóm độ dài** [đề xuất]: `(l_min–0.5, 0.5–1, 1–W]`, dựng từ W, nhóm cuối gồm cả W. Template có L ngoài `[l_min, W]` bị lọc *trước* khi bốc, cho cả nhóm own, strat và hard. Nếu W < 1 thì bỏ nhóm rỗng.
- **Chia tập theo recording** [đề xuất]:
  - val chiếm 8–10% số recording (hiện mặc định 5%, chọn ngẫu nhiên), **phân tầng theo `events_per_min`**. Lý do: overlap rất tập trung, một nửa trong 18.623 sự kiện nằm ở 64/360 recording, còn recording p10 chỉ có ≤ 4 sự kiện;
  - tùy chọn `--group-regex` để gom recording cùng show hoặc host vào cùng một split, tránh lộ giọng giữa train, val và test;
  - **mọi W dùng cùng các lần bốc template** (cùng seed, cùng template và đoạn sạch) để so mô hình trên cùng các overlap.
  - Hai mẫu TSE cùng mix luôn cùng split. TIGER `test/` hiện là bản sao val.

## 3. Định nghĩa và tham số

| Thuật ngữ | Định nghĩa | Tham số |
|---|---|---|
| Overlap | hai segment của hai người khác nhau chồng nhau | ≥ 0.1 s (`--min-overlap`) [đã có] |
| Sự kiện overlap | các khoảng chồng cách nhau ≤ 1.0 s gộp lại | `--merge-gap 1.0` [đã có] |
| Đoạn sạch | phần segment của một người, không ai khác nói trong ±0.3 s, không nhạc | `--margin 0.3` [đã có]; dài ≥ `--min-clip` 0.25 s [đề xuất: thêm cờ; hiện `main()` truyền cứng 0.25] |
| Lõi (core) | phần cả hai cùng nói, dài L = cột `overlap` của `pairs.csv`; lưu L *thực dựng* | `--l-min` 0.25 ≤ L ≤ W [đề xuất] |
| Biên lõi | phần ngữ cảnh tối thiểu mỗi bên lõi | 0.25 s (khác với margin đoạn sạch ±0.3 s) |
| Solo | đoạn sạch chỉ một người nói, lấy từ nơi khác trong cùng recording | ≥ 0.3 s mỗi khúc (độc lập với `--min-clip`) |
| Ngữ cảnh (ctx) | phần nối tiếp liền mạch của **chính đoạn sạch dùng cho lõi** | ≤ 2.0 s mỗi bên |
| Người giữ sàn A / người chen B | A có lượt bắt đầu trước trong dòng `pairs.csv`. `pairs.csv` không có ID người nói, nên renderer gán A và B cho hai speaker ngẫu nhiên của recording | |
| Stem | tín hiệu sạch của một người trong mẫu | ≥ 0.15 s tiếng nói (`MIN_ACTIVE`) |

Kiểu chồng (do hàm `classify` gán):
- **full**: đầu và cuối hai lượt lệch nhau ≤ 0.25 s.
- **contained**: B kết thúc không muộn hơn A quá 0.25 s.
- **partial**: các trường hợp còn lại; B bắt đầu trong lượt của A và kéo dài sang sau khi A dừng.

`first_len` và `second_len` chỉ được lưu lại, **không ràng buộc độ dài**. Template chỉ quyết định kiểu chồng, L và chênh mức; ctx do W quyết định [đề xuất].

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
- L < 0.25 s chiếm phần đáng kể của overlap thật. Ước thô, 0.3 s đã bỏ khoảng 20–25%. Tỷ lệ chính xác cần đo trên `pairs.csv` [DGX].

Giới hạn của số đo:
- Chênh mức lấy từ mức khi nói một mình gần nhất, không phải mức tức thời trong overlap.
- Overlap lấy từ ranh giới segment DiariZen. 1.54% thấp hơn ngưỡng cảnh báo 3% của `stage_output_service.py`, nên đây có thể là cận dưới.
- Nhiễu nền có thể lẫn tiếng cười hoặc nhạc sót.

## 5. Mẫu BSS: đúng W giây, dựng theo thứ tự ưu tiên [đề xuất]

### 5.1 Ngân sách độ dài (`utils/separation_layout.py`)

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
- X ≠ Y, và chọn ngẫu nhiên mỗi mẫu. Planner bốc theo seed của từng sự kiện, rồi thử thứ tự kia nếu thiếu đoạn sạch, để phân phối thứ tự giống train.
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

### 5.2 Khi thiếu đoạn sạch: một chuỗi dự phòng, dùng chung với planner

Khi đoạn nối tiếp của lõi ngắn hơn ngân sách ctx, hoặc một người thiếu solo, áp lần lượt:

1. lấy đoạn sạch dài nhất còn vừa;
2. thu ctx về tối thiểu 0.25 s;
3. dồn phần thời gian thừa vào solo thêm;
4. để im lặng ở mép ngoài;
5. nếu vẫn không đủ lõi hoặc ctx 0.25 s thì **bỏ mẫu**.

Mẫu giữ lại nhờ các bước 1–4 ghi tên bước vào trường `fallback` (`null` nếu không cần dự phòng); tên lý do trong code `utils/separation_layout.py`: `ctx_short` (ctx co về tối thiểu, phần thiếu dồn vào solo thêm), `solo_to_ctx` (thiếu solo, dồn vào ctx), `silence_pad` (để im lặng ở mép), `tse_shrunk` (mix TSE co lại vì nguồn ngắn); drop khi thiếu lõi hoặc ctx tối thiểu. Code mặc định `--tse-mix none` (không dựng TSE cho đỡ tốn render); dùng `plain` để dựng. Planner áp đúng chuỗi này lúc suy luận.

### 5.3 Các trường hợp

| # | Trường hợp | Dựng | Train? |
|---|---|---|---|
| 1 | contained | lõi = lượt của B nằm trong A; ctx hai bên nối tiếp đoạn A | có |
| 2 | partial | ctx trái nối tiếp đoạn A, ctx phải nối tiếp đoạn B | có |
| 3 | full | ctx im lặng trên cả hai stem | có (1.3%) |
| 4 | nhiều overlap trong một sự kiện | không dựng ở v1. Cặp/sự kiện ≈ 1.021, nên tối đa ~2.1% sự kiện bị ảnh hưởng. Nếu đo lại > 5% thì đưa vào v1; có thể dựng lại từ `pairs.csv` (nhóm dòng cách nhau ≤ `--merge-gap`) | không (v1) |
| 5 | không đủ đoạn sạch cho lõi hoặc ctx 0.25 s | không dựng | không |
| 6 | L < `--l-min` hoặc L > W | không dựng | không |

- Trường hợp 4–6 **không sinh dòng** nào trong `chunks.jsonl`; số lượng theo lý do ghi vào `render_report.json`.
- Lúc suy luận vẫn cố tách cả ba trường hợp này:
  - Lõi > W được chia ở chỗ lặng thành ⌈L/W⌉ khúc ≤ W. Mỗi khúc tương ứng với mẫu L = W đã có trong train. Quy tắc `_split_core_bounds` hiện tại (`min_piece` 2.5 s, chỉnh cho 15 s) phải đổi theo.
  - Khoảng gộp cặp của planner phải bằng `--merge-gap`, để lõi lúc suy luận cùng đơn vị với train.

### 5.4 Nguồn tín hiệu và ghép

- **Lõi tổng hợp** từ hai đoạn sạch, không dùng audio overlap thật.
  - Contained: một đoạn sạch của A dài ≥ L + 2·ctx.
  - Partial: đoạn A ≥ ctx trái + L, kết thúc ở cuối lõi; đoạn B ≥ L + ctx phải, bắt đầu ở đầu lõi.
  - Full: hai đoạn ≥ L.
- **Lõi B ngắn**: ưu tiên đoạn sạch ngắn có sẵn của B dài ≈ L, dùng nguyên đoạn (giữ đầu và cuối tự nhiên). Nếu không có thì cắt từ đoạn dài hơn, chỉ dời điểm cắt ≤ 20 ms. L thực dựng được lưu vào dòng. Khi sai lệch L đo trên DGX là nhỏ, hạ `--l-min` xuống 0.2 [DGX].
- **Không giao nhau**: trong một mẫu, lõi/ctx và solo của cùng một người lấy từ các đoạn sạch không giao nhau về thời gian. Giữa các mẫu được dùng lại đoạn sạch, nhưng vị trí cắt bốc ngẫu nhiên.
- **Điểm cắt** bằng `AcousticBoundaryFinder` (năng lượng; `--vad silero` khi có torch), thay cho `snap` ±80 ms hiện tại. Mép ngoài của ctx chỉ được dời ra ngoài, để ctx không xuống dưới 0.25 s; sai lệch dồn vào ctx dư. Mỗi dòng ghi `cut_method`. Lúc suy luận dùng Silero [DGX].
- **Ghép**: mọi mối nối crossfade **chồng 20 ms**, giống `WindowPlanner._assemble`, kể cả mối nối đổi người (solo B → ctx A: một stem fade-out, stem kia fade-in, chồng nhau 20 ms). Crossfade không bao giờ ăn vào lõi. Ngân sách cộng thêm số mẫu bị chồng để tổng độ dài đúng W·sr.
- **Mức**:
  - Mỗi người dùng **một gain** cho mọi khúc của mình, tính trên khúc lõi/ctx. Giữ chênh lệch mức tự nhiên giữa solo và lõi như lúc suy luận; không chuẩn hóa từng khúc.
  - A về −23 dB, B về −23 dB + chênh mức của template (± jitter 1 dB; ca hard dùng chênh mức của nó).
- **Resample**: mỗi khúc cắt (kèm phần đệm) được resample 24 → 16 kHz, rồi ghép, rồi mới tính `M = s1 + s2 (+ nhiễu)` ở float32. Không resample mix riêng. Đỉnh của cả ba (và stem nhiễu) chia cùng một hệ số để < 0.95.
- **Nhiễu** [đã có]:
  - nền của chính recording (các đoạn không có segment), cộng vào mix với xác suất 0.5;
  - SNR 25–45 dB (80%) hoặc 10–25 dB (20%);
  - đích là stem không nhiễu;
  - [đề xuất] lưu `noise.wav` (sau khi chia hệ số đỉnh) cho cột `noise_wav` của SpeechBrain và cho việc kiểm `mix = s1 + s2 + noise`.

### 5.5 Câu hỏi thường gặp

- **Đoạn cắt có cần đủ chữ không?** Không. Chữ không vào loss. Nhưng ngữ cảnh liên tục và điểm cắt ở chỗ lặng vẫn cần để giống lúc suy luận.
- **Có thêm cửa sổ chỉ có solo không?** Không. Một stem toàn 0 làm SI-SNR không xác định, và mẫu đó không dạy thêm việc tách hai người.

## 6. Mẫu TSE: không dựng cửa sổ solo, enrollment ngoài mix [đề xuất]

```
 Enrollment A (≤ 4 s, sạch, ngoài mix) ─┐
                                         ├─► mô hình TSE ─► target A (tiếng sạch của A trong mix)
 Mix (T = W_tse, lõi nguyên vẹn)  ───────┘
```

- **Mix**: lõi + phần nối tiếp liền mạch của hai đoạn sạch quanh nó.
  - T = `W_tse`, độ dài đoạn của mô hình TSE đang dùng.
  - ctx mỗi bên = `(T − L)/2`, trần 2.0 s; phần còn lại là im lặng ở mép. Partial: trái từ A, phải từ B.
  - **Không solo, không biên 0.25 s bắt buộc.**
  - Đoạn nối tiếp ngắn thì thu T lại; lúc suy luận áp cùng quy tắc.
  - Dựng bằng cùng `separation_layout.py`, với solo = 0.
  - Dùng lại mix W giây của BSS chỉ là **ablation** (`--tse-mix bss`), không phải mặc định.
- **Enrollment**:
  - đoạn sạch của người mục tiêu, cùng recording, không giao với mọi khúc của mix, không ai khác nói trong ±0.3 s, không nhạc;
  - ghép các đoạn sạch **gần mix nhất về thời gian trước**, tối đa 4 s, tối thiểu 1 s (thiếu thì bỏ mẫu);
  - lưu thành **một kho enrollment cho mỗi (recording, speaker)** và tối đa 8 chỉ số ứng viên mỗi mẫu (kèm `src_start`/`src_end` để kiểm không giao), không ghi wav riêng cho từng mẫu;
  - train: bốc lại trong 8 ứng viên mỗi epoch (việc của dataloader phía trainer); suy luận: ứng viên gần nhất.
  - `collect_overlap_samples.py` hiện lấy đoạn dài nhất trước, tới 30 s (`--ref-seconds`). Phải đổi sang cùng quy tắc cho tập test TSE.
- **Target**: stem của người mục tiêu, đúng độ dài mix.
- **Hai mẫu mỗi mix**: (enrollment A → A) và (enrollment B → B). Báo kết quả tách theo vai (giữ sàn / chen); vai chen là ca khó mà pipeline cần.
- **Mẫu âm** (người mục tiêu không nói): không dùng cho tới khi chọn loss riêng [mở].

| | BSS | TSE |
|---|---|---|
| Độ dài mix | đúng W (theo mô hình BSS) | `W_tse`, thu lại khi thiếu đoạn sạch |
| Solo trong mix | có (mục 5.1) | không |
| Đoạn sạch ngoài mix | không | enrollment ≤ 4 s |
| Loss | PIT SI-SNR | SI-SNR trực tiếp |

## 7. W và định dạng theo từng mô hình

Mỗi mô hình có một bộ render riêng, `--window W`. Lúc suy luận, planner dùng **cùng W** với mô hình đang chạy, và mô hình phải giải mã **cả cửa sổ W trong một lượt**. Riêng MossFormer2, mặc định giải mã từng đoạn 2 s với bước 75%; phải đặt cửa sổ giải mã bằng W. Cột W dưới đây là gợi ý, chưa chốt.

| Mô hình | W gợi ý | Định dạng [đã có] | Phải đặt |
|---|---|---|---|
| MossFormer2 | 4 s | danh sách 3 cột `mix s1 s2`, 16 kHz (`tr_list`/`cv_list`) | độ dài đoạn train = W (mặc định cắt ngẫu nhiên 2 s); cửa sổ giải mã = W |
| SpeechBrain | 4 s (recipe dùng 4–6 s) | CSV `ID, duration, mix_wav, s1_wav, s2_wav` (+ `noise_wav` [đề xuất]) | độ dài = W; kiểm recipe chấp nhận `noise_wav` rỗng [DGX] |
| TIGER | 4 s (mặc định trainer 3 s) | `mix.json`/`s1.json`/`s2.json`, mỗi phần tử là cặp `[đường dẫn tuyệt đối, số mẫu]` | độ dài = W (file ngắn hơn bị bỏ im lặng) |
| TSE | `W_tse` theo mô hình | manifest jsonl, 2 dòng mỗi mix; schema theo trainer TSE chọn sau [mở] | |

Chưa kiểm bằng cách chạy từng trainer [DGX].

## 8. Bảng nguồn sự thật [đề xuất]

Mỗi dòng `chunks.jsonl` giữ các trường hiện có (`id`, `split`, `recording`, `kind`, `window`, `sr`, `samples`, `paths`, `pattern`, `delta_db`, `noise_snr_db`, `target_order`, `template_overlap`, …), và thêm:
- `L`: L thực dựng;
- `first_len`, `second_len`: chỉ lưu;
- `layout`: danh sách `{kind: solo|ctx|core, who: A|B, start, end (mẫu trong sample), src_start, src_end (giây trên recording)}`. Lõi xuất hiện một lần cho mỗi người;
- `fallback`: `null` hoặc tên bước dự phòng;
- `cut_method`;
- `paths.noise`.

Ngoài ra, `render_report.json` ghi số template đã thử, số mẫu bị bỏ theo lý do (trường hợp 4, 5, 6) và số mẫu theo từng `fallback`.

## 9. Kiểm tra và đánh giá

`validate_mix_dataset.py`:
- [đã có]: file, sample rate, NaN/clip, `mix = s1 + s2` cho chunk không nhiễu, stem có tiếng, không recording nào ở cả hai split, không dùng recording test.
- [đề xuất]:
  - `samples == round(W·sr)`;
  - `mix = s1 + s2 + noise` khi có `paths.noise`;
  - `layout` đúng thứ tự `[solo]?[ctx][core][ctx][solo]?`; khúc liền kề chồng đúng 20 ms (hoặc 0 ở chỗ im lặng hay mép lõi), không có khe; phủ đúng `[0, W·sr)`; hai solo khác người;
  - ctx nối tiếp đúng đoạn của lõi theo kiểu chồng; solo không giao với lõi;
  - độ dài các khúc khớp `separation_layout` trừ dòng có `fallback`; `l_min ≤ L ≤ W`;
  - enrollment TSE không giao với mix, cùng recording, cùng split;
  - so phân phối L dựng được với quantile cột `overlap` của `pairs.csv` (cùng đơn vị cặp). Hiện validator so với `event_duration_s` của sự kiện;
  - báo **tỷ lệ lõi trong cửa sổ** và tỷ lệ `fallback`, so với tỷ lệ dự phòng của planner trên clip thật.
- **Đánh giá**:
  - SI-SDRi trên cả cửa sổ chủ yếu đo phần solo/ctx (ở W = 4, L = 0.46, lõi chỉ chiếm ~12%), và tăng theo W. Vì vậy **SI-SDRi vùng lõi** (đọc `layout`, lõi ± 0.1 s) là bắt buộc khi chọn mô hình, cùng các chỉ số trên clip thật (mục 6 của kế hoạch vòng).
  - Loss phụ có trọng số vùng lõi chỉ là ablation [mở].
- Nghe ngẫu nhiên ~20 mẫu mỗi bộ [DGX].

## 10. Số lượng

- Kế hoạch cũ nêu 10–30 giờ đoạn đã render (ví dụ `--n 20000`, ≈ 22 h ở W = 4). Con số cuối chưa chốt.
- Giới hạn thật là số giờ **nội dung sạch duy nhất**. `audit_clean_pool.py` chưa viết; nó sẽ đếm số giờ sạch theo speaker, số đoạn A ≥ L + 2·ctx theo nhóm L, và kho của người chen ở `--min-clip` 0.25 so với 0.3 [DGX].
- Chưa chạy trên cache thật. Mẫu thử định dạng: `--n 200 --files 5` cho mỗi W [DGX].

## 11. Việc cần làm để code khớp thiết kế

Trên nhánh này:
1. `utils/separation_layout.py`: ngân sách (mục 5.1) và chuỗi dự phòng (5.2), kèm test thuần (bảng mục 5.1, tổng đúng W·sr).
2. Port `utils/acoustic_boundary.py` giữ nguyên byte từ `origin/solid-architecture`.
3. `render_mix_dataset.py`:
   - tách `plan_chunk` (chỉ bố cục) và `render_layout` (audio);
   - bỏ `MARGIN`, `place_window`, `snap`;
   - thêm cờ `--l-min`, `--min-clip` (sửa lỗi `main()` truyền cứng 0.25), `--tse-mix bss`, `--group-regex`, `--vad`;
   - nhóm L theo W; val phân tầng;
   - lưu `noise.wav`; ghi `layout`, `fallback`, `render_report.json`;
   - manifest TSE và kho enrollment; cột `noise_wav` cho SpeechBrain.
4. `validate_mix_dataset.py`: các kiểm tra [đề xuất] ở mục 9 và helper SI-SDRi vùng lõi.
5. `collect_overlap_samples.py`: tham chiếu TSE theo quy tắc enrollment ở mục 6.

Trên `solid-architecture` (sau khi `separation_layout.py` và `acoustic_boundary.py` có ở cả hai nhánh):

6. `WindowPlanner`:
   - mục tiêu W thay cho 15 s cứng (`self.target`);
   - dùng `separation_layout` thay `preferred_context`, `padding_min_seconds` / `minimum` (1 s), dung sai 0.35 s;
   - thứ tự solo theo seed của sự kiện; khoảng gộp = `--merge-gap`;
   - chia lõi > W thành ⌈L/W⌉ khúc;
   - TSE: cửa sổ không solo theo mục 6.

Trên DGX:

7. `audit_clean_pool.py`, mẫu thử, validator, nghe thử, chạy thử từng trainer.

## 12. Câu hỏi còn mở (danh sách duy nhất)

1. `--l-min` 0.25 → 0.2: chờ đo tỷ lệ L < 0.25 / 0.3 trên `pairs.csv` và sai lệch L thực dựng [DGX].
2. Trường hợp 4: đo tỷ lệ sự kiện có nhiều cặp theo khoảng gộp của planner; nếu > 5% thì đưa vào v1 [DGX].
3. `--group-regex`: có metadata show/host không; báo kết quả cả hai cách chia.
4. W cuối cho từng mô hình và `W_tse`.
5. Schema manifest TSE theo trainer TSE được chọn (kế hoạch vòng, nhánh TSE).
6. Mẫu âm TSE: loss nào (ví dụ loss năng lượng cho đích 0); nguồn mẫu âm từ vùng diarization báo overlap sai.
7. Loss phụ có trọng số vùng lõi cho BSS (ablation); jitter chia ctx dư khi train (ablation, mặc định tắt).
8. Số giờ render cuối, sau `audit_clean_pool.py`.
