# Thử nghiệm MossFormer2 song song với Sidon

Đo xem một bộ tách theo **masking** có tránh được ba vấn đề đã đo được của Sidon
hay không. Không thay gì trong pipeline — chỉ chạy trên dữ liệu đã có và in ra
bảng so sánh.

## Vì sao thử

Sidon là latent diffusion: nó **tổng hợp lại** giọng chứ không lọc mixture. Đo
trên 44 track thật của `hoahau`:

| Chỉ số | Mixture | Sidon |
|---|---|---|
| Tương quan sóng (A+B với mix) | — | **0.054** |
| Nền so với đỉnh | −38.6 dB | **−75.4 dB** |
| Khung bị cắt câm | 6.3% | **48.5%** |
| Lệch phổ so mixture | 0 | **2.10 dB** |
| Biến động giữa các lần chạy | 2.7 dB* | **4.8 dB** |

\* đối chứng: tản phổ giữa các đoạn mixture khác nhau, tức mức biến thiên "bình thường" do nội dung nói.

Tương quan **0.054** là bằng chứng cứng: đầu ra gần như không liên quan sóng gốc.

Masking thì dự đoán một bộ lọc trên chính mixture, nên **về nguyên tắc** không thể
tự bịa ra âm sắc hay đẩy một khung xuống câm tuyệt đối. Đó là giả thuyết cần kiểm.

Thêm một lý do: ở [REAL-TSE Challenge (SLT 2026)](https://arxiv.org/abs/2607.15198),
các hệ thống dẫn đầu đều dựa trên **BSRNN và TF-GridNet** — đều là masking miền
thời gian–tần số, không phải diffusion.

## Cái giá đã biết trước

`MossFormer2_SS_16K` chạy ở **16 kHz**, pipeline ở **24 kHz**. Vòng 24→16→24 cắt
mọi thứ trên 8 kHz.

Đo trên corpus này: chỉ **0.2%** năng lượng toàn tín hiệu nằm trên 4 kHz — nghe
như không đáng kể. Nhưng riêng **khung phụ âm xát** thì là **18.2%**. Đó chính là
dải làm nên `-c`, `-t`, `/s/`, `/ch/`.

Nên bảng kết quả có cột `fric >4k` để thấy cái giá đó, thay vì giấu đi.

## Chuẩn bị

### Dữ liệu — đã có sẵn

Không cần chạy lại pipeline. Script đọc phần dump từ lần chạy trước:

```
02_separation/audio/raw/separated/
    1141.11_1_2_mix.wav       ← mixture gốc, đầu vào cho MossFormer2
    1141.11_1_2_trackA.wav    ← Sidon đã tách
    1141.11_1_2_trackB.wav
```

Hiện có:

| Thư mục | Số mixture |
|---|---|
| `kaggle 2/.../hoahau` | **22** ← dùng cái này |
| `kaggle 2/.../thu_that_thach_10m` | 2 |
| `kaggle/.../hoahau` | 22 (lần chạy cũ, chưa bật EQ) |

Tổng ~42 MB. Nếu chạy trên A100 thì copy thư mục `kaggle 2` lên máy đó.

### Thư viện

Chỉ thiếu một cái:

```bash
pip install clearvoice
```

`librosa`, `soundfile`, `torch`, `numpy` đã có sẵn trong môi trường pipeline.

Lần chạy đầu sẽ tự tải `alibabasglab/MossFormer2_SS_16K` từ HuggingFace. Nếu A100
chạy offline thì tải trước:

```bash
huggingface-cli download alibabasglab/MossFormer2_SS_16K
```

## Chạy

Mọi đường dẫn dưới đây tính từ `podcast-pipeline/`.

```bash
cd podcast-pipeline

python tools/compare_separators.py \
  "../kaggle 2/working/vi_audio/_final/-tse-True-demucs-True-vad-True-diaModel-diarizen-initPrompt-True-merge_gap-0.3-seg_th-0.11-cl_min-11-cl-th-0.5-LLM-case_0/hoahau" \
  --dump moss_out/
```

**Chỉ rõ đường dẫn, đừng bỏ trống.** Khi không có tham số, script tự dò và có thể
chọn nhầm `thu_that_thach_10m` (2 mixture) thay vì `hoahau` (22) — thứ tự `glob`
không xác định.

Thử nhanh trước bằng `--limit 3` để chắc model tải và chạy được.

## Đọc kết quả

```
mixtures: 22   sidon tracks: 44   mossformer tracks: 44

              floor dB   gated  tilt dB  fric >4k
mixture          -38.6    6.3%     0.00     18.2%
sidon            -75.4   48.5%     2.10     41.3%
mossformer         ...     ...      ...       ...
```

Bốn cột, và cách đọc từng cột:

**`floor dB`** — mức nền so với đỉnh. **Đích là dòng `mixture` (−38.6)**, không phải
càng thấp càng tốt. Sidon ở −75.4 tức nó đào nền sâu hơn thực tế 37 dB, và đó là
lý do nghe "câm tuyệt đối" giữa các từ.

**`gated`** — tỉ lệ khung bị đẩy xuống dưới −50 dB. Mixture 6.3% là bình thường;
Sidon 48.5% nghĩa là gần nửa thời lượng bị xóa trắng.

**`tilt dB`** — lệch trung bình so với hình dạng phổ của mixture. **0 là trung
thành.** Sidon 2.10 dB là méo âm sắc đã biết (mất trầm, thừa 1–2 kHz).

**`fric >4k`** — năng lượng trên 4 kHz trong khung phụ âm. Đây là **cái giá của
16 kHz**, đối chiếu với phần thắng ở ba cột kia.

### Kết luận thế nào

- `floor dB` và `gated` của MossFormer2 **gần dòng mixture hơn** → thắng ở đúng chỗ
  Sidon yếu nhất, và đó là phần ảnh hưởng cả ASR lẫn dataset fullduplex.
- `tilt dB` **thấp hơn 2.10** → giữ âm sắc thật hơn.
- `fric >4k` **tụt mạnh** → mất phụ âm do resample. Cân với ba cột trên.

Nếu MossFormer2 thắng ba cột đầu mà `fric >4k` không tụt quá nhiều, hướng masking
đáng theo tiếp.

## Nghe thử

`--dump moss_out/` ghi ra từng bộ ba để so bằng tai:

```
moss_out/
  1141.11_1_2_0_mix.wav        mixture gốc
  1141.11_1_2_1_sidon_0.wav    Sidon
  1141.11_1_2_1_sidon_1.wav
  1141.11_1_2_2_moss_0.wav     MossFormer2
  1141.11_1_2_2_moss_1.wav
```

Nghe hai thứ: **khoảng lặng giữa các từ** (Sidon câm tuyệt đối, MossFormer2 nên
còn nền phòng) và **âm cuối** (`-c` trong "khác", `-t` trong "mát").

## Điều thử nghiệm này KHÔNG trả lời

**Không đo WER.** Cả hai đều là blind separation — chúng trả về hai track mà không
biết track nào của ai. Pipeline dùng ECAPA để gán, và bước đó không có trong script
này. Muốn biết WER thì phải nối MossFormer2 vào pipeline thật, và chỉ nên làm sau
khi bảng trên cho thấy nó đáng.

**Không đo tốc độ.** MossFormer2 chạy một lượt; Sidon chạy 100 bước khuếch tán. Chênh
lệch sẽ lớn nhưng script không tính giờ.

## Khắc phục sự cố

**`MossFormer2 unavailable (No module named 'clearvoice')`** — chưa `pip install clearvoice`.

**`no mixture/track dumps under ...`** — sai đường dẫn, hoặc lần chạy đó không bật
dump. Kiểm tra:
```bash
ls "<run_dir>/02_separation/audio/raw/separated" | head
```

**`MossFormer2 failed on <file>`** — script bỏ qua mixture đó và chạy tiếp; nếu tất
cả đều lỗi thì thường là model chưa tải xong hoặc hết VRAM.

**`nothing could be measured`** — không mixture nào qua được. Chạy lại với
`--limit 1` để thấy lỗi thật.

## Sau đó

Nếu MossFormer2 thắng rõ, bước tiếp là nối nó vào pipeline **song song** với Sidon
(công tắc trong config, mặc định tắt) rồi đo WER trên nhóm `tse: true` — đúng cách
đã làm với `spectral_restore`.

Nếu không thắng, kết quả vẫn có giá trị: nó xác nhận vấn đề nằm ở chỗ khác, và
loại được một hướng khỏi danh sách.
