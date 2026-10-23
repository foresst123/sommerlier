# Lọc nhiễu hai tầng, và tầng 3 để làm sau

Tài liệu này ghi lại (1) kế hoạch hai tầng đã viết xong nhưng **chưa chạy trên DGX**, (2) tầng 3 (enhance sau hai tầng) mới chỉ **nghiên cứu, chưa có mã**, (3) kết quả tra cứu bản nâng cấp của từng mô hình, ngày 2026-10-23. Nó bổ sung cho [so_sanh_loc_nhieu.md](so_sanh_loc_nhieu.md) (cơ chế `--denoise`, công cụ so sánh), không thay thế.

Quy ước: **[đã xác minh]** là đã mở trang chính hoặc chạy; **[chưa xác minh]** là suy luận hoặc chỉ thấy qua kết quả tìm kiếm. "Không tìm thấy" không có nghĩa là không tồn tại.

## 1. Tình trạng

| Việc | Trạng thái |
|---|---|
| Backend RE-USE (`re_use`, `re_use_laughter`), chạy trong môi trường riêng | Có mã và test; **đã chạy `re_use` và `re_use.music-scope-spans` trên DGX, `ok`**; `re_use_laughter` chưa chạy |
| Bước `dryNN` và tập 16 biến thể hai tầng (`tools/variants_two_stage.json`) | Có mã và test; **chưa chạy trên DGX** |
| UniPASE (thử bằng mã chính thức) | Chưa chạy được, môi trường `unipase` thiếu `torch` |
| Tầng 3 | **Chưa có gì trong mã.** Chỉ có mục 5 dưới đây |

## 2. Kế hoạch hai tầng

Mỗi biến thể là một chuỗi: một mô hình tầng 1 nhẹ, trộn lại một phần tín hiệu gốc, rồi một mô hình tầng 2 mạnh hơn nhưng cũng được làm nhẹ tay.

```
<tầng 1> > blend25 > <tầng 2> > dry50
```

- Tầng 1: `mpsenet_dns`, `mf2se48` (có thêm bản `scope-spans`: `pipeline.music_scope=spans`), `rf_crowd`, `mf2se48`.
- Tầng 2: `dfn3`, `bsrnn26`, `mpsenet`, `re_use`.
- 4 × 4 = **16 biến thể**, nhóm `two_stage` (id ví dụ `mf2se48+blend25+re_use+dry50.music-scope-spans`).
- `blend25` và `dry50` là **giá trị khởi đầu tự đặt**, chưa tối ưu. Chỉnh bằng nghe thử.

### `blendNN` và `dryNN`

| Token | Công thức | Trộn lại cái gì |
|---|---|---|
| `blendNN` | `(1 − NN/100) × hiện tại + (NN/100) × đầu vào cả chuỗi` | Tín hiệu **gốc đầu chuỗi** |
| `dryNN` | `(1 − NN/100) × hiện tại + (NN/100) × đầu vào của bước lọc ngay trước nó` | Thứ mà bộ lọc ngay trước đó **được đưa vào** |

Vì sao có `dryNN`: ở tầng 2, `blendNN` sẽ trộn lại **sóng gốc**, đưa lại cả nhiễu mà tầng 1 đã khử. `dryNN` chỉ làm tầng 2 nhẹ tay so với đầu ra tầng 1. Ví dụ `half>blend50>half>dry50` (bộ lọc giả nhân 0.5): tầng 1 ra 0.75x, tầng 2 nhận 0.75x và ra 0.375x, `dry50` ra 0.5625x. Đã có test cho các trường hợp này (`tests/test_denoise_pseudo.py`).

Mức hạ nhiễu tối đa của một lần trộn (cận trên suy từ phép trộn biên độ, giả định hai tín hiệu thẳng hàng, **chưa đo**): nhiễu còn lại ≥ `NN/100` × nhiễu đầu vào, nên hạ nhiều nhất `−20·log10(NN/100)` dB: `25` ≈ 12 dB, `50` ≈ 6 dB, `75` ≈ 2.5 dB.

### Chạy trên DGX (chỉ tới bước denoise)

```bash
export REUSE_PYTHON=/home/lamkd2/miniforge3/envs/reuse/bin/python
export REUSE_DIR=~/NemotronLabs_REUSE
python tools/run_variants.py --files /home/lamkd2/data/data_input/data_movie/a1.mp3 \
  --variants tools/variants_two_stage.json --only-available \
  --out-root /home/lamkd2/data/denoise_two_stage --cache-dir /home/lamkd2/data/denoise_two_stage_cache \
  --env a100 --stop-after denoise --gpus 0 --parallel 1
```

- **Dùng `--out-root` mới.** Khi dùng `--only-ids`, runner ghi lại `variants_manifest.json` chỉ với các biến thể được chọn, làm mất mục của các biến thể cũ trong cùng thư mục.
- Chạy đúng **một** runner (kiểm tra `jobs`), không ngắt bằng `^C` khi đang chạy.
- `--only-available` bỏ qua biến thể nào thiếu môi trường và nói lý do.
- Ước lượng thời gian (đoán): vài chục phút cho 16 tổ hợp, lâu nhất là những tổ hợp có `re_use` (lần chạy trước 256 s) và `rf_crowd`.

### Chỉnh độ mạnh theo từng mô hình tầng 2

| Mô hình | Núm chỉnh riêng | Ghi chú |
|---|---|---|
| DFN3 | `atten_lim_db` (giới hạn mức hạ, dB; mặc định không giới hạn). `post_filter` tắt, bật thì mạnh hơn | **Chưa đặt được từ dòng lệnh** (cấu hình từng bộ lọc lấy từ khóa cấp trên cùng `denoise` của config, mà `--set` ghi vào bên trong profile). **Bẫy cache:** tên cache chỉ gồm tên chuỗi, đổi tham số vẫn nạp bản cũ. Cách sạch: thêm bí danh có mức cố định (`dfn3_lim6`, `dfn3_lim12`), **chưa làm** |
| BSRNN (`bsrnn26`) | Không có | Chỉ còn cách trộn |
| MP-SENet | Không có (có `ckpt_name`: `g_best_vb`/`g_best_dns`, là chọn checkpoint khác, chưa biết cái nào nhẹ hơn) | Chỉ còn cách trộn |
| RE-USE | Không có (mã NVIDIA chỉ có kích thước đoạn, bước nhảy, mở rộng băng tần) | Chỉ còn cách trộn. Có `re_use_laughter` là checkpoint khác |

## 3. Bản nâng cấp theo từng mô hình (tra cứu 2026-10-23)

| Tầng | Kết quả |
|---|---|
| 1: MP-SENet | Không có bản kế nhiệm có trọng số do chính nhóm tác giả phát hành. Cùng họ mapping có **SEMamba** (RoyChao19477, MIT, `ckpts/vd.pth` huấn luyện VCTK-Demand và DNS-2020) và **MambAttention** (MIT, có checkpoint DNS 2020). Cả hai cần `mamba_ssm`. RE-USE được xây trên mã SEMamba, nên môi trường `reuse` có thể chạy SEMamba (chưa thử) |
| 1: MossFormer2_SE_48K | Không có mô hình SE kế nhiệm có tài liệu. Hugging Face `alibabasglab` có `a_frcrn_tflocoformer` (07/2025) và `EEYD_locoformer` (12/2025), README trống |
| 1: họ RoFormer | Khử vang mới của anvuew (`dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt` là bản nhẹ tay nhất), **giấy phép GPL-3.0**. Không thấy checkpoint bỏ đám đông hay bỏ nhiễu mới hơn |
| 2: DeepFilterNet3 | **Không có DFN4.** Kho Rikorose/DeepFilterNet không có release, commit mới nhất 2024-10-17. Không có mô hình nào được chứng minh vượt DFN3 trong cùng một bảng đo. FastEnhancer 48 kHz (MIT) chỉ đo trên VoiceBank-DEMAND |
| 2: BSRNN | Hạng 1 URGENT 2025 (BSRNN-FAN), USEMamba, FUSE: **không có mã hay trọng số**. Kho URGENT 2026 chỉ có `bsrnn.ckpt` (151 MB) và `flow_bsrnn.ckpt` (1.24 GB), Apache-2.0 |
| 2: RE-USE | **NemotronLabs RE-USE** (OpenMDW-1.1, ghi "ready for commercial use", ngày 2026-10-22) và **RE-USE-laughter** (cùng ngày, 9.6M tham số, giữ tiếng cười). Bản `nvidia/RE-USE` cũ và `Real-time_RE-USE` là giấy phép phi thương mại |

URGENT 2026 Track 1: hạng 1 khách quan là đội WR (GAP-URGENet, **không có mã hay trọng số**; đội WR bị loại khỏi xếp hạng chính thức vì đăng ký trùng). Xếp hạng chính thức: 1 baird, 2 subatomicseer, 3 Ali-Universal-SE, 4 RICK2000; không tìm thấy mã của các đội này.

## 4. Giấy phép cần xem lại

| Mục | Giấy phép | Ảnh hưởng |
|---|---|---|
| **DialogueSidon** (bộ tách chồng lấn đang dùng trong pipeline) | **CC-BY-NC-4.0** [đã xác minh] | Phi thương mại. Cần xem xét nếu dataset dùng thương mại |
| anvuew dereverb (Mel-Band) | GPL-3.0 | Cần ý kiến pháp lý |
| Sucial dereverb-echo | CC BY-NC-SA 4.0 | Phi thương mại |
| nvidia/RE-USE cũ, Real-time_RE-USE | NSCLv1, chỉ nghiên cứu | Không dùng cho dataset thương mại |
| NemotronLabs RE-USE | OpenMDW-1.1 | Mới đọc dòng "ready for commercial use" trên trang, **chưa đọc văn bản giấy phép** |
| Sidon v0.1, UniPASE, Resemble Enhance, AnyEnhance | MIT / (UniPASE: GitHub ghi MIT, HF ghi Apache-2.0) | Trọng số của WavLM nền chưa kiểm tra |

## 5. Tầng 3 (làm sau)

**Mục tiêu:** một lượt enhance/phục hồi **sau** hai tầng khử nhiễu để cải thiện thêm chất lượng giọng. Chủ dự án chấp nhận enhance miễn tách được giọng, nhưng đầu ra vẫn đi vào ASR, diarization và dataset, nên ảo giác (bịa từ, đổi giọng, bịa nội dung vào chỗ im lặng hay chỗ mất gói) vẫn là rủi ro chính.

### Ứng viên

| Mô hình | Loại | Ghi chú | Ưu tiên |
|---|---|---|---|
| **UniPASE** (Xiaobin-Rong; [mã](https://github.com/Xiaobin-Rong/unipase), [trọng số](https://huggingface.co/Xiaobin-Rong/unipase)) | Sinh: DeWavLM-Omni, bộ chuyển đổi, vocoder 16 kHz, PostNet lên 48 kHz | 545.7M tham số, 79.2 GMACs/s. Số liệu bài báo (DNS 2020 không vang): giống âm vị 0.97, giống giọng 0.96, sai từ 2.17%; có vang sai từ 8.16%. WavLM chỉ tiền huấn luyện tiếng Anh. **Cờ `--enable_plc` mặc định bật và có thể bịa nội dung vào chỗ mất gói; cờ khai báo `type=bool` nên `--enable_plc False` vẫn là True, phải truyền chuỗi rỗng `--enable_plc ""`.** Mã tự chuẩn hóa đỉnh đầu ra theo đỉnh đầu vào | Cao |
| **Sidon v0.1** (sarulab-speech, MIT) | Sinh nhẹ: w2v-BERT 2.0 + vocoder, 48 kHz | Theo bài báo ~250M tham số, huấn luyện 104 ngôn ngữ. WER/CER sau phục hồi nhỉnh hơn đầu vào nhiễu (ví dụ 0.045 so với 0.040), tức mất ít nội dung. Không có bằng chứng tiếng Việt, không có phân tích ảo giác. **Trang Hugging Face không có mã chạy mẫu**, chưa rõ cách gọi suy luận | Cao |
| Resemble Enhance (MIT) | Sinh, bộ khử nhiễu + enhancer, 44.1 kHz | Chỉ có bằng chứng tiếng Anh. Dùng làm đối chứng | Vừa |
| AudioSR (checkpoint `speech`) | Khuếch tán, mở rộng băng tần | Giấy phép trọng số chưa rõ. Chỉ khi cần băng tần cao, đặt sau khi đã sạch | Thấp |
| AnyEnhance (checkpoint nền, MIT, 45.68M) | Sinh mặt nạ | Cần môi trường Python 3.9 riêng; không chắc là bản mạnh nhất | Thấp |

**Loại:** Miipher / Miipher-2 (không có trọng số công khai), FINALLY (không thấy trọng số), Apollo (nhạc), FlowHigh (liên kết 404), GAP-URGENet (không công bố), SAM-Audio (bị khóa, giấy phép riêng).

### Rào chắn chống ảo giác (đề xuất)

1. So ASR trước và sau theo từng đoạn; đoạn nào lệch quá ngưỡng thì dùng bản trước (ngưỡng cần hiệu chỉnh trên dữ liệu thật, ví dụ 5 đến 10%).
2. So độ giống giọng trước và sau; dưới ngưỡng thì dùng bản trước.
3. Giữ im lặng ở chỗ tầng 2 coi là im lặng (cổng năng lượng hoặc VAD); với tiếng đệm, thở, cười, đo thêm độ lệch năng lượng và phổ so với đầu vào.
4. Trộn lại đầu vào theo tỉ lệ (ví dụ 20 đến 40%).
5. Chỉ áp dụng ở đoạn một người nói, xử lý theo đoạn ngắn.

### Vị trí trong pipeline (vấn đề thiết kế chưa giải quyết)

Chuỗi khử nhiễu hiện tại chạy **trước diarization, trên bản trộn hai người**. Agent nghiên cứu khuyên chạy tầng 3 **trên từng kênh sau bước tách chồng lấn** (Sidon chưa được huấn luyện cho bản trộn hai người). Nghĩa là tầng 3 là **một stage mới**, không phải thêm một bước vào chuỗi `--denoise`. DialogueSidon (24 kHz) cũng đã tái tạo các đoạn chồng lấn bằng mô hình sinh, nên thêm một lần tái tạo nữa lên cùng đoạn đó cần cân nhắc.

### Việc cần làm khi bắt đầu tầng 3

- [ ] Chạy thử UniPASE bằng mã chính thức trên đoạn 60 giây (môi trường `unipase`, Python ghi trong README là chưa rõ; `requirements.txt` ghim `torch==2.6.0+cu118`, `espnet==202511`, `numpy==2.5.1`...). Gửi lỗi nếu `pip install` dừng.
- [ ] Tìm mã suy luận chính thức của Sidon v0.1 (trang HF chỉ trỏ tới Space demo).
- [ ] Quyết định vị trí stage (từng kênh sau tách chồng lấn, hay sau bước xuất).
- [ ] Đo WER, tỉ lệ chèn, độ giống giọng, bảo toàn tiếng đệm/thở/cười trên tiếng Việt thật; chưa có bằng chứng tiếng Việt cho mô hình nào ở đây.
- [ ] Xem xét vấn đề giấy phép DialogueSidon (mục 4).

## 6. Chưa xác minh

- Mọi bằng chứng tiếng Việt hoặc thanh điệu cho các mô hình trên (RE-USE chỉ có Quan Thoại làm đại diện).
- Ảnh hưởng của OpenMDW-1.1 lên dataset phát hành (chưa đọc văn bản).
- Ngày cập nhật, kích thước của hầu hết checkpoint (trang không hiển thị).
- Tên kho `NVIDIA-NemotronLabs-RE-USE-laughter` đã được mở trang xác nhận tồn tại, nhưng **bố cục thư mục** (có giống kho RE-USE thường không) chưa kiểm tra.
- `SEMamba.from_pretrained(thư_mục_cục_bộ)` trong worker RE-USE (mới chạy được mã chính thức bằng id kho).

## 7. Nguồn

- RE-USE: <https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE>, <https://arxiv.org/abs/2603.02641>, <https://arxiv.org/abs/2606.25621>
- URGENT 2026: <https://arxiv.org/html/2601.13531v1>, GAP-URGENet <https://arxiv.org/abs/2604.01832>, Alibaba hybrid <https://arxiv.org/html/2601.19113>
- UniPASE: <https://arxiv.org/html/2604.14606v1>, <https://github.com/Xiaobin-Rong/unipase>
- Sidon: <https://arxiv.org/html/2509.17052v2>, <https://huggingface.co/sarulab-speech/sidon-v0.1>; DialogueSidon: <https://huggingface.co/sarulab-speech/DialogueSidon>, <https://arxiv.org/pdf/2604.09344>
- SEMamba: <https://github.com/RoyChao19477/SEMamba>; MambAttention: <https://github.com/NikolaiKyhne/MambAttention>
- DeepFilterNet: <https://github.com/Rikorose/DeepFilterNet>; FastEnhancer: <https://github.com/aask1357/fastenhancer>
- Khử vang anvuew: <https://huggingface.co/anvuew/dereverb_mel_band_roformer>
- Resemble Enhance: <https://github.com/resemble-ai/resemble-enhance>; AudioSR: <https://github.com/haoheliu/versatile_audio_super_resolution>; AnyEnhance: <https://github.com/viewfinder-annn/anyenhance-v1-ccf-aatc>
