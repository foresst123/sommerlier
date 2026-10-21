# So sánh bộ lọc nhiễu (denoise) trên pipeline

## Đây là gì

Một bước tùy chọn `--denoise SPEC` và một bộ công cụ để so sánh nhiều biến thể lọc nhiễu trên cùng tập file.

- Bước lọc chạy **sau khi bỏ nhạc, trước diarization**. `SPEC` là `none` (mặc định, pipeline không đổi gì), một backend (xem bảng dưới) hoặc chuỗi nối tiếp `dfn3>mf2se48`. Backend thử nghiệm: `identity`, `fake_hp`.
- Diarization, separation, ASR và các bước sau đều dùng âm thanh đã lọc (lấy lại từ cache ở mỗi lượt của chế độ stage-major).
- Đầu ra: `<save_path>/<tên file>/01b_denoise/{after_denoise.wav,denoise.json}`. `01_music/after_music.wav` vẫn là âm thanh **trước** khi lọc.
- Độ dài không đổi (backend được ép đúng số mẫu, bù trễ nếu cấu hình). Chuỗi nào lỗi, ra sai độ dài, NaN hoặc câm đều bị bỏ cả chuỗi, giữ âm thanh gốc, lỗi ghi trong `denoise.json["errors"]`.
- Cache `denoised_audio` có namespace `<chain_id>-<namespace processed_audio>`; đổi chuỗi thì tính lại. Không có `--denoise` thì không đổi tên checkpoint, namespace hay file đầu ra.

Công cụ so sánh: `tools/run_variants.py` (chạy file x biến thể, mỗi biến thể một tiến trình `main.py` riêng, `--job_id exp-<id>`, `--save_path` riêng), `tools/build_compare_data.py` (gộp thành `compare_data.json`, xem `tools/compare_lib/SCHEMA.md`), `tools/make_compare_page.py` (trang HTML một file).

## Backend và vai trò

| Backend | Vai trò | Ghi chú |
|---|---|---|
| `dfn3`, `mf2se48`, `mpsenet`, `mpsenet_dns` | masking (khử nhiễu nền) | `mpsenet_dns` = cùng MP-SENet nhưng dùng `best_ckpt/g_best_dns` (repo có cả `g_best_vb` và `g_best_dns`, cùng config 16 kHz; README upstream không nói `g_best_dns` huấn luyện trên gì, tên gợi ý DNS Challenge, chưa xác nhận) |
| `bsrnn26` | masking | URGENT 2026 Track 1, BSRNN 37,8 M tham số, 8-48 kHz, không có tiếng Việt trong dữ liệu huấn luyện |
| `dbs_tiger` | **analysis_only** | tách stem speech/music/sfx (44,1 kHz, 1,0 M tham số, huấn luyện trên DnR v2 tiếng Anh); cắt nhạc/sfx, KHÔNG xóa giọng đám đông; có thể cắt nhầm tiếng cười, thở, ừ/à |
| `mambattention` | **analysis_only** | chạy ở Python env riêng (torch 2.2.2 + mamba_ssm) qua tiến trình con, 16 kHz |
| `identity`, `fake_hp` | thử nghiệm | dùng cho test |

Mọi backend `analysis_only` là để phân tích: **không xuất đầu ra làm audio huấn luyện** trước khi nghe thử và qua các cổng kiểm tra (danh tính người nói, WER so với baseline). Không backend nào ở đây đã được kiểm chứng trên tiếng Việt, và không backend nào xóa giọng người lạ (crowd voices) vì đều không phân biệt người nói chính.

### Cài đặt và biến môi trường

| Backend / chức năng | Biến môi trường | Cài đặt |
|---|---|---|
| `dfn3`, `mf2se48` | `DFN_MODEL_DIR`, `MF2SE48_DIR` | `requirements-denoise.txt`, xem `scripts/denoise_weights.md` |
| `mpsenet`, `mpsenet_dns` | `MPSENET_DIR` (clone yxlu-0102/MP-SENet), `MPSENET_CKPT` (tùy chọn) | `requirements-denoise.txt` |
| `bsrnn26` | `URGENT26_DIR`, `URGENT26_BSRNN_CKPT` | `requirements-denoise-bsrnn.txt`, `scripts/denoise_bsrnn_urgent.md` |
| `dbs_tiger` | `DBS_TIGER_DIR`, `DBS_TIGER_CKPT` | `requirements-denoise-dbs-tiger.txt`, `scripts/denoise_dbs_tiger.md` |
| `mambattention` | `MAMBATTENTION_PYTHON`, `MAMBATTENTION_DIR` | env riêng, `requirements-denoise-mambattention.txt`, `scripts/denoise_mambattention.md` |
| Learned MOS (Uni-VERSA-Ext) | `UNIVERSA_MODEL`, `UNIVERSA_CONFIG`, `UNIVERSA_REPO` | tùy chọn; không đặt thì cột để trống |
| `ecapa_cos` | `WESPEAKER_MODEL_PATH` | file ONNX WeSpeaker; không có thì để trống |

## Lệnh trên DGX

```bash
cd podcast-pipeline
pip install -r requirements-denoise.txt          # chỉ cần khi dùng dfn3/mf2se48/mpsenet
# tải trọng số theo scripts/denoise_weights.md; kiểm tra:
python -c "from models.denoise import available_backends as a; [print(k, v) for k, v in a().items()]"

# MỘT LỆNH chạy cả kế hoạch (baseline + masking + chains + analysis_only):
python tools/run_variants.py --audio-dir /data/vi_podcasts_20 \
    --variants tools/variants.example.json --out-root /data/denoise_cmp \
    --env a100 --stop-after asr --gpus 0,1 --parallel 2 --resume --only-available \
    --reference-dir /data/vi_podcasts_20/ref          # --dry-run để chỉ in lệnh

python tools/build_compare_data.py --manifest /data/denoise_cmp/variants_manifest.json \
    --out /data/denoise_cmp/compare_data.json         # thêm --no-embed nếu không có WESPEAKER_MODEL_PATH
python tools/make_compare_page.py /data/denoise_cmp/compare_data.json \
    -o /data/denoise_cmp/compare.html --embed clips
```

`--only-available`: trước khi chạy, hỏi `available_backends()` về mọi backend trong chuỗi; biến thể nào có backend không dùng được thì **bỏ qua có ghi lại** (status `skipped`, lý do trong `variants_manifest.json`, dòng `SKIPPED` trên console, ghi chú trên chip của trang). Bỏ qua không làm mã thoát khác 0. `--groups masking,chains` chỉ chạy các nhóm được chọn (`group` là khóa tùy chọn của mỗi biến thể trong file JSON; baseline luôn chạy; tên nhóm lạ thì báo lỗi). Nhóm hiện có trong `variants.example.json`: `masking`, `chains`, `analysis_only`. Mỗi biến thể có thể có `note` (hiện ở tooltip của chip) và nhóm hiện thành thẻ nhỏ cạnh chip.

Khi tự gõ `--denoise` trong shell, phải đặt chuỗi trong nháy: `--denoise 'dfn3>mf2se48'`. Cấu hình riêng từng backend đặt ở khóa cấp cao nhất `"denoise": {"dfn3": {...}, "chunk_s": 30}` của config.json (không phải `pipeline.denoise`, đó chỉ là chuỗi SPEC).

## Các tab của trang

- **Text**: so transcript mỗi biến thể với baseline theo từng đoạn, từ sai/thừa/thiếu, WER so với baseline (và so với bản tham chiếu nếu có), nút nghe đoạn gốc và đoạn đã lọc.
- **Speakers**: dải thời gian người nói của baseline và biến thể, các khoảng bất đồng (`swap`, `missing`, `extra`, `overlap`), số người nói, `der_vs_baseline`.
- **Metrics**: nhóm audio, diarization, separation, asr; bảng tổng (trung bình qua các file) và theo từng file, kèm chiều "tốt hơn" khi có.

## Đọc số liệu

**Khối Learned MOS** (Uni-VERSA-Ext, 5 điểm xấp xỉ DNSMOS/UTMOS/NISQA/SCOREQ + MOS người, 16 kHz nên không thấy phần trên 8 kHz): chỉ nhìn **delta** (biến thể trừ bản gốc), không bao giờ dùng làm cổng. Bộ dự đoán này thưởng cho âm thanh trôi chảy kể cả khi mô hình bịa nội dung, và chưa được kiểm trên tiếng Việt. Chỉ tính khi bật `UNIVERSA_*` và build có embed (không `--no-embed`).


Không có ground truth thì **"khác" không có nghĩa là "tốt hơn"**. `wer_vs_baseline` là khoảng cách tới transcript của baseline, không phải tỉ lệ lỗi; baseline cũng có thể sai. Chỉ `wer_vs_reference` (cần bản chép tay) đo đúng/sai. `noise_floor_db` thấp hơn có thể là lọc sạch hoặc là gating quá tay (xem `gated_frame_pct`, `words_in_quiet_frames`). `lag_samples` khác 0 nghĩa là mô hình làm trễ tín hiệu. `corr_with_raw` gần 1 là bộ lọc gần như không đổi gì. Nên nghe lại các đoạn bất đồng trước khi kết luận.

## Đã chứng minh và chỉ là giả định

| Đã chứng minh (có test) | Chỉ là giả định (chưa chứng minh) |
|---|---|
| Import `models.denoise` không nạp torch hay thư viện nặng | Chất lượng của bất kỳ backend nào trên tiếng Việt |
| Bỏ qua có lý do, nhóm, ghi chú đi từ file JSON qua manifest tới trang | Backend nào xóa được giọng đám đông (không cái nào) |
| Độ dài không đổi, chuỗi lỗi giữ nguyên âm thanh gốc | Stem `dbs_tiger` giữ được tiếng cười/thở/ừ-à |
| Alias `mpsenet_dns` chọn đúng `g_best_dns` (bằng checkpoint giả) | Điểm Learned MOS tương quan với nghe thực tế |

## Chưa kiểm chứng (UNTESTED)

- Backend thật `dfn3`, `mf2se48`, `mpsenet`, `mpsenet_dns` (chưa chạy với mô hình/trọng số thật; chỉ test bằng backend giả; `g_best_dns` chỉ được xác nhận là file tồn tại trong repo clone).
- `dbs_tiger` nạp mô hình thật (chưa chạy checkpoint thật, chưa nghe thử stem trên tiếng Việt).
- `mambattention`: worker thật và bước build `mamba_ssm` (cần CUDA) chưa chạy; test chỉ dùng worker giả.
- `bsrnn26`: chạy trên GPU và vòng 44,1 kHz (resample vào/ra) chưa kiểm; bản CPU đã chạy được với checkpoint thật khi tích hợp.
- Learned MOS: suy luận thật với Uni-VERSA-Ext chưa chạy (test dùng bộ dự đoán giả).
- Chạy `run_variants.py --only-available` trên DGX với tất cả backend cùng lúc (test chỉ dùng main giả và kiểm tra khả dụng giả/thật trong môi trường dev).
- Một lần chạy pipeline thật có `--denoise` (chỉ test logic bằng fake, kể cả cache và stage-major).
- Phát audio trong trang trên trình duyệt thật (đã mở trang bằng dữ liệu tổng hợp, không có lỗi console; chưa bấm nghe).
- `ecapa_cos` cần `WESPEAKER_MODEL_PATH` trỏ tới file ONNX WeSpeaker; không có thì để trống.

## Hạn chế đã biết

- Mặc định bản đồ nhiễu (`noise_map`) và phân tích nhạc đo trên âm thanh **trước khi lọc** (chọn có chủ đích: chúng chạy trước bước lọc, và `after_music.wav` giữ nguyên). Chúng không phản ánh tác dụng của bộ lọc.
- Mọi biến thể tính lại phân tích nhạc từ đầu (mỗi biến thể một `job_id`); tốn thời gian nhưng không bao giờ dùng nhầm cache của biến thể khác.
- Bước lọc và `compare_lib` nạp cả file vào RAM (24 kHz float32 khoảng 345 MB/giờ, nhân vài bản sao). Chấp nhận được cho file vài giờ trên DGX; backend cắt khối (`chunk_s`) nên bộ nhớ GPU không tăng theo độ dài.
- `run_variants.py` dùng `os.killpg` và `start_new_session`, chỉ chạy trên Linux/macOS, không chạy trên Windows.
- Thời gian trong trang là trên timeline đã cắt nhạc, cùng hệ với các thư mục `0x_*`.
- Biến thể baseline không truyền `--denoise`; nếu profile đặt `pipeline.denoise` khác `none` thì baseline sẽ không còn là "không lọc".
