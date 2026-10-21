# So sánh bộ lọc nhiễu (denoise) trên pipeline

## Đây là gì

Một bước tùy chọn `--denoise SPEC` và một bộ công cụ để so sánh nhiều biến thể lọc nhiễu trên cùng tập file.

- Bước lọc chạy **sau khi bỏ nhạc, trước diarization**. `SPEC` là `none` (mặc định, pipeline không đổi gì), một backend (`dfn3`, `mf2se48`, `mpsenet`) hoặc chuỗi nối tiếp `dfn3>mf2se48`. Backend thử nghiệm: `identity`, `fake_hp`.
- Diarization, separation, ASR và các bước sau đều dùng âm thanh đã lọc (lấy lại từ cache ở mỗi lượt của chế độ stage-major).
- Đầu ra: `<save_path>/<tên file>/01b_denoise/{after_denoise.wav,denoise.json}`. `01_music/after_music.wav` vẫn là âm thanh **trước** khi lọc.
- Độ dài không đổi (backend được ép đúng số mẫu, bù trễ nếu cấu hình). Chuỗi nào lỗi, ra sai độ dài, NaN hoặc câm đều bị bỏ cả chuỗi, giữ âm thanh gốc, lỗi ghi trong `denoise.json["errors"]`.
- Cache `denoised_audio` có namespace `<chain_id>-<namespace processed_audio>`; đổi chuỗi thì tính lại. Không có `--denoise` thì không đổi tên checkpoint, namespace hay file đầu ra.

Công cụ so sánh: `tools/run_variants.py` (chạy file x biến thể, mỗi biến thể một tiến trình `main.py` riêng, `--job_id exp-<id>`, `--save_path` riêng), `tools/build_compare_data.py` (gộp thành `compare_data.json`, xem `tools/compare_lib/SCHEMA.md`), `tools/make_compare_page.py` (trang HTML một file).

## Lệnh trên DGX

```bash
cd podcast-pipeline
pip install -r requirements-denoise.txt          # chỉ cần khi dùng dfn3/mf2se48/mpsenet
# tải trọng số theo scripts/denoise_weights.md; kiểm tra:
python -c "from models.denoise import available_backends as a; [print(k, v) for k, v in a().items()]"

python tools/run_variants.py --audio-dir /data/vi_podcasts_20 \
    --variants tools/variants.example.json --out-root /data/denoise_cmp \
    --env a100 --stop-after asr --gpus 0,1 --parallel 2 --resume \
    --reference-dir /data/vi_podcasts_20/ref          # --dry-run để chỉ in lệnh

python tools/build_compare_data.py --manifest /data/denoise_cmp/variants_manifest.json \
    --out /data/denoise_cmp/compare_data.json         # thêm --no-embed nếu không có WESPEAKER_MODEL_PATH
python tools/make_compare_page.py /data/denoise_cmp/compare_data.json \
    -o /data/denoise_cmp/compare.html --embed clips
```

Khi tự gõ `--denoise` trong shell, phải đặt chuỗi trong nháy: `--denoise 'dfn3>mf2se48'`. Cấu hình riêng từng backend đặt ở khóa cấp cao nhất `"denoise": {"dfn3": {...}, "chunk_s": 30}` của config.json (không phải `pipeline.denoise`, đó chỉ là chuỗi SPEC).

## Các tab của trang

- **Text**: so transcript mỗi biến thể với baseline theo từng đoạn, từ sai/thừa/thiếu, WER so với baseline (và so với bản tham chiếu nếu có), nút nghe đoạn gốc và đoạn đã lọc.
- **Speakers**: dải thời gian người nói của baseline và biến thể, các khoảng bất đồng (`swap`, `missing`, `extra`, `overlap`), số người nói, `der_vs_baseline`.
- **Metrics**: nhóm audio, diarization, separation, asr; bảng tổng (trung bình qua các file) và theo từng file, kèm chiều "tốt hơn" khi có.

## Đọc số liệu

Không có ground truth thì **"khác" không có nghĩa là "tốt hơn"**. `wer_vs_baseline` là khoảng cách tới transcript của baseline, không phải tỉ lệ lỗi; baseline cũng có thể sai. Chỉ `wer_vs_reference` (cần bản chép tay) đo đúng/sai. `noise_floor_db` thấp hơn có thể là lọc sạch hoặc là gating quá tay (xem `gated_frame_pct`, `words_in_quiet_frames`). `lag_samples` khác 0 nghĩa là mô hình làm trễ tín hiệu. `corr_with_raw` gần 1 là bộ lọc gần như không đổi gì. Nên nghe lại các đoạn bất đồng trước khi kết luận.

## Chưa kiểm chứng (UNTESTED)

- Backend thật `dfn3`, `mf2se48`, `mpsenet` (chưa chạy với mô hình/trọng số thật; chỉ test bằng backend giả).
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
