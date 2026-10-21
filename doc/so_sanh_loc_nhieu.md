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

`--only-available`: trước khi chạy, hỏi `available_backends()` về mọi backend trong chuỗi; biến thể nào có backend không dùng được thì **bỏ qua có ghi lại** (status `skipped`, lý do trong `variants_manifest.json`, dòng `SKIPPED` trên console, ghi chú trên chip của trang). Bỏ qua không làm mã thoát khác 0. `--groups masking,chains` chỉ chạy các nhóm được chọn (`group` là khóa tùy chọn của mỗi biến thể trong file JSON; baseline luôn chạy; tên nhóm lạ thì báo lỗi). Nhóm hiện có trong `variants.example.json`: `masking`, `chains`, `analysis_only`, `ctrl_music`, `pipe_music`, `pipe_reverb`, `pipe_crowd`, `pipe_order`, `pipe_stack`, `pipe_blend`, `analysis_pipes` (xem mục chuỗi nhiều giai đoạn). Mỗi biến thể có thể có `note` (hiện ở tooltip của chip) và nhóm hiện thành thẻ nhỏ cạnh chip.

Khi tự gõ `--denoise` trong shell, phải đặt chuỗi trong nháy: `--denoise 'dfn3>mf2se48'`. Cấu hình riêng từng backend đặt ở khóa cấp cao nhất `"denoise": {"dfn3": {...}, "chunk_s": 30}` của config.json (không phải `pipeline.denoise`, đó chỉ là chuỗi SPEC).

## Chuỗi nhiều giai đoạn (multi-stage)

### Backend `audio-separator` và alias

Backend chung `audio_separator` (khóa `checkpoint`, `stem_keep`, `invert`) cùng 7 alias. Đây là bộ tách nhạc/giọng/vang, không phải bộ tăng cường giọng nói: đầu ra làm đổi tín hiệu nên phải qua cổng kiểm chứng trước khi xuất. Cài: `requirements-denoise-audio-separator.txt`, chi tiết ở `podcast-pipeline/scripts/denoise_audio_separator.md`.

| Alias | Checkpoint | Stem giữ lại | Giấy phép / ghi chú |
|---|---|---|---|
| `rf_vocals_3005` | model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt | `Vocals` | không nêu giấy phép; chính là RoFormer của bước nhạc profile a100 |
| `rf_vocals_kim` | mel_band_roformer_kim_ft2_bleedless_unwa.ckpt | `vocals` | không nêu giấy phép |
| `rf_denoise` | denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt | `dry` | không nêu giấy phép, không có model card |
| `rf_denoise_aggr` | denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt | `dry` | mạnh hơn; có thể ăn phụ âm nhẹ (giả định) |
| `rf_dereverb` | deverb_bs_roformer_8_384dim_10depth.ckpt | `noreverb` | có thể GPL-3.0 nếu là bản của anvuew (tác giả chưa xác minh) |
| `uvr_deecho` | UVR-DeEcho-DeReverb.pth | `No Reverb` | kiến trúc VR; căn thời gian và mức âm chưa kiểm |
| `rf_crowd` | mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt | `other` (= đầu vào trừ đám đông) | không nêu giấy phép; làm cho tiếng vỗ tay, tác dụng với người đứng cạnh nói chuyện CHƯA KIỂM |

Bằng chứng: danh sách mô hình của audio-separator 0.47.1. Không checkpoint nào có giấy phép được nêu (mã UVR là MIT, trọng số thì không), và **không cái nào được kiểm chứng trên tiếng Việt**.

### Bước giả `blendNN`

`blendNN` (NN từ 0 đến 100) không phải backend, do `DenoiseService` xử lý: `ra = (1 - NN/100) * hiện_tại + (NN/100) * đầu_vào_của_chuỗi`. "Đầu vào của chuỗi" là sóng âm đi vào cả chuỗi (cũng đúng khi blend nằm giữa hai backend). Không được đứng đầu chuỗi và chuỗi phải có ít nhất một backend. `blend0` giữ nguyên kết quả lọc, `blend100` trả đầu vào. Không đăng ký trong `BACKENDS`; `--only-available` bỏ qua chúng qua `models.denoise.pseudo.backend_tokens`. `denoise.json` có thêm `steps` và `blend`. Cơ sở ý tưởng: Iwamoto 2022 (arXiv 2201.06685), Kimi-Audio trộn 1:1.

### `--set` và `--music_scope`

```bash
python main.py ... --set steps.music_removal=false            # giá trị là JSON; không phải JSON thì là chuỗi
python main.py ... --set pipeline.music_scope='"spans"' --set steps.noise_map=false
python main.py ... --music_scope spans                         # tương đương --set pipeline.music_scope=spans
```

`--set KEY=VALUE` (lặp được) ghi đè khóa chấm trong profile như thể có trong config.json (`utils/config_override.py`).

### Trường của mỗi biến thể

| Trường | Ý nghĩa |
|---|---|
| `chain` | chuỗi SPEC, ví dụ `rf_vocals_3005>dfn3>blend50` |
| `id` | tùy chọn; mặc định suy ra từ chuỗi, thêm `.<hậu tố>` khi có ghi đè (ví dụ `none.music-removal-off`; quá dài thì dùng hash8) |
| `tag` | nhãn ngắn, đi vào manifest, `compare_data.json` và hiện thành thẻ trên trang |
| `set` | object các khóa chấm sang giá trị JSON, thành `--set` |
| `music_separator` | thành `--music_separator` (checkpoint của bước nhạc) |
| `extra_args` | danh sách đối số thêm cho main.py |
| `group`, `label`, `note` | như trước |

### CẢNH BÁO: chuỗi chứa `rf_vocals_*` dựa vào việc tắt bước nhạc

Profile a100 đã chạy RoFormer ở bước nhạc (`music_removal` bật, `music_scope` full). Chuỗi có `rf_vocals_*` mà không tắt bước đó sẽ áp RoFormer **hai lần**. Vì vậy mọi biến thể loại này có `"set": {"steps.music_removal": false}` và `"tag": "music-stage-off"`, và `tests/test_variants_integrity.py` kiểm điều đó (ngoại lệ phải có `double-music` trong tag). Khi tự thêm biến thể, nhớ đặt như vậy.

### Các họ biến thể

Tổng 58 biến thể (đã gồm baseline).

| Nhóm | Số | Mục đích | Nhìn gì trong kết quả |
|---|---|---|---|
| `baseline` / `masking` / `chains` | 1 / 5 / 3 | không lọc; từng bộ lọc mặt nạ; hai giai đoạn | nền so sánh; sai khác khi nghe, tạo tác khi xếp chồng |
| `analysis_only` | 3 | dbs_tiger, mambattention (chỉ phân tích) | chỉ nghe; không xuất |
| `ctrl_music` | 5 | đối chứng bước nhạc của pipeline: raw (tắt hẳn), `music_scope=spans`, separator kim | bước nhạc tự nó đóng góp gì (nền nhiễu, WER so baseline); spans so với full |
| `pipe_music` | 10 | bỏ nhạc bằng chuỗi (bước nhạc tắt), thêm 1-2 bộ lọc phía sau | còn nhạc không; WER so baseline; độ tương đồng giọng |
| `pipe_reverb` | 7 | khử vang/echo, đứng riêng hoặc sau `rf_vocals_3005` | vang giảm không mà giọng không bị khô/méo; độ lệch thời gian (`lag_samples`) |
| `pipe_crowd` | 6 | giọng nền/đám đông (tác dụng với người nói chuyện bên cạnh CHƯA KIỂM) | nghe xem tiếng nền có giảm không, giọng chính có bị cắt không; thứ tự crowd trước/sau |
| `pipe_order` | 4 | thứ tự có quan trọng không | hai thứ tự cùng bộ lọc cho khác biệt WER/Learned MOS bao nhiêu |
| `pipe_stack` | 4 | tạo tác tích lũy khi xếp chồng bộ lọc mặt nạ, kể cả lọc hai lần cùng mô hình | delta Learned MOS hoặc độ tương đồng giọng có giảm khi thêm giai đoạn không |
| `pipe_blend` | 7 | trộn lại đầu vào chưa lọc (`blend25/30/50`) | WER so baseline có tiến về baseline trong khi nền nhiễu (`noise_floor_db_after`) vẫn giảm không |
| `analysis_pipes` | 3 | chuỗi thử nghiệm với dbs_tiger/mambattention (chỉ phân tích) | chỉ nghe; chưa xuất trước khi nghe thử |

### Ước lượng quy mô: `--list`

```bash
python tools/run_variants.py --audio-dir /data/vi_podcasts_20 --variants tools/variants.example.json \
    --out-root /data/denoise_cmp --only-available --list --minutes-per-file-run 20
```

`--list` in từng biến thể (id, nhóm, chuỗi, tag, ghi đè, khả dụng), số biến thể, số file, số lượt biến thể x file và ước lượng tuần tự. **20 phút cho mỗi lượt (biến thể, file) là GIẢ ĐỊNH, chưa đo**; đổi bằng `--minutes-per-file-run`. Thu nhỏ bằng `--groups`, `--only-ids a,b`, `--max-variants N` (có `--shuffle-seed S` để lấy ngẫu nhiên lặp lại được; tính cả baseline). Với 3 file, cả 58 biến thể là 174 lượt, tức 3480 phút (58 giờ) tuần tự theo giả định đó, chia theo `--parallel`/`--gpus`.

### Chạy theo giai đoạn (khuyến nghị)

```bash
# 1) trước hết: đối chứng bước nhạc + masking + blend, trên 3 file
python tools/run_variants.py --audio-dir /data/vi_podcasts_3 --variants tools/variants.example.json \
    --out-root /data/denoise_cmp1 --env a100 --stop-after asr --gpus 0,1 --parallel 2 --resume \
    --only-available --groups ctrl_music,masking,pipe_blend
# 2) sau đó các nhóm còn lại
python tools/run_variants.py --audio-dir /data/vi_podcasts_3 --variants tools/variants.example.json \
    --out-root /data/denoise_cmp2 --env a100 --stop-after asr --gpus 0,1 --parallel 2 --resume \
    --only-available --groups pipe_music,pipe_reverb,pipe_crowd,pipe_order,pipe_stack,chains,analysis_only,analysis_pipes
```

### Nhắc nhở

- Backend lỗi thì **rơi về âm thanh chưa lọc**; `run_variants.py` giờ tự đọc `01b_denoise/denoise.json` và đánh dấu biến thể `degraded` (xem mục dưới), không còn để status `ok`.
- Âm thanh xuất ra phải qua các cổng kiểm chứng (giữ danh tính người nói, WER ASR so baseline, không khẳng định xóa giọng đám đông).
- Biến thể `analysis_only` và `analysis_pipes` không được xuất trước khi nghe thử.
- Nhóm `pipe_crowd`, `pipe_reverb` chưa kiểm chứng trên tiếng Việt; tên checkpoint và stem lấy từ danh sách mô hình, chưa chạy thật.

## Biến thể lỗi lặng lẽ (`degraded`)

Khi backend không nạp được hoặc chạy lỗi, `DenoiseService` giữ nguyên âm thanh chưa lọc và chỉ ghi lỗi vào `01b_denoise/denoise.json` (khóa `errors`). Sau mỗi biến thể, `run_variants.py` đọc file này cho từng file đầu vào và gán một verdict:

| verdict | ý nghĩa |
|---|---|
| `ok` | chuỗi là `none`, hoặc không có lỗi |
| `fallback` | `errors` khác rỗng: chuỗi bị bỏ, giữ nguyên đầu vào |
| `missing` | chuỗi khác `none` nhưng không có (hoặc hỏng) `denoise.json`, hoặc không có thư mục đầu ra của file: bước denoise chưa chạy hay đã sập |
| `suspect_noop` | không có lỗi nhưng `corr_with_input` >= 0.99999 với backend thật (không phải `identity`, không phải `blendNN`): bộ lọc không đổi gì, đáng ngờ |

Status của biến thể là `degraded` (ngoài `ok/failed/skipped/missing`) khi tiến trình thoát 0 nhưng có file `fallback` hoặc `missing`. `suspect_noop` một mình không đổi status, chỉ in cảnh báo và ghi vào manifest.

- `variants_manifest.json` có thêm `denoise_report` (`{stem: {verdict, errors, corr_with_input, lag_samples}}`), `degraded_files`, `suspect_files`, `total_files`.
- Cuối lần chạy in khối `DEGRADED` liệt kê id biến thể, số file ảnh hưởng và dòng lỗi đầu tiên.
- Biến thể `degraded` **không** ghi `.done`, nên `--resume` sẽ chạy lại sau khi sửa môi trường. `--accept-degraded` vẫn ghi `.done` (status trong manifest vẫn là `degraded`).
- `--strict`: thoát với mã 3 nếu có biến thể `degraded` hoặc `failed`. Mặc định mã thoát không đổi (`degraded` không làm mã khác 0).
- Trang so sánh: huy hiệu đỏ trên chip ("fell back to unfiltered audio in N/M files"), huy hiệu vàng cho `suspect_noop`, banner đầu trang, dòng cảnh báo ở tab Text/Speakers/Metrics, và ghi chú `summary_n` / `summary_excluded_degraded` ở tab Metrics (trung bình vẫn tính cả file `fallback`, không bỏ lặng lẽ).

**Biến thể `degraded` không có nghĩa là "bộ lọc không có tác dụng"**: âm thanh của nó chính là bản chưa lọc, nên mọi chênh lệch (hay không chênh lệch) so với baseline không nói gì về bộ lọc. Sửa lỗi (xem `errors` hoặc `logs/<id>.log`) rồi chạy lại với `--resume` trước khi kết luận.

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

- Backend `audio_separator` và 7 alias chạy với checkpoint thật; `blendNN` và các chuỗi nhiều giai đoạn trên GPU thật (test chỉ dùng backend giả/nhẹ); nghe thử mọi họ biến thể mới.
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

## Xử lý sự cố: lần chạy "đứng im" ở dòng `start`

`run_variants.py` chỉ in dòng bắt đầu và dòng kết thúc của mỗi biến thể; mọi tiến trình thật nằm trong `OUT/logs/<id>.log`. Xem `tail -f` file đó.

Nếu log chỉ có đúng dòng lệnh, `[bootstrap] main environment OK` và `Output folder (clean two-channel data):`, thì `main.py` đang chờ bạn trả lời câu hỏi tương tác (`utils/env_bootstrap.ask_paths` hỏi thư mục khi stdin là terminal). Các phiên bản trước của bộ chạy để tiến trình con dùng chung terminal nên nó chờ Enter ở cửa sổ chạy lệnh. Từ commit sửa lỗi này, tiến trình con nhận `stdin=/dev/null` (không còn hỏi) và mọi lệnh đều kèm `--set outputs.clean_two_channel_dir=""` để lần chạy so sánh không ghi vào thư mục dữ liệu sạch dùng chung. Một biến thể vẫn có thể ghi đè bằng `set` riêng. Nếu thiếu `HF_TOKEN` mà một mô hình cần, tiến trình con sẽ báo lỗi ngay thay vì chờ nhập.

Chọn GPU: `--gpus` gán MỘT GPU cho mỗi biến thể (vòng tròn). Profile a100 ánh xạ model vào hai GPU (`gpu_1: 0`, `gpu_2: 1`), nên với một lần chạy riêng lẻ hãy bỏ `--gpus` và đặt `CUDA_VISIBLE_DEVICES=2,3 python tools/run_variants.py ...` trên hai GPU còn trống.

### Chọn đúng trình thông dịch (`--python`) và kiểm tra trước khi chạy

Tiến trình con chạy bằng `--python` (mặc định là Python đang chạy `run_variants.py`). Nếu bạn đang ở trong môi trường khác (ví dụ một venv của dự án khác kích hoạt chồng lên môi trường pipeline), `main.py` sẽ chạy nhầm môi trường và hỏng khi nhập thư viện, ví dụ `PyTorch has CUDA version 12.6 whereas TorchAudio has CUDA version 12.8`. Trước khi bắt đầu, bộ chạy nhập `torch,torchaudio` (đổi bằng `--preflight-modules`) trong trình thông dịch đó; nếu lỗi nó dừng với mã 2 và in nguyên nhân. Dòng `[preflight] ok: <python> torch=... torchaudio=...` cho biết trình thông dịch nào sẽ chạy. Bỏ qua bằng `--no-preflight`. Môi trường đúng là môi trường nơi `python -c 'import torch, torchaudio'` chạy được với `requirements.txt` (torch 2.8.0 và torchaudio 2.8.0); truyền `--python /đường/dẫn/env/bin/python`.
