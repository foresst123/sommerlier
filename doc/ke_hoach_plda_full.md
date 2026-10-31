# Kế hoạch PLDA → full: lệnh chạy và tiêu chí qua

Chạy trên DGX, từ `~/sommerlier/podcast-pipeline`. Mỗi giai đoạn có lệnh, tiêu chí qua, và trạng thái code.

## Biến môi trường

```bash
cd ~/sommerlier/podcast-pipeline
export SOM_PY=/home/lamkd2/miniforge3/envs/som/bin/python
export DZ_PY=/home/lamkd2/diarizen_env/bin/python
export NEMOTRON_PY=/home/lamkd2/nemotron_env/bin/python
export VOX_ROOT=/home/lamkd2/data/external/voxceleb2/aac
export PLDA_ROOT=/home/lamkd2/data/plda
export OUT=/home/lamkd2/data/output/sitcom/final/lhotse/diar_arms_src30
```

## Trạng thái code

| Giai đoạn | File | Commit | Trạng thái |
|---|---|---|---|
| 0 Kiểm tra trước | `tools/preflight_check.py` | `589db37` | Đã có, 8 test pass |
| 1 Trích VoxCeleb2 | `tools/extract_voxceleb_embeddings.py` | `fbb6bba` | Đã có, 7 test pass |
| 2 Huấn luyện PLDA | `tools/train_plda.py` | `8879477` (có trước) | Đã có, không sửa |
| 3 So sánh PLDA | `tools/compare_plda_runs.py` | `d071f36` | Đã có, 5 test pass |
| 4 Embedder khác | — | — | Chưa làm, xem mục "Chưa làm" |
| 5 Chạy đầy đủ | `tools/variants.py` (có sẵn) | `d9c0a40` | Cần điền tham số, xem giai đoạn 5 |
| 6 Chia A/B | `tools/make_split.py`, `analyze_arms_detail.py --split` | `b62392f` | Đã có, 7 test pass |

## Giai đoạn 0: kiểm tra trước

```bash
python tools/preflight_check.py --out $PLDA_ROOT --audio-root $VOX_ROOT --expect-speakers 5994 \
  --env som=$SOM_PY:pyannote.audio \
  --env diarizen=$DZ_PY:diarizen,pyannote.audio \
  --env nemotron=$NEMOTRON_PY
```

Qua khi thoát mã 0 (mọi mục bắt buộc đạt). Nếu thiếu speaker, giải nén tiếp archive trước khi chạy tiếp.

## Giai đoạn 1: trích embedding VoxCeleb2

Thử 20 speaker trước:

```bash
python tools/extract_voxceleb_embeddings.py --audio-root $VOX_ROOT --out $PLDA_ROOT/r34lm_test \
  --embedder r34lm --device cuda --per-speaker 50 --limit-speakers 20
```

Kiểm tra `progress.jsonl` (số dùng, quá ngắn, bị từ chối). Nếu ổn, chạy đầy đủ (lệnh chạy lại sẽ bỏ qua speaker đã có):

```bash
python tools/extract_voxceleb_embeddings.py --audio-root $VOX_ROOT --out $PLDA_ROOT/r34lm_vox2 \
  --embedder r34lm --device cuda --per-speaker 50
python tools/extract_voxceleb_embeddings.py --out $PLDA_ROOT/r34lm_vox2 --embedder r34lm --merge
```

Qua khi: `train.npz` tồn tại, không có lỗi chiều, số speaker gần 5.994 và số embedding gần 5.994 × 50.

## Giai đoạn 2: huấn luyện PLDA

```bash
python tools/train_plda.py --train $PLDA_ROOT/r34lm_vox2/train.npz --out $PLDA_ROOT/plda_r34lm \
  --embedder-id r34lm --lda-dim 128 --holdout-speakers 0.2
```

Qua khi: `xvec_transform.npz`, `plda.npz`, `plda_meta.json` đều có. Dòng "held-out EER" in ra phải có giá trị PLDA thấp hơn cosine. Chưa có EER trên danh sách trial VoxCeleb1 vì chưa tải audio VoxCeleb1 test.

## Giai đoạn 3: kiểm chứng trên 30 mẫu

Chạy J hai lần: PLDA gốc của DiariZen (mặc định, không truyền `--plda-dir`) và PLDA mới. Lệnh chạy J cụ thể lấy từ `variants.py commands`; chỉ khác `--plda-dir`.

```bash
python tools/compare_plda_runs.py $OUT_BASE $OUT_NEW --arm nemotron_dz --max-mean-diff 0.5 \
  --csv $OUT_NEW/analysis_detail/per_sample_plda.csv
```

Qua khi: báo "ĐẠT", tức |trung bình hiệu| ≤ 0,5 điểm DER. Nếu không đạt, kiểm tra lại dữ liệu trích (giai đoạn 1) trước khi đi tiếp.

## Giai đoạn 4: embedder khác

Chưa làm. Lý do:
- **SimAM-ResNet100:** trọng số CC BY-NC-SA 4.0 (không thương mại), và adapter chưa xác minh. Chờ quyết định về giấy phép.
- **ESPnet:** adapter chưa xác minh. Chưa có trọng số đã kiểm tra.
- **ResNet-293 (`wespeaker_r293`):** có sẵn adapter ONNX. Có thể chạy giai đoạn 1–3 với `--embedder wespeaker_r293 --r293-onnx <đường dẫn>`, cùng quy trình như r34lm.

## Giai đoạn 5: chạy đầy đủ

1. Sao chép file mẫu rồi điền 16 khoá đang là null:

```bash
cp tools/variant_params.example.json variant_params.json
```

   Các giá trị cần điền là quyết định của bạn, không phải mặc định của code.
2. In lệnh của từng biến thể để kiểm tra trước khi chạy (lệnh này chỉ in, không chạy):

```bash
python tools/variants.py commands --params variant_params.json --samples tools/diar_arms_samples.json --out $OUT
```

3. Chạy trên toàn bộ mẫu, GPU 0, tối đa 2–3 tiến trình song song (`--reuse` để không tính lại cache).
4. Phân tích:

```bash
python tools/compare_arms.py $OUT --ref labels --arms nemotron,diarizen_auto,nemotron_dz --csv $OUT/der_labels.csv
python tools/analyze_arms_detail.py $OUT --arms nemotron,diarizen_auto,nemotron_dz --postprocess
```

Qua khi: mọi arm có đủ `segments_*.json` cho các mẫu có nhãn.

## Giai đoạn 6: tách A/B

```bash
python tools/make_split.py $OUT --seed 0 --out $OUT/split_seed0.json
```

- Chọn tham số (Fb, thr, PLDA, embedder) chỉ bằng số liệu nửa A:

```bash
python tools/analyze_arms_detail.py $OUT --arms ... --split $OUT/split_seed0.json --half A
```

- Báo cáo kết quả chính bằng nửa B, không chọn lại trên B:

```bash
python tools/analyze_arms_detail.py $OUT --arms ... --split $OUT/split_seed0.json --half B
```

Không đổi file split sau khi đã chọn tham số. Nếu bắt buộc đổi, ghi rõ trong tài liệu.

## Giai đoạn 7: ghi nhận

- Cập nhật `doc/tong_hop_kien_truc_diarization_sota.md` với kết quả giai đoạn 3 và 6.
- Commit tài liệu và mã (không commit audio, embedding, hay `train.npz`).

## Chưa làm

- Tải audio VoxCeleb1 test để tính EER trên `veri_test2.txt`.
- SimAM và ESPnet (chờ xác minh và quyết định giấy phép).
- SC-pNA: chưa cài đặt.
- Các biến thể `emb_*_vbx` và `sc_pna` đang ở trạng thái `todo` trong `variants.py`, chưa chạy được.
