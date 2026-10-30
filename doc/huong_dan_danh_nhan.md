# Hướng dẫn gán nhãn tay cho ~30 clip

Mục tiêu: chọn khoảng 30 clip (45-60 s) từ hai nguồn, chạy ba arm diarization (A, B, J) trên đó, rồi gán nhãn tay để chấm điểm.

Đường dẫn dùng trong hướng dẫn:

```bash
AUDIO_YT=/home/lamkd2/data/data_input/craw_data/data/youtube_sicom_audio
AUDIO_MOVIE=/home/lamkd2/data/data_input/data_movie
OUT=/home/lamkd2/data/data_output/denoise_V4
```

**Lưu ý:** không có bước nào trong luồng này được chạy trong sandbox. Các lệnh dưới đây lấy từ `main()` của từng script, chưa được kiểm chứng trên dữ liệu thật.

## 1. Cập nhật nhánh

```bash
cd <repo>/podcast-pipeline
git pull origin exp/denoise-compare
```

## 2. Chọn clip

```bash
python tools/make_label_samples.py \
  --audio-dir $AUDIO_YT --audio-dir $AUDIO_MOVIE \
  --out $OUT/samples_label30.json
```

Mặc định: 20 clip từ manifest xuất (round-robin theo category, mỗi file nguồn một clip), 10 clip ngẫu nhiên có tiếng nói, dài 50 s, seed 0. Các mẫu có sẵn trong `tools/diar_arms_samples.json` bị loại. Thêm `--dry-run` để xem tóm tắt mà không ghi file.

## 3. Chạy arm A, B, J

```bash
python tools/diar_arms_experiment.py \
  --samples $OUT/samples_label30.json \
  --out $OUT/diar_arms_label30 \
  --only-arms A,B,J
```

Mỗi mẫu đã có đường dẫn `audio` tuyệt đối, nên không cần `--audio-dir`. Thư mục `diar_arms_label30` là thư mục mới, không dùng chung với kết quả cũ. Chạy lại vào cùng thư mục sẽ ghi đè các arm đã có.

## 4. Mở trang gán nhãn

```bash
python tools/label_server.py \
  --root $OUT --arms-dir $OUT/diar_arms_label30 \
  --audio-dir $AUDIO_YT --port 8002
```

Trên máy laptop, mở đường hầm rồi vào trang:

```bash
ssh -L 8002:localhost:8002 <dgx>
# trình duyệt: http://localhost:8002/label
```

## 5. Cách gán nhãn

1. Bấm "Nạp" để lấy kết quả một arm (A, B hoặc J) làm điểm xuất phát, rồi sửa.
2. Chọn một vùng bằng hai tay cầm (hai biên) trên sóng, rồi bấm tên người nói trong thanh vùng.
3. Vùng có hai người nói cùng lúc: gán nhãn cho cả hai, các đoạn chồng lấn được phép.
4. Đoạn không ai nói: bấm "Không ai nói".
5. Âm thanh không rõ (tiếng ồn, nhạc, nói nhỏ, hai người lẫn vào nhau): bấm "Chưa chắc". Đoạn này không được tính điểm.
6. "Hít biên arm" bật sẵn, tự kéo biên về biên của arm. Giữ phím Alt để tắt tạm thời.
7. Ô "±" là dung sai (mặc định 0,25 s) quanh mỗi biên nhãn khi chấm điểm. Chỉnh ở đây nếu cần.
8. Ghi chú về clip điền vào ô ghi chú, ví dụ có nhạc nền hoặc có người thứ ba.
9. Bấm lưu thường xuyên. Mỗi lần lưu, bản cũ được giữ trong `labels_history/`.
10. Kết quả nằm ở `$OUT/diar_arms_label30/<NN_tên_mẫu>/labels.json`, cạnh `clip.wav`.
11. Nếu hai tab cùng sửa một mẫu, trang sẽ báo xung đột thay vì ghi đè.

## 6. Chấm điểm các arm

```bash
python tools/compare_arms.py $OUT/diar_arms_label30 --ref labels
```

Thêm `--csv $OUT/diar_arms_label30/compare_labels.csv` nếu cần bảng dạng CSV. Dung sai lấy từ `labels.json` của từng mẫu, trừ khi truyền `--collar`.
