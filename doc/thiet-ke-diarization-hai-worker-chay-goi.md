# Thiết kế: diarization hai worker chạy gối

Ngày: 2026-09-21. Nhánh: `solid-architecture`.

Trạng thái: **bản thiết kế đã được duyệt, chưa triển khai.**

## 1. Vấn đề

Diarization hiện chạy hai pha nối tiếp trong một worker. Đo trên Kaggle 2×T4
ngày 2026-09-20, file `hoahau.mp3` (49 phút audio, 44 phút diarization):

| Giai đoạn | GPU0 | GPU1 |
| --- | --- | --- |
| 18:24 → 18:45 (20 phút) | 5,36 GiB — segmentation chạy | 0,29 GiB — model nằm chờ |
| 18:45 → 19:08 (22 phút) | 5,36 GiB — model nằm không | 2,05 GiB — embedding chạy |

`split_components` đã đặt hai model lên hai card, nhưng DiariZen gọi
`get_segmentations()` xong hết mới tới `get_embeddings()`, nên mỗi card nằm
không đúng một nửa thời gian. Các file cũng chạy nối tiếp: `thu_that_thach`
chỉ bắt đầu lúc 19:01 khi `hoahau` đã xong hoàn toàn.

Mục tiêu: pha segmentation của file N+1 chạy **song song** với pha
embedding/clustering của file N. Với các file cùng cỡ, tổng thời gian stage
tiến dần về `max(Σ segmentation, Σ embedding)` thay vì tổng của cả hai.

## 2. Vì sao cắt ở đây là an toàn

`DiariZenPipeline.__call__` là một chuỗi tuyến tính, các pha nhận dữ liệu qua
tham số chứ không qua trạng thái ẩn (inference.py ở commit `844f5555`):

```python
segmentations = self.get_segmentations(audio, soft=False)        # pha A
segmentations.data = median_filter(segmentations.data, size=(1, 11, 1), mode='reflect')
binarized_segmentations = segmentations                          # cùng một object
count = self.speaker_count(binarized, self._segmentation.model._receptive_field, warm_up=(0.0, 0.0))
embeddings = self.get_embeddings(audio, binarized, exclude_overlap=self.embedding_exclude_overlap)
hard_clusters, _, _ = self.clustering(embeddings, binarized, min_clusters=..., max_clusters=...)
discrete_diarization, _ = self.reconstruct(segmentations, hard_clusters, count)
```

Điều này khác hẳn phương án chia nhỏ **trong cùng một file** mà
`ke-hoach-toi-uu-hieu-nang-toan-pipeline.md` §8 đặt 6 điều kiện khắt khe
(giữ lưới window, median filter đủ ngữ cảnh ở biên, mask khớp nhau...).
Ở đây mỗi pha vẫn xử lý **trọn vẹn một file**: lưới window nguyên vẹn, median
filter chạy trên toàn bộ mảng, và clustering vẫn nhìn toàn bộ embedding của
file một lượt. Không điều kiện nào trong số đó bị đụng tới.

Ranh giới cắt nằm ngay sau `median_filter`. Vì `binarized_segmentations` chính
là `segmentations`, chỉ **một mảng** đi qua ranh giới process.

## 3. Kiến trúc

```
worker A (GPU0)   file1 seg ──► file2 seg ──► file3 seg ──► ... ─► dừng
                       │            │            │
                    [pool đĩa]   [pool đĩa]   [pool đĩa]
                       ▼            ▼            ▼
worker B (GPU1)      file1 emb+cluster ──► file2 emb+cluster ──► ... ─► dừng
```

### 3.1 Một script, hai vai

`diarizen_worker.py` nhận thêm `--role seg|emb`. Không tách script mới: phần
nạp model, áp config và kiểm tra tham số giống hệt nhau, và tách ra sẽ tạo hai
bản sao phải sửa song song mỗi lần đổi config.

| | vai `seg` | vai `emb` |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | GPU0 | GPU1 |
| Giữ trên GPU | model segmentation | model embedding |
| Đẩy sang CPU | model embedding | model segmentation |
| Lệnh nhận | `{"cmd":"segment","id","audio_path","out_path"}` | `{"cmd":"embed","id","audio_path","seg_path","meta"}` |
| Việc làm | `get_segmentations` + `median_filter`, lưu mảng ra đĩa | `speaker_count` → `get_embeddings` → `clustering` → `reconstruct` → `Binarize` |
| Phản hồi | `{"id","seg_path","meta","seconds","bytes"}` | `{"id","segments":[{"start","end","speaker"}]}` |

Vai `emb` trả đúng định dạng mà `diarize()` trả hôm nay, nên
`diarization_service` và mọi bước sau **không phải sửa**.

Model không dùng được đẩy sang CPU chứ không xoá, vì vai `emb` vẫn cần
`self._segmentation.model._receptive_field` cho `speaker_count`, và vì giữ
nguyên object pipeline là cách chắc chắn nhất để không lệch config.

### 3.2 Bàn giao qua đĩa

`segmentations.data` lưu `.npy` trong `<cache_dir>/<job_id>/diarization_pool/`,
tên theo `id` của file. Metadata cửa sổ trượt (`start`, `duration`, `step`) đi
trong JSON phản hồi — ba số float, đủ để vai `emb` dựng lại
`SlidingWindowFeature`.

Chọn đĩa thay vì RAM vì pool không giới hạn: worker A không bao giờ phải chờ
B, nhưng 20 file × ~120 MB nằm trên đĩa chứ không cộng vào một tiến trình đã
đỉnh 13,2 GB RSS. File `.npy` bị xoá ngay khi B tiêu thụ xong, và toàn bộ thư
mục bị xoá khi stage kết thúc.

Kích thước mảng ước tính 30–120 MB cho file 49 phút (9.226 chunk), tuỳ dtype.
Đây là **ước lượng suy từ số chunk, chưa đo**; vai `seg` báo `bytes` thật trong
phản hồi để lần chạy đầu tiên cho con số chính xác.

### 3.3 Điều phối

Khi stage diarization bắt đầu, orchestrator gửi **toàn bộ** yêu cầu
segmentation của batch cho A. A xử lý tuần tự theo thứ tự file và không bao giờ
bị chặn — đó chính là "pool không giới hạn".

Một luồng nền trong tiến trình chính đọc phản hồi của A vào một dict theo `id`.
Khi `run()` xử lý tới file N, client chờ phản hồi N của A (thường đã sẵn), gửi
sang B, rồi chặn chờ B — đúng như `diarize()` hiện nay.

Nhờ vậy vòng lặp stage, checkpoint và `run()` **không đổi một dòng**.

Để lấy danh sách file, `run_batch_by_stage` gọi thêm hook
`pipeline.prepare_stage(stage, pending)` ngay sau `begin_stage_scope()`.
`PipelineService.prepare_stage` chuyển tiếp cho diarization service khi stage
là `diarization`, và không làm gì cho các stage khác. Hook này là no-op khi
pipeline không có nó, nên các test dựng pipeline giả không bị ảnh hưởng.

### 3.4 Lỗi và vòng đời

A lỗi ở file N thì phản hồi mang `error`; file N bị đánh dấu hỏng như hôm nay
và các file khác chạy tiếp. B lỗi cũng vậy. Worker chết thì các file sau lỗi
theo, giống hành vi hiện tại của worker đơn.

Hai worker đăng ký trong `worker_services` dưới tên `diarizen_seg` và
`diarizen_emb`, nên `_release_worker` và `end_stage_scope` sẵn có tự dừng
chúng khi hết stage — trước khi separation bắt đầu, và `_reclaim_vram()` ghi
lại VRAM còn trống để đối chiếu.

### 3.5 Bật/tắt

`performance.stages.diarization.placement` nhận thêm giá trị `"pipelined"`,
bên cạnh `"single"` và `"split_components"` đang có.

Chỉ bật khi đủ cả ba: `performance.enabled`, `placement == "pipelined"`, và
`gpu_1 != gpu_2`. Thiếu bất kỳ điều kiện nào thì rơi về worker đơn như hiện
nay, không đổi hành vi. Máy một GPU vì thế không bị ảnh hưởng.

## 4. Rủi ro chưa xác minh

**Model segmentation ở CPU có đủ cho vai `emb` không.** Theo mã nguồn,
`speaker_count` chỉ cần `_receptive_field` (metadata), nhưng `reconstruct` có
thể đụng thuộc tính khác của model. DiariZen không cài trên máy phát triển nên
chưa chạy thử được.

*Xử lý:* việc đầu tiên khi triển khai là một kiểm chứng nhỏ chạy trên Kaggle —
gọi `get_embeddings`/`clustering`/`reconstruct` với model segmentation ở CPU và
so kết quả với đường chạy hiện tại trên cùng một file. Nếu không đạt, phương án
dự phòng là giữ model segmentation trên GPU1 ở vai `emb` (tốn thêm ~5,4 GiB
trên card còn trống 12,5 GiB), và nếu vẫn không đạt thì dừng thiết kế này lại.

**Kết quả phải giống đường chạy hiện tại.** Cắt pha không được đổi nhãn
speaker. Nghiệm thu: chạy cùng một file qua cả hai đường, so số speaker, DER
sau khi căn hoán vị nhãn, và ranh giới segment.

## 5. Kiểm thử

Bằng worker giả, không cần GPU:

- A không bị chặn khi B còn chậm: A nhận hết N yêu cầu trước khi B xong file 1.
- B nhận đúng thứ tự file, và kết quả trả về khớp `id` chứ không theo thứ tự
  phản hồi.
- Một file lỗi ở A không làm chết các file còn lại.
- `SlidingWindowFeature` dựng lại từ `.npy` + metadata khớp mảng gốc.
- Một GPU, hoặc `performance.enabled=false`, hoặc `placement` khác: rơi về
  worker đơn, và không worker nào của chế độ mới được khởi động.
- File `.npy` bị xoá sau khi B tiêu thụ và khi stage kết thúc.

Trên Kaggle, ngoài kiểm chứng ở mục 4: đối chiếu `stage_seconds` của
diarization trước và sau, và xem `resource_samples.jsonl` để xác nhận hai card
cùng bận thay vì luân phiên.

## 6. Cái giá cần nói rõ

Thiết kế này chỉ đáng giá khi `segmentation_step` giữ ở `0.02`. Hop hiện tại
tạo 9.226 chunk cho 49 phút audio (chồng lấn 98%, chính DiariZen in cảnh báo).
Nếu đưa hop về `0.1`, pha segmentation rút từ 20 phút xuống còn khoảng 4 phút,
và phần chồng lấn chỉ còn cứu được ~4 phút mỗi file thay vì ~20.

Phép đo DER giữa hop `0.02` và `0.1` chưa được thực hiện. Người dùng đã chọn
triển khai chồng lấn trước; ghi lại ở đây để quyết định này không bị hiểu nhầm
là đã có bằng chứng ủng hộ.
