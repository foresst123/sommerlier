# Rà soát logic xây cửa sổ v13

Ngày: 2026-09-14. Nhánh: `solid-architecture`.
Policy: `padding-reserved-context-capped-v13`.
Phạm vi: mã hiện tại, gồm các thay đổi chưa commit của các lượt trước.
Lượt rà soát này không sửa thuật toán hay config.

## Kết luận

Chưa thể coi việc xây window là đúng theo chính sách đã thống nhất. Trần context
2.2 giây vẫn bị vượt trong nhánh bù; lựa chọn padding có thể bỏ nguồn giọng sạch
đang có. Giữ nguyên overlap và ánh xạ sample qua bước ghép đang hoạt động trong
các ca đã kiểm tra.

Các ví dụ dưới đây là ca tổng hợp kiểm chứng logic. Chúng không phải kết quả đo
chất lượng âm thanh thực tế hoặc benchmark Sidon/WeSpeaker. Âm có năng lượng
đều được dùng để kiểm tra nhánh không có điểm nghỉ; một ca bổ sung sử dụng sóng
83 Hz với khoảng nghỉ 80 ms mỗi 0.5 giây.

## 1. [P1] Trần context 2.2 giây chưa được giữ ở mọi nhánh

Nguồn: `podcast-pipeline/utils/separation_window.py:523`, đặc biệt dòng 537-544;
đường gọi bù tại dòng 1003-1044.

Caller tính lượng còn được mở mỗi phía, nhưng `_expand_context` lại chuyển lượng
không lấy được ở phía bị chặn sang phía còn lại. Hàm này chỉ kiểm tra tổng window
15 giây và biên file/SP3; nó không kiểm tra trần context từng phía.

Tái hiện, sample rate 1000 Hz, file 20 giây:

- A: 0-10; B: 0.1-0.5.
- Context ban đầu: trái 0.1, phải 2.0 giây.
- Context cuối: trái 0.1, phải **6.4 giây**; window dài 6.9 giây.
- Đảo ca sang cuối file cho kết quả trái 6.4, phải 0.1 giây.
- SP3 kết thúc 0.1 giây trước overlap tạo cùng lỗi; SP3 sát cuối overlap có thể
  đẩy context trái lên **6.6 giây**.

Một đường vượt trần khác nằm ở constructor dòng 241:
`max_context = max(context, configured_max)`. Đặt target 3.0 và max 2.2 được chấp
nhận và thực tế dùng 3.0 giây mỗi phía. Đây không phải max độc lập.

Hướng sửa: kiểm soát giới hạn tuyệt đối từ core ở hàm mở rộng dùng chung, kể cả
khi chuyển budget; kiểm tra config bất hợp lệ và kiểm tra trần lần cuối trước
assemble. Không chỉ giới hạn tolerance của lần mở đầu.

## 2. [P1] Đang tính tiếng overlap là evidence sạch để chọn padding

Nguồn: `podcast-pipeline/utils/separation_window.py:860` và `:937`.

`base_evidence` có tính riêng `solo_ranges`, nhưng điểm chọn pad lại dùng tổng
thời gian có tiếng của từng speaker, bao gồm overlap. `padding_min` được so với
tổng này. Do đó giá trị config 1.5 giây không bảo đảm có 1.5 giây pad hoặc probe
sạch cho từng speaker.

Tái hiện:

- A và B cùng có đoạn 18-24, overlap dài 6 giây.
- Pad sạch A: 0-2; pad sạch B: 30-32.4.
- Hai nguồn sạch đều tồn tại. Window được chọn dài 10.4 giây, context 2.2/2.2.
- Padding **0/0**, clean probe **A=0, B=0**, nhưng estimated voice là **6/6**.
- Cả hai candidate thực tế đều vừa budget mỗi phía. Ghép hai pad này với base
  ban đầu tạo window **14.36 giây**, probe sạch **A=1.98, B=2.38 giây**; vậy
  đây là lựa chọn bị điểm số loại bỏ, không phải thiếu nguồn hoặc hết chỗ.

Tiêu chí cân bằng tại dòng 950-956 đứng trước độ dài và vị trí core. Bỏ cả hai
pad giữ tỉ lệ 1:1 nên thắng lựa chọn có evidence thật nhưng tỉ lệ hơi lệch.

Hướng sửa: dùng lượng solo/probe sạch, trừ vùng crossfade, để đánh giá thiếu mẫu.
Thời gian giọng trong overlap có thể là một chỉ số khác, không thay cho clean
evidence. Ưu tiên đáp ứng evidence trước rồi mới cân bằng trong một khoảng dung sai.

## 3. [P2] Loại ứng viên padding quá sớm, bỏ cả nguồn còn hợp lệ

Nguồn: `podcast-pipeline/utils/separation_window.py:277`, `:319` và `:379`.

Planner lấy tối đa 12 segment gần nhất, rồi chỉ giữ hai candidate mỗi bucket
độ dài trước khi `_sequences` xét biên SP3 và phần đã dùng làm base.

Tái hiện:

- A: 98-110; B overlap: 100-100.4; SP3: 90-95.
- B sạch: 81-84, 85-88, 130-133; cả ba cùng dài 3 giây.
- Hai nguồn trước SP3 được giữ vì gần hơn; nguồn 130-133 bị bỏ.
- Sau kiểm tra biên 95 giây, hai nguồn đã giữ đều không dùng được: pad B = 0.
- Bỏ hai nguồn không hợp lệ khỏi dữ liệu đầu vào thì chính nguồn 130-133 được
  chọn, tạo pad/probe B = **2.98 giây**.

Hướng sửa: kiểm tra tính hợp lệ trong window trước khi rút gọn candidate; nếu
nguồn gần thất bại cần tìm tiếp các nguồn xa hơn còn trong bán kính cho phép.

## 4. [P2] Các pad ngắn chưa được gom để đạt tổng evidence

Nguồn: `podcast-pipeline/utils/separation_window.py:246`, `:314` và `:349`.

Mỗi support piece bị yêu cầu dài ít nhất 1.5 giây và có ít nhất 1 giây voice trước
khi đến bước ghép nhiều piece. Các ngưỡng này được hard-code, độc lập với config
`padding_min_per_speaker_seconds`.

Tái hiện: A 10-30, B overlap 20-20.4, B sạch 0-1.4 và 40-41.4. Có tổng 2.8 giây
nguồn sạch nhưng cả hai bị loại; window dài 4.8 giây, pad B/probe B bằng 0.

Hướng sửa: tách ngưỡng chất lượng mỗi clip khỏi ngưỡng tổng evidence mỗi speaker;
cho phép ghép các clip ngắn hợp lệ để đạt tổng, có tính hao hụt crossfade.

## 5. [P2] Budget cứng quanh 7.5 giây có thể bỏ pad vốn vừa tổng window

Nguồn: `podcast-pipeline/utils/separation_window.py:900` và `:950`.

Mỗi phía bị giới hạn budget nhằm đặt tâm core ở 7.5 giây. Nhưng sau đó điểm
vị trí core lại xếp cuối cùng, sau cân bằng giọng và độ dài. Hai bước tối ưu
không cùng mục tiêu: vừa giới hạn mạnh nguồn có thể chọn, vừa không giữ tốt
vị trí core sau khi chọn.

Tái hiện nhánh không có điểm nghỉ trong pad:

- A 10-30; B overlap 20-20.4; B sạch 40-46 (6 giây).
- Base khoảng 4.4 giây, còn tổng khoảng 10.6 giây trước giới hạn 15 giây.
- Pad 6 giây vẫn bị bỏ vì không vừa budget khoảng 5.3 giây của mỗi phía và không
  có acoustic cut nội bộ. Kết quả window 4.8 giây, probe B = 0.
- Chỉ cần cho phép lệch tâm core một chút là nguyên pad này vừa giới hạn tổng.

Trong ca có khoảng nghỉ đều, overlap 20-20.45 và nhiều nguồn sạch cho cả hai
speaker, window được chọn dài 8.75 giây, tâm core 6.405 giây, pad chỉ ở một phía.
Window ngắn tự nó không phải lỗi; điểm cần sửa là mất evidence khi bố cục khác
vẫn khả thi.

Hướng sửa: dùng vị trí 7.5 giây làm ưu tiên mềm trong tìm bố cục, thử phân bổ
bất đối xứng khi pad không vừa, và tránh việc một chênh lệch tỉ lệ giọng nhỏ
quyết định toàn bộ bố cục.

## 6. [P2] Retry cuối chủ động tắt padding dù vẫn có nguồn sạch

Nguồn: `podcast-pipeline/services/separation_service.py:1214` và `:1228`.

`padding=next_attempt == 1` khiến retry thứ hai luôn không thêm pad. Mô hình giả
trả track rỗng để buộc đi đủ ba lần đã cho kết quả:

| Lần chạy | Context trái/phải | Pad trái/phải | Probe sạch B |
| --- | --- | --- | --- |
| Ban đầu | 2.2 / 2.2 | 3.98 / 0 | 3.98 |
| Retry 1 | 1.0 / 1.0 | 3.98 / 1.98 | 3.98 |
| Retry 2 | 2.2 / 2.2 | 0 / 0 | 0 |

Đây là hành vi chắc chắn của code; chưa đo xem nó làm chất lượng model thật giảm
bao nhiêu. Tuy nhiên nó trái mục tiêu cố giữ evidence sạch khi nguồn vẫn còn.

Hướng sửa: retry nên đổi context hoặc chọn pad khác dựa trên lý do fail. Chỉ bỏ
pad như một phương án có lý do riêng, không áp dụng bắt buộc ở lượt cuối.

## 7. [P1] SP3 ngắn có thể làm bỏ cả một phần overlap dài không có SP3

Nguồn: `podcast-pipeline/utils/separation_window.py:738` và
`podcast-pipeline/services/separation_service.py:1303`.

Tái hiện qua cả planner và separation service với model giả:

- A 0-40, B 10-30: overlap A/B dài 20 giây.
- C chỉ nói 17-18.
- Overlap được chia thành 10-20 và 20-30.
- Toàn bộ 10-20 bị đánh dấu `multi_speaker`, không retry.
- C chỉ chiếm 1 giây nhưng 9 giây A/B hợp lệ tại 10-17 và 18-20 cũng bị bỏ.

Hướng sửa: dùng biên xuất hiện/kết thúc SP3 để chia vùng trước khi lập model job.
Chặn đúng phần ba speaker, giữ và xử lý các phần còn lại. Tiếp tục ghi rõ phần
bị chặn vào báo cáo coverage.

## 8. [P2] Chỉnh config có thể vẫn dùng window trong checkpoint cũ

Nguồn: `podcast-pipeline/services/pipeline_service.py:260` và `:477`;
`podcast-pipeline/utils/checkpoint.py:11`.

Namespace cache chỉ chứa `POLICY_VERSION`, không chứa các giá trị context,
padding hay search. Nếu đã có checkpoint v13, đổi max 2.2 thành 1.8 trong config
không đổi đường dẫn checkpoint. Kiểm tra với `CheckpointManager` cho thấy vẫn
cache hit và trả dữ liệu đã tạo theo 2.2.

Bump policy v12 -> v13 giúp loại cache của bản thuật toán trước, nhưng không
giải quyết các lần tự chỉnh config trong cùng phiên bản. Full rerun có xóa cache
thì không gặp trường hợp này.

Hướng sửa: thêm dấu vân tay của cấu hình thực dùng sau override môi trường vào
namespace, truyền cùng dấu vân tay xuống ASR và các bước phụ thuộc.

## 9. [P2] Report chưa kiểm tra đúng điều nó gọi là đạt

Nguồn: `podcast-pipeline/utils/separation_window.py:1098` và `:1122`;
`podcast-pipeline/services/separation_service.py:301`.

`context_target_met` chỉ kiểm tra thiếu context, không kiểm tra vượt trần.
Mục tiêu mỗi phía đã bị kẹp theo file/SP3 trước khi tính shortfall, vì vậy thiếu
khách quan do SP3 cũng có thể hiện là `ok`. `context_priority_verified` được
gán trực tiếp từ `context_target_met`, không kiểm tra padding hoặc evidence.

Trong ca context 0.1/6.4 giây, report vẫn ghi `context_status=ok`. Trong ca hai
probe sạch đều 0 giây, report cũng vẫn ghi `ok`. Các cờ này không chứng minh
window có đủ điều kiện nhận dạng hai speaker.

Ngoài ra, `quality_split_seconds=12.0` trong report không khớp việc đường chính
chỉ chia core dài hơn 15 giây. Config `boundary_search_seconds` thực tế điều khiển
bán kính tìm support, không điều khiển khoảng tìm acoustic cut; tolerance của
context vẫn cố định 0.35 giây.

Hướng sửa: xuất riêng target, max, context thực mỗi phía, lượng probe sạch và pad
từng speaker, lý do bỏ candidate, lý do không đạt và cấu hình thực của từng retry.
Window vẫn có thể được thử best-effort nhưng cần đánh dấu chất lượng suy giảm.

## Những phần đã kiểm tra đạt

- 26 kịch bản hình học, tổng 38 window ở sample rate 1000 và 16000 Hz.
- Độ dài overlap từ 0.0001 đến 31 giây, gồm ca sát và vượt 15 giây.
- Mọi window trong nhóm này không vượt 15 giây và giữ đúng mẫu core nguồn.
- Các mảnh chia từ core dài không hở hoặc chồng sample.
- Crossfade bảo vệ core và loại vùng nối khỏi probe trong các test hiện có.
- Test retry giữ nguyên speaker đã splice thành công đạt.
- Ghép trả dùng `core_source_samples` để tính offset, tránh lấy pad ở xa làm
  timestamp liên tục của nguồn.
- Qua seam và overlap A/B khác là hành vi chủ ý hiện tại, không coi là lỗi.

## Tình trạng test

Lệnh chạy bằng Python Homebrew có các dependency cần cho test, tắt pytest plugin
autoload. Không tải hoặc chạy mô hình GPU.

```text
tests/test_window_context_balance.py
tests/test_separation_window.py
tests/test_separation_recovery.py
32 passed, 6 failed
```

Các thất bại hiện có:

- Test tên phương pháp cắt vẫn đòi `energy`, trong khi finder trả `energy_pause`
  hoặc `energy_valley`.
- Test vẫn đòi chặn overlap A/B rời nhau.
- Test vẫn đòi từ chối secondary overlap gần mép segment.
- Test vẫn đòi từ chối overlap 10 giây và seam.
- Test vẫn đòi core overlap ngắn tự nở ra ngoài overlap.
- Test QC đòi từ chối coverage 19/59, nhưng ngưỡng hiện tại chấp nhận từ 20%.

Đây không phải sáu bằng chứng độc lập của lỗi mới trong planner. Cần cập nhật
những kỳ vọng lỗi thời và xác định lại kỳ vọng QC. Các ca tái hiện bổ sung trong
báo cáo chứng minh các lỗi còn thiếu kiểm thử, nhất là trần context từng phía,
clean evidence, thứ tự lọc candidate và bảo toàn vùng hợp lệ quanh SP3.

## Thứ tự sửa đề xuất

1. Chặn max context ở mọi đường mở rộng; xác thực target/max.
2. Chấm thiếu evidence bằng solo/probe sạch; sửa việc xếp hạng bỏ pad.
3. Lọc candidate hợp lệ trước khi rút gọn; cho gom pad ngắn và thử budget lệch.
4. Sửa retry và chia riêng vùng SP3 để không bỏ phần A/B hợp lệ.
5. Bổ sung audit report và fingerprint config cho checkpoint.
6. Bổ sung regression tests cho các ca trên, cập nhật test đã lệch chính sách.
