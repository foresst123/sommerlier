# Thiết kế stage `persona_profile`

Ngày: 2026-10-12 · Nhánh: `solid-architecture` · Trạng thái: đã chốt (qua 4 vòng review, vòng 4: READY)

Stage opt-in, chạy sau `transcript_review` và trước `conversation_selection`, trích **hồ sơ từng người nói** (văn nói, từ đệm, tiểu từ, xưng hô, backchannel, ngắt lời, nhịp lượt) từ toàn bộ transcript của một file ghi âm. Hồ sơ là đầu vào cho bước gom persona và sinh hội thoại synthetic về sau; stage này không sinh dữ liệu và không sửa transcript.

Phần A mô tả vị trí, luồng và tích hợp. Phần B là hợp đồng dữ liệu (schema, công thức, test) và là **nguồn chuẩn** khi hai phần nói khác nhau. Mục C ghi các quyết định đã chốt.

Mọi đường dẫn tính từ `podcast-pipeline/`. Số dòng lấy từ code ở nhánh `solid-architecture`, commit `cbce204`.

# PHẦN A. Vị trí, luồng, tích hợp

## A1. Mục tiêu và phạm vi

**Mục tiêu.** Với mỗi file ghi âm, trích một hồ sơ cho từng người nói trong file đó. Hồ sơ gồm phong cách văn nói, từ đệm, tiểu từ cuối câu, cách xưng hô, tần suất backchannel, tần suất và kiểu ngắt lời, nhịp lượt. Nguồn là TOÀN BỘ transcript của file, không chỉ các conversation đã được `conversation_selection` chọn. Hồ sơ là đầu vào cho một script offline sau này (gom persona, sinh hội thoại synthetic cho huấn luyện full-duplex tiếng Việt Lychee-FD).

**Trong phạm vi.**
- Stage riêng `persona_profile`, opt-in qua `opt_in_step_enabled(args, "persona_profile")` (`utils/steps.py:35-44`), mặc định tắt.
- Mọi con số, kể cả danh sách từ đệm / tiểu từ / xưng hô, do code tính từ các từ điển cố định có phiên bản (B4.4). Qwen 9B chỉ làm hai việc: (1) viết `style.description`; (2) đề xuất form MỚI ngoài từ điển để mở rộng từ điển offline. Đề xuất của Qwen không đi vào bất kỳ số liệu nào của profile.
- Ghi `profile.json` và `stats.json` vào thư mục stage `07c_persona_profile/` qua `StageOutputService`, kèm checkpoint có phiên bản.

**Ngoài phạm vi (không làm trong stage này).**
- Không sinh dữ liệu, không sửa `text`/`speaker`/thời gian của transcript, không đặt cờ mới lên segment. Stage chỉ đọc (làm việc trên bản sao, P0).
- Không làm pair dynamics, exemplars, generation constraints, `persona_id`, `created_at`.
- Không gom persona. Việc đó là script offline ngoài pipeline.
- Không tạo định danh người nói xuyên file. Nhãn speaker là nhãn của từng file (diarizer ra `'0'..'4'` hoặc `SPEAKER_01`; xem `services/speaker_relabel_service.py:130-138`), nên khoá của hồ sơ là `(file, speaker_label)`.

## A2. Vị trí trong pipeline

### Sơ đồ luồng

Thứ tự stage lấy từ `_STOP_ORDER` (`services/pipeline_service.py:592-595`), `PIPELINE_STAGES` (`utils/batch.py:144-148`) và các khối trong `run()` (`pipeline_service.py:1264-1920`). Cổng `input_audio_quality` chạy ở `run()` dòng 1294 (`_check_input_audio_quality`, định nghĩa dòng 138), trước mọi worker.

```
input_audio_quality (cổng DNSMOS, pipeline_service.py:138, gọi ở 1294)
  -> music                  (01_music)
  -> diarization            (02_diarization)
  -> separation             (03_separation)
  -> music_removal          (04_music_removal; không là pass riêng trong batch)
  -> asr                    (05_asr)
  -> captioning             (tắt mặc định)
  -> refinement             (06_refinement; Qwen 9B nạp)
  -> speaker_relabel        (07_relabel; opt-in)
  -> transcript_review      (07b_transcript_review; opt-in; ghi review_flags)
  -> [persona_profile]      (07c_persona_profile; opt-in)   <== STAGE MỚI
  -> conversation_selection (chỉ khi conversation_exports bật; Qwen)
  -> word_alignment         (08_word_alignment; _release_llm_before gỡ Qwen)
  -> conversation_exports   (09_conversation_exports; chỉ ghi)
  -> clean-data + export    (pass cuối, stage=None)
```

### Vì sao đặt ngay sau `transcript_review`, trước `conversation_selection`

1. **Word alignment chỉ align segment đã chọn khi bật exports.** `_align_selected` (`pipeline_service.py:1045-1066`) lấy `selection.segments_to_align()` làm tập con (dòng 1054) và trả về luôn nếu rỗng (dòng 1055-1056). Nó được gọi ở dòng 1879 khi `exports_planned and alignment_on`. Đặt stage sau word_alignment thì phần lớn segment không có word time.
2. **Qwen bị gỡ trước word_alignment.** `_release_llm_before(args, "word alignment")` (định nghĩa dòng 450-466) được gọi ở dòng 1060 (đường chọn lọc) và dòng 1861 (đường align cả file). Sau điểm đó không còn `refinement_svc` resident, nên không còn LLM cho P3.
3. **Sau `transcript_review`, Qwen vẫn nạp và transcript còn nguyên.** Khối review ở dòng 1798-1812 chạy ngay trước khối "7c Forced alignment" (dòng 1815). `review_flags` đã gắn lên segment (`apply_review_flags`, `services/transcript_review_service.py:282`). Nếu giữ Qwen resident (xem cảnh báo bên dưới) thì không tốn cold load.
4. **Hệ quả cho độ chính xác thời gian.** Tại vị trí này mọi phép đo thời gian ở mức segment. `TranscriptSegment.words` được điền từ Whisper (`services/asr_service.py:1005`); refinement sửa `text` nhưng không xoá `words` (không có lệnh gán `words` nào trong `services/diarization_refinement_service.py`), và chính `pipeline_service.py:1815-1817` ghi rằng word time cũ không còn hợp lệ sau refinement. Chỉ `apply_word_alignments` (`services/word_alignment_service.py:135-145`) mới xoá `words` cũ và gán lại. `has_complete_word_alignment` (`utils/conversation_selection.py:305-325`) chỉ so SỐ từ, không so nội dung, nên không phân biệt được `words` Whisper cũ với `words` đã align. Quy tắc: bản sao làm việc LUÔN có `words=None` (P0); `analyze()` rơi về timing mức segment (`utils/conversation_events.py:105-111`). Không có chế độ "mixed".

### Điều kiện chạy lại LLM giữa các pass (cảnh báo vận hành)

`_release_llm_after(args, "transcript_review")` (dòng 517-535) giữ LLM chỉ khi `_next_llm_step_after(...)` khác None và `_keep_llm_between_passes(args)` đúng (đòi `performance_config.enabled`, `stages.refinement.keep_llm_across_relabel`, `batch_continues`; dòng 509-515). Nếu không, LLM bị gỡ giữa hai pass và `persona_profile` sẽ nạp lại qua `llm.ensure_loaded()` (như `review()` làm ở `transcript_review_service.py:381-385`). Tức đúng nhưng tốn một lần cold load. Muốn tránh thì mở rộng `_next_llm_step_after` (A5).

## A3. Luồng nội bộ P0-P5

Đơn vị xử lý: một file. Input chung: `transcripts` (list `TranscriptSegment`, đã qua refinement/relabel/review nếu bật), `self.timeline`, `self.noise_track`, `audio_path`.

**P0. Chuẩn bị (trên bản sao).**
1. Bản sao `dataclasses.replace(seg, words=None)` cho mọi segment; sắp theo `start` (sắp ổn định).
2. Chuẩn hoá `text` của bản sao sang NFC (C3).
3. `utils.provenance.annotate(copies, self.timeline, noise=self.noise_track)` (`utils/provenance.py:37`; yêu cầu danh sách đã sắp theo `start`, `:40`). Chạy trên bản sao nên `transcripts` thật không bị ghi gì. Cần vì khi relabel/review tắt thì `gap_before`, `crosses_cut`, `noise_breakdown` chưa có.
4. `source_digest` = băm (sha1 của JSON `sort_keys`) mọi trường mà P1/P2 đọc, theo từng segment: `index, start, end, speaker, text` (đã NFC), `has_music, unseparated, noise_breakdown, crosses_cut, gap_before, review_flags`. Dùng cho namespace checkpoint (A6).
- Giữ NGUYÊN nhãn speaker sau relabel (không đổi sang A/B).
- `recording_id = os.path.splitext(os.path.basename(audio_path))[0]`.

**P1. Lọc (chỉ đánh dấu).** Mỗi segment nhận đúng một lý do loại theo thứ tự ưu tiên (B4.1 bước 1): `review_flag`, `music`, `noise`, `unseparated_overlap`, `empty_text`. Mọi segment (kể cả `empty_text`) vẫn tham gia chia region vì mang `gap_before`/`crosses_cut`; sau bước chia region, `empty_text` bị bỏ khỏi các bước sau, còn bốn lý do kia ở lại làm ngữ cảnh cho `analyze()` nhưng không đóng góp số liệu. Ra tập `usable` và `excluded_s` cho `stats.json`. `review_flags is None` (chưa review) không bị loại, chỉ sinh cảnh báo `not_reviewed`.

**P2. Đo bằng code (không LLM).**
- Phong cách (B4.4): tính MỘT lần trên toàn file; unit dựng trên mọi segment không rỗng (làm ngữ cảnh), chỉ đếm unit gồm toàn segment `usable` (B4.2), độc lập với region/cặp.
- Tương tác (B4.5): cắt region (B4.1 bước 3), với mỗi cặp speaker trong region chạy `analyze()` và che người thứ ba (B4.1 bước 4-5).

**P3. Qwen.** Chi tiết ở A4. Chỉ chạy cho speaker đủ exposure phong cách.

**P4. Gộp.** Áp ngưỡng B5 theo quy tắc B2, dựng `PersonaProfile` và `PersonaStats` (B7).

**P5. Ghi và checkpoint.** `checkpoint.namespaces["persona_profile"] = svc.checkpoint_namespace_for(source_digest)`; `checkpoint.save` chỉ khi pass trọn vẹn (A6); `stage_out.write_persona_profile(profile, stats)` ghi đúng hai file `07c_persona_profile/profile.json` và `stats.json` (B1).

## A4. Dùng Qwen

**Nguồn LLM.** Dùng chung instance `refinement_svc` như `TranscriptReviewService`. Ở `main.py:911-922`, `review_svc` được dựng bằng `TranscriptReviewService(refinement_svc, logger=logger, **review_cfg)` chỉ khi `opt_in_step_enabled(args, "transcript_review")`. `persona_svc = PersonaProfileService(refinement_svc, logger=logger, **models.persona_profile)` dựng trong cùng khu vực, chỉ khi `opt_in_step_enabled(args, "persona_profile")`. Config chỉ có ba khoá `llm_enabled` (mặc định true), `thinking` (mặc định false), `max_new_tokens` (mặc định 1024). KHÔNG quay về `models.relabel` (các khoá cửa sổ của relabel không áp dụng cho stage này). Khoá lạ làm constructor nâng `TypeError` (như relabel/review), để gõ nhầm không bị bỏ qua.

**Phiên bản prompt.** Hằng `PROFILE_PROMPT_VERSION = "persona-v1"` trong `services/persona_profile_service.py`, ngang vai `REVIEW_PROMPT_VERSION = "review-v3"` (`transcript_review_service.py:49`). Đổi prompt hay luật xác nhận thì tăng hằng này; nó nằm trong namespace checkpoint.

**Mẫu hỏi.** Mỗi speaker đủ exposure phong cách (`speech_min >= MIN_EXPOSURE_MIN`, B4.4) có đúng một "cửa sổ" hỏi (`stats.llm_windows` đếm số cửa sổ này). Cửa sổ chỉ chứa lời của speaker đó (các unit không phải backchannel, B4.4), lấy cách đều theo `start` tới `LLM_WINDOW_CHARS` ký tự, nên không cần đặt tên A/B hay ánh xạ ngược. Speaker không đủ exposure không được hỏi: `llm_candidate_forms` và `description` là `null`.

**Prompt.** Kèm các danh sách từ điển hiện có (để Qwen không lặp lại) và các phần số đã tính của speaker (nhịp, độ dài câu, các form có đếm > 0). Yêu cầu trả JSON `{"new_forms":[...],"description":"..."}`: `new_forms` chỉ liệt kê từ/cụm (từ đệm, tiểu từ cuối câu hoặc xưng hô) xuất hiện nguyên văn trong chữ và chưa có trong từ điển; `description` là 1-2 câu tiếng Việt.

**Ngân sách token.** `thinking: false`, `max_new_tokens` mặc định 1024. Số cửa sổ mỗi lần gọi: `min(llm.batch_size, max(1, llm.max_batch_tokens // 4000))` khi `max_batch_tokens` có đặt (cùng dạng `_windows_per_call()`, `transcript_review_service.py:368-373`; 4000 là ước lượng token của một cửa sổ `LLM_WINDOW_CHARS`).

**Gọi và retry.**
- Dùng `ask_in_batches` (`utils/llm_batches.py:66`), trả `(replies, unanswered)`.
- Trả lời không đọc được: theo mẫu `_ask` (`transcript_review_service.py:502-531`), retry một lần với hậu tố bắt buộc JSON. Parse bằng `objects_in` (`utils/llm_json.py:108`); trả lời "đọc được" là dict có khoá `new_forms` hoặc `description`. `is_cut_in_thought` (`llm_json.py:75`) được cảnh báo "nâng max_new_tokens".

**Xác nhận bằng code (không tin Qwen).** Một form trong `new_forms` chỉ được giữ nếu: sau NFC và `_fold` (B4.4) gồm 1-3 token; không trùng form nào của ba từ điển; và dãy token của nó xuất hiện (ranh giới token, không phải chuỗi con) trong chữ thật của đúng speaker đó. Không dùng `find_phrase` (`transcript_review_service.py:218-236`): nó so chuỗi con và sẽ khớp "ạ" trong "bạn". Form hợp lệ được khử trùng, sắp theo thứ tự chuỗi, giữ tối đa `MAX_CANDIDATE_FORMS`, ghi vào `style.llm_candidate_forms`. Chúng KHÔNG được đếm vào `fillers`/`particles`/`address_terms` hay bất kỳ số liệu nào.

## A5. Tích hợp code

| File | Thay đổi |
|---|---|
| `services/persona_profile_service.py` (mới) | Lớp `PersonaProfileService(llm, logger, llm_enabled=True, thinking=False, max_new_tokens=1024)`. Hằng: `PROFILE_PROMPT_VERSION`, `PARAMS_VERSION`, mọi ngưỡng và từ điển ở B5, prompt hệ thống. Thuộc tính `checkpoint_namespace_for(source_digest)` (chuỗi an toàn như `transcript_review_service.py:332-345`: `PROFILE_PROMPT_VERSION`, `think/nothink`, `llm/nollm`, tên model, `new{max_new_tokens}`, `PARAMS_VERSION`, `source_digest`). Phương thức `profile(segments, recording_id) -> ProfileResult` chỉ tính, không ghi (cùng nguyên tắc "compute rồi caller checkpoint" của `review()`). Hàm thuần cho P0-P2, P4 để test không cần GPU. |
| `services/pipeline_service.py` | (a) `__init__` (dòng 50, 64): nhận `persona_svc=None` cạnh `review_svc`. (b) Thêm `_profile_personas(self, args, checkpoint, stage_out, transcripts, audio_path, *, allow_compute)` sau `_review_transcript` (dòng 975-1009), theo mẫu `_align_words` (dòng 1011-1042; `allow_compute` ở 1011-1012, 1031-1034): đặt namespace, `load_if_committed`; có checkpoint thì log, ghi lại hai file nếu thiếu và trả về; không có checkpoint mà `allow_compute` sai thì log và trả về (không nạp Qwen, không ghi `profile.json`); ngược lại tính, lưu, `stage_out.write_persona_profile`. (c) Trong `run()`, chèn khối "7b''" ngay sau khối stop của `transcript_review` (sau dòng 1812) và trước "7c" (dòng 1815): `persona_on = (self.persona_svc is not None and opt_in_step_enabled(args, "persona_profile"))`; `allow_compute = not bool(getattr(args, "postprocess_only", False)) and (getattr(args, "batch_continues", None) is None or getattr(args, "stop_after", None) == "persona_profile")` (`postprocess_only` chỉ được gán ở dòng 1822, sau điểm chèn nên phải đọc lại từ `args`; `batch_continues` chỉ được gán trong pass của `run_batch_by_stage`, `utils/batch.py:258`, nên chạy từng file, kể cả với `--stop_after word_alignment`/`conversation_exports`, vẫn tính, còn trong batch theo stage chỉ pass `persona_profile` mới tính); chạy `_profile_personas`; nếu `stop_after == "persona_profile"` thì `_release_llm_after(args, "persona_profile")`, log, `stage_out.write_manifest({... "stopped_after": "persona_profile"})`, return `transcripts` (đúng khuôn dòng 1806-1812). Các pass sau (`conversation_selection`, `word_alignment`, `conversation_exports`, `postprocess_only`) đi qua khối này nhưng `allow_compute` sai nên chỉ nạp checkpoint, không bao giờ nạp lại Qwen. (d) `_STOP_ORDER` (dòng 592-595): thêm `"persona_profile"` giữa `"transcript_review"` và `"conversation_selection"`. (e) `_next_llm_step_after` (dòng 484-506): thêm `persona_on` (chỉ tính khi `persona_svc.llm_enabled`, vì tắt LLM thì stage không dùng Qwen); `refinement`/`speaker_relabel`/`transcript_review` trả `persona_profile` nếu bật; thêm nhánh `stage == "persona_profile"` trả `conversation_selection` nếu `selection_on`. (f) `_release_llm_after` (dòng 527): thêm `"persona_profile"` vào tuple `("speaker_relabel", "transcript_review")`. (g) `parallel_stage_view` (dòng 220): thêm `"persona_profile"` vào danh sách `cross_file`. (h) Điều kiện gỡ LLM cuối `run()` (dòng 1917-1920): thêm `or persona_on`. |
| `utils/batch.py` | (a) `PIPELINE_STAGES` (dòng 144-148): chèn `"persona_profile"` giữa `"transcript_review"` và `"conversation_selection"`. (b) `_opt_in_stages` (dòng 197-198): thêm `"persona_profile"` để đi qua `opt_in_step_enabled(args, stage)` thay vì `step_enabled` (nếu quên, `step_enabled` coi bước chưa liệt kê là BẬT, `utils/steps.py:26-32`, và pass sẽ chạy dù profile tắt). (c) `_stage_parallelism` (dòng 461): thêm vào tuple `("speaker_relabel", "transcript_review")`, dùng chung `relabel_files_in_flight`. (d) Vòng lặp stage-major không cần sửa thêm: mỗi phần tử của `PIPELINE_STAGES` là một pass thật, `stage_args.stop_after = stage` (dòng 255) và `batch_continues = stop_gate != stage` (dòng 258) đã đủ. |
| `main.py` | (a) `--stop_after` choices (dòng 142-144): thêm `"persona_profile"`. (b) Sau khối dựng `review_svc` (dòng 911-922): dựng `persona_svc` chỉ khi `opt_in_step_enabled(args, "persona_profile")`, import trễ, cấu hình theo A4. (c) Truyền `persona_svc=persona_svc` vào `PipelineService(...)` (dòng 946-964, cạnh `review_svc=review_svc` ở dòng 960). Khoá `steps.persona_profile` tự thành `args.step_persona_profile` qua vòng `setattr` ở dòng 248-249, và `--steps persona_profile=on` dùng cùng đường. |
| `services/stage_output_service.py` | Thêm `"persona_profile": "07c_persona_profile"` vào `STAGES` (dòng 99-110, sau `transcript_review: "07b_transcript_review"`). Sửa chữ ký `_write_json` (dòng 125-134) thành `_write_json(self, path, payload, *, allow_nan=True)`: serialize bằng `text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=allow_nan)` ra chuỗi TRƯỚC, rồi mới `open(path, "w")` và ghi `text`, để payload có NaN/inf làm `json.dumps` lỗi trước khi file được mở (bản hiện tại mở file "w" rồi mới `json.dump`, nên lỗi để lại file viết dở); mặc định `allow_nan=True` giữ hành vi cũ cho mọi stage khác. Thêm `write_persona_profile(profile: dict, stats: dict)`: tôn trọng cờ `enabled`, ghi đúng hai file `profile.json` và `stats.json` với `allow_nan=False` (nếu `_write_json` trả False thì raise `RuntimeError`), rồi ghi manifest như `_finish` (dòng 293-294). Không ghi `report.json`/`replies.json`; trả lời thô của Qwen chỉ ghi log mức debug. |
| `config.json` | Thêm `steps.persona_profile` (opt-in; thiếu hoặc `false` là tắt) vào khối `environments.<env>.steps` cần bật (khối `a100` có `transcript_review: true` ở dòng 250; khối `kaggle` bắt đầu dòng 25 và không có `transcript_review`, xem C7). Thêm khối `models.persona_profile` cạnh `models.transcript_review` (dòng 366-373), chứa đúng ba khoá `llm_enabled`, `thinking`, `max_new_tokens`. |
| `utils/performance_report.py` | `_STAGE_ORDER` (dòng 10-13): thêm `"persona_profile"` sau `"transcript_review"`, để báo cáo hiệu năng không bỏ sót stage. |
| `tests/` | Theo B8. Cập nhật `tests/test_batch.py` (dòng 97-136) và `tests/test_review_select_align_order.py` (dòng 276-336). |

## A6. Checkpoint, phiên bản, idempotent, chạy lại

**Cơ chế.** `CheckpointManager` (`utils/checkpoint.py:33-95`) lưu theo `job_dir/<stage>/<namespace>/result.pkl`. `checkpoint.namespaces[stage]` chọn thư mục con (dòng 43-44). `save` ghi atomic (tạo file tạm, `os.replace`, ghi manifest có digest, dòng 48-71). Mẫu của review: `checkpoint.namespaces["transcript_review"] = svc.checkpoint_namespace` (`pipeline_service.py:982`), `load_if_committed` (dòng 983).

**Phiên bản.** `PARAMS_VERSION` là chuỗi viết tay trong code, ghi vào `profile.json`. Nó PHẢI đổi khi đổi bất kỳ ngưỡng, từ điển (B5), `EventParams()` mặc định, hay công thức nào ảnh hưởng số. Một test chống quên (B8 #16) so dấu vân tay của các hằng và từ điển với giá trị ghi sẵn. `PROFILE_PROMPT_VERSION` đổi khi đổi prompt. Stage không đọc ngưỡng từ config và dùng `EventParams()` mặc định cố định, nên mọi thứ ảnh hưởng số đều nằm trong code và trong `source_digest`.

**Namespace** gồm: `PROFILE_PROMPT_VERSION`, `think/nothink`, `llm/nollm`, tên model, `new{max_new_tokens}`, `PARAMS_VERSION`, `source_digest` (P0). Đổi bất kỳ giá trị nào thì namespace đổi, checkpoint cũ không bị dùng nhầm và cũng không bị xoá (đúng tinh thần comment ở `utils/checkpoint.py:41-42`).

**`NOISE_NOTICEABLE`.** Biến môi trường này, đọc một lần lúc import thành hằng `noise_map.NOTICEABLE` (`utils/noise_map.py:49`), đổi `noise_kind` do `annotate` ghi. Stage KHÔNG dùng `noise_kind` mà tính lại `dominant_kind(seg.noise_breakdown, floor=NOISE_FLOOR)` (`noise_map.py:57-78`) với `NOISE_FLOOR = 0.10` là hằng của stage. `noise_breakdown` không phụ thuộc biến môi trường, nên số liệu chỉ đổi khi đổi hằng này (kéo theo đổi `PARAMS_VERSION`).

**Checkpoint lưu gì.** Lưu `{"profile": ..., "stats": ...}` (dict đã dựng, không lưu transcript). Stage không sửa transcript nên không có bước "áp lại" khi nạp; nạp xong chỉ cần ghi lại file đầu ra nếu thiếu.

**Khi nào KHÔNG lưu** (để lần sau chạy lại): `llm_enabled` nhưng LLM không nạp được, hoặc có cửa sổ không trả lời / không đọc được (cùng dạng điều kiện `pipeline_service.py:1004-1005`). `llm_enabled: false` thì phần số là kết quả cuối và được lưu. Chỉ pass có `allow_compute` (A5) mới lưu.

**Idempotent.** Stage chỉ đọc `transcripts` thật; `annotate` chạy trên bản sao. Output không chứa thời điểm tạo hay id ngẫu nhiên, nên cùng input cho ra file giống từng byte (test B8 #15).

**Chạy lại.** Đổi hằng, từ điển, prompt hay transcript đều đổi namespace, nên tính lại toàn bộ kể cả gọi Qwen. Không tách cache riêng cho phần LLM ở bản đầu.

## A7. Xử lý lỗi và trường hợp biên

| Trường hợp | Hành vi |
|---|---|
| `speaker_relabel` hoặc `transcript_review` tắt | Vẫn chạy. `review_flags is None` thì không loại gì ở lý do `review_flag`; `stats.warnings` có `not_reviewed`. Nhãn speaker giữ như diarizer. |
| LLM tắt (`llm_enabled: false`) hoặc `ensure_loaded()` False | Phần số và ba danh sách từ điển vẫn ghi đầy đủ. `llm_candidate_forms` và `description` là `null`. Tắt bằng cấu hình thì checkpoint; LLM không nạp được thì không checkpoint. |
| LLM chạy nhưng không có form mới hợp lệ | `llm_candidate_forms` là `[]` (khác `null`, cùng quy ước `review_flags`). |
| Một số cửa sổ LLM lỗi | Speaker không có trả lời đọc được thì `llm_candidate_forms` và `description` là `null`; danh sách từ điển không bị ảnh hưởng. `stats.llm_ok < llm_windows`. Không checkpoint, chạy lại lần sau. |
| File 1 speaker | Mọi chỉ số tương tác có exposure 0 nên `value = null`; phong cách vẫn tính (B4.4). |
| >2 speaker | Chạy `analyze` theo từng cặp trong region, che người thứ ba (B4.1 bước 4, B4.6). |
| Không có gap hợp lệ | `response_latency_s` có ba phân vị `null`, `n` thật (có thể 0). Không điền 0. |
| Speaker quá ít dữ liệu | Vẫn có mục trong `profile.json`, đủ khoá, giá trị `null` theo ngưỡng B5; `n`/`count`/`exposure_min` giữ số thật. |
| `transcripts` rỗng hoặc None | Không ghi checkpoint, không ghi `profile.json` (giống `review()`, `transcript_review_service.py:378-380`). |
| Pass sau chạy lại `run()` (`allow_compute` sai) | Chỉ nạp checkpoint; không có thì bỏ qua, không nạp Qwen, không ghi đè `profile.json` đã có. Vì thế một pass chạy lại trong batch (không phải lựa chọn `--stop_after` riêng; `main.py:142-144` không có `conversation_selection`) sẽ không sinh profile. |
| Lỗi ngoại lệ trong `PersonaProfileService` | Bắt ở `_profile_personas`, ghi log, không checkpoint; stage sau vẫn chạy vì stage này chỉ đọc. |
| `--stop_after persona_profile` | Dừng và `write_manifest(... "stopped_after": "persona_profile")`; `main.py` coi là chạy dở, không đánh dấu file xong (`main.py:1013-1021`). |

---

# PHẦN B. Hợp đồng dữ liệu, công thức, test

Mọi đường dẫn tương đối so với `podcast-pipeline/`. Số dòng ghi theo trạng thái repo lúc viết (nhánh `solid-architecture`).

Nguyên tắc xuyên suốt: mọi con số do code tính; Qwen chỉ viết `description` và đề xuất form để mở rộng từ điển offline. **Thiếu dữ liệu không bao giờ được ghi thành 0; 0 chỉ có nghĩa "có cơ hội làm mà không làm".**

---

## B1. Cây thư mục và vai trò từng file

Tên thư mục: `07c_persona_profile/`. Lý do: `StageOutputService.STAGES` (`services/stage_output_service.py:99-110`) đang có `07b_transcript_review` và `08_word_alignment`; stage mới chạy sau review, trước select/align/export nên chèn `"persona_profile": "07c_persona_profile"` giữa hai dòng đó, không đánh số lại các thư mục sau (C5).

```
<output_dir>/
└── 07c_persona_profile/
    ├── profile.json     # kết quả chính: hồ sơ từng người của một file ghi âm
    └── stats.json       # chẩn đoán của lượt chạy: giây bị loại, LLM chạy được bao nhiêu, cảnh báo
```

| File | Vai trò | Ai đọc |
|---|---|---|
| `profile.json` | Hồ sơ từng người nói của MỘT file ghi âm. Khóa định danh hồ sơ = `recording_id` + nhãn speaker (nhãn sau relabel, tức `seg.speaker`; `speaker_original` bị bỏ qua). Không có `persona_id`, không có thời điểm tạo. | Bước gom persona sau này |
| `stats.json` | Số liệu chạy: giây bị loại theo lý do, số mối nối không xác định bị bỏ qua, tình trạng LLM, cảnh báo. | Người vận hành |

Ghi bằng `StageOutputService._write_json` (`:125`, `ensure_ascii=False`, `indent=2`) qua writer mới `write_persona_profile(profile: dict, stats: dict)` (A5); không ghi thêm file nào khác (không `summary.txt`, không exemplars).

Cấu trúc theo file ghi âm: mỗi lượt chạy pipeline xử lý một file ghi âm (xem `folders` trong `run_test_all.sh`), nên `profile.json` có đúng một `recording_id`. Việc "chia theo file rồi theo người" nằm ở cấp gộp thư viện persona sau này, không ở stage này.

---

## B2. Bốn dạng giá trị và quy tắc null / 0 / []

| Dạng | Hình JSON | Khi nào là `null` | Khi nào là 0 / rỗng | Mẫu số |
|---|---|---|---|---|
| **Phân bố** | `{"p10":..,"p50":..,"p90":..,"n":int}` | `n < MIN_DIST_N` thì CẢ ba phân vị là `null`, `n` vẫn ghi số thật | Không có "0 = không làm": một phân bố có `n >= ngưỡng` luôn có số thật (kể cả giá trị âm, ví dụ `response_latency_s`) | `n` = số quan sát hợp lệ |
| **Tần suất** | `{"value":float\|null,"count":int,"exposure_min":float}` | `exposure_min < ngưỡng` thì `value=null`; `count` và `exposure_min` vẫn ghi thật | `value=0.0` khi `exposure_min >= ngưỡng` và `count == 0`: có cơ hội mà thật sự không làm | `exposure_min` = số phút cơ hội (định nghĩa riêng từng chỉ số ở B4) |
| **Tỉ lệ** | `{"value":float\|null,"n":int}` | `n < MIN_RATIO_N` thì `value=null`; `n` vẫn ghi | `value=0.0` khi `n >= ngưỡng` và tử số 0 | `n` = mẫu số |
| **Danh sách form** | (a) từ điển: `[{form, <số>}]` hoặc `null`; (b) Qwen: `[str]` hoặc `null` | (a) `null` CHỈ khi exposure của người này thiếu. Không phụ thuộc LLM. (b) `null` = LLM không chạy / không trả JSON đọc được cho người này, hoặc người này không đủ exposure để hỏi | (a) luôn đủ MỌI form của từ điển theo thứ tự cố định, form không thấy có giá trị `0`/`0.0`; không bao giờ là `[]`. (b) `[]` = LLM đã chạy, không có form mới hợp lệ | – |

Quy tắc cứng (đều có test ở B8):

1. Không bao giờ thay `null` bằng `0`/`0.0`/`[]` khi serialize, cũng không đọc `0` rồi coi là thiếu. `json.dumps` giữ `None→null`, `0.0→0.0`, `[]→[]`; dataclass không dùng `or 0`/`or []` ở bất kỳ đâu (kể cả `x or []` kiểu `review_flags_of` ở `utils/conversation_selection.py:74-77` vốn gộp None và [] – stage này KHÔNG dùng hàm đó cho việc phân biệt).
2. `count` và `exposure_min` trong tần suất, `n` trong phân bố/tỉ lệ luôn là số thật, kể cả khi `value=null`, để bước gom persona tự áp ngưỡng khác.
3. Cùng quy ước với `TranscriptSegment.review_flags` (`schemas/transcript.py:68-71`: None = chưa review, [] = đã review không thấy) và `gap_before` (`:42-47`, `utils/provenance.py:21-27`: None = không xác định, không phải 0).
4. Khi người nói không đủ dữ liệu, các khối vẫn hiện diện với `null`; không bỏ khóa. Schema luôn đủ khóa.
5. Số float làm tròn 3 chữ số khi ghi (làm tròn sau khi so ngưỡng, không trước).

---

## B3. Schema `profile.json`

### B3.1 Cấp file

| Trường | Kiểu | Đơn vị | Nguồn | Ghi chú |
|---|---|---|---|---|
| `schema_version` | int | – | hằng số trong code | Đổi khi đổi cấu trúc JSON. Bắt đầu `1`. |
| `params_version` | string | – | hằng `PARAMS_VERSION` trong code, viết tay | Đổi khi đổi bất kỳ hằng/từ điển/công thức nào ảnh hưởng số (A6). Ví dụ `"pp-2026.10.2"`. |
| `source_type` | `"real"` \| `"synthetic"` | – | hằng | Stage này luôn ghi `"real"`. `"synthetic"` chỉ do bước kiểm định sau ghi; bước gom persona tự kiểm trường này và từ chối hồ sơ `synthetic`. |
| `recording_id` | string | – | `splitext(basename(audio_path))[0]` | Cùng tên với file ghi âm của lượt chạy. |
| `usable_ratio` | float \| null | 0..1 | code | `Σ usable_s / Σ speech_s` trên mọi speaker (B4.3). `null` nếu tổng giây nói = 0. |
| `speakers` | object | – | code | Khóa = nhãn speaker. |

### B3.2 Cấp người (`speakers[<nhãn>]`)

| Trường | Kiểu | Đơn vị | Dạng giá trị | Nguồn |
|---|---|---|---|---|
| `speech_s` | float | giây | số thường (luôn có) | code: hợp (union) khoảng `[start,end]` mọi segment của speaker, KỂ CẢ segment bị loại; đây là lượng nói thô |
| `style.speech_rate_wps` | object | từ/giây (từ = âm tiết tách bởi dấu cách, xem B4.4) | Phân bố | code |
| `style.words_per_sentence` | object | từ/câu | Phân bố | code |
| `style.fillers` | list \| null | `[{form:str, per_min:float}]` | Danh sách (a) | code, từ điển `FILLERS` |
| `style.particles` | list \| null | `[{form:str, per_100_sent:float}]` | Danh sách (a) | code, từ điển `PARTICLES` |
| `style.address_terms` | list \| null | `[{form:str, count:int}]` | Danh sách (a) | code, từ điển `ADDRESS_TERMS` |
| `style.llm_candidate_forms` | list \| null | `[str]` | Danh sách (b) | Qwen đề xuất, code xác nhận (A4); KHÔNG vào số liệu, chỉ để mở rộng từ điển offline |
| `style.description` | string \| null | – | LLM tự do | Qwen viết trên số liệu đã tính; `null` nếu LLM lỗi hoặc người này không đủ dữ liệu |
| `backchannel.per_min_of_other` | object | lần/phút lời của người kia | Tần suất | code: event `backchannel` |
| `backchannel.lexicon` | list \| null | `[{form:str, share:float}]`, tổng share = 1 | số liệu code | code: nhóm theo `_fold(text)` của các event `backchannel`; `null` nếu số backchannel < `MIN_LEXICON_N` |
| `backchannel.overlap_ratio` | object | tỉ lệ 0..1 | Tỉ lệ | code, `n` = số backchannel |
| `interruption.per_hour_of_other` | object | lần/giờ lời của người kia | Tần suất | code: event `interruption` có `by == speaker` |
| `interruption.yield_rate` | object | tỉ lệ 0..1 | Tỉ lệ | code, `n` = số lần người này bị cắt vào (xem C1) |
| `turn.duration_s` | object | giây | Phân bố | code |
| `turn.response_latency_s` | object | giây, ÂM = vào trước khi người kia dứt | Phân bố | code |

Lưu ý: `lexicon` không có LLM, dùng quy ước `null` khi thiếu mẫu; `[]` không xảy ra (nếu đủ mẫu thì luôn có ≥ 1 form vì mẫu số là chính các backchannel đó).

### B3.3 Ví dụ mô phỏng (3 speaker, 3 trạng thái)

`SPEAKER_00` có hành vi; `SPEAKER_01` đủ dữ liệu nhưng không ngắt lời nào (value 0.0) và Qwen không thấy form mới (`[]`); `SPEAKER_02` quá ít dữ liệu nên mọi thứ `null`. (Ví dụ minh họa: ba người cùng một file ghi âm vì khả thi trong dữ liệu thật; số liệu của họ được tính theo từng cặp, B4.1 bước 4.) Với mỗi người có số liệu, `backchannel.per_min_of_other` và `interruption.per_hour_of_other` dùng cùng `exposure_min` (cùng `other_talk_s`, B4.5).

```json
{
  "schema_version": 1,
  "params_version": "pp-2026.10.2",
  "source_type": "real",
  "recording_id": "podcast_ep017",
  "usable_ratio": 0.959,
  "speakers": {
    "SPEAKER_00": {
      "speech_s": 1520.4,
      "style": {
        "speech_rate_wps": {"p10": 2.4, "p50": 3.6, "p90": 4.9, "n": 412},
        "words_per_sentence": {"p10": 4.0, "p50": 11.0, "p90": 24.0, "n": 305},
        "fillers": [
          {"form": "ừm", "per_min": 1.2}, {"form": "ờm", "per_min": 0.0},
          {"form": "ờ", "per_min": 1.8}, {"form": "ưm", "per_min": 0.0},
          {"form": "hừm", "per_min": 0.0}, {"form": "đại loại", "per_min": 0.3}
        ],
        "particles": [
          {"form": "nhé", "per_100_sent": 6.23}, {"form": "nhỉ", "per_100_sent": 1.967},
          {"form": "nha", "per_100_sent": 1.311}, {"form": "nhá", "per_100_sent": 0.0},
          {"form": "nè", "per_100_sent": 0.0}, {"form": "nhể", "per_100_sent": 0.0},
          {"form": "hả", "per_100_sent": 0.328}, {"form": "hử", "per_100_sent": 0.0},
          {"form": "ạ", "per_100_sent": 0.0}, {"form": "chứ", "per_100_sent": 0.0}
        ],
        "address_terms": [
          {"form": "anh", "count": 37}, {"form": "chị", "count": 0}, {"form": "em", "count": 0},
          {"form": "mình", "count": 21}, {"form": "bạn", "count": 0}, {"form": "tớ", "count": 0},
          {"form": "cậu", "count": 0}, {"form": "tôi", "count": 3}
        ],
        "llm_candidate_forms": ["tức là", "đúng không"],
        "description": "Nói nhanh, giọng dẫn dắt, hay chốt ý bằng 'nhé'; xưng 'mình', gọi người kia là 'anh'."
      },
      "backchannel": {
        "per_min_of_other": {"value": 0.9, "count": 18, "exposure_min": 20.0},
        "lexicon": [{"form": "ừ", "share": 0.611}, {"form": "dạ", "share": 0.278}, {"form": "đúng rồi", "share": 0.111}],
        "overlap_ratio": {"value": 0.667, "n": 18}
      },
      "interruption": {
        "per_hour_of_other": {"value": 6.0, "count": 2, "exposure_min": 20.0},
        "yield_rate": {"value": 0.8, "n": 10}
      },
      "turn": {
        "duration_s": {"p10": 1.6, "p50": 5.8, "p90": 17.2, "n": 96},
        "response_latency_s": {"p10": -0.4, "p50": 0.3, "p90": 1.2, "n": 88}
      }
    },
    "SPEAKER_01": {
      "speech_s": 1310.0,
      "style": {
        "speech_rate_wps": {"p10": 2.1, "p50": 3.1, "p90": 4.2, "n": 377},
        "words_per_sentence": {"p10": 5.0, "p50": 14.0, "p90": 31.0, "n": 270},
        "fillers": [
          {"form": "ừm", "per_min": 0.0}, {"form": "ờm", "per_min": 0.9},
          {"form": "ờ", "per_min": 0.0}, {"form": "ưm", "per_min": 0.0},
          {"form": "hừm", "per_min": 0.0}, {"form": "đại loại", "per_min": 0.0}
        ],
        "particles": [
          {"form": "nhé", "per_100_sent": 0.0}, {"form": "nhỉ", "per_100_sent": 0.741},
          {"form": "nha", "per_100_sent": 0.0}, {"form": "nhá", "per_100_sent": 0.0},
          {"form": "nè", "per_100_sent": 0.0}, {"form": "nhể", "per_100_sent": 0.0},
          {"form": "hả", "per_100_sent": 0.0}, {"form": "hử", "per_100_sent": 0.0},
          {"form": "ạ", "per_100_sent": 9.259}, {"form": "chứ", "per_100_sent": 0.0}
        ],
        "address_terms": [
          {"form": "anh", "count": 30}, {"form": "chị", "count": 0}, {"form": "em", "count": 44},
          {"form": "mình", "count": 5}, {"form": "bạn", "count": 0}, {"form": "tớ", "count": 0},
          {"form": "cậu", "count": 0}, {"form": "tôi", "count": 0}
        ],
        "llm_candidate_forms": [],
        "description": "Nói chậm, lễ phép, câu dài; xưng 'em'."
      },
      "backchannel": {
        "per_min_of_other": {"value": 1.403, "count": 31, "exposure_min": 22.1},
        "lexicon": [{"form": "dạ", "share": 0.516}, {"form": "vâng", "share": 0.323}, {"form": "ừ", "share": 0.161}],
        "overlap_ratio": {"value": 0.355, "n": 31}
      },
      "interruption": {
        "per_hour_of_other": {"value": 0.0, "count": 0, "exposure_min": 22.1},
        "yield_rate": {"value": 0.889, "n": 9}
      },
      "turn": {
        "duration_s": {"p10": 2.0, "p50": 7.4, "p90": 21.5, "n": 83},
        "response_latency_s": {"p10": 0.1, "p50": 0.6, "p90": 1.5, "n": 80}
      }
    },
    "SPEAKER_02": {
      "speech_s": 41.7,
      "style": {
        "speech_rate_wps": {"p10": null, "p50": null, "p90": null, "n": 9},
        "words_per_sentence": {"p10": null, "p50": null, "p90": null, "n": 4},
        "fillers": null,
        "particles": null,
        "address_terms": null,
        "llm_candidate_forms": null,
        "description": null
      },
      "backchannel": {
        "per_min_of_other": {"value": null, "count": 1, "exposure_min": 1.1},
        "lexicon": null,
        "overlap_ratio": {"value": null, "n": 1}
      },
      "interruption": {
        "per_hour_of_other": {"value": null, "count": 0, "exposure_min": 1.1},
        "yield_rate": {"value": null, "n": 0}
      },
      "turn": {
        "duration_s": {"p10": null, "p50": null, "p90": null, "n": 3},
        "response_latency_s": {"p10": null, "p50": null, "p90": null, "n": 3}
      }
    }
  }
}
```

Kiểm tra ví dụ theo quy tắc B2/B4: `SPEAKER_00` backchannel `18 / 20.0 = 0.9`, interruption `2 / (20.0 / 60) = 6.0` lần/giờ (exposure 20.0 ≥ `MIN_EXPOSURE_INTERRUPTION_MIN`), `yield_rate` `8/10`, lexicon 11+5+2 = 18 backchannel. `SPEAKER_01.interruption.per_hour_of_other.value = 0.0` (exposure 22.1 phút ≥ 20, count 0) khác hẳn `SPEAKER_02` (`value = null`, exposure 1.1 phút < ngưỡng); `yield_rate` `8/9`; lexicon 16+10+5 = 31. `fillers` của hai người đầy đủ 6 form kể cả form `0.0`; `SPEAKER_02` có `speech_s` 41.7 giây (0.7 phút) < `MIN_EXPOSURE_MIN` nên ba danh sách từ điển là `null`, không hỏi Qwen nên `llm_candidate_forms` và `description` là `null`. `SPEAKER_01.llm_candidate_forms: []` (Qwen đã chạy, không form mới hợp lệ) khác `null`.

---

## B4. Công thức chính xác từng chỉ số

### B4.1 Dữ liệu vào và chuẩn bị

Đầu vào: bản sao đã qua P0 (A3): `words=None`, text NFC, sắp theo `start`, đã `annotate` (có `gap_before`, `crosses_cut`, `noise_breakdown`). `EventParams()` mặc định cố định (`utils/conversation_events.py:31-46`; giá trị dùng: `ipu_gap` 0.2, `overlap_min` 0.2, `backchannel_max_seconds` 1.0, `backchannel_max_words` 2, `backchannel_context` 1.0, `yield_within_seconds` 1.5). Chúng trùng mặc định của `ConversationSelectionConfig` (`utils/conversation_selection.py:185-190`) nhưng stage không đọc config đó.

Các bước:

1. **Gán lý do loại cho từng segment** (đúng MỘT lý do theo thứ tự ưu tiên, để `excluded_s` cộng đúng; giá trị là tổng `end - start` của segment):

   | Thứ tự | Khoá `excluded_s` | Điều kiện | Xử lý |
   |---|---|---|---|
   | 1 | `review_flag` | `hard_review_kind(seg)` khác None (`utils/conversation_selection.py:80-84`: `speaker_turn`, `asr_hallucination`, `asr_garbage`), hoặc `foreign_seconds(seg) > 0` (`:87-91`, cờ `not_vietnamese`), hoặc `soft_review_count(seg) > 0` (`:94-95`, cờ `speaker_phrase`: seg.text có lời người kia) | ở lại làm ngữ cảnh, không đóng góp số liệu |
   | 2 | `music` | `seg.has_music` True (`schemas/transcript.py:21`) | như trên |
   | 3 | `noise` | `dominant_kind(seg.noise_breakdown, floor=NOISE_FLOOR)` thuộc `{"noise_speech","noise_env","noise_room"}`. `"clean"` (đã đo, không nhóm nào ≥ `NOISE_FLOOR`) và `None` (`noise_breakdown is None`, chưa đo) đều KHÔNG bị loại; `None` được đếm cho cảnh báo `noise_unmeasured` | như trên |
   | 4 | `unseparated_overlap` | `_unseparated_seconds(seg) > 0` (`:291-302`; nếu không đọc được thì cả segment nghi ngờ). Âm thanh còn trộn hai giọng nên chữ không tin được; backchannel/lexicon dựa trên chữ đó cũng không tin, nên loại hẳn cả chữ lẫn timing (C9) | như trên |
   | 5 | `empty_text` | `_fold(text) == ""` (`:257-258`) | bỏ khỏi mọi bước sau bước 3 (không phải lời nói; vẫn tham gia chia region ở bước 3 vì mang `gap_before`/`crosses_cut`), nhưng vẫn tính vào `speech_s` và `excluded_s` |

   Tập `usable` = segment không thuộc lý do nào trong 1-5. Dùng để chọn unit sạch cho phong cách (B4.2) và cho `usable_s`. Mọi lý do 1-4 đều che số liệu tương tác (bước 4).

2. **Bản sao không có `words`** (P0). Hệ quả: `_speech_pieces` (`utils/conversation_events.py:105-111`) coi cả segment là một mảnh, token = `text.split()`; `merged_backchannel` (`:274-286`) không bao giờ phát vì `_phrase_times` (`:176-186`) cần `words`. Timing luôn là mức segment (C2).

3. **Chia region liền mạch.** Duyệt TOÀN BỘ segment đã sắp, kể cả `empty_text` (chỉ bỏ chúng khỏi các bước sau khi đã chia xong region; `gap_before` đo so với `end` của segment liền trước kể cả segment rỗng, nên bỏ trước sẽ làm mất mối cắt đứng cạnh nó); bắt đầu region mới (a) tại mọi segment không phải đầu file có `gap_before is None` (`provenance.py:59-65`: None ở đầu file hoặc ngang chỗ cắt nhạc), (b) tại mọi segment `crosses_cut` (segment đó đứng thành region riêng, segment kế tiếp mở region mới). Lý do (b): `gap_before` chỉ so với `end` của segment liền trước (`provenance.py:63`), nên một mối cắt nằm TRONG segment `crosses_cut` của người này sẽ lọt qua khi xét cặp unit của hai người khác; tách region ở chính segment đó chặn cả hai phía. Vì segment `crosses_cut` đứng ngoài region kế tiếp, phần lời của nó trải sang region đó được che bằng M (bước 4(c)). `unknown_gaps_skipped` (stats) = số segment không phải đầu file có `gap_before is None` (cùng khái niệm `provenance.summary`, `:89-90`; đếm trên danh sách đầy đủ, gồm segment rỗng). Ví dụ A[0,5], cắt ở 6, E rỗng [7,7.5], B[8,9]: E có `gap_before=None` nên mở region mới và B (gap 0.5 so với E) nằm cùng E, tách khỏi A; quan sát A→B không xuất hiện. **`gap_before` None KHÔNG BAO GIỜ được đưa vào bất kỳ phân bố/độ trễ nào**: nó chỉ dùng để tách region.

4. **Chạy theo cặp trong region.** Với mỗi region R có ≥ 2 speaker và mỗi cặp speaker (X, Y) có mặt trong R:
   * Đầu vào `analyze(segments_XY, EventParams())` (`utils/conversation_events.py:189`) = mọi segment của X và Y trong R (kể cả loại, làm ngữ cảnh). Trong tập này `other_of` (`:199-201`) luôn xác định.
   * **Vùng che M** = hợp các khoảng `[start - ipu_gap, end + ipu_gap]` của (a) mọi segment trong R của speaker thứ ba, bất kể trạng thái nhưng KHÔNG gồm `empty_text` (lý do 5, không phải lời nói); (b) segment lý do 1-4 của X hoặc Y (cũng KHÔNG gồm `empty_text`); (c) mọi segment NGOÀI R mà khoảng `[start, end]` chồng thời gian lên `[R.start, R.end]`, theo mốc thời gian chứ không theo lời nên không xét lý do loại. Thực tế (c) chủ yếu là segment `crosses_cut` liền trước R (cũng có thể là segment của region trước kéo dài sang R; luật viết tổng quát nên vẫn đúng) (bước 3(b) đã tách nó khỏi R, còn `gap_before` của segment sau nó đo từ `end` của nó), nên lời nó trải sang R không bị đếm thành lời của R. Ví dụ A[0,10] `crosses_cut`, B[5,6] (3 từ), A2[10.1,14]: R mở ở B, M chứa `[-0.2, 10.2]` nên B bị trừ khỏi `other_talk_s(A)` và unit `u`=A2 của cặp (A,B) giao M nên bị bỏ, không sinh độ trễ 4.1 s giả.
   * **Phạm vi S** = `[R.start, R.end - EDGE_MARGIN_S]`, `R.start`/`R.end` = min `start`/max `end` của segment trong R. Bỏ cặp nếu S rỗng.
   * Một event / lượt / quan sát bị bỏ nếu khoảng thời gian của nó giao M, hoặc có `start` sau cuối S (không đủ thời gian biết người kia có nhường hay không).
   * Exposure `other_talk_s(X)` của cặp = độ dài của (hợp các unit không phải backchannel của Y) ∩ S, trừ M; đối xứng cho Y. Cùng một vùng M và S cho tử và mẫu nên hai bên cùng phạm vi.
   * Event dict chỉ mang `segment` = key của unit đầu (`:219,257`), không đủ để biết người giữ lượt; nên che theo thời gian.

5. **Không đếm trùng khi một speaker có nhiều đối tác.** Exposure của X là tổng `other_talk_s` qua các cặp; mỗi cặp chỉ chứa lời của đúng đối tác của nó, và lời của người thứ ba nằm trong M, nên mỗi giây lời của người khác được tính tối đa một lần. Event / lượt / quan sát độ trễ được khử trùng toàn file theo khoá `(loại, speaker, start, end)` (event: `type, by, start, end`; lượt: `X, start unit đầu, end unit cuối`; độ trễ: `X, u.start, u.end`) trước khi đếm, vì cùng một unit của X có thể xuất hiện ở hai cặp khi người thứ ba nói ngay trước/sau ngoài khoảng ε.

### B4.2 Đơn vị xử lý chung

* `Unit` = IPU (`utils/conversation_events.py:49-62`). `unit.backchannel` do `_mark_backchannels` (`:143-155`) gán: `duration ≤ 1.0 s`, `n_words ≤ 2`, `_fold(text) ∈ BACKCHANNELS` (`utils/conversation_selection.py:47-50`), VÀ người kia nói liền trước (trong 1.0 s) và liền sau (trong 1.0 s). Hệ quả đã biết: định nghĩa chặt, nên số đếm là **cận dưới**; lời "ừ" đứng một mình giữa hai lượt của cùng một người, hoặc khi diarization bỏ sót overlap, không được tính.
* **Đơn vị phong cách** (B4.4): `speech_units(non_empty, EventParams())` (`:114-136`) rồi `_mark_backchannels` (`:143-155`), tính MỘT lần trên toàn file, độc lập với region và cặp. `non_empty` = mọi segment không phải `empty_text`, kể cả segment lý do 1-4 (làm ngữ cảnh, để "ờ" đáp lời bên cạnh một lời bị loại của người kia vẫn được đánh dấu backchannel và lượt không bị ghép xuyên lời bị loại). Sau khi dựng, mọi phép đếm phong cách chỉ dùng **unit sạch** = unit mà mọi segment trong `unit.keys` đều `usable`; unit có segment bị loại bị bỏ khỏi đếm, và lượt (B4.4) chứa unit không sạch bị bỏ khỏi `words_per_sentence`. Hệ quả: một IPU ghép segment usable với segment bị loại cạnh nhau bị bỏ cả IPU (chấp nhận, thiên về an toàn). Backchannel ở đây xét với mọi speaker khác (chỉ để loại lời đáp ngắn khỏi phong cách; khác định nghĩa theo cặp ở B4.5, và chấp nhận sự khác này).
* **Đơn vị tương tác** (B4.5): unit do `analyze` trả về trong từng cặp.
* Backchannel unit KHÔNG đưa vào `speech_rate`, `words_per_sentence`, từ điển, `turn`.
* `unit.text` là token nối bằng dấu cách (`:110-111` khi timing segment, tức `seg.text.split()` nối lại); dấu câu ở lại trong token (dùng cho tách câu ở B4.4).
* Phân vị dùng nội suy tuyến tính kiểu NumPy (`method="linear"`), tự cài bằng thuần Python trong module để test không cần numpy; sắp xếp giá trị rồi `p = x[floor(k)] + (k-floor(k))*(x[ceil(k)] - x[floor(k)])`, `k = q*(n-1)`.

### B4.3 `speech_s`, `usable_ratio`, exposure

* `speech_s(X)` = độ dài hợp các đoạn `[start,end]` của mọi segment có `speaker == X` (dùng lại logic `_union_seconds`, `utils/conversation_selection.py:277-288`). Bao gồm cả segment bị loại.
* `usable_s(X)` = hợp các đoạn của segment `usable` của X.
* `usable_ratio` (cấp file) = `Σ usable_s(X) / Σ speech_s(X)` trên mọi speaker; `null` nếu mẫu số 0.
* **Exposure cho tần suất tương tác**: `other_talk_s` (B4.1 bước 4). **Exposure phong cách**: `speech_min` (B4.4).

### B4.4 `style`

Tính một lần trên đơn vị phong cách (B4.2); "unit" ở toàn mục này luôn là unit sạch (gồm toàn segment `usable`), unit chỉ làm ngữ cảnh không được đếm. `speech_min(X)` = tổng giây của các unit sạch không phải backchannel của X / 60.

**speech_rate_wps** (từ/giây; "từ" = token cách bởi dấu cách = âm tiết tiếng Việt, nên đơn vị thực là âm tiết/giây; tên trường giữ `wps`).

* Quan sát = mỗi unit không phải backchannel của X có `n_words ≥ RATE_MIN_WORDS` (3) và `duration ≥ RATE_MIN_UNIT_S` (1.0 s). Giá trị = `n_words / duration`.
* Vì mọi timing là mức segment, unit bắt đầu/kết thúc ở biên segment gồm cả khoảng lặng VAD đệm hai đầu, nên nhịp **thấp hơn một chút** nhịp thật. Gộp các segment cách nhau ≤ `ipu_gap` thành IPU không loại bỏ khoảng lặng này. Giá trị dùng để so sánh tương đối giữa các speaker trong cùng pipeline, không phải nhịp tuyệt đối.
* Cắt ngoại lai: giá trị ngoài `[RATE_MIN_WPS, RATE_MAX_WPS]` = `[0.5, 9.0]` coi là lỗi timing, bỏ, không tính vào `n`.
* Phân bố `{p10,p50,p90,n}`; `null` ba phân vị nếu `n < MIN_DIST_N`.

**words_per_sentence** (từ/câu).

* Nguồn câu: ghép chữ theo **lượt** của X (chuỗi cực đại các unit không phải backchannel của X liên tiếp theo `start` mà không có unit không-phải-backchannel của người khác xen vào) rồi nối bằng dấu cách. Không tách theo từng segment vì segment là mảnh VAD, không phải câu.
* Tách câu tiếng Việt (code thuần, không LLM): dùng regex `(?<!\d)[.!?…]+(?=\s|$)`, tức dấu kết câu `. ! ? …` (một hay nhiều) theo sau là khoảng trắng hoặc hết chuỗi, **không** ngay sau chữ số (để không cắt "3.5", "2.000"). Không tách ở dấu phẩy, dấu chấm phẩy, hay dấu hai chấm. Từ viết tắt có chấm (vd "TP.HCM") không có khoảng trắng sau chấm nên không bị cắt.
* Mảnh cuối của lượt không có dấu kết câu bị **bỏ** (có thể bị cắt giữa chừng do ranh giới lượt).
* Số từ của câu = số token cách bởi dấu cách sau khi bỏ token chỉ gồm dấu câu. Giữ câu có `SENTENCE_MIN_WORDS ≤ số từ ≤ SENTENCE_MAX_WORDS` (2..80); ngoài khoảng đó bỏ (tránh 1 từ do "Vâng." và lỗi ASR lặp).
* Phân bố `n` = số câu giữ lại (`n_sentences`); đây cũng là **mẫu số** của `particles`.

**fillers**, **particles**, **address_terms** (từ điển cố định, không phụ thuộc LLM).

Từ điển là hằng trong code (thứ tự dưới đây là thứ tự xuất ra), thuộc `PARAMS_VERSION`. Chọn bảo thủ: chỉ giữ form hầu như luôn có đúng vai trò đó; form đa nghĩa bị loại vì đếm sẽ trộn nghĩa từ vựng/ngữ pháp với vai trò cần đo.

| Danh sách | Form (theo thứ tự) | Form bị loại và lý do |
|---|---|---|
| `FILLERS` | ừm, ờm, ờ, ưm, hừm, đại loại | `là`, `thì`, `tức là`, `như là`, `kiểu như`: từ nối ngữ pháp thật, phần lớn lần xuất hiện là cú pháp; `kiểu`: danh từ ("kiểu" = loại); `ừ`: chủ yếu là lời đồng ý/trả lời, không phải do dự (có trong `BACKCHANNELS`). Nguyên tắc với form trong `BACKCHANNELS`: `ừm`, `ờm`, `ờ` vẫn giữ vì là tiếng do dự, và đếm filler đã loại unit backchannel nên không đếm trùng; `ừ` bị loại vì nghĩa chính là đồng ý; `à`: cảm thán/backchannel và cũng là hạt cuối câu |
| `PARTICLES` | nhé, nhỉ, nha, nhá, nè, nhể, hả, hử, ạ, chứ | `đấy`, `đó`, `thế`, `vậy`, `đâu`, `cơ`: đại từ chỉ định, từ nghi vấn hoặc danh từ khi đứng cuối câu ("ở đó", "như vậy", "ở đâu", "tập cơ"); `mà`, `thôi`, `đi`: liên từ/phó từ/động từ ("nhưng mà", "thôi!", "đi chợ"); `à`: như trên |
| `ADDRESS_TERMS` | anh, chị, em, mình, bạn, tớ, cậu, tôi | `con`, `cô`: loại từ hoặc danh từ chung ("con mèo", "cô đơn"); `ông`, `bà`: vừa là xưng hô vừa là danh từ chỉ người ("ông ấy", "ông bà", "bà ngoại"), loại cả hai theo hướng bảo thủ; `chú`, `bác`: thường nằm trong "chú ý", "bác sĩ", "bác bỏ" |

* **Chuẩn hoá**: chữ và form đều NFC rồi `_fold` (`utils/conversation_selection.py:257-258`: hạ chữ thường, bỏ dấu câu). So khớp theo **ranh giới token**: dãy token của form phải là dãy con liên tiếp của dãy token đã `_fold` của chữ. Không dùng `find_phrase`.
* **Code luôn đếm đủ MỌI form** của từ điển và ghi cả form có giá trị 0; không top-N, không ngưỡng đếm tối thiểu, nên form vắng mặt không bao giờ xảy ra.
* **Phạm vi đếm**: chữ của các unit không phải backchannel của X (toàn file).
    * `fillers[f].per_min = count(f) / speech_min`. Một "ừ" đứng riêng bị đánh dấu backchannel nên không bị đếm là filler.
    * `particles[f].per_100_sent = 100 * count_cuối_câu(f) / n_sentences`, với `count_cuối_câu(f)` = số câu giữ lại (theo cách tách ở trên) mà token cuối (sau `_fold`) là `f`.
    * `address_terms[f].count = count(f)` toàn bộ chữ (số đếm thô, không chuẩn hoá theo phút, không phân biệt ngôi vì dùng để suy ra xưng hô, không phải tần suất).
* `null` CHỈ khi exposure thiếu: `fillers` và `address_terms` khi `speech_min < MIN_EXPOSURE_MIN`; `particles` khi `n_sentences < MIN_DIST_N`. Không phụ thuộc LLM, và LLM lỗi không làm các danh sách này `null`.

**llm_candidate_forms** (list[str] | null): xác nhận và ghi theo A4. `null` khi `llm_enabled` sai, LLM không nạp được, speaker không được hỏi (`speech_min < MIN_EXPOSURE_MIN`), hoặc trả lời không đọc được sau retry. `[]` khi trả lời đọc được mà không form nào hợp lệ.

**description** (string | null). Prompt của Qwen nhận phần số đã tính và trả một đoạn mô tả tiếng Việt (A4). `null` nếu: speaker không được hỏi hoặc LLM lỗi, hoặc `style.speech_rate_wps.n < MIN_DIST_N`, hoặc kết quả rỗng / dài quá `DESCRIPTION_MAX_CHARS` (400). Không có nguồn thứ hai kiểm chứng, nên bước gom persona không được dựng điều kiện logic dựa trên `description`.

### B4.5 `backchannel`, `interruption`, `turn`

Tất cả tính từ output `analyze()` của từng cặp trong region, sau che M và cắt S (B4.1 bước 4-5).

**Lời của người kia (`other_talk_s`)**: xác định ở B4.1 bước 4. Trên cả hồ sơ, `other_talk_s(X)` cộng qua mọi cặp có X. Cả `backchannel.per_min_of_other` và `interruption.per_hour_of_other` dùng đúng `exposure_min = other_talk_s(X) / 60` này.

**backchannel.per_min_of_other**

* Tử: số event `{"type":"backchannel","by":X}` (`utils/conversation_events.py:216-220`) sau che. Chỉ tính event có `in_turn == True` (`:220`: nằm trong lượt của người kia): backchannel không chạm lời người kia là sai nghĩa "listener word while the other speaker talks on".
* Mẫu: `exposure_min = other_talk_s(X) / 60`.
* `value = count / exposure_min`, `null` nếu `exposure_min < MIN_EXPOSURE_MIN` (5.0 phút). `value = 0.0` khi `exposure_min` đủ và `count = 0`.
* Cận dưới: định nghĩa backchannel chặt (B4.2), diarization bỏ sót overlap (người nghe "ừ" đè lên giọng người nói thường bị gộp vào segment người nói) và lý do 1-4 che số liệu nên thực tế ≥ số này. Cảnh báo ghi ở `stats.warnings`.

**backchannel.lexicon**

* Từ các event `backchannel` của X (đã khử trùng): `form = _fold(event.text)`, `share = count(form) / tổng số backchannel của X`. Sắp giảm dần; làm tròn sau khi sắp. Phần tử có `share` sau làm tròn là 0.0 vẫn giữ (tổng 1 trước làm tròn).
* `null` nếu tổng backchannel < `MIN_LEXICON_N` (5).

**backchannel.overlap_ratio** (Tỉ lệ)

* `n` = số event backchannel của X (cùng tập với `per_min_of_other`, `in_turn == True`).
* Tử = số trong đó `overlap > 0` (`event["overlap"]`, `:218`, = thời gian chồng giọng với người kia, `min(shared, duration)`).
* `value = tử / n`; `null` nếu `n < MIN_RATIO_N` (8). `n` khớp `backchannel.per_min_of_other.count`.

**interruption.per_hour_of_other**

* Tử: số event `{"type":"interruption","by":X}` (`:270-272`) sau che. KHÔNG tính `competitive_overlap` (cắt vào mà cả hai cùng nói tiếp, `:258-261`: X không giành được lượt); hạn chế đã biết: người hay nói đè kiểu cạnh tranh (cả hai cùng nói tiếp) có thể ra giá trị thấp hoặc 0 dù hành vi đè lời rõ rệt; công thức giữ nguyên, bước gom persona cần biết điều này; nó được tính ở mẫu số của `yield_rate` của người bị cắt.
* Hạn chế đã biết của `response_latency_s` (điều kiện 2): trao lượt có chồng giọng ≥ `overlap_min` mà Y chỉ nói một mình ngắn (vd X nhường sau 0.3 s, Y nói thêm 1.3 s, X đáp lại) bị bỏ. Đây là mất mẫu bảo thủ, không sinh giá trị sai.
* Mẫu: `exposure_min = other_talk_s(X) / 60` (cùng exposure với backchannel): `value = count / (exposure_min / 60)` = lần/giờ.
* `MIN_EXPOSURE_INTERRUPTION_MIN` = 20.0 phút (cao hơn 5.0 vì ngắt hiếm; 5 phút quá nhiễu cho tần suất theo giờ). `value = null` nếu `exposure_min` dưới ngưỡng này.

**interruption.yield_rate** (Tỉ lệ)

* Định nghĩa: *khi X đang nắm lượt mà bị cắt vào, X nhường bao nhiêu phần trăm* (C1).
* Mẫu = các event có `of == X` thuộc một trong hai loại: `interruption` (response `yield` hoặc `continue`, `:263-272`) và `competitive_overlap` (holder không dừng trong `yield_within_seconds`, `:258-261`). `competitive_overlap` là ca "không nhường", nếu bỏ nó thì tỉ lệ lệch về 1.0.
* `n` = số event đó; tử = số `interruption` có `response == "yield"` (vì `marked=None` nên `intent=None` và `adapt` không bao giờ xuất hiện, `:268-269`).
* `value = tử / n`; `null` nếu `n < MIN_RATIO_N`.

**turn.duration_s** (Phân bố)

* Lượt của X trong một cặp = chuỗi cực đại các unit không phải backchannel của X liên tiếp theo `start` mà không có unit không-phải-backchannel của Y nằm xen. Độ dài = `end` unit cuối − `start` unit đầu (gồm cả các khoảng lặng nội lượt ngắn hơn khoảng đổi lượt).
* Bỏ lượt đầu và lượt cuối của cặp trong region (bị cắt). Bỏ lượt nếu cả khoảng `[start lượt, end lượt]` giao M (không chỉ khi một unit giao M: nếu không, lời A1 – B – A2 của cặp (A,C) với khoảng lặng > `ipu_gap` bị ghép thành một lượt trùm cả lời B), hoặc `start` lượt sau cuối S.
* `n` = số lượt; `null` ba phân vị nếu `n < MIN_DIST_N`.

**turn.response_latency_s** (Phân bố; âm = X vào trước khi người kia dứt)

* Mỗi quan sát = một lần X nhận lượt từ Y: `u` là unit đầu của một lượt X (unit không phải backchannel liền trước theo `start` là của Y), và `p` = unit không phải backchannel của Y có `start ≤ u.start` và `end` muộn nhất. Quan sát = `u.start − p.end` (giây).
* Điều kiện giữ quan sát (chống lệch):
    1. `p.end ≤ u.end`: `u` không nằm trọn trong lời của Y (nếu không, X đáp chồng lên Y mà không "nhận lượt").
    2. Unit không phải backchannel của X liền trước `p` (nếu có) phải kết thúc không muộn hơn `p.start`. Điều kiện này chỉ áp dụng khi `p` thực sự chen vào lời X, tức `X_prev.end − p.start ≥ overlap_min` (0.2 s; chồng nhỏ hơn là trao lượt tự nhiên, ví dụ X_prev kết thúc 5.2, Y [5.1, 6.5], X đáp 6.8 thì quan sát 0.3 được giữ), và `p` là lời chen ngắn: `p.end ≤ X_prev.end + yield_within_seconds` (1.5 s). Nếu `p.end` muộn hơn mức đó thì Y đã giành lượt thật (X nhường) và quan sát được giữ. Chọn ngưỡng này vì dùng lại hằng số sẵn có, không cần khớp với event `interruption`. Khi áp dụng điều kiện, `p` là Y chen vào giữa lời X (ví dụ chen 3 từ không phải backchannel), `u` chỉ là phần X nói tiếp chứ không phải phản hồi một lượt, và nếu giữ sẽ ra giá trị âm/nhỏ giả. Các ca chen này đã được đo ở `interruption`.
    3. Khoảng `[p.start, u.end]` không giao M (phủ cả `p`, `u` và quãng giữa, nên lời người thứ ba chen vào giữa `p` và `u` làm bỏ quan sát); không thuộc lượt đầu/cuối; `u.start` không sau cuối S. Nếu giao M thì bỏ quan sát, không lùi về `p` sớm hơn. Ví dụ C → B → A: cặp (A,C) có `p`=C, khoảng `[C.start, u.end]` chứa lời B nên bị bỏ; chỉ cặp (A,B) cho độ trễ đúng, nên không còn hai giá trị khác nhau cùng khoá khử trùng `(X, u.start, u.end)`.
    4. Khoảng hợp lệ `[LATENCY_FLOOR_S, LATENCY_CAP_S]` = `[-3.0, +6.0]` giây. Ngoài khoảng đó không phải "phản ứng với lượt trước" (đổi chủ đề sau im lặng dài, hoặc ghép nhầm `p` quá xa); bỏ.
* Nguồn: **tự tính từ units**, không lấy event `turn_switch` và không lấy `gap_before`. Lý do:
    * Event `turn_switch` (`:288-300`) chỉ phát khi `unit.start ≥ reach` (`:294-298`), tức khoảng trống ≥ 0 và không overlap; không có giá trị âm. Nếu chỉ dùng nó, `response_latency_s` mất toàn bộ phần âm và lệch dương.
    * `seg.gap_before` (`provenance.py:59-68`) là khoảng trống giữa hai SEGMENT liền nhau (không phân biệt người nói và không phân biệt IPU nội/ngoại lượt); chỉ được dùng để tách region (B4.1 bước 3).
* Quan sát âm bao gồm cả các cú vào chồng nhỏ dưới `overlap_min` mà `analyze` không coi là `interruption`. Với overlap ≥ `overlap_min` đã thành event `interruption`/`competitive_overlap` thì độ trễ này vẫn được ghi nếu qua các điều kiện trên; hai thống kê độc lập và không cộng dồn.
* `n` = số quan sát; `null` ba phân vị nếu `n < MIN_DIST_N`.
* Độ chính xác: timing mức segment cộng sai số biên VAD (tới vài trăm ms). Đừng diễn giải p10..p90 dưới ~0.2 s.

### B4.6 Chỗ `analyze()` có vấn đề, và cách xử lý

1. **>2 speaker làm `other_of()` trả None.** `other_of` (`utils/conversation_events.py:199-201`) chỉ trả speaker khác khi `len(rest) == 1`. Gọi `analyze` trên nhiều hơn hai speaker thì event `backchannel` có `during = None` (`:216-217`), event `merged_backchannel` bị bỏ (`:281-283`), và các tổng `shared` (`:207-208`) và `holders` (`:231-232`) cộng chéo mọi speaker khác, nên một backchannel với người thứ ba bị coi là chồng giọng với người khác. Xử lý: chạy theo cặp và che người thứ ba (B4.1 bước 4), thay cho cách chia cửa sổ tham lam (cách đó làm mất backchannel của người thứ ba, cắt đôi lượt và nhân đôi vùng chồng). Cặp hai người cho `other_of` luôn xác định.
2. **`turn_switch` không phát khi có overlap** (`:288-300`): không dùng để tính độ trễ, xem B4.5.
3. **`marked`**: không có finder ở stage này, nên `intent=None`, `response` chỉ là `yield`/`continue`.
4. **Event dict chỉ có `segment` = key unit đầu**: xử lý bằng che theo thời gian (B4.1 bước 4).
5. **Backchannel chặt**: `unit.n_words ≤ 2` và `duration ≤ 1.0` và có người kia nói cả trước lẫn sau trong 1.0 s (`:143-155`). Backchannel "dạ vâng ạ" (3 từ) hoặc đứng cuối lượt người kia bị bỏ. Đây là lý do cảnh báo "cận dưới".
6. **Hiệu năng**: `_mark_backchannels` (`:143-155`) và vòng interruption (`:222-272`) là O(n²) trên số unit của một lần `analyze`. Region bị cắt ở mọi mối nối và theo cặp nên n nhỏ; không có giới hạn độ dài cửa sổ (C8).

---

## B5. Hằng số và từ điển (gắn với `PARAMS_VERSION`)

Mọi giá trị dưới đây là hằng trong `services/persona_profile_service.py`, KHÔNG nằm trong config. Đổi bất kỳ hằng nào, từ điển nào, hay `EventParams()` mặc định thì phải đổi `PARAMS_VERSION`; bước gom persona không được trộn hồ sơ khác `params_version`.

| Hằng | Giá trị | Ý nghĩa |
|---|---|---|
| `MIN_DIST_N` | 20 | Phân bố cần ≥ n quan sát mới có p10/p50/p90, ngược lại `null`; cũng là ngưỡng `n_sentences` cho `particles` |
| `MIN_RATIO_N` | 8 | Tỉ lệ cần mẫu số ≥ n, ngược lại `value=null` |
| `MIN_EXPOSURE_MIN` | 5.0 | Số phút cơ hội tối thiểu cho tần suất (backchannel) và cho `fillers`, `address_terms`, hỏi Qwen (so với `speech_min`) |
| `MIN_EXPOSURE_INTERRUPTION_MIN` | 20.0 | Ngưỡng exposure riêng cho `interruption.per_hour_of_other` (hiếm hơn) |
| `MIN_LEXICON_N` | 5 | Số backchannel tối thiểu để công bố `lexicon` |
| `RATE_MIN_WORDS` / `RATE_MIN_UNIT_S` | 3 / 1.0 | Unit phải có ≥ n từ và dài ≥ s giây để tính nhịp |
| `RATE_MIN_WPS` / `RATE_MAX_WPS` | 0.5 / 9.0 | Ngoài khoảng là lỗi timing, loại |
| `SENTENCE_MIN_WORDS` / `SENTENCE_MAX_WORDS` | 2 / 80 | Câu ngoài khoảng bị bỏ |
| `LATENCY_FLOOR_S` / `LATENCY_CAP_S` | -3.0 / 6.0 | Khoảng độ trễ hợp lệ |
| `EDGE_MARGIN_S` | 1.5 | Cuối region bị bỏ (≥ `yield_within_seconds`), trừ khỏi cả tử lẫn mẫu |
| `NOISE_FLOOR` | 0.10 | Ngưỡng `dominant_kind` của stage, không đọc `NOISE_NOTICEABLE` |
| `LLM_WINDOW_CHARS` | 6000 | Ký tự tối đa của cửa sổ chữ gửi Qwen cho một speaker |
| `MAX_CANDIDATE_FORMS` | 20 | Số form tối đa trong `llm_candidate_forms` |
| `DESCRIPTION_MAX_CHARS` | 400 | Mô tả dài hơn thì `null` |
| `FILLERS`, `PARTICLES`, `ADDRESS_TERMS` | B4.4 | Từ điển có thứ tự |

Config chỉ còn `models.persona_profile.{llm_enabled, thinking, max_new_tokens}` (A4); chúng không ảnh hưởng số liệu ngoài việc bật/tắt hai trường do Qwen sinh, và nằm trong namespace checkpoint.

---

## B6. `stats.json`

| Trường | Kiểu | Ý nghĩa |
|---|---|---|
| `excluded_s` | object | giây bị loại, mỗi segment đúng một khóa theo thứ tự B4.1: `review_flag`, `music`, `noise`, `unseparated_overlap`, `empty_text` |
| `unknown_gaps_skipped` | int | số segment (không kể đầu file) có `gap_before is None`, dùng để cắt region |
| `llm_windows` | int | số cửa sổ chữ gửi Qwen (mỗi speaker đủ exposure một cửa sổ) |
| `llm_ok` | int | số cửa sổ trả JSON đọc được |
| `warnings` | list[str] | xem dưới |

`warnings` gồm (khi áp dụng) các câu cố định:
* `"backchannel_and_interruption_are_lower_bounds: diarization drops overlapped speech, so backchannel and interruption counts under-count"` (luôn có khi có ít nhất một speaker được profile).
* `"not_reviewed: <N> segment(s) have review_flags=None and were treated as unflagged"` khi có segment `review_flags is None`.
* `"noise_unmeasured: <N> segment(s) have no noise measurement and were not excluded"` khi có segment `noise_breakdown is None`.
* `"llm_unavailable: ..."` khi `llm_enabled` và (`ensure_loaded()` False, hoặc `llm_windows > 0` và `llm_ok == 0`).
* `"low_usable_ratio: <giá trị>"` khi `usable_ratio < 0.5`.

Ví dụ (khớp `profile.json` ở B3.3: hai speaker đủ exposure được hỏi, `SPEAKER_02` không):

```json
{
  "excluded_s": {
    "review_flag": 31.4, "music": 12.0, "noise": 55.2,
    "unseparated_overlap": 18.9, "empty_text": 0.0
  },
  "unknown_gaps_skipped": 7,
  "llm_windows": 2,
  "llm_ok": 2,
  "warnings": [
    "backchannel_and_interruption_are_lower_bounds: diarization drops overlapped speech, so backchannel and interruption counts under-count"
  ]
}
```

---

## B7. Dataclass trong `schemas/` và serialize

Phong cách trong repo: `@dataclass` thuần, `Optional[...]`, chú thích tiếng Việt/Anh ngắn bên cạnh trường (`schemas/transcript.py`, `schemas/segment.py`), `List`/`Dict` từ `typing`. File mới: `schemas/persona_profile.py`.

```python
@dataclass
class Dist:                       # phân bố
    p10: Optional[float]
    p50: Optional[float]
    p90: Optional[float]
    n: int

@dataclass
class Rate:                       # tần suất
    value: Optional[float]
    count: int
    exposure_min: float

@dataclass
class Ratio:                      # tỉ lệ
    value: Optional[float]
    n: int

@dataclass
class Filler:       form: str; per_min: float
@dataclass
class Particle:     form: str; per_100_sent: float
@dataclass
class AddressTerm: form: str; count: int
@dataclass
class LexiconItem:  form: str; share: float

@dataclass
class Style:
    speech_rate_wps: Dist
    words_per_sentence: Dist
    fillers: Optional[List[Filler]]            # None = thiếu exposure; luôn đủ mọi form của từ điển, không bao giờ []
    particles: Optional[List[Particle]]
    address_terms: Optional[List[AddressTerm]]
    llm_candidate_forms: Optional[List[str]]   # None = Qwen không chạy / thiếu; [] = chạy, không form mới hợp lệ
    description: Optional[str]

@dataclass
class BackchannelProfile:
    per_min_of_other: Rate
    lexicon: Optional[List[LexiconItem]]
    overlap_ratio: Ratio

@dataclass
class InterruptionProfile:
    per_hour_of_other: Rate
    yield_rate: Ratio

@dataclass
class TurnProfile:
    duration_s: Dist
    response_latency_s: Dist                   # âm = vào trước khi người kia dứt

@dataclass
class SpeakerProfile:
    speech_s: float
    style: Style
    backchannel: BackchannelProfile
    interruption: InterruptionProfile
    turn: TurnProfile

@dataclass
class PersonaProfile:                          # = nội dung profile.json
    schema_version: int
    params_version: str
    source_type: str                           # "real" | "synthetic"
    recording_id: str
    usable_ratio: Optional[float]
    speakers: Dict[str, SpeakerProfile]

@dataclass
class PersonaStats:                            # = nội dung stats.json
    excluded_s: Dict[str, float]               # 5 khoá của B6
    unknown_gaps_skipped: int
    llm_windows: int
    llm_ok: int
    warnings: List[str]
```

Hằng số và từ điển (B5) là hằng module trong `services/persona_profile_service.py`, không có dataclass cấu hình.

Serialize:
* `to_dict()` là `dataclasses.asdict`: giữ nguyên `None`, `0.0`, `[]` (không biến đổi giá trị) và giữ thứ tự khai báo trường nên JSON ổn định. Cấm tự viết `value or 0`, `list or []`, `dict.get(k, 0)`. Không có `from_dict`: stage chỉ ghi, bước gom persona tự đọc JSON.
* `json.dumps(..., ensure_ascii=False, indent=2, allow_nan=False)` ra chuỗi trước rồi mới mở file ghi, qua `_write_json` (A5): giá trị NaN/inf không bao giờ được ghi và không để lại file viết dở (chia cho 0 đã được thay bằng `null` bởi quy tắc ngưỡng; `allow_nan=False` để lỗi lớn tiếng).

---

## B8. Kế hoạch test

Vị trí: `tests/`. Phong cách theo `tests/test_conversation_events.py`: segment giả dạng `TranscriptSegment` thật (stage dùng `dataclasses.replace`), helper tương tự `seg(i, speaker, start, text, step=0.3, flags=None)` (`:20-29`) nhưng không gán `words`, `EventParams()` mặc định, không GPU/LLM/âm thanh, chạy bằng `python -m pytest tests/test_persona_profile.py -q`. LLM thay bằng một lớp fake nhỏ (`FakeLLM.generate(prompt) -> str` trả chuỗi JSON cố định hoặc rác).

**`tests/test_persona_profile.py`** (hàm thuần và stage nhỏ)
1. `test_round_trip_keeps_null_zero_and_empty_distinct`: dựng `PersonaProfile` như ví dụ B3.3 → `asdict` → `json.dumps` → `loads`; assert `Rate(value=0.0,count=0,exposure_min=22.1)` giữ `0.0` (không `None`), `Dist(None,None,None,n=3)` giữ `None` và `n=3`, `fillers` có form `per_min == 0.0`, `llm_candidate_forms=[]` ≠ `None`, `description=None`.
2. `test_nan_is_never_written`: `Rate(value=float("nan"))` làm `write_persona_profile` raise.
3. `test_low_exposure_gives_null_value_but_zero_events_with_enough_exposure_gives_zero`: backchannel count=2, exposure 1.1 phút < 5 → `value is None`, `count==2`, `exposure_min==1.1`; exposure 31 phút, count 0 → `value == 0.0` và `is not None`.
4. `test_distribution_and_ratio_thresholds`: `n < MIN_DIST_N` → ba phân vị `None` nhưng `n` giữ; phân vị tuyến tính cho `[1,2,3,4,5]` khớp giá trị tay; tỉ lệ `n < MIN_RATIO_N` → `None`; `n` đủ với tử 0 → `0.0`.
5. `test_sentence_split_and_particles_at_sentence_end_on_token_boundary`: "Tôi đi chợ. Bạn có đi không? Giá 3.5 triệu nhé!" → 3 câu (không cắt "3.5"); mảnh cuối không có dấu kết bị bỏ; "ạ" trong "bạn" không tính, "nhé" cuối câu tính.
6. `test_dictionary_lists_are_complete_ordered_and_null_only_on_low_exposure`: speaker đủ exposure có đúng mọi form của `FILLERS`/`PARTICLES`/`ADDRESS_TERMS` theo thứ tự, form vắng là `0`; speaker `speech_min < 5` thì ba danh sách là `None`; chữ ở dạng NFD vẫn khớp sau NFC (C3); LLM tắt hoặc trả rác không đổi ba danh sách.
7. `test_llm_candidate_forms_validation`: Qwen đề xuất `["tức là","xyz-không-có","ạ","ừm"]` với chữ chỉ có "tức là", "bạn" và "ừm" → chỉ `["tức là"]` (không có trong chữ bị loại; "ạ" chỉ là chuỗi con của "bạn" nên bị loại; "ừm" đã có trong từ điển nên bị loại); trả `{"new_forms":[]}` → `[]`; trả rác sau retry → `None`; số đếm của các danh sách từ điển không đổi.
8. `test_description_null_rules_and_llm_disabled`: description dài hơn 400 hoặc rỗng → `None`; `llm_enabled=False` → `llm_candidate_forms` và `description` là `None`, mọi số liệu giống hệt bản có LLM.
9. `test_gap_before_none_and_crosses_cut_split_regions`: A→B có `gap_before=None` ở B → không quan sát độ trễ qua mối nối, `unknown_gaps_skipped == 1`, trong khi cặp tương tự `gap_before=0.4` cho quan sát 0.4; segment `crosses_cut` của A nằm giữa hai lời của B và C: không quan sát độ trễ nào đi qua nó; mối cắt nằm cạnh segment `empty_text` (A[0,5], cắt ở 6, E rỗng [7,7.5], B[8,9]) vẫn tách A khỏi B, tức không có quan sát 3.0 s; A[0,10] `crosses_cut`, B[5,6] (3 từ), A2[10.1,14], rồi B2, A3, B3, A4 (A3.start ≤ R.end − 1.5): không có quan sát 4.1 s (u=A2, p=B), B[5,6] không vào `other_talk_s(A)` (M ngoài R, bước 4(c)), còn quan sát B2→A3 (p=B2 bắt đầu sau 10.2) vẫn được giữ.
10. `test_latency_rules`: A nói đến 5.0, B bắt đầu 4.8 → quan sát −0.2 (turn_switch không có giá trị này); `u` nằm trọn trong lời `p` bị bỏ (`p.end > u.end`); X nói đến 5.0, Y chen 3 từ không phải backchannel [4.5, 5.6] (chồng giọng, `p.end ≤ X_prev.end + 1.5`) rồi X nói tiếp → không quan sát; X_prev kết thúc 5.2, Y [5.1, 6.5] (chồng 0.1 s < `overlap_min`, `p.end ≤ 6.7`) rồi X đáp 6.8 → quan sát 0.3 được giữ (trước đây bị bỏ oan); cùng cảnh với Y bắt đầu 4.9 (chồng 0.3 ≥ 0.2) → không quan sát; còn Y vào lúc 4.5 và nói đến 8.0 (giành lượt thật) rồi X đáp → quan sát được giữ; giá trị ngoài `[-3.0, 6.0]` bị bỏ.
11. `test_three_speakers_are_analyzed_per_pair_and_masked`: dãy A–B–A–C("ừ")–A: backchannel của C được ghi ở cặp (A,C) với `during` khác None; lời của C bị che khỏi cặp (A,B) và trừ khỏi exposure; một event có thể xuất hiện ở hai cặp chỉ được đếm một lần; A1 – B – A2 (khoảng lặng > `ipu_gap`) ở cặp (A,C) không thành một lượt (bỏ vì khoảng lượt giao M); C → B → A ở cặp (A,C) không cho quan sát độ trễ; `exposure_min` của backchannel và interruption bằng nhau.
12. `test_backchannel_in_turn_lexicon_and_overlap_ratio`: A nói dài, B chen "ừ" ×3 và "dạ" ×1, một "ừ" lúc im; chỉ các event `in_turn` được tính, `share` tổng 1 (ừ 0.75), `overlap_ratio.n == per_min_of_other.count`.
13. `test_interruption_rate_and_yield_rate`: `per_hour_of_other` dùng `other_talk_s` và không tính `competitive_overlap`; `yield_rate` có `competitive_overlap` với `of == X` trong mẫu số (2 lần cắt: một nhường, một `competitive_overlap` → `n=2`, tử 1); dùng patch `MIN_RATIO_N=1` để test công thức.
14. `test_exclusions`: mỗi lý do (`review_flag` gồm cờ `speaker_phrase`, `music`, `noise`, `unseparated_overlap`, `empty_text`) một segment; `excluded_s` cộng khớp, mỗi segment chỉ vào một khoá theo thứ tự ưu tiên; segment bị loại không vào phong cách và che event tương tác; `noise_breakdown` dưới `NOISE_FLOOR` là `clean` (không loại), `None` không bị loại và sinh cảnh báo `noise_unmeasured`; `monkeypatch.setattr(noise_map, "NOTICEABLE", ...)` sang giá trị khác (biến môi trường được đọc lúc import nên chỉ patch hằng module mới có tác dụng) không đổi kết quả loại nhiễu của stage.
15. `test_stage_output_deterministic_and_checkpoint_only_load`: chạy hai lần với cùng input + FakeLLM → hai file byte-giống nhau (không thời gian tạo, không id ngẫu nhiên), đủ khoá ở mọi speaker, `source_type == "real"`, `recording_id` lấy từ `audio_path`; `_profile_personas(..., allow_compute=False)` với checkpoint có sẵn không gọi LLM và với checkpoint thiếu không gọi LLM cũng không ghi `profile.json`.
16. `test_params_version_guard`: dấu vân tay (băm JSON) của toàn bộ hằng B5, ba từ điển và `asdict(EventParams())` bằng giá trị ghi sẵn trong test cùng `PARAMS_VERSION`; sai lệch báo "đổi hằng thì phải tăng PARAMS_VERSION và cập nhật dấu vân tay".

**Cập nhật test có sẵn**
17. `tests/test_review_select_align_order.py` (dòng 276-336): `_STOP_ORDER` có `transcript_review < persona_profile < conversation_selection` và `_reaches(stop_after="persona_profile", ...)`; với `persona_svc` bật `_next_llm_step_after(args, "transcript_review") == "persona_profile"` và `_release_llm_after(args, "persona_profile")` giữ/gỡ LLM đúng theo `_keep_llm_between_passes`; khối `run()` dừng đúng ở `stop_after == "persona_profile"` (kiểm tra theo mẫu `_source()` ở `:284-286`).
18. `tests/test_batch.py` (dòng 97-136): thứ tự `PIPELINE_STAGES` có `transcript_review < persona_profile < conversation_selection`; khi `step_persona_profile` tắt hoặc thiếu thì pass `persona_profile` bị bỏ (`stage_args` không chạy), khi bật thì chạy đủ mọi file trước pass kế tiếp.

Test bắt buộc theo yêu cầu: round-trip null/0/[] (#1), exposure thấp → null (#3), `gap_before None` bị loại (#9), >2 speaker (#11), form Qwen không có trong text bị loại (#7), pass sau không nạp lại Qwen (#15).

---

# C. Quyết định đã chốt

Các điểm dưới đây do hai lần viết spec và review vòng 1 phát hiện khi đối chiếu với code. Spec ở trên viết theo quyết định cuối.

| # | Vấn đề | Quyết định cuối |
|---|---|---|
| C1 | **Nghĩa của `yield_rate`.** Bản thiết kế trước ghi "khi người này ngắt lời, người bị ngắt dừng lại bao nhiêu" (phụ thuộc người kia). | Dùng nghĩa "khi người này **bị** cắt vào, họ nhường bao nhiêu" (`of == X`): tính cách của chính người nói ("dễ nhường lời"), dùng được khi mô phỏng persona. Mẫu số gồm cả `interruption` và `competitive_overlap` có `of == X` (competitive là ca "không nhường"); tử là `interruption` có `response == "yield"`. |
| C2 | **Timing chỉ ở mức segment.** Trước `word_alignment`, `words` là của ASR trước refinement và không có cách tin cậy để phân biệt với `words` đã align. | Bản sao làm việc LUÔN có `words=None`; timing luôn mức segment; không có `timing_precision` hay chế độ "mixed". Không chạy word alignment toàn file trước stage này (đắt, và phải gỡ Qwen). |
| C3 | **`_fold` không chuẩn hoá NFC** (`utils/conversation_selection.py:257-258`). Transcript dạng NFD làm "ừ" không khớp `BACKCHANNELS`. Lỗi này ảnh hưởng cả `conversation_selection` hiện có. | Stage NFC-normalize bản sao segment trước khi đo. Sửa `_fold` dùng chung là một thay đổi riêng (commit `fix(...)` riêng); khi sửa thì phải tăng `PARAMS_VERSION` vì số liệu của stage đổi. |
| C4 | **Tiêu chí loại nhiễu.** `noise_kind` (mức `NOTICEABLE` đọc từ biến môi trường `NOISE_NOTICEABLE`) hay ngưỡng riêng. | Loại theo nhóm nhiễu, nhưng tính lại bằng `dominant_kind(noise_breakdown, floor=NOISE_FLOOR)` với `NOISE_FLOOR = 0.10` là hằng của stage, để số liệu không đổi theo biến môi trường mà `PARAMS_VERSION` không đổi. `clean` và chưa đo (`None`) đều không bị loại; chưa đo sinh cảnh báo `noise_unmeasured`. |
| C5 | **Tên thư mục.** `STAGES` đánh số cố định, không liên tục (`07b_transcript_review` rồi `08_word_alignment`). | `07c_persona_profile`, không đánh số lại các thư mục sau. |
| C6 | **Qwen chỉ giữ nạp qua các pass khi bật `performance_config` và `keep_llm_across_relabel`** (`pipeline_service.py:509-515`). | Chấp nhận; mở rộng `_next_llm_step_after` (A5) để khi đã bật giữ LLM thì không gỡ giữa `transcript_review` và `persona_profile`. Các pass sau chỉ nạp checkpoint (`allow_compute`), không bao giờ nạp lại Qwen. |
| C7 | **Khối `kaggle` trong `config.json` không bật `transcript_review`.** | Stage chạy được khi không có review (không loại gì theo `review_flag`, chỉ cảnh báo `not_reviewed`). Không thêm `persona_profile` vào khối `kaggle` ở bản đầu. |
| C8 | **Hiệu năng `analyze()` O(n²).** | Không có giới hạn độ dài cửa sổ (`window_max_s`); region đã bị cắt ở mọi mối nối và `analyze` chạy theo cặp. Đo thời gian trên một file thật (vd `vimeanhphanchiatay-145413`, 28 phút) khi triển khai; chỉ thêm giới hạn nếu đo cho thấy cần, và khi đó tăng `PARAMS_VERSION`. |
| C9 | **Segment `unseparated` thường chính là nơi có backchannel/ngắt lời.** | Loại hẳn cả chữ lẫn timing của chúng (một cách duy nhất, không dùng nửa vời chữ-không-tin nhưng backchannel-vẫn-đếm). Chấp nhận số đếm thấp hơn nữa, đã nằm trong cảnh báo "cận dưới". |
| C10 | **Ngưỡng và từ điển.** | Là hằng trong code (B5), không cấu hình được; `EventParams()` mặc định cố định cho stage; `PARAMS_VERSION` viết tay, đổi khi đổi bất kỳ hằng/từ điển nào, có test dấu vân tay (B8 #16) chống quên. Từ điển bảo thủ, loại form đa nghĩa (B4.4); mở rộng offline từ `llm_candidate_forms`. Config chỉ còn `llm_enabled`, `thinking`, `max_new_tokens`, không fallback từ `models.relabel`. |
