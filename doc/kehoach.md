# Dialogue Chunk Assembly — Đặc tả thiết kế

> Từ transcript đã human-verify → dataset hội thoại 2-speaker cho full-duplex (dGSLM/Moshi style).

---

## 1. Bối cảnh

### 1.1. Điểm bắt đầu

Pipeline hiện tại đã chạy tới:

```
audio → music → diarization → separation (TSE) → ASR → refinement
                                                          ↓
                                                  {name}.json (transcript)
                                                          ↓
                                          make_review_page.py → review.html
                                                          ↓
                                            human check + sửa trong HTML
                                                          ↓
                                                  {name}_edited.json
```

`{name}_edited.json` là **input của toàn bộ tài liệu này**. Mỗi segment đã có:

| Field | Nguồn | Ý nghĩa |
|---|---|---|
| `text` / `text_edited` | ASR + human sửa | Text chuẩn |
| `speaker` / `spk_correct` | diarization + human sửa | Speaker chuẩn |
| `start`, `end` | diarization | Timestamp segment (timeline đã cut nhạc) |
| `words` | forced align | Word-level timestamps |
| `mark_music` | human | Có nhạc nền |
| `mark_multi` | human | Multi-speaker (>2 hoặc chồng lỗi) |
| `mark_spk` + `spk_correct` | human | Speaker gán sai → speaker đúng |
| `unseparated` | TSE (auto) | Vùng còn mixture 2 giọng, `[(start,end,reason)]` |
| `has_music`, `bss`, `noise_score` | auto | Trạng thái xử lý |
| `orig_spans`, `crosses_cut`, `gap_before` | provenance | Vị trí gốc, đứt đoạn |

### 1.2. Mục tiêu

Tạo **dataset hội thoại 2-speaker** để train full-duplex model:

- Mỗi sample = 1 đoạn hội thoại 1-5 phút, **đúng 2 speaker**, sạch (không nhạc/noise/multi-speaker).
- Format 2 stream độc lập: mỗi speaker 1 channel audio + 1 stream text (word-level timestamp).
- Giữ overlap tự nhiên (backchannel, interrupt) — full-duplex CẦN cái này.

### 1.3. Nguyên tắc thiết kế

1. **Tính toán từ metadata, không từ audio.** Thuật toán cắt chunk chạy trên list segment `(start, end, speaker, dirty)`. Việc ghép audio N-kênh chỉ để render HTML trace, không nằm trên critical path.
2. **Reject cứng, không fix.** Segment bẩn → loại, không cố sửa audio.
3. **Mỗi chunk 1 cặp speaker.** File 3 speaker (A,B,C) có thể ra chunk (A,B), (A,C), (B,C) riêng biệt.
4. **2 speaker tuyệt đối.** Trong 1 chunk, speaker thứ 3 chạm vào → cắt tại đó.

---

## 2. Config

```python
# Độ dài chunk
CHUNK_MIN_SECONDS      = 60      # bỏ chunk ngắn hơn
CHUNK_MAX_SECONDS      = 300     # 5 phút, split nếu dài hơn
CHUNK_TARGET_SECONDS   = 120     # nhắm ~2 phút khi chia đều

# Chất lượng dialogue
MIN_TURNS_PER_SPEAKER  = 2       # mỗi người ≥ 2 segment (lượt nói)
MAX_IMBALANCE          = 0.15    # 1 người không quá 85% tổng talk time

# Boundary
SILENCE_CUT_SECONDS    = 15      # silence dài hơn → đứt chunk
UNSEPARATED_MASK_MAX   = 3.0     # unseparated ngắn hơn → mask; dài hơn → cắt

# Cờ "bẩn" (bất kỳ cái nào true → segment không dùng được)
DIRTY_FLAGS = ["mark_music", "mark_multi", "has_music"]
DIRTY_NOISE_THRESHOLD  = 0.6     # noise_score cao hơn → bẩn

# SLM coherence (T3.5)
COHERENCE_CONFIDENCE_MIN = 0.8   # dưới ngưỡng → đẩy human review
COHERENCE_MODEL = "qwen3-omni"   # API localhost sẵn có
```

Tất cả đọc từ `config.json` block `dialogue_chunk`, override được qua env.

---

## 3. Thuật toán

### 3.1. Tổng quan 6 tầng

```
{name}_edited.json
      │
      ▼
[T0] Normalize        → segment sạch: speaker chuẩn, cờ dirty
      │
      ▼
[T1] Boundary (mỗi cặp) → vùng chỉ có X,Y + sạch
      │
      ▼
[T2] Slice            → cắt vùng thành chunk 60-300s
      │
      ▼
[T3] Balance filter   → loại monologue / lệch / ít lượt
      │
      ▼
[T3.5] SLM coherence  → đánh giá text: có phải hội thoại 2 người mạch lạc?
      │                  (accept / reject / manual_review)
      │
      ▼
[T4] Assembly         → 2 channel audio + 2 stream text
      │
      ▼
[T5] Export           → chunks/ + HTML trace (optional)
```

### 3.2. T0 — Normalize

Gộp input human + auto thành trạng thái sạch cho mỗi segment.

```python
def normalize(seg):
    return {
        "index":   seg["index"],
        "start":   seg["start"],
        "end":     seg["end"],
        "speaker": seg.get("spk_correct") or seg["speaker"],   # human sửa ưu tiên
        "text":    seg.get("text_edited") or seg["text"],
        "words":   seg.get("words") or [],
        "dirty":   is_dirty(seg),
        "unseparated": seg.get("unseparated") or [],
    }

def is_dirty(seg):
    if seg.get("mark_music") or seg.get("mark_multi") or seg.get("has_music"):
        return True
    if (seg.get("noise_score") or 0) > DIRTY_NOISE_THRESHOLD:
        return True
    # unseparated dài → bẩn; ngắn → xử lý mask ở T4, chưa loại ở đây
    for u in seg.get("unseparated") or []:
        if u["end"] - u["start"] > UNSEPARATED_MASK_MAX:
            return True
    return False
```

**Output**: `segments[]` đã chuẩn hóa.

### 3.3. T1 — Boundary (cho mỗi cặp speaker)

Với **mỗi cặp (X, Y)**, tìm các vùng liên tục chỉ chứa X, Y, silence ngắn và sạch.

```python
def valid_regions(segments, X, Y, total_duration):
    boundaries = []   # các (start, end) mà chunk KHÔNG được bắc qua

    # 1. Speaker thứ 3 → cắt
    for s in segments:
        if s["speaker"] not in (X, Y):
            boundaries.append((s["start"], s["end"]))

    # 2. Segment bẩn → cắt
    for s in segments:
        if s["dirty"]:
            boundaries.append((s["start"], s["end"]))

    # 3. Silence dài giữa 2 segment của X/Y → cắt
    xy = sorted([s for s in segments if s["speaker"] in (X, Y)],
                key=lambda s: s["start"])
    for a, b in zip(xy, xy[1:]):
        gap = b["start"] - a["end"]
        if gap > SILENCE_CUT_SECONDS:
            boundaries.append((a["end"], b["start"]))

    # Vùng hợp lệ = [0, total] trừ đi union(boundaries)
    return subtract([(0, total_duration)], merge_ranges(boundaries))
```

**Nhận xét quan trọng**:
- "2 người tuyệt đối" tự động đúng — mọi segment speaker thứ 3 đã thành boundary.
- Không cần dedup giữa các cặp: đoạn chỉ có A,B nói chỉ hợp lệ cho cặp (A,B); xét cặp (A,C) thì đoạn này C không active → khi filter balance (T3) sẽ loại vì `len(segs_C)=0`.

### 3.4. T2 — Slice

Cắt mỗi vùng hợp lệ thành chunk 60-300s, chia đều để không lệch.

```python
def slice_region(region, segments_in_region):
    L = region.end - region.start
    if L < CHUNK_MIN_SECONDS:
        return []                                  # quá ngắn
    if L <= CHUNK_MAX_SECONDS:
        return [region]                            # 1 chunk

    # Chia đều: nhắm chunk ~CHUNK_TARGET
    n = round(L / CHUNK_TARGET_SECONDS)
    step = L / n
    cuts = []
    for i in range(1, n):
        target = region.start + i * step
        # cắt tại silence gần target nhất, không cắt giữa từ
        cut = nearest_silence(target, segments_in_region,
                              window=step * 0.3)
        cuts.append(cut or target)                 # fallback: cắt cứng
    return split_by_cuts(region, cuts)
```

`nearest_silence`: tìm gap giữa 2 segment gần `target` nhất trong cửa sổ cho phép; nếu không có, cắt cứng tại target.

### 3.5. T3 — Balance filter

Mỗi chunk candidate: kiểm tra đủ lượt và cân bằng.

```python
def accept_chunk(chunk, X, Y):
    segs_X = [s for s in chunk.segments if s["speaker"] == X]
    segs_Y = [s for s in chunk.segments if s["speaker"] == Y]

    # Mỗi người ≥ MIN_TURNS lượt
    if len(segs_X) < MIN_TURNS_PER_SPEAKER: return False
    if len(segs_Y) < MIN_TURNS_PER_SPEAKER: return False

    # Không lệch quá
    time_X = sum(s["end"] - s["start"] for s in segs_X)
    time_Y = sum(s["end"] - s["start"] for s in segs_Y)
    if time_X + time_Y == 0: return False
    if max(time_X, time_Y) / (time_X + time_Y) > MAX_IMBALANCE:
        return False

    return True
```

Monologue tự loại (một trong 2 `len < MIN_TURNS`). Không cần check riêng.

### 3.5.5. T3.5 — SLM coherence check (đánh giá ngữ nghĩa)

**Vấn đề T3 không giải được**: filter cơ học (đủ lượt, cân bằng thời lượng) KHÔNG đảm bảo đó là hội thoại thật. Lọt lưới 4 case:

1. **Hai monologue nối nhau** — A độc thoại 60s, B độc thoại 60s, không ai đáp ai. Cơ học thấy 2 speaker cân bằng nhưng không phải đối thoại.
2. **Người thứ 3 bị gán nhầm** — diarization gán nhầm speaker C vào A/B; text đột ngột đổi chủ đề/giọng điệu.
3. **Turn-taking không mạch lạc** — có qua lại nhưng mỗi người nói chuyện riêng, không đáp lại nhau ("lệch vô lý").
4. **Overlap toàn bộ** — 1 người bị tách nhầm thành 2, "nói" chồng nhau gần hết.

4 case này đều là lỗi **ngữ nghĩa**, không đo bằng số được. Cần LLM đọc text.

**Model dùng**: `Qwen3-Omni` API localhost (đã chạy sẵn trong pipeline cho captioning — `models/qwen3_omni.py`, endpoint `chat/completions`). Gửi text thay vì audio, không tốn thêm model/VRAM.

**Input cho SLM** — transcript interleaved theo timestamp:

```
[A 0.0-2.3]  Hôm nay chúng ta nói về giá xăng
[B 2.1-2.5]  ừ
[A 2.5-5.0]  nó tăng liên tục mấy tháng nay
[B 5.2-8.1]  đúng, nhà tôi đổ đầy bình mất thêm mấy chục nghìn
[A 8.0-8.3]  chuẩn
...
```

**Prompt** (Vietnamese):

```
Đây là bản ghi hội thoại giữa 2 người (A và B) theo thứ tự thời gian.
Đánh giá xem đây có phải MỘT cuộc hội thoại tự nhiên giữa ĐÚNG 2 người không.

Trả về JSON:
{
  "is_dialogue": true/false,      // hội thoại thật hay 2 người độc thoại cạnh nhau
  "is_two_person": true/false,    // đúng 2 người, không có dấu hiệu người thứ 3 lọt vào
  "is_coherent": true/false,      // hai người đáp lại nhau, mạch lạc
  "is_continuous": true/false,    // liên tiếp, ngữ cảnh không đứt gãy vô lý
  "confidence": 0.0-1.0,
  "reason": "giải thích ngắn"
}

Chỉ trả JSON, không giải thích thêm.

Transcript:
{interleaved_text}
```

**Verdict** — kết hợp SLM + human review:

```python
def slm_verdict(chunk):
    text = interleave_by_timestamp(chunk.segments)
    resp = qwen3_omni.chat(COHERENCE_PROMPT + text)   # dùng API sẵn có
    j = parse_json(resp)

    all_pass = (j["is_dialogue"] and j["is_two_person"]
                and j["is_coherent"] and j["is_continuous"])

    if all_pass and j["confidence"] >= 0.8:
        return "accept"                    # SLM tự tin → auto accept
    if not all_pass and j["confidence"] >= 0.8:
        return "reject"                    # SLM tự tin loại → auto reject
    return "manual_review"                 # borderline → human duyệt
```

**3 verdict**:
- `accept` — SLM chắc chắn là hội thoại tốt → vào assembly luôn.
- `reject` — SLM chắc chắn không phải → loại.
- `manual_review` — SLM không chắc (confidence < 0.8, hoặc pass/fail lẫn lộn) → đẩy sang human duyệt.

Human chỉ xem chunk `manual_review` (thường 10-20% tổng chunk), không phải duyệt hết → tiết kiệm lớn.

**Lưu ý độ tin cậy**: Qwen3-Omni là model nhỏ, verdict không hoàn hảo. Vì vậy dùng ngưỡng confidence cao (0.8) để chunk mập mờ luôn rơi vào human review thay vì auto-quyết sai. Tuning ngưỡng này sau khi có vài trăm chunk có nhãn human để đo agreement.

**Output**: mỗi chunk có thêm field `coherence` (JSON verdict SLM) và `verdict` (accept/reject/manual_review).


### 3.6. T4 — Assembly (audio + text)

Với mỗi chunk đã accept, dựng output. **Tính thẳng từ segment**, không ghép N-kênh toàn file trước.

**Audio 2 channel** (cùng độ dài = độ dài chunk, giữ silence timing):

```python
def build_channels(chunk, X, Y, sr):
    dur = chunk.end - chunk.start
    chan_X = np.zeros(int(dur * sr), dtype=np.float32)
    chan_Y = np.zeros(int(dur * sr), dtype=np.float32)

    for s in chunk.segments:
        target = chan_X if s["speaker"] == X else chan_Y
        # audio đã tách của speaker này (từ TSE separation output)
        audio = load_separated_audio(s)            # đã có sẵn
        # mask unseparated ngắn thành silence ở CẢ 2 channel
        audio = mask_unseparated(audio, s, sr)
        dst = int((s["start"] - chunk.start) * sr)
        target[dst:dst + len(audio)] = audio

    return chan_X, chan_Y
```

**Unseparated ngắn (< 3s)**: mask silence ở **cả 2 channel** cùng vị trí (giữ timing đồng bộ). Vì nếu chỉ bỏ ở channel X mà giữ Y → 2 stream lệch nhau.

#### KẾT QUẢ CUỐI — cơ chế build 2 channel

Đây là output cốt lõi của toàn bộ pipeline. Với 1 chunk `[chunk.start, chunk.end]` và cặp (X, Y):

**Khởi tạo**: 2 mảng zero, dài đúng bằng chunk, cùng sample rate.

```
chan_X = zeros(duration)     # bắt đầu TOÀN im lặng
chan_Y = zeros(duration)     # bắt đầu TOÀN im lặng
```

**Ghi audio vào đúng vị trí thời gian gốc**:

- Segment của X → ghi audio X vào `chan_X` tại đúng offset `(seg.start - chunk.start)`. Vị trí đó trên `chan_Y` **để nguyên zero (im lặng)**.
- Segment của Y → ghi vào `chan_Y`, vị trí tương ứng trên `chan_X` **để nguyên zero**.
- Khoảng không ai nói (silence gốc) → cả 2 channel đều zero.
- Khoảng cả 2 cùng nói (overlap/backchannel) → cả 2 channel đều có audio tại cùng thời điểm.

**Minh họa** — chunk 10s, X nói 0-3s và 6-8s, Y nói 3.5-6s và chen "ừ" tại 2.5-3s:

```
timeline:   0    1    2    3    4    5    6    7    8    9   10
            │────────────│         │─────────│              │
chan_X:     [═══ X nói ══]·········[═ X nói ═]··············    (· = zero)
chan_Y:     ··············[═ Y ══]···········[══════════════   
                    ▲                              
              2.5-3s: overlap (cả 2 có audio, "ừ" của Y đè lên X)
```

**Bất biến quan trọng**:

1. `len(chan_X) == len(chan_Y) == duration * sr` — 2 channel LUÔN bằng nhau, khớp timeline chung.
2. Chỗ 1 người nói → người kia im lặng (zero) tại đúng vị trí đó.
3. Timing tuyệt đối theo timeline gốc trong chunk — không nén, không dồn. Đây là cái full-duplex học: **khi nào ai nói, khi nào im, khi nào chồng**.
4. Audio ghi vào là audio ĐÃ TÁCH của riêng speaker đó (từ TSE), không phải mixture. Nên `chan_X` chỉ chứa giọng X, kể cả ở đoạn gốc là overlap.

```python
def build_channels(chunk, X, Y, sr):
    dur_samples = int((chunk.end - chunk.start) * sr)
    chan_X = np.zeros(dur_samples, dtype=np.float32)   # im lặng mặc định
    chan_Y = np.zeros(dur_samples, dtype=np.float32)

    for s in chunk.segments:
        target = chan_X if s["speaker"] == X else chan_Y
        audio = load_separated_audio(s)                # audio đã tách của speaker
        audio = mask_unseparated(audio, s, sr)         # vùng mixture ngắn → 0
        dst = int((s["start"] - chunk.start) * sr)
        end = min(dst + len(audio), dur_samples)
        target[dst:end] = audio[:end - dst]
        # channel còn lại: KHÔNG động vào → giữ nguyên zero tại vị trí này

    return chan_X, chan_Y                              # cùng độ dài, đồng bộ
```

**Text 2 stream** (word timestamp relative đến chunk start):

```python
def build_transcript(chunk, speaker, X):
    words = []
    turns = []
    for s in chunk.segments:
        if s["speaker"] != speaker:
            continue
        for w in s["words"]:
            words.append({
                "w": w["w"],
                "start": w["start"] - chunk.start,   # relative
                "end":   w["end"]   - chunk.start,
            })
        turns.append({
            "start": s["start"] - chunk.start,
            "end":   s["end"]   - chunk.start,
            "text":  s["text"],
        })
    return {"speaker": speaker, "words": words, "turns": turns}
```

### 3.7. T5 — Export

```
chunks/{chunk_id}/
    channel_A.wav        # speaker X, silence khi Y nói
    channel_B.wav        # speaker Y
    transcript_A.json    # words + turns của X (relative chunk)
    transcript_B.json    # words + turns của Y
    mixed.wav            # X + Y (safe_limit), reference nghe
    metadata.json        # xem dưới
```

`metadata.json`:

```json
{
  "chunk_id": "lm8..._000123",
  "source_file": "lm8-vongtaynang-reaction-131131",
  "pair": {"A": "spk_00", "B": "spk_02"},
  "source_start": 245.34,
  "source_end": 487.12,
  "source_start_orig": 268.90,
  "source_end_orig": 510.68,
  "duration": 241.78,
  "stats": {
    "talk_time_A": 118.42,
    "talk_time_B": 95.30,
    "silence_time": 28.06,
    "overlap_time": 12.44,
    "turns_A": 14,
    "turns_B": 11,
    "words_A": 342,
    "words_B": 278,
    "masked_seconds": 1.8
  },
  "provenance": {
    "assembly_version": "1.0",
    "config": { ... }
  }
}
```

---

## 4. Ví dụ chạy tay

File 10 phút, 3 speaker A/B/C:

```
0-180s:   A, B thay phiên (12 seg mỗi người), sạch
180-185s: C chen vào
185-400s: A, B tiếp (sạch)
400-500s: A monologue
500-620s: A, C đối thoại (sạch)
```

**Cặp (A,B)**:
| Vùng | Dài | Slice | Balance | Kết quả |
|---|---|---|---|---|
| [0,180] | 180s | 1-2 chunk | OK | NHẬN |
| [185,400] | 215s | 2 chunk ~107s | OK | NHẬN |
| [400,500] | 100s | thiếu B | fail T3 | LOẠI |

**Cặp (A,C)**:
| Vùng | Dài | Kết quả |
|---|---|---|
| [500,620] | 120s | 1 chunk → NHẬN |

**Cặp (B,C)**: không vùng nào cả 2 cùng active → 0 chunk.

→ ~4-5 chunk, mỗi cái ghi rõ cặp.

---

## 5. Độ phức tạp

- Số cặp: C(n,2). File 3 speaker = 3 cặp, 5 speaker = 10 cặp.
- Mỗi cặp quét segment 1 lần: O(số segment).
- Tổng: O(C(n,2) × segment). Với n nhỏ (2-6 speaker điển hình) → rất nhẹ, chạy tức thì trên CPU.
- Chi phí thật nằm ở T4 (ghi audio WAV) — I/O bound, không phải compute.

---

## 6. Tích hợp vào Sommelier

### 6.1. File mới

```
services/dialogue_chunk_service.py   # orchestrator T0-T5
utils/chunk_boundary.py              # T1 boundary
utils/chunk_slicer.py                # T2 slice + nearest_silence
utils/chunk_stats.py                 # T3 balance + stats
schemas/dialogue_chunk.py            # dataclass Chunk, ChunkMeta
tools/make_chunk_trace_page.py       # T5 HTML trace (optional)
```

### 6.2. Sửa file có sẵn

```
services/export_service.py           # thêm export_dialogue_chunk()
services/separation_service.py       # generalize export_sdlm_dual_channel → N speaker (nếu cần)
```

### 6.3. Entry point

```python
# Chạy độc lập sau khi có {name}_edited.json
python tools/build_dialogue_chunks.py \
    --edited-json path/to/{name}_edited.json \
    --separated-audio path/to/separation/ \
    --out chunks/
```

Không cần chạy trong pipeline chính — là bước hậu kỳ, chạy sau human review.

---

## 7. Các quyết định đã chốt

| Câu hỏi | Quyết định |
|---|---|
| "Chỉ 2 người" | Tuyệt đối. Speaker thứ 3 chạm → cắt |
| File 3+ speaker | Mỗi chunk 1 cặp (A-B, A-C, B-C đều được) |
| Chunk length | 60-300s, nhắm 120s, chia đều |
| Min lượt/người | 2 segment |
| Balance | 1 người ≤ 85% talk time |
| Silence cut | > 15s → đứt chunk |
| Unseparated | < 3s mask silence 2 channel; ≥ 3s → segment bẩn → cắt |
| Text overlap | 2 stream riêng biệt, mỗi cái timestamp relative |
| Bad segment | Reject cứng, không fix |
| Đánh giá hội thoại | SLM (Qwen3-Omni) đọc text: is_dialogue/two_person/coherent/continuous |
| SLM verdict | accept/reject nếu confidence ≥ 0.8, else human review |
| SLM scope | Đánh giá cả chunk 1 lần (không cửa sổ trượt) |

---

## 8. Các điểm cần quyết khi implement

1. **`load_separated_audio(seg)`** — audio đã tách của mỗi speaker lấy từ đâu trong output TSE hiện tại? Cần verify format file trong `separation/` dir.
2. **HTML trace page** — có thật sự cần render N-kênh audio không, hay chỉ cần bảng metadata + link tới từng chunk? (audio embed rất nặng).
3. **Cặp trùng thời lượng** — nếu 1 file cho ra quá nhiều chunk từ nhiều cặp, có cần cap số chunk/file để tránh imbalance dataset không?
4. **Filter thêm cho full-duplex** — có cần `min_overlap_ratio` (chunk phải có ít nhất X% overlap để đảm bảo là dialogue thật, không phải 2 monologue nối nhau)?

---

## 9. Sơ đồ tổng thể

```
{name}_edited.json (human-verified)
      │
      ▼
[T0] normalize ──────────► segments sạch (speaker chuẩn + dirty flag)
      │
      ▼
for mỗi cặp (X,Y):
      │
      ├─[T1] boundary ────► vùng chỉ có X,Y + sạch
      │
      ├─[T2] slice ───────► chunk 60-300s, chia đều
      │
      ├─[T3] balance ─────► loại monologue/lệch/ít lượt
      │
      ├─[T3.5] SLM ───────► hội thoại 2 người mạch lạc?
      │                      accept/reject/manual_review (Qwen3-Omni)
      │
      ▼
[T4] assembly ───────────► build 2 channel:
      │                      chan_X = audio X, im lặng khi Y nói
      │                      chan_Y = audio Y, im lặng khi X nói
      │                      (2 channel bằng nhau, đồng bộ timeline)
      │                      + 2 stream text word-level
      │
      ▼
[T5] export ─────────────► chunks/{id}/ + metadata + HTML trace
      │
      ▼
dataset full-duplex (dGSLM/Moshi)
```