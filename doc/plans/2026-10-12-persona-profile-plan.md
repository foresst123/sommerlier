# Persona Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Hai agent chạy SONG SONG trên cùng working tree (Track 1 và Track 2), sau đó một reviewer; xem "Quy tắc làm việc chung".

**Goal:** Thêm stage opt-in `persona_profile` (sau `transcript_review`, trước `conversation_selection`) trích hồ sơ từng người nói của một file ghi âm, ghi `07c_persona_profile/profile.json` và `stats.json`, có checkpoint có phiên bản.

**Architecture:** Lõi thuần (`schemas/persona_profile.py` + `services/persona_profile_service.py`) nhận bản sao segment đã chuẩn bị, tính mọi con số bằng code từ từ điển cố định, chỉ gọi Qwen để viết `description` và đề xuất form mới (code xác nhận lại). Phần tích hợp (`pipeline_service`, `utils/batch.py`, `main.py`, `stage_output_service`, `config.json`, `performance_report`) chỉ nói chuyện với lõi qua một hợp đồng interface cố định (bên dưới), nên hai track làm độc lập.

**Tech Stack:** Python 3.12, `pytest`, dataclass thuần, `utils.conversation_events.analyze`, `utils.provenance.annotate`, `utils.llm_batches.ask_in_batches`. Không GPU, không numpy trong lõi.

**Spec:** `/Users/lam/Documents/sommelier/sommerlier/doc/specs/2026-10-12-persona-profile-design.md` (Phần B là nguồn chuẩn khi A và B khác nhau). Đọc toàn bộ trước khi làm.

Mọi đường dẫn dưới đây tính từ `podcast-pipeline/` trừ khi ghi rõ. Mọi lệnh chạy từ `podcast-pipeline/`:

```bash
cd /Users/lam/Documents/sommelier/sommerlier/podcast-pipeline
python -m pytest tests/test_persona_profile.py -q
```

## Global Constraints

Copy nguyên từ spec; mọi task ngầm tuân theo.

- Stage opt-in qua `opt_in_step_enabled(args, "persona_profile")`, mặc định tắt. Thư mục output `07c_persona_profile/` (không đánh số lại thư mục khác). Chỉ ghi hai file `profile.json` và `stats.json`.
- Stage chỉ đọc: không sửa `text`/`speaker`/thời gian của transcript thật, không đặt cờ lên segment thật. Mọi tính toán trên BẢN SAO `dataclasses.replace(seg, words=None)` (P0): `words=None`, text NFC, sắp theo `start` (ổn định), rồi `utils.provenance.annotate(copies, timeline, noise=noise_track)`.
- Mọi con số do code tính. Qwen chỉ viết `style.description` và đề xuất `style.llm_candidate_forms`; đề xuất đó KHÔNG vào bất kỳ số liệu nào.
- Thiếu dữ liệu không bao giờ ghi thành 0. `null` (None) = thiếu; `0.0` = có cơ hội mà không làm; `[]` (cho `llm_candidate_forms`) = LLM chạy, không form mới hợp lệ. Cấm `x or 0`, `x or []`, `dict.get(k, 0)` trong lõi. Số float làm tròn 3 chữ số SAU khi so ngưỡng.
- `profile.json` luôn đủ khoá ở mọi speaker; `source_type == "real"`; `schema_version == 1`; `params_version == PARAMS_VERSION == "pp-2026.10.2"`; `recording_id = splitext(basename(audio_path))[0]`; không `persona_id`, không thời điểm tạo, không id ngẫu nhiên (output giống từng byte với cùng input).
- Hằng/từ điển (B5) là hằng module trong `services/persona_profile_service.py`, KHÔNG nằm trong config. Đổi hằng/từ điển/`EventParams()` mặc định/công thức ảnh hưởng số thì phải đổi `PARAMS_VERSION`. `PROFILE_PROMPT_VERSION = "persona-v1"`.
- Config chỉ có `models.persona_profile.{llm_enabled, thinking, max_new_tokens}` (mặc định `True`, `False`, `1024`). Khoá lạ làm constructor raise `TypeError`. Không fallback sang `models.relabel`.
- `_write_json` mặc định `allow_nan=True` giữ hành vi cũ cho mọi stage khác; `write_persona_profile` ghi với `allow_nan=False`, serialize ra chuỗi TRƯỚC khi mở file.
- Python 4 space, snake_case, comment ngắn (AGENTS.md). Test dùng fake nhỏ, không GPU/LLM thật. Không mở rộng phạm vi ngoài spec.
- Commit: prefix conventional (`feat(persona): ...`, `test(persona): ...`), mỗi commit một hành vi, kết thúc bằng dòng:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`

## Review Focus

Năm lớp đầu vào/lỗi spec ngầm ngụ ý, dễ cắn người dùng nhất (mỗi dòng có test ghim ở task ghi trong ngoặc):

1. File chỉ có một speaker, hoặc `transcripts` rỗng/None: không lỗi, không ghi file, không checkpoint rỗng (T1.6 `test_single_speaker_has_null_interaction`, T1.8 `test_empty_input_is_skipped`, T2.4 `test_empty_or_none_transcripts_write_nothing`).
2. Transcript dạng NFD (dấu tổ hợp) vẫn khớp từ điển và `BACKCHANNELS` (T1.3 `test_prepare_normalizes_nfc_sorts_and_never_mutates_the_input`, T1.4 `test_nfd_text_matches_the_dictionary`).
3. LLM trả rác / thiếu / dài quá / trả form không có trong lời hoặc trùng từ điển: số liệu không đổi, `null` vs `[]` đúng, và không checkpoint khi chưa trả lời trọn (T1.7).
4. Mối cắt nhạc (`gap_before is None`, `crosses_cut`, cạnh segment rỗng) không được sinh độ trễ giả, và NaN/inf không bao giờ lọt vào file (T1.5, T2.1).
5. Pass sau trong batch (`allow_compute` sai) không nạp lại Qwen, không ghi đè `profile.json` đã có; pass đầu `--stop_after persona_profile` dừng đúng và ghi manifest (T2.4, T2.5).

## Quy tắc làm việc chung (hai track song song)

- Hai agent làm TRÊN CÙNG working tree. Mỗi agent CHỈ sửa file thuộc track của mình (bảng dưới), chỉ `git add <đường dẫn cụ thể>`, KHÔNG `git add -A`, KHÔNG `git add .`, KHÔNG `git stash`, KHÔNG `git checkout -- <file>` trên file của track kia.
- Nếu `git commit` thất bại vì index đang bị agent kia giữ khoá, chờ vài giây rồi thử lại.
- Nếu test kỳ vọng trong kế hoạch mâu thuẫn với chữ của spec (số học tay sai), SPEC thắng: sửa kỳ vọng trong test, ghi lý do ngắn trong commit message.
- Không sửa test không liên quan đang fail sẵn.

### Phân chia file (không giao nhau)

| Track 1: lõi thuần | Track 2: tích hợp |
|---|---|
| `schemas/persona_profile.py` (mới) | `services/pipeline_service.py` |
| `services/persona_profile_service.py` (mới) | `utils/batch.py` |
| `tests/test_persona_profile.py` (mới) | `main.py` |
| | `services/stage_output_service.py` |
| | `config.json` |
| | `utils/performance_report.py` |
| | `tests/test_stage_output.py` (sửa) |
| | `tests/test_batch.py` (sửa) |
| | `tests/test_review_select_align_order.py` (sửa) |
| | `tests/test_persona_profile_pipeline.py` (mới) |
| | `tests/test_stop_after_config.py` (sửa, T2.6) |

## Baseline test (ghi nhận lúc lập kế hoạch, commit `cbce204`)

`python -m pytest tests -q` cho kết quả `1617 passed, 62 failed, 1 skipped`. 62 test fail sẵn nằm ở các file sau (KHÔNG liên quan, KHÔNG sửa): `test_bss_assignment` (2), `test_config_driven` (2), `test_conversation_export_service` (25), `test_conversation_selection` (1), `test_cpu_optim` (1), `test_ghost_speakers` (4), `test_gpu_memory` (1), `test_mixture_window` (4), `test_music_full_strip` (1), `test_music_separator` (2), `test_overlap_preservation` (4), `test_prefix_cache` (1), `test_refinement_generate` (1), `test_seam_split` (1), `test_separation_logic` (6), `test_separation_recovery` (1), `test_separation_window` (5). Các file mà kế hoạch này sửa/đọc (`test_batch`, `test_review_select_align_order`, `test_stage_output`, `test_stop_after_config`, `test_steps`, `test_conversation_events`, `test_provenance`, `test_transcript_review`) đang PASS toàn bộ.

Trước khi bắt đầu mỗi track, chạy lại để lấy danh sách fail thật (hook `rtk` nén output, dùng `rtk proxy` để thấy tên test):

```bash
rtk proxy python -m pytest tests -q -p no:cacheprovider -rf 2>&1 | grep -E "^FAILED" | sed 's/ - .*//' | sort > "$TMPDIR/baseline_failed_$(date +%s).txt"; wc -l "$TMPDIR"/baseline_failed_*.txt
```

Cuối track, chạy lại và `diff` hai danh sách: chỉ được phép KHÔNG có dòng mới.

---

# HỢP ĐỒNG INTERFACE (cố định; Track 1 cung cấp, Track 2 dùng)

Track 2 không import module của Track 1 trong code sản phẩm của `pipeline_service`/`batch`/`stage_output` (chỉ `main.py` import trễ). Track 2 test bằng fake đúng hợp đồng này.

## Module và hằng

- `schemas/persona_profile.py`: dataclass `Dist, Rate, Ratio, Filler, Particle, AddressTerm, LexiconItem, Style, BackchannelProfile, InterruptionProfile, TurnProfile, SpeakerProfile, PersonaProfile, PersonaStats` đúng spec B7; `PersonaProfile.to_dict()` và `PersonaStats.to_dict()` = `dataclasses.asdict` (giữ `None`, `0.0`, `[]`).
- `services/persona_profile_service.py` xuất:
  - `PROFILE_PROMPT_VERSION: str = "persona-v1"`
  - `PARAMS_VERSION: str = "pp-2026.10.2"`
  - `class ProfileResult` (dataclass): `profile: dict` (nội dung `profile.json`), `stats: dict` (nội dung `stats.json`), `complete: bool` (True = an toàn để checkpoint), `skipped: Optional[str] = None` (`"no_segments"` khi đầu vào rỗng; khi đó `profile == {}`, `stats == {}`, `complete == False`).
  - `class PersonaProfileService`.

## `PersonaProfileService`

```python
PersonaProfileService(llm, logger=None, *, llm_enabled=True, thinking=False, max_new_tokens=1024)
```

Khoá lạ -> `TypeError` (keyword-only, không `**kw`). Thuộc tính công khai: `llm`, `logger`, `llm_enabled: bool`, `thinking: bool`, `max_new_tokens: int` (đã qua `reply_budget`).

| Thành viên | Chữ ký | Ý nghĩa |
|---|---|---|
| `prepare` | `prepare(self, segments, timeline, noise=None) -> list` | P0: trả BẢN SAO đã `words=None`, NFC, sắp theo `start`, đã `annotate`. Không sửa `segments` đầu vào. Không gọi LLM. |
| `source_digest` | `source_digest(self, prepared) -> str` | sha1 hex (40 ký tự) các trường P1/P2 đọc. Ổn định, không phụ thuộc id đối tượng. |
| `checkpoint_namespace_for` | `checkpoint_namespace_for(self, source_digest: str) -> str` | Chuỗi path-safe (`[A-Za-z0-9._-]+`) gồm `PROFILE_PROMPT_VERSION`, `think/nothink`, `llm/nollm`, tên model, `new{max_new_tokens}`, `PARAMS_VERSION`, `source_digest`. |
| `profile` | `profile(self, prepared, recording_id: str) -> ProfileResult` | Chỉ tính, không ghi file, không checkpoint. `prepared` PHẢI là kết quả `prepare`. Có thể raise; caller bắt. |

Cách Track 2 dùng (đúng thứ tự): `prepared = svc.prepare(transcripts, self.timeline, noise)` -> `digest = svc.source_digest(prepared)` -> `checkpoint.namespaces["persona_profile"] = svc.checkpoint_namespace_for(digest)` -> `load_if_committed("persona_profile")` -> nếu thiếu và được phép tính: `result = svc.profile(prepared, recording_id)` -> nếu `result.skipped` thì log và thôi; nếu `result.complete` thì `checkpoint.save("persona_profile", {"profile": result.profile, "stats": result.stats})` -> `stage_out.write_persona_profile(result.profile, result.stats)`.

`pipeline_service` đọc `getattr(svc, "llm_enabled", False)` để quyết định stage có dùng Qwen không (cho `_next_llm_step_after`).

## `StageOutputService` (Track 2 cung cấp, Track 1 không dùng)

- `STAGES["persona_profile"] = "07c_persona_profile"`.
- `_write_json(self, path, payload, *, allow_nan=True) -> bool`.
- `write_persona_profile(self, profile: dict, stats: dict) -> Optional[str]`: tôn trọng `enabled`; ghi đúng `profile.json` + `stats.json`; raise `RuntimeError` nếu một file ghi lỗi (NaN/inf).

## Fake dùng cho test Track 2

```python
class FakePersona:
    llm_enabled = True
    def __init__(self, result=None):
        self.result, self.calls = result, []
    def prepare(self, segments, timeline, noise=None):
        return list(segments)
    def source_digest(self, prepared):
        return "d1"
    def checkpoint_namespace_for(self, digest):
        return f"persona-ns-{digest}"
    def profile(self, prepared, recording_id):
        self.calls.append(recording_id)
        return self.result
```

`result` là `types.SimpleNamespace(profile=..., stats=..., complete=True, skipped=None)`.

---

# TRACK 1: Lõi thuần (8 task)

Files: tạo `schemas/persona_profile.py`, `services/persona_profile_service.py`, `tests/test_persona_profile.py`. Không đụng file nào khác.

Phong cách test: `TranscriptSegment` thật (stage dùng `dataclasses.replace`), không gán `words`, `EventParams()` mặc định, helper ở đầu file test (T1.1 tạo, các task sau dùng tiếp). Ngưỡng exposure thật (5/20 phút) quá lớn cho dữ liệu test: dùng `monkeypatch.setattr(svc_mod, "MIN_EXPOSURE_MIN", 0.0)` v.v. Hàm trong lõi PHẢI đọc hằng module tại thời điểm gọi (không dùng làm default argument) để patch có tác dụng.

### Task 1.1: Schema dataclass + round trip null/0/[]

**Files:**
- Create: `schemas/persona_profile.py`
- Create: `tests/test_persona_profile.py`

**Interfaces:**
- Produces: các dataclass spec B7; `PersonaProfile.to_dict()`, `PersonaStats.to_dict()`.

- [ ] **Step 0: Ghi baseline** bằng lệnh ở mục "Baseline test".

- [ ] **Step 1: Viết test fail** (`tests/test_persona_profile.py`, phần đầu file)

```python
"""Persona profile stage: schema, measures, events, LLM checks.

Run:  python -m pytest tests/test_persona_profile.py -q     (from podcast-pipeline/)
"""
import json
import os
import sys
import unicodedata

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import services.persona_profile_service as svc_mod
from schemas.persona_profile import (
    AddressTerm, BackchannelProfile, Dist, Filler, InterruptionProfile, LexiconItem,
    Particle, PersonaProfile, PersonaStats, Rate, Ratio, SpeakerProfile, Style,
    TurnProfile)
from schemas.transcript import TranscriptSegment

A, B, C = "SPEAKER_00", "SPEAKER_01", "SPEAKER_02"


def seg(i, speaker, start, end, text, **kw):
    """A real TranscriptSegment with no word times."""
    return TranscriptSegment(
        index=f"{i:05d}", start=start, end=end, speaker=speaker, text=text,
        text_whisper="", text_phowhisper="", text_qwen3="", language="vi",
        bs_roformer=False, bss=False, **kw)


def build(segs, gaps=None):
    """Fill gap_before the way provenance does (first None, then start - prev end).

    `gaps` maps a list position to an explicit value (None = a join lies there).
    """
    gaps = gaps or {}
    prev = None
    for pos, s in enumerate(segs):
        s.gap_before = gaps[pos] if pos in gaps else (
            None if prev is None else round(s.start - prev, 3))
        prev = s.end
    return segs


def words(n, word="từ"):
    return " ".join(f"{word}{k}" for k in range(n))


def _style(**over):
    base = dict(
        speech_rate_wps=Dist(None, None, None, 9), words_per_sentence=Dist(None, None, None, 4),
        fillers=None, particles=None, address_terms=None, llm_candidate_forms=None,
        description=None)
    base.update(over)
    return Style(**base)


def _speaker(style, rate):
    return SpeakerProfile(
        speech_s=41.7, style=style,
        backchannel=BackchannelProfile(per_min_of_other=rate, lexicon=None,
                                       overlap_ratio=Ratio(None, 1)),
        interruption=InterruptionProfile(per_hour_of_other=Rate(0.0, 0, 22.1),
                                         yield_rate=Ratio(None, 0)),
        turn=TurnProfile(duration_s=Dist(None, None, None, 3),
                         response_latency_s=Dist(-0.4, 0.3, 1.2, 88)))


def test_round_trip_keeps_null_zero_and_empty_distinct():
    style = _style(fillers=[Filler("ừm", 1.2), Filler("ờm", 0.0)],
                   particles=[Particle("nhé", 6.23)], address_terms=[AddressTerm("anh", 0)],
                   llm_candidate_forms=[], description=None)
    profile = PersonaProfile(
        schema_version=1, params_version="pp-2026.10.2", source_type="real",
        recording_id="ep", usable_ratio=None,
        speakers={A: _speaker(style, Rate(0.0, 0, 22.1)),
                  B: _speaker(_style(), Rate(None, 1, 1.1))})
    back = json.loads(json.dumps(profile.to_dict(), ensure_ascii=False, allow_nan=False))
    a, b = back["speakers"][A], back["speakers"][B]
    assert a["backchannel"]["per_min_of_other"] == {"value": 0.0, "count": 0, "exposure_min": 22.1}
    assert a["style"]["llm_candidate_forms"] == [] and a["style"]["description"] is None
    assert a["style"]["fillers"][1] == {"form": "ờm", "per_min": 0.0}
    assert a["turn"]["response_latency_s"]["p10"] == -0.4
    assert b["backchannel"]["per_min_of_other"]["value"] is None
    assert b["style"]["llm_candidate_forms"] is None and b["style"]["fillers"] is None
    assert b["turn"]["duration_s"] == {"p10": None, "p50": None, "p90": None, "n": 3}
    assert back["usable_ratio"] is None and back["source_type"] == "real"


def test_nan_is_rejected_by_strict_json():
    profile = PersonaProfile(1, "v", "real", "ep", None,
                             {A: _speaker(_style(), Rate(float("nan"), 1, 1.0))})
    with pytest.raises(ValueError):
        json.dumps(profile.to_dict(), allow_nan=False)


def test_stats_to_dict_keeps_all_keys():
    stats = PersonaStats(excluded_s={"review_flag": 0.0}, unknown_gaps_skipped=0,
                         llm_windows=0, llm_ok=0, warnings=[])
    assert stats.to_dict() == {"excluded_s": {"review_flag": 0.0}, "unknown_gaps_skipped": 0,
                               "llm_windows": 0, "llm_ok": 0, "warnings": []}
```

- [ ] **Step 2: Chạy, thấy FAIL**

Run: `python -m pytest tests/test_persona_profile.py -q`
Expected: FAIL `ModuleNotFoundError: schemas.persona_profile` (cũng sẽ fail ở import `services.persona_profile_service`: tạo file rỗng ở Step 3 để import không gãy).

- [ ] **Step 3: Viết code tối thiểu**

`schemas/persona_profile.py`:

```python
"""Persona profile of one recording: what profile.json and stats.json hold.

None means missing data, 0 / 0.0 means "had the chance and did not", [] on
llm_candidate_forms means "the LLM ran and found nothing". Never swap them.
"""
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


@dataclass
class Dist:                       # distribution; p* are None when n is too small
    p10: Optional[float]
    p50: Optional[float]
    p90: Optional[float]
    n: int


@dataclass
class Rate:                       # frequency; value None when exposure is too small
    value: Optional[float]
    count: int
    exposure_min: float


@dataclass
class Ratio:                      # proportion; value None when n is too small
    value: Optional[float]
    n: int


@dataclass
class Filler:
    form: str
    per_min: float


@dataclass
class Particle:
    form: str
    per_100_sent: float


@dataclass
class AddressTerm:
    form: str
    count: int


@dataclass
class LexiconItem:
    form: str
    share: float


@dataclass
class Style:
    speech_rate_wps: Dist
    words_per_sentence: Dist
    fillers: Optional[List[Filler]]            # None = low exposure; never []
    particles: Optional[List[Particle]]
    address_terms: Optional[List[AddressTerm]]
    llm_candidate_forms: Optional[List[str]]   # None = LLM did not run / unreadable; [] = ran, nothing new
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
    response_latency_s: Dist                   # negative = came in before the other finished


@dataclass
class SpeakerProfile:
    speech_s: float
    style: Style
    backchannel: BackchannelProfile
    interruption: InterruptionProfile
    turn: TurnProfile


@dataclass
class PersonaProfile:                          # = profile.json
    schema_version: int
    params_version: str
    source_type: str                           # "real" | "synthetic"
    recording_id: str
    usable_ratio: Optional[float]
    speakers: Dict[str, SpeakerProfile]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PersonaStats:                            # = stats.json
    excluded_s: Dict[str, float]
    unknown_gaps_skipped: int
    llm_windows: int
    llm_ok: int
    warnings: List[str]

    def to_dict(self) -> dict:
        return asdict(self)
```

Tạo `services/persona_profile_service.py` chỉ với docstring một dòng: `"""Persona profile stage (core). See doc/specs/2026-10-12-persona-profile-design.md."""`.

- [ ] **Step 4: Chạy, thấy PASS**

Run: `python -m pytest tests/test_persona_profile.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add schemas/persona_profile.py services/persona_profile_service.py tests/test_persona_profile.py
git commit -m "feat(persona): add profile and stats schema

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 1.2: Hằng, từ điển, PARAMS_VERSION, phân vị/ngưỡng

**Files:**
- Modify: `services/persona_profile_service.py`
- Test: `tests/test_persona_profile.py`

**Interfaces:**
- Produces (module-level, tên cố định): `SCHEMA_VERSION=1`, `PARAMS_VERSION`, `PROFILE_PROMPT_VERSION`, mọi hằng B5, `FILLERS`, `PARTICLES`, `ADDRESS_TERMS`, `EXCLUSION_KEYS`, `EVENT_PARAMS = EventParams()`, `params_snapshot() -> dict`, `percentile(values, q) -> float`, `make_dist(values) -> Dist`, `make_rate(count, exposure_min, min_exposure, per=1.0) -> Rate`, `make_ratio(numerator, n) -> Ratio`, `rnd(x) -> Optional[float]` (round 3, giữ None).

- [ ] **Step 1: Viết test fail** (nối vào `tests/test_persona_profile.py`)

```python
# --- constants and thresholds ----------------------------------------------------------

def test_params_version_guard():
    """Changing any constant, dictionary or EventParams default means bump PARAMS_VERSION.

    Update this snapshot AND PARAMS_VERSION together; a snapshot edit without a
    version bump is exactly the mistake this test exists to make visible.
    """
    assert svc_mod.PARAMS_VERSION == "pp-2026.10.2"
    assert svc_mod.params_snapshot() == {
        "MIN_DIST_N": 20, "MIN_RATIO_N": 8, "MIN_EXPOSURE_MIN": 5.0,
        "MIN_EXPOSURE_INTERRUPTION_MIN": 20.0, "MIN_LEXICON_N": 5,
        "RATE_MIN_WORDS": 3, "RATE_MIN_UNIT_S": 1.0, "RATE_MIN_WPS": 0.5, "RATE_MAX_WPS": 9.0,
        "SENTENCE_MIN_WORDS": 2, "SENTENCE_MAX_WORDS": 80,
        "LATENCY_FLOOR_S": -3.0, "LATENCY_CAP_S": 6.0, "EDGE_MARGIN_S": 1.5,
        "NOISE_FLOOR": 0.10, "LLM_WINDOW_CHARS": 6000, "MAX_CANDIDATE_FORMS": 20,
        "DESCRIPTION_MAX_CHARS": 400,
        "FILLERS": ["ừm", "ờm", "ờ", "ưm", "hừm", "đại loại"],
        "PARTICLES": ["nhé", "nhỉ", "nha", "nhá", "nè", "nhể", "hả", "hử", "ạ", "chứ"],
        "ADDRESS_TERMS": ["anh", "chị", "em", "mình", "bạn", "tớ", "cậu", "tôi"],
        "EVENT_PARAMS": {
            "ipu_gap": 0.2, "overlap_min": 0.2, "backchannel_max_seconds": 1.0,
            "backchannel_max_words": 2, "backchannel_context": 1.0,
            "yield_within_seconds": 1.5, "min_backchannels": 2,
            "half_duplex_max_overlap": 0.2, "role_min_conf": 0.7},
    }
    assert svc_mod.PROFILE_PROMPT_VERSION == "persona-v1"


def test_dictionaries_are_nfc():
    for form in svc_mod.FILLERS + svc_mod.PARTICLES + svc_mod.ADDRESS_TERMS:
        assert unicodedata.normalize("NFC", form) == form


def test_percentile_is_linear_and_distribution_needs_enough_samples(monkeypatch):
    assert svc_mod.percentile([5, 1, 3, 2, 4], 0.1) == pytest.approx(1.4)
    assert svc_mod.percentile([5, 1, 3, 2, 4], 0.5) == 3
    assert svc_mod.percentile([5, 1, 3, 2, 4], 0.9) == pytest.approx(4.6)
    monkeypatch.setattr(svc_mod, "MIN_DIST_N", 5)
    d = svc_mod.make_dist([1, 2, 3, 4, 5])
    assert (d.p10, d.p50, d.p90, d.n) == (1.4, 3, 4.6, 5)
    few = svc_mod.make_dist([1, 2, 3, 4])
    assert (few.p10, few.p50, few.p90, few.n) == (None, None, None, 4)
    empty = svc_mod.make_dist([])
    assert (empty.p50, empty.n) == (None, 0)


def test_low_exposure_gives_null_value_but_enough_exposure_with_no_events_gives_zero():
    low = svc_mod.make_rate(2, 1.1, 5.0)
    assert (low.value, low.count, low.exposure_min) == (None, 2, 1.1)
    zero = svc_mod.make_rate(0, 31.0, 5.0)
    assert zero.value == 0.0 and zero.value is not None
    per_hour = svc_mod.make_rate(2, 20.0, 20.0, per=60.0)
    assert per_hour.value == pytest.approx(6.0)
    assert svc_mod.make_rate(1, 0.0, 0.0, per=60.0).value is None      # never divide by zero


def test_ratio_thresholds(monkeypatch):
    assert svc_mod.make_ratio(3, 7).value is None and svc_mod.make_ratio(3, 7).n == 7
    assert svc_mod.make_ratio(0, 8).value == 0.0
    assert svc_mod.make_ratio(6, 8).value == 0.75
    monkeypatch.setattr(svc_mod, "MIN_RATIO_N", 1)
    assert svc_mod.make_ratio(1, 2).value == 0.5
```

- [ ] **Step 2: Chạy, thấy FAIL**

Run: `python -m pytest tests/test_persona_profile.py -q`
Expected: FAIL `AttributeError: module ... has no attribute 'PARAMS_VERSION'`.

- [ ] **Step 3: Code tối thiểu** (thay nội dung `services/persona_profile_service.py`)

```python
"""Persona profile stage (core): per-speaker habits of one recording.

Every number comes from code over fixed, versioned dictionaries. The LLM only
writes `style.description` and proposes extra forms; neither enters a number.
See doc/specs/2026-10-12-persona-profile-design.md.
"""
import math
from dataclasses import asdict
from typing import Iterable, List, Optional

from schemas.persona_profile import Dist, Rate, Ratio
from utils.conversation_events import EventParams

SCHEMA_VERSION = 1
# Hand-written. Change it whenever a constant, dictionary, EventParams default or
# formula below changes a number (tests/test_persona_profile.py guards this).
PARAMS_VERSION = "pp-2026.10.2"
PROFILE_PROMPT_VERSION = "persona-v1"

MIN_DIST_N = 20                     # distribution needs this many samples
MIN_RATIO_N = 8                     # ratio needs this denominator
MIN_EXPOSURE_MIN = 5.0              # minutes of opportunity for a frequency / style lists
MIN_EXPOSURE_INTERRUPTION_MIN = 20.0
MIN_LEXICON_N = 5
RATE_MIN_WORDS = 3
RATE_MIN_UNIT_S = 1.0
RATE_MIN_WPS = 0.5
RATE_MAX_WPS = 9.0
SENTENCE_MIN_WORDS = 2
SENTENCE_MAX_WORDS = 80
LATENCY_FLOOR_S = -3.0
LATENCY_CAP_S = 6.0
EDGE_MARGIN_S = 1.5                 # >= yield_within_seconds
NOISE_FLOOR = 0.10                  # the stage's own cut; never read NOISE_NOTICEABLE
LLM_WINDOW_CHARS = 6000
MAX_CANDIDATE_FORMS = 20
DESCRIPTION_MAX_CHARS = 400

# Conservative: only forms that almost always play that role (spec B4.4).
FILLERS = ("ừm", "ờm", "ờ", "ưm", "hừm", "đại loại")
PARTICLES = ("nhé", "nhỉ", "nha", "nhá", "nè", "nhể", "hả", "hử", "ạ", "chứ")
ADDRESS_TERMS = ("anh", "chị", "em", "mình", "bạn", "tớ", "cậu", "tôi")

EXCLUSION_KEYS = ("review_flag", "music", "noise", "unseparated_overlap", "empty_text")
EVENT_PARAMS = EventParams()


def params_snapshot() -> dict:
    """Every constant that can change a number, for the PARAMS_VERSION guard."""
    return {
        "MIN_DIST_N": MIN_DIST_N, "MIN_RATIO_N": MIN_RATIO_N,
        "MIN_EXPOSURE_MIN": MIN_EXPOSURE_MIN,
        "MIN_EXPOSURE_INTERRUPTION_MIN": MIN_EXPOSURE_INTERRUPTION_MIN,
        "MIN_LEXICON_N": MIN_LEXICON_N, "RATE_MIN_WORDS": RATE_MIN_WORDS,
        "RATE_MIN_UNIT_S": RATE_MIN_UNIT_S, "RATE_MIN_WPS": RATE_MIN_WPS,
        "RATE_MAX_WPS": RATE_MAX_WPS, "SENTENCE_MIN_WORDS": SENTENCE_MIN_WORDS,
        "SENTENCE_MAX_WORDS": SENTENCE_MAX_WORDS, "LATENCY_FLOOR_S": LATENCY_FLOOR_S,
        "LATENCY_CAP_S": LATENCY_CAP_S, "EDGE_MARGIN_S": EDGE_MARGIN_S,
        "NOISE_FLOOR": NOISE_FLOOR, "LLM_WINDOW_CHARS": LLM_WINDOW_CHARS,
        "MAX_CANDIDATE_FORMS": MAX_CANDIDATE_FORMS,
        "DESCRIPTION_MAX_CHARS": DESCRIPTION_MAX_CHARS,
        "FILLERS": list(FILLERS), "PARTICLES": list(PARTICLES),
        "ADDRESS_TERMS": list(ADDRESS_TERMS), "EVENT_PARAMS": asdict(EVENT_PARAMS),
    }


def rnd(x: Optional[float]) -> Optional[float]:
    """Round to 3 places AFTER any threshold was applied; None stays None."""
    return None if x is None else round(float(x), 3)


def percentile(values: Iterable[float], q: float) -> float:
    """NumPy-style linear interpolation, in pure Python."""
    xs = sorted(values)
    k = q * (len(xs) - 1)
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (k - lo) * (xs[hi] - xs[lo])


def make_dist(values: List[float]) -> Dist:
    n = len(values)
    if n < MIN_DIST_N:
        return Dist(None, None, None, n)
    return Dist(rnd(percentile(values, 0.1)), rnd(percentile(values, 0.5)),
                rnd(percentile(values, 0.9)), n)


def make_rate(count: int, exposure_min: float, min_exposure: float, per: float = 1.0) -> Rate:
    """count per `exposure_min` minutes, times `per`; None under `min_exposure`."""
    if exposure_min < min_exposure or exposure_min <= 0:
        return Rate(None, count, rnd(exposure_min))
    return Rate(rnd(count / exposure_min * per), count, rnd(exposure_min))


def make_ratio(numerator: int, n: int) -> Ratio:
    if n < MIN_RATIO_N:
        return Ratio(None, n)
    return Ratio(rnd(numerator / n), n)
```

- [ ] **Step 4: Chạy, thấy PASS**

Run: `python -m pytest tests/test_persona_profile.py -q`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add services/persona_profile_service.py tests/test_persona_profile.py
git commit -m "feat(persona): add thresholds, dictionaries and null-aware helpers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 1.3: P0 chuẩn bị, lý do loại, source_digest

**Files:**
- Modify: `services/persona_profile_service.py`
- Test: `tests/test_persona_profile.py`

**Interfaces:**
- Produces: `prepare_segments(segments, timeline, noise=None) -> list`, `source_digest_of(prepared) -> str`, `exclusion_reason(seg) -> Optional[str]` (một trong `EXCLUSION_KEYS` trừ khi None; thứ tự ưu tiên spec B4.1), `is_empty(seg) -> bool` (`_fold(text) == ""`), `excluded_seconds(prepared) -> dict` (đủ 5 khoá, giây làm tròn 3), `usable(seg) -> bool` (`exclusion_reason is None`).
- Quyết định: `empty_text` ưu tiên thấp nhất cho `excluded_s`, nhưng MỌI segment có `is_empty` đều bị bỏ khỏi các bước sau bước chia region bất kể lý do.

- [ ] **Step 1: Viết test fail**

```python
# --- P0 and exclusions -----------------------------------------------------------------

def test_prepare_normalizes_nfc_sorts_and_never_mutates_the_input():
    nfd = unicodedata.normalize("NFD", "ừm đúng rồi")
    late, early = seg(1, A, 5.0, 6.0, nfd), seg(0, B, 0.0, 1.0, "xin chào")
    late.words = [{"word": "ừm", "start": 5.0, "end": 5.5}]
    original = [late, early]
    prepared = svc_mod.prepare_segments(original, timeline=None)
    assert [s.index for s in prepared] == ["00000", "00001"]
    assert all(s.words is None for s in prepared)
    assert prepared[1].text == unicodedata.normalize("NFC", nfd)
    assert prepared[0].gap_before is None and prepared[1].gap_before == 4.0
    assert original == [late, early] and late.words is not None
    assert late.text == nfd and late.gap_before is None      # real segments untouched


def test_exclusions_one_reason_per_segment_in_priority_order(monkeypatch):
    flag = lambda kind: [{"kind": kind, "phrase": None, "conf": 0.9, "why": "x"}]
    rows = [
        seg(0, A, 0, 1, "a b c", review_flags=flag("speaker_turn")),
        seg(1, A, 1, 3, "a b c", review_flags=flag("speaker_phrase")),
        seg(2, A, 3, 6, "a b c", review_flags=flag("not_vietnamese")),
        seg(3, A, 6, 10, "a b c", has_music=True, review_flags=flag("speaker_turn")),
        seg(4, A, 10, 15, "a b c", noise_breakdown={"noise_speech": 0.2, "noise_env": 0.0,
                                                     "noise_room": 0.0}),
        seg(5, A, 15, 21, "a b c", unseparated=[{"start": 15, "end": 16, "reason": "x"}]),
        seg(6, A, 21, 28, "   "),
        seg(7, A, 28, 36, "a b c", noise_breakdown={"noise_speech": 0.05, "noise_env": 0.0,
                                                     "noise_room": 0.0}),   # clean
        seg(8, A, 36, 45, "a b c"),                                          # noise unmeasured
    ]
    reasons = [svc_mod.exclusion_reason(s) for s in rows]
    assert reasons == ["review_flag", "review_flag", "review_flag", "review_flag", "noise",
                       "unseparated_overlap", "empty_text", None, None]
    assert svc_mod.is_empty(rows[6]) and not svc_mod.is_empty(rows[0])
    assert svc_mod.excluded_seconds(rows) == {
        "review_flag": 10.0, "music": 0.0, "noise": 5.0,
        "unseparated_overlap": 6.0, "empty_text": 7.0}


def test_noise_cut_ignores_the_environment_variable_constant(monkeypatch):
    from utils import noise_map
    noisy = seg(0, A, 0, 1, "a b c", noise_breakdown={"noise_speech": 0.05, "noise_env": 0.0,
                                                       "noise_room": 0.0})
    before = svc_mod.exclusion_reason(noisy)
    monkeypatch.setattr(noise_map, "NOTICEABLE", 0.01)
    assert svc_mod.exclusion_reason(noisy) == before == None


def test_source_digest_changes_with_every_field_it_reads_and_is_stable():
    base = lambda **kw: build([seg(0, A, 0, 1, "xin chào", **kw), seg(1, B, 1.2, 2, "chào")])
    digest = lambda segs: svc_mod.source_digest_of(segs)
    assert digest(base()) == digest(base()) and len(digest(base())) == 40
    assert digest(base()) != digest(base(has_music=True))
    assert digest(base()) != digest(base(review_flags=[]))      # None vs [] differ
    assert digest(base()) != digest(base(crosses_cut=True))
    other = base()
    other[0].text = "xin chao"
    assert digest(base()) != digest(other)
```

- [ ] **Step 2: Chạy, thấy FAIL** — `python -m pytest tests/test_persona_profile.py -q -k "prepare or exclusions or noise_cut or digest"` -> `AttributeError`.

- [ ] **Step 3: Code** (thêm vào `services/persona_profile_service.py`; thêm import ở đầu)

```python
# imports to add at the top
import hashlib
import json
import unicodedata
from dataclasses import replace
from typing import Dict

from utils.conversation_selection import (
    _fold, _unseparated_seconds, _union_seconds, foreign_seconds, hard_review_kind,
    soft_review_count)
from utils.noise_map import dominant_kind
from utils.provenance import annotate

_NOISY_KINDS = frozenset({"noise_speech", "noise_env", "noise_room"})


def nfc(text) -> str:
    return unicodedata.normalize("NFC", text or "")


def prepare_segments(segments, timeline, noise=None) -> list:
    """P0: working copies (no words, NFC text, sorted by start, provenance filled)."""
    ordered = sorted(segments, key=lambda s: float(s.start))      # stable
    copies = [replace(s, words=None, text=nfc(s.text)) for s in ordered]
    annotate(copies, timeline, noise=noise)
    return copies


def is_empty(seg) -> bool:
    return _fold(getattr(seg, "text", "") or "") == ""


def exclusion_reason(seg) -> Optional[str]:
    """The one reason `seg` adds no numbers, in the spec's priority order; None = usable."""
    if (hard_review_kind(seg) is not None or foreign_seconds(seg) > 0
            or soft_review_count(seg) > 0):
        return "review_flag"
    if getattr(seg, "has_music", False):
        return "music"
    if dominant_kind(getattr(seg, "noise_breakdown", None), floor=NOISE_FLOOR) in _NOISY_KINDS:
        return "noise"
    if _unseparated_seconds(seg) > 0:
        return "unseparated_overlap"
    if is_empty(seg):
        return "empty_text"
    return None


def usable(seg) -> bool:
    return exclusion_reason(seg) is None


def excluded_seconds(prepared) -> Dict[str, float]:
    out = {key: 0.0 for key in EXCLUSION_KEYS}
    for s in prepared:
        reason = exclusion_reason(s)
        if reason is not None:
            out[reason] += float(s.end) - float(s.start)
    return {key: round(v, 3) for key, v in out.items()}


def source_digest_of(prepared) -> str:
    """sha1 of every field P1/P2 read, per segment."""
    rows = [{
        "index": str(s.index), "start": s.start, "end": s.end, "speaker": str(s.speaker),
        "text": s.text, "has_music": bool(getattr(s, "has_music", False)),
        "unseparated": getattr(s, "unseparated", None),
        "noise_breakdown": getattr(s, "noise_breakdown", None),
        "crosses_cut": bool(getattr(s, "crosses_cut", False)),
        "gap_before": getattr(s, "gap_before", None),
        "review_flags": getattr(s, "review_flags", None),
    } for s in prepared]
    blob = json.dumps(rows, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Chạy, thấy PASS** — `python -m pytest tests/test_persona_profile.py -q`.

- [ ] **Step 5: Commit**

```bash
git add services/persona_profile_service.py tests/test_persona_profile.py
git commit -m "feat(persona): prepare working copies, exclusion reasons and source digest

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 1.4: Phong cách (nhịp, độ dài câu, từ điển)

**Files:**
- Modify: `services/persona_profile_service.py`
- Test: `tests/test_persona_profile.py`

**Interfaces:**
- Produces: `split_sentences(text) -> List[str]` (chỉ câu có dấu kết, bỏ mảnh cuối), `sentence_word_count(sentence) -> int`, `count_form(tokens, form_tokens) -> int`, `class StyleMeasure` (dataclass: `style: Style`, `speech_min: float`, `n_sentences: int`, `texts: List[str]` = text các unit sạch không backchannel của speaker theo `start`), `measure_style(prepared) -> Dict[str, StyleMeasure]` (khoá = mọi speaker trong `prepared`).
- Consumes: T1.2 (`make_dist`, hằng, từ điển), T1.3 (`usable`, `is_empty`, `nfc`).

- [ ] **Step 1: Viết test fail**

```python
# --- style ----------------------------------------------------------------------------

def test_sentence_split_keeps_decimals_and_drops_the_unfinished_tail():
    out = svc_mod.split_sentences("Tôi đi chợ. Bạn có đi không? Giá 3.5 triệu nhé! và còn nữa")
    assert out == ["Tôi đi chợ.", " Bạn có đi không?", " Giá 3.5 triệu nhé!"]
    assert svc_mod.sentence_word_count(" Giá 3.5 triệu nhé!") == 4
    assert svc_mod.split_sentences("TP.HCM rất đông") == []


def _talk(rows):
    """Alternating long turns so nothing is a backchannel; returns a prepared list."""
    return build([seg(i, sp, st, en, tx) for i, (sp, st, en, tx) in enumerate(rows)])


def test_style_counts_every_dictionary_form_and_particles_only_at_sentence_end(monkeypatch):
    monkeypatch.setattr(svc_mod, "MIN_EXPOSURE_MIN", 0.0)
    monkeypatch.setattr(svc_mod, "MIN_DIST_N", 1)
    prepared = _talk([
        (A, 0.0, 6.0, "Tôi đi chợ. Bạn có đi không? Giá 3.5 triệu nhé!"),
        (B, 7.0, 12.0, "ừm ờm em thấy ổn lắm ạ."),
    ])
    measures = svc_mod.measure_style(prepared)
    a = measures[A].style
    assert [f.form for f in a.fillers] == list(svc_mod.FILLERS)
    assert [p.form for p in a.particles] == list(svc_mod.PARTICLES)
    assert [t.form for t in a.address_terms] == list(svc_mod.ADDRESS_TERMS)
    assert {p.form: p.per_100_sent for p in a.particles}["nhé"] == pytest.approx(33.333, abs=1e-3)
    assert {p.form: p.per_100_sent for p in a.particles}["ạ"] == 0.0     # "bạn" is not "ạ"
    assert {t.form: t.count for t in a.address_terms}["bạn"] == 1
    assert measures[A].n_sentences == 3 and a.words_per_sentence.n == 3
    b = measures[B].style
    assert {f.form: f.per_min for f in b.fillers}["ừm"] > 0
    assert {t.form: t.count for t in b.address_terms}["em"] == 1
    assert {p.form: p.per_100_sent for p in b.particles}["ạ"] == 100.0
    assert a.llm_candidate_forms is None and a.description is None      # filled later


def test_low_exposure_nulls_dictionary_lists_but_keeps_counts_of_samples():
    prepared = _talk([(A, 0.0, 4.0, "ừm tôi đi chợ nhé."), (B, 5.0, 9.0, "dạ")])
    m = svc_mod.measure_style(prepared)[A]
    assert m.speech_min < svc_mod.MIN_EXPOSURE_MIN
    assert m.style.fillers is None and m.style.address_terms is None
    assert m.style.particles is None                          # n_sentences < MIN_DIST_N
    assert m.style.words_per_sentence.n == 1 and m.style.words_per_sentence.p50 is None


def test_nfd_text_matches_the_dictionary(monkeypatch):
    monkeypatch.setattr(svc_mod, "MIN_EXPOSURE_MIN", 0.0)
    nfd = unicodedata.normalize("NFD", "ừm tôi nghĩ vậy anh nhé.")
    segs = svc_mod.prepare_segments([seg(0, A, 0.0, 5.0, nfd), seg(1, B, 6.0, 9.0, "ok bạn")], None)
    a = svc_mod.measure_style(segs)[A].style
    assert {f.form: f.per_min for f in a.fillers}["ừm"] > 0
    assert {t.form: t.count for t in a.address_terms}["anh"] == 1


def test_excluded_segments_do_not_enter_style_but_stay_as_context(monkeypatch):
    monkeypatch.setattr(svc_mod, "MIN_EXPOSURE_MIN", 0.0)
    prepared = _talk([(A, 0.0, 5.0, "tôi đi chợ nhé."), (A, 6.0, 11.0, "anh ơi nhé.")])
    prepared[1].has_music = True
    a = svc_mod.measure_style(prepared)[A].style
    assert {t.form: t.count for t in a.address_terms}["anh"] == 0


def test_speech_rate_uses_long_enough_units_and_clips_outliers(monkeypatch):
    monkeypatch.setattr(svc_mod, "MIN_DIST_N", 1)
    prepared = _talk([
        (A, 0.0, 4.0, words(8)),          # 2.0 w/s -> kept
        (B, 5.0, 5.9, words(4)),          # < 1.0 s -> skipped
        (A, 7.0, 8.0, words(10)),         # 10 w/s -> outlier, dropped
        (B, 9.0, 12.0, "một hai"),        # < 3 words -> skipped
    ])
    m = svc_mod.measure_style(prepared)
    assert m[A].style.speech_rate_wps.n == 1 and m[A].style.speech_rate_wps.p50 == 2.0
    assert m[B].style.speech_rate_wps.n == 0
```

- [ ] **Step 2: Chạy, thấy FAIL** — `python -m pytest tests/test_persona_profile.py -q -k "sentence or style or nfd or speech_rate or exposure"` -> `AttributeError`.

- [ ] **Step 3: Code**

```python
# imports to add
import re
from dataclasses import dataclass, field

from schemas.persona_profile import AddressTerm, Filler, Particle, Style
from utils.conversation_events import _mark_backchannels, speech_units

_SENT_END = re.compile(r"(?<!\d)[.!?…]+(?=\s|$)")


def split_sentences(text: str) -> List[str]:
    """Finished sentences only; the unfinished tail may be cut by a turn boundary."""
    out, last = [], 0
    for m in _SENT_END.finditer(text):
        out.append(text[last:m.end()])
        last = m.end()
    return out


def sentence_word_count(sentence: str) -> int:
    return sum(1 for t in sentence.split() if re.search(r"\w", t))


def fold_tokens(text: str) -> List[str]:
    return _fold(nfc(text)).split()


def count_form(tokens: List[str], form: List[str]) -> int:
    """Occurrences of `form` as a run of whole tokens (never a substring of a token)."""
    n = len(form)
    return sum(1 for i in range(len(tokens) - n + 1) if tokens[i:i + n] == form)


@dataclass
class StyleMeasure:
    style: Style
    speech_min: float
    n_sentences: int
    texts: List[str] = field(default_factory=list)    # clean, non-backchannel unit texts


def _turn_runs(units):
    """Consecutive non-backchannel units of one speaker, by start."""
    runs = []
    for u in units:
        if u.backchannel:
            continue
        if runs and runs[-1][0] == u.speaker:
            runs[-1][1].append(u)
        else:
            runs.append((u.speaker, [u]))
    return runs


def measure_style(prepared) -> Dict[str, StyleMeasure]:
    """Style of every speaker, once over the whole file (spec B4.4)."""
    speakers = sorted({str(s.speaker) for s in prepared})
    speech = [s for s in prepared if not is_empty(s)]
    clean_keys = {str(s.index) for s in speech if usable(s)}
    units = speech_units(speech, EVENT_PARAMS)
    _mark_backchannels(units, EVENT_PARAMS)
    is_clean = lambda u: all(k in clean_keys for k in u.keys)

    sentences: Dict[str, List[str]] = {sp: [] for sp in speakers}
    for speaker, run in _turn_runs(units):
        if all(is_clean(u) for u in run):
            sentences[speaker] += split_sentences(" ".join(u.text for u in run))

    out: Dict[str, StyleMeasure] = {}
    for sp in speakers:
        mine = [u for u in units if u.speaker == sp and not u.backchannel and is_clean(u)]
        speech_min = sum(u.duration for u in mine) / 60.0
        rates = [u.n_words / u.duration for u in mine
                 if u.n_words >= RATE_MIN_WORDS and u.duration >= RATE_MIN_UNIT_S
                 and RATE_MIN_WPS <= u.n_words / u.duration <= RATE_MAX_WPS]
        kept = [s for s in sentences[sp]
                if SENTENCE_MIN_WORDS <= sentence_word_count(s) <= SENTENCE_MAX_WORDS]
        lengths = [sentence_word_count(s) for s in kept]
        tokens = [t for u in mine for t in fold_tokens(u.text)]
        per_unit = [fold_tokens(u.text) for u in mine]

        def count(form: str) -> int:
            f = fold_tokens(form)
            return sum(count_form(t, f) for t in per_unit)

        enough = speech_min >= MIN_EXPOSURE_MIN
        fillers = ([Filler(f, rnd(count(f) / speech_min)) for f in FILLERS]
                   if enough else None)
        address = [AddressTerm(f, count(f)) for f in ADDRESS_TERMS] if enough else None
        particles = None
        if len(kept) >= MIN_DIST_N:
            ends = [fold_tokens(s)[-1] for s in kept if fold_tokens(s)]
            particles = [Particle(f, rnd(100.0 * sum(1 for e in ends if e == fold_tokens(f)[0])
                                         / len(kept))) for f in PARTICLES]
        style = Style(speech_rate_wps=make_dist(rates), words_per_sentence=make_dist(lengths),
                      fillers=fillers, particles=particles, address_terms=address,
                      llm_candidate_forms=None, description=None)
        out[sp] = StyleMeasure(style, speech_min, len(kept), [u.text for u in mine])
        del tokens
    return out
```

(Bỏ biến `tokens` thừa nếu linter báo; nó không được dùng.)

- [ ] **Step 4: Chạy, thấy PASS** — `python -m pytest tests/test_persona_profile.py -q`.

- [ ] **Step 5: Commit**

```bash
git add services/persona_profile_service.py tests/test_persona_profile.py
git commit -m "feat(persona): measure speech rate, sentence length and dictionary habits

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 1.5: Tương tác (region, cặp, che, backchannel/ngắt lời/lượt/độ trễ)

**Files:**
- Modify: `services/persona_profile_service.py`
- Test: `tests/test_persona_profile.py`

**Interfaces:**
- Produces: `split_regions(prepared) -> Tuple[List[list], int]` (regions gồm cả segment rỗng; số mối nối không xác định), `class Accumulator` (dataclass: `other_talk_s: float`, `backchannels: dict`, `interruptions: dict`, `cut_ins: dict`, `turns: dict`, `latencies: dict`), `measure_interaction(prepared) -> Tuple[Dict[str, Accumulator], int]` (khoá = mọi speaker; số thứ hai = `unknown_gaps_skipped`).
- Consumes: T1.3 (`exclusion_reason`, `is_empty`), T1.2 (`EVENT_PARAMS`, `EDGE_MARGIN_S`, `LATENCY_*`).

- [ ] **Step 1: Viết test fail**

```python
# --- interaction ----------------------------------------------------------------------

def _alt(starts, speakers=(A, B), length=4.5, text=None):
    """Alternating turns at the given starts, each `length` s long."""
    return [seg(i, speakers[i % 2], st, st + length, text or words(5)) for i, st in enumerate(starts)]


def test_gap_before_none_splits_regions_and_is_never_a_latency():
    rows = build(_alt([0, 5.4, 9.8, 14.2, 18.6]))          # gaps 0.9 apart
    acc, unknown = svc_mod.measure_interaction(rows)
    assert unknown == 0
    assert len(acc[B].latencies) == 2 and len(acc[A].latencies) == 1
    rows = build(_alt([0, 5.4, 9.8, 14.2, 18.6]), gaps={1: None})
    acc, unknown = svc_mod.measure_interaction(rows)
    assert unknown == 1
    assert len(acc[B].latencies) == 1                       # the seam into B1 gave nothing
    assert all(v >= svc_mod.LATENCY_FLOOR_S for a in acc.values() for v in a.latencies.values())


def test_seam_next_to_an_empty_segment_still_separates_the_sides():
    rows = build([seg(0, A, 0, 5, words(5)), seg(1, "E", 7, 7.5, ""),
                  seg(2, B, 8, 12, words(5)), seg(3, A, 12.5, 17, words(5)),
                  seg(4, B, 17.5, 22, words(5)), seg(5, A, 22.5, 27, words(5))],
                 gaps={1: None})
    acc, unknown = svc_mod.measure_interaction(rows)
    assert unknown == 1
    assert 3.0 not in {round(v, 3) for a in acc.values() for v in a.latencies.values()}


def test_crosses_cut_segment_stands_alone_and_masks_its_overlap():
    # A[0,10] crosses a cut; B[5,6] talks inside it; A2 follows right after.
    rows = build([seg(0, A, 0, 10, words(10), crosses_cut=True), seg(1, B, 5, 6, words(3)),
                  seg(2, A, 10.1, 14, words(8)), seg(3, B, 14.5, 18, words(6)),
                  seg(4, A, 18.5, 22, words(6)), seg(5, B, 22.5, 26, words(6)),
                  seg(6, A, 26.5, 30, words(6))], gaps={1: None})
    acc, _ = svc_mod.measure_interaction(rows)
    assert not any(abs(v - 4.1) < 1e-6 for a in acc.values() for v in a.latencies.values())
    assert acc[B].other_talk_s <= 12.0 and acc[A].other_talk_s >= 0
    # B[5,6] lies in the crossing segment's span, so it is not A's listening time.
    assert acc[A].other_talk_s < 5.0 + 3.5 + 3.5 + 3.5


def test_latency_rules():
    # A talks to 5.0, B comes in at 4.8: -0.2 (turn_switch would not see it).
    rows = build([seg(0, A, 0, 5.0, words(8)), seg(1, B, 4.8, 9.0, words(8)),
                  seg(2, A, 9.5, 14.0, words(8)), seg(3, B, 14.5, 19.0, words(8)),
                  seg(4, A, 19.5, 24.0, words(8))])
    acc, _ = svc_mod.measure_interaction(rows)
    assert sorted(round(v, 3) for v in acc[B].latencies.values())[0] == -0.2


def test_latency_drops_a_unit_inside_the_other_voice_and_out_of_range_values():
    rows = build([seg(0, A, 0, 20.0, words(30)), seg(1, B, 2.0, 4.0, words(5)),     # B inside A
                  seg(2, A, 21.0, 26.0, words(8)), seg(3, B, 40.0, 44.0, words(8)),  # +14 s: too slow
                  seg(4, A, 44.5, 49.0, words(8)), seg(5, B, 49.5, 54.0, words(8)),
                  seg(6, A, 54.5, 59.0, words(8))])
    acc, _ = svc_mod.measure_interaction(rows)
    for a in acc.values():
        assert all(svc_mod.LATENCY_FLOOR_S <= v <= svc_mod.LATENCY_CAP_S
                   for v in a.latencies.values())
    assert 14.0 not in {round(v, 3) for a in acc.values() for v in a.latencies.values()}


def test_a_cut_in_that_returns_to_the_holder_is_not_a_latency():
    # X (A) stops at 5.0; B cuts in [4.5, 5.6] with 3 non-backchannel words; A goes on.
    rows = build([seg(0, A, 0, 5.0, words(8)), seg(1, B, 4.5, 5.6, words(3)),
                  seg(2, A, 6.0, 10.0, words(8)), seg(3, B, 10.5, 15.0, words(8)),
                  seg(4, A, 15.5, 20.0, words(8)), seg(5, B, 20.5, 25.0, words(8)),
                  seg(6, A, 25.5, 30.0, words(8))])
    acc, _ = svc_mod.measure_interaction(rows)
    # A's answer after B's short cut-in is not "taking the turn": its key (6.0, 10.0) is absent.
    assert (A, 6.0, 10.0) not in acc[A].latencies


def test_three_speakers_pairs_are_masked_and_deduplicated():
    rows = build([seg(0, A, 0, 6, words(8)), seg(1, C, 6.5, 7.2, "ừ"),
                  seg(2, A, 7.4, 13, words(8)), seg(3, B, 13.5, 18, words(8)),
                  seg(4, A, 18.5, 24, words(8)), seg(5, B, 24.5, 30, words(8)),
                  seg(6, A, 30.5, 36, words(8))])
    acc, _ = svc_mod.measure_interaction(rows)
    keys = [k for a in acc.values() for k in a.latencies]
    assert len(keys) == len(set(keys))
    # Exposure of the two backchannel-type rates uses the same other_talk_s.
    assert acc[A].other_talk_s > 0 and acc[C].other_talk_s >= 0


def test_backchannel_events_in_turn_lexicon_and_overlap(monkeypatch):
    # A talks long; B says "ừ" x3 and "dạ" x1 while A goes on, plus one while A is quiet.
    rows = [seg(0, A, 0, 30, words(40))]
    for i, (st, tx) in enumerate([(5.0, "ừ"), (10.0, "ừ"), (15.0, "dạ"), (20.0, "ừ")], start=1):
        rows.append(seg(i, B, st, st + 0.4, tx))
    rows.append(seg(5, A, 32, 40, words(10)))
    rows.append(seg(6, B, 45, 45.4, "ừ"))                      # after A stopped: not in a turn
    rows.append(seg(7, A, 47, 52, words(8)))
    acc, _ = svc_mod.measure_interaction(build(rows))
    kinds = [e["text"] for e in acc[B].backchannels.values()]
    assert kinds.count("ừ") == 3 and kinds.count("dạ") == 1
    assert all(e["in_turn"] for e in acc[B].backchannels.values())


def test_interruption_and_yield_counts_use_cut_ins_on_the_speaker():
    # B cuts into A twice; A yields once (stops) and once keeps talking (competitive).
    rows = build([seg(0, A, 0, 6.0, words(10)), seg(1, B, 5.0, 12.0, words(12)),
                  seg(2, A, 20, 30.0, words(14)), seg(3, B, 24.0, 34.0, words(14)),
                  seg(4, A, 34.2, 44.0, words(14)), seg(5, B, 44.5, 52.0, words(10)),
                  seg(6, A, 52.5, 60.0, words(10))])
    acc, _ = svc_mod.measure_interaction(rows)
    types = sorted(e["type"] for e in acc[A].cut_ins.values())
    assert "interruption" in types
    assert all(e["by"] == B for e in acc[A].cut_ins.values())
    assert all(e["by"] == B for e in acc[B].interruptions.values())
```

(Số học tay trong các test tương tác là gần đúng; khi một kỳ vọng mâu thuẫn spec thì spec thắng và sửa test, xem "Quy tắc làm việc chung". Bắt buộc giữ ý nghĩa: không có độ trễ qua mối nối, không có 4.1 s giả, không đếm trùng, `in_turn` mới tính, `cut_ins` gồm cả `competitive_overlap`.)

- [ ] **Step 2: Chạy, thấy FAIL** — `python -m pytest tests/test_persona_profile.py -q -k "gap_before or seam or crosses or latency or three_speakers or backchannel_events or interruption"` -> `AttributeError: measure_interaction`.

- [ ] **Step 3: Code**

```python
# imports to add
from itertools import combinations
from typing import Tuple

from utils.conversation_events import _merge, analyze


def split_regions(prepared) -> Tuple[List[list], int]:
    """Contiguous regions; a region starts at an unknown gap and a crosses_cut segment stands alone."""
    regions: List[list] = []
    current: list = []
    unknown = 0
    for pos, s in enumerate(prepared):
        new = False
        if pos > 0 and getattr(s, "gap_before", None) is None:
            unknown += 1
            new = True
        crossing = bool(getattr(s, "crosses_cut", False))
        if (new or crossing) and current:
            regions.append(current)
            current = []
        current.append(s)
        if crossing:
            regions.append(current)
            current = []
    if current:
        regions.append(current)
    return regions, unknown


@dataclass
class Accumulator:
    other_talk_s: float = 0.0
    backchannels: dict = field(default_factory=dict)     # (type, by, start, end) -> event
    interruptions: dict = field(default_factory=dict)    # events where this speaker cut in
    cut_ins: dict = field(default_factory=dict)          # events where this speaker was cut into
    turns: dict = field(default_factory=dict)            # (speaker, start, end) -> seconds
    latencies: dict = field(default_factory=dict)        # (speaker, u.start, u.end) -> seconds


def _hits(mask, lo, hi) -> bool:
    return any(a < hi and b > lo for a, b in mask)


def _free_seconds(spans, lo, hi, mask) -> float:
    """Seconds of the union of `spans` clipped to [lo, hi] and outside `mask`."""
    clipped = _merge([(max(a, lo), min(b, hi)) for a, b in spans if min(b, hi) > max(a, lo)])
    inside = [(max(a, x), min(b, y)) for a, b in clipped for x, y in mask
              if min(b, y) > max(a, x)]
    return _union_seconds(clipped) - _union_seconds(inside)


def _mask(members, pair, region_ids, speech, lo, hi):
    """Where a pair's numbers cannot be trusted: third voices, excluded lines, outside overlap."""
    pad = EVENT_PARAMS.ipu_gap
    spans = []
    for s in members:
        if str(s.speaker) not in pair or exclusion_reason(s) is not None:
            spans.append((s.start - pad, s.end + pad))
    for s in speech:
        if id(s) not in region_ids and s.start < hi and s.end > lo:
            spans.append((s.start - pad, s.end + pad))
    return _merge(spans)


def _measure_pair(accs, pair, pair_segs, mask, lo, s_end):
    analysis = analyze(pair_segs, EVENT_PARAMS)
    talk = [u for u in analysis.units if not u.backchannel]
    for me in pair:
        other = pair[1] if me == pair[0] else pair[0]
        accs[me].other_talk_s += _free_seconds(
            [(u.start, u.end) for u in talk if u.speaker == other], lo, s_end, mask)

    for ev in analysis.events:
        if ev["start"] > s_end or _hits(mask, ev["start"], ev["end"]):
            continue
        kind = ev["type"]
        key = (kind, ev["by"], ev["start"], ev["end"])
        if kind == "backchannel" and ev.get("in_turn"):
            accs[ev["by"]].backchannels[key] = ev
        elif kind == "interruption":
            accs[ev["by"]].interruptions[key] = ev
        if kind in ("interruption", "competitive_overlap"):
            accs[ev["of"]].cut_ins[key] = ev

    runs = []
    for u in talk:
        if runs and runs[-1][0] == u.speaker:
            runs[-1][1].append(u)
        else:
            runs.append((u.speaker, [u]))
    p_cfg = EVENT_PARAMS
    for k in range(1, len(runs) - 1):                 # first and last turn are cut by the pair
        speaker, units = runs[k]
        u = units[0]
        start, end = u.start, units[-1].end
        if start <= s_end and not _hits(mask, start, end):
            accs[speaker].turns[(speaker, start, end)] = end - start
        other = runs[k - 1][0]
        before = [v for v in talk if v.speaker == other and v.start <= u.start]
        if not before or u.start > s_end:
            continue
        p = max(before, key=lambda v: v.end)
        if p.end > u.end:
            continue
        mine_before = [v for v in talk if v.speaker == speaker and v.start < p.start]
        prev = max(mine_before, key=lambda v: v.start, default=None)
        if (prev is not None and prev.end - p.start >= p_cfg.overlap_min
                and p.end <= prev.end + p_cfg.yield_within_seconds):
            continue                                  # a short cut-in, not a turn hand-over
        if _hits(mask, p.start, u.end):
            continue
        value = u.start - p.end
        if LATENCY_FLOOR_S <= value <= LATENCY_CAP_S:
            accs[speaker].latencies[(speaker, u.start, u.end)] = value


def measure_interaction(prepared) -> Tuple[Dict[str, Accumulator], int]:
    """Per-pair events, turns and latencies over seam-free regions (spec B4.1, B4.5)."""
    accs = {str(s.speaker): Accumulator() for s in prepared}
    regions, unknown = split_regions(prepared)
    speech = [s for s in prepared if not is_empty(s)]
    for region in regions:
        members = [s for s in region if not is_empty(s)]
        speakers = sorted({str(s.speaker) for s in members})
        if len(speakers) < 2:
            continue
        lo = min(float(s.start) for s in members)
        hi = max(float(s.end) for s in members)
        s_end = hi - EDGE_MARGIN_S
        if s_end <= lo:
            continue
        ids = {id(s) for s in region}
        for pair in combinations(speakers, 2):
            pair_segs = [s for s in members if str(s.speaker) in pair]
            mask = _mask(members, pair, ids, speech, lo, hi)
            _measure_pair(accs, pair, pair_segs, mask, lo, s_end)
    return accs, unknown
```

- [ ] **Step 4: Chạy, thấy PASS** — `python -m pytest tests/test_persona_profile.py -q`. Nếu một test tương tác fail vì số học tay, đối chiếu spec B4.1/B4.5, sửa test hoặc code theo spec.

- [ ] **Step 5: Commit**

```bash
git add services/persona_profile_service.py tests/test_persona_profile.py
git commit -m "feat(persona): measure backchannel, interruption and turn timing per pair

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 1.6: Gộp thành PersonaProfile và PersonaStats (không LLM)

**Files:**
- Modify: `services/persona_profile_service.py`
- Test: `tests/test_persona_profile.py`

**Interfaces:**
- Produces: `class Core` (dataclass: `profile: PersonaProfile`, `stats: PersonaStats`, `measures: Dict[str, StyleMeasure]`), `compute_core(prepared, recording_id) -> Core`, `base_warnings(prepared, usable_ratio) -> List[str]`.
- Warning chuỗi (đúng nguyên văn spec B6): `"backchannel_and_interruption_are_lower_bounds: diarization drops overlapped speech, so backchannel and interruption counts under-count"`, `"not_reviewed: {N} segment(s) have review_flags=None and were treated as unflagged"`, `"noise_unmeasured: {N} segment(s) have no noise measurement and were not excluded"`, `"low_usable_ratio: {value}"` (khi `usable_ratio < 0.5`). Cảnh báo `llm_unavailable` do T1.7 thêm.
- `llm_windows`/`llm_ok` = 0 ở đây. `speech_s` mỗi speaker = `_union_seconds` mọi segment của speaker (kể cả bị loại), làm tròn 3. `usable_ratio = Σ usable_s / Σ speech_s` (null nếu mẫu số 0), làm tròn 3.

- [ ] **Step 1: Viết test fail**

```python
# --- assembly -------------------------------------------------------------------------

def _no_thresholds(monkeypatch):
    for name in ("MIN_EXPOSURE_MIN", "MIN_EXPOSURE_INTERRUPTION_MIN"):
        monkeypatch.setattr(svc_mod, name, 0.0)
    for name in ("MIN_DIST_N", "MIN_RATIO_N", "MIN_LEXICON_N"):
        monkeypatch.setattr(svc_mod, name, 1)


def _talk_file():
    rows = []
    t = 0.0
    for i in range(10):
        rows.append(seg(2 * i, A, t, t + 4.0, "Tôi đi chợ nhé. ừm anh thấy sao?"))
        rows.append(seg(2 * i + 1, B, t + 4.5, t + 8.5, "Em thấy ổn ạ. Dạ vâng."))
        t += 9.0
    return build(rows)


def test_compute_core_has_every_key_and_exact_stats(monkeypatch):
    _no_thresholds(monkeypatch)
    core = svc_mod.compute_core(_talk_file(), "ep01")
    d = json.loads(json.dumps(core.profile.to_dict(), allow_nan=False))
    assert d["schema_version"] == 1 and d["source_type"] == "real"
    assert d["params_version"] == svc_mod.PARAMS_VERSION and d["recording_id"] == "ep01"
    assert d["usable_ratio"] == 1.0 and set(d["speakers"]) == {A, B}
    for speaker in d["speakers"].values():
        assert set(speaker) == {"speech_s", "style", "backchannel", "interruption", "turn"}
        assert set(speaker["style"]) == {
            "speech_rate_wps", "words_per_sentence", "fillers", "particles", "address_terms",
            "llm_candidate_forms", "description"}
        assert speaker["style"]["llm_candidate_forms"] is None
    assert d["speakers"][A]["speech_s"] == 40.0
    s = core.stats.to_dict()
    assert set(s["excluded_s"]) == set(svc_mod.EXCLUSION_KEYS) and s["llm_windows"] == 0
    assert s["warnings"][0].startswith("backchannel_and_interruption_are_lower_bounds")


def test_single_speaker_has_null_interaction(monkeypatch):
    _no_thresholds(monkeypatch)
    rows = build([seg(0, A, 0, 5, "tôi đi chợ."), seg(1, A, 6, 11, "anh nhé.")])
    d = svc_mod.compute_core(rows, "solo").profile.to_dict()["speakers"][A]
    assert d["backchannel"]["per_min_of_other"] == {"value": None, "count": 0, "exposure_min": 0.0}
    assert d["interruption"]["yield_rate"]["value"] is None
    assert d["turn"]["response_latency_s"]["n"] == 0
    assert d["style"]["fillers"] is not None            # style still measured


def test_low_data_speaker_keeps_keys_with_nulls_and_real_counts():
    rows = _talk_file()
    d = svc_mod.compute_core(rows, "ep").profile.to_dict()["speakers"][B]
    assert d["backchannel"]["per_min_of_other"]["value"] is None
    assert d["backchannel"]["per_min_of_other"]["exposure_min"] > 0
    assert d["backchannel"]["lexicon"] is None
    assert d["style"]["fillers"] is None                 # 40 s of speech < 5 min


def test_warnings_and_usable_ratio():
    rows = build([seg(0, A, 0, 10, "tôi đi chợ nhé", review_flags=[{"kind": "asr_garbage"}]),
                  seg(1, B, 10.5, 12, "dạ vâng ạ")])
    stats = svc_mod.compute_core(rows, "ep").stats
    assert any(w.startswith("not_reviewed: 1 segment(s)") for w in stats.warnings)
    assert any(w.startswith("noise_unmeasured: 2 segment(s)") for w in stats.warnings)
    assert any(w.startswith("low_usable_ratio: 0.13") for w in stats.warnings)
    assert stats.excluded_s["review_flag"] == 10.0
```

- [ ] **Step 2: Chạy, thấy FAIL** — `python -m pytest tests/test_persona_profile.py -q -k "compute_core or single_speaker or low_data or warnings_and"` -> `AttributeError: compute_core`.

- [ ] **Step 3: Code**

```python
# imports to add
from schemas.persona_profile import (
    BackchannelProfile, InterruptionProfile, LexiconItem, PersonaProfile, PersonaStats,
    SpeakerProfile, TurnProfile)

LOWER_BOUND_WARNING = ("backchannel_and_interruption_are_lower_bounds: diarization drops "
                       "overlapped speech, so backchannel and interruption counts under-count")


@dataclass
class Core:
    profile: PersonaProfile
    stats: PersonaStats
    measures: Dict[str, StyleMeasure]


def base_warnings(prepared, usable_ratio: Optional[float]) -> List[str]:
    out = [LOWER_BOUND_WARNING]
    unreviewed = sum(1 for s in prepared if getattr(s, "review_flags", None) is None)
    if unreviewed:
        out.append(f"not_reviewed: {unreviewed} segment(s) have review_flags=None "
                   "and were treated as unflagged")
    unmeasured = sum(1 for s in prepared if getattr(s, "noise_breakdown", None) is None)
    if unmeasured:
        out.append(f"noise_unmeasured: {unmeasured} segment(s) have no noise measurement "
                   "and were not excluded")
    if usable_ratio is not None and usable_ratio < 0.5:
        out.append(f"low_usable_ratio: {usable_ratio}")
    return out


def _lexicon(acc: Accumulator) -> Optional[List[LexiconItem]]:
    forms = [_fold(e["text"]) for e in acc.backchannels.values()]
    if len(forms) < MIN_LEXICON_N:
        return None
    counts: Dict[str, int] = {}
    for f in forms:
        counts[f] = counts.get(f, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [LexiconItem(f, rnd(c / len(forms))) for f, c in ranked]


def _speaker_profile(speech_s: float, measure: StyleMeasure, acc: Accumulator) -> SpeakerProfile:
    exposure = acc.other_talk_s / 60.0
    bcs = list(acc.backchannels.values())
    cut_ins = list(acc.cut_ins.values())
    return SpeakerProfile(
        speech_s=rnd(speech_s),
        style=measure.style,
        backchannel=BackchannelProfile(
            per_min_of_other=make_rate(len(bcs), exposure, MIN_EXPOSURE_MIN),
            lexicon=_lexicon(acc),
            overlap_ratio=make_ratio(sum(1 for e in bcs if e["overlap"] > 0), len(bcs))),
        interruption=InterruptionProfile(
            per_hour_of_other=make_rate(len(acc.interruptions), exposure,
                                        MIN_EXPOSURE_INTERRUPTION_MIN, per=60.0),
            yield_rate=make_ratio(
                sum(1 for e in cut_ins
                    if e["type"] == "interruption" and e["response"] == "yield"),
                len(cut_ins))),
        turn=TurnProfile(duration_s=make_dist(list(acc.turns.values())),
                         response_latency_s=make_dist(list(acc.latencies.values()))))


def compute_core(prepared, recording_id: str) -> Core:
    """Everything except the two LLM-written fields (P1, P2, P4)."""
    measures = measure_style(prepared)
    accs, unknown = measure_interaction(prepared)
    speakers = sorted(measures)
    speech_s = {sp: _union_seconds([(float(s.start), float(s.end)) for s in prepared
                                    if str(s.speaker) == sp]) for sp in speakers}
    usable_s = sum(_union_seconds([(float(s.start), float(s.end)) for s in prepared
                                   if str(s.speaker) == sp and usable(s)]) for sp in speakers)
    total = sum(speech_s.values())
    ratio = rnd(usable_s / total) if total > 0 else None
    profile = PersonaProfile(
        schema_version=SCHEMA_VERSION, params_version=PARAMS_VERSION, source_type="real",
        recording_id=recording_id, usable_ratio=ratio,
        speakers={sp: _speaker_profile(speech_s[sp], measures[sp], accs[sp]) for sp in speakers})
    stats = PersonaStats(excluded_s=excluded_seconds(prepared), unknown_gaps_skipped=unknown,
                         llm_windows=0, llm_ok=0, warnings=base_warnings(prepared, ratio))
    return Core(profile, stats, measures)
```

- [ ] **Step 4: Chạy, thấy PASS** — `python -m pytest tests/test_persona_profile.py -q`.

- [ ] **Step 5: Commit**

```bash
git add services/persona_profile_service.py tests/test_persona_profile.py
git commit -m "feat(persona): assemble speaker profiles and run stats from the measures

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 1.7: Qwen (cửa sổ, prompt, retry, xác nhận) và `PersonaProfileService`

**Files:**
- Modify: `services/persona_profile_service.py`
- Test: `tests/test_persona_profile.py`

**Interfaces:**
- Produces: `class ProfileResult`, `class PersonaProfileService` đúng hợp đồng; hàm thuần `pick_window(texts, limit) -> str`, `parse_reply(raw) -> Optional[dict]` (dict có khoá `new_forms` hoặc `description`), `validate_forms(raw_forms, texts, dictionary_forms) -> List[str]`, `validate_description(raw, rate_n) -> Optional[str]`.
- Consumes: T1.2-T1.6; `ask_in_batches` (`utils/llm_batches.py`), `objects_in`, `is_readable` (`utils/llm_json.py`), `reply_budget`.

- [ ] **Step 1: Viết test fail**

```python
# --- LLM ------------------------------------------------------------------------------

class FakeLLM:
    model_name = "fake/model"

    def __init__(self, reply='{"new_forms": [], "description": "Nói chậm."}', loaded=True):
        self.reply, self.loaded = reply, loaded
        self.batch_size, self.max_batch_tokens = 100, 0
        self.calls, self.system_prompts = [], []

    def ensure_loaded(self):
        return self.loaded

    def count_tokens(self, text):
        return len(text.split())

    def generate_texts(self, system_prompt, user_messages, max_new_tokens=512,
                       use_prefix=False, labels=None, **extra):
        self.calls.append(list(user_messages))
        self.system_prompts.append(system_prompt)
        out = [self.reply(m) if callable(self.reply) else self.reply for m in user_messages]
        return True, out


def _service(llm, **kw):
    return svc_mod.PersonaProfileService(llm, **kw)


def _prepared_talk():
    rows, t = [], 0.0
    for i in range(10):
        rows.append(seg(2 * i, A, t, t + 4.0, "Tôi thấy tức là bạn đúng. ừm anh nhé."))
        rows.append(seg(2 * i + 1, B, t + 4.5, t + 8.5, "Em thấy ổn ạ. Dạ vâng."))
        t += 9.0
    return svc_mod.prepare_segments(rows, None)


def test_constructor_rejects_unknown_keys_and_exposes_the_contract():
    with pytest.raises(TypeError):
        _service(FakeLLM(), window_tokens=5)
    s = _service(FakeLLM())
    assert (s.llm_enabled, s.thinking, s.max_new_tokens) == (True, False, 1024)
    assert _service(FakeLLM(), llm_enabled=False).llm_enabled is False


def test_namespace_changes_with_every_part_and_is_path_safe():
    base = _service(FakeLLM()).checkpoint_namespace_for("d1")
    assert base != _service(FakeLLM()).checkpoint_namespace_for("d2")
    assert base != _service(FakeLLM(), llm_enabled=False).checkpoint_namespace_for("d1")
    assert base != _service(FakeLLM(), thinking=True).checkpoint_namespace_for("d1")
    assert base != _service(FakeLLM(), max_new_tokens=2048).checkpoint_namespace_for("d1")
    assert svc_mod.PROFILE_PROMPT_VERSION in base and svc_mod.PARAMS_VERSION in base
    assert re.fullmatch(r"[A-Za-z0-9._-]+", base)


def test_llm_candidate_forms_validation():
    texts = ["tức là bạn đúng", "ừm tôi nghĩ vậy"]
    kept = svc_mod.validate_forms(["tức là", "xyz không có", "ạ", "ừm", "Tức Là", "bạn đúng"],
                                  texts, svc_mod.FILLERS + svc_mod.PARTICLES + svc_mod.ADDRESS_TERMS)
    assert kept == ["bạn đúng", "tức là"]              # sorted, deduplicated; "ạ" is a dictionary form
    assert svc_mod.validate_forms(["a b c d"], ["a b c d"], ()) == []      # more than 3 tokens
    assert svc_mod.validate_forms(["ạn"], ["bạn đúng"], ()) == []          # substring of a token
    many = [f"w{k}" for k in range(40)]
    assert len(svc_mod.validate_forms(many, [" ".join(many)], ())) == svc_mod.MAX_CANDIDATE_FORMS


def test_description_rules():
    assert svc_mod.validate_description("  Nói nhanh. ", 50) == "Nói nhanh."
    assert svc_mod.validate_description("", 50) is None
    assert svc_mod.validate_description("x" * 401, 50) is None
    assert svc_mod.validate_description("ok", svc_mod.MIN_DIST_N - 1) is None
    assert svc_mod.validate_description(None, 50) is None


def test_pick_window_is_even_and_bounded():
    texts = [f"đoạn {k:03d} " + "x" * 20 for k in range(100)]
    window = svc_mod.pick_window(texts, 500)
    assert len(window) <= 500 and window.startswith(texts[0])
    assert svc_mod.pick_window(["ngắn"], 500) == "ngắn"
    assert len(svc_mod.pick_window(["y" * 900], 500)) == 500


def test_llm_forms_description_and_counts_are_separate(monkeypatch):
    monkeypatch.setattr(svc_mod, "MIN_EXPOSURE_MIN", 0.0)
    reply = '{"new_forms": ["tức là", "không có trong lời", "ừm"], "description": "Nói chậm."}'
    llm = FakeLLM(reply)
    with_llm = _service(llm).profile(_prepared_talk(), "ep")
    without = _service(FakeLLM(), llm_enabled=False).profile(_prepared_talk(), "ep")
    a = with_llm.profile["speakers"][A]["style"]
    assert a["llm_candidate_forms"] == ["tức là"] and a["description"] == "Nói chậm."
    assert with_llm.stats["llm_windows"] == 2 and with_llm.stats["llm_ok"] == 2
    assert with_llm.complete is True
    off = without.profile["speakers"][A]["style"]
    assert off["llm_candidate_forms"] is None and off["description"] is None
    assert without.complete is True and llm.calls and not _service(FakeLLM(), llm_enabled=False).llm.calls
    for sp in (A, B):                                  # numbers identical with or without the LLM
        x, y = with_llm.profile["speakers"][sp], without.profile["speakers"][sp]
        for part in ("fillers", "particles", "address_terms", "speech_rate_wps",
                     "words_per_sentence"):
            assert x["style"][part] == y["style"][part]
        assert x["backchannel"] == y["backchannel"] and x["turn"] == y["turn"]


def test_empty_forms_list_is_not_null(monkeypatch):
    monkeypatch.setattr(svc_mod, "MIN_EXPOSURE_MIN", 0.0)
    res = _service(FakeLLM('{"new_forms": [], "description": "ok"}')).profile(_prepared_talk(), "ep")
    assert res.profile["speakers"][A]["style"]["llm_candidate_forms"] == []


def test_unreadable_reply_retries_once_then_gives_null_and_is_not_complete(monkeypatch):
    monkeypatch.setattr(svc_mod, "MIN_EXPOSURE_MIN", 0.0)
    llm = FakeLLM("blah blah không phải json")
    res = _service(llm).profile(_prepared_talk(), "ep")
    st = res.profile["speakers"][A]["style"]
    assert st["llm_candidate_forms"] is None and st["description"] is None
    assert res.stats["llm_windows"] == 2 and res.stats["llm_ok"] == 0
    assert res.complete is False
    assert len(llm.calls) == 2                          # first ask + one strict retry
    assert any(w.startswith("llm_unavailable") for w in res.stats["warnings"])
    assert "JSON" in llm.system_prompts[1]


def test_llm_not_loaded_is_not_complete_but_numbers_are_written(monkeypatch):
    monkeypatch.setattr(svc_mod, "MIN_EXPOSURE_MIN", 0.0)
    res = _service(FakeLLM(loaded=False)).profile(_prepared_talk(), "ep")
    assert res.complete is False and res.profile["speakers"][A]["style"]["fillers"] is not None
    assert any(w.startswith("llm_unavailable") for w in res.stats["warnings"])


def test_speakers_without_exposure_are_not_asked():
    llm = FakeLLM()
    res = _service(llm).profile(_prepared_talk(), "ep")          # 40 s < 5 min each
    assert not llm.calls and res.stats["llm_windows"] == 0
    assert res.profile["speakers"][A]["style"]["description"] is None
    assert res.complete is True
```

Thêm `import re` ở đầu file test.

- [ ] **Step 2: Chạy, thấy FAIL** — `python -m pytest tests/test_persona_profile.py -q -k "constructor or namespace or validation or description_rules or pick_window or llm_ or empty_forms or unreadable or not_loaded or without_exposure"` -> `AttributeError`.

- [ ] **Step 3: Code**

```python
# imports to add
from utils.llm_batches import ask_in_batches, reply_budget
from utils.llm_json import clean_reply, is_readable, objects_in

_WINDOW_TOKEN_ESTIMATE = 4000          # tokens of one LLM_WINDOW_CHARS window

PERSONA_SYSTEM_PROMPT = (
    "Bạn phân tích cách nói của MỘT người trong một cuộc ghi âm tiếng Việt, "
    "chỉ dựa trên lời của người đó và các số liệu đã đo được đưa kèm.\n"
    "Trả về DUY NHẤT một đối tượng JSON: "
    '{"new_forms": [...], "description": "..."}.\n'
    "- new_forms: các từ hoặc cụm (1 đến 3 từ) là từ đệm, tiểu từ cuối câu hoặc từ xưng hô, "
    "xuất hiện NGUYÊN VĂN trong lời của người này và CHƯA có trong các danh sách đã cho. "
    "Không có thì để [].\n"
    "- description: 1 đến 2 câu tiếng Việt mô tả phong cách nói. Không bịa chi tiết ngoài lời và số liệu.")
_STRICT_JSON_SUFFIX = ("\n\nNHẮC LẠI: chỉ trả về một đối tượng JSON như trên. "
                       "Không viết phân tích trước hoặc sau JSON.")


def pick_window(texts: List[str], limit: int) -> str:
    """Texts spread evenly by start until about `limit` characters (never past it)."""
    step = 1
    while step < len(texts) and sum(len(t) + 1 for t in texts[::step]) > limit:
        step += 1
    return "\n".join(texts[::step])[:limit]


def parse_reply(raw: Optional[str]) -> Optional[dict]:
    """The reply's JSON object if it holds `new_forms` or `description`, else None."""
    if raw is None or not is_readable(raw):
        return None
    for obj in objects_in(raw):
        if "new_forms" in obj or "description" in obj:
            return obj
    return None


def validate_forms(raw_forms, texts: List[str], dictionary_forms) -> List[str]:
    """Qwen's proposals that are real, new, 1-3 whole tokens of this speaker's words."""
    if not isinstance(raw_forms, list):
        return []
    known = {tuple(fold_tokens(f)) for f in dictionary_forms}
    token_lists = [fold_tokens(t) for t in texts]
    kept = set()
    for raw in raw_forms:
        if not isinstance(raw, str):
            continue
        form = fold_tokens(raw)
        if not 1 <= len(form) <= 3 or tuple(form) in known:
            continue
        if any(count_form(tokens, form) for tokens in token_lists):
            kept.add(" ".join(form))
    return sorted(kept)[:MAX_CANDIDATE_FORMS]


def validate_description(raw, rate_n: int) -> Optional[str]:
    if not isinstance(raw, str) or rate_n < MIN_DIST_N:
        return None
    text = raw.strip()
    if not text or len(text) > DESCRIPTION_MAX_CHARS:
        return None
    return text


@dataclass
class ProfileResult:
    profile: dict
    stats: dict
    complete: bool                       # True = safe to checkpoint
    skipped: Optional[str] = None


class PersonaProfileService:
    """Compute-only persona profile of one recording; the caller checkpoints and writes."""

    def __init__(self, llm, logger=None, *, llm_enabled: bool = True,
                 thinking: bool = False, max_new_tokens: int = 1024):
        self.llm = llm
        self.logger = logger
        self.llm_enabled = bool(llm_enabled)
        self.thinking = bool(thinking)
        self.max_new_tokens = reply_budget(max_new_tokens, self.thinking)
        self.unreadable_retries = 1

    # -- contract used by the pipeline ------------------------------------
    def prepare(self, segments, timeline, noise=None) -> list:
        return prepare_segments(segments, timeline, noise)

    def source_digest(self, prepared) -> str:
        return source_digest_of(prepared)

    def checkpoint_namespace_for(self, source_digest: str) -> str:
        model = getattr(self.llm, "model_name", None) or "unknown"
        key = (f"{PROFILE_PROMPT_VERSION}-{'think' if self.thinking else 'nothink'}"
               f"-{'llm' if self.llm_enabled else 'nollm'}-{model}"
               f"-new{self.max_new_tokens}-{PARAMS_VERSION}-{source_digest}")
        return re.sub(r"[^A-Za-z0-9._-]+", "_", key)

    def profile(self, prepared, recording_id: str) -> ProfileResult:
        if not prepared:
            return ProfileResult({}, {}, False, "no_segments")
        core = compute_core(prepared, recording_id)
        complete = True
        if self.llm_enabled:
            complete = self._ask_llm(core)
        return ProfileResult(core.profile.to_dict(), core.stats.to_dict(), complete)

    # -- LLM --------------------------------------------------------------
    def _windows_per_call(self) -> int:
        batch = max(1, int(getattr(self.llm, "batch_size", 1) or 1))
        limit = int(getattr(self.llm, "max_batch_tokens", 0) or 0)
        if limit:
            batch = min(batch, max(1, limit // _WINDOW_TOKEN_ESTIMATE))
        return batch

    @staticmethod
    def _message(label: str, measure: StyleMeasure) -> str:
        rate, sent = measure.style.speech_rate_wps, measure.style.words_per_sentence
        seen = [f.form for f in measure.style.fillers or [] if f.per_min > 0]
        seen += [p.form for p in measure.style.particles or [] if p.per_100_sent > 0]
        seen += [a.form for a in measure.style.address_terms or [] if a.count > 0]
        known = ", ".join(FILLERS + PARTICLES + ADDRESS_TERMS)
        return (f"Người nói: {label}\n"
                f"Danh sách đã có (không lặp lại): {known}\n"
                f"Số liệu đã đo: nhịp {rate.p50} từ/giây, độ dài câu trung vị {sent.p50} từ, "
                f"form thấy trong lời: {', '.join(seen) or 'không có'}\n\n"
                f"Lời của người này:\n{pick_window(measure.texts, LLM_WINDOW_CHARS)}\n\n"
                "JSON:")

    def _ask(self, messages: List[str]):
        replies, unanswered = ask_in_batches(
            self.llm, PERSONA_SYSTEM_PROMPT, messages, per_call=self._windows_per_call(),
            max_new_tokens=self.max_new_tokens, label="persona window",
            logger=self.logger, thinking=self.thinking)
        pending = [i for i, r in enumerate(replies)
                   if r is not None and parse_reply(r) is None]
        for _ in range(self.unreadable_retries):
            if not pending:
                break
            retry, _ = ask_in_batches(
                self.llm, PERSONA_SYSTEM_PROMPT + _STRICT_JSON_SUFFIX,
                [messages[i] + _STRICT_JSON_SUFFIX for i in pending],
                per_call=min(self._windows_per_call(), len(pending)),
                max_new_tokens=self.max_new_tokens, label="persona JSON retry",
                logger=self.logger, thinking=False)
            still = []
            for original, reply in zip(pending, retry):
                if reply is not None:
                    replies[original] = reply
                if reply is None or parse_reply(reply) is None:
                    still.append(original)
            pending = still
        return replies

    def _ask_llm(self, core: Core) -> bool:
        """Fill llm_candidate_forms / description. True when nothing is left to retry."""
        asked = [sp for sp, m in sorted(core.measures.items())
                 if m.speech_min >= MIN_EXPOSURE_MIN]
        if not asked:
            return True
        if not self.llm.ensure_loaded():
            core.stats.warnings.append("llm_unavailable: the LLM could not be loaded")
            if self.logger:
                self.logger.warning("[persona] LLM not available; numbers only")
            return False
        replies = self._ask([self._message(sp, core.measures[sp]) for sp in asked])
        core.stats.llm_windows = len(asked)
        dictionary = FILLERS + PARTICLES + ADDRESS_TERMS
        for sp, raw in zip(asked, replies):
            parsed = parse_reply(raw)
            if parsed is None:
                continue
            core.stats.llm_ok += 1
            style = core.profile.speakers[sp].style
            style.llm_candidate_forms = validate_forms(
                parsed.get("new_forms"), core.measures[sp].texts, dictionary)
            style.description = validate_description(
                parsed.get("description"), style.speech_rate_wps.n)
        if core.stats.llm_ok == 0:
            core.stats.warnings.append("llm_unavailable: no window returned readable JSON")
        return core.stats.llm_ok == core.stats.llm_windows
```

(Chú ý: `core.profile.speakers[sp].style` là CÙNG đối tượng `Style` với `measures[sp].style`, nên gán trực tiếp là đủ. `profile()` gọi `.to_dict()` SAU `_ask_llm`.)

- [ ] **Step 4: Chạy, thấy PASS** — `python -m pytest tests/test_persona_profile.py -q`.

- [ ] **Step 5: Commit**

```bash
git add services/persona_profile_service.py tests/test_persona_profile.py
git commit -m "feat(persona): ask Qwen for a description and new forms, verify them in code

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 1.8: Hợp đồng đầu-cuối, tính xác định, đầu vào rỗng

**Files:**
- Test: `tests/test_persona_profile.py` (chỉ thêm test; sửa code nếu test lộ lỗi)
- Modify (nếu cần): `services/persona_profile_service.py`

- [ ] **Step 1: Viết test**

```python
# --- end to end -----------------------------------------------------------------------

def test_empty_input_is_skipped():
    res = _service(FakeLLM()).profile([], "ep")
    assert res.skipped == "no_segments" and res.complete is False
    assert res.profile == {} and res.stats == {}


def test_profile_is_deterministic_and_does_not_touch_the_input(monkeypatch):
    monkeypatch.setattr(svc_mod, "MIN_EXPOSURE_MIN", 0.0)
    rows = [seg(i, A if i % 2 == 0 else B, i * 5.0, i * 5.0 + 4.0,
                "Tôi đi chợ nhé. ừm anh thấy sao?") for i in range(12)]
    snapshot = [(s.words, s.gap_before, s.noise_breakdown, s.text, s.speaker) for s in rows]
    svc = _service(FakeLLM())
    first = svc.profile(svc.prepare(rows, None), "ep")
    second = svc.profile(svc.prepare(rows, None), "ep")
    dump = lambda r: json.dumps(r.profile, ensure_ascii=False, indent=2, allow_nan=False)
    assert dump(first) == dump(second)
    assert json.dumps(first.stats, sort_keys=True) == json.dumps(second.stats, sort_keys=True)
    assert [(s.words, s.gap_before, s.noise_breakdown, s.text, s.speaker) for s in rows] == snapshot
    digest = svc.source_digest(svc.prepare(rows, None))
    assert digest == svc.source_digest(svc.prepare(rows, None))
    assert svc.checkpoint_namespace_for(digest) == svc.checkpoint_namespace_for(digest)
```

- [ ] **Step 2: Chạy.** `python -m pytest tests/test_persona_profile.py -q`. Nếu PASS ngay thì ghi nhận (lõi đã đúng); nếu FAIL, sửa code lõi tối thiểu.

- [ ] **Step 3: Chạy test của track và toàn bộ**

```bash
python -m pytest tests/test_persona_profile.py -q
python -m pytest tests -q
rtk proxy python -m pytest tests -q -p no:cacheprovider -rf 2>&1 | grep -E "^FAILED" | sed 's/ - .*//' | sort > "$TMPDIR/after_track1.txt"
diff "$TMPDIR"/baseline_failed_*.txt "$TMPDIR/after_track1.txt"
```

Expected: `test_persona_profile.py` toàn pass; toàn bộ chỉ có đúng các fail của baseline (62), không dòng mới trong `diff` (test của Track 2 có thể đang dở: nếu thấy fail ở file thuộc Track 2, đó là việc của Track 2, ghi chú trong báo cáo).

- [ ] **Step 4: Commit**

```bash
git add tests/test_persona_profile.py services/persona_profile_service.py
git commit -m "test(persona): pin determinism, empty input and input immutability

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

# TRACK 2: Tích hợp (6 task)

Files: sửa `services/stage_output_service.py`, `utils/performance_report.py`, `utils/batch.py`, `services/pipeline_service.py`, `main.py`, `config.json`; test: sửa `tests/test_stage_output.py`, `tests/test_batch.py`, `tests/test_review_select_align_order.py`, `tests/test_stop_after_config.py`, tạo `tests/test_persona_profile_pipeline.py`. Không import `services.persona_profile_service` ở bất kỳ file nào trừ `main.py` (import trễ). Dùng `FakePersona` ở mục "Hợp đồng".

Trước khi bắt đầu: ghi baseline (mục "Baseline test").

### Task 2.1: `StageOutputService` — thư mục, `_write_json`, `write_persona_profile`

**Files:**
- Modify: `services/stage_output_service.py` (STAGES dòng ~99-110, `_write_json` dòng ~125-134, thêm writer cạnh `write_review`)
- Test: `tests/test_stage_output.py`

**Interfaces:**
- Produces: `STAGES["persona_profile"] == "07c_persona_profile"`; `_write_json(self, path, payload, *, allow_nan=True) -> bool`; `write_persona_profile(self, profile: dict, stats: dict) -> Optional[str]`.

- [ ] **Step 1: Viết test fail** (nối vào `tests/test_stage_output.py`)

```python
def test_persona_profile_writes_exactly_two_files(tmp_path):
    so = StageOutputService(str(tmp_path))
    profile = {"schema_version": 1, "speakers": {"A": {"x": None, "y": 0.0, "z": []}}}
    so.write_persona_profile(profile, {"warnings": []})
    d = tmp_path / "07c_persona_profile"
    assert sorted(p.name for p in d.iterdir()) == ["profile.json", "stats.json"]
    back = json.loads((d / "profile.json").read_text(encoding="utf-8"))
    assert back["speakers"]["A"] == {"x": None, "y": 0.0, "z": []}
    assert "persona_profile" in so.manifest["stages"]


def test_persona_profile_keeps_vietnamese_text_unescaped(tmp_path):
    so = StageOutputService(str(tmp_path))
    so.write_persona_profile({"d": "Nói nhanh"}, {})
    assert "Nói nhanh" in (tmp_path / "07c_persona_profile" / "profile.json").read_text(encoding="utf-8")


def test_persona_profile_with_nan_raises_and_leaves_no_half_written_file(tmp_path):
    so = StageOutputService(str(tmp_path))
    with pytest.raises(RuntimeError):
        so.write_persona_profile({"v": float("nan")}, {})
    assert not (tmp_path / "07c_persona_profile" / "profile.json").exists()


def test_other_stages_still_allow_nan(tmp_path):
    so = StageOutputService(str(tmp_path))
    assert so._write_json(str(tmp_path / "x.json"), {"v": float("nan")}) is True


def test_persona_profile_is_a_noop_when_stage_output_is_disabled(tmp_path):
    so = StageOutputService(str(tmp_path), enabled=False)
    assert so.write_persona_profile({"a": 1}, {}) is None
    assert not (tmp_path / "07c_persona_profile").exists()
```

Thêm `import pytest` ở đầu file nếu chưa có.

- [ ] **Step 2: Chạy, thấy FAIL** — `python -m pytest tests/test_stage_output.py -q` -> `AttributeError: write_persona_profile`.

- [ ] **Step 3: Code**

Trong `STAGES` thêm `"persona_profile": "07c_persona_profile",` ngay sau dòng `"transcript_review": "07b_transcript_review",`. Thay `_write_json`:

```python
    def _write_json(self, path: str, payload, *, allow_nan: bool = True) -> bool:
        try:
            # Serialize first: a payload that cannot be written must not leave a half file.
            text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=allow_nan)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"[stage-out] failed writing {path}: {e}")
            return False
```

Thêm writer (đọc `_finish` để lặp lại đúng phần manifest/log của nó):

```python
    def write_persona_profile(self, profile: dict, stats: dict) -> Optional[str]:
        """profile.json + stats.json of the persona stage; NaN/inf is an error, not a file."""
        if not self.enabled:
            return None
        d = self.stage_dir("persona_profile")
        for name, payload in (("profile.json", profile), ("stats.json", stats)):
            if not self._write_json(os.path.join(d, name), payload, allow_nan=False):
                raise RuntimeError(f"persona_profile: could not write {name}")
        self.manifest["stages"]["persona_profile"] = {
            "dir": self.STAGES["persona_profile"], "stats": stats}
        if self.logger:
            self.logger.info(f"[stage-out] persona_profile -> {d}")
        return d
```

(Nếu `_finish` còn ghi manifest file xuống đĩa sau khi cập nhật `self.manifest`, gọi cùng hàm đó; xem dòng ~293-300 của file.)

- [ ] **Step 4: Chạy, thấy PASS** — `python -m pytest tests/test_stage_output.py -q`.

- [ ] **Step 5: Commit**

```bash
git add services/stage_output_service.py tests/test_stage_output.py
git commit -m "feat(persona): write persona profile and stats from the stage output service

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 2.2: `performance_report` + `utils/batch.py`

**Files:**
- Modify: `utils/performance_report.py` (`_STAGE_ORDER`, dòng ~10-13), `utils/batch.py` (`PIPELINE_STAGES` ~144-148, `_opt_in_stages` ~197-198, `_stage_parallelism` ~461)
- Test: `tests/test_batch.py`

- [ ] **Step 1: Viết test fail** (nối vào `tests/test_batch.py`)

```python
def test_persona_profile_pass_sits_between_review_and_selection():
    assert PIPELINE_STAGES.index("transcript_review") < PIPELINE_STAGES.index(
        "persona_profile") < PIPELINE_STAGES.index("conversation_selection")


def test_persona_profile_pass_is_skipped_unless_opted_in():
    files = ["f1", "f2"]
    pipe = FakePipeline()
    run_batch_by_stage(pipe, _args(
        step_speaker_relabel=False, step_transcript_review=True,
        step_word_alignment=True, step_conversation_exports=False), {}, files)
    assert "persona_profile" not in [s for s, _ in pipe.calls]


def test_persona_profile_pass_runs_for_every_file_before_the_next_stage():
    files = ["f1", "f2", "f3"]
    pipe = FakePipeline()
    run_batch_by_stage(pipe, _args(
        step_speaker_relabel=False, step_transcript_review=True, step_persona_profile=True,
        step_word_alignment=True, step_conversation_exports=True), {}, files)
    order = [s for s, _ in pipe.calls]
    assert [p for s, p in pipe.calls if s == "persona_profile"] == files
    last_persona = max(i for i, s in enumerate(order) if s == "persona_profile")
    first_after = min(i for i, s in enumerate(order) if s == "conversation_selection")
    assert last_persona < first_after


def test_performance_report_knows_the_persona_stage():
    from utils.performance_report import _STAGE_ORDER
    assert _STAGE_ORDER.index("transcript_review") < _STAGE_ORDER.index(
        "persona_profile") < _STAGE_ORDER.index("word_alignment")
```

(Giữ `_args(...)` như các test lân cận; nếu `_args` cần khoá `step_persona_profile` khác tên, đọc helper ở đầu `tests/test_batch.py` và theo đó.)

- [ ] **Step 2: Chạy, thấy FAIL** — `python -m pytest tests/test_batch.py -q -k persona` -> `ValueError: 'persona_profile' is not in tuple` / `AssertionError`.

- [ ] **Step 3: Code**

`PIPELINE_STAGES`: chèn `"persona_profile",` giữa `"transcript_review"` và `"conversation_selection"`.
`_opt_in_stages`: thêm `"persona_profile"`.
`_stage_parallelism`: đổi tuple thành `("speaker_relabel", "transcript_review", "persona_profile")`.
`_STAGE_ORDER`: thêm `"persona_profile"` sau `"transcript_review"`.

- [ ] **Step 4: Chạy, thấy PASS** — `python -m pytest tests/test_batch.py tests/test_performance_report.py -q` (bỏ file không tồn tại; `ls tests | grep performance`) rồi `python -m pytest tests/test_batch.py -q`.

- [ ] **Step 5: Commit**

```bash
git add utils/batch.py utils/performance_report.py tests/test_batch.py
git commit -m "feat(persona): add the persona_profile batch pass and report order

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 2.3: `pipeline_service` — thứ tự dừng, LLM residency, `__init__`

**Files:**
- Modify: `services/pipeline_service.py` (`__init__` ~50-64, `parallel_stage_view` ~220, `_STOP_ORDER` ~592-595, `_next_llm_step_after` ~484-506, `_release_llm_after` ~527)
- Test: `tests/test_review_select_align_order.py`

- [ ] **Step 1: Viết test fail** (nối vào cuối phần "LLM residency" của `tests/test_review_select_align_order.py`)

```python
def test_stop_order_places_persona_profile_between_review_and_selection():
    order = PipelineService._STOP_ORDER
    assert order.index("transcript_review") < order.index("persona_profile") \
        < order.index("conversation_selection")
    args = types.SimpleNamespace(stop_after="persona_profile")
    assert PipelineService._reaches(args, "persona_profile")
    assert not PipelineService._reaches(args, "conversation_selection")
    assert not PipelineService._reaches(args, "word_alignment")


class _Persona:
    llm_enabled = True


def test_next_llm_step_passes_through_persona_when_it_uses_the_llm():
    pipe = _pipeline(review_svc=object(), persona_svc=_Persona(),
                     conversation_export_svc=object())
    args = types.SimpleNamespace(step_transcript_review=True, step_persona_profile=True,
                                 step_conversation_exports=True)
    assert pipe._next_llm_step_after(args, "transcript_review") == "persona_profile"
    assert pipe._next_llm_step_after(args, "persona_profile") == "conversation_selection"
    assert pipe._next_llm_step_after(args, "refinement") == "transcript_review"


def test_a_persona_stage_without_llm_is_not_an_llm_consumer():
    off = _Persona()
    off.llm_enabled = False
    pipe = _pipeline(review_svc=object(), persona_svc=off, conversation_export_svc=object())
    args = types.SimpleNamespace(step_transcript_review=True, step_persona_profile=True,
                                 step_conversation_exports=True)
    assert pipe._next_llm_step_after(args, "transcript_review") == "conversation_selection"


def test_persona_profile_off_changes_nothing_for_existing_stages():
    pipe = _pipeline(review_svc=object(), persona_svc=_Persona(),
                     conversation_export_svc=object())
    args = types.SimpleNamespace(step_transcript_review=True, step_conversation_exports=True)
    assert pipe._next_llm_step_after(args, "transcript_review") == "conversation_selection"


def test_the_llm_is_kept_from_review_into_persona_then_released_after_it():
    pipe = _pipeline(review_svc=object(), persona_svc=_Persona())
    perf = {"enabled": True, "stages": {"refinement": {"keep_llm_across_relabel": True}}}
    args = types.SimpleNamespace(performance_config=perf, batch_continues=True,
                                 step_transcript_review=True, step_persona_profile=True,
                                 keep_models=False)
    unloaded = []
    pipe.refinement_svc = types.SimpleNamespace(unload=lambda: unloaded.append(1))
    pipe._defer_or_run = lambda fn: fn()
    pipe._release_llm_after(args, "transcript_review")
    assert not unloaded                                   # kept for persona_profile
    pipe._release_llm_after(args, "persona_profile")      # nothing after it uses the LLM
    assert unloaded
    unloaded.clear()
    args.batch_continues = False
    pipe._release_llm_after(args, "transcript_review")
    assert unloaded


def test_persona_profile_is_a_cross_file_stage_view():
    pipe = _pipeline(persona_svc=_Persona())
    pipe.diarization_svc = object()
    assert pipe.parallel_stage_view("persona_profile") is pipe
```

(Kiểm `parallel_stage_view` đã dùng chữ ký `parallel_stage_view(self, stage)`; xem dòng ~215 và chỉnh test cho đúng chữ ký thật.)

- [ ] **Step 2: Chạy, thấy FAIL** — `python -m pytest tests/test_review_select_align_order.py -q` -> `TypeError: unexpected keyword persona_svc`.

- [ ] **Step 3: Code**

- `__init__`: thêm tham số cuối `persona_svc=None` (sau `input_audio_gate=None`) và `self.persona_svc = persona_svc` cạnh `self.review_svc`.
- `_STOP_ORDER`: thêm `"persona_profile"` giữa `"transcript_review"` và `"conversation_selection"`.
- `parallel_stage_view`: `cross_file = (stage in ("speaker_relabel", "transcript_review", "persona_profile") or ...)`.
- `_next_llm_step_after`: thêm

```python
        persona_on = (getattr(self, "persona_svc", None) is not None
                      and bool(getattr(self.persona_svc, "llm_enabled", False))
                      and opt_in_step_enabled(args, "persona_profile"))
```
rồi: `refinement` / `speaker_relabel` / `transcript_review` trả `"persona_profile"` nếu `persona_on` (đứng ngay trước nhánh `conversation_selection`, sau `transcript_review`), và thêm `if stage == "persona_profile": return "conversation_selection" if selection_on else None`. Cập nhật docstring thứ tự: `refinement, speaker_relabel, transcript_review, persona_profile, conversation_selection`.
- `_release_llm_after`: tuple `("speaker_relabel", "transcript_review", "persona_profile")`.

- [ ] **Step 4: Chạy, thấy PASS** — `python -m pytest tests/test_review_select_align_order.py tests/test_batch.py -q`.

- [ ] **Step 5: Commit**

```bash
git add services/pipeline_service.py tests/test_review_select_align_order.py
git commit -m "feat(persona): order the persona stage and keep the LLM resident for it

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 2.4: `_profile_personas` (checkpoint, allow_compute, lỗi)

**Files:**
- Modify: `services/pipeline_service.py` (thêm phương thức sau `_review_transcript`, trước `_align_words`)
- Create: `tests/test_persona_profile_pipeline.py`

**Interfaces:**
- Consumes: hợp đồng `PersonaProfileService` (fake), `stage_out.write_persona_profile`, `CheckpointManager` (dùng `_Checkpoint` fake như `tests/test_review_select_align_order.py`).
- Produces: `PipelineService._profile_personas(self, args, checkpoint, stage_out, transcripts, audio_path, *, allow_compute=True) -> None`.

- [ ] **Step 1: Viết test fail** (`tests/test_persona_profile_pipeline.py`)

```python
"""persona_profile wiring in PipelineService, with a fake service (no GPU, no models).

Run:  python -m pytest tests/test_persona_profile_pipeline.py -q     (from podcast-pipeline/)
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.pipeline_service import PipelineService
from services.stage_output_service import StageOutputService

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Checkpoint:
    def __init__(self):
        self.namespaces, self.store = {}, {}

    def _key(self, name):
        return (name, self.namespaces.get(name))

    def load_if_committed(self, name, **kw):
        return self.store.get(self._key(name))

    def save(self, name, payload, **kw):
        self.store[self._key(name)] = payload


class FakePersona:
    llm_enabled = True

    def __init__(self, result=None, boom=False):
        self.result, self.calls, self.boom = result, [], boom

    def prepare(self, segments, timeline, noise=None):
        return list(segments)

    def source_digest(self, prepared):
        return "d1"

    def checkpoint_namespace_for(self, digest):
        return f"persona-ns-{digest}"

    def profile(self, prepared, recording_id):
        self.calls.append(recording_id)
        if self.boom:
            raise ValueError("boom")
        return self.result


def _result(complete=True, skipped=None):
    return types.SimpleNamespace(profile={"recording_id": "ep", "speakers": {}},
                                 stats={"warnings": []}, complete=complete, skipped=skipped)


def _pipe(svc):
    pipe = PipelineService(*(None,) * 8, persona_svc=svc)
    pipe.timeline = []
    return pipe


def _segs():
    return [types.SimpleNamespace(index="0", speaker="A", text="x", start=0.0, end=1.0)]


def _profile_exists(tmp_path):
    return (tmp_path / "07c_persona_profile" / "profile.json").exists()


def test_compute_checkpoints_and_writes_both_files(tmp_path):
    svc, ckpt = FakePersona(_result()), _Checkpoint()
    out = StageOutputService(str(tmp_path))
    _pipe(svc)._profile_personas(types.SimpleNamespace(), ckpt, out, _segs(), "/a/ep.mp3",
                                 allow_compute=True)
    assert svc.calls == ["ep"]
    assert ("persona_profile", "persona-ns-d1") in ckpt.store
    assert ckpt.store[("persona_profile", "persona-ns-d1")]["profile"]["recording_id"] == "ep"
    assert _profile_exists(tmp_path) and (tmp_path / "07c_persona_profile" / "stats.json").exists()


def test_an_incomplete_result_is_written_but_not_remembered(tmp_path):
    svc, ckpt = FakePersona(_result(complete=False)), _Checkpoint()
    out = StageOutputService(str(tmp_path))
    _pipe(svc)._profile_personas(types.SimpleNamespace(), ckpt, out, _segs(), "/a/ep.mp3")
    assert not ckpt.store and _profile_exists(tmp_path)


def test_a_later_pass_loads_the_checkpoint_and_never_calls_the_service(tmp_path):
    svc, ckpt = FakePersona(_result()), _Checkpoint()
    ckpt.namespaces["persona_profile"] = "persona-ns-d1"
    ckpt.store[("persona_profile", "persona-ns-d1")] = {
        "profile": {"recording_id": "ep"}, "stats": {"warnings": []}}
    out = StageOutputService(str(tmp_path))
    _pipe(svc)._profile_personas(types.SimpleNamespace(), ckpt, out, _segs(), "/a/ep.mp3",
                                 allow_compute=False)
    assert svc.calls == [] and _profile_exists(tmp_path)


def test_no_checkpoint_and_no_compute_writes_nothing_and_skips_the_llm(tmp_path):
    svc, ckpt = FakePersona(_result()), _Checkpoint()
    out = StageOutputService(str(tmp_path))
    _pipe(svc)._profile_personas(types.SimpleNamespace(), ckpt, out, _segs(), "/a/ep.mp3",
                                 allow_compute=False)
    assert svc.calls == [] and not _profile_exists(tmp_path)
    assert ckpt.namespaces["persona_profile"] == "persona-ns-d1"      # still selected


def test_empty_or_none_transcripts_write_nothing(tmp_path):
    for transcripts in ([], None):
        svc, ckpt = FakePersona(_result()), _Checkpoint()
        out = StageOutputService(str(tmp_path))
        _pipe(svc)._profile_personas(types.SimpleNamespace(), ckpt, out, transcripts, "/a/ep.mp3")
        assert svc.calls == [] and not ckpt.store and not _profile_exists(tmp_path)


def test_a_skipped_result_is_not_written_or_saved(tmp_path):
    svc, ckpt = FakePersona(_result(complete=False, skipped="no_segments")), _Checkpoint()
    out = StageOutputService(str(tmp_path))
    _pipe(svc)._profile_personas(types.SimpleNamespace(), ckpt, out, _segs(), "/a/ep.mp3")
    assert not ckpt.store and not _profile_exists(tmp_path)


def test_an_exception_is_logged_and_swallowed(tmp_path):
    svc, ckpt = FakePersona(boom=True), _Checkpoint()
    out = StageOutputService(str(tmp_path))
    logged = []
    pipe = _pipe(svc)
    pipe.logger = types.SimpleNamespace(
        info=lambda *a, **k: None, warning=lambda *a, **k: None,
        error=lambda m, *a, **k: logged.append(m), exception=lambda m, *a, **k: logged.append(m))
    pipe._profile_personas(types.SimpleNamespace(), ckpt, out, _segs(), "/a/ep.mp3")
    assert logged and not ckpt.store and not _profile_exists(tmp_path)


def test_the_stage_does_not_touch_the_real_transcripts(tmp_path):
    svc, ckpt = FakePersona(_result()), _Checkpoint()
    segs = _segs()
    before = dict(vars(segs[0]))
    _pipe(svc)._profile_personas(types.SimpleNamespace(), ckpt, StageOutputService(str(tmp_path)),
                                 segs, "/a/ep.mp3")
    assert vars(segs[0]) == before
```

- [ ] **Step 2: Chạy, thấy FAIL** — `python -m pytest tests/test_persona_profile_pipeline.py -q` -> `AttributeError: _profile_personas`.

- [ ] **Step 3: Code** (thêm vào `PipelineService`, sau `_review_transcript`)

```python
    def _profile_personas(self, args, checkpoint, stage_out, transcripts, audio_path, *,
                          allow_compute=True):
        """Per-speaker habits of this recording; reads the transcript, changes nothing.

        Later passes of a batch only reload the checkpoint (`allow_compute` false), so
        the LLM is never loaded again just to find nothing to do.
        """
        svc = self.persona_svc
        if not transcripts:
            return
        try:
            prepared = svc.prepare(transcripts, getattr(self, "timeline", None),
                                   getattr(self, "noise_track", None))
            checkpoint.namespaces["persona_profile"] = svc.checkpoint_namespace_for(
                svc.source_digest(prepared))
            payload = checkpoint.load_if_committed("persona_profile")
            if payload and payload.get("profile") is not None and payload.get("stats") is not None:
                if self.logger:
                    self.logger.info("Loading persona profile from checkpoint")
                stage_out.write_persona_profile(payload["profile"], payload["stats"])
                return
            if not allow_compute:
                if self.logger:
                    self.logger.info("Persona profile has no checkpoint for this pass; skipped")
                return
            recording_id = os.path.splitext(os.path.basename(audio_path))[0]
            result = svc.profile(prepared, recording_id)
            if result.skipped:
                if self.logger:
                    self.logger.info(f"Persona profile skipped: {result.skipped}")
                return
            # An unanswered LLM window must be retried next time, not remembered.
            if result.complete:
                checkpoint.save("persona_profile",
                                {"profile": result.profile, "stats": result.stats})
            stage_out.write_persona_profile(result.profile, result.stats)
        except Exception as exc:
            if self.logger:
                self.logger.error(f"[persona] profile failed, later stages go on: {exc}")
```

- [ ] **Step 4: Chạy, thấy PASS** — `python -m pytest tests/test_persona_profile_pipeline.py tests/test_review_select_align_order.py -q`.

- [ ] **Step 5: Commit**

```bash
git add services/pipeline_service.py tests/test_persona_profile_pipeline.py
git commit -m "feat(persona): compute, checkpoint and write the profile from the pipeline

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 2.5: Khối trong `run()` — dừng, allow_compute, giải phóng LLM

**Files:**
- Modify: `services/pipeline_service.py` (`run()`: chèn sau khối stop của `transcript_review`, trước "7c. Forced alignment"; điều kiện gỡ LLM cuối `run()`)
- Test: `tests/test_persona_profile_pipeline.py` (nối)

- [ ] **Step 1: Viết test fail** (nối vào `tests/test_persona_profile_pipeline.py`; theo mẫu đọc nguồn của `tests/test_review_select_align_order.py`)

```python
def _run_body():
    with open(os.path.join(ROOT, "services", "pipeline_service.py"), encoding="utf-8") as fh:
        src = fh.read()
    return src[src.index("    def run(self, args"):]


def test_run_order_review_then_persona_then_alignment():
    body = _run_body()
    review = body.index('opt_in_step_enabled(args, "transcript_review")')
    persona = body.index('opt_in_step_enabled(args, "persona_profile")')
    call = body.index("self._profile_personas(")
    stop = body.index('getattr(args, "stop_after", None) == "persona_profile"')
    align = body.index("# 7c. Forced alignment")
    selection = body.index("Conversation judging is its own LLM stage")
    assert review < persona < call < stop < align < selection


def test_stop_point_writes_the_manifest_and_releases_the_llm():
    body = _run_body()
    stop = body.index('getattr(args, "stop_after", None) == "persona_profile"')
    block = body[stop:stop + 700]
    assert '_release_llm_after(args, "persona_profile")' in block
    assert '"stopped_after": "persona_profile"' in block
    assert "return transcripts" in block


def test_only_the_persona_pass_or_a_single_file_run_computes():
    body = _run_body()
    call = body.index("self._profile_personas(")
    window = body[call - 700:call + 300]
    assert 'getattr(args, "postprocess_only", False)' in window
    assert 'getattr(args, "batch_continues", None) is None' in window
    assert 'getattr(args, "stop_after", None) == "persona_profile"' in window
    assert "allow_compute=allow_compute" in window


def test_the_final_llm_release_counts_the_persona_stage():
    body = _run_body()
    tail = body[body.rindex("review_on):"):]
    assert "persona_on" in tail[:200]


def test_main_accepts_persona_profile_as_a_stop_point():
    with open(os.path.join(ROOT, "main.py"), encoding="utf-8") as fh:
        assert '"persona_profile"' in fh.read()
```

(Điều kiện gỡ LLM cuối `run()` nằm gần dòng 1917: kiểm tra cụm `or review_on` thực tế và chỉnh `rindex` cho khớp.)

- [ ] **Step 2: Chạy, thấy FAIL** — `python -m pytest tests/test_persona_profile_pipeline.py -q -k "run_order or stop_point or computes or final_llm"` -> `ValueError: substring not found`.

- [ ] **Step 3: Code** — chèn ngay sau khối `if getattr(args, "stop_after", None) == "transcript_review": ...return transcripts` và TRƯỚC comment `# 7c. Forced alignment`:

```python
        # 7b''. Persona profile: per-speaker habits of the whole recording. Reads the
        # transcript (on copies) and writes two files; it changes no data. Opt-in.
        persona_on = (self.persona_svc is not None
                      and opt_in_step_enabled(args, "persona_profile"))
        if persona_on and transcripts is not None:
            # postprocess_only is assigned further down, so read it from args. A batch
            # pass computes only in its own persona_profile pass; later passes reload.
            allow_compute = (not bool(getattr(args, "postprocess_only", False))
                             and (getattr(args, "batch_continues", None) is None
                                  or getattr(args, "stop_after", None) == "persona_profile"))
            self._profile_personas(args, checkpoint, stage_out, transcripts, audio_path,
                                   allow_compute=allow_compute)

        if getattr(args, "stop_after", None) == "persona_profile":
            self._release_llm_after(args, "persona_profile")
            if self.logger:
                self.logger.info(
                    "Stopping pipeline after persona_profile as requested by --stop_after.")
            stage_out.write_manifest({"audio_file": os.path.basename(audio_path),
                                      "stopped_after": "persona_profile"})
            return transcripts
```

Và ở điều kiện gỡ LLM cuối `run()`: `if (self.step_enabled(args, "refinement") or relabel_on or exports_planned or review_on or persona_on):`.

- [ ] **Step 4: Chạy, thấy PASS** — `python -m pytest tests/test_persona_profile_pipeline.py tests/test_review_select_align_order.py tests/test_batch.py -q`.

- [ ] **Step 5: Commit**

```bash
git add services/pipeline_service.py tests/test_persona_profile_pipeline.py
git commit -m "feat(persona): run the persona stage after transcript review, with a stop point

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 2.6: `main.py` + `config.json`

**Files:**
- Modify: `main.py` (`--stop_after` choices ~142-144; dựng `persona_svc` sau khối `review_svc` ~911-922; truyền `persona_svc=persona_svc` vào `PipelineService(...)` ~946-964), `config.json` (`environments.a100.steps.persona_profile`, `environments.a100.models.persona_profile`)
- Test: `tests/test_persona_profile_pipeline.py` (nối), `tests/test_stop_after_config.py` (đã có test choices so với config; chỉ chạy lại)

Quyết định: bật `persona_profile: true` ở khối `a100` (nơi `transcript_review: true`) và để `false` nếu muốn mặc định tắt? Spec A5 nói thêm `steps.persona_profile` vào khối cần bật; mặc định stage tắt. Plan chọn: thêm `"persona_profile": false` vào `a100.steps` (hiện diện nhưng tắt, bật bằng `--steps persona_profile=on`), không thêm vào `kaggle` (C7).

- [ ] **Step 1: Viết test fail**

```python
def _main_src():
    with open(os.path.join(ROOT, "main.py"), encoding="utf-8") as fh:
        return fh.read()


def test_main_builds_the_persona_service_only_when_the_step_is_on():
    src = _main_src()
    gate = src.index('opt_in_step_enabled(args, "persona_profile")')
    assert "PersonaProfileService(" in src[gate:gate + 700]
    assert "refinement_svc, logger=logger" in src[gate:gate + 700]
    assert "persona_svc = None" in src[gate - 200:gate]
    assert "persona_svc=persona_svc" in src
    assert src.index("persona_svc = None") > src.index("review_svc = TranscriptReviewService(")


def test_persona_config_has_exactly_the_three_keys():
    import json
    with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    a100 = cfg["environments"]["a100"]
    assert a100["steps"]["persona_profile"] is False
    assert set(a100["models"]["persona_profile"]) == {"llm_enabled", "thinking", "max_new_tokens"}
    for env in cfg["environments"].values():
        persona = env.get("models", {}).get("persona_profile")
        if persona is not None:
            assert set(persona) <= {"llm_enabled", "thinking", "max_new_tokens"}
```

- [ ] **Step 2: Chạy, thấy FAIL** — `python -m pytest tests/test_persona_profile_pipeline.py -q -k "main_builds or persona_config"` -> `ValueError`/`KeyError`.

- [ ] **Step 3: Code**

`main.py`: thêm `"persona_profile"` vào `choices` của `--stop_after` (giữa `"transcript_review"` và `"word_alignment"`). Sau khối `review_svc`:

```python
        # Persona profile: per-speaker habits; same resident LLM. Built only when the
        # step is on. Its three settings come from models.persona_profile alone, and an
        # unknown key raises like relabel's.
        persona_svc = None
        if opt_in_step_enabled(args, "persona_profile"):
            from services.persona_profile_service import PersonaProfileService
            persona_cfg = dict(env_profile.get("models", {}).get("persona_profile", {}))
            persona_svc = PersonaProfileService(refinement_svc, logger=logger, **persona_cfg)
```

Truyền `persona_svc=persona_svc,` vào `PipelineService(...)` ngay sau `review_svc=review_svc,`.

`config.json`: trong `environments.a100.steps` thêm `"persona_profile": false` (sau `"transcript_review": true`, nhớ dấu phẩy); trong `environments.a100.models`, ngay sau khối `"transcript_review": {...}` thêm

```json
        "persona_profile": {
          "llm_enabled": true,
          "thinking": false,
          "max_new_tokens": 1024
        },
```

- [ ] **Step 4: Chạy, thấy PASS**

```bash
python -m pytest tests/test_persona_profile_pipeline.py tests/test_stop_after_config.py tests/test_steps.py tests/test_main_discard.py -q
python -c "import json; json.load(open('config.json'))"
```

- [ ] **Step 5: Commit**

```bash
git add main.py config.json tests/test_persona_profile_pipeline.py
git commit -m "feat(persona): build the persona service in main and add its config

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Cuối Track 2: chạy test

```bash
python -m pytest tests/test_persona_profile_pipeline.py tests/test_stage_output.py tests/test_batch.py tests/test_review_select_align_order.py tests/test_stop_after_config.py -q
python -m pytest tests -q
rtk proxy python -m pytest tests -q -p no:cacheprovider -rf 2>&1 | grep -E "^FAILED" | sed 's/ - .*//' | sort > "$TMPDIR/after_track2.txt"
diff "$TMPDIR"/baseline_failed_*.txt "$TMPDIR/after_track2.txt"
```

Expected: các file trên pass; toàn bộ chỉ còn các fail của baseline, không dòng mới (test lõi của Track 1 có thể chưa xong lúc đó; không phải lỗi của Track 2).

---

# TÍCH HỢP CUỐI (người điều phối làm, sau khi hai track xong)

- [ ] **I1. Pull trạng thái:** `git status` và `git log --oneline -20` để chắc mọi commit của hai track đã vào; không còn file chưa commit thuộc hai track.
- [ ] **I2. Chạy toàn bộ:** `python -m pytest tests -q` rồi so danh sách fail với baseline (62 fail sẵn). Mong đợi: 1617 + (số test mới của hai track) pass, đúng 62 fail cũ, 1 skip.
- [ ] **I3. Kiểm Track 2 gọi đúng interface THẬT của Track 1.** Chạy script này (không ghi file):

```bash
python - <<'EOF'
import inspect, types
from services.persona_profile_service import PersonaProfileService, ProfileResult, PARAMS_VERSION
from services.pipeline_service import PipelineService

sig = inspect.signature(PersonaProfileService.__init__)
assert list(sig.parameters) == ["self", "llm", "logger", "llm_enabled", "thinking", "max_new_tokens"], sig
for name in ("prepare", "source_digest", "checkpoint_namespace_for", "profile"):
    assert callable(getattr(PersonaProfileService, name)), name
assert set(ProfileResult.__dataclass_fields__) == {"profile", "stats", "complete", "skipped"}

# Real service through the real _profile_personas, with a fake checkpoint and the real writer.
import tempfile
from services.stage_output_service import StageOutputService
from schemas.transcript import TranscriptSegment

class LLM:
    model_name = "fake"; batch_size = 8; max_batch_tokens = 0
    def ensure_loaded(self): return True
    def count_tokens(self, t): return len(t.split())
    def generate_texts(self, s, m, n=512, use_prefix=False, labels=None, **kw):
        return True, ['{"new_forms": [], "description": "ok"}'] * len(m)

class Ckpt:
    def __init__(self): self.namespaces, self.store = {}, {}
    def load_if_committed(self, n, **kw): return self.store.get((n, self.namespaces.get(n)))
    def save(self, n, p, **kw): self.store[(n, self.namespaces.get(n))] = p

segs = [TranscriptSegment(index=f"{i:05d}", start=i*5.0, end=i*5.0+4.0, speaker="AB"[i % 2],
        text="Tôi đi chợ nhé.", text_whisper="", text_phowhisper="", text_qwen3="", language="vi",
        bs_roformer=False, bss=False) for i in range(8)]
pipe = PipelineService(*(None,) * 8, persona_svc=PersonaProfileService(LLM()))
pipe.timeline = []
out_dir = tempfile.mkdtemp()
ck = Ckpt()
pipe._profile_personas(types.SimpleNamespace(), ck, StageOutputService(out_dir), segs, "/x/ep01.mp3")
import os, json
d = os.path.join(out_dir, "07c_persona_profile")
assert sorted(os.listdir(d)) == ["profile.json", "stats.json"]
prof = json.load(open(os.path.join(d, "profile.json"), encoding="utf-8"))
assert prof["recording_id"] == "ep01" and prof["params_version"] == PARAMS_VERSION
assert len(ck.store) == 1
print("OK", out_dir)
EOF
```

Expected: in `OK <thư mục tạm>`. Fail nghĩa là Track 2 và Track 1 lệch hợp đồng: sửa phía lệch hợp đồng (hợp đồng ở đầu kế hoạch là chuẩn).

- [ ] **I4. Chạy lại lần hai cùng input và so byte** `profile.json` (xác định); thêm kiểm `allow_compute=False` với checkpoint sẵn có không gọi LLM.
- [ ] **I5. Kiểm hai tập file không giao nhau:** `git diff --stat cbce204..HEAD` chỉ liệt kê các file ở hai cột bảng "Phân chia file", mỗi file một track.
- [ ] **I6. Review cuối** (reviewer): đối chiếu spec Phần B (công thức B4.1 đến B4.5, ngưỡng B5, cảnh báo B6) với code lõi; kiểm Review Focus 5 mục có test ghim.

# Việc sau (NGOÀI kế hoạch này)

- C8: đo thời gian `analyze()` O(n²) trên file thật (vd `vimeanhphanchiatay-145413`, 28 phút); chỉ thêm giới hạn cửa sổ nếu đo cần, khi đó tăng `PARAMS_VERSION`.
- C3: sửa `_fold` dùng chung sang NFC là commit `fix(...)` riêng; khi làm phải tăng `PARAMS_VERSION`.
- Bật `persona_profile` trong khối `kaggle` (C7), script gom persona offline, mở rộng từ điển từ `llm_candidate_forms`.

# Self-Review (đã chạy khi viết kế hoạch)

- Spec coverage: A2/A5 (vị trí, `_STOP_ORDER`, `_next_llm_step_after`, `_release_llm_after`, `parallel_stage_view`, cuối `run()`) -> T2.3, T2.5; A3 P0-P5 -> T1.3-T1.7, T2.4; A4 (Qwen, retry, xác nhận) -> T1.7; A6 (namespace, không lưu khi chưa trọn, idempotent) -> T1.7, T1.8, T2.4; A7 (bảng lỗi) -> T1.6-T1.8, T2.4, T2.5; B1-B2 -> T1.1, T1.2, T2.1; B3-B4 -> T1.4-T1.6; B5 -> T1.2; B6 -> T1.6, T1.7; B7 -> T1.1; B8 #1-#16 -> T1.1-T1.8 (#2 NaN: T1.1 + T2.1; #15: T1.8 + T2.4), #17 -> T2.3/T2.5, #18 -> T2.2. `config.json`, `main.py` -> T2.6. C8 là việc sau.
- Placeholder: không có TODO/TBD; kỳ vọng số học tay trong test tương tác của T1.5 có quy tắc "spec thắng" ở mục Quy tắc làm việc chung.
- Kiểu/tên nhất quán: `prepare`, `source_digest`, `checkpoint_namespace_for`, `profile`, `ProfileResult(profile, stats, complete, skipped)`, `write_persona_profile(profile, stats)`, `_profile_personas(..., allow_compute=)` dùng đồng nhất ở hợp đồng, T1.7, T2.4, T2.5 và I3.
