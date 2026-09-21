"""`generate_texts` is the model-facing half of refinement, shared with the passes
that reuse the same resident LLM (speaker relabel, conversation-export judging).

It was split out of `_refine_batch`, so what is pinned here is that the split
changed nothing for fusion -- same acceptance, same OOM and fallback behaviour --
and that the new callers get left padding without leaking it.

Run:  python -m pytest tests/test_refinement_generate.py -q     (from podcast-pipeline/)
"""
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

try:
    import services.diarization_refinement_service as refinement
except ImportError as e:  # pragma: no cover - depends on the local install
    refinement = None
    _missing = str(e)

pytestmark = pytest.mark.skipif(
    refinement is None,
    reason="refinement service needs librosa/soundfile via algorithms.asr.rover")

from schemas.transcript import TranscriptSegment


class _Batch:
    """What a tokenizer returns: attribute access, .to(), and ** unpacking."""

    def __init__(self, ids, mask):
        self.input_ids = ids
        self.attention_mask = mask

    def to(self, _device):
        return self

    def keys(self):
        return ("input_ids", "attention_mask")

    def __getitem__(self, key):
        return getattr(self, key)


class _Tok:
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"
    eos_token_id = 1

    def __init__(self):
        self.padding_side = "right"
        self.padding_seen = []
        self.answers = []
        self.thinking_seen = []

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False,
                            enable_thinking=False):
        self.thinking_seen.append(enable_thinking)
        return msgs[-1]["content"]

    def __call__(self, texts, return_tensors=None, padding=False, **_):
        if isinstance(texts, str):
            # A real tokenizer given one string returns a plain list of ids.
            return SimpleNamespace(input_ids=list(range(len(texts.split()))))
        self.padding_seen.append(self.padding_side)
        width = max(len(t.split()) for t in texts)
        ids = torch.ones(len(texts), width, dtype=torch.long)
        return _Batch(ids, torch.ones_like(ids))

    def batch_decode(self, rows, skip_special_tokens=True):
        return [self.answers[int(row[0])] for row in rows]


class _Model:
    """generate() appends one token per row that indexes the tokenizer's answers."""

    def __init__(self, fail=None):
        self.calls = 0
        self.fail = fail

    def generate(self, input_ids, attention_mask, **_):
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        answer = torch.arange(input_ids.shape[0]).unsqueeze(1)
        return torch.cat([input_ids, answer], dim=1)


def _svc(answers, model=None, **kwargs):
    svc = refinement.DiarizationRefinementService(logger=None, **kwargs)
    svc.tokenizer = _Tok()
    svc.tokenizer.answers = list(answers)
    svc.model = model if model is not None else _Model()
    svc._input_device = lambda: "cpu"
    return svc


def _seg(index, text):
    return TranscriptSegment(
        index=index, start=0.0, end=5.0, speaker="SPEAKER_00",
        text=text, text_whisper=text, text_phowhisper=text, text_qwen3=text,
        language="vi", bs_roformer=False, bss=False)


def test_it_returns_one_decoded_text_per_request():
    svc = _svc(["một", "hai", "ba"])
    ok, decoded = svc.generate_texts("sys", ["a b", "c d", "e f"])
    assert ok and decoded == ["một", "hai", "ba"]


def test_left_padding_is_used_during_the_call_and_put_back_after():
    svc = _svc(["x"])
    svc.tokenizer.padding_side = "right"
    svc.generate_texts("sys", ["a b"])
    assert svc.tokenizer.padding_seen == ["left"]
    assert svc.tokenizer.padding_side == "right"


def test_without_a_loaded_model_it_fails_instead_of_loading_one():
    """Loading belongs to ensure_loaded(); a failed load must not be retried
    on every call."""
    svc = _svc([])
    svc.model = None
    assert svc.generate_texts("sys", ["a b"]) == (False, [])


def test_a_batch_over_max_batch_tokens_is_refused_without_generating():
    svc = _svc(["x", "y"], max_batch_tokens=3)
    ok, decoded = svc.generate_texts("sys", ["a b c", "d e f"])
    assert (ok, decoded) == (False, [])
    assert svc.model.calls == 0


def test_out_of_memory_is_reported_as_not_succeeded():
    svc = _svc(["x"], model=_Model(fail=torch.cuda.OutOfMemoryError("oom")))
    assert svc.generate_texts("sys", ["a b"]) == (False, [])


def test_a_failing_pipeline_scheduler_falls_back_to_serial_generation():
    class _BrokenPool:
        def generate(self, *a, **k):
            raise RuntimeError("scheduler broke")

    svc = _svc(["ok"])
    svc.pipeline_pool = _BrokenPool()
    ok, decoded = svc.generate_texts("sys", ["a b"])
    assert ok and decoded == ["ok"]
    assert svc.pipeline_pool is None
    assert svc.model.calls == 1


def test_the_fusion_prefix_cache_is_not_touched_unless_asked_for():
    svc = _svc(["ok"], prefix_cache=True)

    def _boom(*a, **k):
        raise AssertionError("prefix cache used for a non-fusion prompt")

    svc._build_prefix = _boom
    ok, _ = svc.generate_texts("relabel prompt", ["a b"])
    assert ok


# --- _refine_batch still does what it did ------------------------------------

def test_refine_batch_writes_an_accepted_text_and_counts_it():
    seg = _seg("1", "xin chào các bạn")
    svc = _svc(["xin chào các bạn."])
    ok, count = svc._refine_batch([(seg, "user msg one two")], "sys")
    assert (ok, count) == (True, 1)
    assert seg.text == "xin chào các bạn."


def test_fusion_never_asks_the_model_to_think():
    """One call per segment, a fixed answer shape and a few hundred tokens."""
    svc = _svc(["văn bản mới một hai"])
    svc._refine_batch([(_seg("00001", "văn bản cũ"), "user msg one two")], "sys")
    assert svc.tokenizer.thinking_seen and not any(svc.tokenizer.thinking_seen)


def test_refine_batch_keeps_the_original_text_when_the_output_is_rejected():
    seg = _seg("1", "xin chào các bạn")
    svc = _svc(["hoàn toàn khác biệt nội dung gì đó"])
    ok, count = svc._refine_batch([(seg, "user msg one two")], "sys")
    assert (ok, count) == (True, 0)
    assert seg.text == "xin chào các bạn"
    assert svc.rejected == 1


def test_refine_batch_reports_failure_when_generation_does_not_fit():
    seg = _seg("1", "xin chào")
    svc = _svc(["x"], max_batch_tokens=1)
    assert svc._refine_batch([(seg, "a b c d")], "sys") == (False, 0)
    assert seg.text == "xin chào"


# --- what the other passes need before they can use the model -----------------

def test_ensure_loaded_does_not_reload_a_model_that_is_already_resident():
    svc = _svc([])
    resident, tokenizer = svc.model, svc.tokenizer
    assert svc.ensure_loaded() is True      # the real _load_model, which returns early
    assert svc.model is resident and svc.tokenizer is tokenizer


def test_ensure_loaded_says_false_when_the_model_could_not_be_loaded():
    svc = _svc([])
    svc.model = None
    svc._load_model = lambda: None          # the load ran and produced nothing
    assert svc.ensure_loaded() is False


def test_count_tokens_uses_the_models_own_tokenizer_without_special_tokens():
    svc = _svc([])
    assert svc.count_tokens("một hai ba bốn") == 4


# --- the passes that share this model, on the real service surface -------------
#
# The relabel and clip tests use a stand-in LLM. These run the same passes on the
# real DiarizationRefinementService (only its model and tokenizer are fake), so a
# drift between that stand-in and the real signatures cannot go unnoticed.

def test_the_relabel_pass_runs_on_the_real_service():
    from services.speaker_relabel_service import SpeakerRelabelService, apply_relabels

    def seg(i, speaker):
        s = _seg(f"{i:05d}", f"câu số {i} nói gì đó")
        s.start, s.end, s.speaker, s.gap_before = i * 5.0, i * 5.0 + 4.0, speaker, 0.5
        return s

    segs = [seg(i, "SPEAKER_00" if i % 2 == 0 else "SPEAKER_01") for i in range(10)]
    segs[4].speaker = "SPEAKER_01"                      # the mislabel
    # One window covers the file: segment 4 is line 5, and SPEAKER_00 is told as "A".
    reply = '[{"i": 5, "speaker": "A", "conf": 0.9, "why": "x"}]'
    llm = _svc([reply] * 5, max_batch_tokens=100000)

    result = SpeakerRelabelService(llm, window_tokens=100000).relabel(segs)
    apply_relabels(segs, result.mapping)

    assert result.skipped is None and result.failed_windows == 0
    assert segs[4].speaker == "SPEAKER_00" and segs[4].speaker_original == "SPEAKER_01"


def test_the_clip_judge_runs_on_the_real_service():
    import numpy as np
    from services.conversation_export_service import ConversationExportService
    from utils.excise import TimelineMap
    from utils.noise_map import KINDS, NoiseTrack

    segs = []
    for k in range(30):
        s = _seg(f"{k:05d}", f"đoạn {k} bàn về chuyện số {k} rất dài dòng")
        s.start, s.end = k * 8.0, k * 8.0 + 7.0
        s.speaker = "SPEAKER_00" if k % 2 == 0 else "SPEAKER_01"
        segs.append(s)
    noise = NoiseTrack({k: np.full(60000, 0.001, dtype=np.float32) for k in KINDS}, fps=100.0)
    llm = _svc(['{"self_contained": 5, "topic": "thử"}'] * 10, max_batch_tokens=100000)

    import tempfile
    with tempfile.TemporaryDirectory() as out:
        result = ConversationExportService(llm, max_candidates=3).run(
            segs, timeline=TimelineMap(), noise=noise, music_map=None,
            waveform=np.zeros(300 * 8000, dtype=np.float32), sample_rate=8000,
            out_dir=out, base_name="ep")
    assert result.report["shortlisted"] >= 1 and result.report["unanswered"] == 0
    assert result.exports and result.exports[0]["semantic_score"] == 5


# --- the two-GPU load and what a total failure says ---------------------------

def _load_pipelined(monkeypatch, tie):
    """Run the real _load_model with the model libraries replaced, and return the
    device_map it asked transformers for."""
    import transformers

    captured = {}

    class _Cfg:
        num_hidden_layers = 36
        tie_word_embeddings = tie

    class _FakeModel:
        def eval(self):
            return self

    def _from_pretrained(name, **kwargs):
        captured.update(kwargs)
        return _FakeModel()

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained",
                        staticmethod(lambda name, **kw: _Cfg()))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        staticmethod(lambda name, **kw: object()))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained",
                        staticmethod(_from_pretrained))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda index: SimpleNamespace(total_memory=15 * 2 ** 30))
    import services.refinement_pipeline_pool as pool_module
    monkeypatch.setattr(pool_module.RefinementPipelinePool, "__init__",
                        lambda self, *a, **k: None)

    svc = refinement.DiarizationRefinementService(
        logger=None, placement="pipelined", pipeline_devices=[0, 1])
    svc._activate_cpu_threads = lambda: 0
    svc._load_model()
    return captured["device_map"]


def test_the_two_gpu_load_keeps_a_tied_head_with_its_embedding(monkeypatch):
    device_map = _load_pipelined(monkeypatch, tie=True)
    assert device_map["lm_head"] == device_map["model.embed_tokens"] == 0


def test_the_two_gpu_load_leaves_an_untied_model_split_as_before(monkeypatch):
    device_map = _load_pipelined(monkeypatch, tie=False)
    assert device_map["model.embed_tokens"] == 0 and device_map["lm_head"] == 1


def _refine_everything_failing(error):
    svc = _svc(["x"] * 4, model=_Model(fail=error), batch_size=2)
    segs = [_seg(str(i), "xin chào các bạn") for i in range(4)]
    with pytest.raises(RuntimeError) as caught:
        svc.refine(segs)
    return str(caught.value)


def test_a_total_failure_names_a_device_error_instead_of_blaming_memory():
    message = _refine_everything_failing(RuntimeError(
        "Expected all tensors to be on the same device, but got index is on cuda:0"))
    assert "same device" in message
    assert "not an out-of-memory failure" in message
    assert "too little free VRAM" not in message


def test_a_total_failure_from_memory_still_gives_the_memory_advice():
    message = _refine_everything_failing(torch.cuda.OutOfMemoryError("CUDA out of memory"))
    assert "too little free VRAM" in message
    assert "not an out-of-memory" not in message


def test_thinking_is_off_for_the_model_unless_a_pass_asks_for_it():
    svc = _svc(["một", "hai"])
    svc.generate_texts("sys", ["a", "b"])
    assert svc.tokenizer.thinking_seen == [False, False]


def test_a_pass_that_asks_for_thinking_gets_it_in_the_chat_template():
    svc = _svc(["một", "hai"])
    ok, texts = svc.generate_texts("sys", ["a", "b"], thinking=True)
    assert ok and texts == ["một", "hai"]
    assert svc.tokenizer.thinking_seen == [True, True]


def test_thinking_survives_the_serial_fallback_retry():
    svc = _svc(["một"])
    svc.pipeline_pool = SimpleNamespace(
        generate=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("scheduler broke")))
    ok, texts = svc.generate_texts("sys", ["a"], thinking=True)
    assert ok and svc.pipeline_pool is None
    assert svc.tokenizer.thinking_seen == [True, True]
