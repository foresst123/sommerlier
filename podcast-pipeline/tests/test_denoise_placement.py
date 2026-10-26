"""The three noise-filter points: pre_diarization (P1), pre_separation (P2), pre_asr (P3)."""

import json
import os
import re
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from schemas.segment import SpeechSegment
from services import denoise_service
from services.denoise_service import DenoiseService
from services.pipeline_service import PipelineService
from services.stage_output_service import StageOutputService
from utils.checkpoint import CheckpointManager
# the stand-in models.denoise (autouse) and the small helpers of the P1 tests
from test_denoise_stage import FakeBackend, SR, ROOT, args, audio, fake_api, speech  # noqa: F401


def pipe(dropped=None):
    sep = SimpleNamespace(drop_prefetched_plan=lambda path: dropped.append(path))
    svc = PipelineService(*(None,) * 8)
    svc.separation_svc = sep if dropped is not None else None
    return svc


def ck_for(tmp_path, ns="v1-abc"):
    ck = CheckpointManager(str(tmp_path / "cache"), "job")
    ck.namespaces["processed_audio"] = ns
    return ck


def run_p2(tmp_path, spec, p1="none", data=None, dropped=None, computed=None, ck=None):
    ck = ck or ck_for(tmp_path)
    if p1 != "none":
        ck.namespaces["denoised_audio"] = f"{p1}-v1-abc"
    data = data or audio()
    computed = set() if computed is None else computed
    pipe(dropped)._apply_denoise_pre_separation(
        args(denoise=p1, denoise_pre_separation=spec), {}, ck, data,
        StageOutputService(str(tmp_path / "out")), computed, "/a/x.wav")
    return ck, data, computed


def segments(n_audio=2):
    """Two separated cores (audio set) among plain segments, cores as in the separation checkpoint."""
    segs = []
    for i in range(4):
        core = speech(1.0, seed=i + 1) if i < n_audio else None
        segs.append(SpeechSegment(index=f"{i:05d}", start=float(i) * 2, end=float(i) * 2 + 1.0,
                                  speaker="S0", audio=core))
    return segs


def run_p3(tmp_path, spec, segs, data=None, p2="none", p1="none", computed=None, ck=None):
    ck = ck or ck_for(tmp_path)
    ck.namespaces["separation"] = "win1"
    if p1 != "none":   # what the earlier points leave behind in a real run
        ck.namespaces["denoised_audio"] = f"{p1}-v1-abc"
    if p2 != "none":
        ck.namespaces["denoised_audio_pre_separation"] = (
            f"{p2}-{ck.namespaces.get('denoised_audio', 'v1-abc')}")
    data = data or audio()
    computed = set() if computed is None else computed
    out = pipe()._apply_denoise_pre_asr(
        args(denoise=p1, denoise_pre_separation=p2, denoise_pre_asr=spec), {}, ck, data,
        StageOutputService(str(tmp_path / "out")), computed, segs)
    return ck, data, out, computed


# --- DenoiseService.apply_many ------------------------------------------------

def test_apply_many_loads_each_backend_once(fake_api):
    arrays = [speech(2.0, 0), speech(1.0, 1), speech(1.5, 2)]
    before = [a.copy() for a in arrays]
    outs, info = DenoiseService().apply_many(arrays, SR, "half>plus1", None, None)
    assert len(fake_api) == 2, "one backend object per chain step, not per array"
    assert FakeBackend.calls == ["half"] * 3 + ["plus1"] * 3
    assert all(b.unloaded for b in fake_api)
    for o, b in zip(outs, before):
        assert np.allclose(o, b * 0.5 + 0.01, atol=1e-6)
    assert all(np.array_equal(a, b) for a, b in zip(arrays, before)), "inputs are not edited"
    assert info["errors"] == [] and info["filtered"] == 3 and info["arrays"] == 3
    assert info["chain_id"] == "half+plus1" and info["lag_samples"] == 0


def test_apply_many_failure_keeps_every_input(fake_api, monkeypatch):
    class Late(FakeBackend):
        def enhance(self, wav, sr):
            if len(FakeBackend.calls) >= 1:
                raise RuntimeError("out of memory")
            FakeBackend.calls.append("late")
            return (wav * 0.5).astype(np.float32)

    monkeypatch.setattr(denoise_service, "_load_api", lambda: (
        lambda name, cfg=None, logger=None: Late(name, "late"),
        lambda spec: [str(spec)], lambda spec: str(spec)))
    arrays = [speech(1.0, 0), speech(1.0, 1)]
    before = [a.copy() for a in arrays]
    outs, info = DenoiseService().apply_many(arrays, SR, "late", None, None)
    assert all(np.array_equal(o, b) for o, b in zip(outs, before)), "the first array must not stay filtered"
    assert "out of memory" in info["errors"][0] and info["filtered"] == 0


def test_apply_many_bad_input_never_raises():
    outs, info = DenoiseService().apply_many([speech(1.0)], SR, "nonesuch", None, None)
    assert info["errors"] and len(outs) == 1
    outs, info = DenoiseService().apply_many([speech(1.0)], SR, "short", None, None)
    assert "samples" in info["errors"][0]
    outs, info = DenoiseService().apply_many([], SR, "half", None, None)
    assert outs == []


def test_apply_many_skips_arrays_too_short_for_a_backend(fake_api):
    tiny = speech(0.1, 3)
    outs, info = DenoiseService().apply_many([speech(1.0, 0), tiny], SR, "half", None, None)
    assert np.array_equal(outs[1], tiny) and info["skipped_short"] == 1
    assert info["errors"] == [] and info["filtered"] == 1 and FakeBackend.calls == ["half"]
    outs, info = DenoiseService().apply_many([tiny], SR, "half", None, None)
    assert FakeBackend.calls == ["half"], "nothing long enough: the backend is not even called"


# --- P2: after diarization, before separation ----------------------------------

def test_p2_filters_the_waveform_and_writes_its_stage(tmp_path):
    dropped = []
    ck, data, computed = run_p2(tmp_path, "half", dropped=dropped)
    assert np.allclose(data.waveform, speech() * 0.5, atol=1e-6)
    assert len(data.waveform) == len(speech()), "sample aligned"
    assert ck.namespaces["denoised_audio_pre_separation"] == "half-v1-abc"
    stage = tmp_path / "out" / "02b_denoise_pre_separation"
    info = json.loads((stage / "denoise.json").read_text())
    assert info["point"] == "pre_separation" and info["chain"] == ["half"]
    assert (stage / "after_denoise.wav").is_file() and "denoise_pre_separation" in computed
    assert not (tmp_path / "out" / "01b_denoise").exists()
    assert dropped == ["/a/x.wav"], "a window plan prefetched from the unfiltered audio is discarded"


def test_p2_namespace_includes_the_p1_chain(tmp_path):
    ck, data, _ = run_p2(tmp_path, "half", p1="plus1")
    assert ck.namespaces["denoised_audio_pre_separation"] == "half-plus1-v1-abc"


def test_p2_second_entry_reuses_the_cache_and_still_filters(tmp_path):
    run_p2(tmp_path, "half")
    assert FakeBackend.calls == ["half"]
    dropped = []
    ck, data, computed = run_p2(tmp_path, "half", data=audio(), dropped=dropped)
    assert FakeBackend.calls == ["half"], "backend ran again on a cache hit"
    assert np.allclose(data.waveform, speech() * 0.5, atol=1e-6)
    assert "denoise_pre_separation" not in computed and dropped == ["/a/x.wav"]


def test_p2_failure_falls_back_records_the_error_and_does_not_retry(tmp_path):
    ck, data, _ = run_p2(tmp_path, "boom")
    assert np.array_equal(data.waveform, speech())
    info = json.loads((tmp_path / "out" / "02b_denoise_pre_separation" / "denoise.json").read_text())
    assert "out of memory" in info["errors"][0]
    run_p2(tmp_path, "boom")
    assert FakeBackend.calls == ["boom"]


def test_p2_none_touches_nothing(tmp_path):
    for spec in ("none", "", None):
        dropped = []
        data = audio()
        before = data.waveform
        ck, data, computed = run_p2(tmp_path, spec, data=data, dropped=dropped)
        assert data.waveform is before and not computed and not dropped
    assert os.listdir(tmp_path / "cache" / "job") == [] and not (tmp_path / "out").exists()


# --- P3: after separation, before ASR ------------------------------------------

def test_p3_filters_waveform_and_cores_without_touching_the_checkpoint_objects(tmp_path):
    segs = segments()
    cores = [s.audio.copy() if s.audio is not None else None for s in segs]
    originals = list(segs)
    ck, data, out, computed = run_p3(tmp_path, "half", segs)
    assert segs == originals and all(a is b for a, b in zip(segs, originals))
    for s, c in zip(segs, cores):  # the input objects keep their arrays
        assert (s.audio is None) if c is None else np.array_equal(s.audio, c)
    assert np.allclose(data.waveform, speech() * 0.5, atol=1e-6)
    for new, old, c in zip(out, segs, cores):
        assert (new.start, new.end, new.index, new.speaker) == (old.start, old.end, old.index, old.speaker)
        if c is None:
            assert new.audio is None
        else:
            assert new is not old and len(new.audio) == len(c)
            assert np.allclose(new.audio, c * 0.5, atol=1e-6)
    stage = tmp_path / "out" / "04b_denoise_pre_asr"
    info = json.loads((stage / "denoise.json").read_text())
    assert info["point"] == "pre_asr" and info["segments_with_audio"] == 2
    assert info["arrays"] == 3 and info["filtered"] == 3 and info["skipped_short"] == 0
    assert sorted(os.listdir(stage)) == ["after_denoise.wav", "denoise.json"], "no per-segment wavs"
    assert FakeBackend.calls == ["half"] * 3


def test_p3_second_entry_reuses_the_cache_and_still_filters(tmp_path):
    segs = segments()
    run_p3(tmp_path, "half", segs)
    ck, data, out, computed = run_p3(tmp_path, "half", segs, data=audio())
    assert FakeBackend.calls == ["half"] * 3, "no second run on a cache hit"
    assert "denoise_pre_asr" not in computed
    assert np.allclose(data.waveform, speech() * 0.5, atol=1e-6)
    assert np.allclose(out[0].audio, segs[0].audio * 0.5, atol=1e-6)
    assert out[2].audio is None


def test_p3_cache_follows_the_chain_and_what_came_before(tmp_path):
    segs = segments()
    run_p3(tmp_path, "half", segs)
    run_p3(tmp_path, "half", segs, p2="plus1")
    run_p3(tmp_path, "plus1", segs)
    assert FakeBackend.calls == ["half"] * 3 + ["half"] * 3 + ["plus1"] * 3
    ck, *_ = run_p3(tmp_path, "half", segs, p1="plus1", p2="identity")
    assert ck.namespaces["denoised_audio_pre_asr"] == "half-identity-plus1-v1-abc-win1"


def test_p3_failure_keeps_everything_and_is_cached(tmp_path):
    segs = segments()
    ck, data, out, _ = run_p3(tmp_path, "boom", segs)
    assert np.array_equal(data.waveform, speech())
    assert all(np.array_equal(a.audio, b.audio) for a, b in zip(out, segs) if a.audio is not None)
    info = json.loads((tmp_path / "out" / "04b_denoise_pre_asr" / "denoise.json").read_text())
    assert info["errors"] and info["filtered"] == 0
    run_p3(tmp_path, "boom", segs)
    assert FakeBackend.calls == ["boom"], "a failing chain must not retry every pass"


def test_p3_none_or_no_segments_is_a_noop(tmp_path):
    segs = segments()
    for spec in ("none", "", None):
        ck, data, out, computed = run_p3(tmp_path, spec, segs)
        assert out is segs and not computed
    ck, data, out, computed = run_p3(tmp_path, "half", None)
    assert out is None and not computed
    assert os.listdir(tmp_path / "cache" / "job") == [] and not FakeBackend.calls


def test_p3_stale_segment_cache_is_recomputed(tmp_path):
    run_p3(tmp_path, "half", segments(2))
    run_p3(tmp_path, "half", segments(3))     # same chain and namespace, different separation result
    assert FakeBackend.calls == ["half"] * 3 + ["half"] * 4


# --- wiring in run() and the rest ---------------------------------------------

def _run_source():
    src = open(os.path.join(ROOT, "services", "pipeline_service.py"), encoding="utf-8").read()
    return src[src.index("    def run(self, args"):]


def test_run_applies_p2_and_p3_on_every_entry_in_the_right_place():
    src = _run_source()
    p2, p3 = src.index("_apply_denoise_pre_separation("), src.index("_apply_denoise_pre_asr(")
    dia_stop = src.index('Stopping pipeline after diarization as requested by --stop_after.")')
    separation = src.index("# 4. Tách giọng")
    asr_block = src.index("# 5. Nhận dạng lời nói")
    assert dia_stop < p2 < separation, "after the diarization stop, before separation"
    assert src.index('"stopped_after": "music_removal"') < p3 < asr_block
    assert "speech_segments = self._apply_denoise_pre_asr(" in src
    assert p3 < src.index("self._submit_asr_async(", asr_block), "the async ASR path gets the P3 segments"
    prefetch = src[src.index("self.separation_svc.prefetch_overlap_plan(") - 300:][:300]
    assert "denoise_pre_separation" in prefetch, "no prefetch from the unfiltered waveform"


def test_stage_dirs_sort_between_their_neighbours():
    st = StageOutputService.STAGES
    order = ["denoise", "diarization", "denoise_pre_separation", "separation", "music_removal",
             "denoise_pre_asr", "asr"]
    names = [st[k] for k in order]
    assert st["music"] < st["denoise"] < st["diarization"]
    assert names[1:] == sorted(names[1:])


def test_stop_order_has_the_new_points():
    order = PipelineService._STOP_ORDER
    assert order.index("diarization") < order.index("denoise_pre_separation") < order.index("separation")
    assert order.index("music_removal") < order.index("denoise_pre_asr") < order.index("asr")


def test_p1_json_carries_its_point(tmp_path):
    pipe()._apply_denoise(args(denoise="half"), {}, ck_for(tmp_path), audio(),
                          StageOutputService(str(tmp_path / "out")), set())
    info = json.loads((tmp_path / "out" / "01b_denoise" / "denoise.json").read_text())
    assert info["point"] == "pre_diarization"


def test_profiles_carry_the_new_keys():
    with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    for name, profile in cfg["environments"].items():
        if "pipeline" in profile:
            assert profile["pipeline"].get("denoise_pre_separation") == "none", name
            assert profile["pipeline"].get("denoise_pre_asr") == "none", name


def _parse_flags(argv):
    probe = (
        "import sys, json\n"
        "sys.argv = ['main.py'] + " + repr(argv) + "\n"
        "src = open('main.py').read().split('# 2. DELAYED IMPORTS')[0]\n"
        "ns = {'__name__': 'probe', '__file__': 'main.py'}\n"
        "exec(compile(src, 'main.py', 'exec'), ns)\n"
        "a = ns['args']\n"
        "print(json.dumps([a.denoise, a.denoise_pre_separation, a.denoise_pre_asr, a.stop_after]))\n"
    )
    path = os.path.join(ROOT, "_probe_placement_args.py")
    with open(path, "w") as fh:
        fh.write(probe)
    try:
        env = dict(os.environ, SOMMELIER_SKIP_BOOTSTRAP="1")
        return subprocess.run([sys.executable, path], cwd=ROOT, env=env, capture_output=True, text=True)
    finally:
        os.unlink(path)


BASE = ["--audio", "x.wav", "--env", "kaggle"]


def test_flags_default_to_none_and_accept_values():
    out = _parse_flags(BASE)
    assert out.returncode == 0, out.stderr[-1500:]
    assert json.loads(out.stdout.strip().splitlines()[-1]) == ["none", "none", "none", None]
    out = _parse_flags(BASE + ["--denoise_pre_separation", "identity", "--denoise_pre_asr", "identity",
                               "--stop_after", "denoise_pre_asr"])
    if out.returncode == 0:   # validated through models.denoise when it is importable
        assert json.loads(out.stdout.strip().splitlines()[-1]) == ["none", "identity", "identity", "denoise_pre_asr"]
    else:
        assert "--denoise_pre" in out.stderr


def test_a_bad_chain_at_a_new_point_stops_the_run_early():
    out = _parse_flags(BASE + ["--denoise_pre_asr", "a>>b"])
    assert out.returncode != 0 and "--denoise_pre_asr" in out.stderr
