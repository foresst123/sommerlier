"""Which GPU each ASR model sits on, and the cross-file scheduling knobs."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import performance_config as pc


def _profile(asr=None):
    return {"performance": {"enabled": True, "stages": {"asr": asr or {}}}}


def _asr(asr=None):
    return pc.resolve(_profile(asr))["stages"]["asr"]


def test_defaults_reproduce_the_legacy_layout_and_leave_cross_file_off():
    asr = _asr()
    assert asr["cross_file"] is False
    assert pc.resolve_asr_placement(asr, 0, 1) == {"qwen3": 1, "whisper": 1, "phowhisper": 0}


def test_the_a100_layout_puts_qwen_alone_and_the_whisper_pair_together():
    asr = _asr({"qwen3_gpu": "gpu_1", "whisper_gpu": "gpu_2", "phowhisper_gpu": "gpu_2"})
    assert pc.resolve_asr_placement(asr, 0, 1) == {"qwen3": 0, "whisper": 1, "phowhisper": 1}


def test_placement_uses_the_configured_gpu_ids():
    asr = _asr({"qwen3_gpu": "gpu_1", "whisper_gpu": "gpu_2", "phowhisper_gpu": "gpu_2"})
    assert pc.resolve_asr_placement(asr, 2, 3) == {"qwen3": 2, "whisper": 3, "phowhisper": 3}


def test_an_unknown_gpu_name_falls_back_and_is_reported():
    result = pc.resolve(_profile({"qwen3_gpu": "gpu_9"}))
    assert result["stages"]["asr"]["qwen3_gpu"] == "gpu_2"
    assert any("qwen3_gpu" in problem for problem in result["_problems"])


def test_the_scheduling_knobs_have_defaults_and_bounds():
    asr = _asr()
    assert (asr["files_in_flight"], asr["shared_batch_size"], asr["boost_batch_size"]) == (3, 16, 48)
    result = pc.resolve(_profile({"files_in_flight": 0}))
    assert result["stages"]["asr"]["files_in_flight"] == 1
    assert any("files_in_flight" in problem for problem in result["_problems"])


def test_the_new_keys_do_not_change_the_output_fingerprint():
    plain = pc.fingerprint(pc.resolve(_profile()))
    tuned = pc.fingerprint(pc.resolve(_profile({
        "cross_file": True, "files_in_flight": 5, "qwen3_gpu": "gpu_1",
        "phowhisper_gpu": "gpu_2", "boost_batch_size": 64})))
    assert plain == tuned


def test_phowhisper_runs_in_process_unless_workers_are_asked_for():
    assert _asr()["phowhisper_workers"] == 0
    assert _asr({"phowhisper_workers": 3})["phowhisper_workers"] == 3
    too_many = pc.resolve(_profile({"phowhisper_workers": 99}))
    assert too_many["stages"]["asr"]["phowhisper_workers"] == 4      # clamped, and reported
    assert any("phowhisper_workers" in problem for problem in too_many["_problems"])


def test_pho_workers_start_on_its_own_gpu_and_alternate_across_the_cards():
    assert pc.pho_worker_devices(1, [0, 1], 1) == [1]
    assert pc.pho_worker_devices(1, [0, 1], 2) == [1, 0]
    assert pc.pho_worker_devices(1, [0, 1], 4) == [1, 0, 1, 0]
    assert pc.pho_worker_devices(0, [0, 1], 3) == [0, 1, 0]


def test_pho_workers_use_only_the_cards_that_exist():
    assert pc.pho_worker_devices(1, [1], 2) == [1, 1]
    assert pc.pho_worker_devices(5, [0, 1], 2) == [0, 1]     # placement not available
