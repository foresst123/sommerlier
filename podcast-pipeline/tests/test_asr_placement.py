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


def test_assigning_windows_ahead_is_on_unless_switched_off():
    sep = pc.resolve({"performance": {"enabled": True, "stages": {}}})["stages"]["separation"]
    assert sep["postprocess_ahead"] is True
    off = pc.resolve({"performance": {"enabled": True, "stages": {
        "separation": {"postprocess_ahead": False}}}})["stages"]["separation"]
    assert off["postprocess_ahead"] is False


def test_sidon_lookahead_can_be_deep_enough_to_never_wait_on_what_follows():
    sep = pc.resolve({"performance": {"enabled": True, "stages": {
        "separation": {"gpu_prefetch_per_worker": 16}}}})["stages"]["separation"]
    assert sep["gpu_prefetch_per_worker"] == 16


def test_zero_lookahead_is_accepted_as_unlimited():
    sep = pc.resolve({"performance": {"enabled": True, "stages": {
        "separation": {"gpu_prefetch_per_worker": 0}}}})
    assert sep["stages"]["separation"]["gpu_prefetch_per_worker"] == 0
    assert not any("gpu_prefetch_per_worker" in p for p in sep["_problems"])


def test_assignment_worker_threads_default_to_two_and_are_configurable():
    sep = pc.resolve({"performance": {"enabled": True, "stages": {}}})["stages"]["separation"]
    assert sep["assignment_worker_threads"] == 2
    more = pc.resolve({"performance": {"enabled": True, "stages": {
        "separation": {"assignment_worker_threads": 4, "postprocess_workers": 8}}}})
    assert more["stages"]["separation"]["assignment_worker_threads"] == 4
    assert more["stages"]["separation"]["postprocess_workers"] == 8


def test_the_a100_profile_gives_assignment_twice_the_threads_and_eight_post_workers():
    import json
    cfg = json.load(open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "config.json"), encoding="utf-8"))
    sep = cfg["environments"]["a100"]["performance"]["stages"]["separation"]
    assert sep["assignment_threads"] == 24 and sep["assignment_worker_threads"] == 4
    assert sep["postprocess_workers"] == 8


def test_the_a100_profile_runs_four_bs_roformer_instances_two_on_each_card():
    import json
    cfg = json.load(open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "config.json"), encoding="utf-8"))
    music = cfg["environments"]["a100"]["performance"]["stages"]["music"]
    assert music["max_separator_workers"] == 2 and music["workers_per_gpu"] == 2
    assert pc.resolve_music_worker_devices([0, 1], music["workers_per_gpu"], True) == [
        0, 1, 0, 1]


def test_the_a100_profile_runs_diarizen_with_a_batch_of_72():
    import json
    cfg = json.load(open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "config.json"), encoding="utf-8"))
    assert cfg["environments"]["a100"]["models"]["diarizen"]["batch_size"] == 72


def test_the_a100_profile_sends_refinement_to_vllm_in_large_chunks():
    # vLLM queues what does not fit and 95% of the card is KV cache, so a small client
    # batch only left it idle between batches.
    import json
    cfg = json.load(open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "config.json"), encoding="utf-8"))
    assert cfg["environments"]["a100"]["models"]["refinement"]["batch_size"] == 1024
