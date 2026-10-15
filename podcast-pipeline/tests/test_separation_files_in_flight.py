import json
import os
from types import SimpleNamespace

from utils.batch import _stage_parallelism


def _args(**separation):
    return SimpleNamespace(
        separator="sidon",
        performance_config={"enabled": True,
                            "stages": {"separation": separation}})


def test_files_in_flight_sets_how_many_files_separate_at_once():
    assert _stage_parallelism(_args(max_workers=2, files_in_flight=4), "separation") == 4


def test_zero_keeps_the_old_two_file_rule():
    assert _stage_parallelism(_args(max_workers=2, files_in_flight=0), "separation") == 2
    assert _stage_parallelism(_args(max_workers=1), "separation") == 1


def test_files_in_flight_is_ignored_when_performance_is_off():
    args = _args(max_workers=2, files_in_flight=4)
    args.performance_config["enabled"] = False
    assert _stage_parallelism(args, "separation") == 1


def test_the_a100_profile_separates_four_files_and_hf_profile_is_gone():
    path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    with open(path) as handle:
        environments = json.load(handle)["environments"]
    assert "a100_hf" not in environments
    assert environments["a100"]["performance"]["stages"]["separation"][
        "files_in_flight"] == 4
