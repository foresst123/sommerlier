"""Where a run stops is chosen on the command line, not by the a100 profile.

main.py parses sys.argv at import time, so its source text is read, as in
tests/test_main_discard.py.
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _main_src():
    return open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()


def test_the_a100_profile_runs_every_stage_unless_told_to_stop():
    config = json.load(open(os.path.join(ROOT, "config.json"), encoding="utf-8"))
    assert "stop_after" not in config["environments"]["a100"]["pipeline"]


def test_the_configured_stop_is_a_stage_the_command_line_accepts():
    src = _main_src()
    choices = re.search(r'"--stop_after".*?choices=\[(.*?)\]', src, re.S).group(1)
    accepted = set(re.findall(r'"([a-z_]+)"', choices))
    config = json.load(open(os.path.join(ROOT, "config.json"), encoding="utf-8"))
    for env in config["environments"].values():
        stop = env.get("pipeline", {}).get("stop_after")
        assert stop is None or stop in accepted
    assert "none" in accepted


def test_stop_after_none_cancels_a_profile_that_stops_early():
    src = _main_src()
    land = src.index('for k, v in env_profile.get("pipeline", {}).items():')
    cancel = src.index('if getattr(args, "stop_after", None) == "none":')
    assert land < cancel, "the profile value must land before 'none' is resolved"
    assert 'args.stop_after = None' in src[cancel:cancel + 120]
