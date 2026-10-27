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


PROBE = """
import sys, json
sys.argv = ['main.py', '--audio', 'x.wav', '--env', 'kaggle', '--stop_after', 'none']
src = open('main.py', encoding='utf-8').read().split('# 2. DELAYED IMPORTS')[0]
ns = {'__name__': 'probe', '__file__': 'main.py'}
exec(compile(src, 'main.py', 'exec'), ns)
parser = ns['_build_parser']()
accepted, rejected = [], []
for stop in json.loads(sys.stdin.read()):
    try:
        parser.parse_args(['--audio', 'x.wav', '--stop_after', stop])
        accepted.append(stop)
    except SystemExit:
        rejected.append(stop)
print(json.dumps({'accepted': accepted, 'rejected': rejected}))
"""


def _parser_verdicts(values):
    """Run main's real _build_parser() in a child process (main parses argv at import)."""
    import subprocess
    import sys
    env = dict(os.environ, SOMMELIER_SKIP_BOOTSTRAP="1")
    out = subprocess.run([sys.executable, "-c", PROBE], input=json.dumps(values), cwd=ROOT,
                         env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-1500:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_every_stop_point_the_pipeline_honours_parses_on_the_command_line():
    """The regression: choices was a second hand-kept list, and it lacked
    conversation_selection, so run_variants_resident's --stop-after for that point
    killed the LLM process with 'invalid choice'."""
    import sys
    sys.path.insert(0, ROOT)
    from utils.batch import pipeline_stop_points
    stops = list(pipeline_stop_points()) + ["none"]
    verdicts = _parser_verdicts(stops)
    assert verdicts["rejected"] == [], verdicts
    assert verdicts["accepted"] == stops


def test_the_command_line_rejects_a_value_that_is_not_a_stop_point():
    verdicts = _parser_verdicts(["bogus_stage", "conversation_selection"])
    assert verdicts == {"accepted": ["conversation_selection"], "rejected": ["bogus_stage"]}


def test_stop_after_none_cancels_a_profile_that_stops_early():
    src = _main_src()
    land = src.index('for k, v in env_profile.get("pipeline", {}).items():')
    cancel = src.index('if getattr(args, "stop_after", None) == "none":')
    assert land < cancel, "the profile value must land before 'none' is resolved"
    assert 'args.stop_after = None' in src[cancel:cancel + 120]
