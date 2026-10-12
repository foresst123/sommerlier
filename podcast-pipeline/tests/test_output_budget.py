import json
import os

from utils.output_budget import (conversation_seconds, target_reached,
                                 total_conversation_hours)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _src():
    return open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()


def _write(out_dir, durations, name="manifest.jsonl"):
    d = os.path.join(out_dir, "conversation_exports")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), "w") as fh:
        for x in durations:
            fh.write(json.dumps({"duration": x}) + "\n")


def test_sums_durations_over_files(tmp_path):
    _write(str(tmp_path / "a"), [1800, 1800])
    _write(str(tmp_path / "b"), [3600])
    total = total_conversation_hours(["a", "b"], lambda p: str(tmp_path / p))
    assert total == 2.0


def test_quarantine_manifest_is_not_counted(tmp_path):
    _write(str(tmp_path), [3600])
    _write(str(tmp_path), [7200], name="manifest_quarantine.jsonl")
    assert total_conversation_hours(["x"], lambda p: str(tmp_path)) == 1.0


def test_missing_file_is_zero(tmp_path):
    assert conversation_seconds(str(tmp_path / "nope.jsonl")) == 0.0


def test_malformed_blank_and_bad_durations_skipped(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text('\n{"duration": 10}\nnot json\n[1]\n{"duration": "5"}\n'
                 '{"duration": -3}\n{"duration": null}\n{"x": 1}\n{"duration": true}\n'
                 '{"duration": 2.5}\n   \n')
    assert conversation_seconds(str(p)) == 12.5


def test_rerun_replaces_instead_of_adding(tmp_path):
    out = str(tmp_path)
    _write(out, [3600])
    assert total_conversation_hours(["f"], lambda p: out) == 1.0
    _write(out, [1800])
    assert total_conversation_hours(["f"], lambda p: out) == 0.5


def test_failing_out_dir_of_counts_zero(tmp_path):
    _write(str(tmp_path / "ok"), [3600])

    def out_dir_of(p):
        if p == "bad":
            raise RuntimeError("boom")
        return str(tmp_path / p)

    assert total_conversation_hours(["bad", "ok"], out_dir_of) == 1.0


def test_target_reached():
    assert not target_reached(100.0, None)
    assert not target_reached(100.0, 0)
    assert not target_reached(100.0, -1)
    assert not target_reached(9.99, 10)
    assert target_reached(10, 10)
    assert target_reached(11, 10)


def test_resolve_target_prefers_the_command_line_and_treats_junk_as_off():
    from utils.output_budget import resolve_target
    assert resolve_target(12.5, 100) == 12.5
    assert resolve_target(None, 100) == 100.0
    assert resolve_target(None, "40") == 40.0
    for off in (None, 0, 0.0, -3, float("nan"), float("inf"), True):
        assert resolve_target(None, off) is None
    assert resolve_target(0, 100) is None, "an explicit 0 on the command line turns it off"
    try:
        resolve_target(None, "lots")
    except ValueError:
        pass
    else:
        raise AssertionError("text that is not a number must not silently disable the limit")


def test_a_stop_after_run_is_never_held_to_the_target():
    from utils.output_budget import enforced
    assert enforced(50.0, partial_run=False)
    assert not enforced(50.0, partial_run=True)
    assert not enforced(None, partial_run=False)


def test_an_unreadable_manifest_is_counted_as_far_as_it_reads(tmp_path):
    bad = tmp_path / "manifest.jsonl"
    bad.write_bytes(b'{"duration": 60}\n{"duration": 30}\n\xff\xfe{"duration": 5\n')
    assert conversation_seconds(str(bad)) == 90.0
    huge = tmp_path / "huge.jsonl"
    huge.write_text('{"duration": 1' + "0" * 400 + '}\n{"duration": 10}\n')
    assert conversation_seconds(str(huge)) == 10.0


def test_main_wires_the_target_through_the_tested_helpers():
    src = _src()
    assert "--target_output_hours" in src
    assert "resolve_target(args.target_output_hours" in src
    assert "use_target = enforced(target_output_hours, partial_run)" in src
    # The post-pass check belongs to the pass loop, after the ledger is written.
    loop = src.index("ledger.pending(run_snapshot")
    save = src.index("ledger.save()", loop)
    assert save < src.index("Output so far:", save) < src.index("Output target reached", save)


def test_config_has_target_key_in_every_batch():
    config = json.load(open(os.path.join(ROOT, "config.json"), encoding="utf-8"))
    assert "target_output_hours" in config["batch"]
    for name in ("a100", "kaggle"):
        assert "target_output_hours" in config["environments"][name]["batch"]
