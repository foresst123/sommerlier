"""Before it starts, the run looks for what an earlier run left behind and for a machine
that is already saturated, because either one made every measurement after it meaningless
(an orphaned vLLM engine held 12 GiB of a card; the load average was 167 on 24 cores).
Fake process tables, nothing is killed."""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import preflight


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append((record.levelname, record.getMessage()))


def _logger():
    logger = logging.getLogger(f"preflight-{id(object())}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handler = Capture()
    logger.addHandler(handler)
    return logger, handler


def _proc(pid, ppid, cmd, user="lamkd2", age=3600):
    return {"pid": pid, "ppid": ppid, "cmdline": cmd, "username": user, "age": age}


TABLE = [
    _proc(10, 1, ["VLLM::EngineCore"]),                                  # orphaned engine
    _proc(11, 1, ["/env/bin/python", "/repo/sidon_worker.py", "--config", "c.json"]),
    _proc(12, 500, ["VLLM::EngineCore"]),                                # still has a parent
    _proc(13, 1, ["VLLM::EngineCore"], user="someone_else"),             # not ours
    _proc(14, 1, ["/usr/bin/python", "train.py"]),                       # unrelated
    _proc(15, 1, ["/env/bin/python", "/repo/assignment_worker.py"], age=5),  # just started
]


def test_only_our_users_orphaned_engines_and_workers_are_found():
    found = preflight.find_orphans(TABLE, user="lamkd2", min_age=60)
    assert sorted(p["pid"] for p in found) == [10, 11]


def test_a_saturated_machine_is_reported_with_the_numbers():
    text = preflight.load_warning(load=167.0, cores=24)
    assert "167" in text and "24" in text
    assert preflight.load_warning(load=20.0, cores=24) is None


def test_orphans_are_listed_with_the_command_that_removes_them_and_not_killed():
    logger, handler = _logger()
    killed = []
    preflight.run(logger, table=TABLE, user="lamkd2", load=1.0, cores=24,
                  kill=lambda pid: killed.append(pid), kill_orphans=False)

    warnings = [m for level, m in handler.records if level == "WARNING"]
    assert any("2 leftover" in m for m in warnings)
    assert any("kill -9 10 11" in m for m in warnings)
    assert killed == []


def test_the_kill_option_removes_them_and_says_so():
    logger, handler = _logger()
    killed = []
    preflight.run(logger, table=TABLE, user="lamkd2", load=1.0, cores=24,
                  kill=lambda pid: killed.append(pid), kill_orphans=True)

    assert sorted(killed) == [10, 11]
    assert any("Killed 2" in m for _l, m in handler.records)


def test_a_clean_machine_says_nothing_beyond_one_info_line():
    logger, handler = _logger()
    preflight.run(logger, table=[_proc(20, 500, ["python", "main.py"])], user="lamkd2",
                  load=1.0, cores=24, kill=lambda pid: None, kill_orphans=True)
    assert not [1 for level, _m in handler.records if level in ("WARNING", "ERROR")]


def test_a_failure_while_looking_never_stops_the_run():
    logger, handler = _logger()

    def broken():
        raise RuntimeError("no /proc")

    preflight.run(logger, table=broken, user="lamkd2", load=1.0, cores=24,
                  kill=lambda pid: None, kill_orphans=False)
    assert any("could not" in m.lower() for _l, m in handler.records)


def test_main_runs_the_check_and_offers_the_flag():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source = open(os.path.join(root, "main.py"), encoding="utf-8").read()
    assert '"--kill_orphans"' in source and "preflight.run(" in source
