"""Claims and the merging ledger that let pipelines share one audio directory.

Run:  python -m pytest tests/test_file_claims.py -q     (from podcast-pipeline/)
"""
import json
import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.file_claims import FileClaims, claim_group
from utils.progress import ProgressLedger


def _files(tmp_path, n):
    out = []
    for i in range(n):
        p = tmp_path / f"f{i:02d}.mp3"
        p.write_bytes(b"x")
        out.append(str(p))
    return out


def _dead_pid():
    pid = 2 ** 22 + 12345
    while True:
        try:
            os.kill(pid, 0)
            pid += 1
        except ProcessLookupError:
            return pid
        except PermissionError:
            pid += 1


def test_two_claimers_cannot_share_a_file(tmp_path):
    f = _files(tmp_path, 1)[0]
    a, b = FileClaims(str(tmp_path)), FileClaims(str(tmp_path))
    assert a.try_claim(f)
    assert not b.try_claim(f)
    assert b.is_claimed_by_other(f) and not a.is_claimed_by_other(f)
    assert b.claimed_by_others([f]) == {f}
    assert a.held() == {f}


def test_release_only_removes_own_claim(tmp_path):
    f = _files(tmp_path, 1)[0]
    a, b = FileClaims(str(tmp_path)), FileClaims(str(tmp_path))
    a.try_claim(f)
    b.release(f)
    assert b.is_claimed_by_other(f)
    a.release(f)
    assert not b.is_claimed_by_other(f)
    assert b.try_claim(f)


def test_claim_of_dead_pid_is_taken_over(tmp_path):
    f = _files(tmp_path, 1)[0]
    a = FileClaims(str(tmp_path))
    assert a.try_claim(f)
    claim = a._claim_path(f)
    data = json.load(open(claim))
    data.update(pid=_dead_pid(), host=socket.gethostname(), token="ghost")
    json.dump(data, open(claim, "w"))
    b = FileClaims(str(tmp_path))
    assert not b.is_claimed_by_other(f)
    assert b.try_claim(f)
    assert json.load(open(claim))["token"] == b.token


def test_old_mtime_is_stale_but_fresh_is_not(tmp_path):
    f = _files(tmp_path, 1)[0]
    a = FileClaims(str(tmp_path), ttl=60)
    a.try_claim(f)
    claim = a._claim_path(f)
    data = json.load(open(claim))
    data.update(host="other-host", token="remote")
    json.dump(data, open(claim, "w"))
    b = FileClaims(str(tmp_path), ttl=60)
    assert not b.try_claim(f)
    old = time.time() - 120
    os.utime(claim, (old, old))
    assert b.try_claim(f)


def test_concurrent_takeover_of_stale_claim_has_one_winner(tmp_path):
    f = _files(tmp_path, 1)[0]
    a = FileClaims(str(tmp_path), ttl=60)
    a.try_claim(f)
    claim = a._claim_path(f)
    old = time.time() - 600
    os.utime(claim, (old, old))
    claimers = [FileClaims(str(tmp_path), ttl=60) for _ in range(8)]
    wins, barrier = [], threading.Barrier(8)

    def go(c):
        barrier.wait()
        wins.append(c.try_claim(f))

    ts = [threading.Thread(target=go, args=(c,)) for c in claimers]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sum(wins) == 1


def test_heartbeat_keeps_claim_alive_past_ttl(tmp_path):
    f = _files(tmp_path, 1)[0]
    a = FileClaims(str(tmp_path), ttl=1)
    b = FileClaims(str(tmp_path), ttl=1)
    # make the claim look foreign so only mtime decides
    with a:
        a.try_claim(f)
        data = json.load(open(a._claim_path(f)))
        data.update(host="other-host")
        json.dump(data, open(a._claim_path(f), "w"))
        time.sleep(2.5)
        assert b.is_claimed_by_other(f)
    # stopped heartbeat and released
    assert not b.is_claimed_by_other(f)


def test_disabled_is_a_noop(tmp_path):
    f = _files(tmp_path, 1)[0]
    a = FileClaims(str(tmp_path), enabled=False)
    b = FileClaims(str(tmp_path), enabled=False)
    assert a.try_claim(f) and b.try_claim(f)
    assert not b.is_claimed_by_other(f)
    a.release_all()
    assert not os.path.exists(os.path.join(str(tmp_path), "_sommelier_claims"))


def test_ledgers_do_not_lose_each_others_entries(tmp_path):
    f = _files(tmp_path, 6)
    a, b = ProgressLedger(str(tmp_path)), ProgressLedger(str(tmp_path))
    a.mark_done(f[0]); a.save()
    b.mark_done(f[1]); b.mark_failed(f[2], "boom"); b.mark_rejected(f[3], "bad"); b.save()
    a.mark_failed(f[2], "boom"); a.mark_failed(f[2], "boom"); a.save()
    b.set_quality(f[4], {"mos": 3})
    c = ProgressLedger(str(tmp_path))
    assert {os.path.basename(f[0]), os.path.basename(f[1])} <= set(c.done)
    assert c.attempts(f[2]) == 2
    assert c.is_rejected(f[3])
    assert c.get_quality(f[4]) == {"mos": 3}


def test_done_beats_failed_for_the_same_file(tmp_path):
    f = _files(tmp_path, 1)[0]
    a, b = ProgressLedger(str(tmp_path)), ProgressLedger(str(tmp_path))
    a.mark_failed(f, "oom")
    b.mark_done(f); b.save()
    a.save()
    c = ProgressLedger(str(tmp_path))
    assert c.is_done(f) and not c.failed
    assert a.is_done(f)


def test_refresh_sees_other_ledgers_done(tmp_path):
    f = _files(tmp_path, 3)
    a, b = ProgressLedger(str(tmp_path)), ProgressLedger(str(tmp_path))
    b.mark_done(f[0]); b.save()
    assert a.pending(f) == f
    a.refresh()
    assert a.pending(f) == f[1:]


def test_non_persistent_ledger_never_touches_disk(tmp_path):
    led = ProgressLedger(str(tmp_path), persist=False)
    led.mark_done("x.mp3"); led.save(); led.refresh()
    assert os.listdir(str(tmp_path)) == []


def _plan(todo, max_hours, logger=None):
    return [todo[:2]]


def test_claim_group_claims_only_the_group(tmp_path):
    f = _files(tmp_path, 5)
    led = ProgressLedger(str(tmp_path))
    mine = FileClaims(str(tmp_path))
    group, others = claim_group(led, mine, f, set(), 1.0, planner=_plan)
    assert group == f[:2] and others == 0
    assert mine.held() == set(f[:2])
    other = FileClaims(str(tmp_path))
    group2, _ = claim_group(led, other, f, set(), 1.0, planner=_plan)
    assert group2 == f[2:4]


def test_claim_group_stops_when_rest_is_held_by_others(tmp_path):
    f = _files(tmp_path, 3)
    led = ProgressLedger(str(tmp_path))
    other = FileClaims(str(tmp_path))
    for p in f:
        other.try_claim(p)
    mine = FileClaims(str(tmp_path))
    group, others = claim_group(led, mine, f, set(), 1.0, planner=_plan)
    assert group == [] and others == 3
    assert mine.held() == set()


def test_claim_group_replans_around_a_lost_race(tmp_path):
    f = _files(tmp_path, 4)
    led = ProgressLedger(str(tmp_path))
    mine = FileClaims(str(tmp_path))
    real = mine.try_claim
    thief = FileClaims(str(tmp_path))

    def racing(p):  # another pipeline grabs f[0] between check and claim
        if p == f[0] and not thief.held():
            thief.try_claim(p)
        return real(p)

    mine.try_claim = racing
    group, others = claim_group(led, mine, f, set(), 1.0, planner=_plan)
    assert f[0] not in group and group == [f[1]]
    assert others >= 1


def test_claim_group_drops_files_finished_between_refresh_and_claim(tmp_path):
    f = _files(tmp_path, 3)
    led_a, led_b = ProgressLedger(str(tmp_path)), ProgressLedger(str(tmp_path))
    ca, cb = FileClaims(str(tmp_path)), FileClaims(str(tmp_path))
    assert ca.try_claim(f[0])

    def plan(todo, max_hours, logger=None):
        # A finishes f[0] (save, then release) after B's refresh, before B claims.
        led_a.mark_done(f[0]); led_a.save(); ca.release(f[0])
        return [todo[:1]]

    group, _ = claim_group(led_b, cb, f, set(), 1.0, planner=plan)
    assert f[0] not in group and f[0] not in cb.held()
    assert group == [f[1]]


_WORKER = '''
import os, sys, time
sys.path.insert(0, sys.argv[1])
from utils.file_claims import FileClaims, claim_group
from utils.progress import ProgressLedger
d, log = sys.argv[2], sys.argv[3]
files = sorted(os.path.join(d, n) for n in os.listdir(d) if n.endswith(".mp3"))
led, cl = ProgressLedger(d), FileClaims(d)
plan = lambda todo, h, logger=None: [todo[:2]]
while True:
    group, _ = claim_group(led, cl, files, set(), 1.0, planner=plan)
    if not group:
        break
    for p in group:
        with open(log, "a") as f:
            f.write(os.path.basename(p) + "\\n")
        time.sleep(0.01)
        led.mark_done(p); led.save()
        cl.release(p)
'''


def test_three_processes_run_each_file_exactly_once(tmp_path):
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = tmp_path / "worker.py"
    script.write_text(_WORKER)
    for rnd in range(5):
        d = tmp_path / f"r{rnd}"
        d.mkdir()
        _files(d, 12)
        log = tmp_path / f"log{rnd}.txt"
        procs = [subprocess.Popen([sys.executable, str(script), root, str(d), str(log)])
                 for _ in range(3)]
        assert all(p.wait(timeout=60) == 0 for p in procs)
        ran = log.read_text().split()
        assert sorted(ran) == sorted(f"f{i:02d}.mp3" for i in range(12)), ran
