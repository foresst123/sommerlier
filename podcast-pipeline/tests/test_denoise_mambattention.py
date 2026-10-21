import os
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest
from scipy.signal import firwin

from models.denoise import subproc
from models.denoise.base import DenoiseError
from models.denoise.mambattention import MambAttentionBackend

PY = sys.executable

_HEAD = """
import argparse, json, sys
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp"); ap.add_argument("--out"); ap.add_argument("--sr", type=int); ap.add_argument("--cfg-json")
a = ap.parse_args()
x = np.load(a.inp)
cfg = json.load(open(a.cfg_json))
"""


def _worker(tmp_path, body, name="w.py"):
    p = tmp_path / name
    p.write_text(_HEAD + textwrap.dedent(body))
    return str(p)


def _sig(n, sr, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    noise = np.convolve(rng.standard_normal(n + 200), firwin(201, 3000, fs=sr), mode="valid")
    return (0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * noise).astype(np.float32)


def _snapshot():
    import tempfile
    return set(d for d in os.listdir(tempfile.gettempdir()) if d.startswith("denoise_worker_"))


# --------------------------------------------------------------- subprocess plumbing

def test_run_worker_success_and_cfg_passed(tmp_path):
    w = _worker(tmp_path, "np.save(a.out, (x * cfg['gain']).astype(np.float32))")
    x = _sig(4000, 16000)
    y = subproc.run_worker(PY, w, x, 16000, {"gain": 0.5}, timeout=60)
    assert y.dtype == np.float32 and y.shape == x.shape
    np.testing.assert_allclose(y, x * 0.5, atol=1e-7)


def test_run_worker_rejects_wrong_length(tmp_path):
    w = _worker(tmp_path, "np.save(a.out, x[:-1])")
    with pytest.raises(DenoiseError, match="shape"):
        subproc.run_worker(PY, w, _sig(1000, 16000), 16000, {}, timeout=60)


def test_run_worker_rejects_non_finite(tmp_path):
    w = _worker(tmp_path, "y = x.copy(); y[3] = np.nan; np.save(a.out, y)")
    with pytest.raises(DenoiseError, match="non-finite"):
        subproc.run_worker(PY, w, _sig(1000, 16000), 16000, {}, timeout=60)


def test_run_worker_rejects_2d_and_missing_output(tmp_path):
    w2 = _worker(tmp_path, "np.save(a.out, x[None])", "w2.py")
    with pytest.raises(DenoiseError, match="shape"):
        subproc.run_worker(PY, w2, _sig(100, 16000), 16000, {}, timeout=60)
    w3 = _worker(tmp_path, "pass", "w3.py")
    with pytest.raises(DenoiseError, match="no output"):
        subproc.run_worker(PY, w3, _sig(100, 16000), 16000, {}, timeout=60)


def test_run_worker_nonzero_exit_has_stderr_tail(tmp_path):
    w = _worker(tmp_path, "sys.stderr.write('boom: CUDA out of memory\\n'); sys.exit(3)")
    with pytest.raises(DenoiseError) as e:
        subproc.run_worker(PY, w, _sig(100, 16000), 16000, {}, timeout=60)
    assert "code 3" in str(e.value) and "CUDA out of memory" in str(e.value)


def test_run_worker_timeout_kills_child_group(tmp_path):
    pidfile = tmp_path / "grandchild.pid"
    body = f"""
    import subprocess, time
    g = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    open({str(pidfile)!r}, "w").write(str(g.pid))
    time.sleep(60)
    """
    w = _worker(tmp_path, body)
    t0 = time.time()
    with pytest.raises(DenoiseError, match="timed out"):
        subproc.run_worker(PY, w, _sig(100, 16000), 16000, {}, timeout=3)
    assert time.time() - t0 < 30
    pid = int(pidfile.read_text())
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
        try:  # reaped by init may lag: zombie counts as dead
            if subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip().startswith("Z"):
                break
        except OSError:
            pass
    else:
        os.kill(pid, 9)
        pytest.fail("grandchild survived the timeout")


def test_run_worker_cleans_temp_files(tmp_path):
    before = _snapshot()
    ok = _worker(tmp_path, "np.save(a.out, x)", "ok.py")
    bad = _worker(tmp_path, "sys.exit(1)", "bad.py")
    subproc.run_worker(PY, ok, _sig(100, 16000), 16000, {}, timeout=60)
    with pytest.raises(DenoiseError):
        subproc.run_worker(PY, bad, _sig(100, 16000), 16000, {}, timeout=60)
    with pytest.raises(DenoiseError):
        subproc.run_worker(PY, bad, _sig(100, 16000), 16000, {}, timeout=60)
    assert _snapshot() == before


def test_run_worker_missing_interpreter(tmp_path):
    w = _worker(tmp_path, "np.save(a.out, x)")
    with pytest.raises(DenoiseError, match="cannot start"):
        subproc.run_worker(str(tmp_path / "nope" / "python"), w, _sig(100, 16000), 16000, {}, timeout=10)


def test_worker_gets_project_root_on_pythonpath(tmp_path):
    w = _worker(tmp_path, "from models.denoise.chunking import fit_length\nnp.save(a.out, fit_length(x, len(x)))")
    x = _sig(500, 16000)
    np.testing.assert_array_equal(subproc.run_worker(PY, w, x, 16000, {}, timeout=60), x)


# ------------------------------------------------------------------ availability

def test_check_python_modules():
    assert subproc.check_python_modules(PY, ["numpy", "os"]) == (True, "")
    ok, why = subproc.check_python_modules(PY, ["numpy", "no_such_mod_xyz"])
    assert not ok and "no_such_mod_xyz" in why
    ok, why = subproc.check_python_modules("/nonexistent/python", ["numpy"])
    assert not ok and "not found" in why
    assert not subproc.check_python_modules("", ["numpy"])[0]


def test_check_available_messages(tmp_path, monkeypatch):
    for k in ("MAMBATTENTION_PYTHON", "MAMBATTENTION_DIR"):
        monkeypatch.delenv(k, raising=False)
    ok, why = MambAttentionBackend.check_available({})
    assert not ok and "MAMBATTENTION_PYTHON" in why
    ok, why = MambAttentionBackend.check_available({"python": "/nonexistent/python"})
    assert not ok and "MAMBATTENTION_DIR" in why
    ok, why = MambAttentionBackend.check_available({"python": "/nonexistent/python", "repo_dir": str(tmp_path)})
    assert not ok and "generator.py" in why
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "generator.py").write_text("")
    ok, why = MambAttentionBackend.check_available({"python": PY, "repo_dir": str(tmp_path)})
    assert not ok and "checkpoint not found" in why
    ck = tmp_path / "ck.pth"
    ck.write_text("")
    ok, why = MambAttentionBackend.check_available({"python": PY, "repo_dir": str(tmp_path), "checkpoint": str(ck)})
    assert not ok and "yaml" in why
    yml = tmp_path / "m.yaml"
    yml.write_text("")
    cfg = {"python": "/nonexistent/python", "repo_dir": str(tmp_path), "checkpoint": str(ck), "config_yaml": str(yml)}
    ok, why = MambAttentionBackend.check_available(cfg)
    assert not ok and "interpreter" in why
    cfg["python"] = PY
    ok, why = MambAttentionBackend.check_available(cfg)  # this interpreter has no torch/mamba_ssm... or does
    assert ok or "lacks python package" in why


def test_load_raises_denoise_error_when_unavailable(monkeypatch):
    monkeypatch.delenv("MAMBATTENTION_PYTHON", raising=False)
    with pytest.raises(DenoiseError, match="MAMBATTENTION_PYTHON"):
        MambAttentionBackend({}).load()


# ---------------------------------------------------------- backend with fake worker

@pytest.mark.parametrize("sr,n", [(24000, 24000 * 3 + 7), (16000, 16001), (48000, 48000 + 123), (22050, 5000)])
def test_backend_length_and_resampling_invariants(tmp_path, sr, n):
    w = _worker(tmp_path, """
    assert a.sr == 16000
    assert cfg["chunk_s"] == 4.0 and cfg["overlap_s"] == 0.5
    np.save(a.out, x * np.float32(0.5))
    """)
    x = _sig(n, sr)
    be = MambAttentionBackend({"python": PY, "worker": w})
    y = be.enhance(x, sr)
    assert y.dtype == np.float32 and y.shape == x.shape and np.all(np.isfinite(y))
    m = slice(sr // 10, -sr // 10) if n > sr // 2 else slice(None)
    np.testing.assert_allclose(y[m], 0.5 * x[m], atol=0.03)


def test_backend_removes_constant_lag(tmp_path):
    w = _worker(tmp_path, "y = np.zeros_like(x); y[25:] = x[:-25]; np.save(a.out, y)")
    x = _sig(16000 * 3, 16000)
    y = MambAttentionBackend({"python": PY, "worker": w}).enhance(x, 16000)
    assert np.max(np.abs(y[1000:-1000] - x[1000:-1000])) < 1e-4


def test_backend_worker_failure_surfaces(tmp_path):
    w = _worker(tmp_path, "sys.stderr.write('mamba_ssm exploded'); sys.exit(1)")
    with pytest.raises(DenoiseError, match="mamba_ssm exploded"):
        MambAttentionBackend({"python": PY, "worker": w}).enhance(_sig(8000, 16000), 16000)


def test_backend_timeout_config(tmp_path):
    w = _worker(tmp_path, "import time; time.sleep(60)")
    with pytest.raises(DenoiseError, match="timed out"):
        MambAttentionBackend({"python": PY, "worker": w, "timeout_s": 2}).enhance(_sig(8000, 16000), 16000)


def test_backend_empty_input():
    assert len(MambAttentionBackend({}).enhance(np.zeros(0, dtype=np.float32), 24000)) == 0


def test_import_does_not_import_torch():
    code = ("import sys; import models.denoise.mambattention, models.denoise.subproc; "
            "assert 'torch' not in sys.modules, 'torch imported'")
    res = subprocess.run([PY, "-c", code], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "."})
    assert res.returncode == 0, res.stderr
