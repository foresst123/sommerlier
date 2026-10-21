"""Run a denoise model in a separate Python environment through a worker script.

For models whose dependencies conflict with the pipeline's (e.g. MambAttention needs torch 2.2.2).
The parent only needs numpy: audio goes in and out as .npy files, the worker is a plain script:

    python SCRIPT --in X.npy --out Y.npy --sr N --cfg-json PATH
"""
import json
import os
import signal
import subprocess
import tempfile
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .base import DenoiseError

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _tail(text: str, n: int = 1500) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else "..." + text[-n:]


def _kill_group(proc: subprocess.Popen):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def check_python_modules(python: str, modules: Sequence[str], timeout: float = 60.0) -> Tuple[bool, str]:
    """(ok, reason): `python` exists and can find every module (find_spec, nothing is imported)."""
    if not python:
        return False, "no python interpreter configured"
    if not (os.path.isfile(python) and os.access(python, os.X_OK)):
        return False, f"python interpreter not found or not executable: {python}"
    code = ("import importlib.util as u, json, sys\n"
            "print(json.dumps([m for m in json.loads(sys.argv[1]) if u.find_spec(m) is None]))")
    try:
        res = subprocess.run([python, "-I", "-c", code, json.dumps(list(modules))], capture_output=True,
                             text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"cannot run {python}: {e}"
    if res.returncode != 0:
        return False, f"{python} failed: {_tail(res.stderr, 300)}"
    try:
        missing = json.loads(res.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return False, f"unexpected output from {python}: {_tail(res.stdout, 200)}"
    if missing:
        return False, f"{python} lacks python package(s): {', '.join(missing)}"
    return True, ""


def run_worker(python: str, script: str, wav: np.ndarray, sr: int, cfg: dict, timeout: float,
               env: Optional[Dict[str, str]] = None) -> np.ndarray:
    """Denoise `wav` (float32 mono 1-D) with `script` run by `python`; returns float32 of the same length.

    The child runs in its own process group and is killed on timeout or KeyboardInterrupt.
    Raises DenoiseError (with the tail of the worker's stderr) on any failure or invalid output.
    """
    wav = np.ascontiguousarray(wav, dtype=np.float32)
    n = len(wav)
    child_env = dict(os.environ)
    child_env.update(env or {})
    parts = [PROJECT_ROOT] + [p for p in child_env.get("PYTHONPATH", "").split(os.pathsep) if p]
    child_env["PYTHONPATH"] = os.pathsep.join(parts)
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    with tempfile.TemporaryDirectory(prefix="denoise_worker_") as tmp:
        f_in, f_out, f_cfg = (os.path.join(tmp, k) for k in ("in.npy", "out.npy", "cfg.json"))
        np.save(f_in, wav)
        with open(f_cfg, "w") as f:
            json.dump(cfg, f, default=str)
        cmd = [python, script, "--in", f_in, "--out", f_out, "--sr", str(int(sr)), "--cfg-json", f_cfg]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                    env=child_env, start_new_session=True)
        except OSError as e:
            raise DenoiseError(f"cannot start worker {python} {script}: {e}") from e
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as e:
            _kill_group(proc)
            raise DenoiseError(f"worker timed out after {timeout:g} s and was killed ({script})") from e
        except BaseException:  # KeyboardInterrupt etc.
            _kill_group(proc)
            raise
        if proc.returncode != 0:
            raise DenoiseError(f"worker exited with code {proc.returncode}: {_tail(stderr)}")
        if not os.path.isfile(f_out):
            raise DenoiseError(f"worker wrote no output file: {_tail(stderr)}")
        try:
            y = np.load(f_out, allow_pickle=False)
        except Exception as e:
            raise DenoiseError(f"worker output unreadable: {e}") from e
    if y.ndim != 1 or len(y) != n:
        raise DenoiseError(f"worker returned shape {y.shape}, expected ({n},)")
    if not np.issubdtype(y.dtype, np.floating):
        raise DenoiseError(f"worker returned dtype {y.dtype}, expected float32")
    y = y.astype(np.float32, copy=False)
    if not np.all(np.isfinite(y)):
        raise DenoiseError(f"worker returned {int(np.count_nonzero(~np.isfinite(y)))} non-finite sample(s)")
    return y
