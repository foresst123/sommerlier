import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from tools import preflight_check as P  # noqa: E402


def test_parse_gpu_csv_reads_free_and_total_in_gb():
    text = "NVIDIA A100-SXM4-40GB, 39000, 40960\nNVIDIA A100, 1024, 2048\nbad line\n"
    gpus = P.parse_gpu_csv(text)
    assert gpus[0][0] == "NVIDIA A100-SXM4-40GB"
    assert round(gpus[0][1], 2) == round(39000 / 1024, 2)
    assert round(gpus[1][2], 1) == 2.0
    assert len(gpus) == 2


def test_check_gpu_fails_without_nvidia_smi():
    c = P.check_gpu(1.0, nvidia_smi="/nonexistent/nvidia-smi")
    assert not c.ok and "nvidia-smi" in c.detail


def test_check_disk_passes_for_tiny_threshold_and_fails_for_huge(tmp_path):
    assert P.check_disk(str(tmp_path / "new" / "dir"), 0.0001).ok
    assert not P.check_disk(str(tmp_path), 10 ** 9).ok


def test_count_voxceleb_layout(tmp_path):
    root = tmp_path / "aac"
    (root / "id00001" / "v1").mkdir(parents=True)
    (root / "id00001" / "v1" / "a.m4a").write_bytes(b"x")
    (root / "id00001" / "v1" / "b.m4a").write_bytes(b"x")
    (root / "id00002" / "v2").mkdir(parents=True)
    (root / "id00002" / "v2" / "c.wav").write_bytes(b"x")
    assert P.count_voxceleb(str(root), "m4a") == (2, 2, 2)
    assert P.count_voxceleb(str(root), "wav") == (2, 1, 2)
    assert P.count_voxceleb(str(root), "m4a", sample=1) == (2, 2, 1)
    assert P.count_voxceleb(str(tmp_path / "missing"), "m4a") == (0, 0, 0)


def test_check_voxceleb_compares_speaker_count(tmp_path):
    root = tmp_path / "aac"
    (root / "id00001" / "v1").mkdir(parents=True)
    (root / "id00001" / "v1" / "a.m4a").write_bytes(b"x")
    assert P.check_voxceleb(str(root), "m4a", 1).ok
    assert not P.check_voxceleb(str(root), "m4a", 5994).ok


def test_check_env_reports_missing_python_and_missing_module(tmp_path):
    assert not P.check_env("x", "/nonexistent/python").ok
    proc_python = os.sys.executable
    assert P.check_env("here", proc_python).ok
    bad = P.check_env("here", f"{proc_python}:module_that_does_not_exist_xyz")
    assert not bad.ok


def test_check_out_writable_creates_and_cleans_probe(tmp_path):
    out = tmp_path / "plda"
    c = P.check_out_writable(str(out))
    assert c.ok and out.is_dir()
    assert not (out / ".preflight_write_test").exists()


def test_main_returns_nonzero_when_a_required_check_fails(tmp_path, capsys):
    rc = P.main(["--out", str(tmp_path), "--min-free-gpu-gb", "1e9", "--min-free-disk-gb", "0",
                 "--expect-speakers", "1"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "gpu" in out and "mục đạt" in out
