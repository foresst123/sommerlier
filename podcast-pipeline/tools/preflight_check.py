#!/usr/bin/env python3
"""Kiểm tra trước khi chạy kế hoạch PLDA → full (giai đoạn 0). Chỉ đọc, không sửa gì.

    python tools/preflight_check.py --out /home/lamkd2/data/plda \
        --audio-root /home/lamkd2/data/external/voxceleb2/aac --expect-speakers 5994 \
        --env som=/home/lamkd2/miniforge3/envs/som/bin/python:pyannote.audio \
        --env diarizen=/home/lamkd2/diarizen_env/bin/python:diarizen,pyannote.audio \
        --env nemotron=/home/lamkd2/nemotron_env/bin/python \
        [--min-free-gpu-gb 20] [--min-free-disk-gb 150] [--test-archives 'aac.7z.*']

Kiểm tra:
  gpu       nvidia-smi thấy ít nhất một GPU, và GPU có đủ bộ nhớ trống.
  disk      dung lượng trống của thư mục --out.
  ffmpeg    có trong PATH (để giải mã m4a).
  env       mỗi --env: Python tồn tại, chạy được, và import được các module ghi sau dấu hai chấm.
  voxceleb2 số thư mục speaker bằng --expect-speakers, và số file audio có đuôi --ext.
  archives  (tuỳ chọn, --test-archives) chạy 7z t trên các file khớp mẫu; chậm vì phải đọc toàn bộ.
Thoát với mã 0 nếu mọi mục bắt buộc đạt, 1 nếu có mục không đạt.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def parse_gpu_csv(text: str) -> list[tuple[str, float, float]]:
    """Dòng của `nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader,nounits`
    -> [(name, free_gb, total_gb)]."""
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        name, free_mb, total_mb = parts
        gpus.append((name, float(free_mb) / 1024, float(total_mb) / 1024))
    return gpus


def check_gpu(min_free_gb: float, nvidia_smi: str = "nvidia-smi") -> Check:
    if shutil.which(nvidia_smi) is None:
        return Check("gpu", False, "nvidia-smi không có trong PATH")
    proc = subprocess.run([nvidia_smi, "--query-gpu=name,memory.free,memory.total", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        return Check("gpu", False, f"nvidia-smi lỗi: {proc.stderr.strip()}")
    gpus = parse_gpu_csv(proc.stdout)
    if not gpus:
        return Check("gpu", False, "không thấy GPU nào")
    best = max(free for _, free, _ in gpus)
    detail = "; ".join(f"{i}: {n}, trống {f:.1f}/{t:.1f} GB" for i, (n, f, t) in enumerate(gpus))
    return Check("gpu", best >= min_free_gb, f"{detail} (cần ≥ {min_free_gb} GB trống)")


def check_disk(path: str, min_free_gb: float) -> Check:
    probe = path
    while not os.path.exists(probe) and os.path.dirname(probe) != probe:
        probe = os.path.dirname(probe)
    free_gb = shutil.disk_usage(probe).free / 1024 ** 3
    return Check("disk", free_gb >= min_free_gb, f"{probe}: trống {free_gb:.1f} GB (cần ≥ {min_free_gb} GB)")


def check_ffmpeg() -> Check:
    path = shutil.which("ffmpeg")
    return Check("ffmpeg", path is not None, path or "không có trong PATH")


def check_env(name: str, spec: str) -> Check:
    """spec = /đường/dẫn/python[:mod1,mod2]"""
    python, _, mods = spec.partition(":")
    if not os.path.isfile(python):
        return Check(f"env {name}", False, f"không thấy {python}")
    code = "import importlib,sys\nfor m in sys.argv[1:]: importlib.import_module(m)\nprint('ok')"
    args = [python, "-c", code] + [m for m in mods.split(",") if m]
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr.strip().splitlines() or ["lỗi không rõ"])[-1]
        return Check(f"env {name}", False, f"{python}: {tail}")
    shown = f" import được {mods}" if mods else ""
    return Check(f"env {name}", True, f"{python}{shown}")


FILE_SAMPLE_SPEAKERS = 20


def count_voxceleb(audio_root: str, ext: str, sample: int = FILE_SAMPLE_SPEAKERS) -> tuple[int, int, int]:
    """(số thư mục speaker, số file .ext trong `sample` speaker đầu, số speaker được đếm file).

    Đếm file trên mọi speaker chậm trên ổ mạng (VoxCeleb2 có hàng triệu file), nên chỉ đếm trên một mẫu."""
    if not os.path.isdir(audio_root):
        return 0, 0, 0
    speakers = sorted(d for d in os.listdir(audio_root) if os.path.isdir(os.path.join(audio_root, d)))
    suffix = "." + ext.lower().lstrip(".")
    files = 0
    sampled = speakers[:sample]
    for spk in sampled:
        for _, _, names in os.walk(os.path.join(audio_root, spk)):
            files += sum(1 for n in names if n.lower().endswith(suffix))
    return len(speakers), files, len(sampled)


def check_voxceleb(audio_root: str, ext: str, expect_speakers: int) -> Check:
    n_spk, n_files, n_sampled = count_voxceleb(audio_root, ext)
    ok = n_spk == expect_speakers and n_files > 0
    return Check("voxceleb2", ok, f"{audio_root}: {n_spk} speaker (cần {expect_speakers}), "
                                  f"{n_files} file .{ext} trong {n_sampled} speaker đầu (mẫu)")


def check_archives(pattern: str) -> list[Check]:
    files = sorted(glob.glob(pattern))
    if not files:
        return [Check("archives", False, f"không khớp file nào: {pattern}")]
    out = []
    for f in files:
        proc = subprocess.run(["7z", "t", f], capture_output=True, text=True)
        out.append(Check(f"7z t {os.path.basename(f)}", proc.returncode == 0,
                         "không lỗi" if proc.returncode == 0 else (proc.stdout + proc.stderr).strip()[-200:]))
    return out


def check_out_writable(out: str) -> Check:
    try:
        os.makedirs(out, exist_ok=True)
        probe = os.path.join(out, ".preflight_write_test")
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("ok")
        os.remove(probe)
        return Check("out", True, f"{out} ghi được")
    except OSError as exc:
        return Check("out", False, f"{out}: {exc}")


def run_checks(a) -> list[Check]:
    checks = [check_gpu(a.min_free_gpu_gb), check_disk(a.out, a.min_free_disk_gb), check_ffmpeg(),
              check_out_writable(a.out)]
    for item in a.env:
        name, _, spec = item.partition("=")
        checks.append(check_env(name, spec))
    if a.audio_root:
        checks.append(check_voxceleb(a.audio_root, a.ext, a.expect_speakers))
    if a.test_archives:
        checks += check_archives(a.test_archives)
    return checks


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="thư mục đầu ra (dùng để đo dung lượng trống)")
    ap.add_argument("--audio-root", default=None, help="thư mục VoxCeleb2 (layout speaker/video/file)")
    ap.add_argument("--ext", default="m4a")
    ap.add_argument("--expect-speakers", type=int, default=5994, help="5994 là số speaker của VoxCeleb2 dev")
    ap.add_argument("--env", action="append", default=[], metavar="NAME=PYTHON[:mod1,mod2]")
    ap.add_argument("--min-free-gpu-gb", type=float, default=20.0)
    ap.add_argument("--min-free-disk-gb", type=float, default=150.0)
    ap.add_argument("--test-archives", default=None, help="glob các phần archive, ví dụ 'aac.7z.*' (chậm)")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    checks = run_checks(a)
    width = max(len(c.name) for c in checks)
    for c in checks:
        print(f"[{'OK ' if c.ok else 'LỖI'}] {c.name:<{width}}  {c.detail}")
    failed = [c for c in checks if not c.ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} mục đạt")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
