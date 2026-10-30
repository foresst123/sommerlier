#!/usr/bin/env python3
"""Diarization arms on fixed conversation samples, shown as waveform lanes (one row per speaker, no text).

    python tools/diar_arms_experiment.py --audio-dir SRC --samples tools/diar_arms_samples.json --out OUT/diar_arms \
        [--final OUT/final] [--gpu 6] [--env a100] [--only-arms A,C,F,G] [--fusion] [--k-min-clean 1.0] [--reuse] \
        [--windowed-piece-s 240] [--windowed-hop-s 10] [--windowed-python /path/to/nemotron/python] \
        [--python /path/to/som/python] [--vbx-ahc-threshold 0.6] [--vbx-fa 0.07] [--vbx-fb 0.8] [--plda-dir DIR] \
        [--vbx-python /path/to/dz/python] [--demucs-python /path/to/demucs/python] [--fb-sweep 0.2,0.4]

Per sample (a conversation clip of about 40 s inside a long source recording):
  1. cut a 5 min chunk of the source around the clip (it starts 2 min before the clip, clamped to the file), 16 kHz mono.
  2. arm A, Nemotron on the chunk. K = labels whose clean speech (utils.nemotron_two_pass.clean_speech_duration)
     reaches --k-min-clean seconds inside the clip, at least 1. The default 1.0 s keeps backchannel listeners that
     say only 'ừ à' (under 1.5 s) in K.
  3. arm B, DiariZen with its defaults. Arm C, DiariZen with min = max = K. Arm D, DiariZen with min 1, max K.
  4. arm E (--fusion), B's boundaries with labels renamed to A's labels by greedy one-to-one overlap.
     E is built from B, so it also runs A and B. C and D need A's K, so they run A too.
  4b. arm F (--only-arms F, not in the default set), nemotron_windowed: the same two steps as G, but the clustering
     is the constrained AHC of utils/windowed_clustering and the embeddings come from the repo's WeSpeaker ResNet-293
     ONNX model, run on the GPU in the som env (the Nemotron env cannot: onnxruntime needs CUDA 13 there). Silero
     pieces of about --windowed-piece-s seconds with 30 s overlap, 30 s Nemotron windows every --windowed-hop-s
     seconds, per-frame probabilities merged DiariZen style. F needs no K. UNVERIFIED on GPU.
  4c. arm G (--only-arms G, not in the default set), nemotron_windowed_vbx: the same windows as F. Stage 1 (--windowed-python) runs tools/windowed_nemotron_run.py --dump-probs into runs/<arm>/probs.
     Stage 2 (--python, the som env) runs tools/windowed_vbx_stage.py: WeSpeaker ResNet34 embeddings, then VBx
     with the DiariZen PLDA (--plda-dir, else the HF cache) and the per-window Hungarian reassignment.
     --vbx-ahc-threshold, --vbx-fa, --vbx-fb go to stage 2. G needs no K. UNVERIFIED on GPU.
  4d. arm H (--only-arms H, not in the default set), nemotron_one_pass: Nemotron's legacy one-pass path
     (models.nemotron.legacy_one_pass), the whole chunk in one request, no windows and no external stitching. It
     tests whether the model's own speaker cache (AOSC) holds the identities over about 5 minutes. H needs no K.
  4e. arm I (--only-arms I, not in the default set), nemotron_dense_vbx: arm G with DiariZen's window (--dense-window-s,
     16 s) and a hop of --dense-hop-s (0.8 s), so every frame is seen by about 20 windows and VBx gets hundreds of
     embeddings per 5 minutes instead of a few dozen. Same stages, same flags as G. I needs no K. UNVERIFIED on GPU.
  4f. arm J (--only-arms J, not in the default set), nemotron_dz: arm I's windows (16 s, hop 0.8 s) through DiariZen's
     own steps, utils/dz_pipeline.py: binarised and median-filtered slots, a per-frame speaker count, one masked
     embedding of the whole window per slot (the ResNet gets the window and `weights` = the slot's clean frames), the
     10% clean-frame rule, VBx, a Hungarian assignment per window and the top-count reconstruction. Stage 2 should run
     with --vbx-python set to the DiariZen env. --dz-thr and --dz-median-s set the Nemotron binarisation. UNVERIFIED on GPU.
  4g. arm K (--only-arms K, not in the default set), nemotron_dz_demucs: arm J with HT Demucs on the EMBEDDING branch
     only (the probabilities keep the original audio). Stage 1 is not rerun: K reuses arm J's probs folder of the same
     sample (J's stage 1 runs first when its plan.json is missing). Then tools/demucs_vocals.py (--demucs-python, default
     --python) writes chunk_vocals.wav, the vocals stem of chunk.wav; --reuse skips it when the file exists. Stage 2 is
     J's with --embed-audio chunk_vocals.wav. HT Demucs is a MUSIC separator. UNVERIFIED on GPU.
  4h. --fb-sweep "0.2,0.4" (values above 0): for each selected arm J and K, stage 2 only, with --fb <value>, reusing
     the probs (and K's vocals wav), into runs/<arm>_fb<value>. These extra arms (nemotron_dz_fb0.4, nemotron_dz_demucs_fb0.2)
     are listed like the others. The base arms keep --vbx-fb. No P_loop: DiariZen's VBx runs with loopProb 0.
  5. every arm is cropped to the clip window and shifted to 0, written as segments_<arm>.json, and drawn in one
     lanes.html per sample (tools/make_diar_lanes_page.py). index.html links the samples with a table of
     speakers, K, segments under 1 s, speech seconds and overlap seconds per arm. No DER, no text.

Sample span (seconds in the source), in order of use:
  - "start"/"end" when given (end defaults to start + duration);
  - else --final DIR: the conversation folder whose name is <recording>_<folder_suffix>s, with its
    conversation.json source_start/source_end (the export writes them);
  - else the fallback, an assumption and not verified: start = folder_suffix (the second in the folder name)
    and end = start + duration.
The folder_suffix is read from the sample's "folder_suffix". The stem may also carry the suffix, as <stem>_NNNs.

Arm runs use tools/job_diar.build_cmd (main.py --stop_after diarization). CUDA_VISIBLE_DEVICES is set to --gpu.
--reuse skips an arm whose segments.json already exists. A failed arm is recorded and the others still run.
"""
import argparse
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.job_diar import INPUT_NAME, build_cmd  # noqa: E402
from tools.make_diar_lanes_page import build as lanes_build  # noqa: E402
from tools.make_variant_lanes_page import render  # noqa: E402
from utils.nemotron_two_pass import clean_speech_duration  # noqa: E402
from utils.worker_env import resolve_worker_python  # noqa: E402

SR = 16000
CHUNK_S = 300.0         # length of the cut around each clip
LEAD_S = 120.0          # chunk starts this long before the clip
SHORT_S = 1.0           # segments shorter than this are counted as short
AUDIO_EXT = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".mp4", ".webm")
ARMS = {"A": "nemotron", "B": "diarizen_auto", "C": "diarizen_k", "D": "diarizen_1_k", "E": "fusion",
        "F": "nemotron_windowed", "G": "nemotron_windowed_vbx", "H": "nemotron_one_pass",
        "I": "nemotron_dense_vbx", "J": "nemotron_dz", "K": "nemotron_dz_demucs"}
ARM_ORDER = "ABCDEFGHIJK"
DEFAULT_ARMS = "A,B,C,D"
WINDOWED_PIECE_S = 240.0
WINDOWED_HOP_S = 10.0
DENSE_WINDOW_S = 16.0       # arm I: DiariZen's window and a hop of 0.8 s, so every frame is seen by about 20 windows
DENSE_HOP_S = 0.8
WINDOWED_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "windowed_nemotron_run.py")
VBX_STAGE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "windowed_vbx_stage.py")
DEMUCS_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demucs_vocals.py")
VOCALS_NAME = "chunk_vocals.wav"
VBX_ARMS = ("G", "I", "J", "K")     # stage 2 with VBx or DiariZen steps, in the --vbx-python env
DZ_ARMS = ("J", "K")                # stage 2 with DiariZen's own steps (--clustering dz)
DENSE_ARMS = ("I", "J", "K")        # DiariZen-like windows: 16 s, hop 0.8 s
FB_SWEEP_ARMS = ("J", "K")          # the arms the --fb-sweep values are run for
STAGE2_LABEL = {"G": "VBx", "I": "VBx", "J": "DiariZen steps", "K": "DiariZen steps"}
VBX_AHC_THRESHOLD = 0.6     # UNVERIFIED: not checked against the DiariZen config
VBX_FA = 0.07               # UNVERIFIED
VBX_FB = 0.8                # UNVERIFIED
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diar_arms_samples.json")

Runner = Callable[[List[str], dict], int]


def safe_stem(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def ascii_slug(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Za-z0-9]+", "_", folded).strip("_") or "sample"


def split_folder_suffix(stem: str) -> Tuple[str, Optional[int]]:
    """('rec', 576) for 'rec_576s' (and 'rec_576s_b'); (stem, None) when there is no suffix."""
    m = re.fullmatch(r"(.+?)_(\d+)s(_b)*", stem)
    return (m.group(1), int(m.group(2))) if m else (stem, None)


def parse_mmss(value) -> Optional[float]:
    """'0:45' -> 45.0, '1:02:03' -> 3723.0, a number stays a number."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    secs = 0.0
    for part in str(value).strip().split(":"):
        secs = secs * 60 + float(part)
    return secs


def find_source(audio_dir: Optional[str], stem: str) -> str:
    """The one audio file under audio_dir whose stem is `stem` or starts with it (sanitised names compared)."""
    if not audio_dir:
        raise ValueError(f"no audio path for {stem!r}: pass --audio-dir or an explicit \"audio\"")
    want = safe_stem(split_folder_suffix(stem)[0])
    exact, prefix = [], []
    for cur, dirs, files in os.walk(audio_dir):
        dirs.sort()
        for name in sorted(files):
            base, ext = os.path.splitext(name)
            if ext.lower() not in AUDIO_EXT:
                continue
            s = safe_stem(base)
            if s == want:
                exact.append(os.path.join(cur, name))
            elif re.match(re.escape(want) + r"[^A-Za-z0-9]", s):
                prefix.append(os.path.join(cur, name))
    hits = exact or prefix
    if not hits:
        raise FileNotFoundError(f"no audio under {audio_dir} for stem {stem!r}")
    if len(hits) > 1:
        raise ValueError(f"stem {stem!r} matches several files: {hits}")
    return hits[0]


def probe_duration(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", path],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def ffmpeg_cut(src: str, start: float, dur: float, dst: str) -> None:
    """Write [start, start + dur) of src as 16 kHz mono PCM WAV."""
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{start:.3f}",
                    "-t", f"{dur:.3f}", "-i", src, "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", dst],
                   check=True)


def chunk_window(clip_start: float, clip_end: float, total: Optional[float] = None) -> Tuple[float, float, float, float]:
    """(chunk_start, chunk_end, clip_start_in_chunk, clip_end_in_chunk), in source seconds.

    The chunk starts LEAD_S before the clip and lasts CHUNK_S. Near the end of the file it moves back so it
    still lasts CHUNK_S, and it never starts before 0.
    """
    if clip_end <= clip_start or clip_start < 0:
        raise ValueError(f"bad clip window {clip_start}-{clip_end}")
    if total is not None and clip_end > total:
        raise ValueError(f"clip ends at {clip_end:.1f} s, the source is {total:.1f} s long")
    start = max(0.0, clip_start - LEAD_S)
    if total is not None:
        start = min(start, max(0.0, total - CHUNK_S))
    end = start + CHUNK_S
    if total is not None:
        end = min(end, total)
    if clip_end > end:
        raise ValueError(f"clip {clip_start}-{clip_end} does not fit in the chunk {start}-{end}")
    return start, end, clip_start - start, clip_end - start


def _lookup_final(final: str, prefix: str, suffix: int) -> Optional[Tuple[float, float]]:
    """(source_start, source_end) of the conversation folder <recording>_<suffix>s under final, if one matches."""
    want = safe_stem(prefix)
    spans = set()
    for cur, _dirs, files in os.walk(final):
        if "conversation.json" not in files:
            continue
        rec, n = split_folder_suffix(os.path.basename(cur))
        if n != suffix or not safe_stem(rec).startswith(want):
            continue
        try:
            with open(os.path.join(cur, "conversation.json"), encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            continue
        s, e = meta.get("source_start"), meta.get("source_end")
        if isinstance(s, (int, float)) and isinstance(e, (int, float)):
            spans.add((float(s), float(e)))
    if len(spans) > 1:
        raise ValueError(f"{prefix}_{suffix}s matches {len(spans)} conversations under {final}")
    return spans.pop() if spans else None


def resolve_span(sample: dict, final: Optional[str]) -> Tuple[float, float, str]:
    """(start, end, how) in source seconds; how is 'sample', 'final' or 'suffix'."""
    prefix, suffix = split_folder_suffix(str(sample.get("stem") or ""))
    if sample.get("folder_suffix") is not None:
        suffix = int(sample["folder_suffix"])
    start, end = sample.get("start"), sample.get("end")
    found = None
    if final and suffix is not None and (start is None or end is None):
        found = _lookup_final(final, prefix, suffix)
    if start is not None:
        how = "sample"
    elif found:
        start, how = found[0], "final"
    elif suffix is not None:
        start, how = float(suffix), "suffix"
    else:
        raise ValueError(f"{sample.get('name')}: no start, no --final match and no folder_suffix")
    if end is None:
        if found:
            end = found[1]
        else:
            dur = parse_mmss(sample.get("duration"))
            if dur is None:
                raise ValueError(f"{sample.get('name')}: no end and no duration")
            end = float(start) + dur
    return float(start), float(end), how


def read_rows(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    rows = data.get("segments", data) if isinstance(data, dict) else data
    out = []
    for r in rows or []:
        try:
            s, e = float(r["start"]), float(r["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if e > s:
            out.append({"start": s, "end": e, "speaker": str(r.get("speaker", "?"))})
    return out


def crop(segments: Sequence[dict], lo: float, hi: float) -> List[dict]:
    """Segments clipped to [lo, hi) and shifted so lo is 0."""
    out = []
    for r in segments:
        s, e = max(r["start"], lo), min(r["end"], hi)
        if e > s:
            out.append({"start": round(s - lo, 3), "end": round(e - lo, 3), "speaker": r["speaker"]})
    return sorted(out, key=lambda r: (r["start"], r["end"]))


def estimate_k(segments: Sequence[dict], k_min_clean: float) -> int:
    labels = {r["speaker"] for r in segments}
    count = sum(1 for lab in labels if clean_speech_duration(list(segments), lab) >= k_min_clean)
    return max(1, count)


def fuse(boundaries: Sequence[dict], labels: Sequence[dict]) -> List[dict]:
    """Boundaries of one run with the speaker names of another: greedy one-to-one by overlap seconds.

    A boundary label that gets no partner keeps its identity as 'x_<label>', so it never merges into a name.
    """
    pair: Dict[Tuple[str, str], float] = {}
    for a in boundaries:
        for b in labels:
            ov = min(a["end"], b["end"]) - max(a["start"], b["start"])
            if ov > 0:
                key = (a["speaker"], b["speaker"])
                pair[key] = pair.get(key, 0.0) + ov
    mapping: Dict[str, str] = {}
    used = set()
    for (dz, name), _ov in sorted(pair.items(), key=lambda kv: (-kv[1], kv[0])):
        if dz in mapping or name in used:
            continue
        mapping[dz] = name
        used.add(name)
    return [dict(r, speaker=mapping.get(r["speaker"], f"x_{r['speaker']}")) for r in boundaries]


def _union(intervals: Sequence[Tuple[float, float]]) -> float:
    total, cur = 0.0, None
    for s, e in sorted(intervals):
        if cur is None or s > cur[1]:
            if cur is not None:
                total += cur[1] - cur[0]
            cur = [s, e]
        else:
            cur[1] = max(cur[1], e)
    if cur is not None:
        total += cur[1] - cur[0]
    return total


def stats(segments: Sequence[dict]) -> dict:
    overlap = []
    for i, a in enumerate(segments):
        for b in segments[i + 1:]:
            if a["speaker"] != b["speaker"]:
                s, e = max(a["start"], b["start"]), min(a["end"], b["end"])
                if e > s:
                    overlap.append((s, e))
    return {
        "speakers": len({r["speaker"] for r in segments}),
        "segments": len(segments),
        "short": sum(1 for r in segments if r["end"] - r["start"] < SHORT_S),
        "speech_s": round(_union([(r["start"], r["end"]) for r in segments]), 2),
        "overlap_s": round(_union(overlap), 2),
    }


def arm_settings(arm: str, k: Optional[int]) -> Tuple[str, dict]:
    """(backend, --set values) of one run arm."""
    if arm == "A":
        return "nemotron", {}
    if arm == "H":      # one Nemotron request over the whole chunk: its own speaker cache keeps the identities
        return "nemotron", {"models.nemotron.legacy_one_pass": True}
    if arm == "B":
        return "diarizen", {}
    if k is None:
        raise RuntimeError("no K: arm A failed or was not run")
    if arm == "C":
        return "diarizen", {"models.diarizen.min_speakers": k, "models.diarizen.max_speakers": k}
    return "diarizen", {"models.diarizen.min_speakers": 1, "models.diarizen.max_speakers": k}


def run_main(cmd: List[str], env: dict) -> int:
    # No terminal on stdin: main.py then takes its folders from config instead of asking at a prompt.
    return subprocess.run(cmd, cwd=ROOT, env=env, check=False, stdin=subprocess.DEVNULL).returncode


def run_arm(sample_dir: Path, arm: str, values: dict, backend: str, cfg: dict, runner: Runner) -> Path:
    """Run main.py for one arm on sample_dir/chunk.wav; return its segments.json."""
    run_dir = sample_dir / "runs" / ARMS[arm]
    result = run_dir / "out" / Path(INPUT_NAME).stem / "02_diarization" / "segments.json"
    if cfg["reuse"] and result.is_file():
        return result
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(sample_dir / "chunk.wav", run_dir / INPUT_NAME)
    ctx = {"python": cfg["python"], "env_name": cfg["env"], "config_path": cfg["config"], "pipeline_dir": ROOT}
    params = dict(values, **{"models.diarization_backend": backend})
    cmd = build_cmd(run_dir, {"params": params}, ctx) + ["--diarization_backend", backend]
    code = runner(cmd, dict(os.environ, CUDA_VISIBLE_DEVICES=str(cfg["gpu"])))
    if code != 0 or not result.is_file():
        raise RuntimeError(f"arm {ARMS[arm]} failed (exit {code}), see {run_dir}")
    return result


def arm_key(letter: str, fb: Optional[str] = None) -> str:
    """'J' for the base arm, 'J@0.4' for its --fb-sweep arm with Fb 0.4 (the text as printed in the arm name)."""
    return letter if fb is None else f"{letter}@{fb}"


def split_key(key: str) -> Tuple[str, Optional[str]]:
    letter, _, fb = key.partition("@")
    return letter, (fb or None)


def arm_name(key: str) -> str:
    """The arm's name in the output: nemotron_dz, or nemotron_dz_fb0.4 for a sweep arm."""
    letter, fb = split_key(key)
    return ARMS[letter] if fb is None else f"{ARMS[letter]}_fb{fb}"


def parse_fb_sweep(text: Optional[str]) -> List[str]:
    """'0.2,0.4' -> ['0.2', '0.4']: each value above 0, formatted with %g as it appears in the arm name."""
    if text is None:
        return []
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"empty value in --fb-sweep {text!r}")
        try:
            value = float(part)
        except ValueError:
            raise ValueError(f"--fb-sweep value {part!r} is not a number")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--fb-sweep values must be above 0, got {part!r}")
        out.append("%g" % value)
    if len(set(out)) != len(out):
        raise ValueError(f"--fb-sweep repeats a value: {text!r}")
    return out


def gpu_env(cfg: dict) -> dict:
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(cfg["gpu"]))


def dump_probs(arm: str, sample_dir: Path, cfg: dict, runner: Runner, skip_existing: bool) -> Path:
    """Stage 1 of a windowed arm (Nemotron env): the per-window probabilities in runs/<arm>/probs.

    With skip_existing and a plan.json already there, nothing runs.
    """
    run_dir = sample_dir / "runs" / ARMS[arm]
    probs = run_dir / "probs"
    if skip_existing and (probs / "plan.json").is_file():
        return probs
    run_dir.mkdir(parents=True, exist_ok=True)
    dense = arm in DENSE_ARMS
    window_s = cfg["dense_window_s"] if dense else None
    hop_s = cfg["dense_hop_s"] if dense else cfg["windowed_hop_s"]
    cmd1 = [cfg["windowed_python"], WINDOWED_SCRIPT, "--audio", str(sample_dir / "chunk.wav"),
            "--out", str(run_dir / "unused.json"), "--dump-probs", str(probs),
            "--piece-s", str(cfg["windowed_piece_s"]), "--hop-s", str(hop_s),
            "--config", cfg["config"], "--env", cfg["env"]]
    if window_s is not None:
        cmd1 += ["--window-s", str(window_s)]
    code = runner(cmd1, gpu_env(cfg))
    if code != 0 or not (probs / "plan.json").is_file():
        raise RuntimeError(f"arm {ARMS[arm]} stage 1 (probabilities) failed (exit {code}), see {run_dir}")
    return probs


def ensure_vocals(sample_dir: Path, cfg: dict, runner: Runner, skip_existing: bool) -> Path:
    """Arm K's embedding input: chunk_vocals.wav, the HT Demucs vocals stem of chunk.wav (tools/demucs_vocals.py)."""
    wav = sample_dir / VOCALS_NAME
    if skip_existing and wav.is_file():
        return wav
    cmd = [cfg["demucs_python"], DEMUCS_SCRIPT, "--in", str(sample_dir / "chunk.wav"), "--out", str(wav)]
    code = runner(cmd, gpu_env(cfg))
    if code != 0 or not wav.is_file():
        raise RuntimeError(f"arm {ARMS['K']} demucs step failed (exit {code}), see {wav}")
    return wav


def stage2_cmd(arm: str, sample_dir: Path, probs: Path, result: Path, cfg: dict, fb: str,
               embed_audio: Optional[Path] = None) -> List[str]:
    """The stage 2 command of a windowed arm. Without embed_audio it is the same command as before arm K existed."""
    python = (cfg.get("vbx_python") or cfg["python"]) if arm in VBX_ARMS else cfg["python"]
    cmd = [python, VBX_STAGE_SCRIPT, "--audio", str(sample_dir / "chunk.wav"), "--probs-dir", str(probs),
           "--out", str(result), "--min-speakers", str(cfg["windowed_min_speakers"])]
    if arm in VBX_ARMS:
        if arm in DZ_ARMS:
            cmd += ["--clustering", "dz", "--thr", str(cfg["dz_thr"]), "--median-s", str(cfg["dz_median_s"])]
        cmd += ["--ahc-threshold", str(cfg["vbx_ahc_threshold"]), "--fa", str(cfg["vbx_fa"]), "--fb", fb]
        if cfg["plda_dir"]:
            cmd += ["--plda-dir", str(cfg["plda_dir"])]
    else:
        cmd += ["--clustering", "ahc", "--embedder", "wespeaker"]
    if embed_audio is not None:
        cmd += ["--embed-audio", str(embed_audio)]
    return cmd


def run_windowed_arm(arm: str, sample_dir: Path, cfg: dict, runner: Runner) -> Path:
    """Arms F, G, I, J and K on sample_dir/chunk.wav, in two steps.

    Stage 1 (Nemotron env) dumps the per-window probabilities. Stage 2 (--python, the som env) embeds on the GPU
    and clusters: arm F with the repo's WeSpeaker ResNet-293 and constrained AHC, arm G with pyannote's ResNet34
    and DiariZen's VBx. The Nemotron env cannot run ONNX Runtime on the GPU (it needs CUDA 13), so F no longer
    embeds there. Arm K has no stage 1 of its own: it reuses arm J's probs, runs Demucs for the embedding input,
    and then stage 2 with --embed-audio.
    """
    run_dir = sample_dir / "runs" / ARMS[arm]
    result = run_dir / "out" / "segments.json"
    if cfg["reuse"] and result.is_file():
        return result
    run_dir.mkdir(parents=True, exist_ok=True)
    if arm == "K":
        probs = dump_probs("J", sample_dir, cfg, runner, skip_existing=True)
        embed_audio = ensure_vocals(sample_dir, cfg, runner, skip_existing=cfg["reuse"])
    else:
        probs = dump_probs(arm, sample_dir, cfg, runner, skip_existing=cfg["reuse"])
        embed_audio = None
    cmd2 = stage2_cmd(arm, sample_dir, probs, result, cfg, fb=str(cfg["vbx_fb"]), embed_audio=embed_audio)
    code = runner(cmd2, gpu_env(cfg))
    if code != 0 or not result.is_file():
        raise RuntimeError(f"arm {ARMS[arm]} stage 2 ({STAGE2_LABEL.get(arm, 'AHC')}) failed (exit {code}), see {run_dir}")
    return result


def run_fb_arm(arm: str, fb: str, sample_dir: Path, cfg: dict, runner: Runner) -> Path:
    """A --fb-sweep arm: stage 2 only, with --fb <fb>, on arm J's probs (and K's vocals). Never reruns stage 1 or Demucs."""
    name = arm_name(arm_key(arm, fb))
    run_dir = sample_dir / "runs" / name
    result = run_dir / "out" / "segments.json"
    if cfg["reuse"] and result.is_file():
        return result
    run_dir.mkdir(parents=True, exist_ok=True)
    probs = dump_probs("J", sample_dir, cfg, runner, skip_existing=True)
    embed_audio = ensure_vocals(sample_dir, cfg, runner, skip_existing=True) if arm == "K" else None
    cmd2 = stage2_cmd(arm, sample_dir, probs, result, cfg, fb=fb, embed_audio=embed_audio)
    code = runner(cmd2, gpu_env(cfg))
    if code != 0 or not result.is_file():
        raise RuntimeError(f"arm {name} stage 2 ({STAGE2_LABEL[arm]}) failed (exit {code}), see {run_dir}")
    return result


def process_sample(index: int, sample: dict, cfg: dict, runner: Runner, log: Callable[[str], None] = print) -> dict:
    name = str(sample["name"])
    sdir = Path(cfg["out"]) / f"{index:02d}_{ascii_slug(name)}"
    sdir.mkdir(parents=True, exist_ok=True)
    src = sample.get("audio") or find_source(cfg["audio_dir"], str(sample["stem"]))
    start, end, how = resolve_span(sample, cfg["final"])
    cs, ce, rs, re_ = chunk_window(start, end, probe_duration(src))
    chunk, clip = sdir / "chunk.wav", sdir / "clip.wav"
    if not (cfg["reuse"] and chunk.is_file()):
        ffmpeg_cut(src, cs, ce - cs, str(chunk))
    ffmpeg_cut(str(chunk), rs, re_ - rs, str(clip))
    log(f"{name}: source {os.path.basename(src)} clip {start:.2f}-{end:.2f} ({how}) chunk {cs:.1f}-{ce:.1f}")

    segs: Dict[str, List[dict]] = {}
    errors: Dict[str, str] = {}
    k: Optional[int] = None
    for key in cfg["keys"]:              # base arms in letter order, each followed by its --fb-sweep arms
        arm, fb = split_key(key)
        try:
            if arm == "E":
                if "A" not in segs or "B" not in segs:
                    raise RuntimeError("fusion needs arms A and B")
                segs[key] = fuse(segs["B"], segs["A"])
                continue
            if arm == "F":                      # windowed Nemotron: no K needed
                segs[key] = crop(read_rows(str(run_windowed_arm("F", sdir, cfg, runner))), rs, re_)
                continue
            if arm in VBX_ARMS:                 # windowed Nemotron + VBx (G sparse, I dense), DiariZen steps (J, K)
                path = run_fb_arm(arm, fb, sdir, cfg, runner) if fb else run_windowed_arm(arm, sdir, cfg, runner)
                segs[key] = crop(read_rows(str(path)), rs, re_)
                continue
            backend, values = arm_settings(arm, k)
            path = run_arm(sdir, arm, values, backend, cfg, runner)
            segs[key] = crop(read_rows(str(path)), rs, re_)
            if arm == "A":
                k = estimate_k(segs["A"], cfg["k_min_clean"])
        except Exception as exc:  # the other arms still run
            errors[key] = str(exc)
            log(f"  arm {arm_name(key)}: {exc}")
    if k is not None:
        log(f"  K = {k} (from A, k_min_clean {cfg['k_min_clean']})")

    for key in cfg["keys"]:
        if key in segs:
            with open(sdir / f"segments_{arm_name(key)}.json", "w", encoding="utf-8") as fh:
                json.dump({"segments": segs[key]}, fh, ensure_ascii=False, indent=1)
    lanes = None
    diars = [(arm_name(a), str(sdir / f"segments_{arm_name(a)}.json")) for a in cfg["keys"] if a in segs]
    if diars:
        data = lanes_build(str(clip), diars, str(sdir / "lanes.html"), audio_url="clip.wav")
        if data["variants"][0]["peaks"] is None:
            raise RuntimeError(f"cannot read {clip}")
        data["file"] = name
        with open(sdir / "lanes.html", "w", encoding="utf-8") as fh:
            fh.write(render(data))
        lanes = f"{sdir.name}/lanes.html"
    return {"name": name, "slug": sdir.name, "kind": sample.get("kind"), "turns": sample.get("turns"),
            "span": [start, end, how], "k": k, "arms": {a: stats(segs[a]) for a in segs},
            "errors": errors, "lanes": lanes}


def _cell(value) -> str:
    return html.escape(str(value))


def write_index(out: Path, summaries: List[dict], arms: Sequence[str]) -> Path:
    rows = []
    for s in summaries:
        name = f'<a href="{_cell(s["lanes"])}">{_cell(s["name"])}</a>' if s.get("lanes") else _cell(s["name"])
        if "error" in s:
            rows.append(f'<tr><td>{name}</td><td colspan="8">error: {_cell(s["error"])}</td></tr>')
            continue
        for arm in arms:
            if arm in s["arms"]:
                st = s["arms"][arm]
                tail = f'<td>{s["k"]}</td><td>{st["speakers"]}</td><td>{st["short"]}</td>' \
                       f'<td>{st["speech_s"]:.2f}</td><td>{st["overlap_s"]:.2f}</td><td>ok</td>'
            else:
                tail = f'<td>{s["k"] if s["k"] is not None else "-"}</td><td colspan="4">-</td>' \
                       f'<td>{_cell(s["errors"].get(arm, "not run"))}</td>'
            rows.append(f"<tr><td>{name}</td><td>{split_key(arm)[0]} {arm_name(arm)}</td>{tail}</tr>")
    body = "\n".join(rows)
    page = f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Diarization arms</title>
<style>
:root{{--bg:#fbfaf7;--ink:#1d1d1b;--line:#e4e1da}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#161614;--ink:#ece9e2;--line:#3a3834}}}}
:root[data-theme="dark"]{{--bg:#161614;--ink:#ece9e2;--line:#3a3834}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--ink);font:14px/1.4 system-ui,sans-serif}}
table{{border-collapse:collapse;width:100%}}
td,th{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}}
a{{color:inherit}}
</style></head><body>
<h1 style="font-size:18px">Diarization arms: {len(summaries)} samples</h1>
<table><thead><tr><th>sample</th><th>arm</th><th>K</th><th>speakers</th><th>segments &lt; 1 s</th>
<th>speech s</th><th>overlap s</th><th>status</th></tr></thead>
<tbody>
{body}
</tbody></table>
</body></html>
"""
    path = out / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def load_samples(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        samples = json.load(fh)
    if not isinstance(samples, list):
        raise ValueError(f"{path} must hold a JSON list")
    for s in samples:
        if not s.get("name") or not (s.get("stem") or s.get("audio")):
            raise ValueError(f"each sample needs a name and a stem (or audio): {s}")
    return samples


def main(argv: Optional[Sequence[str]] = None, runner: Optional[Runner] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-dir", help="folder with the source recordings (searched by stem)")
    ap.add_argument("--samples", required=True, help="JSON list of samples (see tools/diar_arms_samples.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--final", help="export folder, to look up each conversation's source_start/source_end")
    ap.add_argument("--gpu", default="0", help="physical GPU id, set as CUDA_VISIBLE_DEVICES")
    ap.add_argument("--env", default="a100", help="environment profile in config.json")
    ap.add_argument("--only-arms", default=DEFAULT_ARMS,
                    help="comma list of A,B,C,D,F,G,H,I,J,K (E needs --fusion; F is the windowed Nemotron arm, "
                         "G the same windows with VBx clustering, H Nemotron in one pass over the chunk, "
                         "I dense windows like DiariZen with VBx, J DiariZen's own pipeline steps, K J with "
                         "Demucs vocals for the embeddings)")
    ap.add_argument("--fusion", action="store_true", help="also build arm E")
    ap.add_argument("--k-min-clean", type=float, default=1.0, help="clean seconds for a speaker to count toward K")
    ap.add_argument("--reuse", action="store_true", help="skip arms whose segments.json already exists")
    ap.add_argument("--config", default=os.path.join(ROOT, "config.json"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--windowed-piece-s", type=float, default=WINDOWED_PIECE_S,
                    help="arm F piece length (min and max are 0.75x and 1.25x of it)")
    ap.add_argument("--windowed-hop-s", type=float, default=WINDOWED_HOP_S, help="arm F window hop (window is 30 s)")
    ap.add_argument("--windowed-python",
                    help="interpreter of the Nemotron env for arms F and G (default: NEMOTRON_PYTHON or worker_envs)")
    ap.add_argument("--vbx-ahc-threshold", type=float, default=VBX_AHC_THRESHOLD,
                    help="arm G: euclidean cut of the AHC init before VBx (UNVERIFIED default)")
    ap.add_argument("--dense-window-s", type=float, default=DENSE_WINDOW_S,
                    help="arm I: Nemotron window length (DiariZen uses 16 s)")
    ap.add_argument("--dense-hop-s", type=float, default=DENSE_HOP_S,
                    help="arm I: hop between windows; 0.8 s gives hundreds of embeddings per 5 minutes like DiariZen")
    ap.add_argument("--vbx-python",
                    help="python of stage 2 for the VBx arms G, I and J; the DiariZen env gives arm J DiariZen's own "
                         "pyannote 3.1.1 (default: --python)")
    ap.add_argument("--demucs-python",
                    help="arm K: python with the demucs package (HT Demucs, tools/demucs_vocals.py) (default: --python)")
    ap.add_argument("--fb-sweep",
                    help="comma list of VBx Fb values above 0 (e.g. 0.2,0.4): for each selected arm J and K, run "
                         "stage 2 only with that --fb, into runs/<arm>_fb<value>")
    ap.add_argument("--dz-thr", type=float, default=0.5, help="arm J: slot probability threshold")
    ap.add_argument("--dz-median-s", type=float, default=0.22,
                    help="arm J: median filter of the binarised slots in seconds (DiariZen: 11 frames of about 20 ms); "
                         "Nemotron's output is 10 ms per frame")
    ap.add_argument("--windowed-min-speakers", type=int, default=1,
                    help="arms F and G: at least this many speakers per piece (G falls back to AHC below it)")
    ap.add_argument("--vbx-fa", type=float, default=VBX_FA, help="arm G: VBx Fa (UNVERIFIED default)")
    ap.add_argument("--vbx-fb", type=float, default=VBX_FB, help="arm G: VBx Fb (UNVERIFIED default)")
    ap.add_argument("--plda-dir", help="arm G: folder with xvec_transform.npz and plda.npz (default: HF cache)")
    a = ap.parse_args(argv)

    letters = {x.strip().upper() for x in a.only_arms.split(",") if x.strip()}
    if letters - set(ARM_ORDER):
        ap.error(f"unknown arms in --only-arms: {sorted(letters - set(ARM_ORDER))}")
    try:
        fb_values = parse_fb_sweep(a.fb_sweep)
    except ValueError as exc:
        ap.error(str(exc))
    if a.fb_sweep is not None and not letters & set(FB_SWEEP_ARMS):
        ap.error("--fb-sweep needs arm J or K in --only-arms")
    if a.fusion:
        letters.add("E")
    run = set(letters)
    if run & {"C", "D", "E"}:
        run.add("A")            # K comes from arm A
    if "E" in run:
        run.add("B")            # fusion boundaries come from arm B
    keys = []                   # base arms in letter order, each followed by its --fb-sweep arms
    for letter in ARM_ORDER:
        if letter in run:
            keys.append(arm_key(letter))
            if letter in FB_SWEEP_ARMS:
                keys += [arm_key(letter, fb) for fb in fb_values]
    windowed_python = None
    if run & {"F", "G", "I", "J", "K"}:
        windowed_python = a.windowed_python or resolve_worker_python("nemotron")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"audio_dir": a.audio_dir, "final": a.final, "out": str(out), "gpu": a.gpu, "env": a.env,
           "run": run, "keys": keys, "k_min_clean": a.k_min_clean, "reuse": a.reuse, "config": a.config,
           "python": a.python, "demucs_python": a.demucs_python or a.python,
           "windowed_python": windowed_python, "windowed_piece_s": a.windowed_piece_s,
           "windowed_hop_s": a.windowed_hop_s, "dense_window_s": a.dense_window_s, "dense_hop_s": a.dense_hop_s,
           "vbx_python": a.vbx_python, "dz_thr": a.dz_thr, "dz_median_s": a.dz_median_s,
           "windowed_min_speakers": a.windowed_min_speakers, "vbx_ahc_threshold": a.vbx_ahc_threshold, "vbx_fa": a.vbx_fa,
           "vbx_fb": a.vbx_fb, "plda_dir": a.plda_dir}
    summaries = []
    for i, sample in enumerate(load_samples(a.samples), 1):
        try:
            summary = process_sample(i, sample, cfg, runner or run_main)
        except Exception as exc:
            summary = {"name": str(sample.get("name")), "error": str(exc), "lanes": None, "arms": {}, "k": None,
                       "errors": {}}
            print(f"{sample.get('name')}: error: {exc}", file=sys.stderr)
        summaries.append(summary)
    index = write_index(out, summaries, keys)
    print(f"-> {index}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
