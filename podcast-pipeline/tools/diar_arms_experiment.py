#!/usr/bin/env python3
"""Diarization arms on fixed conversation samples, shown as waveform lanes (one row per speaker, no text).

    python tools/diar_arms_experiment.py --audio-dir SRC --samples tools/diar_arms_samples.json --out OUT/diar_arms \
        [--final OUT/final] [--gpu 6] [--env a100] [--only-arms A,C,F,G] [--fusion] [--k-min-clean 1.0] [--reuse] \
        [--windowed-piece-s 240] [--windowed-hop-s 10] [--windowed-python /path/to/nemotron/python] \
        [--python /path/to/som/python] [--vbx-ahc-threshold 0.6] [--vbx-fa 0.07] [--vbx-fb 0.8] [--plda-dir DIR] \
        [--vbx-python /path/to/dz/python] [--demucs-python /path/to/demucs/python] [--fb-sweep 0.2,0.4] \
        [--overlap-variants [--ov-thr 0.35] [--ov-min-on-s 0.2]] [--embedder-trials r34lm,espnet_spk \
        [--cos-thr 0.4,0.5,0.6] [--embedder-python NAME=PATH]]

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
  4i. --overlap-variants (experiment): for each selected arm J and K, stage 2 only, with --thr --ov-thr (default 0.35)
     and --min-on-s --ov-min-on-s (default 0.2), into runs/nemotron_dz_ov_thr0.35_on0.2 and
     runs/nemotron_dz_demucs_ov_thr0.35_on0.2. Same Fb, median filter and probs as the base arm; K's vocals for the embeddings. Independent of --fb-sweep: the sweep arms
     keep the base threshold. The base J and K commands do not change (no --min-on-s). UNVERIFIED on GPU.
     --ov-thr and --ov-min-on-s take comma lists: every combination is one arm per J and K, named
     nemotron_dz_ov_thr<thr>_on<min_on> (e.g. nemotron_dz_ov_thr0.35_on0.2, nemotron_dz_demucs_ov_thr0.3_on0). At most
     OV_MAX_COMBOS (24) combinations per base arm; repeats after %g formatting and values outside (0, 1) / below 0 are refused.
  4j. --embedder-trials LIST (experiment, off by default): for each sample and each kind in LIST (r34lm, wespeaker_r293,
     wespeaker_simam100, espnet_spk) and each cosine cut of --cos-thr (default 0.4,0.5,0.6), stage 2 only on J's probs and
     the chunk, with --clustering dz-ahc (utils/dz_pipeline.make_dz_ahc_fn: average-linkage cosine AHC, not VBx, because
     DiariZen's PLDA only fits ResNet34-LM). Into runs/nemotron_dz_emb_<kind>_c<cos>. The embeddings of a sample are
     cached in <sample>/embcache, so the other cos cuts make no model call. --embedder-python NAME=PATH picks the python
     of one kind (default --vbx-python, then --python); --r293-onnx, --simam-onnx or --simam-dir, --espnet-tag pass
     through. A failing embedder is logged, its arms are listed as errors, the other arms and samples still run, and a
     summary of the failed (sample, embedder) pairs is printed. The SimAM and ESPnet adapters are UNVERIFIED and NOT run.
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

from tools import embedders as EMB  # noqa: E402
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
OV_ARMS = ("J", "K")                # the arms --overlap-variants adds one arm per grid point for
OV_THR = 0.35                       # first guess, not validated: a lower slot threshold keeps more overlap
OV_MIN_ON_S = 0.2                   # stage 2 drops ON runs shorter than this (--min-on-s)
OV_MAX_COMBOS = 24                  # --ov-thr x --ov-min-on-s combinations per base arm, at most
TRIAL_COS = (0.4, 0.5, 0.6)         # --cos-thr default: cosine AHC cuts of the embedder trials
TRIAL_CACHE_DIR = "embcache"        # per sample: shared by the trial arms, so another cos threshold reads the embeddings
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


def fmt(value: float) -> str:
    """A number as it appears in a key and an arm name: %g, so 0.35 -> '0.35', 0.2 -> '0.2', 0 -> '0'."""
    return "%g" % value


def ov_key(letter: str, thr: float, min_on_s: float) -> str:
    """'J+ov|0.35|0.2' for the --overlap-variants arm of J with --ov-thr 0.35 and --ov-min-on-s 0.2."""
    return f"{letter}+ov|{fmt(thr)}|{fmt(min_on_s)}"


def is_ov(key: str) -> bool:
    return "+ov|" in key


def ov_params(key: str) -> Tuple[float, float]:
    """(thr, min_on_s) of an overlap variant key: 'J+ov|0.35|0.2' -> (0.35, 0.2)."""
    _, _, rest = key.partition("+ov|")
    thr, _, on = rest.partition("|")
    return float(thr), float(on)


def trial_key(kind: str, cos: float, letter: str = "J") -> str:
    """'J+emb|r34lm|0.5' for the embedder trial of kind at cosine cut 0.5 (stage 2 on J's probs)."""
    return f"{letter}+emb|{kind}|{fmt(cos)}"


def is_trial(key: str) -> bool:
    return "+emb|" in key


def trial_params(key: str) -> Tuple[str, float]:
    """(kind, cos) of an embedder trial key: 'J+emb|r34lm|0.5' -> ('r34lm', 0.5)."""
    _, _, rest = key.partition("+emb|")
    kind, _, cos = rest.partition("|")
    return kind, float(cos)


def split_key(key: str) -> Tuple[str, Optional[str]]:
    """(letter, fb). An overlap variant or a trial gives its base letter with fb None (check is_ov and is_trial)."""
    letter, _, fb = key.split("+", 1)[0].partition("@")
    return letter, (fb or None)


def arm_name(key: str) -> str:
    """The arm's name in the output, from the key.

    nemotron_dz (base J), nemotron_dz_fb0.4 (a --fb-sweep arm), nemotron_dz_ov_thr0.35_on0.2 (an overlap variant),
    nemotron_dz_emb_r34lm_c0.5 (an embedder trial).
    """
    letter, fb = split_key(key)
    if is_ov(key):
        thr, min_on_s = ov_params(key)
        return f"{ARMS[letter]}_ov_thr{fmt(thr)}_on{fmt(min_on_s)}"
    if is_trial(key):
        kind, cos = trial_params(key)
        return f"{ARMS[letter]}_emb_{kind}_c{fmt(cos)}"
    return ARMS[letter] if fb is None else f"{ARMS[letter]}_fb{fb}"


def parse_grid(text: Optional[str], flag: str, valid: Callable[[float], bool], rule: str) -> List[float]:
    """'0.3,0.35' -> [0.3, 0.35]: a comma list of numbers that all pass valid, each kept as %g (the value the key and
    the stage 2 command use). A repeat after that formatting is refused, as is an empty list."""
    if text is None or not str(text).strip():
        raise ValueError(f"{flag} needs at least one value")
    out: List[float] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"empty value in {flag} {text!r}")
        try:
            value = float(part)
        except ValueError:
            raise ValueError(f"{flag} value {part!r} is not a number")
        if not (math.isfinite(value) and valid(value)):
            raise ValueError(f"{flag} values {rule}, got {part!r}")
        out.append(float(fmt(value)))
    if len(set(fmt(v) for v in out)) != len(out):
        raise ValueError(f"{flag} repeats a value: {text!r}")
    return out


def ov_grid(thrs: Sequence[float], min_ons: Sequence[float]) -> List[Tuple[float, float]]:
    """Every (thr, min_on_s) pair, thr outer. More than OV_MAX_COMBOS pairs is refused: each one is a stage 2 run."""
    combos = [(t, m) for t in thrs for m in min_ons]
    if len(combos) > OV_MAX_COMBOS:
        raise ValueError(f"--ov-thr x --ov-min-on-s gives {len(combos)} combinations per arm, at most {OV_MAX_COMBOS}")
    return combos


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


def parse_kinds(text: Optional[str]) -> List[str]:
    """'r34lm,espnet_spk' -> ['r34lm', 'espnet_spk']; '' -> [] (trials off). Unknown or repeated kinds are refused."""
    if not text or not text.strip():
        return []
    out = [part.strip() for part in text.split(",")]
    for kind in out:
        if kind not in EMB.KINDS:
            raise ValueError(f"--embedder-trials: unknown embedder {kind!r}, one of {', '.join(EMB.KINDS)}")
    if len(set(out)) != len(out):
        raise ValueError(f"--embedder-trials repeats a kind: {text!r}")
    return out


def parse_embedder_python(items: Sequence[str], kinds: Sequence[str]) -> Dict[str, str]:
    """['r34lm=/x/python'] -> {'r34lm': '/x/python'}. The kind must be one of the trials; a repeat is refused."""
    out: Dict[str, str] = {}
    for item in items or []:
        kind, sep, path = str(item).partition("=")
        kind, path = kind.strip(), path.strip()
        if not sep or not kind or not path:
            raise ValueError(f"--embedder-python takes NAME=PATH, got {item!r}")
        if kind not in kinds:
            raise ValueError(f"--embedder-python {kind!r}: that embedder is not in --embedder-trials")
        if kind in out:
            raise ValueError(f"--embedder-python repeats {kind!r}")
        out[kind] = path
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
               embed_audio: Optional[Path] = None, thr: Optional[float] = None,
               min_on_s: Optional[float] = None) -> List[str]:
    """The stage 2 command of a windowed arm. Without embed_audio, thr and min_on_s it is the same command as before.

    thr replaces --dz-thr and min_on_s adds --min-on-s; both are given only by the --overlap-variants arms.
    """
    python = (cfg.get("vbx_python") or cfg["python"]) if arm in VBX_ARMS else cfg["python"]
    cmd = [python, VBX_STAGE_SCRIPT, "--audio", str(sample_dir / "chunk.wav"), "--probs-dir", str(probs),
           "--out", str(result), "--min-speakers", str(cfg["windowed_min_speakers"])]
    if arm in VBX_ARMS:
        if arm in DZ_ARMS:
            dz_thr = cfg["dz_thr"] if thr is None else thr
            cmd += ["--clustering", "dz", "--thr", str(dz_thr), "--median-s", str(cfg["dz_median_s"])]
        cmd += ["--ahc-threshold", str(cfg["vbx_ahc_threshold"]), "--fa", str(cfg["vbx_fa"]), "--fb", fb]
        if cfg["plda_dir"]:
            cmd += ["--plda-dir", str(cfg["plda_dir"])]
    else:
        cmd += ["--clustering", "ahc", "--embedder", "wespeaker"]
    if embed_audio is not None:
        cmd += ["--embed-audio", str(embed_audio)]
    if min_on_s is not None:
        cmd += ["--min-on-s", fmt(min_on_s)]          # %g, as in the arm name: 0 -> "0", 0.2 -> "0.2"
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


def run_stage2_arm(arm: str, name: str, sample_dir: Path, cfg: dict, runner: Runner, fb: str,
                   thr: Optional[float] = None, min_on_s: Optional[float] = None) -> Path:
    """Stage 2 only, into runs/<name>, on arm J's probs (and K's vocals). Never reruns stage 1 or Demucs: missing
    inputs are made as the base arm makes them, existing ones are used. Used by the --fb-sweep and --overlap-variants arms.
    """
    run_dir = sample_dir / "runs" / name
    result = run_dir / "out" / "segments.json"
    if cfg["reuse"] and result.is_file():
        return result
    run_dir.mkdir(parents=True, exist_ok=True)
    probs = dump_probs("J", sample_dir, cfg, runner, skip_existing=True)
    embed_audio = ensure_vocals(sample_dir, cfg, runner, skip_existing=True) if arm == "K" else None
    cmd2 = stage2_cmd(arm, sample_dir, probs, result, cfg, fb=fb, embed_audio=embed_audio, thr=thr,
                      min_on_s=min_on_s)
    code = runner(cmd2, gpu_env(cfg))
    if code != 0 or not result.is_file():
        raise RuntimeError(f"arm {name} stage 2 ({STAGE2_LABEL[arm]}) failed (exit {code}), see {run_dir}")
    return result


def run_fb_arm(arm: str, fb: str, sample_dir: Path, cfg: dict, runner: Runner) -> Path:
    """A --fb-sweep arm: stage 2 only, with --fb <fb>, on arm J's probs (and K's vocals). Never reruns stage 1 or Demucs."""
    return run_stage2_arm(arm, arm_name(arm_key(arm, fb)), sample_dir, cfg, runner, fb=fb)


def run_ov_arm(key: str, sample_dir: Path, cfg: dict, runner: Runner) -> Path:
    """An --overlap-variants arm (J+ov|thr|on, K+ov|...): the base arm's stage 2 with --thr thr and --min-on-s on.

    Everything else is the base arm's: the same Fb (--vbx-fb), median filter and probs; K's vocals for the embeddings.
    """
    arm, _ = split_key(key)
    thr, min_on_s = ov_params(key)
    return run_stage2_arm(arm, arm_name(key), sample_dir, cfg, runner, fb=str(cfg["vbx_fb"]),
                          thr=thr, min_on_s=min_on_s)


def trial_cmd(kind: str, cos: float, sample_dir: Path, probs: Path, result: Path, cfg: dict) -> List[str]:
    """The stage 2 command of one embedder trial: cosine AHC (--clustering dz-ahc) with embedder `kind` on the chunk.

    The python is --embedder-python for that kind, else --vbx-python, else --python. The embeddings are cached in
    sample_dir/embcache, shared by the trials of the sample, so each --cos-thr after the first reads them.
    """
    python = cfg["embedder_python"].get(kind) or cfg.get("vbx_python") or cfg["python"]
    cmd = [python, VBX_STAGE_SCRIPT, "--audio", str(sample_dir / "chunk.wav"), "--probs-dir", str(probs),
           "--out", str(result), "--min-speakers", str(cfg["windowed_min_speakers"]),
           "--clustering", "dz-ahc", "--embedder-kind", kind, "--cos-threshold", fmt(cos),
           "--thr", str(cfg["dz_thr"]), "--median-s", str(cfg["dz_median_s"]),
           "--embcache-dir", str(sample_dir / TRIAL_CACHE_DIR)]
    if kind == "wespeaker_r293" and cfg.get("r293_onnx"):
        cmd += ["--r293-onnx", str(cfg["r293_onnx"])]
    if kind == "wespeaker_simam100":
        if cfg.get("simam_onnx"):
            cmd += ["--simam-onnx", str(cfg["simam_onnx"])]
        elif cfg.get("simam_dir"):
            cmd += ["--simam-dir", str(cfg["simam_dir"])]
    if kind == "espnet_spk":
        cmd += ["--espnet-tag", str(cfg["espnet_tag"])]
    return cmd


def run_trial_arm(key: str, sample_dir: Path, cfg: dict, runner: Runner) -> Path:
    """An embedder trial (J+emb|kind|cos): stage 2 only, on J's probs, into runs/nemotron_dz_emb_<kind>_c<cos>.

    J's probs are made as for J when missing (stage 1 never reruns once they exist). Nothing else runs: no Demucs,
    the chunk is the embedding audio.
    """
    kind, cos = trial_params(key)
    name = arm_name(key)
    run_dir = sample_dir / "runs" / name
    result = run_dir / "out" / "segments.json"
    if cfg["reuse"] and result.is_file():
        return result
    run_dir.mkdir(parents=True, exist_ok=True)
    probs = dump_probs("J", sample_dir, cfg, runner, skip_existing=True)
    code = runner(trial_cmd(kind, cos, sample_dir, probs, result, cfg), gpu_env(cfg))
    if code != 0 or not result.is_file():
        raise RuntimeError(f"arm {name} stage 2 (cosine AHC, embedder {kind}) failed (exit {code}), see {run_dir}")
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
            if is_trial(key):                   # --embedder-trials: stage 2 only, after every base arm
                segs[key] = crop(read_rows(str(run_trial_arm(key, sdir, cfg, runner))), rs, re_)
                continue
            if is_ov(key):                      # --overlap-variants: stage 2 only, after the base arm in the key order
                segs[key] = crop(read_rows(str(run_ov_arm(key, sdir, cfg, runner))), rs, re_)
                continue
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
    ap.add_argument("--overlap-variants", action="store_true",
                    help="for each selected arm J and K, one stage 2 run per --ov-thr x --ov-min-on-s combination, into "
                         "runs/nemotron_dz_ov_thr<thr>_on<min_on> (stage 1 and Demucs are not rerun)")
    ap.add_argument("--ov-thr", default=str(OV_THR),
                    help="--overlap-variants: comma list of slot probability thresholds in (0, 1), e.g. 0.3,0.35,0.4 "
                         "(first guess, not validated). Each combination with --ov-min-on-s is one arm per J and K")
    ap.add_argument("--ov-min-on-s", default=str(OV_MIN_ON_S),
                    help="--overlap-variants: comma list of minimum ON-run lengths in seconds, 0 or more, e.g. 0,0.2: "
                         "stage 2 drops each cluster's ON runs shorter than this")
    ap.add_argument("--embedder-trials", default="",
                    help="comma list of embedder kinds (r34lm, wespeaker_r293, wespeaker_simam100, espnet_spk): for each "
                         "sample, stage 2 with cosine AHC (--clustering dz-ahc) on J's probs and the chunk, one arm per "
                         "kind and --cos-thr value (nemotron_dz_emb_<kind>_c<thr>). Off when empty")
    ap.add_argument("--cos-thr", default=",".join(fmt(c) for c in TRIAL_COS),
                    help="--embedder-trials: comma list of cosine-distance cuts of the average-linkage AHC, in (0, 2)")
    ap.add_argument("--embedder-python", action="append", default=[], metavar="NAME=PATH",
                    help="--embedder-trials: python for one kind, e.g. r34lm=/path/to/dz/python (repeatable). "
                         "Default: --vbx-python, then --python")
    ap.add_argument("--r293-onnx", help="--embedder-trials, wespeaker_r293: ONNX file (default: the HF hub copy)")
    ap.add_argument("--simam-onnx", help="--embedder-trials, wespeaker_simam100: ONNX file (UNVERIFIED export names)")
    ap.add_argument("--simam-dir", help="--embedder-trials, wespeaker_simam100: folder with config.yaml and avg_model.pt "
                                        "(UNVERIFIED API); give one of --simam-onnx and --simam-dir")
    ap.add_argument("--espnet-tag", default=EMB.DEFAULT_ESPNET_TAG,
                    help="--embedder-trials, espnet_spk: ESPnet model tag (UNVERIFIED)")
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
    try:
        ov_thrs = parse_grid(a.ov_thr, "--ov-thr", lambda v: 0 < v < 1, "must be in (0, 1)")
        ov_min_ons = parse_grid(a.ov_min_on_s, "--ov-min-on-s", lambda v: v >= 0, "must be 0 or more")
        ov_combos = ov_grid(ov_thrs, ov_min_ons)
        cos_thrs = parse_grid(a.cos_thr, "--cos-thr", lambda v: 0 < v < 2, "must be in (0, 2)")
        trial_kinds = parse_kinds(a.embedder_trials)
        embedder_python = parse_embedder_python(a.embedder_python, trial_kinds)
        for kind in trial_kinds:
            EMB.validate_args(kind, r293_onnx=a.r293_onnx, simam_onnx=a.simam_onnx, simam_dir=a.simam_dir)
    except ValueError as exc:
        ap.error(str(exc))
    if a.overlap_variants and not letters & set(OV_ARMS):
        ap.error("--overlap-variants needs arm J or K in --only-arms")
    if a.fusion:
        letters.add("E")
    run = set(letters)
    if run & {"C", "D", "E"}:
        run.add("A")            # K comes from arm A
    if "E" in run:
        run.add("B")            # fusion boundaries come from arm B
    keys = []                   # base arms in letter order, each followed by its --fb-sweep and overlap arms
    for letter in ARM_ORDER:
        if letter in run:
            keys.append(arm_key(letter))
            if letter in FB_SWEEP_ARMS:
                keys += [arm_key(letter, fb) for fb in fb_values]
            if a.overlap_variants and letter in OV_ARMS:
                keys += [ov_key(letter, thr, on) for thr, on in ov_combos]
    keys += [trial_key(kind, cos) for kind in trial_kinds for cos in cos_thrs]   # after every base arm
    windowed_python = None
    if run & {"F", "G", "I", "J", "K"} or trial_kinds:       # trials need J's probs, stage 1 of the Nemotron env
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
           "vbx_fb": a.vbx_fb, "plda_dir": a.plda_dir,
           "embedder_python": embedder_python, "r293_onnx": a.r293_onnx, "simam_onnx": a.simam_onnx,
           "simam_dir": a.simam_dir, "espnet_tag": a.espnet_tag}
    summaries = []
    for i, sample in enumerate(load_samples(a.samples), 1):
        try:
            summary = process_sample(i, sample, cfg, runner or run_main)
        except Exception as exc:
            summary = {"name": str(sample.get("name")), "error": str(exc), "lanes": None, "arms": {}, "k": None,
                       "errors": {}}
            print(f"{sample.get('name')}: error: {exc}", file=sys.stderr)
        summaries.append(summary)
    failures = embedder_failures(summaries)
    if failures:
        print(f"embedder trials failed for {len(failures)} (sample, embedder) pairs:")
        for (name, kind), items in failures.items():
            cos = ", ".join(fmt(c) for c, _ in items)
            print(f"  {name} / {kind} (cos {cos}): {items[0][1]}")
    index = write_index(out, summaries, keys)
    print(f"-> {index}")
    return 0


def embedder_failures(summaries: Sequence[dict]) -> Dict[Tuple[str, str], List[Tuple[float, str]]]:
    """(sample name, embedder kind) -> [(cos, message)] of the trial arms that failed, in key order."""
    out: Dict[Tuple[str, str], List[Tuple[float, str]]] = {}
    for s in summaries:
        for key, msg in (s.get("errors") or {}).items():
            if is_trial(key):
                kind, cos = trial_params(key)
                out.setdefault((str(s.get("name")), kind), []).append((cos, msg))
    return out


if __name__ == "__main__":
    sys.exit(main())
