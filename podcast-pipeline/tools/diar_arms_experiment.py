#!/usr/bin/env python3
"""Diarization arms on fixed conversation samples, shown as waveform lanes (one row per speaker, no text).

    python tools/diar_arms_experiment.py --audio-dir SRC --samples tools/diar_arms_samples.json --out OUT/diar_arms \
        [--final OUT/final] [--gpu 6] [--env a100] [--only-arms A,C,F] [--fusion] [--k-min-clean 1.0] [--reuse] \
        [--windowed-piece-s 240] [--windowed-hop-s 10] [--windowed-python /path/to/nemotron/python]

Per sample (a conversation clip of about 40 s inside a long source recording):
  1. cut a 5 min chunk of the source around the clip (it starts 2 min before the clip, clamped to the file), 16 kHz mono.
  2. arm A, Nemotron on the chunk. K = labels whose clean speech (utils.nemotron_two_pass.clean_speech_duration)
     reaches --k-min-clean seconds inside the clip, at least 1. The default 1.0 s keeps backchannel listeners that
     say only 'ừ à' (under 1.5 s) in K.
  3. arm B, DiariZen with its defaults. Arm C, DiariZen with min = max = K. Arm D, DiariZen with min 1, max K.
  4. arm E (--fusion), B's boundaries with labels renamed to A's labels by greedy one-to-one overlap.
     E is built from B, so it also runs A and B. C and D need A's K, so they run A too.
  4b. arm F (--only-arms F, not in the default set), nemotron_windowed: tools/windowed_nemotron_run.py in the
     Nemotron environment (--windowed-python, else NEMOTRON_PYTHON / worker_envs). Silero pieces of about
     --windowed-piece-s seconds with 30 s overlap, 30 s Nemotron windows every --windowed-hop-s seconds, per-frame
     probabilities merged by utils/windowed_clustering.run_file (global clustering with cannot-link, DiariZen-style
     frame averaging). F needs no K. UNVERIFIED on GPU: not run on a real model yet.
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
        "F": "nemotron_windowed"}
ARM_ORDER = "ABCDEF"
DEFAULT_ARMS = "A,B,C,D"
WINDOWED_PIECE_S = 240.0
WINDOWED_HOP_S = 10.0
WINDOWED_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "windowed_nemotron_run.py")
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


def run_windowed(sample_dir: Path, cfg: dict, runner: Runner) -> Path:
    """Run arm F (tools/windowed_nemotron_run.py in the Nemotron env) on sample_dir/chunk.wav."""
    run_dir = sample_dir / "runs" / ARMS["F"]
    result = run_dir / "out" / "segments.json"
    if cfg["reuse"] and result.is_file():
        return result
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [cfg["windowed_python"], WINDOWED_SCRIPT, "--audio", str(sample_dir / "chunk.wav"),
           "--out", str(result), "--piece-s", str(cfg["windowed_piece_s"]),
           "--hop-s", str(cfg["windowed_hop_s"]), "--config", cfg["config"], "--env", cfg["env"]]
    code = runner(cmd, dict(os.environ, CUDA_VISIBLE_DEVICES=str(cfg["gpu"])))
    if code != 0 or not result.is_file():
        raise RuntimeError(f"arm {ARMS['F']} failed (exit {code}), see {run_dir}")
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
    for arm in ARM_ORDER:
        if arm not in cfg["run"]:
            continue
        try:
            if arm == "E":
                if "A" not in segs or "B" not in segs:
                    raise RuntimeError("fusion needs arms A and B")
                segs["E"] = fuse(segs["B"], segs["A"])
                continue
            if arm == "F":                      # windowed Nemotron: no K needed
                segs["F"] = crop(read_rows(str(run_windowed(sdir, cfg, runner))), rs, re_)
                continue
            backend, values = arm_settings(arm, k)
            path = run_arm(sdir, arm, values, backend, cfg, runner)
            segs[arm] = crop(read_rows(str(path)), rs, re_)
            if arm == "A":
                k = estimate_k(segs["A"], cfg["k_min_clean"])
        except Exception as exc:  # the other arms still run
            errors[arm] = str(exc)
            log(f"  arm {arm}: {exc}")
    if k is not None:
        log(f"  K = {k} (from A, k_min_clean {cfg['k_min_clean']})")

    for arm in ARM_ORDER:
        if arm in segs:
            with open(sdir / f"segments_{ARMS[arm]}.json", "w", encoding="utf-8") as fh:
                json.dump({"segments": segs[arm]}, fh, ensure_ascii=False, indent=1)
    lanes = None
    diars = [(ARMS[a], str(sdir / f"segments_{ARMS[a]}.json")) for a in ARM_ORDER if a in segs]
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
            rows.append(f"<tr><td>{name}</td><td>{arm} {ARMS[arm]}</td>{tail}</tr>")
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
                    help="comma list of A,B,C,D,F (E needs --fusion; F is the windowed Nemotron arm)")
    ap.add_argument("--fusion", action="store_true", help="also build arm E")
    ap.add_argument("--k-min-clean", type=float, default=1.0, help="clean seconds for a speaker to count toward K")
    ap.add_argument("--reuse", action="store_true", help="skip arms whose segments.json already exists")
    ap.add_argument("--config", default=os.path.join(ROOT, "config.json"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--windowed-piece-s", type=float, default=WINDOWED_PIECE_S,
                    help="arm F piece length (min and max are 0.75x and 1.25x of it)")
    ap.add_argument("--windowed-hop-s", type=float, default=WINDOWED_HOP_S, help="arm F window hop (window is 30 s)")
    ap.add_argument("--windowed-python",
                    help="interpreter of the Nemotron env for arm F (default: NEMOTRON_PYTHON or worker_envs)")
    a = ap.parse_args(argv)

    letters = {x.strip().upper() for x in a.only_arms.split(",") if x.strip()}
    if letters - set(ARM_ORDER):
        ap.error(f"unknown arms in --only-arms: {sorted(letters - set(ARM_ORDER))}")
    if a.fusion:
        letters.add("E")
    run = set(letters)
    if run & {"C", "D", "E"}:
        run.add("A")            # K comes from arm A
    if "E" in run:
        run.add("B")            # fusion boundaries come from arm B
    arms = [x for x in ARM_ORDER if x in run]
    windowed_python = None
    if "F" in run:
        windowed_python = a.windowed_python or resolve_worker_python("nemotron")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"audio_dir": a.audio_dir, "final": a.final, "out": str(out), "gpu": a.gpu, "env": a.env,
           "run": run, "k_min_clean": a.k_min_clean, "reuse": a.reuse, "config": a.config, "python": a.python,
           "windowed_python": windowed_python, "windowed_piece_s": a.windowed_piece_s,
           "windowed_hop_s": a.windowed_hop_s}
    summaries = []
    for i, sample in enumerate(load_samples(a.samples), 1):
        try:
            summary = process_sample(i, sample, cfg, runner or run_main)
        except Exception as exc:
            summary = {"name": str(sample.get("name")), "error": str(exc), "lanes": None, "arms": {}, "k": None,
                       "errors": {}}
            print(f"{sample.get('name')}: error: {exc}", file=sys.stderr)
        summaries.append(summary)
    index = write_index(out, summaries, arms)
    print(f"-> {index}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
