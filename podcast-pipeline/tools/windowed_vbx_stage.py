#!/usr/bin/env python3
"""Arms F and G, stage 2: global clustering of the windowed Nemotron probabilities. Runs in the `som` env.

    <som python> tools/windowed_vbx_stage.py --audio chunk.wav --probs-dir DIR --out segments.json \
        [--clustering vbx|ahc] [--embedder pyannote|wespeaker] [--max-embed-s 6] \
        [--plda-dir DIR] [--embedding-model pyannote/wespeaker-voxceleb-resnet34-LM] [--ahc-threshold 0.6] \
        [--fa 0.07] [--fb 0.8] [--lda-dim 128] [--max-iters 20] [--init-smoothing 7.0] [--device cuda:0] \
        [--embed-audio VOCALS.wav] [--min-on-s 0.2]

--embed-audio (arm K) is the waveform the embedder cuts its windows from: a Demucs vocals stem of --audio with the
same sample rate and sample count. The probabilities and masks still come from --audio. Without it nothing changes.

Arm J is --clustering dz: DiariZen's own pipeline steps (utils/dz_pipeline.py) with masked embeddings of the whole
window, a per-frame speaker count and top-k reconstruction. Run it with the python of the DiariZen env, which has
the pyannote 3.1.1 fork DiariZen is built on.

EXPERIMENT: --clustering dz-ahc is arm J's steps with cosine AHC instead of VBx (utils/dz_pipeline.make_dz_ahc_fn), so
any embedder can be tried; the DiariZen PLDA only fits WeSpeaker ResNet34-LM. --embedder-kind picks it (tools/embedders.py):
r34lm (the same masked call as dz), wespeaker_r293 (--r293-onnx), wespeaker_simam100 (--simam-onnx or --simam-dir),
espnet_spk (--espnet-tag). The other kinds get the clean frames of each window as audio (make_concat_embedder, 0.5 s
minimum). Embeddings are cached in --embcache-dir (default OUT_DIR/embcache) by piece, window span and frame mask, so
another --cos-threshold reads the cache and calls no model. NOT RUN on a real model here: fakes only.

Arm G is the default (--clustering vbx, --embedder pyannote). Arm F is --clustering ahc --embedder wespeaker:
the constrained AHC of utils/windowed_clustering and the repo's WeSpeaker ResNet-293 ONNX model
(models/wespeaker_embedding.py), which runs on the GPU here; in the Nemotron env onnxruntime cannot load CUDA 13
and falls back to the CPU. VBx needs the ResNet34 embedder its PLDA was trained for, so vbx with wespeaker is refused.

Reads DIR/plan.json and DIR/piece_NNN.npz written by tools/windowed_nemotron_run.py --dump-probs (stage 1, NeMo
env). For each piece it runs utils/windowed_clustering.run_file with make_vbx_cluster_fn: the speaker embeddings
come from pyannote's WeSpeaker ResNet34 (the model DiariZen's PLDA was trained on, not the ONNX model of arm F),
and the clustering is DiariZen's: centroid AHC init, then VBx over the PLDA space, then the per-window Hungarian
reassignment. Writes {"segments", "pieces", "frame_s", "params"} to --out, same segments shape as arm F.

PLDA folder: --plda-dir, else the newest snapshot under the HF hub cache of BUT-FIT/diarizen-wavlm-large-s80-md-v2
that holds xvec_transform.npz and plda.npz. The chosen folder and all candidates are printed in the start line.

UNVERIFIED on GPU and on the installed packages: the defaults (ahc_threshold 0.6, Fa 0.07, Fb 0.8) are not
checked against the DiariZen config; the pyannote call (PretrainedSpeakerEmbedding(path, device=...), then
emb(waveforms) with a (1, 1, T) float tensor, returning (1, D)) is written from memory of pyannote 3.1+; crops
shorter than emb.min_num_samples, if the attribute exists, are zero-padded (run_piece needs one row per request).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

import utils.windowed_clustering as wc  # noqa: E402
import utils.dz_pipeline as dz  # noqa: E402
from tools import embedders as EMB  # noqa: E402

SR = 16000
VBX_REPO = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
PLDA_FILES = ("xvec_transform.npz", "plda.npz")
EMBEDDING_REPO = "pyannote/wespeaker-voxceleb-resnet34-LM"


def hf_hub_root() -> str:
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return os.path.join(home, "hub")


def plda_candidates(cache_root: str | None = None, repo: str = VBX_REPO) -> list[str]:
    """plda/ folders of every snapshot that holds both PLDA files, newest folder first."""
    base = os.path.join(cache_root or hf_hub_root(), "models--" + repo.replace("/", "--"), "snapshots")
    found = [d for d in glob.glob(os.path.join(base, "*", "plda"))
             if all(os.path.isfile(os.path.join(d, f)) for f in PLDA_FILES)]
    return sorted(found, key=os.path.getmtime, reverse=True)


def read_plda_meta(plda_dir: str):
    """The parsed plda_meta.json of a PLDA folder (tools/train_plda.py writes it), or None when the file is absent."""
    path = os.path.join(plda_dir, "plda_meta.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def plda_embedder_problem(kind: str, plda_meta) -> str | None:
    """Why a PLDA folder cannot be used with embedder `kind`, or None when it can. Pure: no files are read here.

    plda_meta is the parsed plda_meta.json, or None when the file is absent. The file is required for every kind but
    r34lm: the default DiariZen PLDA (no --plda-dir) and any --plda-dir folder without a meta file are ResNet34-LM.
    """
    if plda_meta is None:
        if kind == "r34lm":
            return None
        return (f"plda_meta.json is absent, so nothing says this PLDA was fitted for {kind}; "
                "train it with tools/train_plda.py --embedder-id " + kind)
    got = plda_meta.get("embedder_id")
    if got != kind:
        return f"the PLDA was fitted for embedder_id {got!r}, but the embedder is {kind!r}"
    return None


def find_plda_dir(explicit: str | None = None) -> str:
    if explicit:
        if not all(os.path.isfile(os.path.join(explicit, f)) for f in PLDA_FILES):
            raise FileNotFoundError(f"--plda-dir {explicit} lacks {' or '.join(PLDA_FILES)}")
        return explicit
    found = plda_candidates()
    if not found:
        raise FileNotFoundError(f"no PLDA folder with {PLDA_FILES} under {hf_hub_root()}; pass --plda-dir")
    return found[0]


def build_wespeaker(device: str):
    """The repo's WeSpeaker ResNet-293 ONNX embedder (arm F). Returns (embed_clip, providers)."""
    from models.wespeaker_embedding import WeSpeakerONNXEmbedder
    emb = WeSpeakerONNXEmbedder(device=device)
    providers = emb.active_providers()
    if str(device).startswith("cuda") and "CUDAExecutionProvider" not in providers:
        print(f"WARNING: WeSpeaker has no CUDA provider here, providers={providers}: it runs on the CPU",
              file=sys.stderr, flush=True)

    def embed_clip(clip):
        out = emb.embed(np.asarray(clip, dtype=np.float32), SR)
        return np.asarray(out.detach().cpu().numpy() if hasattr(out, "detach") else out, dtype=float).reshape(-1)

    return embed_clip, providers


def import_vbx():
    try:
        from diarizen.clustering.VBx import cluster_vbx, vbx_setup
    except ImportError as exc:
        raise SystemExit(f"windowed_vbx_stage needs the diarizen package in this env (som): {exc}")
    return cluster_vbx, vbx_setup


def build_embedder(model: str, device: str):
    """pyannote WeSpeaker ResNet34, the embedding model DiariZen's PLDA was trained on.

    PretrainedSpeakerEmbedding(path, device=...) and emb(waveforms, masks=...) are the calls DiariZen's pyannote fork
    makes (speaker_verification.py). The pyannote 3.1.1 loader of the DiariZen env needs torch.load back at
    weights_only=False (torch >= 2.6), as the DiariZen worker does, so the shim goes in before the model is built.
    """
    try:
        import torch
        from utils.torch_compat import install_torch_load_shim
        install_torch_load_shim()
        from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
    except ImportError as exc:
        raise SystemExit(f"windowed_vbx_stage needs torch and pyannote.audio in this env: {exc}")
    if os.path.isfile(model):
        path = model
    else:
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=model, filename="pytorch_model.bin")
        except Exception as exc:
            raise SystemExit(f"cannot download {model}/pytorch_model.bin: {exc}")
    if "wespeaker" not in path.lower():
        raise SystemExit(f"embedding model path must name WeSpeaker (pyannote picks the class from it): {path}")
    emb = PretrainedSpeakerEmbedding(path, device=torch.device(device))
    return emb, torch, path


def make_clip_embedder(emb, torch, device: str):
    """(clip float32 array) -> unit-free (D,) numpy row, one clip per call."""
    min_n = getattr(emb, "min_num_samples", None)
    counts = {"padded": 0}

    def embed_clip(clip):
        x = np.asarray(clip, dtype=np.float32)
        if min_n and len(x) < int(min_n):
            x = np.pad(x, (0, int(min_n) - len(x)))     # run_piece needs one row per request
            counts["padded"] += 1
        if len(x) == 0:
            raise ValueError("empty clip for the embedder")
        wav = torch.from_numpy(x).to(device).reshape(1, 1, -1)
        with torch.no_grad():
            out = emb(wav)
        if hasattr(out, "detach"):
            out = out.detach().cpu().numpy()
        return np.asarray(out, dtype=float).reshape(-1)  # (1, D) -> (D,)

    return embed_clip, counts


def make_masked_embedder(emb, torch, device: str):
    """(window audio, frame mask) -> (D,) raw embedding, or None when the model refuses (arm J, as DiariZen calls it)."""
    failed = {"n": 0}

    def embed(clip, mask):
        x = torch.from_numpy(np.asarray(clip, dtype=np.float32)).to(device).reshape(1, 1, -1)
        w = torch.from_numpy(np.asarray(mask, dtype=np.float32)).to(device).reshape(1, -1)
        try:
            out = emb(x, masks=w)
        except (ValueError, RuntimeError):
            failed["n"] += 1
            return None
        return np.asarray(out, dtype=float).reshape(-1)

    return embed, failed


def load_plan(probs_dir: str):
    with open(os.path.join(probs_dir, "plan.json"), encoding="utf-8") as fh:
        plan = json.load(fh)
    pieces, tables = [], {}
    for entry in plan["pieces"]:
        piece, spans, probs, frame_s = wc.load_piece_probs(os.path.join(probs_dir, entry["file"]))
        if abs(frame_s - float(plan["frame_s"])) > 1e-9:
            raise ValueError(f"{entry['file']} has frame_s {frame_s}, plan has {plan['frame_s']}")
        pieces.append(piece)
        tables[piece_key(piece)] = dict(zip(spans, probs))
    return plan, pieces, tables


def piece_key(piece) -> tuple[float, float]:
    return (round(piece.audio_start, 6), round(piece.audio_end, 6))


def embedder_waveform(audio_wave, audio_sr: int, embed_wave, embed_sr: int):
    """The waveform the embedder cuts its windows from: --audio when --embed-audio is not given, else that file.

    An --embed-audio file must have the sample rate and the exact number of samples of --audio, so every frame of
    the probabilities still lines up with it (arm K: the same chunk, its vocals stem).
    """
    if embed_wave is None:
        return audio_wave
    if int(embed_sr) != int(audio_sr):
        raise ValueError(f"--embed-audio is {embed_sr} Hz, --audio is {audio_sr} Hz: they must match")
    if len(embed_wave) != len(audio_wave):
        raise ValueError(f"--embed-audio has {len(embed_wave)} samples, --audio has {len(audio_wave)}: "
                         "they must be the same length so the frames line up")
    return embed_wave


def crop_for_embedder(wave, piece_audio_start: float, span) -> np.ndarray:
    """The samples of span (s, e), seconds from the piece's start, as the embedder gets them."""
    s, e = span
    return wave[int(round((piece_audio_start + s) * SR)): int(round((piece_audio_start + e) * SR))]


def file_fingerprint(path: str) -> list:
    """[absolute path, size, mtime in ns] of a file: what the embedding cache needs to know about an input file."""
    st = os.stat(path)
    return [os.path.abspath(path), int(st.st_size), int(st.st_mtime_ns)]


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def embedding_cache_path(cache_dir: str, embedder_id: str, inputs: dict) -> str:
    """<cache_dir>/<embedder id>_<hash of inputs>.npz. The inputs are everything that decides an embedding except the
    embedder itself and the cosine threshold, which is not an input: a run with another --cos-threshold finds the file.
    """
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, f"{embedder_id}_{digest}.npz")


class EmbeddingCache:
    """Embeddings of --clustering dz-ahc, keyed by what run_piece_dz asks the embedder: the piece's start on the file,
    the window span and the frame mask (hashed). A cached entry is the embedding that same call would give, so a run
    that finds every key makes no model call. Entries that gave None are cached too (None is a result, not a miss).

    Stored as one .npz (keys, ok, vecs), written through a temporary file and then renamed.
    """

    def __init__(self, path: str):
        self.path = path
        self.table: dict = {}
        self.hits = 0
        self.misses = 0
        self._new = 0

    def load(self) -> "EmbeddingCache":
        if not os.path.isfile(self.path):
            return self
        with np.load(self.path, allow_pickle=False) as z:
            keys = [str(k) for k in z["keys"]]
            ok = np.asarray(z["ok"], dtype=bool)
            vecs = np.asarray(z["vecs"], dtype=float)
        for i, key in enumerate(keys):
            self.table[key] = vecs[i].copy() if ok[i] else None
        return self

    def save(self) -> bool:
        """Writes the file when this run added entries. Returns whether it wrote."""
        if not self._new:
            return False
        keys = sorted(self.table)
        dims = {len(v) for v in self.table.values() if v is not None}
        if len(dims) > 1:
            raise ValueError(f"cached embeddings have different sizes {sorted(dims)}; one embedder gives one size")
        dim = dims.pop() if dims else 0
        vecs = np.zeros((len(keys), dim))
        ok = np.zeros(len(keys), dtype=bool)
        for i, key in enumerate(keys):
            v = self.table[key]
            if v is not None:
                vecs[i], ok[i] = v, True
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp.npz"
        np.savez(tmp, keys=np.array(keys, dtype=str), ok=ok, vecs=vecs)
        os.replace(tmp, self.path)
        self._new = 0
        return True

    def wrap(self, piece_start: float, embed_fn):
        """embed_fn(span, mask) -> the same, answered from the table when the key is there."""

        def cached(span, mask):
            m = np.ascontiguousarray(np.asarray(mask, dtype=np.float32))
            key = (f"{round(float(piece_start), 6):.6f}|{float(span[0]):.6f}|{float(span[1]):.6f}|"
                   f"{hashlib.sha1(m.tobytes()).hexdigest()}")
            if key in self.table:
                self.hits += 1
                v = self.table[key]
                return None if v is None else v.copy()
            self.misses += 1
            v = embed_fn(span, mask)
            v = None if v is None else np.asarray(v, dtype=np.float64).reshape(-1)
            self.table[key] = v
            self._new += 1
            return None if v is None else v.copy()

        return cached


def run_dz_stage(duration: float, plan: dict, pieces: list, tables: dict, make_embed_fn, dz_vbx, *, thr: float,
                 median_frames: int, min_frames_ratio: float, min_num_frames: int, max_speakers: int,
                 min_on_s: float) -> dict:
    """The dz steps over all pieces (utils/dz_pipeline.run_piece_dz through wc.run_file). Torch-free: the callers pass
    the embedding function of each piece (make_embed_fn(piece) -> embed_fn(span, mask)) and the clustering function
    (dz_vbx: make_dz_vbx_fn or make_dz_ahc_fn), so the tests can run it on fakes."""
    def make_prob_fn(piece):
        table = tables[piece_key(piece)]

        def prob_fn(s, e):
            return table[(float(s), float(e))]
        return prob_fn

    return wc.run_file(duration, None, make_prob_fn=make_prob_fn, make_embed_fn=make_embed_fn,
                       pieces=pieces, piece_s=plan["params"]["piece_s"], window_s=plan["params"]["window_s"],
                       hop_s=plan["params"]["hop_s"], frame_s=float(plan["frame_s"]), piece_fn=dz.run_piece_dz,
                       vbx_fn=dz_vbx, thr=thr, median_frames=median_frames, min_frames_ratio=min_frames_ratio,
                       min_num_frames=min_num_frames, max_speakers=max_speakers, min_on_s=min_on_s)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", required=True, help="16 kHz mono WAV, the same chunk stage 1 read")
    ap.add_argument("--probs-dir", required=True, help="folder written by windowed_nemotron_run.py --dump-probs")
    ap.add_argument("--out", required=True, help="segments.json to write")
    ap.add_argument("--clustering", choices=("vbx", "ahc", "dz", "dz-ahc"), default="vbx",
                    help="vbx = arm G, ahc = arm F (constrained AHC at --cosine-threshold), dz = arm J (DiariZen steps), "
                         "dz-ahc = arm J's steps with cosine AHC (embedder trials; see --embedder-kind)")
    ap.add_argument("--embedder", choices=("pyannote", "wespeaker"),
                    help="default: pyannote for vbx, wespeaker (ResNet-293 ONNX) for ahc. Not for dz-ahc.")
    ap.add_argument("--embedder-kind", choices=EMB.KINDS, default=None,
                    help="dz-ahc only: r34lm (default, arm J's embedder), wespeaker_r293, wespeaker_simam100, espnet_spk")
    ap.add_argument("--cos-threshold", type=float, default=0.5,
                    help="dz-ahc only: cosine-distance cut of the average-linkage AHC, in (0, 2). Not --cosine-threshold, "
                         "which is arm F's")
    ap.add_argument("--r293-onnx", help="dz-ahc, wespeaker_r293: ONNX file (default: the HF hub copy of models/wespeaker_embedding)")
    ap.add_argument("--simam-onnx", help="dz-ahc, wespeaker_simam100: ONNX file (UNVERIFIED export names)")
    ap.add_argument("--simam-dir", help="dz-ahc, wespeaker_simam100: folder with config.yaml and avg_model.pt (UNVERIFIED API)")
    ap.add_argument("--espnet-tag", default=EMB.DEFAULT_ESPNET_TAG, help="dz-ahc, espnet_spk: ESPnet model tag (UNVERIFIED)")
    ap.add_argument("--embcache-dir",
                    help="dz-ahc: folder of the embedding cache (default: embcache/ next to --out). Share one folder "
                         "between runs to reuse embeddings across --cos-threshold values")
    ap.add_argument("--cosine-threshold", type=float, default=0.5, help="cosine cut of the AHC of --clustering ahc")
    ap.add_argument("--min-speakers", type=int, default=1,
                    help="clusters per piece at least; below it VBx falls back to AHC (arm F's AHC stops at it too)")
    ap.add_argument("--thr", type=float, default=0.5, help="dz: slot probability threshold of the local segmentation")
    ap.add_argument("--median-s", type=float, default=0.22,
                    help="dz: median filter along time of each window's binarised slots, in seconds "
                         "(DiariZen: 11 frames of about 20 ms); turned into an odd number of output frames")
    ap.add_argument("--median-frames", type=int,
                    help="dz: the median filter size in output frames, overriding --median-s")
    ap.add_argument("--min-frames-ratio", type=float, default=0.1,
                    help="dz: share of a window's frames a slot needs alone to enter the clustering (DiariZen 0.1)")
    ap.add_argument("--max-speakers", type=int, default=20, help="dz: cap of the per-frame speaker count")
    ap.add_argument("--min-on-s", type=float, default=0.0,
                    help="dz (experiment): drop each cluster's ON runs shorter than this many seconds after the "
                         "binarisation; 0 = off (DiariZen has no minimum). Must be 0 or more.")
    ap.add_argument("--max-embed-s", type=float, default=6.0,
                    help="clean seconds embedded per (window, label); 0 = all of it")
    ap.add_argument("--plda-dir", help="folder with xvec_transform.npz and plda.npz (default: HF cache)")
    ap.add_argument("--embedding-model", default=EMBEDDING_REPO,
                    help="HF repo (pytorch_model.bin) or a local .bin file; the path must name WeSpeaker")
    ap.add_argument("--ahc-threshold", type=float, default=0.6, help="euclidean cut of the AHC init (UNVERIFIED)")
    ap.add_argument("--fa", type=float, default=0.07, help="VBx Fa (UNVERIFIED)")
    ap.add_argument("--fb", type=float, default=0.8, help="VBx Fb (UNVERIFIED)")
    ap.add_argument("--lda-dim", type=int, default=128)
    ap.add_argument("--max-iters", type=int, default=20)
    ap.add_argument("--init-smoothing", type=float, default=7.0)
    ap.add_argument("--device", help="cuda:0 or cpu (default: cuda:0 when available)")
    ap.add_argument("--embed-audio", default=None,
                    help="WAV the embedder cuts its windows from (arm K: the Demucs vocals of --audio). Same sample "
                         "rate and sample count as --audio. Probabilities, masks and timing still use --audio.")
    a = ap.parse_args(argv)
    if not a.min_on_s >= 0:                       # also catches nan
        ap.error(f"--min-on-s must be 0 or more, got {a.min_on_s}")
    if a.clustering == "dz-ahc":
        if a.embedder:
            ap.error("--clustering dz-ahc takes its embedder from --embedder-kind, not --embedder")
        if not (math.isfinite(a.cos_threshold) and 0 < a.cos_threshold < 2):
            ap.error(f"--cos-threshold must be in (0, 2), got {a.cos_threshold}")
        try:
            EMB.validate_args(a.embedder_kind or "r34lm", r293_onnx=a.r293_onnx, simam_onnx=a.simam_onnx,
                              simam_dir=a.simam_dir)
        except ValueError as exc:
            ap.error(str(exc))
    elif a.embedder_kind:
        ap.error("--embedder-kind is only for --clustering dz-ahc")

    import soundfile as sf
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(f"windowed_vbx_stage needs torch in this env (som): {exc}")

    device = a.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    kind = a.embedder_kind or "r34lm"
    if a.clustering == "dz-ahc":
        embedder = "dz-ahc"
    else:
        embedder = a.embedder or ("wespeaker" if a.clustering == "ahc" else "pyannote")
    if a.clustering in ("vbx", "dz") and embedder != "pyannote":
        raise SystemExit("--clustering vbx and dz need --embedder pyannote: the PLDA was trained for WeSpeaker ResNet34")
    plan, pieces, tables = load_plan(a.probs_dir)
    params = {"clustering": a.clustering, "embedder": embedder, "max_embed_s": a.max_embed_s, "device": device,
              "window_s": plan["params"]["window_s"], "hop_s": plan["params"]["hop_s"],
              "piece_s": plan["params"]["piece_s"]}
    cluster_fn = None
    dz_vbx = None
    if a.clustering in ("vbx", "dz"):
        if a.plda_dir:                            # refuse a PLDA fitted for another embedder before any model load
            problem = plda_embedder_problem(kind, read_plda_meta(a.plda_dir))
            if problem:
                raise SystemExit(f"--plda-dir {a.plda_dir}: {problem}")
        cluster_vbx, vbx_setup = import_vbx()
        plda_dir = find_plda_dir(a.plda_dir)
        xvec_tf, plda_tf, plda_psi = vbx_setup(plda_dir)
        params.update({"Fa": a.fa, "Fb": a.fb, "ahc_threshold": a.ahc_threshold, "lda_dim": a.lda_dim,
                       "max_iters": a.max_iters, "init_smoothing": a.init_smoothing, "plda_dir": plda_dir,
                       "embedding_model": a.embedding_model})
        if a.clustering == "vbx":
            cluster_fn = wc.make_vbx_cluster_fn(cluster_vbx, xvec_tf, plda_tf, plda_psi,
                                                ahc_threshold=a.ahc_threshold, Fa=a.fa, Fb=a.fb, lda_dim=a.lda_dim,
                                                max_iters=a.max_iters, init_smoothing=a.init_smoothing)
        else:
            dz_vbx = dz.make_dz_vbx_fn(cluster_vbx, xvec_tf, plda_tf, plda_psi, ahc_threshold=a.ahc_threshold,
                                       Fa=a.fa, Fb=a.fb, lda_dim=a.lda_dim, max_iters=a.max_iters)
            median_frames = a.median_frames or dz.frames_for_seconds(a.median_s, float(plan["frame_s"]))
            params.update({"thr": a.thr, "median_frames": median_frames, "median_s": a.median_s,
                           "frame_s": float(plan["frame_s"]), "min_frames_ratio": a.min_frames_ratio,
                           "max_speakers": a.max_speakers, "min_on_s": a.min_on_s})
        start = {"plda_dir": plda_dir, "plda_candidates": plda_candidates()}
    elif a.clustering == "dz-ahc":                # no PLDA: cosine AHC, so no VBx import and no PLDA folder
        dz_vbx = dz.make_dz_ahc_fn(a.cos_threshold)
        median_frames = a.median_frames or dz.frames_for_seconds(a.median_s, float(plan["frame_s"]))
        params.update({"embedder": kind, "cos_threshold": a.cos_threshold, "thr": a.thr,
                       "median_frames": median_frames, "median_s": a.median_s, "frame_s": float(plan["frame_s"]),
                       "min_frames_ratio": a.min_frames_ratio, "max_speakers": a.max_speakers,
                       "min_on_s": a.min_on_s})
        start = {}
    else:
        params["cosine_threshold"] = a.cosine_threshold
        start = {}
    print(json.dumps({"status": "start", "pieces": len(pieces), "params": params, **start},
                     ensure_ascii=False), flush=True)

    counts = {"padded": 0}
    dz_embed = None                       # the dz embedding call (masked for r34lm, see make_dz_embed_fn)
    failed = {"fn": lambda: 0}            # refused embeddings so far, for the progress lines
    adapter = None
    if embedder == "pyannote":
        emb, _torch, emb_path = build_embedder(a.embedding_model, device)
        params["embedding_path"] = emb_path
        embed_clip, counts = make_clip_embedder(emb, torch, device)
        masked_embed, masked_failed = make_masked_embedder(emb, torch, device)
        min_samples = getattr(emb, "min_num_samples", None)
        dz_embed = masked_embed
        failed["fn"] = lambda: masked_failed["n"]
    elif embedder == "dz-ahc":
        adapter = EMB.make_embedder(kind, device=device, r293_onnx=a.r293_onnx, simam_onnx=a.simam_onnx,
                                    simam_dir=a.simam_dir, espnet_tag=a.espnet_tag)
        params["embedder_id"] = adapter.id_string
        lazy = {}

        def dz_embed(clip, mask):         # the model loads at the first call, i.e. the first cache miss
            if "fn" not in lazy:
                if kind == "r34lm":
                    lazy["fn"], masked_failed = adapter.masked_embed_fn()
                    lazy["failed"] = lambda: masked_failed["n"]
                else:
                    concat = EMB.make_concat_embedder(adapter, min_s=EMB.DEFAULT_MIN_S)
                    lazy["fn"], lazy["failed"] = concat, lambda: concat.counts["refused"]
                failed["fn"] = lazy["failed"]
            return lazy["fn"](clip, mask)
    else:
        embed_clip, providers = build_wespeaker(device)
        params["onnx_providers"] = providers
        print(json.dumps({"status": "embedder", "wespeaker_providers": providers}), flush=True)

    wave, sr = sf.read(a.audio, dtype="float32", always_2d=False)
    if getattr(wave, "ndim", 1) == 2:
        wave = wave.mean(axis=1)
    if sr != SR:
        raise SystemExit(f"expected {SR} Hz audio, got {sr} Hz")
    duration = len(wave) / float(SR)
    if abs(duration - float(plan["duration"])) > 0.05:
        raise SystemExit(f"audio is {duration:.2f} s but the probabilities were made for {plan['duration']:.2f} s")
    embed_wave = wave
    if a.embed_audio:
        ew, esr = sf.read(a.embed_audio, dtype="float32", always_2d=False)
        if getattr(ew, "ndim", 1) == 2:
            ew = ew.mean(axis=1)
        try:
            embed_wave = embedder_waveform(wave, sr, ew, esr)
        except ValueError as exc:
            raise SystemExit(str(exc))
        params["embed_audio"] = a.embed_audio

    cache = None
    if embedder == "dz-ahc":
        # Keyed by the inputs of every embedding: audio, probabilities, mask-making settings, the embed audio. Not by
        # --cos-threshold, which only the clustering reads. Other embedder ids get their own file.
        cache_dir = a.embcache_dir or os.path.join(os.path.dirname(os.path.abspath(a.out)), "embcache")
        inputs = {"audio": file_fingerprint(a.audio),
                  "plan": file_sha256(os.path.join(a.probs_dir, "plan.json")),
                  "pieces": [file_fingerprint(os.path.join(a.probs_dir, e["file"])) for e in plan["pieces"]],
                  "thr": a.thr, "median_frames": median_frames, "median_s": a.median_s,
                  "min_frames_ratio": a.min_frames_ratio, "min_s": EMB.DEFAULT_MIN_S,
                  "embed_audio": file_fingerprint(a.embed_audio) if a.embed_audio else None}
        cache_file = embedding_cache_path(cache_dir, adapter.id_string, inputs)
        cache = EmbeddingCache(cache_file).load()
        params.update({"embedding_cache": cache_file, "embedding_cache_entries_loaded": len(cache.table)})

    def make_prob_fn(piece):
        table = tables[piece_key(piece)]

        def prob_fn(s, e):
            return table[(float(s), float(e))]
        return prob_fn

    progress = {"crops": 0, "failed": 0, "t0": time.time()}

    def make_dz_embed_fn(piece):
        def embed_fn(span, mask):
            progress["crops"] += 1
            if progress["crops"] % 50 == 0:
                print(json.dumps({"status": "embedding", "embeddings": progress["crops"],
                                  "failed": failed["fn"](),
                                  "seconds": round(time.time() - progress["t0"], 1)}), flush=True)
            return dz_embed(crop_for_embedder(embed_wave, piece.audio_start, span), mask)
        return cache.wrap(piece.audio_start, embed_fn) if cache is not None else embed_fn

    def make_embed_fn(piece):
        def embed_fn(spans):
            rows = []
            for s, e in spans:
                try:
                    rows.append(embed_clip(crop_for_embedder(embed_wave, piece.audio_start, (s, e))))
                except ValueError:                       # the embedder refuses this crop; run_piece drops its row
                    rows.append(None)
                    progress["failed"] += 1
                progress["crops"] += 1
                if progress["crops"] % 50 == 0:
                    print(json.dumps({"status": "embedding", "crops": progress["crops"],
                                      "failed": progress["failed"],
                                      "seconds": round(time.time() - progress["t0"], 1)}), flush=True)
                if progress["crops"] >= 20 and progress["failed"] > 0.3 * progress["crops"]:
                    raise RuntimeError(f"the embedder refused {progress['failed']} of {progress['crops']} crops; "
                                       "the ONNX GPU provider may be broken here, try --device cpu")
            dim = next((len(r) for r in rows if r is not None), None)
            if dim is None:
                raise RuntimeError("the embedder refused every crop of this piece")
            return np.stack([r if r is not None else np.full(dim, np.nan) for r in rows])
        return embed_fn

    t0 = time.time()
    if a.clustering in ("dz", "dz-ahc"):
        if a.clustering == "dz":
            win_frames = int(round(float(plan["params"]["window_s"]) / float(plan["frame_s"])))
            min_frames = (math.ceil(win_frames * min_samples / (float(plan["params"]["window_s"]) * SR))
                          if min_samples else 1)
        else:
            # The same formula gives 1 for any min_num_samples up to 1280 (WeSpeaker and pyannote use 400), so 1 here.
            min_frames = 1
        params["min_num_frames"] = min_frames
        result = run_dz_stage(duration, plan, pieces, tables, make_dz_embed_fn, dz_vbx, thr=a.thr,
                              median_frames=median_frames, min_frames_ratio=a.min_frames_ratio,
                              min_num_frames=min_frames, max_speakers=a.max_speakers, min_on_s=a.min_on_s)
        progress["failed"] = failed["fn"]()
        if cache is not None:
            params["embedding_cache_hits"] = cache.hits
            params["embedding_cache_misses"] = cache.misses
            params["embedding_cache_written"] = cache.save()
    else:
        result = wc.run_file(duration, None, make_prob_fn=make_prob_fn, make_embed_fn=make_embed_fn,
                             pieces=pieces, piece_s=plan["params"]["piece_s"], window_s=plan["params"]["window_s"],
                             hop_s=plan["params"]["hop_s"], frame_s=float(plan["frame_s"]), cluster_fn=cluster_fn,
                             max_embed_s=a.max_embed_s, min_speakers=a.min_speakers,
                             **({} if a.clustering == "vbx" else {"ahc_threshold": a.cosine_threshold}))
    params["embedding_padded_crops"] = counts["padded"]
    params["embedding_failed_crops"] = progress["failed"]
    params["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"segments": result["segments"], "pieces": result["pieces"], "frame_s": float(plan["frame_s"]),
                   "params": params}, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"status": "pieces", "clusters": [p["clusters"] for p in result["pieces"]],
                      "stats": [p["stats"] for p in result["pieces"]]}), flush=True)
    print(json.dumps({"status": "done", "segments": len(result["segments"]), "pieces": len(result["pieces"]),
                      "vbx_fallback_pieces": sum(1 for p in result["pieces"] if p["stats"].get("vbx_fallback"))}),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
