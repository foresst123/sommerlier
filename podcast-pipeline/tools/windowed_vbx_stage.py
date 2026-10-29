#!/usr/bin/env python3
"""Arms F and G, stage 2: global clustering of the windowed Nemotron probabilities. Runs in the `som` env.

    <som python> tools/windowed_vbx_stage.py --audio chunk.wav --probs-dir DIR --out segments.json \
        [--clustering vbx|ahc] [--embedder pyannote|wespeaker] [--max-embed-s 6] \
        [--plda-dir DIR] [--embedding-model pyannote/wespeaker-voxceleb-resnet34-LM] [--ahc-threshold 0.6] \
        [--fa 0.07] [--fb 0.8] [--lda-dim 128] [--max-iters 20] [--init-smoothing 7.0] [--device cuda:0]

Arm J is --clustering dz: DiariZen's own pipeline steps (utils/dz_pipeline.py) with masked embeddings of the whole
window, a per-frame speaker count and top-k reconstruction. Run it with the python of the DiariZen env, which has
the pyannote 3.1.1 fork DiariZen is built on.

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
    """pyannote WeSpeaker ResNet34 (the embedding model DiariZen's PLDA was trained on). UNVERIFIED API."""
    try:
        import torch
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", required=True, help="16 kHz mono WAV, the same chunk stage 1 read")
    ap.add_argument("--probs-dir", required=True, help="folder written by windowed_nemotron_run.py --dump-probs")
    ap.add_argument("--out", required=True, help="segments.json to write")
    ap.add_argument("--clustering", choices=("vbx", "ahc", "dz"), default="vbx",
                    help="vbx = arm G, ahc = arm F (constrained AHC at --cosine-threshold), dz = arm J (DiariZen steps)")
    ap.add_argument("--embedder", choices=("pyannote", "wespeaker"),
                    help="default: pyannote for vbx, wespeaker (ResNet-293 ONNX) for ahc")
    ap.add_argument("--cosine-threshold", type=float, default=0.5, help="cosine cut of the AHC of --clustering ahc")
    ap.add_argument("--min-speakers", type=int, default=1,
                    help="clusters per piece at least; below it VBx falls back to AHC (arm F's AHC stops at it too)")
    ap.add_argument("--thr", type=float, default=0.5, help="dz: slot probability threshold of the local segmentation")
    ap.add_argument("--median-frames", type=int, default=3,
                    help="dz: median filter along time of each window's binarised slots (DiariZen: 11 frames of 20 ms)")
    ap.add_argument("--min-frames-ratio", type=float, default=0.1,
                    help="dz: share of a window's frames a slot needs alone to enter the clustering (DiariZen 0.1)")
    ap.add_argument("--max-speakers", type=int, default=20, help="dz: cap of the per-frame speaker count")
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
    a = ap.parse_args(argv)

    import soundfile as sf
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(f"windowed_vbx_stage needs torch in this env (som): {exc}")

    device = a.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
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
            params.update({"thr": a.thr, "median_frames": a.median_frames, "min_frames_ratio": a.min_frames_ratio,
                           "max_speakers": a.max_speakers})
        start = {"plda_dir": plda_dir, "plda_candidates": plda_candidates()}
    else:
        params["cosine_threshold"] = a.cosine_threshold
        start = {}
    print(json.dumps({"status": "start", "pieces": len(pieces), "params": params, **start},
                     ensure_ascii=False), flush=True)

    counts = {"padded": 0}
    if embedder == "pyannote":
        emb, _torch, emb_path = build_embedder(a.embedding_model, device)
        params["embedding_path"] = emb_path
        embed_clip, counts = make_clip_embedder(emb, torch, device)
        masked_embed, masked_failed = make_masked_embedder(emb, torch, device)
        min_samples = getattr(emb, "min_num_samples", None)
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

    def idx(t):
        return int(round(t * SR))

    def make_prob_fn(piece):
        table = tables[piece_key(piece)]

        def prob_fn(s, e):
            return table[(float(s), float(e))]
        return prob_fn

    progress = {"crops": 0, "failed": 0, "t0": time.time()}

    def make_dz_embed_fn(piece):
        win_s = float(plan["params"]["window_s"])

        def embed_fn(span, mask):
            progress["crops"] += 1
            if progress["crops"] % 50 == 0:
                print(json.dumps({"status": "embedding", "embeddings": progress["crops"],
                                  "failed": masked_failed["n"],
                                  "seconds": round(time.time() - progress["t0"], 1)}), flush=True)
            return masked_embed(wave[idx(piece.audio_start + span[0]): idx(piece.audio_start + span[1])], mask)
        return embed_fn

    def make_embed_fn(piece):
        def embed_fn(spans):
            rows = []
            for s, e in spans:
                try:
                    rows.append(embed_clip(wave[idx(piece.audio_start + s): idx(piece.audio_start + e)]))
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
    if a.clustering == "dz":
        win_frames = int(round(float(plan["params"]["window_s"]) / float(plan["frame_s"])))
        min_frames = (math.ceil(win_frames * min_samples / (float(plan["params"]["window_s"]) * SR))
                      if min_samples else 1)
        params["min_num_frames"] = min_frames
        result = wc.run_file(duration, None, make_prob_fn=make_prob_fn, make_embed_fn=make_dz_embed_fn,
                             pieces=pieces, piece_s=plan["params"]["piece_s"], window_s=plan["params"]["window_s"],
                             hop_s=plan["params"]["hop_s"], frame_s=float(plan["frame_s"]), piece_fn=dz.run_piece_dz,
                             vbx_fn=dz_vbx, thr=a.thr, median_frames=a.median_frames,
                             min_frames_ratio=a.min_frames_ratio, min_num_frames=min_frames,
                             max_speakers=a.max_speakers)
        progress["failed"] = masked_failed["n"]
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
