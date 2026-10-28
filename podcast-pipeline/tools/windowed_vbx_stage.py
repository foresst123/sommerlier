#!/usr/bin/env python3
"""Arm G, stage 2: VBx global clustering of the windowed Nemotron probabilities. Runs in the `som` env.

    <som python> tools/windowed_vbx_stage.py --audio chunk.wav --probs-dir DIR --out segments.json \
        [--plda-dir DIR] [--embedding-model pyannote/wespeaker-voxceleb-resnet34-LM] [--ahc-threshold 0.6] \
        [--fa 0.07] [--fb 0.8] [--lda-dim 128] [--max-iters 20] [--init-smoothing 7.0] [--device cuda:0]

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
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

import utils.windowed_clustering as wc  # noqa: E402

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
    cluster_vbx, vbx_setup = import_vbx()
    plda_dir = find_plda_dir(a.plda_dir)
    xvec_tf, plda_tf, plda_psi = vbx_setup(plda_dir)
    plan, pieces, tables = load_plan(a.probs_dir)
    params = {"Fa": a.fa, "Fb": a.fb, "ahc_threshold": a.ahc_threshold, "lda_dim": a.lda_dim,
              "max_iters": a.max_iters, "init_smoothing": a.init_smoothing, "plda_dir": plda_dir,
              "embedding_model": a.embedding_model, "device": device,
              "window_s": plan["params"]["window_s"], "hop_s": plan["params"]["hop_s"],
              "piece_s": plan["params"]["piece_s"]}
    print(json.dumps({"status": "start", "plda_dir": plda_dir, "plda_candidates": plda_candidates(),
                      "pieces": len(pieces), "params": params}, ensure_ascii=False), flush=True)

    emb, _torch, emb_path = build_embedder(a.embedding_model, device)
    params["embedding_path"] = emb_path
    embed_clip, counts = make_clip_embedder(emb, torch, device)

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

    def make_embed_fn(piece):
        def embed_fn(spans):
            rows = [embed_clip(wave[idx(piece.audio_start + s): idx(piece.audio_start + e)]) for s, e in spans]
            return np.stack(rows)
        return embed_fn

    cluster_fn = wc.make_vbx_cluster_fn(cluster_vbx, xvec_tf, plda_tf, plda_psi, ahc_threshold=a.ahc_threshold,
                                        Fa=a.fa, Fb=a.fb, lda_dim=a.lda_dim, max_iters=a.max_iters,
                                        init_smoothing=a.init_smoothing)
    t0 = time.time()
    result = wc.run_file(duration, None, make_prob_fn=make_prob_fn, make_embed_fn=make_embed_fn,
                         pieces=pieces, piece_s=plan["params"]["piece_s"], window_s=plan["params"]["window_s"],
                         hop_s=plan["params"]["hop_s"], frame_s=float(plan["frame_s"]), cluster_fn=cluster_fn)
    params["embedding_padded_crops"] = counts["padded"]
    params["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"segments": result["segments"], "pieces": result["pieces"], "frame_s": float(plan["frame_s"]),
                   "params": params}, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"status": "done", "segments": len(result["segments"]), "pieces": len(result["pieces"]),
                      "vbx_fallback_pieces": sum(1 for p in result["pieces"] if p["stats"].get("vbx_fallback"))}),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
