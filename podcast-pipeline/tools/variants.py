#!/usr/bin/env python3
"""Variant registry of the diarization arms (J, K and their overlap, Fb and embedder variants), and a dry-run.

    python tools/variants.py list
    python tools/variants.py commands --params tools/variant_params.example.json \
        --samples tools/diar_arms_samples.json --out OUT [--only J_base,K_base] \
        [--python SOM_PY] [--vbx-python DZ_PY] [--windowed-python NEMOTRON_PY] [--demucs-python DEMUCS_PY] \
        [--embedder-python NAME=PATH] [--env a100] [--config config.json]

Nothing runs here: `commands` prints the command lines, and refuses the variants that need a parameter still null.

TWO PHASES, THEN THE RUN
  1. Build and validate the variants (this module). Parameter values are NOT decided here.
  2. Measure the PLDA for each embedder: tools/train_plda.py fits LDA+PLDA on labelled embeddings and writes
     DIR/xvec_transform.npz, plda.npz and plda_meta.json (embedder_id = the kind). Record the folder, the lda_dim
     and the holdout result in a copy of tools/variant_params.example.json (call it params.json).
     Cosine cuts (cos.<kind>) are measured the same way, on hand-labelled clips.
  3. Fill params.json: every null value a variant needs is then a number or a path. `commands` refuses, with the
     parameter name, any variant that still needs a null value.
  4. Run (NOT WRITTEN YET): the printed commands are independent once their sample's shared steps are done. A later
     step runs them in parallel across the two GPUs, one command per GPU with CUDA_VISIBLE_DEVICES=0 or =1, and runs
     each sample's shared steps (stage 1 probabilities, then Demucs for the K branch) before its stage 2 commands.

BRANCHES
  nemo_dz        Nemotron probabilities (stage 1, the same windows as arm J) and DiariZen's steps (stage 2).
  nemo_dz_demucs the same, with the embeddings cut from chunk_vocals.wav (HT Demucs). Arm K.
  nemo_sc        RESERVED. SC-pNA is NOT implemented: its variants have status "todo" and nothing runs them.

STATUS
  ready  the variant has a command line; it still needs its null parameters (if any) filled.
  todo   the variant cannot run yet; the reason is in todo_reason and `commands` refuses it.

The dry-run builds each command with the same helpers the arms script uses (tools/diar_arms_experiment.py:
stage2_cmd, trial_cmd, dump_probs and ensure_vocals), so a variant's command is the arms script's command with the
variant's values. tests/test_variants.py checks that for J.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import diar_arms_experiment as DA  # noqa: E402
from tools import embedders as EMB  # noqa: E402

EXAMPLE_PARAMS = os.path.join(ROOT, "tools", "variant_params.example.json")
BRANCH_LETTER = {"nemo_dz": "J", "nemo_dz_demucs": "K"}
BRANCHES = ("nemo_dz", "nemo_dz_demucs", "nemo_sc")
CLUSTERINGS = ("dz", "dz-ahc")
PLDA_DEFAULT = "diarizen_default"
NOTE_PARAMS = "_note"


@dataclass(frozen=True)
class Variant:
    id: str
    branch: str
    segmentation: str
    embedder_kind: str            # r34lm (DiariZen's ResNet34-LM), wespeaker_r293, wespeaker_simam100, espnet_spk
    clustering: str               # dz = DiariZen's VBx with a PLDA; dz-ahc = cosine average-linkage AHC
    plda: str                     # none | diarizen_default | params:<key> (the PLDA folder)
    cos_threshold: object         # number | params:<key> | None (dz only)
    thr: float                    # slot probability threshold (--thr)
    min_on_s: Optional[float]     # None: no --min-on-s flag
    median_s: float               # --median-s
    fb: float                     # VBx Fb (dz only)
    fa: float                     # VBx Fa (dz only)
    ahc_threshold: float          # VBx AHC init cut (dz only)
    lda_dim: object = None        # None: the script's default (128, no flag); int; params:<key>
    arm_key: Optional[str] = None     # the arms script's key, when the variant is one of its arms (J, K@0.2, ...)
    run_name: Optional[str] = None    # output folder name for variants without an arm key (see run_name())
    status: str = "ready"         # ready | todo
    todo_reason: str = ""
    notes: str = ""


class VariantRefused(Exception):
    """A variant that cannot run now: a parameter is null, or the variant is todo. The message names the cause."""


def _vbx_defaults(letter: str, **over) -> dict:
    """Arm J (or K) with the values of the arms script's defaults, as one dict of Variant fields."""
    branch = next(b for b, l in BRANCH_LETTER.items() if l == letter)
    base = dict(branch=branch, segmentation="nemotron", embedder_kind="r34lm", clustering="dz",
                plda=PLDA_DEFAULT, cos_threshold=None, thr=0.5, min_on_s=None, median_s=0.22, fb=0.8, fa=0.07,
                ahc_threshold=0.6)
    base.update(over)
    return base


def _arm(vid: str, letter: str, arm_key: str, **over) -> Variant:
    fields = _vbx_defaults(letter, **over)
    return Variant(id=vid, arm_key=arm_key, run_name=None, **fields)


def _emb_dzahc(kind: str) -> Variant:
    fields = _vbx_defaults("J", embedder_kind=kind, clustering="dz-ahc", plda="none",
                           cos_threshold=f"params:cos.{kind}",
                           notes="cosine AHC on J's probs and the chunk (arm J's steps, not VBx)")
    return Variant(id=f"emb_{kind}_dzahc", arm_key=None, run_name=None, **fields)


def _emb_vbx(kind: str) -> Variant:
    fields = _vbx_defaults("J", embedder_kind=kind, clustering="dz", plda=f"params:plda.{kind}.dir",
                           lda_dim=f"params:plda.{kind}.lda_dim",
                           notes="VBx with a PLDA re-estimated for this embedder (train_plda.py)")
    if kind == "r34lm":
        return Variant(id="emb_r34lm_vbx", arm_key=None, run_name="nemotron_dz_vbx_r34lm", **fields)
    return Variant(id=f"emb_{kind}_vbx", arm_key=None, run_name=f"nemotron_dz_vbx_{kind}", status="todo",
                   todo_reason="windowed_vbx_stage.py: --clustering dz needs --embedder pyannote (ResNet34) and "
                               "refuses --embedder-kind on the VBx path; the path must take the kind first",
                   **fields)


def _build_registry() -> List[Variant]:
    out: List[Variant] = [
        _arm("J_base", "J", "J", notes="arm J as now: DiariZen's PLDA, no --plda-dir, no --min-on-s"),
        _arm("K_base", "K", "K", notes="arm K as now: J with Demucs vocals for the embeddings"),
    ]
    for letter in ("J", "K"):
        out.append(_arm(f"{letter}_ov_thr0.35_on0.2", letter, f"{letter}+ov|0.35|0.2", thr=0.35, min_on_s=0.2,
                        notes="overlap pair: slot threshold 0.35 and ON runs under 0.2 s dropped (first guesses)"))
    for letter in ("J", "K"):
        for fb in ("0.2", "0.4"):
            out.append(_arm(f"{letter}_fb{fb}", letter, f"{letter}@{fb}", fb=float(fb),
                            notes="Fb sweep value (stage 2 only; first guess, not validated)"))
    for kind in EMB.KINDS:
        out.append(_emb_dzahc(kind))
    for kind in EMB.KINDS:
        out.append(_emb_vbx(kind))
    out.append(Variant(id="sc_pna", branch="nemo_sc", segmentation="sc_pna", embedder_kind="r34lm",
                       clustering="dz", plda=PLDA_DEFAULT, cos_threshold=None, thr=0.5, min_on_s=None,
                       median_s=0.22, fb=0.8, fa=0.07, ahc_threshold=0.6, run_name="nemotron_sc_pna",
                       status="todo", todo_reason="SC-pNA is not implemented: no segmentation or stage code exists",
                       notes="reserved"))
    return out


REGISTRY: List[Variant] = _build_registry()
BY_ID: Dict[str, Variant] = {v.id: v for v in REGISTRY}


def param_refs(v: Variant) -> List[str]:
    """Every parameter key a variant reads, in a fixed order. A VBx variant with its own PLDA also needs the holdout
    result of that PLDA (plda.<kind>.holdout), recorded after measurement."""
    keys: List[str] = []
    for value in (v.plda, v.cos_threshold, v.lda_dim):
        if isinstance(value, str) and value.startswith("params:"):
            keys.append(value[len("params:"):])
    if v.clustering == "dz" and v.plda.startswith("params:"):
        keys.append(v.plda[len("params:"):].rsplit(".", 1)[0] + ".holdout")
    return keys


def _param_value(params: dict, key: str, variant_id: str):
    value = params.get(key)
    if value is None:
        raise VariantRefused(f"{variant_id}: parameter {key!r} is null (filled after measurement); not run")
    return value


def resolve(v: Variant, params: dict) -> dict:
    """The concrete values of a variant with params filled in. Raises VariantRefused, naming the cause."""
    if v.status != "ready":
        raise VariantRefused(f"{v.id}: status todo: {v.todo_reason}")
    for key in param_refs(v):
        _param_value(params, key, v.id)
    plda_dir = None
    if v.plda.startswith("params:"):
        plda_dir = str(_param_value(params, v.plda[len("params:"):], v.id))
    lda_dim = None
    if isinstance(v.lda_dim, str):
        lda_dim = _param_value(params, v.lda_dim[len("params:"):], v.id)
    elif v.lda_dim is not None:
        lda_dim = v.lda_dim
    cos = None
    if isinstance(v.cos_threshold, str):
        cos = float(_param_value(params, v.cos_threshold[len("params:"):], v.id))
    elif v.cos_threshold is not None:
        cos = float(v.cos_threshold)
    return {"plda_dir": plda_dir, "lda_dim": lda_dim, "cos": cos}


def run_name(v: Variant, resolved: dict) -> str:
    """The output folder under runs/, the arms script's name for the same arm (nemotron_dz, nemotron_dz_fb0.2, ...)."""
    if v.clustering == "dz-ahc":
        return DA.arm_name(DA.trial_key(v.embedder_kind, resolved["cos"]))
    if v.arm_key is not None:
        return DA.arm_name(v.arm_key)
    return str(v.run_name)


def check_registry(params_keys: Sequence[str]) -> List[str]:
    """Problems in the registry, as messages; an empty list means it is consistent. params_keys: the example file's keys."""
    problems: List[str] = []
    seen = set()
    for v in REGISTRY:
        if v.id in seen:
            problems.append(f"duplicate variant id {v.id!r}")
        seen.add(v.id)
        if v.status not in ("ready", "todo"):
            problems.append(f"{v.id}: status {v.status!r} is not ready or todo")
        if v.status == "todo" and not v.todo_reason:
            problems.append(f"{v.id}: todo without a reason")
        if v.branch not in BRANCHES:
            problems.append(f"{v.id}: unknown branch {v.branch!r}")
        if v.branch == "nemo_sc" and v.status != "todo":
            problems.append(f"{v.id}: branch nemo_sc must stay todo (SC-pNA is not implemented)")
        if v.embedder_kind not in EMB.KINDS:
            problems.append(f"{v.id}: unknown embedder {v.embedder_kind!r}")
        if v.clustering not in CLUSTERINGS:
            problems.append(f"{v.id}: unknown clustering {v.clustering!r}")
        if v.clustering == "dz-ahc":
            if v.branch != "nemo_dz":
                problems.append(f"{v.id}: dz-ahc is only built on branch nemo_dz (the trial command has no --embed-audio)")
            if v.plda != "none":
                problems.append(f"{v.id}: dz-ahc uses no PLDA; plda must be none")
            if v.cos_threshold is None:
                problems.append(f"{v.id}: dz-ahc needs cos_threshold")
        else:
            if v.cos_threshold is not None:
                problems.append(f"{v.id}: cos_threshold is for dz-ahc only")
            if v.plda == "none":
                problems.append(f"{v.id}: dz (VBx) needs a PLDA")
        if v.arm_key is None and v.clustering == "dz" and v.run_name is None:
            problems.append(f"{v.id}: needs an arm_key or a run_name")
        for key in param_refs(v):
            if key not in params_keys:
                problems.append(f"{v.id}: parameter {key!r} is not in the example file")
    return problems


def load_params(path: str) -> dict:
    """A params JSON object. Keys starting with '_' are notes and are ignored; null means not measured yet."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a JSON object")
    return {k: v for k, v in data.items() if not k.startswith("_")}


def arms_cfg(args_python: str, vbx_python: Optional[str], windowed_python: str, demucs_python: str,
             embedder_python: Dict[str, str], env: str, config: str, gpu: str = "<gpu>") -> dict:
    """The cfg dict of tools/diar_arms_experiment.main, with the arms script's defaults for every value a variant
    does not set. Used by the dry-run and by the tests (which compare with the arms script's own commands)."""
    return {"python": args_python, "vbx_python": vbx_python, "windowed_python": windowed_python,
            "demucs_python": demucs_python, "embedder_python": dict(embedder_python), "env": env, "config": config,
            "gpu": gpu, "windowed_min_speakers": 1, "windowed_piece_s": DA.WINDOWED_PIECE_S,
            "windowed_hop_s": DA.WINDOWED_HOP_S, "dense_window_s": DA.DENSE_WINDOW_S, "dense_hop_s": DA.DENSE_HOP_S,
            "dz_thr": 0.5, "dz_median_s": 0.22, "vbx_ahc_threshold": 0.6, "vbx_fa": 0.07, "vbx_fb": 0.8,
            "plda_dir": None, "r293_onnx": None, "simam_onnx": None, "simam_dir": None,
            "espnet_tag": EMB.DEFAULT_ESPNET_TAG}


def stage1_command(sample_dir: Path, cfg: dict) -> List[str]:
    """The stage 1 command of arm J's probabilities (mirrors DA.dump_probs for J: dense windows, hop 0.8 s)."""
    run_dir = sample_dir / "runs" / DA.ARMS["J"]
    probs = run_dir / "probs"
    return [cfg["windowed_python"], DA.WINDOWED_SCRIPT, "--audio", str(sample_dir / "chunk.wav"),
            "--out", str(run_dir / "unused.json"), "--dump-probs", str(probs),
            "--piece-s", str(cfg["windowed_piece_s"]), "--hop-s", str(cfg["dense_hop_s"]),
            "--config", cfg["config"], "--env", cfg["env"], "--window-s", str(cfg["dense_window_s"])]


def demucs_command(sample_dir: Path, cfg: dict) -> List[str]:
    """The Demucs command of arm K's embedding audio (mirrors DA.ensure_vocals)."""
    return [cfg["demucs_python"], DA.DEMUCS_SCRIPT, "--in", str(sample_dir / "chunk.wav"),
            "--out", str(sample_dir / DA.VOCALS_NAME)]


def stage2_command(v: Variant, sample_dir: Path, cfg: dict, resolved: dict) -> List[str]:
    """The windowed_vbx_stage.py command of one variant on one sample (mirrors DA.stage2_cmd / DA.trial_cmd)."""
    letter = BRANCH_LETTER[v.branch]
    probs = sample_dir / "runs" / DA.ARMS["J"] / "probs"
    result = sample_dir / "runs" / run_name(v, resolved) / "out" / "segments.json"
    vcfg = dict(cfg, dz_thr=v.thr, dz_median_s=v.median_s, vbx_fa=v.fa, vbx_fb=v.fb,
                vbx_ahc_threshold=v.ahc_threshold, plda_dir=resolved["plda_dir"])
    if v.clustering == "dz-ahc":
        return DA.trial_cmd(v.embedder_kind, resolved["cos"], sample_dir, probs, result, vcfg)
    embed = sample_dir / DA.VOCALS_NAME if letter == "K" else None
    cmd = DA.stage2_cmd(letter, sample_dir, probs, result, vcfg, fb=str(v.fb), embed_audio=embed,
                        thr=v.thr, min_on_s=v.min_on_s)
    if resolved["lda_dim"] is not None:
        cmd += ["--lda-dim", str(resolved["lda_dim"])]
    return cmd


def build_plan(variants: Sequence[Variant], params: dict, samples: Sequence[dict], out: Path, cfg: dict):
    """(runnable, refused). runnable: [(sample index, sample name, variant, command)] in sample then variant order;
    refused: [message]. Refusals are per variant, not per sample."""
    runnable, refused = [], []
    resolved_by_id, errors = {}, {}
    for v in variants:
        try:
            resolved_by_id[v.id] = resolve(v, params)
        except VariantRefused as exc:
            errors[v.id] = str(exc)
            refused.append(str(exc))
    for i, sample in enumerate(samples, 1):
        sdir = Path(out) / f"{i:02d}_{DA.ascii_slug(str(sample['name']))}"
        for v in variants:
            if v.id in resolved_by_id:
                cmd = stage2_command(v, sdir, cfg, resolved_by_id[v.id])
                runnable.append((i, str(sample["name"]), v, cmd, sdir, run_name(v, resolved_by_id[v.id])))
    return runnable, refused


def prerequisites(runnable_variants: Sequence[Variant], sample_dir: Path, cfg: dict) -> List[List[str]]:
    """Shared steps of one sample: stage 1 (always, every variant needs J's probabilities) and Demucs (K branch)."""
    out = [stage1_command(sample_dir, cfg)]
    if any(v.branch == "nemo_dz_demucs" for v in runnable_variants):
        out.append(demucs_command(sample_dir, cfg))
    return out


def select(only: Optional[str]) -> List[Variant]:
    if not only:
        return list(REGISTRY)
    ids = [x.strip() for x in only.split(",") if x.strip()]
    unknown = [x for x in ids if x not in BY_ID]
    if unknown:
        raise SystemExit(f"unknown variant ids: {', '.join(unknown)}; see `python tools/variants.py list`")
    return [BY_ID[x] for x in ids]


def cmd_list(_args) -> int:
    print(f"{'id':<28} {'branch':<15} {'emb':<18} {'clust':<7} {'plda':<26} {'status':<6} needs")
    for v in REGISTRY:
        needs = ",".join(param_refs(v)) or "-"
        print(f"{v.id:<28} {v.branch:<15} {v.embedder_kind:<18} {v.clustering:<7} {v.plda:<26} {v.status:<6} {needs}")
    print(f"{len(REGISTRY)} variants")
    return 0


def cmd_commands(a) -> int:
    variants = select(a.only)
    params = load_params(a.params) if a.params else {}
    with open(EXAMPLE_PARAMS, encoding="utf-8") as fh:
        example_keys = [k for k in json.load(fh) if not k.startswith("_")]
    problems = check_registry(example_keys)
    if problems:
        raise SystemExit("registry problems:\n  " + "\n  ".join(problems))
    samples = DA.load_samples(a.samples)
    emb_py = {}
    for item in a.embedder_python or []:
        kind, _, path = item.partition("=")
        if kind not in EMB.KINDS or not path:
            raise SystemExit(f"--embedder-python takes NAME=PATH with NAME one of {', '.join(EMB.KINDS)}")
        emb_py[kind] = path
    cfg = arms_cfg(a.python, a.vbx_python, a.windowed_python, a.demucs_python or a.python, emb_py, a.env, a.config)
    runnable, refused = build_plan(variants, params, samples, Path(a.out), cfg)
    for message in refused:
        print(f"REFUSED {message}", file=sys.stderr)
    ready_ids = {v.id for _, _, v, _, _, _ in runnable}
    for i, sample in enumerate(samples, 1):
        sdir = Path(a.out) / f"{i:02d}_{DA.ascii_slug(str(sample['name']))}"
        used = [v for v in variants if v.id in ready_ids]
        if not used:
            continue
        print(f"# sample {i:02d} {sample['name']}: shared steps (run once, before stage 2)")
        for cmd in prerequisites(used, sdir, cfg):
            print(shlex.join(cmd))
        for idx, name, v, cmd, sd, rname in runnable:
            if idx == i:
                print(f"# {v.id}  ->  runs/{rname}/out/segments.json")
                print(shlex.join(cmd))
    print(f"# {len(runnable)} command(s) printed for {len(samples)} sample(s); {len(refused)} variant(s) refused",
          file=sys.stderr)
    return 2 if refused else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="print the registry")
    c = sub.add_parser("commands", help="print the command lines (nothing runs)")
    c.add_argument("--params", help="params JSON (copy of tools/variant_params.example.json, filled)")
    c.add_argument("--samples", required=True, help="samples JSON (tools/diar_arms_samples.json)")
    c.add_argument("--out", required=True, help="the arms output folder the commands write to")
    c.add_argument("--only", help="comma list of variant ids (default: all)")
    c.add_argument("--python", default="<som-python>", help="som env python: stage 2 of the Nemotron-free arms")
    c.add_argument("--vbx-python", default="<dz-python>", help="DiariZen env python: stage 2 of J, K and the VBx arms")
    c.add_argument("--windowed-python", default="<nemotron-python>", help="Nemotron env python: stage 1")
    c.add_argument("--demucs-python", help="python with demucs (default: --python)")
    c.add_argument("--embedder-python", action="append", default=[], metavar="NAME=PATH")
    c.add_argument("--env", default="a100", help="environment profile in config.json")
    c.add_argument("--config", default=os.path.join(ROOT, "config.json"))
    a = ap.parse_args(argv)
    if a.cmd == "list":
        return cmd_list(a)
    try:
        return cmd_commands(a)
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    sys.exit(main())
