"""Environment variables the denoise lab exports inside every GPU job.

Some denoise backends run in their own conda env or need downloaded weights, and are SKIPPED by
`run_variants.py --only-available` until a variable points at them (REUSE_PYTHON, AUDIO_SEPARATOR_MODEL_DIR, ...).
`lab_server.py` imports this file at start-up, resolves every variable and exports the result in the job, so nothing
has to be typed on the command line. Edit CANDIDATES for your machine; nothing else needs to change.

For each variable the first of these wins:
    1. --set-env NAME=VALUE on the lab_server command line
    2. NAME already set in the shell that started lab_server
    3. the first existing path in CANDIDATES[NAME]   (a "$OTHER" entry means "the value of $OTHER, if set")
A variable with no existing candidate is left out and reported at start-up, so you see what is still missing.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HOME = Path.home()
CONDA_ROOTS = [HOME / "miniforge3", HOME / "miniconda3", HOME / "anaconda3", HOME / "mambaforge"]


def _conda_envs(home: Path = HOME, environ=os.environ) -> List[Path]:
    """`<conda base>/envs` folders: the active conda first, then the usual install places."""
    roots = []
    exe = environ.get("CONDA_EXE")
    if exe:
        roots.append(Path(exe).resolve().parent.parent)
    roots += [home / n for n in ("miniforge3", "miniconda3", "anaconda3", "mambaforge")]
    return [r / "envs" for r in roots]


def default_candidates(home: Path = HOME, environ=os.environ) -> Dict[str, list]:
    envs = _conda_envs(home, environ)
    py = lambda name: [e / name / "bin" / "python" for e in envs]
    return {
        # RE-USE (re_use, re_use_laughter): own env with mamba_ssm + the downloaded Hugging Face folder (model.safetensors)
        "REUSE_PYTHON": py("reuse"),
        "REUSE_DIR": [home / "NemotronLabs_REUSE", home / "data" / "NemotronLabs_REUSE", home / "models" / "NemotronLabs_REUSE"],
        "REUSE_LAUGHTER_DIR": [home / "NemotronLabs_REUSE_laughter", home / "data" / "NemotronLabs_REUSE_laughter"],
        # audio-separator checkpoints (rf_denoise, rf_denoise_aggr, rf_vocals_*, rf_dereverb, uvr_deecho, rf_crowd):
        # the folder with the .ckpt + .yaml + download_checks.json. The BS-RoFormer folder the pipeline already uses is reused.
        "AUDIO_SEPARATOR_MODEL_DIR": ["$BS_ROFORMER_MODEL_DIR", home / "audio_separator_models", home / "models" / "audio_separator",
                                      home / "data" / "audio_separator_models"],
        # MambAttention: own env + a clone of the repo
        "MAMBATTENTION_PYTHON": py("mambattention"),
        "MAMBATTENTION_DIR": [home / "MambAttention", home / "sommerlier" / "MambAttention", home / "models" / "MambAttention"],
        # other backends that need a clone / checkpoint (leave out when you do not use them)
        "URGENT26_DIR": [home / "urgent2026_challenge_track1"],
        "URGENT26_BSRNN_CKPT": [home / "urgent2026_challenge_track1" / "bsrnn.ckpt"],
        "DBS_TIGER_DIR": [home / "Dynamic-Band-Splitting"],
        "DFN_MODEL_DIR": [],
    }


def resolve(cli: Optional[Dict[str, str]] = None, environ=os.environ, candidates: Optional[Dict[str, list]] = None,
            home: Path = HOME) -> Tuple[Dict[str, str], List[str]]:
    """-> ({NAME: value} to export in the job, report lines for the start-up log)."""
    cands = default_candidates(home, environ) if candidates is None else candidates
    cli = dict(cli or {})
    out: Dict[str, str] = {}
    report: List[str] = []
    for name, options in cands.items():
        if name in cli:
            out[name] = cli.pop(name)
            report.append(f"{name}={out[name]}  (--set-env)")
            continue
        if environ.get(name):
            out[name] = environ[name]
            report.append(f"{name}={out[name]}  (shell)")
            continue
        found = None
        for opt in options:
            if isinstance(opt, str) and opt.startswith("$"):
                val = environ.get(opt[1:])
                if val and Path(os.path.expandvars(os.path.expanduser(val))).exists():
                    found = (val, f"${opt[1:]}")
                    break
            elif Path(opt).exists():
                found = (str(opt), "file")
                break
        if found:
            out[name] = found[0]
            report.append(f"{name}={found[0]}  ({found[1]})")
        elif options:
            report.append(f"{name}: KHÔNG TÌM THẤY (đã thử: {', '.join(str(o) for o in options[:3])}{' ...' if len(options) > 3 else ''})")
    for name, val in cli.items():  # --set-env for a name this file does not know
        out[name] = val
        report.append(f"{name}={val}  (--set-env)")
    return out, report
