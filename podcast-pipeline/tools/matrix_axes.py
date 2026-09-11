"""Shared, dependency-free experiment definitions."""

import hashlib
import itertools
import json

AXES = {
    "A": ["mel3005", "bsr368"], "B": ["nodenoise", "aufr33"],
    "C": ["nocut", "cut"], "G": ["nog", "ina"],
    "D": ["mdv2", "mlc"], "E": ["nomem", "mem"],
    "F": ["raw", "resemble", "audiosr"],
}
STAGES = ("music", "diarization", "separation")
OWNS = {"music": ("A", "B", "C", "G"),
        "diarization": ("A", "B", "C", "G", "D"),
        "separation": ("A", "B", "C", "G", "D", "E")}
PREFIX = {"music": "s1", "diarization": "s3", "separation": "s4"}
STAGE_DIR = {"music": "01_music", "diarization": "02_diarization",
             "separation": "03_separation"}
MUSIC_MODELS = {
    "mel3005": "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
    "bsr368": "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
}
DENOISE_MODEL = "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt"
DIARIZEN_MODELS = {"mdv2": "BUT-FIT/diarizen-wavlm-large-s80-md-v2",
                   "mlc": "BUT-FIT/diarizen-wavlm-large-s80-mlc"}
POLISH_SUFFIX = {"raw": "", "resemble": ".re", "audiosr": ".sr"}


def prefix_axes(cell, stage):
    return {key: cell[key] for key in OWNS[stage]}


def job_id(cell, stage):
    key = "-".join(f"{k}={cell[k]}" for k in OWNS[stage])
    return f"{PREFIX[stage]}-{hashlib.sha1(key.encode()).hexdigest()[:10]}"


def parents(cell, stage):
    return [job_id(cell, s) for s in STAGES[:STAGES.index(stage)]][::-1]


def combo_id(cell):
    return "_".join(cell[key] for key in AXES)


def cells(selection=None, smoke=False):
    choices = {k: list(v) for k, v in AXES.items()}
    if smoke:
        choices = {k: (v if k in ("D", "E") else v[:1]) for k, v in choices.items()}
    for key, values in (selection or {}).items():
        if key not in AXES or not values or not set(values) <= set(AXES[key]):
            raise ValueError(f"Invalid axis selection: {key}={values}")
        choices[key] = list(values)
    return [dict(zip(choices, values)) for values in itertools.product(*choices.values())]


def fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()
