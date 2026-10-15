# Nemotron-3-Diarization experiment

Optional diarization backend next to DiariZen. Default behaviour, output folder
names and checkpoints for DiariZen runs are unchanged. **The code was written
on a Mac without a GPU or NeMo; it has never run against the real model.**

## What was added

- `podcast-pipeline/nemotron_worker.py`: JSON-lines worker (same protocol and
  segment shape as `diarizen_worker.py`). Loads
  `SortformerEncLabelModel.from_pretrained("nvidia/Nemotron-3-Diarization")` and
  calls `diarize(audio=[path], batch_size=1)`.
- `podcast-pipeline/utils/nemotron_segments.py`: pure conversion (no NeMo/torch):
  raw segments to `{start, end, speaker}`, speakers renamed `SPEAKER_00..` by
  first appearance, optional same-speaker gap merge and minimum duration.
- `podcast-pipeline/services/nemotron_worker_service.py`: thin subclass of the
  DiariZen worker service.
- `podcast-pipeline/utils/diarization_backend.py`: backend selection.
- `podcast-pipeline/requirements-nemotron.txt`: separate env (NeMo pins numba
  0.61.0, which conflicts with the main env, so it is not in `requirements.txt`).
- Wiring in `main.py` (new `--diarization_backend` flag) and
  `services/pipeline_service.py`. The worker occupies the same pipeline slot as
  DiariZen, so downstream code and `model_loader.py` are untouched.
- Tests: `podcast-pipeline/tests/test_nemotron_backend.py`.

## Selecting the backend

Order: `--diarization_backend`, then `models.diarization_backend` in the config
profile, then `diarizen`. Unknown values raise `ValueError`; combining with
`--dia3` is rejected. The profile key is optional and not in `config.json`; add
`"diarization_backend": "nemotron"` under `models` of a profile to use it.

Output folder: a Nemotron run uses `-diaModel-nemotron-...` instead of
`-diaModel-diarizen-...`, so the two never share checkpoints. The `diarizen`
name is untouched.

Optional keys under `models.nemotron` in the profile: `chunk_len` (streaming
parameter in 80 ms frames), `merge_gap`, `min_duration` (seconds, default 0).

## Create the env and run on the DGX

```bash
# next to the repository
python -m venv nemotron_env
nemotron_env/bin/pip install torch==2.8.0 torchaudio==2.8.0
nemotron_env/bin/pip install -r sommerlier/podcast-pipeline/requirements-nemotron.txt
export NEMOTRON_PYTHON=$PWD/nemotron_env/bin/python   # optional, else auto-found

cd sommerlier/podcast-pipeline
# smoke test of the worker alone
echo '{"audio_path": "/abs/path/test.wav"}' | CUDA_VISIBLE_DEVICES=0 \
  $NEMOTRON_PYTHON nemotron_worker.py --config config.json --env a100

# full pipeline, same flags you use for DiariZen plus the backend
python main.py <your usual flags> --env a100 --gpu_1 0 --gpu_2 1 \
  --diarization_backend nemotron
```

## Comparing with DiariZen

Run both backends on the same 3-5 files (different overlap levels, speaker
counts, one music-heavy). Each lands in its own `_final/...-diaModel-<name>-...`
folder. Compare per file:

- speaker count and per-speaker speaking time
- number of raw segments and of segments after merge
- overlap seconds (time with two or more active speakers)
- diarization stage wall time and GPU memory
- DER (collar 0.25 s and 0) if a reference RTTM exists
- downstream: number of kept clips, BSS-failed spans

## Verified from the model card

Loading call, `diarize(audio=[...], batch_size=1)`, segments as
begin/end/speaker index, 16 kHz mono input, up to 8 speakers ordered by first
arrival, offline and streaming modes, OpenMDW 1.1 license.

## UNVERIFIED assumptions (check on the first run)

1. Each segment is a string `"begin end speaker_N"` (the card only says
   "begin_seconds, end_seconds, speaker_index"); `predicted[0]` is the list for
   the first file. The parser also accepts tuples and `{Start, End, Speaker}`.
2. `nemo-toolkit[asr]` unpinned installs a NeMo version that exposes
   `SortformerEncLabelModel` and works with torch 2.8.
3. `.eval()` is enough; the model lands on `cuda:0` without an explicit `.to()`.
4. Pipeline temp WAVs (16 kHz mono) are accepted as-is, including long files
   with default (non-streaming) settings; very long files may need `chunk_len`
   or the streaming presets.
5. `num_speakers/min_speakers/max_speakers` are ignored; the profile's
   `max_speakers: 10` cannot be honoured (the model has 8 slots).
6. `sortformer_modules.chunk_len` plus `_check_streaming_parameters()` is the
   right way to change chunking (from the card).
7. Multiple workers (`performance.stages.diarization.workers > 1`) each fit one
   GPU; `split_components` placement is not applied to this backend.
8. The `-preview` gated repo has an evaluation-only license; this uses the
   main repo.
