# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Python code and tests live in `podcast-pipeline/`. Run commands from there unless noted.

```bash
# Setup (PyTorch first, then the rest; see README.md for the full order)
pip install torch==2.7.1 torchaudio==2.7.1
pip install -r requirements.txt

# Tests
python -m pytest tests -q
python -m pytest tests/test_separation_logic.py -q        # one file
python -m pytest tests/test_label_tools.py::test_history_is_pruned -q   # one test by name

# Run the pipeline
bash run_test_all.sh                 # edit the `folders` array first
python main.py --audio_dir /path/to/audio --config config.json --job_id myjob
python main.py --audio /path/to/one.mp3 --config config.json
```

- `tests/conftest.py` sets `SOMMELIER_SKIP_BOOTSTRAP=1`. Tests must not build environments, so keep it set when adding tests that import `main.py`.
- No lint or format config was found in the repository. Do not invent one.
- Some tests need GPU model packages (torch, NeMo, pyannote). A few `test_nemotron_two_pass.py` cases fail on a machine without torch; that is expected there.

## Big picture

**Entry and profile.** `main.py` parses the CLI and reads `config.json`. Each `environments.<name>` block holds `steps` (one boolean per stage) and per-environment paths and GPU ids. `utils/steps.py` decides whether a stage runs (`step_enabled`), and maps older flags to the stage names through `LEGACY_FLAG`. Prefer the `steps` block when adding a switch.

**Orchestration.** `services/pipeline_service.py` (`PipelineService`) runs the stages in order: music analysis and removal, diarization, separation, ASR, refinement, speaker relabel, transcript review, word alignment, then export. Each stage writes its outputs through `services/stage_output_service.py`. `utils/checkpoint.py` records progress per `--job_id`, so a rerun resumes from the last completed stage. Stage outputs are checked by size and mtime, not by a full hash on every load.

**Stage-major batching.** `utils/batch.py` (`run_batch_by_stage`) runs a stage over many files before the next one starts. Parallel stages (music, diarization, separation) get an isolated per-file view from `PipelineService.parallel_stage_view`.

**Model workers.** Heavy models run in separate worker processes, each with its own environment, to avoid dependency clashes. Pattern: `<name>_worker.py` (the code that loads and runs the model) with `services/<name>_worker_service.py` (the client that starts it, pools it and collects results). Examples: `nemotron_worker.py`, `diarizen_worker.py`, `whisper_vllm_worker.py`, `qwen3_worker.py`, `sidon_worker.py`, `word_alignment_worker.py`. `worker_vllm_env.py` and `worker_errors.py` hold the shared environment and error handling. `services/worker_pool_service.py` manages the pool.

**Diarization.** Current production diarization is Nemotron (`nemotron_worker.py`, `utils/nemotron_two_pass.py`): a two-pass run over 25 s windows, with speaker labels stitched across windows by WeSpeaker cosine (0.6). Post-processing thresholds live in the profile. DiariZen (`diarizen_worker.py`) is the alternative. Diarization sits in `services/diarization_service.py` and `services/diarization_refinement_service.py`.

**Music and denoise.** `utils/music_map.py` and `utils/excise.py` build a timeline of music and cut it out. The `MUSIC_CUT_SHARE_LIMIT` guard (default 0.60) refuses a music map that would remove most of the audio. Denoise and separation are optional stages, configured per environment.

**Timeline and outputs.** `utils/final_output.py` writes the final layout (`_final/…`). Conversation clips are chosen after relabel and word alignment. Optional exports include a lhotse manifest (`services/lhotse_manifest_service.py`).

**Experiment area vs production.** `tools/` is research and evaluation code, not the pipeline. It contains the arm comparison (`diar_arms_experiment.py`, `compare_arms.py`), the hand-labelling tools (`label_page.html`, `label_server.py`, `label_store.py`), and a large number of one-off `exp*.py` scripts. Do not wire `tools/` code into `services/` without an explicit request. Diarization arms that need a second Python environment take `--windowed-python` and `--vbx-python`.

**Docs.** `doc/` has the working plans and reports (for example `doc/bao_cao_buoc_diarization.md`, `doc/tong_hop_kien_truc_diarization_sota.md`). `AGENTS.md` holds the repository guidelines: commit prefixes, test naming, and the Kaggle workflow.

## Working agreements

- Current work is on branch `exp/denoise-compare`. Commit and push there; do not open a pull request unless asked.
- When a result is measured, say how (command, data, collar, mapping). Diarization numbers from non-hand-labelled data are relative only.
- Comments and docs in this repo are mostly Vietnamese. Match the language of the file you edit.

## Team roles

- **Bộ não (mô hình suy luận, mặc định):** the default agent for planning, analysis, reviewing results and diffs, deciding what to change, and writing the final answer. Use it for judgement, not for repetitive edits.
- **Chân tay (mô hình thực thi):** the agent for concrete work: writing and editing code, searching documentation and data, running commands and tests, collecting files and numbers. Hand it a precise task with the file paths, the expected output and the check that proves it is done.
- Review the execution agent's diff before it is committed.
