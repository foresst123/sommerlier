# Repository Guidelines

## Project Structure & Module Organization

This repository contains the Sommelier audio-processing pipeline. Core Python code lives in `podcast-pipeline/`: `services/` coordinates stages, `models/` wraps inference backends, `algorithms/` contains diarization/ASR helpers, `schemas/` defines data objects, and `utils/` holds shared audio, timeline, and reporting utilities. Tests live in `podcast-pipeline/tests/`. Operational docs are in `doc/`, assets in `img/`, helper scripts in `scripts/` and `podcast-pipeline/scripts/`, and sample data/config files sit at the repository root.

## Build, Test, and Development Commands

Set up from `podcast-pipeline/`:

```bash
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0
pip install -r requirements.txt
```

Run focused tests before broad checks:

```bash
cd podcast-pipeline
python -m pytest tests/test_separation_logic.py -q
python -m pytest tests -q
```

Run the pipeline with `bash run_test_all.sh` after editing the `folders` array. For direct runs, use `python main.py ...` with the needed stage flags.

## Coding Style & Naming Conventions

Use Python with 4-space indentation, clear function names, and type hints where existing code uses them. Keep comments short and only explain non-obvious audio/timeline logic. Existing code favors explicit stage names (`separation`, `music_map`, `timeline`) and snake_case for functions, variables, and test files.

## Testing Guidelines

Tests use `pytest` and are named `test_*.py`. Add focused regression tests for behavior changes, especially around overlap accounting, timestamp mapping, checkpoint compatibility, and output layout. Prefer small fake models over GPU/model-heavy tests when validating control flow.

## Commit & Pull Request Guidelines

Recent commits use conventional prefixes such as `fix(...)`, `feat(...)`, and `docs:`. Keep commits scoped to one behavior or document update. PR descriptions should explain the trigger, the changed behavior, and the validation command, for example `python -m pytest tests/test_separation_logic.py -q`.

## Agent-Specific Instructions

For Kaggle work, use the external Chrome profile, not the Codex in-app browser. Verify Google Account `lamkdhe180931@fpt.edu.vn`, then open Kaggle. To download notebook outputs, use the Kaggle sidebar: hover/click `/kaggle/working`, expand with the small arrow, hover the target file such as `result.zip`, click its three-dot menu, then choose **Download**. Do not rely on `FileLink` or direct `kkb-production...` URLs; they can open as `404`.
# test push Fri Sep 11 04:21:59 UTC 2026
