# Repository Guidelines

## Project Structure & Module Organization

The main Python pipeline lives in `podcast-pipeline/`. Stage orchestration belongs in `services/`, inference wrappers in `models/`, ASR and diarization logic in `algorithms/`, data contracts in `schemas/`, and shared audio, timeline, checkpoint, and reporting code in `utils/`. Worker entry points sit at the pipeline root.

Tests are in `podcast-pipeline/tests/`. Use `tools/` only for research, evaluation, and lab utilities; do not couple experiments into production services without an explicit requirement. Supporting material lives in `scripts/`, `doc/`, `docs/superpowers/`, `notebooks/`, `data/`, and `img/`. `training/minicpmo45_audio_text/` is an independent training package.

## Build, Test, and Development Commands

Run pipeline commands from `podcast-pipeline/`:

```bash
python -m pip install -r requirements.txt
python main.py --help
python main.py --audio /path/to/file --config config.json
python main.py --audio_dir /path/to/directory --config config.json
python -m pytest tests/test_separation_logic.py -q
python -m pytest tests -q
```

`requirements.txt` is authoritative and currently pins Torch 2.8.0. Do not use the obsolete commands in `README.md` or `run_test_all.sh`; they reference missing entry points.

For `training/minicpmo45_audio_text/`, create a Python 3.11 virtual environment, run `pip install -e ".[dev]"`, then `pytest`.

## Coding Style & Configuration

Use 4-space indentation, `snake_case`, descriptive stage-oriented names, and type hints where practical. Keep comments brief and reserve them for non-obvious audio or timeline behavior. No repository-wide formatter or linter is configured.

Treat `config.json` profiles as environment-specific configuration. Keep Nemotron, vLLM, and MambAttention dependencies isolated. Never commit secrets, tokens, model weights, source audio, caches, or generated outputs; use `HUGGINGFACE_TOKEN`, `HF_TOKEN`, or `OPENAI_API_KEY` as applicable.

## Testing Guidelines

Use `pytest` and name files `test_*.py`. Add focused regression tests with fake workers or models; avoid GPU-heavy inference when testing control flow. `tests/conftest.py` disables environment bootstrap. Do not assume the suite is green—report the exact command and failures observed.

## Commit and PR Guidelines

Follow scoped Conventional Commit forms such as `feat(scope):`, `fix(scope):`, `perf(scope):`, `refactor(scope):`, `chore:`, and `docs:`. Keep each commit to one behavior or documentation change. Do not open a pull request unless requested.

For Kaggle tasks, use the external Chrome profile and verify the intended Google account before proceeding.
