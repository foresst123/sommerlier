#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_DIR="${1:-${PIPELINE_DIR}/../vllm_env}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it first: https://docs.astral.sh/uv/" >&2
  exit 1
fi

uv venv --python 3.12 "${ENV_DIR}"
uv pip install --python "${ENV_DIR}/bin/python" \
  -r "${PIPELINE_DIR}/requirements-vllm.txt"

"${ENV_DIR}/bin/python" - <<'PY'
import qwen_asr
import vllm
print("vLLM:", vllm.__version__)
print("qwen-asr:", getattr(qwen_asr, "__version__", "installed"))
PY

echo
echo "Environment ready: ${ENV_DIR}"
echo "Run: export VLLM_PYTHON=${ENV_DIR}/bin/python"
