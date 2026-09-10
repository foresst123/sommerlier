"""Ô nối lại — dán vào Kaggle khi kernel đã restart.

Triệu chứng nó chữa:

    /bin/bash: line 1: {python_bin}: command not found

`python_bin`, `ENV_DIR`, `DZ_PY`, `Q_PY`, `hf_token` là biến trong bộ nhớ kernel.
Restart hoặc nhảy vào giữa notebook là mất, và IPython chuyền nguyên chuỗi
`{python_bin}` cho bash. Ô 2 của notebook không chạy lại được để lấy chúng: nó
`rm -rf` rồi clone lại repo.

Ô này dựng lại mọi đường dẫn TỪ ĐĨA. Không cài gì, không đụng repo đã clone.
Chạy xong thì tiếp tục từ ô 4 (trỏ pipeline vào worker env).

Đây là bản sao của ô 3e trong notebooks/sommelier-full-kaggle.ipynb — sửa thì
sửa cả hai.
"""

import os, glob, shutil

BASE_DIR = '/kaggle/working' if os.path.exists('/kaggle/working') else os.getcwd()
PROJECT_DIR  = os.path.join(BASE_DIR, 'sommerlier')
PIPELINE_DIR = os.path.join(PROJECT_DIR, 'podcast-pipeline')
AUDIO_DIR    = os.path.join(BASE_DIR, 'vi_audio')
OUT_DIR      = os.path.join(BASE_DIR, 'out')
CONSTRAINTS  = os.path.join(BASE_DIR, 'constraints.txt')

# Tim env da cai, thay vi doan: o 2 chon o con nhieu cho nhat luc do, va lan
# nay o do co the khac.
def _find_env(name):
    for root in ("/kaggle/temp", "/tmp", BASE_DIR):
        py = os.path.join(root, name, "bin", "python")
        if os.path.exists(py):
            return os.path.dirname(os.path.dirname(py)), py
    return None, None

ENV_DIR, python_bin      = _find_env("sommelier_env")
DIARIZEN_ENV_DIR, DZ_PY  = _find_env("diarizen_env")
QWEN3_ENV_DIR, Q_PY      = _find_env("qwen3_env")

missing = [n for n, v in (("sommelier_env", python_bin),
                          ("diarizen_env", DZ_PY),
                          ("qwen3_env", Q_PY)) if v is None]
if missing:
    raise SystemExit(f"Chua cai: {missing}. Chay lai cac o 3a-3d.")

ENV_ROOT = os.path.dirname(ENV_DIR)
os.environ["UV_CACHE_DIR"] = os.path.join(ENV_ROOT, ".uv_cache")
TORCH = "torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0"
CU126 = "--extra-index-url https://download.pytorch.org/whl/cu126"

def disk(tag):
    u = shutil.disk_usage(ENV_ROOT)
    print(f"  [{tag}] con {u.free/2**30:.1f}G tren {ENV_ROOT}")

hf_token = os.environ.get("HUGGINGFACE_TOKEN", "")
if not hf_token:
    try:
        from kaggle_secrets import UserSecretsClient
        s = UserSecretsClient()
        for key in ("HUGGINGFACE_TOKEN", "HF_TOKEN"):
            try:
                hf_token = s.get_secret(key) or ""
                if hf_token: break
            except Exception:
                pass
    except Exception:
        pass
if hf_token:
    os.environ["HUGGINGFACE_TOKEN"] = hf_token
    os.environ["HF_TOKEN"] = hf_token

EXTS = ('.mp3','.wav','.flac','.m4a','.aac','.ogg','.opus')
audio_files = sorted(p for p in glob.glob(os.path.join(AUDIO_DIR, '*'))
                     if p.lower().endswith(EXTS))

for label, value in (("ENV_ROOT", ENV_ROOT), ("python_bin", python_bin),
                     ("DZ_PY", DZ_PY), ("Q_PY", Q_PY),
                     ("repo", PIPELINE_DIR)):
    print(f"  {label:12} {value}")
print(f"  {'token':12} {'co' if hf_token else 'CHUA CO — chay lai o 5'}")
print(f"  {'audio':12} {len(audio_files)} file")
print(f"  {'output':12} {OUT_DIR} ({'co' if os.path.exists(OUT_DIR) else 'chua chay'})")
