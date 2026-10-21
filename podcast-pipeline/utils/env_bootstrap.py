"""Make sure each worker's virtualenv and the HuggingFace token exist before a run.

Replaces the setup notebook cells: an environment is checked first (its interpreter exists
and its probe import works) and only built when that check fails. Torch / CUDA wheels are
pinned to the versions the workers were tested with.
"""

import getpass
import json
import os
import subprocess
import sys
import urllib.request

TORCH_PINS = ["torch==2.8.0", "torchaudio==2.8.0", "torchvision==0.23.0"]
TORCH_INDEX = ["--extra-index-url", "https://download.pytorch.org/whl/cu126"]

DIARIZEN_PIN = "844f5555b0a98acd0931511fc641a8c5b8ba92c7"
SIDON_PIN = "c8cde2b24e4c77c599ad43a9871140cdc9beeffa"
VLLM_WHEEL = ("https://github.com/vllm-project/vllm/releases/download/v0.23.0/"
              "vllm-0.23.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl")

# worker name (as resolve_worker_python sees it) -> probe import proving the env is usable
PROBES = {
    "sidon": "import torch, sidon",
    "qwen3": "from transformers import AutoProcessor, AutoModelForMultimodalLM",
    "diarizen": "import torch, diarizen, pyannote.audio",
    "refinement": "from transformers import AutoModelForCausalLM",
    "vllm": "import vllm, qwen_asr",
}


def env_root() -> str:
    """Where the `<name>_env` folders live; PIPELINE_ENV_ROOT overrides the default."""
    explicit = os.environ.get("PIPELINE_ENV_ROOT")
    if explicit:
        return explicit
    for root in ("/kaggle/temp", "/kaggle/working"):
        if os.path.isdir(root):
            return root
    # Beside the repository, not inside it: a `git clean` or a fresh clone must not take the
    # venvs with it. resolve_worker_python searches this same place (<pipeline>/../..).
    pipeline_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.abspath(os.path.join(pipeline_dir, "..", ".."))


def _python(env_dir: str) -> str:
    return os.path.join(env_dir, "bin", "python")


def _run(cmd, logger):
    logger.info("[bootstrap] $ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _uv_pip(env_dir, args, logger):
    _run(["uv", "pip", "install", "--python", env_dir, *args], logger)


def _venv(env_dir, logger, python="3.12"):
    _run(["uv", "venv", "--allow-existing", env_dir, "--python", python], logger)


def is_healthy(env_dir: str, worker: str, run=subprocess.run) -> bool:
    """True when the env's interpreter exists and its probe import succeeds."""
    py = _python(env_dir)
    if not os.path.exists(py):
        return False
    result = run([py, "-c", PROBES[worker]], capture_output=True)
    return result.returncode == 0


def _ensure_uv(logger):
    if subprocess.run(["uv", "--version"], capture_output=True).returncode != 0:
        _run([sys.executable, "-m", "pip", "install", "-q", "uv"], logger)


# ---------------------------------------------------------------- the main interpreter

# One import per heavy requirement; if any is missing the whole requirements file is applied.
MAIN_PROBES = ("torch", "torchaudio", "numpy", "librosa", "soundfile", "transformers",
               "pyannote.audio", "whisperx", "faster_whisper", "audio_separator")
ORT_GPU = ["onnxruntime-gpu==1.23.2", "nvidia-npp-cu12", "wrapt==2.3.0"]


def missing_main_modules(find_spec=None):
    """Modules from MAIN_PROBES the running interpreter cannot import."""
    import importlib.util
    find_spec = find_spec or importlib.util.find_spec
    missing = []
    for name in MAIN_PROBES:
        try:
            found = find_spec(name)
        except (ImportError, ValueError):       # a parent package that is itself missing
            found = None
        if found is None:
            missing.append(name)
    return missing


def ensure_main_env(logger, requirements, missing=None, python=None, ensure_uv=None, run=None):
    """Install the pipeline's own requirements into the running interpreter when any is missing.

    Must run before the first ``import torch``. Torch is installed first, pinned, from the
    CUDA 12.6 index; the CPU ``onnxruntime`` some audio packages pull in is then replaced by
    the GPU wheel, which owns the same module tree.
    """
    missing = missing_main_modules() if missing is None else missing
    if not missing:
        logger.info("[bootstrap] main environment OK")
        return False
    python = python or sys.executable
    ensure_uv = ensure_uv or _ensure_uv
    run = run or _run
    logger.warning(f"[bootstrap] main environment is missing {', '.join(missing)}; "
                   f"installing {requirements} into {python}")
    ensure_uv(logger)
    run(["uv", "pip", "install", "--python", python, *TORCH_PINS, *TORCH_INDEX], logger)
    run(["uv", "pip", "install", "--python", python, "-r", requirements, *TORCH_INDEX,
         "--index-strategy", "unsafe-best-match"], logger)
    run(["uv", "pip", "install", "--python", python, *ORT_GPU, "--reinstall", "--no-deps"],
        logger)
    still = missing_main_modules()
    if still:
        raise RuntimeError(f"main environment still lacks {', '.join(still)} after installing "
                           f"{requirements}")
    return True


def ensure_ffmpeg(logger, which=None, run=None, conda_prefix=None):
    """Make sure `ffmpeg` is on PATH; inside a conda env, install it from conda-forge.

    audio-separator (BS-RoFormer) shells out to ffmpeg, so without it every music span fails
    and the mixture is kept. Without a conda env there is nothing installed here: apt needs
    root, so the missing binary is reported with the command to run.
    """
    import shutil
    which = which or shutil.which
    if which("ffmpeg"):
        return False
    conda_prefix = os.environ.get("CONDA_PREFIX") if conda_prefix is None else conda_prefix
    if not conda_prefix or not which("conda"):
        raise RuntimeError(
            "ffmpeg is not installed and this is not a conda environment. Install it with "
            "`sudo apt install ffmpeg` (or `conda install -c conda-forge ffmpeg`) and rerun.")
    logger.warning("[bootstrap] ffmpeg is missing; installing it from conda-forge")
    (run or _run)(["conda", "install", "-y", "-p", conda_prefix, "-c", "conda-forge", "ffmpeg"],
                  logger)
    # conda puts it in <prefix>/bin, which is on PATH only for an activated env.
    bin_dir = os.path.join(conda_prefix, "bin")
    if bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    if not which("ffmpeg"):
        raise RuntimeError("ffmpeg is still not on PATH after installing it with conda")
    return True


# ---------------------------------------------------------------- per-worker installers

def _install_qwen3(env_dir, logger):
    _venv(env_dir, logger)
    _uv_pip(env_dir, [*TORCH_PINS, *TORCH_INDEX], logger)
    _uv_pip(env_dir, ["transformers>=5.13.0", "huggingface-hub>=1.5.0", "accelerate",
                      "soundfile", "librosa"], logger)


def _install_refinement(env_dir, logger):
    _venv(env_dir, logger)
    _uv_pip(env_dir, ["torch==2.8.0", *TORCH_INDEX], logger)
    _uv_pip(env_dir, ["transformers>=5.2.0", "huggingface-hub>=1.5.0", "accelerate"], logger)


def _install_vllm(env_dir, logger):
    _venv(env_dir, logger)
    _uv_pip(env_dir, ["--torch-backend=cu129", f"vllm[audio] @ {VLLM_WHEEL}",
                      "qwen-asr==0.0.6", "soundfile>=0.13,<1"], logger)


def _install_diarizen(env_dir, logger):
    git = "git+https://github.com/BUTSpeechFIT/DiariZen.git@" + DIARIZEN_PIN
    _venv(env_dir, logger)
    _uv_pip(env_dir, [*TORCH_PINS, "--index-url", "https://download.pytorch.org/whl/cu126"],
            logger)
    _uv_pip(env_dir, ["numpy==1.26.4", "scipy==1.13.1", "pandas==2.2.3",
                      "numba==0.59.1", "llvmlite==0.42.0"], logger)
    _uv_pip(env_dir, ["soundfile", "librosa==0.10.2.post1", "matplotlib==3.9.4",
                      "pyparsing", "accelerate==1.12.0"], logger)
    _uv_pip(env_dir, [f"{git}#subdirectory=pyannote-audio"], logger)
    _uv_pip(env_dir, [git], logger)
    _uv_pip(env_dir, ["speechbrain==1.0.2", "toml==0.10.2", "wrapt==2.3.0", "psutil"], logger)


def _flash_attn_wheel(py, repo, torch_ver, abi, pages=3):
    names = []
    for page in range(1, pages + 1):
        try:
            req = urllib.request.Request(
                f"https://api.github.com/repos/{repo}/releases?per_page=100&page={page}",
                headers={"Accept": "application/vnd.github+json"})
            releases = json.load(urllib.request.urlopen(req, timeout=30))
        except Exception:
            break
        if not releases:
            break
        names += [a["browser_download_url"] for r in releases for a in r.get("assets", [])
                  if a["name"].startswith("flash_attn-") and f"torch{torch_ver}" in a["name"]
                  and "cp312" in a["name"] and "linux_x86_64" in a["name"]]
    return (next((u for u in names if f"cxx11abi{abi}" in u), None)
            or next((u for u in names if "cxx11abi" not in u), None))


def _install_sidon(env_dir, logger):
    py = _python(env_dir)
    _venv(env_dir, logger)
    _uv_pip(env_dir, [*TORCH_PINS, *TORCH_INDEX], logger)
    _uv_pip(env_dir, ["hatchling", "packaging", "ninja"], logger)

    # flash-attn from a prebuilt wheel: the host has no nvcc. Without a wheel Sidon uses SDPA.
    torch_ver, abi = subprocess.check_output(
        [py, "-c", "import torch;print('.'.join(torch.__version__.split('+')[0].split('.')[:2]));"
                   "print('TRUE' if torch.compiled_with_cxx11_abi() else 'FALSE')"],
        text=True).split()
    url = (_flash_attn_wheel(py, "Dao-AILab/flash-attention", torch_ver, abi)
           or _flash_attn_wheel(py, "mjun0812/flash-attention-prebuild-wheels", torch_ver, abi, 2))
    if url:
        _uv_pip(env_dir, [url], logger)
        if subprocess.run([py, "-c", "import flash_attn"]).returncode:
            logger.warning("[bootstrap] flash-attn wheel has the wrong ABI; removing it")
            subprocess.run(["uv", "pip", "uninstall", "--python", env_dir, "flash-attn"])
    else:
        logger.warning("[bootstrap] no flash-attn wheel for this torch; Sidon will use SDPA")

    # --no-deps so Sidon does not drag flash-attn in to build from source.
    _uv_pip(env_dir, ["--no-deps",
                      f"git+https://github.com/sarulab-speech/Sidon.git@{SIDON_PIN}"], logger)
    reqs = subprocess.check_output(
        [py, "-c", "import importlib.metadata as m;print('\\n'.join(m.requires('sidon') or []))"],
        text=True)
    deps = [s.split(";")[0].strip() for s in reqs.splitlines() if s.strip()]
    deps = [d for d in deps
            if not d.lower().replace("_", "-").startswith(("flash-attn", "torch"))]
    torch_args = ["torch==2.8.0", "torchaudio==2.8.0", *TORCH_INDEX]
    try:
        _uv_pip(env_dir, [*deps, *torch_args], logger)
    except subprocess.CalledProcessError:
        logger.warning("[bootstrap] dependency conflict; dropping training-only packages")
        _uv_pip(env_dir, [*[d for d in deps if not d.startswith(("datasets", "s3fs", "ray"))],
                          *torch_args], logger)


INSTALLERS = {
    "sidon": _install_sidon,
    "qwen3": _install_qwen3,
    "diarizen": _install_diarizen,
    "refinement": _install_refinement,
    "vllm": _install_vllm,
}


def ensure_worker_envs(workers, logger, root=None, check=is_healthy, installers=None):
    """Build any of ``workers``' virtualenvs that is missing or broken; return their paths.

    An explicit ``{WORKER}_PYTHON`` variable means the user manages that env: it is left alone.
    """
    root = root or env_root()
    installers = installers or INSTALLERS
    ready = {}
    uv_checked = False
    for worker in workers:
        if os.environ.get(f"{worker.upper()}_PYTHON"):
            continue
        env_dir = os.path.join(root, f"{worker}_env")
        if check(env_dir, worker):
            logger.info(f"[bootstrap] {worker} env OK: {env_dir}")
            ready[worker] = env_dir
            continue
        if not uv_checked:
            _ensure_uv(logger)
            uv_checked = True
        logger.warning(f"[bootstrap] {worker} env missing or broken; building {env_dir}")
        installers[worker](env_dir, logger)
        if not check(env_dir, worker):
            raise RuntimeError(f"{worker} env at {env_dir} still fails its probe import "
                               f"({PROBES[worker]!r}) after installing")
        ready[worker] = env_dir
    return ready


# ---------------------------------------------------------------- HuggingFace token

def _cached_hf_token():
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


def ensure_hf_token(config, logger, prompt=getpass.getpass, interactive=None, required=True):
    """Return a HuggingFace token, asking for one when none is configured.

    Order: config value, HUGGINGFACE_TOKEN / HF_TOKEN, a token saved by ``huggingface-cli
    login``, then a hidden prompt. The token is exported to the environment for this process
    and its workers only; it is never written to config.json.

    With ``required=False`` nothing is asked: a token found is exported, otherwise None is
    returned and whatever really needs one (the gated pyannote pipeline) asks when it loads.
    """
    token = (config.get("huggingface_token")
             or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
             or _cached_hf_token())
    if not token and not required:
        logger.info("[bootstrap] no HuggingFace token set; it will only be asked for if a "
                    "gated model (--dia3) is loaded")
        return None
    if not token:
        if interactive is None:
            interactive = sys.stdin.isatty()
        if not interactive:
            raise RuntimeError(
                "No HuggingFace token (pyannote models are gated). Set HF_TOKEN, add a "
                "Kaggle secret, or run `huggingface-cli login`.")
        token = prompt("HuggingFace token (needs pyannote gated-model access): ").strip()
        if not token:
            raise RuntimeError("A HuggingFace token is required.")
    config["huggingface_token"] = token
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    logger.info("[bootstrap] HuggingFace token ready")
    return token


# ---------------------------------------------------------------- input / output paths

def _input_prefilled(text, default):
    """input() with `default` already typed on the line, so it can be edited in place.

    None when the terminal cannot do it (no readline, or a build without the pre-input hook,
    such as macOS libedit); the caller then falls back to a plain prompt.
    """
    try:
        import readline
        hook = readline.set_pre_input_hook
    except (ImportError, AttributeError):
        return None

    def prefill():
        readline.insert_text(default)
        readline.redisplay()

    hook(prefill)
    try:
        return input(text)
    finally:
        hook()


def ask_paths(args, config, prompt=input, interactive=None):
    """Ask for the input folder and the clean-data output folder, defaulting to config.json.

    On a terminal the default is typed onto the line for the user to edit; clearing the line
    leaves the answer empty (output export off).

    Defaults: ``entrypoint.input_folder_path`` and the active profile's
    ``outputs.clean_two_channel_dir`` (empty there means the export is off, so Enter keeps it
    off). ``--audio`` / ``--audio_dir`` on the command line skip the input question. Without
    a terminal the config values are used silently. The answers are stored in ``args`` and
    ``config``, which is where the pipeline reads them.
    """
    if interactive is None:
        interactive = sys.stdin.isatty()

    def ask(label, default):
        if not interactive:
            return default
        if prompt is input:
            typed = _input_prefilled(f"{label}: ", default)
            if typed is not None:       # the default was editable text; what is left is the answer
                return typed.strip()
        shown = f" [{default}]" if default else " [off]"
        return prompt(f"{label}{shown}: ").strip() or default

    if not (args.audio or args.audio_dir):
        default = (config.get("entrypoint") or {}).get("input_folder_path", "")
        args.audio_dir = ask("Input audio folder", default) or None

    profile = config.setdefault("environments", {}).setdefault(args.env, {})
    outputs = profile.setdefault("outputs", {})
    outputs["clean_two_channel_dir"] = ask("Output folder (clean two-channel data)",
                                           outputs.get("clean_two_channel_dir", ""))
    return args
