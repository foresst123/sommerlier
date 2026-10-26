# Source me (do not execute):   source tools/denoise_env.sh [--download]
#
# Finds where the checkpoints of rf_crowd, rf_vocals_resurrection, rf_dereverb_anvuew and mpsenet_dns
# live on this machine, exports AUDIO_SEPARATOR_MODEL_DIR and MPSENET_DIR, and prints whether the
# pipeline can use each model. Anything missing is downloaded only when --download is given
# (needs network): the three Roformers through audio-separator, MP-SENet by git clone.
#
# Extra places to look:   DENOISE_SEARCH="/raid /scratch" source tools/denoise_env.sh
# Where downloads go:      DENOISE_WEIGHTS_DIR (default ~/denoise_weights)

_dn_roots="$HOME $DENOISE_SEARCH /data /mnt /raid /scratch /workspace"
_dn_dest="${DENOISE_WEIGHTS_DIR:-$HOME/denoise_weights}"
_dn_download=0
[ "$1" = "--download" ] && _dn_download=1

# checkpoint names come from the backends themselves, so this cannot drift from the code
_dn_ckpts=$(python - <<'PY' 2>/dev/null
from models.denoise import BACKENDS
for n in ("rf_crowd", "rf_vocals_resurrection", "rf_dereverb_anvuew"):
    print(BACKENDS[n].checkpoint)
PY
)
if [ -z "$_dn_ckpts" ]; then
  echo "denoise_env: run this from podcast-pipeline/ with the pipeline's python environment active" >&2
  return 1 2>/dev/null || exit 1
fi

_dn_find_dir() {   # first folder holding file $1 (and download_checks.json, which audio-separator reads on every load)
  local f d
  for f in $(find $_dn_roots -maxdepth 6 -name "$1" 2>/dev/null); do
    d=$(dirname "$f")
    [ -f "$d/download_checks.json" ] && { echo "$d"; return; }
  done
  f=$(find $_dn_roots -maxdepth 6 -name "$1" 2>/dev/null | head -1)
  [ -n "$f" ] && dirname "$f"
}

# --- audio-separator folder -------------------------------------------------------------------------
if [ -z "$AUDIO_SEPARATOR_MODEL_DIR" ] || [ ! -d "$AUDIO_SEPARATOR_MODEL_DIR" ]; then
  unset AUDIO_SEPARATOR_MODEL_DIR
  for _dn_c in $_dn_ckpts; do
    _dn_d=$(_dn_find_dir "$_dn_c")
    [ -n "$_dn_d" ] && { export AUDIO_SEPARATOR_MODEL_DIR="$_dn_d"; break; }
  done
fi
if [ -z "$AUDIO_SEPARATOR_MODEL_DIR" ] && [ "$_dn_download" = 1 ]; then
  export AUDIO_SEPARATOR_MODEL_DIR="$_dn_dest/audio_separator"
fi
if [ -n "$AUDIO_SEPARATOR_MODEL_DIR" ] && [ "$_dn_download" = 1 ]; then
  mkdir -p "$AUDIO_SEPARATOR_MODEL_DIR"
  _dn_absent=""
  for _dn_c in $_dn_ckpts; do [ -f "$AUDIO_SEPARATOR_MODEL_DIR/$_dn_c" ] || _dn_absent="$_dn_absent $_dn_c"; done
  if [ -n "$_dn_absent" ] || [ ! -f "$AUDIO_SEPARATOR_MODEL_DIR/download_checks.json" ]; then
    echo "denoise_env: downloading into $AUDIO_SEPARATOR_MODEL_DIR:${_dn_absent:- (download_checks.json only)}"
    python - $_dn_absent <<'PY'
import os, sys
from audio_separator.separator import Separator
s = Separator(model_file_dir=os.environ["AUDIO_SEPARATOR_MODEL_DIR"], log_level=40)
for ckpt in sys.argv[1:] or [None]:
    if ckpt:
        s.load_model(ckpt)          # fetches the checkpoint, its YAML and download_checks.json
    else:
        s.list_supported_model_files()
PY
  fi
fi

# --- MP-SENet clone ---------------------------------------------------------------------------------
if [ -z "$MPSENET_DIR" ] || [ ! -f "$MPSENET_DIR/best_ckpt/g_best_dns" ]; then
  unset MPSENET_DIR
  _dn_f=$(find $_dn_roots -maxdepth 6 -path '*best_ckpt/g_best_dns' 2>/dev/null | head -1)
  [ -n "$_dn_f" ] && export MPSENET_DIR=$(dirname "$(dirname "$_dn_f")")
fi
if [ -z "$MPSENET_DIR" ] && [ "$_dn_download" = 1 ]; then
  echo "denoise_env: cloning MP-SENet into $_dn_dest/MP-SENet"
  mkdir -p "$_dn_dest"
  git clone --depth 1 https://github.com/yxlu-0102/MP-SENet "$_dn_dest/MP-SENet" \
    && export MPSENET_DIR="$_dn_dest/MP-SENet"
fi

echo "AUDIO_SEPARATOR_MODEL_DIR=${AUDIO_SEPARATOR_MODEL_DIR:-<chưa tìm thấy>}"
echo "MPSENET_DIR=${MPSENET_DIR:-<chưa tìm thấy>}"
python - <<'PY'
from models.denoise import available_backends
r = available_backends()
for n in ("rf_crowd", "rf_vocals_resurrection", "rf_dereverb_anvuew", "mpsenet_dns"):
    ok, why = r[n]
    print(f"  {'OK   ' if ok else 'THIẾU'} {n}{'' if ok else ' - ' + why}")
PY
echo "(chưa tìm thấy gì thì chạy lại: source tools/denoise_env.sh --download)"
unset _dn_roots _dn_dest _dn_download _dn_ckpts _dn_c _dn_d _dn_f _dn_absent
unset -f _dn_find_dir
