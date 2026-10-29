#!/usr/bin/env bash
# Windows (Git Bash): install prebuilt llama.cpp CUDA binaries into
# external/llama.cpp/build/bin (replacing its contents) instead of building
# (no CUDA Toolkit / MSVC needed). The llama.cpp source checkout is still required
# (convert_hf_to_gguf.py, commit pin check in setup.sh).
#
# Release b11007 is commit 2f3fd02, the parent of the pinned c6824a9; the two differ
# only in .github/workflows (git diff --stat 2f3fd02 c6824a9: 6 workflow files).
# llama-bench therefore reports build_commit 2f3fd02 for these binaries.
#
# Usage: scripts/install_llama_win.sh [13.4|12.4]
#   default: 13.4 (native Blackwell sm_120 code), falling back to 12.4 (PTX JIT)
#   when the 13.4 binaries cannot see a CUDA device (e.g. driver older than R580).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source scripts/common.sh

TAG=b11007
URL="https://github.com/ggml-org/llama.cpp/releases/download/$TAG"
BIN_DIR=external/llama.cpp/build/bin
CACHE_DIR=external/downloads
declare -A SHA256=(
  [llama-$TAG-bin-win-cuda-13.4-x64.zip]=b556f369a5429a84ab55dd262122a55e335ec90e88975d505d250bfdf4e5899a
  [cudart-llama-bin-win-cuda-13.4-x64.zip]=738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668
  [llama-$TAG-bin-win-cuda-12.4-x64.zip]=6b33beb85991aaefb17a1b7e79a204f7a64feaad23ff3811605014b503b8f8fa
  [cudart-llama-bin-win-cuda-12.4-x64.zip]=8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6
)

log() { echo "[install_llama_win] $*"; }

fetch() {  # fetch FILE -> verified path in CACHE_DIR
  local f="$1" p="$CACHE_DIR/$1"
  if [ ! -f "$p" ] || [ "$(sha256sum "$p" | cut -d' ' -f1)" != "${SHA256[$f]}" ]; then
    log "downloading $f"
    curl -fL --retry 3 -o "$p.part" "$URL/$f" || return 1
    mv -f "$p.part" "$p" || return 1
  fi
  if [ "$(sha256sum "$p" | cut -d' ' -f1)" != "${SHA256[$f]}" ]; then
    log "FAIL: sha256 mismatch for $f"
    return 1
  fi
}

# Runs inside `if`, where errexit is off: every step returns explicitly.
install() {  # install CUDA_VERSION; returns 0 if a CUDA device is visible
  local v="$1" llama="llama-$TAG-bin-win-cuda-$1-x64.zip" cudart="cudart-llama-bin-win-cuda-$1-x64.zip"
  fetch "$llama" && fetch "$cudart" || return 1
  rm -rf "$BIN_DIR" && mkdir -p "$BIN_DIR" || return 1
  "$PYTHON" -c 'import sys, zipfile
for z in sys.argv[2:]:
    zipfile.ZipFile(z).extractall(sys.argv[1])' "$BIN_DIR" "$CACHE_DIR/$llama" "$CACHE_DIR/$cudart" || return 1
  local devices
  devices="$("$BIN_DIR/llama-cli.exe" --list-devices 2>&1 | tr -d '\r' || true)"
  printf '    %s\n' "${devices//$'\n'/$'\n'    }"
  grep -q 'CUDA0' <<<"$devices" || return 1
  printf 'release %s\ncuda %s\n%s %s\n%s %s\n' "$TAG" "$v" \
    "$llama" "${SHA256[$llama]}" "$cudart" "${SHA256[$cudart]}" > "$BIN_DIR/PREBUILT.txt"
}

[ -x "$PYTHON" ] || { log "FAIL: .venv not found (py -3 -m venv .venv)"; exit 1; }
mkdir -p "$CACHE_DIR"
for v in ${1:-13.4 12.4}; do
  if install "$v"; then
    log "OK: llama.cpp $TAG CUDA $v binaries in $BIN_DIR"
    exit 0
  fi
  log "CUDA $v: no CUDA device visible"
done
log "FAIL: no CUDA device visible with the prebuilt binaries (check: nvidia-smi)"
exit 1
