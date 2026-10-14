#!/usr/bin/env bash
# Pipeline stage: Environment Validation
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="$ROOT_DIR/.venv"
LLAMA_DIR="$ROOT_DIR/external/llama.cpp"
LOG_FILE="$ROOT_DIR/logs/setup.log"
mkdir -p "$ROOT_DIR/logs"

log() { echo "[setup] $*" | tee -a "$LOG_FILE"; }

if [ ! -x "$VENV_DIR/bin/python" ]; then
  log "FAIL: .venv not found at $VENV_DIR. Create it first (python3 -m venv .venv)."
  exit 1
fi

PYTHON="$VENV_DIR/bin/python"

log "Installing project requirements..."
"$PYTHON" -m pip install -q -r "$ROOT_DIR/requirements.txt"

log "Installing llama.cpp GGUF conversion requirements..."
"$PYTHON" -m pip install -q -r "$LLAMA_DIR/requirements/requirements-convert_hf_to_gguf.txt"

log "Verifying llama.cpp binaries..."
for bin in llama-quantize llama-bench llama-cli llama-perplexity llama-server; do
  BIN_PATH="$LLAMA_DIR/build/bin/$bin"
  if [ ! -x "$BIN_PATH" ]; then
    log "FAIL: missing or non-executable binary: $BIN_PATH"
    exit 1
  fi
  log "OK: $bin found at $BIN_PATH"
done

log "Recording hardware profile..."
HW_ID="$("$PYTHON" "$ROOT_DIR/src/common.py" hwid)"
HW_DIR="$ROOT_DIR/config/hardware/$HW_ID"
mkdir -p "$HW_DIR"
{
  echo "Q-Bench Experimental Environment (hw_id: $HW_ID)"
  echo
  date
  echo
  case "$(uname -s)" in
    Darwin)
      # Device identifiers are filtered out before writing (privacy).
      system_profiler SPHardwareDataType SPDisplaysDataType 2>/dev/null \
        | grep -vE 'Serial Number|Hardware UUID|Provisioning UDID' | sed 's/^/    /'
      echo
      sw_vers
      ;;
    Linux)
      uname -sr
      grep -E '^PRETTY_NAME=' /etc/os-release 2>/dev/null || true
      grep -qi microsoft /proc/version 2>/dev/null && echo "WSL: yes"
      echo
      lscpu 2>/dev/null | grep -vE '^(Flags|Vulnerability)' || true
      echo
      grep -E '^(MemTotal|SwapTotal):' /proc/meminfo
      echo
      if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
      else
        echo "nvidia-smi: not found"
      fi
      ;;
  esac
} > "$HW_DIR/profile.txt"
log "OK: ${HW_DIR#"$ROOT_DIR"/}/profile.txt updated"

"$PYTHON" -m pip freeze > "$HW_DIR/requirements-lock.txt"
log "OK: ${HW_DIR#"$ROOT_DIR"/}/requirements-lock.txt recorded (pip freeze)"

log "Verifying llama.cpp commit pin..."
ACTUAL_COMMIT="$(git -C "$LLAMA_DIR" rev-parse HEAD)"
PINNED_COMMIT="$(cat "$ROOT_DIR/config/llama_cpp_commit.txt" 2>/dev/null | tr -d '[:space:]')"
if [ -z "$PINNED_COMMIT" ]; then
  echo "$ACTUAL_COMMIT" > "$ROOT_DIR/config/llama_cpp_commit.txt"
  log "OK: recorded llama.cpp commit $ACTUAL_COMMIT"
elif [ "$ACTUAL_COMMIT" != "$PINNED_COMMIT" ]; then
  log "FAIL: external/llama.cpp HEAD ($ACTUAL_COMMIT) does not match config/llama_cpp_commit.txt ($PINNED_COMMIT)"
  exit 1
else
  log "OK: llama.cpp commit matches pinned $PINNED_COMMIT"
fi

log "Verifying python packages importable..."
"$PYTHON" - <<'EOF'
import importlib.util
mods = ["yaml", "pandas", "numpy", "scipy", "matplotlib", "psutil", "huggingface_hub", "tqdm", "pyarrow"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"FAIL: missing python modules: {missing}")
print("OK: all required python modules importable")
EOF

log "Environment validation PASSED."
