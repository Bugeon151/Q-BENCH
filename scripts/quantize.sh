#!/usr/bin/env bash
# Pipeline stage: Model Validation (download -> convert -> quantize -> smoke test)
# Usage: scripts/quantize.sh [MODEL|all]   (default: config default_model)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/common.sh"
COMMON="$ROOT_DIR/src/common.py"
LOG_FILE="$ROOT_DIR/logs/model_validation.log"
mkdir -p "$ROOT_DIR/logs"

log() { echo "[quantize] $*" | tee -a "$LOG_FILE"; }

read_cfg() {
  py - "$1" <<'EOF'
import sys, yaml
node = yaml.safe_load(open("config/experiment.yaml"))
for part in sys.argv[1].split("."):
    node = node[part]
print(node)
EOF
}

HF_ROOT="$(read_cfg model_dirs.hf)"
BASE_ROOT="$(read_cfg model_dirs.base)"
QUANT_OUT_DIR="$(read_cfg quantization.output_dir)"
LLAMA_QUANTIZE="$(py "$COMMON" bin benchmark.llama_quantize_bin)"
LLAMA_CLI="$(py "$COMMON" bin benchmark.llama_cli_bin)"

QUANT_LEVELS=()
while IFS= read -r lvl; do QUANT_LEVELS+=("$lvl"); done < <(py - <<'EOF'
import yaml
for lvl in yaml.safe_load(open("config/experiment.yaml"))["quantization"]["levels"]:
    print(lvl)
EOF
)

MODEL_LINES=()
while IFS= read -r line; do MODEL_LINES+=("$line"); done < <(py "$COMMON" models "${1:-}")

mkdir -p "$QUANT_OUT_DIR"
for LINE in "${MODEL_LINES[@]}"; do
  MODEL="${LINE%% *}"
  HF_REPO="${LINE#* }"
  LOCAL_HF_DIR="$HF_ROOT/$MODEL"
  BASE_GGUF="$BASE_ROOT/$MODEL-F16.gguf"

  # 1. Download base checkpoint
  if [ -d "$LOCAL_HF_DIR" ] && [ -n "$(ls -A "$LOCAL_HF_DIR" 2>/dev/null)" ]; then
    log "SKIP download: $LOCAL_HF_DIR already populated"
  else
    log "Downloading $HF_REPO -> $LOCAL_HF_DIR"
    mkdir -p "$LOCAL_HF_DIR"
    "$PYTHON" - "$HF_REPO" "$LOCAL_HF_DIR" <<'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2],
                  ignore_patterns=["*.bin", "*.h5", "*.msgpack", "*.ot", "*.gguf"])
EOF
    log "OK: download complete"
  fi

  # 2. Convert to baseline F16 GGUF
  if [ -f "$BASE_GGUF" ]; then
    log "SKIP convert: $BASE_GGUF already exists"
  else
    log "Converting $LOCAL_HF_DIR -> $BASE_GGUF (F16)"
    mkdir -p "$(dirname "$BASE_GGUF")"
    "$PYTHON" "$ROOT_DIR/external/llama.cpp/convert_hf_to_gguf.py" \
      "$LOCAL_HF_DIR" --outfile "$BASE_GGUF" --outtype f16
    log "OK: baseline GGUF created"
  fi

  # 3. Quantize into each target level
  ALL_GGUFS=("F16:$BASE_GGUF")
  for LEVEL in "${QUANT_LEVELS[@]}"; do
    OUT_GGUF="$QUANT_OUT_DIR/${MODEL}-${LEVEL}.gguf"
    ALL_GGUFS+=("$LEVEL:$OUT_GGUF")
    if [ -f "$OUT_GGUF" ]; then
      log "SKIP quantize $LEVEL: $OUT_GGUF already exists"
    else
      log "Quantizing $LEVEL -> $OUT_GGUF"
      "$LLAMA_QUANTIZE" "$BASE_GGUF" "$OUT_GGUF" "$LEVEL"
      log "OK: $LEVEL created"
    fi
  done

  # 4. Smoke test (skipped for combos over this device's memory budget)
  for ENTRY in "${ALL_GGUFS[@]}"; do
    LEVEL="${ENTRY%%:*}"
    GGUF="${ENTRY#*:}"
    if ! WHY="$(py "$COMMON" fits "$MODEL" "$LEVEL" 512)"; then
      log "SKIP smoke test $GGUF: over memory budget ($WHY)"
      continue
    fi
    log "Smoke testing $GGUF"
    if ! "$LLAMA_CLI" -m "$GGUF" -p "Hello" -n 8 -c 512 -st --simple-io >>"$LOG_FILE" 2>&1; then
      log "FAIL: smoke test failed for $GGUF"
      exit 1
    fi
    log "OK: $GGUF loads and generates"
  done
done

log "All models prepared and validated."
