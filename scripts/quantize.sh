#!/usr/bin/env bash
# Pipeline stage: Model Validation (download -> convert -> quantize -> smoke test)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="$ROOT_DIR/.venv/bin/python"
CONFIG="$ROOT_DIR/config/experiment.yaml"
LOG_FILE="$ROOT_DIR/logs/model_validation.log"
mkdir -p "$ROOT_DIR/logs"

log() { echo "[quantize] $*" | tee -a "$LOG_FILE"; }

read_cfg() {
  "$PYTHON" - "$1" <<'EOF'
import sys, yaml
cfg = yaml.safe_load(open("config/experiment.yaml"))
key = sys.argv[1]
node = cfg
for part in key.split("."):
    node = node[part]
print(node)
EOF
}

HF_REPO="$(read_cfg base_model.hf_repo)"
LOCAL_HF_DIR="$(read_cfg base_model.local_hf_dir)"
BASE_GGUF="$(read_cfg base_model.base_gguf)"
QUANT_OUT_DIR="$(read_cfg quantization.output_dir)"
LLAMA_QUANTIZE="$(read_cfg benchmark.llama_quantize_bin)"
LLAMA_CLI="$(read_cfg benchmark.llama_cli_bin)"

QUANT_LEVELS=()
while IFS= read -r lvl; do
  QUANT_LEVELS+=("$lvl")
done < <("$PYTHON" - <<'EOF'
import yaml
cfg = yaml.safe_load(open("config/experiment.yaml"))
for lvl in cfg["quantization"]["levels"]:
    print(lvl)
EOF
)

# 1. Download base checkpoint
if [ -d "$LOCAL_HF_DIR" ] && [ -n "$(ls -A "$LOCAL_HF_DIR" 2>/dev/null)" ]; then
  log "SKIP download: $LOCAL_HF_DIR already populated"
else
  log "Downloading $HF_REPO -> $LOCAL_HF_DIR"
  mkdir -p "$LOCAL_HF_DIR"
  "$PYTHON" - "$HF_REPO" "$LOCAL_HF_DIR" <<'EOF'
import sys
from huggingface_hub import snapshot_download
repo_id, local_dir = sys.argv[1], sys.argv[2]
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    ignore_patterns=["*.bin", "*.h5", "*.msgpack", "*.ot", "*.gguf"],
)
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
mkdir -p "$QUANT_OUT_DIR"
MODEL_STEM="$(basename "$BASE_GGUF" | sed -E 's/-F16\.gguf$//')"
for LEVEL in "${QUANT_LEVELS[@]}"; do
  OUT_GGUF="$QUANT_OUT_DIR/${MODEL_STEM}-${LEVEL}.gguf"
  if [ -f "$OUT_GGUF" ]; then
    log "SKIP quantize $LEVEL: $OUT_GGUF already exists"
  else
    log "Quantizing $LEVEL -> $OUT_GGUF"
    "$LLAMA_QUANTIZE" "$BASE_GGUF" "$OUT_GGUF" "$LEVEL"
    log "OK: $LEVEL created"
  fi
done

# 4. Smoke test every produced model (F16 + all quant levels)
ALL_GGUFS=("$BASE_GGUF")
for LEVEL in "${QUANT_LEVELS[@]}"; do
  ALL_GGUFS+=("$QUANT_OUT_DIR/${MODEL_STEM}-${LEVEL}.gguf")
done

for GGUF in "${ALL_GGUFS[@]}"; do
  log "Smoke testing $GGUF"
  if ! "$LLAMA_CLI" -m "$GGUF" -p "Hello" -n 8 -c 512 -st --simple-io >>"$LOG_FILE" 2>&1; then
    log "FAIL: smoke test failed for $GGUF"
    exit 1
  fi
  log "OK: $GGUF loads and generates"
done

log "All models prepared and validated."
