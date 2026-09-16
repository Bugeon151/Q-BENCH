#!/usr/bin/env bash
# Full Q-Bench v0.1 pipeline:
# Environment Validation -> Model Validation -> Benchmark (warm-up + measurement)
# -> Raw Data Collection -> Data Processing -> Statistical Analysis -> Visualization -> Report
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs
LOG_FILE="logs/run_all_$(date +%Y%m%dT%H%M%S).log"

log() { echo "[run_all] $*" | tee -a "$LOG_FILE"; }

log "Stage 1/4: Environment Validation"
bash scripts/setup.sh

log "Stage 2/4: Model Validation"
bash scripts/quantize.sh

log "Stage 3/4: Benchmark (warm-up + measurement + raw data collection)"
.venv/bin/python src/benchmark.py

log "Stage 4/4: Data Processing + Statistical Analysis + Visualization + Report"
.venv/bin/python src/analyze.py

log "Q-Bench v0.1 pipeline complete."
