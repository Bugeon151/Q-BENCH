#!/usr/bin/env bash
# Q-Bench pipeline. Completed items are skipped on re-run (resume).
#
# Usage: scripts/run_all.sh [--stage perf|ctx|quality|all] [--model NAME|all]
#   perf    v0.1  benchmark at inference.context_length
#   ctx     v0.3  benchmark over inference.contexts
#   quality v0.2  perplexity + task accuracy (llama-server)
#   all     perf + ctx + quality
# Default: --stage perf, --model = config default_model. Analysis always runs last.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAGE="perf"
MODEL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --stage) STAGE="${2:?--stage needs a value}"; shift 2 ;;
    --model) MODEL="${2:?--model needs a value}"; shift 2 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$STAGE" in perf|ctx|quality|all) ;; *) echo "invalid --stage: $STAGE" >&2; exit 2 ;; esac

source scripts/common.sh
MODEL_ARGS=()
[ -n "$MODEL" ] && MODEL_ARGS=(--model "$MODEL")

mkdir -p logs
LOG_FILE="logs/run_all_$(date +%Y%m%dT%H%M%S).log"
log() { echo "[run_all] $*" | tee -a "$LOG_FILE"; }

log "stage=$STAGE model=${MODEL:-<default>}"
log "Environment Validation"
bash scripts/setup.sh

log "Model Validation"
bash scripts/quantize.sh "$MODEL"

if [ "$STAGE" = perf ] || [ "$STAGE" = all ]; then
  log "Benchmark (v0.1, default ctx)"
  "$PYTHON" src/benchmark.py ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"} --contexts default
fi
if [ "$STAGE" = ctx ] || [ "$STAGE" = all ]; then
  log "Benchmark (v0.3, context sweep)"
  "$PYTHON" src/benchmark.py ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"} --contexts all
fi
if [ "$STAGE" = quality ] || [ "$STAGE" = all ]; then
  log "Perplexity (v0.2)"
  "$PYTHON" src/perplexity_eval.py ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"}
  log "Task accuracy (v0.2)"
  "$PYTHON" src/quality_eval.py ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"}
fi

log "Analysis + Visualization + Report"
"$PYTHON" src/analyze.py
"$PYTHON" src/perfmodel.py
"$PYTHON" src/consistency.py
"$PYTHON" src/frontier.py
"$PYTHON" src/loadtest.py --summarize-only
"$PYTHON" src/report.py

log "Q-Bench pipeline complete (stage=$STAGE)."
