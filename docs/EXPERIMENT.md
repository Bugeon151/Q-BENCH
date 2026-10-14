# Experiment Protocol

Q-Bench 실험 절차서 (v0.1 → v1.0). 기기별 실행 명령은 `docs/RUNBOOK.md`.

## 목적

Apple M5(Metal)에서 양자화 수준별 LLM 추론 성능 기준선 데이터를 수집한다.

## 실험 설정

`config/experiment.yaml` 참조.

| 항목 | 값 |
|---|---|
| Models | Qwen2.5-1.5B / 3B / 7B / 14B-Instruct (기본: 7B, `--model`) |
| Quant levels | F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K |
| Context length | 4096 (v0.1), 512 / 2048 / 4096 / 8192 (v0.3) |
| Max output | 256 tokens (tg는 KV depth = ctx에서 측정) |
| -ngl / -fa | 99 / auto (config `benchmark`) |
| Temperature / Seed | 0 / 42 — 품질 평가 전용. llama-bench 경로에서는 사용되지 않음 |
| Warm-up / Measurement | 3 / 10 runs |
| Backend | 런타임 감지 (Metal / CUDA / CPU) |
| llama.cpp | pinned `config/llama_cpp_commit.txt` |
| 반복 순서 | model × quant × ctx × (pp, tg, pg) |

## 절차

### 1. 환경 검증 (`scripts/setup.sh`)

* venv / requirements 확인
* llama-quantize, llama-bench, llama-cli 존재 확인
* llama-perplexity, llama-server 존재 확인
* `config/hardware/<hw_id>/profile.txt` 하드웨어 프로파일 기록 (macOS: Serial/UUID/UDID 제거한 system_profiler, Linux: lscpu, /proc/meminfo, nvidia-smi)
* `config/hardware/<hw_id>/requirements-lock.txt` (pip freeze)
* llama.cpp commit pin 확인
* `config/hardware.txt`는 v0.1 프로파일(식별자 제거본)로 보존

### 2. 모델 준비 (`scripts/quantize.sh`)

1. HF 원본 checkpoint 다운로드 → `models/huggingface/`
2. F16 GGUF 변환 → `models/base/*-F16.gguf`
3. 6개 quant level 생성 → `models/quantized/`
4. 스모크 테스트 (llama-cli). 메모리 예산 초과 조합은 SKIP

### 3. 벤치마크 (`src/benchmark.py`)

model × quant × ctx × 3 test group (pp / tg / pg):

* 메모리 예산 검사: GGUF 크기 + f16 KV cache(ctx) + overhead(추정)가 예산을 넘으면 SKIP하고 로그 기록 (`memory_budget`)
* warm-up 3회 (폐기)
* 측정 10회 → `data/raw/<hw_id>/benchmark_raw.csv` (llama-bench 메타데이터, OS/HW, peak RSS/VRAM, swap 전후 포함)
* 기존 `data/raw/benchmark_raw.csv`는 legacy(v0.1)로 읽기만 함

### 3b. 품질 평가 (v0.2, `src/perplexity_eval.py`, `src/quality_eval.py`)

* wikitext-2 raw test perplexity → `data/quality/<hw_id>/perplexity.csv`
* llama-server(-np 슬롯) + OpenAI 호환 API로 MMLU 570 / GSM8K 250 / HumanEval 164 → `data/quality/<hw_id>/`
* **HumanEval은 모델이 생성한 코드를 이 기기에서 실행합니다 (subprocess + timeout, 샌드박스 아님).**

### 4. 분석/시각화 (`src/analyze.py`)

* 통계 계산 → `data/processed/benchmark_stats.csv`
* 요약 (hw_id, model, ctx, quant) → `results/tables/summary.csv` (CV > 3% 경고 포함)
* 품질 / Quality Retention → `results/tables/quality.csv`
* 하드웨어 간 비교 → `results/tables/hardware_comparison.csv`
* 그래프 (GB 단위, 95% CI error bar) → `results/figures/<hw_id>/<model>/`, `results/figures/quality/`, `results/figures/hardware/`
* `results/summary.json` (실험 config + hardware + 결과)

## 실행

```bash
./scripts/run_all.sh [--stage perf|ctx|quality|all] [--model NAME|all]
```

NVIDIA 기기용 llama.cpp 빌드: `cmake -B build -DGGML_CUDA=ON` (RTX 50 시리즈는 CUDA 12.8 이상). 상세는 RUNBOOK.

## 결과 해석 관점

* F16 대비 quant level별 크기/메모리 감소율
* throughput 변화 (prompt processing vs generation)
* TTFT / 레이턴시
* M5 기준선 데이터 기록 (향후 타 하드웨어와 수직 비교용)

## 재현 조건

* 동일 하드웨어 (hw_id)
* 동일 llama.cpp commit
* 동일 config
* 동일 Python 패키지 (`config/hardware/<hw_id>/requirements-lock.txt`)
* 기존 raw 결과를 재사용(SKIP)하므로, 재실행해도 `data/raw/`는 보존됨