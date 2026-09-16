# Experiment Protocol

Q-Bench v0.1 — 신규 하드웨어(Apple M5) 성능 프로파일링 실험 절차서.

## 목적

Apple M5(Metal)에서 양자화 수준별 LLM 추론 성능 기준선 데이터를 수집한다.

## 실험 설정

`config/experiment.yaml` 참조.

| 항목 | 값 |
|---|---|
| Base model | Qwen/Qwen2.5-7B-Instruct |
| Quant levels | F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K |
| Context length | 4096 |
| Max output | 256 tokens |
| Temperature / Seed | 0 / 42 |
| Warm-up / Measurement | 3 / 10 runs |
| Backend | Metal |
| llama.cpp | pinned `config/llama_cpp_commit.txt` |

## 절차

### 1. 환경 검증 (`scripts/setup.sh`)

* venv / requirements 확인
* llama-quantize, llama-bench, llama-cli 존재 확인
* `config/hardware.txt` 하드웨어 프로파일 자동 기록
* llama.cpp commit pin 확인

### 2. 모델 준비 (`scripts/quantize.sh`)

1. HF 원본 checkpoint 다운로드 → `models/huggingface/`
2. F16 GGUF 변환 → `models/base/*-F16.gguf`
3. 6개 quant level 생성 → `models/quantized/`
4. 전 모델 스모크 테스트 (llama-cli)

### 3. 벤치마크 (`src/benchmark.py`)

각 모델 × 3 test group (pp / tg / pg):

* warm-up 3회 (폐기)
* 측정 10회 → `data/raw/benchmark_raw.csv` (hardware metadata 포함)

### 4. 분석/시각화 (`src/analyze.py`)

* 통계 계산 → `data/processed/benchmark_stats.csv`
* quant별 요약 → `results/tables/summary.csv`
* 그래프 5종 → `results/figures/`
* `results/summary.json` (실험 config + hardware + 결과)

## 실행

```bash
./scripts/run_all.sh
```

## 결과 해석 관점

* F16 대비 quant level별 크기/메모리 감소율
* throughput 변화 (prompt processing vs generation)
* TTFT / 레이턴시
* M5 기준선 데이터 기록 (향후 타 하드웨어와 수직 비교용)

## 재현 조건

* 동일 하드웨어 (Apple M5)
* 동일 llama.cpp commit
* 동일 config
* 기존 raw/processed 결과를 재사용하므로, 재실행해도 `data/`는 보존됨