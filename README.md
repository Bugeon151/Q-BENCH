# Q-Bench

> Quantization Benchmark for Local LLM Inference — 신규 하드웨어(Apple M5) 성능 프로파일링

동일한 기반 모델(Qwen2.5-7B-Instruct)을 여러 GGUF 양자화 수준(F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K)으로 변환하고, **Apple M5(Metal)** 에서 모델 크기 · 메모리 · 처리량 · 레이턴시를 정량 측정하여 기준선 데이터를 수집합니다.

## 실행

```bash
./scripts/run_all.sh
```

파이프라인: 환경 검증 → 모델 준비(다운로드/변환/양자화/스모크 테스트) → 벤치마크(pp/tg/pg) → 통계 분석 → 시각화 → report.

## 출력

* `data/raw/benchmark_raw.csv` — raw 측정 (hardware metadata 포함)
* `data/processed/benchmark_stats.csv` — 통계 요약
* `results/tables/summary.csv`, `results/summary.json` — quant별 요약
* `results/figures/` — 그래프 5종

## 문서

* `docs/PRD.md` — 요구사항 (신규 하드웨어 프로파일링)
* `docs/METRICS.md` — 지표 정의
* `docs/EXPERIMENT.md` — 실험 절차서
* `docs/` 상세

## 환경

* Apple M5, 24 GB Unified Memory, Metal
* llama.cpp (pinned: `config/llama_cpp_commit.txt`)
* 하드웨어 프로파일: `config/hardware.txt`