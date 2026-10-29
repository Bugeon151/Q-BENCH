# Q-Bench

> Quantization Benchmark for Local LLM Inference — 신규 하드웨어(Apple M5) 성능 프로파일링

Qwen2.5-Instruct(1.5B/3B/7B/14B)를 여러 GGUF 양자화 수준(F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K)으로 변환하고, **Apple M5(Metal)** 와 NVIDIA(CUDA) 기기에서 모델 크기 · 메모리 · 처리량 · 레이턴시 · 품질을 정량 측정합니다.

## 실행

```bash
./scripts/run_all.sh --stage perf                     # v0.1 성능 (기본 모델, ctx 4096)
./scripts/run_all.sh --stage ctx                      # v0.3 context 512–8192
./scripts/run_all.sh --stage quality                  # v0.2 PPL + MMLU/GSM8K/HumanEval
./scripts/run_all.sh --stage ctx --model all          # v1.0 전체 모델
```

파이프라인: 환경 검증 → 모델 준비(다운로드/변환/양자화/스모크 테스트) → 벤치마크(model × quant × ctx) / 품질 평가 → 통계 분석 → 시각화 → report. 완료 항목은 SKIP(이어서 실행). 기기별 절차: `docs/RUNBOOK.md`.

테스트: `pip install -r requirements-dev.txt && pytest -q`

## 출력

* `data/raw/<hw_id>/benchmark_raw.csv` — raw 측정 (llama-bench/하드웨어 메타데이터 포함)
* `data/raw/benchmark_raw.csv` — v0.1 legacy raw (수정 금지, 분석 시 `legacy_v0.1`로 읽음)
* `data/quality/<hw_id>/` — perplexity, 과제별 응답/채점
* `data/processed/benchmark_stats.csv` — 통계 요약
* `results/tables/{summary,quality,hardware_comparison}.csv`, `results/summary.json`
* `results/figures/<hw_id>/<model>/` — PRD 그림 1–5, 7, context 그래프 / `results/figures/quality/` — 그림 6 / `results/figures/hardware/`

## 문서

* `docs/PRD.md` — 요구사항 (신규 하드웨어 프로파일링)
* `docs/METRICS.md` — 지표 정의
* `docs/EXPERIMENT.md` — 실험 절차서
* `docs/RUNBOOK.md` — 기기별 실행 순서, 예상 시간
* `docs/` 상세

## 환경

* Apple M5, 24 GB Unified Memory, Metal (성능 측정)
* Intel Core Ultra 9 275HX, 32 GB, RTX 5070 Laptop 8 GB, CUDA, Windows 11 (품질 평가, 하드웨어 비교)
* llama.cpp (pinned: `config/llama_cpp_commit.txt`)
* 하드웨어 프로파일: `config/hardware/<hw_id>/profile.txt` (v0.1: `config/hardware.txt`)