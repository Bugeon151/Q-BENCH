# Q-Bench

> Quantization Benchmark for Local LLM Inference — 양자화가 로컬 LLM의 속도·정확도·답변 일관성·서빙 용량에 미치는 영향을 측정하고, 그 측정으로 배포를 결정하는 도구

Qwen2.5-Instruct(1.5B/3B/7B/14B)를 여러 GGUF 양자화 수준(F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K, 저비트는 imatrix 재양자화 포함)으로 변환하고, 모델 크기 · 메모리 · 처리량 · 레이턴시 · perplexity · 과제 정확도 · 답변 flip과 재실행 노이즈 · 동시 사용자 용량을 측정합니다. v0.1 성능 기준선은 Apple M5(Metal), v0.2–v1.0은 RTX 5070 Laptop(CUDA)에서 측정했습니다.

## 결과 요약 (RTX 5070 Laptop 8 GB, 전체 해석은 [`docs/REPORT.md`](docs/REPORT.md), 수치는 [`docs/RESULTS.md`](docs/RESULTS.md))

* **같은 Q3도 모델마다 다르다.** Q3_K_M은 7B·14B의 HumanEval을 2점 미만 깎지만 3B는 73.2 → 6.1로 무너뜨린다. 3B Q2_K는 perplexity 22,582로 아무것도 못 맞힌다. CPU 전용 대조로 GPU 커널이 아니라 가중치 문제임을 확인했고, 같은 크기의 imatrix 재양자화로 3B Q2_K는 PPL 11.9, 정확도 50/72/46%로 회복된다.
* **정확도가 같아도 답이 바뀐다.** 7B Q4_K_M은 평균 정확도 −0.5점이지만 답의 4.4%가 정답↔오답으로 뒤집힌다. <!-- README_NOISE -->
* **164문항 HumanEval로는 Q4 교체를 증명할 수 없다.** 필요한 문항 수는 flip 비율이 정한다: 7B Q8_0은 160문항이면 되지만 Q4_K_M은 798문항이 필요하다.
* **같은 메모리면 큰 모델 저비트가 이긴다, 약 4비트까지 (imatrix면 3비트까지).** 2.8 GiB 7B Q2_K는 일반 양자화로는 3.1 GiB 3B Q8_0에 −2.7점이지만, imatrix로는 +4.1점(p = 0.006). Pareto frontier에 F16·Q8_0은 하나도 없다.
* **1분짜리 perplexity가 flip을 예측한다.** 27개 설정에서 Spearman ρ = 0.88. 전체 평가 전 스크린으로 쓴다.
* **정확도 순위와 서빙 용량 순위는 다르다.** 7B Q6_K와 Q4_K_M의 정확도 차이는 통계적으로 구분되지 않는다(984문항 대응 비교 −0.3점, 95% CI −1.6~+1.0). 그런데 interactive SLA(MLPerf v5.0)에서 Q4_K_M은 4명, Q6_K는 1명을 감당한다. `recommend.py --users 4 --sla interactive`가 이 차이를 반영한다.
* **배포는 모델 이름이 아니라 매니페스트다.** `deploy.py`는 선택, GGUF·서버 바이너리 SHA-256, 이 기기에서의 항목별 인수 테스트, 용량을 JSON으로 남긴다. 두 번 실행해도 선택·해시·인수 결과가 같았다.

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