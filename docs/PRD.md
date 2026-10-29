# Q-Bench

> Quantization Benchmark for Local LLM Inference
> 신규 하드웨어(Apple M5) 성능 프로파일링 관점 벤치마크

> **범위 변경 (2026-10-30)** — 소유자 결정으로 M5(맥북)에서는 더 이상 실행하지 않는다.
> M5 데이터는 v0.1 결과(7B, ctx 4096)만 `legacy_v0.1`로 유지하고, v0.2–v1.0(품질, context,
> 1.5B/3B/7B/14B)은 RTX 5070 Laptop 8 GB(Windows 11, CUDA)에서 측정했다. 14B F16은 이 노트북의
> RAM 31 GiB + VRAM 8 GB에 들어가지 않아(스왑 발생) 품질 평가에서 제외하고 14B는 Q8_0을 기준으로
> 비교한다. PRD 이후 추가된 분석·도구(회귀 게이트, 답변 뒤집힘과 재실행 노이즈, 크기 대 비트,
> perplexity 스크린, 동시 사용자 용량, 디코드 속도 예측, imatrix 복구, 배포 키트)는
> `docs/REPORT.md`에 정리한다.

## 1. 프로젝트 개요

Q-Bench는 **신규 하드웨어(Apple M5)에서 로컬 LLM 추론의 양자화 수준별 성능 특성을 프로파일링**하기 위한 벤치마크 프로젝트이다.

동일한 기반 모델을 여러 양자화 방식으로 변환하고, 동일한 하드웨어 및 추론 조건에서 실행하여 모델 크기, 메모리 사용량, 추론 속도 사이의 trade-off를 정량적으로 측정한다.

측정 결과는 **신규 하드웨어의 실제 추론 성능 기준선(first-hand benchmark data)** 으로 남겨 둔다. Apple M5는 출시 초기 단계의 최신 칩으로, 공개된 LLM 추론 성능 데이터가 부족하다. Q-Bench는 이 초기 데이터를 직접 확보하고, 동일 실험을 다른 하드웨어/조건에서 재실행할 수 있는 **재현 가능한 프로파일링 프레임워크**를 제공하는 것을 목표로 한다.

### 실험 플랫폼

* Device: MacBook Pro
* SoC: Apple M5 (10-core, 24 GB Unified Memory)
* Inference Runtime: llama.cpp (Metal backend)
* Base Model: Qwen2.5-7B-Instruct

---

## 2. 문제 정의

LLM quantization은 모델의 저장 공간과 실행 메모리 요구량을 크게 줄일 수 있지만, 낮은 정밀도가 항상 높은 실제 추론 성능으로 이어지는 것은 아니다.

특히 **신규 하드웨어에서의 실측 데이터가 부족**하기 때문에, 실제 배포/사용 환경에서 어떤 양자화 조건이 최적인지에 대한 근거가 명확하지 않다.

따라서 본 프로젝트는 **신규 하드웨어(Apple M5)에서 측정한 신뢰할 수 있는 성능/메모리 기준선**을 확보하고, 이를 기반으로 "품질 손실 대비 최고의 추론 효율을 제공하는 양자화 조건"에 대한 판단 근거를 만드는 것을 목적으로 한다.

---

## 3. 핵심 연구 질문

### Primary Research Question

**Apple M5(Metal)에서 로컬 LLM 추론의 quantization level은 처리량(throughput), 메모리, 레이턴시에 어떤 영향을 미치는가?**

### Secondary Questions

1. Quantization level이 감소하면 모델 파일 크기와 실제 피크 메모리는 얼마나 감소하는가?
2. Quantization level에 따라 prompt processing throughput은 어떻게 변화하는가?
3. Quantization level에 따라 token generation throughput은 어떻게 변화하는가?
4. Quantization level에 따라 TTFT와 전체 latency는 어떻게 변화하는가?
5. **신규 하드웨어 기준선**: 동일한 설정으로 수집한 M5 데이터는 어떤 특징을 가지며, 기존 세대(참고 자료)와 비교했을 때 어떤 차이가 있는가?
6. M5 24GB 환경에서 품질·메모리·속도를 동시에 고려한 권장 quant configuration은 무엇인가?

---

## 4. 연구 가설

### H1 — Memory

Quantization bit-width가 감소할수록 모델 크기와 실행 시 메모리 사용량은 감소할 것이다.

### H2 — Throughput

중간 수준의 quantization은 높은 정밀도의 모델보다 높은 inference throughput을 제공할 가능성이 있다. 단, bit-width와 throughput 사이에는 완전한 선형 관계가 존재하지 않을 것으로 예상한다.

### H3 — New Hardware Baseline

신규 하드웨어(M5)에서 측정된 기준선은 기존 세대 칩(M1–M4 등)에서 보고된 성능 수치 대비 prompt/generation throughput이 유의하게 높은 수준으로 나타날 것이다.

### H4 — Efficiency Frontier

특정 quantization 구간에서 추가적인 메모리 절감 및 throughput 개선보다 품질 손실의 증가폭이 커지는 trade-off point가 존재할 것이다. (품질 지표는 v0.2부터 추가)

---

## 5. 프로젝트 범위

### Q-Bench v0.1 — Hardware Performance Baseline

목표:

**신규 하드웨어(M5) 기준선 확보 + Quantization → Memory / Inference Performance**

하나의 기반 모델을 사용하여, 동일한 실험 조건으로 재현 가능한 기준선을 만든다.

측정 대상:

* Model file size
* Peak memory
* Prompt processing throughput
* Generation throughput
* TTFT
* Total latency

### Q-Bench v0.2

추가:

**Quantization → Model Quality**

* Perplexity
* Knowledge/Reasoning benchmark
* Mathematical reasoning benchmark
* Coding benchmark

### Q-Bench v0.3

추가:

**Quantization × Context Length**

다양한 context length에서 양자화 효과를 측정한다.

### Q-Bench v1.0

최종적으로 다음 관계를 분석한다.

**Model Size × Quantization × Context Length → Quality / Memory / Latency / Throughput**

※ v1.0 이후, 동일 파이프라인을 다른 하드웨어(다른 Apple Silicon 세대, NVIDIA 등)에 적용하여 하드웨어 간 수직 비교를 수행한다.

---

## 6. 실험 환경

### Hardware (대상 장치: Apple M5 신규 하드웨어)

* Device: MacBook Pro
* SoC: Apple M5
* CPU: 10-core (4 performance + 6 efficiency)
* Unified Memory: 24 GB
* Architecture: Apple Silicon

### Software

* OS: macOS
* Inference Runtime: llama.cpp (pin 커밋 기록: `config/llama_cpp_commit.txt`)
* GPU Backend: Metal
* Model Format: GGUF
* Analysis: Python

---

## 7. Quantization Conditions

비교 대상 (동일한 원본 checkpoint로부터 생성):

* F16
* Q8_0
* Q6_K
* Q5_K_M
* Q4_K_M
* Q3_K_M
* Q2_K

---

## 8. 통제 변수

Quantization 이외의 조건은 동일하게 유지한다.

* Base model
* Prompt dataset
* Context length
* Maximum output length
* Sampling parameters
* Random seed
* Inference runtime
* Runtime version (pinned commit)
* Metal configuration
* Hardware
* OS
* Background workload

초기 기본값:

* Context Length: 4096 (tg는 KV depth 4096(`-d 4096`)에서 256 토큰 생성으로 측정)
* Maximum Output: 256 tokens
* Temperature: 0
* Seed: 42

※ 성능 측정(llama-bench) 경로는 합성 토큰으로 처리량만 측정하므로 **temperature, seed, 프롬프트 데이터셋을 사용하지 않는다.** 이 값들은 v0.2 품질 평가(llama-server) 경로에만 적용된다.

---

## 9. 반복 측정 / 통계

각 quantization configuration별:

* Warm-up: 3 runs (측정에서 제외)
* Measurement: 10 runs

통계량:

* Mean / Median / Standard deviation / 95% confidence interval

---

## 10. 주요 Metrics

### Storage

* GGUF File Size
* Compression Ratio

### Memory

* Peak Process Memory
* Unified Memory Usage

### Performance

* Prompt Processing Throughput (tokens/s)
* Generation Throughput (tokens/s)
* Time To First Token (ms)
* Total Generation Latency (s)

### Hardware Metadata (본 프로젝트의 차별점)

모든 결과(raw CSV, processed, summary)에 대상 하드웨어 정보를 포함한다.

* SoC (Apple M5)
* Unified Memory (24 GB)
* Backend (Metal)
* llama.cpp commit
* OS 정보

이를 통해 **같은 실험을 다른 하드웨어에서 실행했을 때 수직 비교 가능**하게 한다.

### Quality (v0.2부터)

* Perplexity
* Task Accuracy
* Quality Retention

---

## 11. 데이터 관리

```text
data/
├── raw/          # raw benchmark 결과 (hardware metadata 포함)
└── processed/    # 통계 요약

results/
├── figures/
├── tables/
└── summary.json  # 실험 config + hardware + 결과 포함
```

Raw data는 분석 과정에서 직접 수정하지 않는다.

---

## 12. 자동화 목표

```bash
./scripts/run_all.sh
```

Pipeline:

```text
Environment Validation → Model Validation → Warm-up
→ Benchmark → Raw Data Collection → Data Processing
→ Statistical Analysis → Visualization → Report
```

---

## 13. 결과 시각화

v0.1에서 생성:

1. Quantization vs Model Size
2. Quantization vs Peak Memory
3. Quantization vs Prompt Throughput
4. Quantization vs Generation Throughput
5. Quantization vs TTFT

v0.2 이후 추가:

6. Quantization vs Model Quality
7. Quality–Performance Pareto Frontier

---

## 14. 성공 기준 — v0.1 (Hardware Performance Baseline)

* 동일한 기반 모델에서 여러 quantization configuration 준비
* 모든 실험 조건 자동 기록 (config/hardware.txt, pinned commit 포함)
* 모든 quantization configuration 자동 실행
* Raw benchmark 결과 자동 저장 (hardware metadata 포함)
* 반복 측정 완료 및 통계 분석 자동화
* 주요 결과 그래프 자동 생성
* **신규 하드웨어(M5) 기준선 데이터 수집 완료**
* 실험을 동일 환경에서 재현 가능

---

## 15. 장기 확장

### Multiple Hardware Platforms (핵심 확장 방향)

* 다른 Apple Silicon 세대 (M1/M2/M3/M4 등)
* NVIDIA GPU 시스템
* CPU-only 시스템

→ 동일 파이프라인으로 **하드웨어 간 성능 비교 레퍼런스** 구축

### Multiple Models

* Small / Mid-size / Large LLM

### Multiple Context Lengths

* 512 / 2048 / 4096 / 8192+

### Multiple Runtime Backends

* llama.cpp / MLX

이를 통해 **신규 하드웨어 등장 시 데이터가 없는 단계에서 최적 양자화 조건을 빠르게 결정할 수 있는 hardware-aware LLM inference benchmark framework**로 확장하는 것을 장기 목표로 한다.