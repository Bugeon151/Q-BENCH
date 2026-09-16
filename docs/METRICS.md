# Metrics

Q-Bench v0.1에서 측정하는 지표 정의서.

## Storage

| Metric | 정의 |
|---|---|
| GGUF File Size | 각 quant level GGUF 파일의 바이트 크기 |
| Compression Ratio | `F16 size / quant size` |

## Memory

| Metric | 정의 |
|---|---|
| Peak Process Memory (RSS) | llama-bench 프로세스의 최대 resident set size (bytes) — `memory_monitor.py`가 50ms 간격 폴링 |

## Performance

| Metric | 정의 |
|---|---|
| Prompt Processing Throughput | prompt 처리 속도 (tokens/s) — llama-bench `pp` |
| Generation Throughput | 토큰 생성 속도 (tokens/s) — llama-bench `tg` |
| TTFT | 첫 토큰까지 시간 (ms) — `pp` 1회 랜더에서의 latency |
| Total Generation Latency | 전체 생성 레이턴시 (s) — llama-bench `pg` |

## Hardware Metadata (본 프로젝트 차별점)

모든 출력에 포함:

| Field | 예시 |
|---|---|
| SoC | Apple M5 |
| Unified Memory | 24 GB |
| Backend | Metal |
| llama.cpp commit | pinned in `config/llama_cpp_commit.txt` |
| OS | macOS |

→ 동일 파이프라인을 다른 하드웨어에서 재실행 시 수직 비교 가능.

## 통계

반복 측정 10회 (warm-up 3회 제외)에 대해:

* Mean
* Median
* Standard deviation
* 95% confidence interval (t-distribution)

## 품질 지표 (v0.2 예정)

* Perplexity
* Task Accuracy (knowledge/reasoning/math/coding)
* Quality Retention vs F16