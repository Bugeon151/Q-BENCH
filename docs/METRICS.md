# Metrics

Q-Bench 지표 정의서 (v0.1 성능, v0.2 품질, v0.3 context, v1.0 통합).

## Storage

| Metric | 정의 |
|---|---|
| GGUF File Size | 각 quant level GGUF 파일의 바이트 크기 |
| Compression Ratio | `F16 size / quant size` |

## Memory

| Metric | 정의 |
|---|---|
| Peak Process Memory (RSS) | llama-bench 프로세스의 최대 resident set size — `memory_monitor.py`가 50ms 간격 폴링. Metal(통합 메모리)에서도 GPU 버퍼가 RSS에 모두 잡힌다는 보장은 없음 |
| Peak VRAM (NVIDIA) | `nvidia-smi -lms 100` 스트리밍으로 얻은 장치 memory.used의 최댓값 − 실행 직전 값. 장치 단위(프로세스 단위 아님) |
| Swap growth | 측정 실행 전후 `psutil.swap_memory().used` 차이. 0보다 크면 메모리 압박 신호 |

그래프와 표의 GB는 10^9 bytes 입니다.

## Performance

| Metric | 정의 |
|---|---|
| Prompt Processing Throughput | prompt 처리 속도 (tokens/s) — llama-bench `-p ctx -n 0` |
| Generation Throughput | 토큰 생성 속도 (tokens/s) — llama-bench `-d ctx -p 0 -n 256`. **KV depth = ctx(기본 4096)** 상태에서 256 토큰 생성 |
| Partial offload | VRAM 예산을 넘는 CUDA 조합의 실사용 속도. 기본 ctx에서만 llama-bench `-fitt 1024`(1 GiB 여유를 두고 GPU에 들어가는 만큼 올리고 나머지는 CPU)로 측정하고, raw의 `n_gpu_layers`=`fit:1024MiB`, summary의 `partial_offload`=True로 표시. 반복 수가 적어(측정 3회) CI가 넓음 |
| TTFT | 첫 토큰까지 시간 (ms) — `pp` 1회 런에서 ctx 토큰을 처리한 latency |
| Total Generation Latency | 전체 생성 레이턴시 (ms) — llama-bench `-pg ctx,256 -p 0 -n 0` |

llama-bench는 합성 토큰으로 처리량만 측정합니다. **temperature, seed, 프롬프트 데이터셋은 이 경로에서 사용되지 않습니다** (v0.2 품질 평가에만 적용).

v0.1 legacy 데이터(`data/raw/benchmark_raw.csv`)의 pg 그룹에는 llama-bench 기본 테스트(pp512, tg128)가 `pg` 라벨로 섞여 있어(quant당 30행 중 20행), 분석 시 `n_prompt > 0 and n_gen > 0` 행만 사용합니다. 새 측정은 `-p 0 -n 0`으로 이 문제가 없습니다.

## Hardware Metadata (본 프로젝트 차별점)

모든 출력에 포함:

| Field | 예시 |
|---|---|
| hw_id | 런타임 감지 (`src/common.py hwid`), raw 저장 경로 `data/raw/<hw_id>/` |
| OS / CPU / GPU / RAM / VRAM / Backend | 런타임 감지 (macOS: sysctl, Linux: /proc, nvidia-smi) |
| llama-bench 메타데이터 | build_commit, n_threads, n_gpu_layers, flash_attn, n_batch, n_ubatch, type_k, type_v, gpu_info, cpu_info, backends |
| llama.cpp commit | pinned in `config/llama_cpp_commit.txt` |

→ 동일 파이프라인을 다른 하드웨어에서 재실행 시 수직 비교 가능.

## 통계

반복 측정 10회 (warm-up 3회 제외)에 대해:

* Mean
* Median
* Standard deviation
* 95% confidence interval (t-distribution) — 그래프 error bar
* Coefficient of variation (std/mean of tokens/s) — `analysis.cv_warn_threshold`(기본 3%) 초과 시 `summary.csv`의 `cv_warning`에 표시

참고: v0.1 F16은 tg CV 8.0%(다른 quant 0.5–1.1%)이고 피크 RSS(13.09GB)가 파일 크기(15.24GB)보다 작아 메모리 압박이 의심됩니다. 해석에 주의하세요.

## 품질 지표 (v0.2)

| Metric | 정의 |
|---|---|
| Perplexity | llama-perplexity, wikitext-2 raw test (`wiki.test.raw`), ctx 512, `Final estimate: PPL = x +/- err` |
| MMLU accuracy | 과목별 10문항 층화 표본(57과목, 570문항), zero-shot, 응답의 첫 독립 문자 A–D가 정답과 일치 |
| GSM8K accuracy | 250문항 무작위 표본, `The answer is N` 또는 마지막 숫자를 정답(`####` 뒤)과 비교 |
| HumanEval pass@1 | 164문항 전체, greedy 1회 생성, 테스트를 subprocess + timeout(10s)으로 실행 |
| Quality Retention | quant 정확도 / 같은 모델 F16 정확도, 과제 평균. 과제 결과가 없으면 F16 PPL / PPL로 대체(`retention_basis`) |

표본 ID는 seed 42로 고정되어 `prompts/quality.json`에 데이터셋 revision과 함께 저장됩니다. 생성은 temperature 0, seed 42.
`-np`(parallel slot)로 여러 요청을 함께 배치 처리하므로 출력이 실행마다 비트 단위로 같다는 보장은 없습니다(llama.cpp server README: 배치 크기에 따라 logits가 bit-for-bit 동일하지 않을 수 있음). 요청에는 `cache_prompt: false`를 사용합니다.