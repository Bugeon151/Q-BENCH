# RUNBOOK

실행 순서와 명령. 모든 단계는 이어서 실행(resume)됩니다. 이미 기록된 항목은 `SKIP`으로 건너뜁니다.

| 기기 | 역할 | hw_id (자동 감지 예시) |
|---|---|---|
| M5 맥 (Apple M5, 24GB, macOS 26.4, Metal) | 성능 측정: v0.1, v0.3, v1.0 | `apple-m5-24gb` |
| 노트북 (Core Ultra 9, 32GB, RTX 5070 Ti Laptop 12GB) | 품질 평가 v0.2, 하드웨어 비교 | `nvidia-geforce-rtx-5070-ti-laptop-gpu-12gb` 형태 |

hw_id는 `python src/common.py hwid`로 확인합니다. 고정하려면 `config/experiment.yaml`의 `hardware.id`를 설정하세요.

## 0. 공통 준비 (두 기기 모두)

```bash
git clone https://github.com/Bugeon151/q-bench && cd q-bench
python3 -m venv .venv
git clone https://github.com/ggml-org/llama.cpp external/llama.cpp
git -C external/llama.cpp checkout c6824a9e42ceeda5d58089fa274ddd816e59e68e   # config/llama_cpp_commit.txt
```

llama.cpp 빌드:

```bash
# M5 (Metal은 macOS에서 기본 활성화)
cmake -S external/llama.cpp -B external/llama.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build external/llama.cpp/build --config Release -j

# 노트북 (Linux / WSL2, CUDA). RTX 50 시리즈(Blackwell, sm_120)는 CUDA Toolkit 12.8 이상이 필요합니다
# (ggml/src/ggml-cuda/CMakeLists.txt @ c6824a9 주석 참조).
cmake -S external/llama.cpp -B external/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build external/llama.cpp/build --config Release -j
```

`scripts/setup.sh`가 확인하는 바이너리는 llama-quantize, llama-bench, llama-cli, llama-perplexity, llama-server입니다.
setup은 `config/hardware/<hw_id>/profile.txt`(macOS는 Serial/UUID/UDID 제거)와 `requirements-lock.txt`(pip freeze)를 기록합니다.

## 1. M5 맥: 성능 측정

```bash
# v0.1  7B, ctx 4096 (실측 약 1.5시간: 기존 v0.1 실행 기준)
./scripts/run_all.sh --stage perf

# v0.3  7B, ctx 512/2048/4096/8192 (추정 약 4.5시간 추가. ctx 4096은 SKIP됨)
./scripts/run_all.sh --stage ctx

# v1.0  나머지 모델 (모델별 실행 권장). 4개 모델 × 4 ctx 전체는 추정 약 17시간(7B 포함)
./scripts/run_all.sh --stage ctx --model Qwen2.5-1.5B-Instruct
./scripts/run_all.sh --stage ctx --model Qwen2.5-3B-Instruct
./scripts/run_all.sh --stage ctx --model Qwen2.5-14B-Instruct   # F16·Q8_0은 자동 SKIP
```

- 결과: `data/raw/apple-m5-24gb/benchmark_raw.csv` (기존 `data/raw/benchmark_raw.csv`는 수정하지 않음)
- 시간 추정 방법: v0.1 실측 pp/tg 속도로 반복 13회(warm-up 3 + 측정 10) × 3 그룹을 계산하고, 모델 크기에 반비례하게 스케일했습니다. 긴 ctx에서는 attention 비용 때문에 실제보다 짧게 추정될 수 있습니다.
- 측정 중 다른 앱을 종료하세요. swap 증가가 감지되면 로그에 `WARN ... swap grew`가 남습니다.

## 2. 노트북: 품질 평가 (v0.2)와 하드웨어 비교

```bash
# v0.2  7B 7종 quant: perplexity + MMLU/GSM8K/HumanEval (추정 2~4시간)
./scripts/run_all.sh --stage quality

# 하드웨어 비교용 성능 측정 (12GB VRAM 초과 조합은 자동 SKIP: 예) 7B F16)
./scripts/run_all.sh --stage perf
./scripts/run_all.sh --stage ctx        # 선택
```

- 첫 실행 시 `prompts/quality.json`(seed 42 표본 ID와 데이터셋 revision)을 만듭니다. **이 파일을 커밋해서** 다른 기기에서도 같은 표본을 쓰세요. 표본만 만들려면 `.venv/bin/python src/quality_eval.py --build-sample-only`.
- 품질 단계는 `-ngl auto`(llama.cpp `--fit`)를 써서 VRAM이 부족하면 일부 레이어를 CPU에 둡니다. 그래서 7B F16도 품질 평가가 가능합니다(속도는 느림).
- 부분집합을 쓰는 이유: M5에서 GSM8K 전체(1,319문항) × 7종은 실측 생성 속도 기준 약 40시간(추정)이 걸립니다.
- 데이터셋 다운로드에 Hugging Face 접속이 필요합니다(`cais/mmlu`, `openai/gsm8k`, `openai/openai_humaneval`, `ggml-org/ci` wikitext-2).

> **경고 (신뢰할 수 없는 코드 실행)**: HumanEval 채점은 모델이 생성한 Python 코드를 이 기기에서 `subprocess`와 timeout만으로 실행합니다. 샌드박스가 아닙니다. 불안하면 컨테이너/VM에서 실행하거나 `src/quality_eval.py --tasks mmlu gsm8k`로 HumanEval을 제외하세요.

## 3. 결과 합치기

두 기기의 `data/raw/<hw_id>/`, `data/quality/<hw_id>/`, `config/hardware/<hw_id>/`를 커밋한 뒤 한 기기에서:

```bash
.venv/bin/python src/analyze.py
```

출력: `results/tables/{summary,quality,hardware_comparison}.csv`, `results/figures/<hw_id>/<model>/`, `results/figures/quality/`, `results/figures/hardware/`, `results/summary.json`.

## 4. Claude에 결과 공유할 때 (토큰 절약)

- 공유: `results/tables/summary.csv`(필요하면 `quality.csv`)와, 실패 시 **에러 마지막 20줄**만.
  ```bash
  tail -n 20 "$(ls -t logs/*.log | head -n 1)"
  ```
- raw CSV, jsonl, 전체 로그, 그림 파일은 붙이지 마세요.

## 5. 문제 해결

| 증상 | 조치 |
|---|---|
| `FAIL: external/llama.cpp HEAD ... does not match` | `git -C external/llama.cpp checkout <config/llama_cpp_commit.txt>` 후 재빌드 |
| `SKIP ... over memory budget` | 의도된 동작. 규칙은 `config/experiment.yaml`의 `memory_budget` |
| `WARN CV>3%` (analyze) | 결과 신뢰도 낮음. 다른 앱을 끄고 재측정 권장. raw는 수정하지 말고 `hardware.id`를 바꿔 새 디렉터리에 측정 |
| llama-server가 뜨지 않음 | `logs/server_<model>_<quant>.log` 마지막 20줄 확인 |
