# Q-Bench report: what quantization does to a local LLM on an 8 GB laptop GPU

Qwen2.5-Instruct 1.5B / 3B / 7B / 14B, seven GGUF levels each (F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M,
Q3_K_M, Q2_K), measured end to end on one consumer laptop: speed, memory, perplexity, task accuracy
on fixed item samples, answer-level changes, serving capacity, and the tooling that turns these
measurements into deployment decisions. Every number below is generated from the result tables
(`docs/RESULTS.md`, written by `src/report.py`); this page interprets them.

## TL;DR

<!-- TLDR -->

## 1. Setup

| | |
|---|---|
| Machine | RTX 5070 Laptop GPU 8 GB, Intel Core Ultra 9 275HX, 31 GiB RAM, Windows 11 (build 26200) |
| Runtime | llama.cpp pinned at `c6824a9` (prebuilt CUDA binaries of release b11007 = `2f3fd02`, parent commit; differs only in CI workflows) |
| Speed | `llama-bench`, ctx 512 / 2048 / 4096 / 8192, warm-up + repeated runs, VRAM/RSS/swap and GPU temperature/clock/power sampled during each run |
| Perplexity | `llama-perplexity`, wikitext-2 raw **test**, ctx 512, all 584 chunks |
| Tasks | `llama-server` (OpenAI API, 4 slots, temperature 0, seed 42), fixed samples with IDs in `prompts/quality.json`: MMLU 570 items (10 per subject), GSM8K 250, HumanEval 164 (all; scored by executing the tests) |
| Statistics | Wilson 95 % CI per accuracy; paired comparisons on the same items: exact McNemar test, paired bootstrap 95 % CI of the difference (10,000 resamples), Bonferroni over the three tasks |

Reference per model is F16. 14B F16 (29.5 GB) does not fit this machine: `llama-bench` decoded at
2.5 tok/s while swap grew by 3.06 GB, so 14B quality is measured against its Q8_0 (on 7B, Q8_0 vs
F16 differs by +0.09 % perplexity and 1.1 % of answers).

Sanity check against the model publisher (Qwen blog, Qwen2.5-LLM post; its MMLU column is
MMLU-Redux, so a few points apart is expected): 7B F16 71.9 / 94.0 / 82.9 vs published 75.4 / 91.6 /
84.8 (MMLU / GSM8K / HumanEval), 14B Q8_0 77.9 / 94.8 / 82.3 vs 80.0 / 94.8 / 83.5, 3B F16 64.4 /
89.6 / 73.2 vs 64.4 / 86.7 / 74.4, 1.5B F16 54.2 / 70.4 / 53.0 vs 50.7 / 73.2 / 61.6.

## 2. Findings

### 2.1 The same quantization level does very different damage to different models

| HumanEval pass@1 (greedy) | F16 / ref | Q4_K_M | Q3_K_M | Q2_K |
|---|---|---|---|---|
| 1.5B | 53.0 | 48.8 | 39.0 | 1.2 |
| 3B | 73.2 | 68.9 | **6.1** | **0.0** |
| 7B | 82.9 | 79.9 | 81.1 | 59.8 |
| 14B (ref Q8_0) | 82.3 | 82.9 | 80.5 | 59.1 |

Q3_K_M costs 7B and 14B under 2 points of coding accuracy but destroys 3B (73.2 → 6.1); 3B Q2_K
answers nothing correctly (perplexity 22,582 vs 8.99 for F16; it emits digit noise). A rule like
"Q3 is fine" is model-specific; it has to be checked per model, which is what the gate (§2.3) is
for. 3B Q3_K_M also made `llama-server` return HTTP 500 on 6 of 984 requests, because the reply
contained broken UTF-8 byte sequences its chat-output parser rejects: in production those users get
an error, not a wrong answer. These items are scored wrong and counted separately.

<!-- IMATRIX -->

### 2.2 Accuracy hides how many answers change

Dutta et al. (2024) showed that compressed models can keep accuracy while many individual answers
"flip" between correct and wrong. Same here, on every model:

| answers flipped vs reference (all 984 items) | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_M | Q2_K |
|---|---|---|---|---|---|---|
| 1.5B | 5.5 % | 8.7 % | 11.9 % | 16.2 % | 29.0 % | 52.6 % |
| 3B | 3.3 % | 5.2 % | 6.9 % | 11.9 % | 35.6 % | 72.3 % |
| 7B | 1.1 % | 1.6 % | 3.3 % | 4.4 % | 7.7 % | 20.6 % |
| 14B (vs Q8_0) | — | 1.9 % | 3.3 % | 3.8 % | 6.5 % | 17.0 % |

7B Q4_K_M changes mean accuracy by −0.5 points but flips 4.4 % of all answers and 9.1 % of
HumanEval problems. The effect is not run-to-run noise:

<!-- NOISE -->

Smaller models are more fragile at the same bit width: at Q8_0, 5.5 % of 1.5B answers flip vs
1.1 % of 7B answers.

### 2.3 A regression gate, and what it costs to certify "no regression"

`src/gate.py` decides per task whether a candidate may replace the reference: **FAIL** if it is
significantly worse (McNemar, Bonferroni), **PASS** if the lower bound of the paired 95 % CI of the
difference is above −3 points (non-inferiority), otherwise **INCONCLUSIVE**. Exit codes make it a CI
check.

* FAIL: 1.5B Q3_K_M and Q2_K, 3B Q3_K_M and Q2_K, 7B Q2_K, 14B Q2_K.
* PASS on all three tasks: 14B Q6_K and Q5_K_M (vs Q8_0). Everything else is INCONCLUSIVE,
  mostly because of HumanEval.

Why HumanEval cannot certify: the variance of a paired difference is set by the share of
**discordant** items (the flips of §2.2), not by accuracy. The gate reports the items needed for an
80 % chance of PASS if the true difference is 0 (Wald approximation for the paired design, Connor
1987; checked by simulation in `tests/test_gate.py`):

| | discordant | items needed | HumanEval has |
|---|---|---|---|
| 7B Q8_0, HumanEval | 1.8 % | 160 | 164 |
| 7B Q4_K_M, HumanEval | 9.1 % | 798 | 164 |
| 14B Q4_K_M, HumanEval | 10.4 % | 905 | 164 |
| 1.5B Q8_0, GSM8K | 11.6 % | 1,012 | 1,319 (full test set) |

So flips are not only a user-facing problem; they are the statistical cost of proving a model swap
safe. A public 164-problem benchmark can certify a near-lossless Q8_0 swap, not a Q4 one.

### 2.4 Same memory: a bigger model at fewer bits wins down to about 4 bits

Dettmers & Zettlemoyer (2023) found 4-bit almost universally optimal for zero-shot accuracy per
total model bits. On downstream tasks, comparing each bigger-model configuration with the
closest-size configuration of a smaller model on the same 984 items:

| bigger | GiB | smaller | GiB | bigger − smaller (95 % CI) |
|---|---|---|---|---|
| 14B Q8_0 | 14.6 | 7B F16 | 14.2 | +3.6 (+1.2 to +6.0) |
| 14B Q4_K_M | 8.4 | 7B Q8_0 | 7.5 | +4.8 (+2.4 to +7.1) |
| 7B Q5_K_M | 5.1 | 3B F16 | 5.8 | +6.7 (+4.1 to +9.3) |
| 3B Q4_K_M | 1.8 | 1.5B Q8_0 | 1.5 | +13.0 (+9.9 to +16.2) |
| 14B Q3_K_M | 6.8 | 7B Q8_0 | 7.5 | +2.4 (0.0 to +4.9), p = 0.06 |
| 7B Q3_K_M | 3.5 | 3B Q8_0 | 3.1 | +5.9 (+3.2 to +8.5) |
| 3B Q3_K_M | 1.5 | 1.5B Q8_0 | 1.5 | −11.3 (−14.8 to −7.7) |
| 7B Q2_K | 2.8 | 3B Q8_0 | 3.1 | −2.7 (−5.7 to +0.1), p = 0.08 |
| 14B Q2_K | 5.4 | 7B Q5_K_M | 5.1 | −5.9 (−8.6 to −3.3) |

With ≥ 4.8 bits per weight the bigger model wins every time. At ~4 bits (Q3_K_M) it still wins
except where the model collapses (3B). At ~3.2 bits (Q2_K) it loses or ties against the next
smaller model at 5–8 bits (−5.9, −2.7, −58.7 points) and only beats a model 4–5× smaller (7B Q2_K vs
1.5B F16: +11.2). No F16 or Q8_0 configuration is on the accuracy-per-GiB Pareto frontier: above
1.2 GiB every frontier point is a Q3_K_M–Q6_K build of a 3B–14B model
(`results/figures/frontier.png`). Note that the gate (vs own F16) and the frontier
answer different questions: 7B Q2_K FAILs against 7B F16, yet beats 1.5B F16 of the same size by
11.2 points.

### 2.5 A one-minute perplexity run predicts answer flips

Across 23 quantized configurations, ln(PPL / PPL_ref) ranks the share of flipped answers with
Spearman ρ = 0.86 (p = 1.6e-7) and the drop in mean accuracy with ρ = 0.79. Perplexity on
wikitext-2 took 0.7–11 minutes per configuration here, the task sample 5–150 minutes. It flags
collapses immediately (3B Q2_K: 22,582) and orders the rest, so it works as a screen before the
full gate, not as a replacement: it does not certify anything.

### 2.6 What an 8 GB laptop GPU actually runs

Decode speed at ctx 4096 (tok/s): 7B Q6_K 49.2, Q4_K_M 69.3 (fits in VRAM); 7B Q8_0 20.1 and F16
7.2 (part of the layers on the CPU); 14B Q4_K_M 10.8, 14B Q2_K 42.5 (fits); 14B F16 2.5 with swap.
Crossing the VRAM boundary costs 2.4–3× (7B Q6_K → Q8_0, 14B Q2_K → Q3_K_M). The same 7B Q4_K_M decodes at 22.4 tok/s on an Apple M5
(v0.1 legacy run), 3.1× slower than this laptop. Over the run the GPU reached 88 °C.

<!-- CAPACITY -->

### 2.7 Predicting speed instead of measuring it

Decode is memory-bandwidth bound: tokens/s ≈ effective bandwidth / bytes read per token (weights +
KV cache), plus a fixed per-token overhead. Fitted per device, this explains 79 laptop
configurations with 7.0 % mean error (M5: 3.4 %). Transferred across devices with published
bandwidth only, the error is 12 % if this laptop's memory runs at the 384 GB/s figure and 32 % at
the 512 GB/s figure that also circulates; laptop GPU specs are the weakest input. Two `llama-bench`
runs of the smallest model on the target machine (≈1 min) predict the other 51 configurations with
10.7 % mean error and no spec at all: 2.7–8 % for F16 to Q4_K_M, but 17–21 % over-prediction for
Q3_K_M and Q2_K (max 29.5 %, 3B Q2_K), where dequantization compute starts to matter.

## 3. Tools built on these measurements

| tool | what it does |
|---|---|
| `src/gate.py` | regression gate with items-needed planning; also compares across models (`--candidate-model`) for upgrade decisions |
| `src/recommend.py` | picks (model, quant) for a VRAM/RAM budget and speed floor; shows accuracy, gate, flips, measured or predicted speed, and SLA capacity |
| `src/deploy.py` | offline deployment: detect → select → SHA256 of model and server binary → serve → acceptance test on this machine vs recorded per-item results (FAIL blocks) → capacity → JSON audit manifest |
| `src/loadtest.py` | closed-loop concurrency sweep on `llama-server`, streamed TTFT/TPOT percentiles, capacity under MLPerf-derived SLAs |
| `src/consistency.py`, `src/frontier.py` | flips with re-run noise floor; Pareto frontier, matched-memory duels, perplexity screen |
| `src/imatrix_fix.py` | diagnoses a collapsed quant (CPU-only perplexity prefix rules out a GPU kernel fault) and re-quantizes it with an importance matrix calibrated on the wikitext-2 train split |
| `src/perfmodel.py` | decode-speed model, cross-device validation, on-site calibration |

<!-- DEPLOY -->

## 4. Limitations

* One machine for v0.2–v1.0; the M5 contributes only the v0.1 speed baseline (7B, ctx 4096).
* Task samples are subsets (MMLU 570 of 14,042; GSM8K 250 of 1,319) with greedy decoding and our
  own prompts; absolute scores differ from the publisher's protocol by a few points (§1).
* 14B is compared with Q8_0, not F16.
* llama-server runs 4 parallel slots; batching can change greedy outputs (measured in §2.2).
* Speed configurations that spill to the CPU depend on RAM speed and the `--fit` split; they are
  reported as measured, not modeled.
* Results are for llama.cpp K-quants without an importance matrix unless marked `-imat`.

## 5. Reproduce

`docs/RUNBOOK.md` (Windows runner: `run_qbench.sh`; every step resumes). Analysis:
`analyze.py → perfmodel.py → consistency.py → frontier.py → loadtest.py --summarize-only →
report.py`.

## References

* Dutta, Krishnan, Kwatra, Ramjee. *Accuracy is Not All You Need.* NeurIPS 2024. arXiv:2407.09141.
* Dettmers, Zettlemoyer. *The case for 4-bit precision: k-bit Inference Scaling Laws.* ICML 2023, PMLR 202:7750–7774.
* Connor. *Sample size for testing differences in proportions for the paired-sample design.* Biometrics 43(1), 1987.
* MLCommons. *MLPerf Inference v5.0 Advances Language Model Capabilities for GenAI* (2025-04-02): Llama-2-70B server TTFT 2 s / TPOT 200 ms; interactive p99 TTFT 450 ms / TPOT 40 ms.
* Qwen team. Qwen2.5-LLM blog post, instruction-tuned model performance tables (https://qwenlm.github.io/blog/qwen2.5-llm/).
