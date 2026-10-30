# Q-Bench report: what quantization does to a local LLM on an 8 GB laptop GPU

Qwen2.5-Instruct 1.5B / 3B / 7B / 14B, seven GGUF levels each (F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M,
Q3_K_M, Q2_K) plus importance-matrix re-quantizations of the low-bit 3B and 7B files, measured end
to end on one consumer laptop: speed, memory, perplexity, task accuracy on fixed item samples,
answer-level changes and their run-to-run noise, serving capacity under latency SLAs, and the
tooling that turns these measurements into deployment decisions. Every number below is generated
from the result tables (`docs/RESULTS.md`, written by `src/report.py`, per-item data in `data/`);
this page interprets them.

## TL;DR

* **"Q3 is fine" depends on the model and on how you round.** Q3_K_M costs 7B and 14B under 2
  points of HumanEval but takes 3B from 73.2 % to 6.1 %; 3B Q2_K answers nothing (perplexity
  22,582). A CPU-only re-check shows the weights, not a GPU kernel, are broken. Re-quantizing the
  same file size with an importance matrix brings 3B Q2_K back to perplexity 11.9 and 50 / 72 / 46 %.
* **Accuracy hides answer churn; concurrent batching hides irreproducibility.** 7B Q4_K_M moves
  mean accuracy by −0.5 points but flips 4.4 % of answers between right and wrong. Separately,
  re-running an identical quantized configuration under concurrent batching changes 24–31 % of
  response texts (F16: 1.6 %); with one server slot, two runs are byte-identical <!-- TLDR_NP1 -->.
  The deterministic run shows the flips are the quantization's (7B: 4.6 %); subtracting a re-run
  noise floor, the usual correction, would have understated them by more than half.
* **A 164-problem benchmark can certify Q8_0 but not Q4.** The share of flipped items sets the
  sample size: 160 HumanEval problems suffice for 7B Q8_0, about 800 for Q4_K_M.
* **At equal memory, a bigger model at fewer bits wins down to ~4 bits, and to ~3 with an
  importance matrix.** A 2.8 GiB 7B Q2_K loses to a 3.1 GiB 3B Q8_0 by 2.7 points with plain
  K-quants and beats it by 4.1 (p = 0.006) with an imatrix. No F16 or Q8_0 file is on the
  accuracy-per-GiB Pareto frontier.
* **A one-minute perplexity run ranks answer flips** across 27 configurations (Spearman ρ = 0.88):
  a screen before the full gate.
* **Accuracy and serving capacity rank configurations differently.** 7B Q6_K and Q4_K_M are
  statistically indistinguishable on 984 paired items (−0.3 points, 95 % CI −1.6 to +1.0), but
  under MLPerf's interactive latency SLA Q4_K_M serves 4 concurrent users and Q6_K serves 1.
* **A deployment is a manifest.** `deploy.py` selects for the hardware and user count, hashes model
  and server binary, re-scores recorded items on the target machine, measures capacity, and writes
  an audit record; two runs produced the same selection, hashes and verdicts.

## 1. Setup

| | |
|---|---|
| Machine | RTX 5070 Laptop GPU 8 GB, Intel Core Ultra 9 275HX, 31 GiB RAM, Windows 11 (build 26200), driver 591.91 |
| Runtime | llama.cpp pinned at `c6824a9` (prebuilt CUDA 13.4 binaries of release b11007 = `2f3fd02`, parent commit; differs only in CI workflows; archive SHA-256s in the deploy manifests) |
| Speed | `llama-bench`, ctx 512 / 2048 / 4096 / 8192, warm-up + repeated runs, VRAM/RSS/swap and GPU temperature/clock/power sampled during each run |
| Perplexity | `llama-perplexity`, wikitext-2 raw **test**, ctx 512, all 584 chunks |
| Tasks | `llama-server` (OpenAI API, 4 slots, temperature 0, seed 42), fixed samples with IDs in `prompts/quality.json`: MMLU 570 items (10 per subject), GSM8K 250, HumanEval 164 (all; scored by executing the tests) |
| Statistics | Wilson 95 % CI per accuracy; paired comparisons on the same items: exact McNemar test, paired bootstrap 95 % CI of the difference (10,000 resamples), Bonferroni over the three tasks |
| Noise | identical re-runs of four configurations; the same configurations with one server slot (no concurrent batching), run twice |
| Load | `llama-server` 16 slots × 1,024 tokens, 1–16 closed-loop users, GSM8K prompts, ≤ 256 output tokens, streamed TTFT/TPOT |
| imatrix | `llama-imatrix` on wikitext-2 **train** (128 chunks) from the F16 model, then `llama-quantize --imatrix` |

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

**Is it the GPU or the weights?** `src/imatrix_fix.py` recomputes the first 20 perplexity chunks
with no CUDA device visible (`CUDA_VISIBLE_DEVICES=-1`, `-ngl 0 --no-op-offload`; without the
last flag llama.cpp still sends large matrix products to the GPU) and compares them with the GPU
run after the same chunks:

| first 20 chunks | GPU | CPU only | CPU / GPU |
|---|---|---|---|
| 7B Q3_K_M | 8.118 | 8.123 | 1.0005 |
| 7B Q2_K | 11.348 | 11.340 | 0.9993 |
| 3B Q3_K_M | 18.01 | 15.23 | 0.85 |
| 3B Q2_K | 22,506 | 25,304 | 1.12 |

On healthy files the CUDA kernels reproduce the CPU to 0.1 %, and the 3B files are broken on the
CPU too: the damage is in the quantized weights, not a GPU kernel bug. That the two backends then
disagree by 12–15 % is itself a symptom: in a model this close to collapse, differences in
summation order that are invisible elsewhere get amplified.

**Same file size, different rounding.** Without an importance matrix, `llama-quantize` picks each
block's scales by minimizing a rounding error weighted only by the weights themselves. With one
(llama.cpp PR #4861), the error is also weighted by how strongly each input column is activated on
calibration text, so the columns the model actually relies on are rounded more carefully. Re-quantizing with an imatrix from the F16 model on wikitext-2 *train* (the test
split stays unseen) gives files with the same tensor types and size (+352 bytes of metadata):

| | PPL | MMLU | GSM8K | HumanEval | flips vs F16 | gate |
|---|---|---|---|---|---|---|
| 3B F16 | 8.99 | 64.4 | 89.6 | 73.2 | — | ref |
| 3B Q3_K_M → imatrix | 15.84 → 14.25 | 51.2 → 56.7 | 60.4 → 74.0 | **6.1 → 58.5** | 35.6 → 21.8 % | FAIL → FAIL |
| 3B Q2_K → imatrix | **22,582 → 11.88** | 0.0 → 50.2 | 0.0 → 72.0 | 0.0 → 46.3 | 72.3 → 28.4 % | FAIL → FAIL |
| 7B F16 | 7.89 | 71.9 | 94.0 | 82.9 | — | ref |
| 7B Q3_K_M → imatrix | 8.54 → 8.24 | 70.9 → 71.2 | 92.0 → 93.6 | 81.1 → 80.5 | 7.7 → 7.4 % | INCONCL. → INCONCL. |
| 7B Q2_K → imatrix | 11.26 → 9.24 | 66.7 → 68.6 | 81.6 → 90.8 | 59.8 → 79.9 | 20.6 → 11.6 % | **FAIL → INCONCL.** |

The imatrix matters in proportion to the damage: nothing measurable at 7B Q3_K_M, half the flips
and a lifted FAIL at 7B Q2_K, a dead model brought back at 3B Q2_K. It does not make 3B safe at
3–4 bits (still FAIL on all three tasks), and the 6 HTTP 500 errors of 3B Q3_K_M disappear (0 of
984). The practical rule: never ship a K-quant below 4 bits without an imatrix, and gate the
result anyway. §2.4 shows what this does to the size-versus-bits trade-off.

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
HumanEval problems. Smaller models are more fragile at the same bit width: at Q8_0, 5.5 % of 1.5B
answers flip vs 1.1 % of 7B answers.

**Is it the quantization or the server?** Re-running the identical configuration (same file,
binary, settings, seed, temperature 0) does not reproduce a quantized model's answers:

| identical re-run, 4 server slots | answers flipped | response text changed |
|---|---|---|
| 1.5B F16 | 0.2 % | 1.6 % |
| 1.5B Q4_K_M | 4.1 % | 30.8 % |
| 3B Q4_K_M | 3.8 % | 26.1 % |
| 7B Q4_K_M | 2.5 % | 24.4 % |

The cause is concurrent batching. `llama-server` decodes its 4 slots together, so the number of
sequences in a forward pass changes from step to step with request timing, and GPU kernels are
not batch-invariant: the same row is summed in a different order depending on what else is in the
batch (Thinking Machines Lab, 2025). The same configurations with one slot
(`quality_eval.py --parallel 1`), run twice:

| Q4_K_M | re-run, 4 slots: flipped / text changed | re-run, 1 slot: flipped / text changed | flips vs reference, 4-slot run | flips vs reference, 1-slot run |
|---|---|---|---|---|
| 1.5B | 4.1 % / 30.8 % | **0 / 0** (984 of 984 records identical) | 16.2 % | 14.8 % |
| 7B | 2.5 % / 24.4 % | <!-- 7B_NP1 --> | 4.4 % | 4.6 % |

With a fixed batch the server is deterministic on this machine and binary. And the deterministic
run settles the question above: the flip rate against the reference is the quantization's, not
the server's (1.5B 14.8 % vs 16.2 %, 7B 4.6 % vs 4.4 %). Batching noise changes *which* items flip
more than *how many*: of the 45 items 7B Q4_K_M flips in the one-slot run, 34 also flip in the
4-slot run, and the rest are swapped for others. That is what noise acting on items near a decision
boundary looks like: it moves them in both directions relative to the reference, so it largely
cancels in the rate. So the textbook correction,
subtracting a re-run noise floor from the flip rate (the `attributable` column in
`docs/RESULTS.md`, 1.9 points for 7B Q4_K_M), understates the quantization effect by more than
half; the reliable measurement is a deterministic candidate run.

Why F16 barely moves under batching while Q4_K_M does is not isolated here. The CUDA path at the
pinned commit suggests an amplifier: for quantized weights the activations are first rounded to
8-bit blocks (`q8_1`) and multiplied by a kernel whose warp layout, and with it the order of the
partial sums, is chosen per number of batch columns (`ggml-cuda/mmvq.cu`,
`calc_nwarps(type, ncols_dst, …)`, used up to 8 columns). A last-bit difference before that
rounding can move an activation to the next 8-bit level, a far larger step than the
floating-point difference itself; F16 weights skip that rounding.

What this means in practice:

* **A quantized model behind a batching server is not a fixed function of its input.** Audit
  trails, regression tests and response caches that assume it is will see a quarter to a third of
  responses change on a re-run. One slot restores reproducibility at a throughput cost: 7B Q4_K_M
  delivers 57 tok/s with one request at a time vs 238 tok/s with 16 concurrent users (§2.6).
* **Measure flips with a deterministic candidate run**, not by subtracting a noise floor.
* **Accuracy and flip rates are robust to the batching noise; individual answers are not.** The
  accuracy difference vs the reference barely moves with a deterministic run (7B −0.5 → −0.3
  points, 1.5B −3.4 → −3.5), so the gate (§2.3) is not distorted; a test that asserts an exact
  response is.

### 2.3 A regression gate, and what it costs to certify "no regression"

`src/gate.py` decides per task whether a candidate may replace the reference: **FAIL** if it is
significantly worse (McNemar, Bonferroni), **PASS** if the lower bound of the paired 95 % CI of the
difference is above −3 points (non-inferiority), otherwise **INCONCLUSIVE**. Exit codes make it a CI
check.

* FAIL: 1.5B Q3_K_M and Q2_K, 3B Q3_K_M and Q2_K (with and without imatrix), 7B Q2_K, 14B Q2_K.
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
safe. A public 164-problem benchmark can certify a near-lossless Q8_0 swap, not a Q4 one, and
running the candidate deterministically does not lower the cost: the discordance is the
quantization's (§2.2).

### 2.4 Same memory: a bigger model at fewer bits wins down to about 4 bits, or 3 with an imatrix

Dettmers & Zettlemoyer (2023) found 4-bit almost universally optimal for zero-shot accuracy per
total model bits. On downstream tasks, comparing each bigger-model configuration with the
closest-size configuration of a smaller model on the same 984 items:

| bigger | GiB | smaller | GiB | bigger − smaller, points (95 % CI) |
|---|---|---|---|---|
| 14B Q8_0 | 14.6 | 7B F16 | 14.2 | +3.6 (+1.2 to +6.0) |
| 14B Q4_K_M | 8.4 | 7B Q8_0 | 7.5 | +4.8 (+2.4 to +7.1) |
| 7B Q5_K_M | 5.1 | 3B F16 | 5.8 | +6.7 (+4.1 to +9.3) |
| 3B Q4_K_M | 1.8 | 1.5B Q8_0 | 1.5 | +13.0 (+9.9 to +16.2) |
| 14B Q3_K_M | 6.8 | 7B Q8_0 | 7.5 | +2.4 (0.0 to +4.9), p = 0.06 |
| 7B Q3_K_M | 3.5 | 3B Q8_0 | 3.1 | +5.9 (+3.2 to +8.5) |
| 3B Q3_K_M | 1.5 | 1.5B Q8_0 | 1.5 | −11.3 (−14.8 to −7.7) |
| 3B Q3_K_M **imatrix** | 1.5 | 1.5B Q8_0 | 1.5 | +4.1 (+0.8 to +7.3), p = 0.016 |
| 7B Q2_K | 2.8 | 3B Q8_0 | 3.1 | −2.7 (−5.7 to +0.1), p = 0.08 |
| 7B Q2_K **imatrix** | 2.8 | 3B Q8_0 | 3.1 | **+4.1 (+1.2 to +6.8), p = 0.006** |
| 14B Q2_K | 5.4 | 7B Q5_K_M | 5.1 | −5.9 (−8.6 to −3.3) |

With ≥ 4.8 bits per weight the bigger model wins every time. At ~4 bits (Q3_K_M) it still wins
except where the plain quantization collapses (3B). At ~3.2 bits (Q2_K) plain K-quants lose or tie
against the next smaller model at 5–8 bits (−5.9, −2.7, −58.7 points) and only beat a model 4–5×
smaller (7B Q2_K vs 1.5B F16: +11.2). With an importance matrix the same 3.2-bit file of 7B beats
3B at 8.5 bits while being 8 % smaller: where the "bigger model at fewer bits" rule stops depends on
the rounding method, not only on the bit count. On the accuracy-per-GiB Pareto frontier
(`results/figures/frontier.png`) there is no F16 or Q8_0 configuration; between 1.4 and 6 GiB the
frontier is 3B Q3_K_M-imatrix → 3B Q4–Q6 → 7B Q2_K-imatrix → 7B Q3_K_M-imatrix → 7B Q5_K_M/Q6_K.
The gate and the frontier answer different questions: 7B Q2_K FAILs against 7B F16, yet beats
1.5B F16 of the same size by 11.2 points.

### 2.5 A one-minute perplexity run predicts answer flips

Across 27 quantized configurations (including the imatrix files), ln(PPL / PPL_ref) ranks the
share of flipped answers with Spearman ρ = 0.88 (p = 1.1e-9) and the drop in mean accuracy with
ρ = 0.86. Perplexity on wikitext-2 took 0.7–11 minutes per configuration here, the task sample
5–150 minutes. It flags collapses immediately (3B Q2_K: 22,582) and orders the rest, so it works
as a screen before the full gate, not as a replacement: it does not certify anything.

### 2.6 What an 8 GB laptop GPU actually runs

Decode speed at ctx 4096 (tok/s): 7B Q6_K 49.2, Q4_K_M 69.3 (fits in VRAM); 7B Q8_0 20.1 and F16
7.2 (part of the layers on the CPU); 14B Q4_K_M 10.8, 14B Q2_K 42.5 (fits); 14B F16 2.5 with swap.
Crossing the VRAM boundary costs 2.4–3× (7B Q6_K → Q8_0, 14B Q2_K → Q3_K_M). The same 7B Q4_K_M
decodes at 22.4 tok/s on an Apple M5 (v0.1 legacy run), 3.1× slower than this laptop. Over the run
the GPU reached 88 °C.

Single-stream speed is not serving capacity. `src/loadtest.py` drives `llama-server` (16 slots of
1,024 tokens) with 1, 2, 4, 8 and 16 closed-loop users sending GSM8K prompts (≤ 256 output tokens)
and reads time-to-first-token (TTFT) and time-per-output-token (TPOT) from the stream. A level
meets an SLA if its p95 is within the MLPerf Inference v5.0 limits for Llama-2-70B: **server**
TTFT 2 s / TPOT 200 ms, **interactive** TTFT 450 ms / TPOT 40 ms (MLPerf uses p99; with 16–64
requests per level we use p95).

| config | users, server SLA | users, interactive SLA | 1 user, tok/s | total tok/s at the largest passing level |
|---|---|---|---|---|
| 1.5B Q4_K_M | ≥ 16 | ≥ 16 | 240 | 759 |
| 3B Q4_K_M | ≥ 16 | 8 | 142 | 590 |
| 7B Q4_K_M | ≥ 16 | 4 | 58 | 238 |
| 7B Q6_K | 8 | 1 | 27 | 96 |
| 14B Q4_K_M | 2 | 0 | 12 | 20 |

* The interactive TPOT limit (40 ms, i.e. 25 tok/s per user) binds first: 7B Q4_K_M has p95 TPOT
  26 ms at 4 users and 41 ms at 8.
* **7B Q6_K and Q4_K_M are a tie on accuracy and not on serving.** Paired on all 984 items, Q4_K_M
  is −0.3 points vs Q6_K (95 % CI −1.6 to +1.0, McNemar p = 0.77), yet it carries 2× the users
  within the server SLA and 4× within the interactive one. A quality-first recommendation picks
  Q6_K for this laptop (§3); `recommend.py --users 4 --sla interactive` picks Q4_K_M.
* 7B Q6_K decodes 27 tok/s for one user in the server vs 49 tok/s in `llama-bench` (Q4_K_M: 58 vs
  69). The server holds a 16,384-token KV cache (0.875 GiB in f16) and we reserve 2 GiB of VRAM
  (`--fit-target 2048`, added after 14B Q6_K crashed with CUDA out-of-memory); 8 GiB minus 2 GiB
  is less than 5.8 GiB of weights plus that cache, so part of Q6_K probably runs from system RAM. The server log does
  not report the split at default verbosity; this is inferred from the speed.
* 14B Q4_K_M (8.4 GiB) does not fit in 8 GB of VRAM: the most accurate model here is a
  single-user model on this machine.

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
| `src/recommend.py` | picks (model, quant) for a VRAM/RAM budget and speed floor; with `--users N --sla server\|interactive` only configurations measured to serve N users; shows accuracy, gate, flips, measured or predicted speed, and capacity |
| `src/deploy.py` | offline deployment: detect → select → SHA-256 of model and server binary → serve → acceptance test on this machine vs recorded per-item results (FAIL blocks) → capacity → JSON audit manifest |
| `src/loadtest.py` | closed-loop concurrency sweep on `llama-server`, streamed TTFT/TPOT percentiles, capacity under MLPerf-derived SLAs |
| `src/consistency.py`, `src/frontier.py` | flips with re-run and one-slot noise floors; Pareto frontier, matched-memory duels, perplexity screen |
| `src/imatrix_fix.py` | diagnoses a collapsed quant (CPU-only perplexity prefix rules out a GPU kernel fault) and re-quantizes it with an importance matrix calibrated on the wikitext-2 train split |
| `src/perfmodel.py` | decode-speed model, cross-device validation, on-site calibration |

**A deployment is a manifest, not a model name.** `python src/deploy.py --items 20 --capacity` on
the laptop (it needs no network once the GGUF files and the dataset cache are on the machine):

1. detected the RTX 5070 Laptop GPU (budget 7.56 GiB = 95 % of VRAM) and selected 7B Q6_K from the
   committed tables (best mean accuracy that fits, gate not FAIL);
2. hashed the 6.25 GB GGUF (`sha256 9a817640…`) and the `llama-server` binary (`e209b447…`) and
   recorded the SHA-256 of the prebuilt release archives;
3. re-ran the first 20 items of each task and compared them item by item with the results recorded
   for the same file: 60 of 60 outcomes identical (MMLU 0.60, GSM8K 1.00, HumanEval 1.00), PASS;
4. smoke request answered in 0.25 s after a 4.6 s load; capacity 8 users (server SLA), 1
   (interactive);
5. wrote `deploy/20261030T014520Z_Qwen2.5-7B-Instruct_Q6_K.json`, status VERIFIED: hardware, driver,
   selection with the runners-up, hashes, dataset revisions and sample seed, acceptance statistics,
   capacity per load level.

A second run 36 minutes later (`deploy/20261030T022150Z_…`) produced the same selection, hashes and
acceptance outcome; only timestamps and the smoke latency differ (capacity was reused, not
re-measured). The acceptance step is a smoke-level regression check: it catches a broken build such
as 3B Q3_K_M (HumanEval 6.1 %) in minutes, but 20 items cannot certify equivalence (§2.3), and it
compares like with like only because the per-item results were recorded with the same file and
server settings. The capacity numbers above argue for running it with `--users N --sla …` when
the box serves a team.

## 4. Limitations

* One machine for v0.2–v1.0; the M5 contributes only the v0.1 speed baseline (7B, ctx 4096).
* Task samples are subsets (MMLU 570 of 14,042; GSM8K 250 of 1,319) with greedy decoding and our
  own prompts; absolute scores differ from the publisher's protocol by a few points (§1).
* 14B is compared with Q8_0, not F16.
* Noise floors are measured for four configurations (one-slot runs for two); other configurations
  use a proxy from the same model's quantized re-runs.
* The importance matrix is calibrated on wikitext-2 train and perplexity is measured on wikitext-2
  test, the same distribution, so the perplexity gains may flatter it; the task gains (MMLU, GSM8K,
  HumanEval) do not share that bias. Only 3B and 7B Q2_K/Q3_K_M were re-quantized.
* Capacity is measured for five configurations, with p95 instead of MLPerf's p99 (16–64 requests
  per level), on GSM8K prompts. 7B Q6_K under 16 slots is probably partly on the CPU (§2.6).
* Speed configurations that spill to the CPU depend on RAM speed and llama.cpp's `--fit` split;
  they are reported as measured, not modeled.

## 5. Reproduce

`docs/RUNBOOK.md` (Windows runner: `run_qbench.sh`; every step resumes). Analysis:
`analyze.py → perfmodel.py → consistency.py → frontier.py → loadtest.py --summarize-only →
report.py`. Per-item outputs (prediction, response text, correctness) of every run are in
`data/quality*/`, load-test requests in `data/capacity/`, deploy manifests in `deploy/`.

## References

* Dutta, Krishnan, Kwatra, Ramjee. *Accuracy is Not All You Need.* NeurIPS 2024. arXiv:2407.09141.
* Dettmers, Zettlemoyer. *The case for 4-bit precision: k-bit Inference Scaling Laws.* ICML 2023, PMLR 202:7750–7774.
* Connor. *Sample size for testing differences in proportions for the paired-sample design.* Biometrics 43(1), 1987.
* MLCommons. *MLPerf Inference v5.0 Advances Language Model Capabilities for GenAI* (2025-04-02): Llama-2-70B server TTFT 2 s / TPOT 200 ms; interactive p99 TTFT 450 ms / TPOT 40 ms.
* Thinking Machines Lab. *Defeating Nondeterminism in LLM Inference* (September 2025), https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/: inference servers are nondeterministic mainly because batch size varies with load and kernels are not batch-invariant.
* llama.cpp. *Importance Matrix calculation*, PR #4861 (https://github.com/ggml-org/llama.cpp/pull/4861); `tools/imatrix/README.md`.
* Qwen team. Qwen2.5-LLM blog post, instruction-tuned model performance tables (https://qwenlm.github.io/blog/qwen2.5-llm/).
