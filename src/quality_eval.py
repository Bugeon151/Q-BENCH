"""Pipeline stage (v0.2): task accuracy via llama-server (OpenAI-compatible API).

Tasks (fixed subsets, IDs in prompts/quality.json, sampled with seed 42):
  mmlu      subject-stratified sample, scored by answer-letter match
  gsm8k     random sample, scored by final-number match
  humaneval all 164 problems, scored by executing the unit tests

WARNING: HumanEval scoring executes model-generated code on this machine
(subprocess + timeout only, no sandbox). Run on a machine you can afford to
expose, or skip the task with --tasks mmlu gsm8k.

Per-item results: data/quality/<hw_id>/<model>/<quant>/<task>.jsonl (resume).
Summary:          data/quality/<hw_id>/quality.csv
--out-root DIR writes the same layout under DIR (identical re-runs for the noise floor).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from common import (ROOT_DIR, check_budget, detect_hw, gguf_path, load_config, make_logger,
                    quant_levels, resolve_bin, select_models)

TASKS = ("mmlu", "gsm8k", "humaneval")
_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# --------------------------------------------------------------------------- prompts / scoring


def mmlu_prompt(row: dict) -> str:
    opts = "\n".join(f"{l}. {c}" for l, c in zip("ABCD", row["choices"]))
    subject = row["subject"].replace("_", " ")
    return (f"The following is a multiple choice question about {subject}. "
            f"Answer with only the letter of the correct option.\n\n{row['question']}\n{opts}\nAnswer:")


def mmlu_extract(text: str) -> str | None:
    m = re.search(r"\b([ABCD])\b", text)
    return m.group(1) if m else None


def gsm8k_prompt(row: dict) -> str:
    return ("Solve the following math problem step by step. At the end, write the final answer "
            f"as 'The answer is <number>'.\n\nQuestion: {row['question']}")


_NUM = r"-?\$?\s*\d[\d,]*(?:\.\d+)?"


def _norm_num(s: str) -> float | None:
    s = s.replace(",", "").replace("$", "").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return None


def gsm8k_gold(answer: str) -> float | None:
    return _norm_num(answer.split("####")[-1].strip())


def gsm8k_extract(text: str) -> float | None:
    hits = re.findall(rf"answer is[:\s]*\**\s*({_NUM})", text, flags=re.I)
    if not hits:
        hits = re.findall(_NUM, text)
    return _norm_num(hits[-1]) if hits else None


def humaneval_prompt(row: dict) -> str:
    return ("Complete the following Python function. Respond with the complete function "
            f"implementation in a single ```python code block.\n\n```python\n{row['prompt']}```")


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    return m.group(1) if m else text


def humaneval_program(row: dict, completion: str) -> str:
    code = extract_code(completion)
    if re.search(rf"^\s*def\s+{re.escape(row['entry_point'])}\s*\(", code, flags=re.M):
        body = row["prompt"] + "\n\n" + code   # full function: redefine after the stub
    else:
        body = row["prompt"] + code            # body only: continue the stub
    return f"{body}\n\n{row['test']}\n\ncheck({row['entry_point']})\n"


def run_program(program: str, timeout: float) -> bool:
    """Executes untrusted code in an isolated interpreter with a timeout."""
    # ignore_cleanup_errors: on Windows a timed-out program's leftover child can keep files locked
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        path = Path(d) / "prog.py"
        path.write_text(program, encoding="utf-8")  # Python reads source as UTF-8 (PEP 3120)
        try:
            r = subprocess.run([sys.executable, "-I", str(path)], cwd=d, stdin=subprocess.DEVNULL,
                               capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
    return r.returncode == 0


def score(task: str, row: dict, text: str, exec_timeout: float) -> tuple[object, bool]:
    if task == "mmlu":
        pred = mmlu_extract(text)
        return pred, pred == "ABCD"[int(row["answer"])]
    if task == "gsm8k":
        pred, gold = gsm8k_extract(text), gsm8k_gold(row["answer"])
        return pred, pred is not None and gold is not None and abs(pred - gold) < 1e-6
    passed = run_program(humaneval_program(row, text), exec_timeout)
    return None, passed


PROMPTS = {"mmlu": mmlu_prompt, "gsm8k": gsm8k_prompt, "humaneval": humaneval_prompt}

# --------------------------------------------------------------------------- datasets / sample


def _dataset_file(repo: str, prefix: str, revision: str | None = None) -> tuple[str, str]:
    from huggingface_hub import HfApi
    api = HfApi()
    sha = revision or api.dataset_info(repo).sha
    files = [f for f in api.list_repo_files(repo, repo_type="dataset", revision=sha)
             if f.startswith(prefix) and f.endswith(".parquet")]
    if len(files) != 1:
        raise SystemExit(f"FAIL: expected one parquet '{prefix}*' in {repo}, found {files}")
    return files[0], sha


def _load_df(repo: str, filename: str, sha: str):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    return pd.read_parquet(hf_hub_download(repo, filename, repo_type="dataset", revision=sha))


def build_sample(cfg: dict) -> dict:
    qcfg = cfg["quality"]
    out = {"seed": qcfg["sample_seed"], "created": datetime.now(timezone.utc).isoformat(), "datasets": {}}
    for task, spec in qcfg["datasets"].items():
        filename, sha = _dataset_file(spec["repo"], spec["prefix"])
        df = _load_df(spec["repo"], filename, sha)
        rng = random.Random(qcfg["sample_seed"])
        if task == "mmlu":
            ids = []
            for _, g in sorted(df.groupby("subject"), key=lambda kv: kv[0]):
                ids += sorted(rng.sample(list(g.index), min(spec["per_subject"], len(g))))
        elif spec.get("n"):
            ids = sorted(rng.sample(range(len(df)), spec["n"]))
        else:
            ids = list(range(len(df)))
        out["datasets"][task] = {"repo": spec["repo"], "revision": sha, "file": filename,
                                 "n": len(ids), "ids": [int(i) for i in ids]}
    return out


def load_sample(cfg: dict, log) -> dict:
    path = ROOT_DIR / cfg["quality"]["sample_file"]
    if path.exists() and path.stat().st_size and "datasets" in json.loads(path.read_text(encoding="utf-8")):
        return json.loads(path.read_text(encoding="utf-8"))
    log(f"building fixed sample (seed {cfg['quality']['sample_seed']}) -> {path}")
    sample = build_sample(cfg)
    path.write_text(json.dumps(sample, indent=1), encoding="utf-8", newline="\n")  # LF on every OS (committed)
    return sample


def load_items(sample: dict, task: str) -> list[dict]:
    d = sample["datasets"][task]
    df = _load_df(d["repo"], d["file"], d["revision"])
    return [{"id": i, **df.loc[i].to_dict()} for i in d["ids"]]

# --------------------------------------------------------------------------- server


def start_server(cfg: dict, model_path: Path, log_path: Path, n_parallel: int | None = None,
                 ctx_per_slot: int | None = None) -> subprocess.Popen:
    q = cfg["quality"]
    n_parallel, ctx_per_slot = n_parallel or q["n_parallel"], ctx_per_slot or q["ctx_per_slot"]
    cmd = [str(resolve_bin(q["llama_server_bin"])), "-m", str(model_path), "--host", "127.0.0.1",
           "--port", str(q["port"]), "-np", str(n_parallel), "-c", str(n_parallel * ctx_per_slot),
           "-ngl", str(q["n_gpu_layers"]), "--seed", str(cfg["inference"]["seed"])]
    if q.get("fit_target_mib"):  # free VRAM llama.cpp's fit keeps per device when -ngl auto offloads
        cmd += ["--fit-target", str(q["fit_target_mib"])]
    proc = subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    deadline = time.time() + 900
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"FAIL: llama-server exited ({proc.returncode}); see {log_path}")
        try:
            with _NO_PROXY.open(f"http://127.0.0.1:{q['port']}/health", timeout=5) as r:
                if r.status == 200:
                    return proc
        except OSError:
            pass
        time.sleep(2)
    proc.terminate()
    raise SystemExit(f"FAIL: llama-server not healthy after 900 s; see {log_path}")


def post(cfg: dict, path: str, body: dict) -> dict:
    q = cfg["quality"]
    req = urllib.request.Request(f"http://127.0.0.1:{q['port']}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with _NO_PROXY.open(req, timeout=q["request_timeout_sec"]) as r:
        return json.loads(r.read())


PARSE_ERROR = "does not match the expected"  # llama-server could not parse the reply into its chat format


def parse_error(e: urllib.error.HTTPError) -> str | None:
    """The server's message if llama-server rejected a generated reply as unparseable, else None."""
    detail = e.read().decode("utf-8", "replace")
    return detail[:300] if e.code == 500 and PARSE_ERROR in detail else None


def chat(cfg: dict, prompt: str, max_tokens: int, errors: list | None = None) -> str:
    """Chat completion. llama-server answers HTTP 500 when the generated reply does not fit the
    model's chat-output format (seen with collapsed low-bit quants: tool-call-like text, or broken
    UTF-8 byte sequences). Then the same prompt is rendered with /apply-template and regenerated
    with /completion (same sampling); if that is rejected as well, the client never gets an
    answer and the reply is empty (scored wrong). Each server error is appended to `errors`."""
    errors = errors if errors is not None else []
    sampling = {"temperature": cfg["inference"]["temperature"], "seed": cfg["inference"]["seed"], "cache_prompt": False}
    messages = [{"role": "user", "content": prompt}]
    try:
        r = post(cfg, "/v1/chat/completions", {"messages": messages, "max_tokens": max_tokens, **sampling})
        return r["choices"][0]["message"]["content"] or ""
    except urllib.error.HTTPError as e:
        detail = parse_error(e)
        if detail is None:
            raise
        errors.append(detail)
    try:
        raw = post(cfg, "/apply-template", {"messages": messages})["prompt"]
        return post(cfg, "/completion", {"prompt": raw, "n_predict": max_tokens, **sampling}).get("content") or ""
    except urllib.error.HTTPError as e:
        detail = parse_error(e)
        if detail is None:
            raise
        errors.append(detail)
        return ""

# --------------------------------------------------------------------------- main


def read_done(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def write_summary(qdir: Path, hw_id: str, sample: dict) -> Path:
    rows = []
    for f in sorted(qdir.glob("*/*/*.jsonl")):
        model, quant, task = f.parent.parent.name, f.parent.name, f.stem
        recs = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
        n, correct = len(recs), sum(r["correct"] for r in recs)
        rows.append({"hw_id": hw_id, "model": model, "quant": quant, "task": task, "n": n,
                     "expected_n": sample["datasets"][task]["n"], "correct": correct,
                     "accuracy": correct / n if n else None})
    qdir.mkdir(parents=True, exist_ok=True)
    out = qdir / "quality.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["hw_id", "model", "quant", "task", "n", "expected_n", "correct", "accuracy"])
        w.writeheader()
        w.writerows(rows)
    return out


SERVER_RESTARTS = 2  # llama-server crashes (e.g. CUDA out of memory with CPU offload) are retried


def run_tasks(cfg: dict, items: dict, tasks: list[str], out_dir: Path, tag: str, lock, log) -> None:
    """Runs the items of each task not yet in out_dir/<task>.jsonl (resumable)."""
    q = cfg["quality"]
    for task in tasks:
        out_file = out_dir / f"{task}.jsonl"
        seen = read_done(out_file)
        pending = [it for it in items[task] if it["id"] not in seen]
        if not pending:
            continue
        spec = q["datasets"][task]
        log(f"{tag}/{task}: {len(pending)} items")

        def work(row: dict, task=task, spec=spec, out_file=out_file) -> None:
            errors: list[str] = []
            text = chat(cfg, PROMPTS[task](row), spec["max_tokens"], errors)
            pred, correct = score(task, row, text, q["exec_timeout_sec"])
            rec = {"id": row["id"], "pred": pred, "correct": bool(correct), "response": text}
            if errors:
                rec["api_error"] = errors[0]
            with lock, open(out_file, "a", encoding="utf-8", newline="") as fh:
                fh.write(json.dumps(rec) + "\n")

        with ThreadPoolExecutor(q["n_parallel"]) as ex:
            list(ex.map(work, pending))
        log(f"{tag}/{task}: done")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", help="model name from config, or 'all' (default: default_model)")
    ap.add_argument("--quants", nargs="+", help="quant levels in run order (default: all, F16 first)")
    ap.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    ap.add_argument("--build-sample-only", action="store_true", help="write prompts/quality.json and exit")
    ap.add_argument("--out-root", help="results root instead of paths.quality_dir (e.g. data/quality_rerun)")
    args = ap.parse_args()

    cfg = load_config()
    q = cfg["quality"]
    log = make_logger(cfg, "quality")
    sample = load_sample(cfg, log)
    if args.build_sample_only:
        return
    hw = detect_hw()
    qdir = ROOT_DIR / (args.out_root or cfg["paths"]["quality_dir"]) / hw["hw_id"]
    items = {t: load_items(sample, t) for t in args.tasks}
    lock = threading.Lock()

    for model in select_models(cfg, args.model):
        name = model["name"]
        for quant in args.quants or quant_levels(cfg):
            tag = f"{name}/{quant}"
            out_dir = qdir / name / quant
            todo = {}
            for t in args.tasks:
                seen = read_done(out_dir / f"{t}.jsonl")
                todo[t] = [it for it in items[t] if it["id"] not in seen]
            if not any(todo.values()):
                log(f"SKIP {tag}: all tasks complete")
                continue
            ok, why = check_budget(cfg, name, quant, q["n_parallel"] * q["ctx_per_slot"], hw, allow_host_offload=True)
            if not ok:
                log(f"SKIP {tag}: over memory budget ({why})")
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            for attempt in range(1 + SERVER_RESTARTS):
                log(f"{tag}: starting llama-server ({why})")
                suffix = f".restart{attempt}" if attempt else ""  # keep the crashed server's log
                server = start_server(cfg, gguf_path(cfg, name, quant),
                                      ROOT_DIR / cfg["paths"]["logs_dir"] / f"server_{name}_{quant}{suffix}.log")
                try:
                    run_tasks(cfg, items, args.tasks, out_dir, tag, lock, log)
                    break
                except (ConnectionError, urllib.error.URLError) as e:
                    if isinstance(e, urllib.error.HTTPError) or attempt == SERVER_RESTARTS:
                        raise
                    try:
                        server.wait(timeout=15)  # a crashed server is exiting
                    except subprocess.TimeoutExpired:
                        raise e from None  # server still running: not a crash
                    log(f"{tag}: llama-server exited with {server.returncode} ({e!r}); restart {attempt + 1}/"
                        f"{SERVER_RESTARTS}, resuming the remaining items")
                finally:
                    if server.poll() is None:
                        server.terminate()
                        server.wait(timeout=30)

    log(f"Quality complete. Summary: {write_summary(qdir, hw['hw_id'], sample)}")


if __name__ == "__main__":
    main()
