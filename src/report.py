"""Writes docs/RESULTS.md: every number of the report, generated from results/tables (no hand copying).

Sections: setup, accuracy vs the published scores, speed, regression gate with items needed,
answer flips with the re-run noise floor, size vs bits, perplexity screen, capacity under the
latency SLAs, decode-speed model. Missing tables are skipped. Run after the analysis scripts.

    python src/report.py [--hw HW_ID]
"""

from __future__ import annotations

import argparse
import json
import math

import pandas as pd
import yaml

from analyze import QUANT_ORDER
from common import GIB, ROOT_DIR, load_config, params_b

TASKS = ["mmlu", "gsm8k", "humaneval"]


def md(df: pd.DataFrame) -> str:
    """Markdown table; None/NaN as blank."""
    def cell(v) -> str:
        return "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)
    head = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(head + ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)])


def pct(x, nd: int = 1) -> str:
    return "" if x is None or pd.isna(x) else f"{100 * x:.{nd}f}"


def pp(x, nd: int = 1) -> str:
    return "" if x is None or pd.isna(x) else f"{100 * x:+.{nd}f}"


def qsort(df: pd.DataFrame, model: str = "model", quant: str = "quant") -> pd.DataFrame:
    return df.assign(_m=df[model].map(params_b), _q=df[quant].map({q: i for i, q in enumerate(QUANT_ORDER)})) \
        .sort_values(["_m", "_q"]).drop(columns=["_m", "_q"])


def short(model: str) -> str:
    return model.replace("Qwen2.5-", "").replace("-Instruct", "")


def api_errors(quality_dir) -> dict[tuple[str, str], int]:
    """Items per (model, quant) whose chat request failed because llama-server could not parse the
    reply (scored on the raw completion instead; field api_error in the item files)."""
    out: dict[tuple[str, str], int] = {}
    for f in quality_dir.glob("*/*/*/*.jsonl"):
        n = sum('"api_error"' in line for line in f.read_text(encoding="utf-8").splitlines())
        if n:
            key = (f.parent.parent.name, f.parent.name)
            out[key] = out.get(key, 0) + n
    return out


def accuracy_section(t: dict, ref: dict, n_items: dict, errors: dict | None = None) -> list[str]:
    f, q = t.get("frontier"), t.get("quality")
    if f is None or q is None:
        return []
    q = q.set_index(["model", "quant"])
    errors = errors or {}
    rows = []
    for r in qsort(f).itertuples():
        row = {"model": short(r.model), "quant": r.quant, "GiB": f"{r.size_gib:.2f}", "bits/w": f"{r.bpw:.2f}"}
        for task in TASKS:
            lo, hi = q.loc[(r.model, r.quant), f"{task}_ci_low"], q.loc[(r.model, r.quant), f"{task}_ci_high"]
            row[task] = f"{pct(getattr(r, task))} ({pct(lo)}-{pct(hi)})"
        row.update({"mean": pct(r.quality), "PPL": f"{r.ppl:.3f}" if pd.notna(r.ppl) else "",
                    "gate vs ref": getattr(r, "gate", ""), "Pareto": "yes" if r.pareto else ""})
        if errors:
            row["chat API errors"] = errors.get((r.model, r.quant), "")
        rows.append(row)
        if r.quant == getattr(r, "reference", "F16") and r.model in ref["scores"]:
            s = ref["scores"][r.model]
            rows.append({"model": short(r.model), "quant": "published", "GiB": "", "bits/w": "",
                         **{t_: f"{s[t_]:.1f}" for t_ in TASKS}, "mean": "", "PPL": "", "gate vs ref": "", "Pareto": "",
                         **({"chat API errors": ""} if errors else {})})
    sizes = ", ".join(f"{k} {v}" for k, v in n_items.items())
    return ["## 1. Accuracy (95% Wilson CI) and the published scores", "",
            f"Accuracy in % on the fixed sample ({sizes} items); `published` rows: {ref['source']} ({ref['url']}), "
            "MMLU column = MMLU-Redux there. `mean` = unweighted mean of the three tasks. "
            "Pareto: no other configuration is both smaller and more accurate."
            + (" `chat API errors`: items where llama-server returned HTTP 500 because the generated reply did not "
               "fit its chat-output format; scored on a raw /completion of the same prompt, or as wrong when that "
               "was rejected too (the client got no answer)." if errors else ""),
            "", md(pd.DataFrame(rows)), ""]


def speed_section(t: dict, hw: str, ctx: int) -> list[str]:
    s = t.get("summary")
    if s is None:
        return []
    s = qsort(s[(s.hw_id == hw) & (s.ctx == ctx)])
    if s.empty:
        return []
    rows = [{"model": short(r.model), "quant": r.quant, "decode tok/s": f"{r.generation_throughput_tps:.1f}",
             "prefill tok/s": f"{r.prompt_throughput_tps:.0f}",
             "peak VRAM GiB": f"{r.peak_vram_bytes / GIB:.2f}" if pd.notna(r.peak_vram_bytes) else "",
             "CPU offload": "yes" if r.partial_offload else "",
             "GPU max C": f"{r.tg_gpu_temp_max_c:.0f}" if "tg_gpu_temp_max_c" in s and pd.notna(r.tg_gpu_temp_max_c) else ""}
            for r in s.itertuples()]
    return [f"## 2. Speed on {hw} (llama-bench, ctx {ctx})", "", md(pd.DataFrame(rows)), ""]


def gate_section(t: dict) -> list[str]:
    g = t.get("gate")
    if g is None or g.empty:
        return []
    overall = g[g.task == "overall"].set_index(["model", "candidate"]).verdict
    g = qsort(g[g.task != "overall"], quant="candidate")
    rows = [{"model": short(r.model), "reference": r.reference, "candidate": r.candidate, "task": r.task, "n": int(r.n),
             "diff %p": pp(r.diff), "95% CI %p": f"{pp(r.diff_ci_low)} to {pp(r.diff_ci_high)}",
             "McNemar p": f"{r.p_mcnemar:.2g}", "discordant %": pct(r.p_discordant),
             "items needed": "" if not math.isfinite(r.n_needed) else f"{r.n_needed:.0f}",
             "task verdict": r.verdict, "overall": overall.get((r.model, r.candidate), "")}
            for r in g.itertuples()]
    return ["## 3. Regression gate vs the reference and items needed", "",
            "Reference: F16, or Q8_0 for a model whose F16 does not fit this machine.", "",
            "PASS: lower 95% CI bound above -3 %p; FAIL: significantly worse (McNemar, Bonferroni over tasks). "
            "`items needed`: items for an 80% chance of PASS if the true difference is 0, at the observed "
            "discordance (src/gate.py).", "", md(pd.DataFrame(rows)), ""]


def flips_section(t: dict) -> list[str]:
    f = t.get("flips")
    if f is None or f.empty:
        return []
    out = ["## 4. Answer flips vs the reference and the re-run noise floor", "",
           "Flips: items whose correctness differs from the reference (F16; Q8_0 where F16 was not evaluated; "
           "Dutta et al. 2024). Noise floor: flips between two identical runs of the same configuration "
           "(`same`) or, without one, of the model's quantized re-runs (`proxy`). attributable = flips - noise "
           "floor, a conservative lower bound.", ""]
    v = qsort(f[f.kind == "vs_ref"])
    rows = [{"model": short(r.model), "quant": r.quant, "vs": r.reference, "task": r.task, "n": int(r.n),
             "accuracy diff %p": pp(r.acc_diff),
             "flips %": f"{pct(r.flips)} ({pct(r.flips_ci_low)}-{pct(r.flips_ci_high)})",
             "noise floor %": pct(r.noise_floor), "noise basis": getattr(r, "noise_basis", "") or "",
             "attributable %p": pp(r.attributable)} for r in v.itertuples()]
    out += [md(pd.DataFrame(rows)), ""]
    rr = f[f.kind == "rerun"]
    if not rr.empty:
        rows = [{"model": short(r.model), "quant": r.quant, "task": r.task, "n": int(r.n),
                 "flips between identical runs %": f"{pct(r.flips)} ({pct(r.flips_ci_low)}-{pct(r.flips_ci_high)})",
                 "accuracy diff %p": pp(r.acc_diff)} for r in qsort(rr).itertuples()]
        out += ["Identical re-runs:", "", md(pd.DataFrame(rows)), ""]
    return out


def duel_section(t: dict) -> list[str]:
    d, s = t.get("size_vs_bits"), t.get("ppl_screen")
    out = []
    if d is not None and not d.empty:
        d = d[d.task == "all"]
        rows = [{"bigger model": r.big.replace("Qwen2.5-", "").replace("-Instruct", ""), "GiB": f"{r.big_gib:.2f}",
                 "smaller model": r.small.replace("Qwen2.5-", "").replace("-Instruct", ""), "GiB ": f"{r.small_gib:.2f}",
                 "n": int(r.n), "bigger - smaller %p": pp(r.big_minus_small),
                 "95% CI %p": f"{pp(r.diff_ci_low)} to {pp(r.diff_ci_high)}", "McNemar p": f"{r.p_mcnemar:.2g}"}
                for r in d.itertuples()]
        out += ["## 5. Size vs bits at matched memory", "",
                "Each configuration of a bigger model vs the closest-size configuration of each smaller model "
                "(size ratio 0.8-1.25), on the same items pooled over the three tasks.", "", md(pd.DataFrame(rows)), ""]
    if s is not None and not s.empty:
        rows = [{"ln(PPL/PPL_ref) vs": r.metric, "n configs": int(r.n), "Spearman rho": f"{r.spearman_rho:.2f}",
                 "p": f"{r.p:.2g}"} for r in s.itertuples()]
        out += ["## 6. Perplexity as a screen", "", md(pd.DataFrame(rows)), ""]
    return out


def capacity_section(t: dict, cfg: dict) -> list[str]:
    c, lv = t.get("capacity"), t.get("capacity_levels")
    if c is None or c.empty:
        return []
    lt = cfg["loadtest"]
    sla = "; ".join(f"{k}: TTFT <= {v['ttft_s'] * 1000:.0f} ms and TPOT <= {v['tpot_s'] * 1000:.0f} ms"
                    for k, v in lt["sla"].items())
    rows = [{"hw": r.hw_id, "model": short(r.model), "quant": r.quant, "users (server SLA)": int(r.capacity_server),
             "users (interactive SLA)": int(r.capacity_interactive),
             "tok/s at server capacity": f"{r.out_tps_total_at_server:.0f}" if pd.notna(r.out_tps_total_at_server) else "",
             "max tok/s": f"{r.max_out_tps_total:.0f}"} for r in qsort(c).itertuples()]
    out = ["## 7. Capacity: concurrent users within the latency SLAs", "",
           f"Closed loop, GSM8K prompts, up to {lt['max_tokens']} output tokens, {lt['slots']} llama-server slots; "
           f"SLA at p{lt['sla_percentile']} ({sla}; MLPerf Inference v5.0 Llama-2-70B limits).", "",
           md(pd.DataFrame(rows)), ""]
    if lv is not None and not lv.empty:
        rows = [{"model": short(r.model), "quant": r.quant, "users": int(r.level), "requests": int(r.n),
                 "TTFT p50/p95 s": f"{r.ttft_p50_s:.2f} / {r.ttft_p95_s:.2f}",
                 "TPOT p50/p95 ms": f"{r.tpot_p50_s * 1000:.0f} / {r.tpot_p95_s * 1000:.0f}",
                 "tok/s per user (p50)": f"{r.user_tps_p50:.1f}", "tok/s total": f"{r.out_tps_total:.0f}"}
                for r in qsort(lv).itertuples()]
        out += [md(pd.DataFrame(rows)), ""]
    return out


def perf_section(t: dict) -> list[str]:
    p = t.get("perf_model")
    if p is None or p.empty:
        return []
    rows = [{"hw": r.hw_id, "configs": int(r.n), "published GB/s": f"{r.peak_bw_gbs:.0f}",
             "efficiency": f"{r.efficiency:.2f}", "fit error (MAPE)": f"{pct(min(r.mape_1p, r.mape_2p))}%",
             "cross-device error (MAPE)": f"{pct(r.loho_mape_min)}-{pct(r.loho_mape_max)}%",
             "calibrated error (MAPE / max)": f"{pct(r.calib_mape)}% / {pct(r.calib_max_ape)}% ({int(r.calib_n_pred)} configs)"
             if "calib_mape" in p and pd.notna(r.calib_mape) else ""} for r in p.itertuples()]
    return ["## 8. Decode-speed model (src/perfmodel.py)", "",
            "tokens/s = effective bandwidth / bytes read per token. Cross-device: predicted from the other "
            "device's efficiency and this device's published bandwidth. Calibrated: two llama-bench runs of the "
            "smallest model on this device (F16, Q4_K_M) predict the bigger models.", "", md(pd.DataFrame(rows)), ""]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hw", help="hw_id for the speed table (default: first CUDA hw_id)")
    args = ap.parse_args()
    cfg = load_config()
    tables = ROOT_DIR / cfg["paths"]["tables_dir"]
    names = ["summary", "quality", "frontier", "gate", "flips", "size_vs_bits", "ppl_screen", "capacity",
             "capacity_levels", "perf_model"]
    t = {n: pd.read_csv(tables / f"{n}.csv") for n in names if (tables / f"{n}.csv").exists()}
    ref = yaml.safe_load((ROOT_DIR / "config" / "reference_scores.yaml").read_text(encoding="utf-8"))
    hws = sorted(set(t["summary"].hw_id)) if "summary" in t else []
    hw = args.hw or next((h for h in hws if h.startswith("nvidia")), hws[0] if hws else "")
    meta = json.loads((ROOT_DIR / cfg["paths"]["summary_json"]).read_text(encoding="utf-8")) \
        if (ROOT_DIR / cfg["paths"]["summary_json"]).exists() else {}
    h = meta.get("hardware", {}).get(hw, {})
    first = lambda k: (h.get(k) or [""])[0]  # noqa: E731
    lines = ["# Q-Bench results", "",
             "Generated by `src/report.py` from `results/tables/` - do not edit by hand.", "",
             f"Hardware: {first('hardware_gpu')} ({float(first('hardware_vram_bytes') or 0) / GIB:.1f} GiB VRAM), "
             f"{first('hardware_cpu')}, {float(first('hardware_memory_bytes') or 0) / GIB:.0f} GiB RAM, {first('os')}; "
             f"llama.cpp {meta.get('llama_cpp_commit', '')} (binaries build {first('build_commit')}).", ""]
    sample = ROOT_DIR / cfg["quality"]["sample_file"]
    n_items = {k: d["n"] for k, d in json.loads(sample.read_text(encoding="utf-8"))["datasets"].items()} \
        if sample.exists() else {}
    errors = api_errors(ROOT_DIR / cfg["paths"]["quality_dir"])
    for sec in (accuracy_section(t, ref, n_items, errors), speed_section(t, hw, cfg["inference"]["context_length"]), gate_section(t),
                flips_section(t), duel_section(t), capacity_section(t, cfg), perf_section(t)):
        lines += sec
    out = ROOT_DIR / "docs" / "RESULTS.md"
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"wrote {out.relative_to(ROOT_DIR).as_posix()} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
