"""Controlled throughput check: baseline vs steered on identical batches, tightly paired.

The grid timings in timing/*.jsonl come from passes run minutes apart, and on this 8 GB laptop GPU the
per-step time flips between a fast and a slow state for reasons unrelated to steering (FINDINGS R3).
So: short runs with a fixed number of decode steps (min_new_tokens = max_new_tokens), the conditions
back to back inside each trial, order rotated, and **medians of the within-trial ratios**, which a
state flip in the middle of the benchmark cannot move much.

Conditions
  baseline      no hooks
  steered       SafeTune RefusalDirectionModel exactly as in the grids (direction held on the CPU in float32;
                the hook moves it to the GPU on every call, 36 hooks per decode step)
  steered_gpu   same hooks, direction placed on the GPU once (same numbers; removes the per-call transfer)

  python src/bench_throughput.py --run-dir results/final      -> <run-dir>/throughput_bench.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_eval_prompts, write_json  # noqa: E402
from run_final import load_model  # noqa: E402

CONDS = ("baseline", "steered", "steered_gpu")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=str(ROOT / "results" / "final"))
    ap.add_argument("--trials", type=int, default=24, help="upper bound; one trial = one batch under all conditions")
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--budget-min", type=float, default=7.0, help="stop starting new trials after this many minutes")
    args = ap.parse_args()
    import torch
    from safetune._refusal_helpers import _encode_prompts, _left_padding
    from safetune.steer import RefusalDirectionModel

    run = Path(args.run_dir)
    state = json.loads((run / "state.json").read_text(encoding="utf-8"))
    bundle = torch.load(run / "direction.pt", map_location="cpu")
    model, tok = load_model(state["model"], state["quant"])
    dev = next(model.parameters()).device
    rows = load_eval_prompts(ROOT / "prompts", ["en", "hi", "gu"])
    langs = ("en", "hi", "gu")
    encs = {}
    for lg in langs:
        prompts = [r["prompt"] for r in rows if r["lang"] == lg and r["set"] == "benign_edgy"][:args.batch]
        with _left_padding(tok):
            enc = _encode_prompts(tok, prompts, return_tensors="pt", padding=True)
        encs[lg] = {k: v.to(dev) for k, v in enc.items()}
    wrap = {"steered": RefusalDirectionModel(model, direction=bundle["direction"], mode="steer", strength=state["alpha"]),
            "steered_gpu": RefusalDirectionModel(model, direction=bundle["direction"].to(dev), mode="steer",
                                                 strength=state["alpha"])}

    def once(lg: str, cond: str) -> float:
        w = wrap.get(cond)
        if w is not None:
            w.install()
        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                model.generate(**encs[lg], max_new_tokens=args.steps, min_new_tokens=args.steps, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
            torch.cuda.synchronize()
            return time.perf_counter() - t0
        finally:
            if w is not None:
                w.remove()
            torch.cuda.empty_cache()

    for c in CONDS:  # warm-up, discarded
        once("en", c)
    trials, t_start = [], time.time()
    for i in range(args.trials):
        if time.time() - t_start > args.budget_min * 60:
            break
        lg = langs[i % 3]
        order = CONDS[i % 3:] + CONDS[:i % 3]  # rotate which condition goes first
        secs = {c: once(lg, c) for c in order}
        trials.append({"lang": lg, "order": list(order), **{c: round(secs[c], 3) for c in CONDS}})
        print(f"trial {i:2d} {lg}  " + "  ".join(f"{c} {secs[c]:6.2f}s" for c in CONDS)
              + f"   steered/baseline {secs['steered'] / secs['baseline']:.2f}  gpu/baseline "
                f"{secs['steered_gpu'] / secs['baseline']:.2f}", flush=True)

    toks = args.batch * args.steps

    def med_tok_s(c, part):
        return round(statistics.median(toks / t[c] for t in part), 1)

    def summary(part):
        r1 = [t["steered"] / t["baseline"] for t in part]
        r2 = [t["steered_gpu"] / t["baseline"] for t in part]
        return {"trials": len(part), **{f"{c}_tok_s_median": med_tok_s(c, part) for c in CONDS},
                "baseline_ms_per_step_median": round(1000 * statistics.median(t["baseline"] for t in part) / args.steps, 1),
                "steered_over_baseline_time_median": round(statistics.median(r1), 3),
                "steered_gpu_over_baseline_time_median": round(statistics.median(r2), 3),
                "steered_over_baseline_time_iqr": [round(x, 3) for x in statistics.quantiles(r1, n=4)[::2]] if len(r1) >= 4 else None,
                "steered_gpu_over_baseline_time_iqr": [round(x, 3) for x in statistics.quantiles(r2, n=4)[::2]] if len(r2) >= 4 else None}

    fastest = min(t["baseline"] for t in trials)
    out = {"model": state["model"], "quant": state["quant"], "alpha": state["alpha"], "batch": args.batch,
           "steps": args.steps, "all": summary(trials),
           "by_lang": {lg: summary([t for t in trials if t["lang"] == lg]) for lg in langs
                       if any(t["lang"] == lg for t in trials)},
           # trials where the baseline ran within 1.5x of the fastest baseline seen: the GPU's undisturbed state
           "fast_state": summary([t for t in trials if t["baseline"] <= 1.5 * fastest]),
           "trials_raw": trials}
    write_json(run / "throughput_bench.json", out)
    print(json.dumps({k: out[k] for k in ("all", "fast_state")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
