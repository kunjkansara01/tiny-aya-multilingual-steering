"""Controlled throughput check: baseline vs steered on identical batches, interleaved.

The grid timings in timing/*.jsonl come from two passes run minutes apart, so they also pick up whatever
else the GPU was doing. Here the same batches are generated back to back (baseline, steered, baseline, ...)
with a fixed number of decode steps (min_new_tokens = max_new_tokens), so the only difference is the
36 forward hooks that add ``alpha * direction``.

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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=str(ROOT / "results" / "final"))
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=96)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    import torch
    from safetune._refusal_helpers import _encode_prompts, _left_padding
    from safetune.steer import RefusalDirectionModel

    run = Path(args.run_dir)
    state = json.loads((run / "state.json").read_text(encoding="utf-8"))
    bundle = torch.load(run / "direction.pt", map_location="cpu")
    model, tok = load_model(state["model"], state["quant"])
    rows = load_eval_prompts(ROOT / "prompts", ["en", "hi", "gu"])
    batches = {lg: [r["prompt"] for r in rows if r["lang"] == lg and r["set"] == "benign_edgy"][:args.batch]
               for lg in ("en", "hi", "gu")}
    wrapped = RefusalDirectionModel(model, direction=bundle["direction"], mode="steer", strength=state["alpha"])

    def once(prompts, steered: bool) -> float:
        with _left_padding(tok):
            enc = _encode_prompts(tok, prompts, return_tensors="pt", padding=True)
        enc = {k: v.to(next(model.parameters()).device) for k, v in enc.items()}
        if steered:
            wrapped.install()
        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                model.generate(**enc, max_new_tokens=args.steps, min_new_tokens=args.steps, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
            torch.cuda.synchronize()
            return time.perf_counter() - t0
        finally:
            wrapped.remove()

    once(batches["en"], False), once(batches["en"], True)  # warm-up, discarded
    times: dict = {lg: {"baseline": [], "steered": []} for lg in batches}
    for rnd in range(args.rounds):
        for lg, prompts in batches.items():
            order = (False, True) if rnd % 2 == 0 else (True, False)  # alternate which condition goes first
            for steered in order:
                secs = once(prompts, steered)
                times[lg]["steered" if steered else "baseline"].append(secs)
                print(f"round {rnd} {lg} {'steered ' if steered else 'baseline'} {secs:.2f}s "
                      f"{args.batch * args.steps / secs:.1f} tok/s", flush=True)
    out = {"model": state["model"], "quant": state["quant"], "alpha": state["alpha"], "batch": args.batch,
           "steps": args.steps, "rounds": args.rounds, "by_lang": {}}
    all_b, all_s = [], []
    for lg, t in times.items():
        b = [args.batch * args.steps / x for x in t["baseline"]]
        s = [args.batch * args.steps / x for x in t["steered"]]
        all_b += b
        all_s += s
        out["by_lang"][lg] = {"baseline_tok_s": round(statistics.mean(b), 1), "steered_tok_s": round(statistics.mean(s), 1),
                              "baseline_sd": round(statistics.pstdev(b), 1), "steered_sd": round(statistics.pstdev(s), 1)}
    mb, ms = statistics.mean(all_b), statistics.mean(all_s)
    out["all"] = {"baseline_tok_s": round(mb, 1), "steered_tok_s": round(ms, 1),
                  "baseline_sd": round(statistics.pstdev(all_b), 1), "steered_sd": round(statistics.pstdev(all_s), 1),
                  "overhead_pct": round(100 * (mb / ms - 1), 1), "n_timings_per_condition": len(all_b)}
    write_json(run / "throughput_bench.json", out)
    print(json.dumps(out["all"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
