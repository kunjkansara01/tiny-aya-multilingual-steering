"""Day-2 stretch: WHERE does the refusal direction live? Safety-neuron localisation with SafeTune's
interpret pillar, run on the direction saved by eval_steering.py.

Status: STUB that already runs end to end for the cheap "weight" method. The Day-2 analysis ideas are
listed under TODO at the bottom; nothing here is needed for the Phase-1 steering results.

What it does today
  * loads <run-dir>/direction.pt (per-layer refusal directions from the steering calibration);
  * weight method  (no forward passes): ranks each MLP neuron by |cos(down_proj column, refusal direction)|
    with ``safetune.interpret.identify_safety_neurons``;
  * activation method (--activation, needs forward passes): ranks neurons by harmful-vs-harmless
    activation contrast, optionally once per language (--per-lang) to compare EN and HI neuron sets;
  * writes <run-dir>/interpret/neurons_<method>[_<lang>].json and a per-layer summary.

Caveats from SafeTune's own docs (keep them in the article):
  * both scores are SafeTune heuristics ("Variant" badge), not the metric of Chen et al. 2406.14144;
  * the activation method reads raw prompt tokens WITHOUT the chat template and keeps the last 64 tokens.

Usage
  python src/interpret_run.py --run-dir results/dryrun --dry-run
  python src/interpret_run.py --run-dir results/tiny-aya --activation --per-lang
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_calibration, parse_csv, write_json  # noqa: E402


def report_to_json(report, n_layers: int) -> dict:
    per_layer = {int(l): [(int(u), float(s)) for u, s in units] for l, units in report.per_layer.items()}
    flat = sorted(((abs(s), l, u, s) for l, units in per_layer.items() for u, s in units), reverse=True)
    return {
        "method": report.method, "target_module": report.target_module, "n_layers": n_layers,
        "top_overall": [{"layer": l, "neuron": u, "score": round(s, 5)} for _, l, u, s in flat[:50]],
        "layer_mean_abs_score": {str(l): round(sum(abs(s) for _, s in units) / max(1, len(units)), 5)
                                 for l, units in sorted(per_layer.items())},
        "per_layer": {str(l): [[u, round(s, 5)] for u, s in units] for l, units in sorted(per_layer.items())},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="a run directory that contains direction.pt")
    ap.add_argument("--model", default=None, help="defaults to the model recorded in direction.pt")
    ap.add_argument("--top-k", type=int, default=16, help="neurons kept per layer")
    ap.add_argument("--activation", action="store_true", help="also run the activation-contrast method")
    ap.add_argument("--per-lang", action="store_true", help="activation method once per calibration language")
    ap.add_argument("--calib-langs", default="en,hi")
    ap.add_argument("--prompt-dir", default=str(ROOT / "prompts"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dry-run", action="store_true", help="CPU, weight method only, top-k 4")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import safetune
    import torch
    from safetune.interpret import SafetyNeuronConfig, identify_safety_neurons
    from safetune.runner.utils import load_model, load_tok

    run = Path(args.run_dir)
    bundle = torch.load(run / "direction.pt", map_location="cpu")
    model_id = args.model or bundle["model"]
    if args.dry_run:
        args.device, args.top_k, args.activation = "cpu", 4, False
    safetune.configure(device=args.device)
    tok, model = load_tok(model_id), load_model(model_id)
    out = run / "interpret"
    n_layers = bundle["n_layers"]

    # 1. weight method: which MLP neurons write onto the refusal direction of their own layer?
    rep = identify_safety_neurons(model, bundle["per_layer"], SafetyNeuronConfig(method="weight", top_k_per_layer=args.top_k))
    res = report_to_json(rep, n_layers)
    res["steering_layer"] = bundle["layer"]
    write_json(out / "neurons_weight.json", res)
    print(f"[weight] top neurons (direction picked at layer {bundle['layer']} of {n_layers}):")
    for t in res["top_overall"][:8]:
        print(f"  layer {t['layer']:2d}  neuron {t['neuron']:6d}  |cos| {t['score']:.3f}")

    # 2. activation method: which neurons fire differently on harmful vs harmless prompts?
    if args.activation:
        harm, safe = load_calibration(args.prompt_dir, parse_csv(args.calib_langs))
        groups = {"all": (harm, safe)}
        if args.per_lang:
            groups.update({lg: ([r for r in harm if r["lang"] == lg], [r for r in safe if r["lang"] == lg])
                           for lg in parse_csv(args.calib_langs)})
        tops: dict[str, set] = {}
        for name, (h, s) in groups.items():
            rep = identify_safety_neurons(
                model, {}, SafetyNeuronConfig(method="activation", top_k_per_layer=args.top_k),
                tokenizer=tok, harmful_prompts=[r["prompt"] for r in h], harmless_prompts=[r["prompt"] for r in s])
            res = report_to_json(rep, n_layers)
            write_json(out / f"neurons_activation_{name}.json", res)
            tops[name] = {(l, u) for l, units in rep.per_layer.items() for u, _ in units}
            print(f"[activation/{name}] strongest layer by mean |contrast|: "
                  f"{max(res['layer_mean_abs_score'], key=res['layer_mean_abs_score'].get)}")
        if args.per_lang and len(tops) > 2:
            langs = [k for k in tops if k != "all"]
            a, b = tops[langs[0]], tops[langs[1]]
            jac = len(a & b) / max(1, len(a | b))
            write_json(out / "language_overlap.json", {"langs": langs[:2], "jaccard_top_neurons": jac,
                                                       "shared": len(a & b), "top_k_per_layer": args.top_k})
            print(f"[activation] {langs[0]} vs {langs[1]} top-neuron Jaccard overlap: {jac:.3f}")
    print(f"wrote {out}")
    return 0

# TODO (Day 2, only if Phase-1 results are in):
#   1. Layer profile: plot layer_mean_abs_score against cosine_of_layer_direction_to_picked (calibration.json):
#      do the "writer" neurons cluster around the layer the steering sweep picked?
#   2. Language overlap: --activation --per-lang gives the Jaccard overlap of EN vs HI safety neurons.
#      Low overlap would explain weaker refusal in HI/GU; high overlap supports one shared direction.
#   3. Causal check: zero the top-k weight-method neurons (forward hook on mlp.down_proj input) and
#      re-run eval_steering.py --limit 10 to see whether harmful refusal drops; compare to k random neurons.
#   4. Publish neurons_*.json with the run (the notebook's upload cell already pushes the whole run dir).


if __name__ == "__main__":
    raise SystemExit(main())
