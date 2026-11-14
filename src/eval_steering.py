"""Refusal-direction steering sweep with SafeTune: calibrate once, then generate baseline + one pass per alpha.

Pipeline
  1. calibrate   extract the refusal direction (SafeTune ``extract_refusal_direction``) from harmful vs
                 harmless calibration prompts; the layer comes from SafeTune's validation sweep, or from
                 --pick-layer. Saved to <out-dir>/direction.pt + calibration.json.
  2. generate    for every condition (baseline, alpha_X.XX, optional ablate) answer every eval prompt.
                 One JSONL row is appended per answer, so an interrupted run resumes where it stopped.
  3. score       python src/score.py --run-dir <out-dir>     (separate script)

Why not ``RefusalDirectionTrainer``? Its ``calibrate()`` builds the direction with the default config,
so it cannot take ``pick_layer`` and it would redo the extraction for every alpha. This script calls
the same two SafeTune pieces the trainer wraps: ``extract_refusal_direction`` and
``RefusalDirectionModel(mode="steer", strength=alpha)``.

Examples
  python src/eval_steering.py --dry-run --out-dir results/dryrun            # CPU, Qwen 0.5B, 6 prompts/set, EN
  python src/eval_steering.py --model CohereLabs/tiny-aya-global \
         --alpha 0.15,0.3,0.5,0.8 --langs en,hi,gu --out-dir results/tiny-aya
  python src/eval_steering.py --model CohereLabs/tiny-aya-global --pick-layer 24 --out-dir results/tiny-aya-l24
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ROOT, append_jsonl, load_calibration, load_eval_prompts, parse_csv,  # noqa: E402
                    read_jsonl, write_json)

DEFAULT_MODEL = "CohereLabs/tiny-aya-global"
DRY_RUN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_ALPHAS = "0.15,0.3,0.5,0.8"
log = logging.getLogger("eval_steering")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help=f"HF model id (default {DEFAULT_MODEL}; {DRY_RUN_MODEL} with --dry-run)")
    ap.add_argument("--alpha", default=None, help=f"comma-separated steering strengths (default {DEFAULT_ALPHAS})")
    ap.add_argument("--pick-layer", type=int, default=None,
                    help="take the direction from this layer and skip SafeTune's sweep (README example for Tiny Aya: 24)")
    ap.add_argument("--langs", default=None, help="comma-separated eval languages (default en,hi,gu)")
    ap.add_argument("--prompt-dir", default=str(ROOT / "prompts"))
    ap.add_argument("--out-dir", default=None, help="run directory (default results/<model name>)")
    ap.add_argument("--dry-run", action="store_true",
                    help="CPU smoke test: small model, 6 prompts/set, EN only, one alpha, short generations")
    # less common knobs
    ap.add_argument("--limit", type=int, default=None, help="first N prompts of every (language, set)")
    ap.add_argument("--max-new-tokens", type=int, default=None, help="default 128 (40 with --dry-run)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--steer-layers", default="all",
                    help="where the direction is added: 'all' (SafeTune default), 'pick' (only the direction's layer), or 'a-b'")
    ap.add_argument("--calib-source", choices=("local", "safetune"), default="local",
                    help="local = prompts/calib/*.jsonl (aya_redteaming held-out vs benign requests); "
                         "safetune = BeaverTails vs Alpaca via safetune.runner.utils (English)")
    ap.add_argument("--calib-langs", default=None, help="languages of the local calibration set (default en,hi)")
    ap.add_argument("--calib-n", type=int, default=None, help="calibration prompts per side (default: all local / 64 safetune)")
    ap.add_argument("--n-val", type=int, default=None, help="prompts held out by SafeTune's layer sweep (default 16; 4 with --dry-run)")
    ap.add_argument("--with-ablation", action="store_true",
                    help="also run an 'ablate' condition (direction projected out): checks the direction is causal")
    ap.add_argument("--stage", choices=("all", "calibrate", "generate"), default="all")
    ap.add_argument("--recalibrate", action="store_true", help="ignore an existing direction.pt")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dry = args.dry_run
    args.model = args.model or (DRY_RUN_MODEL if dry else DEFAULT_MODEL)
    args.alphas = parse_csv(args.alpha or ("0.3" if dry else DEFAULT_ALPHAS), float)
    args.langs = parse_csv(args.langs or ("en" if dry else "en,hi,gu"))
    args.calib_langs = parse_csv(args.calib_langs or ("en" if dry else "en,hi"))
    args.limit = args.limit if args.limit is not None else (6 if dry else None)
    args.max_new_tokens = args.max_new_tokens or (40 if dry else 128)
    args.n_val = args.n_val or (4 if dry else 16)
    if dry and args.calib_n is None:
        args.calib_n = 16
    if dry and args.device == "auto":
        args.device = "cpu"
    args.out_dir = Path(args.out_dir or ROOT / "results" / args.model.split("/")[-1])
    return args


# --------------------------------------------------------------------------- calibration data
def calibration_prompts(args) -> tuple[list[str], list[str], str]:
    """(harmful, harmless, description). English prompts come first: SafeTune's layer sweep scores the
    first n_val prompts with an English refusal-substring check, so the held-out slice must be English."""
    rng = random.Random(args.seed)
    if args.calib_source == "safetune":
        from safetune.runner.utils import refusal_prompt_pairs_large
        n = args.calib_n or 64
        harmful, harmless = refusal_prompt_pairs_large(n)
        return list(harmful), list(harmless), f"safetune BeaverTails vs Alpaca, {len(harmful)} per side (en)"

    harm_rows, safe_rows = load_calibration(args.prompt_dir, args.calib_langs)

    def order(rows: list[dict]) -> list[str]:
        out: list[str] = []
        langs = ["en"] + [lg for lg in args.calib_langs if lg != "en"]
        per_lang = (args.calib_n // len(args.calib_langs)) if args.calib_n else None
        for lg in langs:
            part = [r["prompt"] for r in rows if r["lang"] == lg]
            rng.shuffle(part)
            out += part[:per_lang] if per_lang else part
        return out

    harmful, harmless = order(harm_rows), order(safe_rows)
    m = min(len(harmful), len(harmless))
    desc = (f"local prompts/calib ({'+'.join(args.calib_langs)}), {m} per side: "
            "aya_redteaming held-out harmful vs hand-written benign requests")
    return harmful[:m], harmless[:m], desc


# --------------------------------------------------------------------------- model helpers
def load(args):
    import safetune
    import torch
    from safetune.runner.utils import load_model, load_tok

    torch.manual_seed(args.seed)
    safetune.configure(device=args.device)
    tok = load_tok(args.model)
    model = load_model(args.model)
    model.eval()
    return model, tok


def residual_norms(model, tok, prompts: list[str]) -> dict[int, float]:
    """Mean L2 norm of the last-token residual stream after each decoder block. Alpha is added as
    ``alpha * unit_vector`` per layer, so this is the scale alpha should be read against."""
    import torch
    from safetune._refusal_helpers import _encode_prompts, _left_padding

    sums: dict[int, float] = {}
    for i in range(0, len(prompts), 2):  # two prompts at a time: all hidden states of a long batch do not fit 8 GB
        part = prompts[i:i + 2]
        with _left_padding(tok):
            enc = _encode_prompts(tok, part, return_tensors="pt", padding=True)
        enc = {k: v.to(next(model.parameters()).device) for k, v in enc.items()}
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        # hidden_states[0] is the embedding output; hidden_states[l + 1] is the output of block l
        for l, h in enumerate(hs[1:]):
            sums[l] = sums.get(l, 0.0) + float(h[:, -1, :].float().norm(dim=-1).sum())
        del hs
    return {l: s / len(prompts) for l, s in sums.items()}


def calibrate(args, model, tok) -> dict:
    """Extract (or reload) the refusal direction. Returns the direction bundle."""
    import torch
    from safetune._refusal_helpers import _get_decoder_layers
    from safetune.steer import RefusalDirectionConfig, extract_refusal_direction

    path = args.out_dir / "direction.pt"
    if path.exists() and not args.recalibrate:
        bundle = torch.load(path, map_location="cpu")
        if bundle.get("model") == args.model and bundle.get("pick_layer") == args.pick_layer:
            log.info("reusing %s (layer %d). Pass --recalibrate to redo.", path, bundle["layer"])
            return bundle
        log.info("direction.pt was made with other settings; recalibrating.")
    # A new direction makes earlier steered answers stale: move them aside so they are regenerated.
    # The baseline does not depend on the direction and is kept.
    gen = args.out_dir / "generations"
    stale = [f for f in gen.glob("*.jsonl") if f.stem != "baseline"] if (path.exists() and gen.exists()) else []
    if stale:
        aside = args.out_dir / f"generations_stale_{int(time.time())}"
        aside.mkdir()
        for f in stale:
            f.rename(aside / f.name)
        log.info("moved %d steered generation files made with the old direction to %s", len(stale), aside)

    harmful, harmless, desc = calibration_prompts(args)
    n_layers = len(_get_decoder_layers(model))
    log.info("calibrating on %d harmful / %d harmless prompts (%s); %d decoder layers",
             len(harmful), len(harmless), desc, n_layers)
    cfg = RefusalDirectionConfig(pick_layer=args.pick_layer, n_val=args.n_val)

    # SafeTune reports the sweep (per-layer bypass / KL, and why a layer was chosen) through logging.
    records: list[str] = []

    class _Grab(logging.Handler):
        def emit(self, record):
            msg = record.getMessage()
            if not msg.startswith("RefusalDirectionModel: installed"):  # one line per hook install: noise
                records.append(msg)

    grab = _Grab(level=logging.INFO)
    st_log = logging.getLogger("safetune")
    prev_level = st_log.level
    st_log.setLevel(logging.INFO)
    st_log.addHandler(grab)
    t0 = time.time()
    try:
        direction, layer, per_layer = extract_refusal_direction(model, tok, harmful, harmless, cfg)
    finally:
        st_log.removeHandler(grab)
        st_log.setLevel(prev_level)
    secs = time.time() - t0

    reason = next((m.split("(", 1)[1].split(";")[0] for m in records
                   if m.startswith("RefusalDirection: selected layer") and "(" in m), "unknown")
    fell_back = "fallback" in reason
    if fell_back:
        log.warning("SafeTune's sweep fell back to the middle layer (%d). Consider --pick-layer.", layer)
    norms = residual_norms(model, tok, harmless[:8])
    cos_to_picked = {int(l): float(torch.nn.functional.cosine_similarity(v.float(), direction.float(), dim=0))
                     for l, v in per_layer.items()}
    bundle = {
        "model": args.model, "pick_layer": args.pick_layer, "layer": int(layer), "n_layers": n_layers,
        "direction": direction.detach().float().cpu(),
        "per_layer": {int(l): v.detach().float().cpu() for l, v in per_layer.items()},
    }
    torch.save(bundle, path)
    write_json(args.out_dir / "calibration.json", {
        "model": args.model, "layer": int(layer), "n_layers": n_layers, "layer_reason": reason,
        "sweep_fell_back_to_middle": fell_back, "pick_layer": args.pick_layer,
        "calibration_desc": desc, "n_harmful": len(harmful), "n_harmless": len(harmless), "n_val": args.n_val,
        "seconds": round(secs, 1), "hidden_size": int(direction.shape[0]),
        "resid_norm_last_token_by_layer": {str(k): round(v, 3) for k, v in norms.items()},
        "cosine_of_layer_direction_to_picked": {str(k): round(v, 4) for k, v in sorted(cos_to_picked.items())},
        "safetune_log": records,
    })
    log.info("direction from layer %d of %d (%s) in %.0fs; residual norm there ~%.1f -> saved %s",
             layer, n_layers, reason, secs, norms.get(int(layer), float("nan")), path)
    return bundle


def steer_layer_list(spec: str, picked: int, n_layers: int):
    if spec == "all":
        return None
    if spec == "pick":
        return [picked]
    lo, hi = (int(x) for x in spec.split("-"))
    if not 0 <= lo <= hi < n_layers:
        raise SystemExit(f"--steer-layers {spec} is out of range for {n_layers} layers")
    return list(range(lo, hi + 1))


def generate_batch(model, tok, prompts: list[str], max_new_tokens: int) -> list[tuple[str, int]]:
    import torch
    from safetune._refusal_helpers import _encode_prompts, _left_padding

    with _left_padding(tok):
        # no truncation / max_length: Tiny Aya's chat template alone is ~366 tokens
        enc = _encode_prompts(tok, prompts, return_tensors="pt", padding=True)
    enc = {k: v.to(next(model.parameters()).device) for k, v in enc.items()}
    n_in = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    new = out[:, n_in:]
    pad = tok.pad_token_id if tok.pad_token_id is not None else -1
    texts = tok.batch_decode(new, skip_special_tokens=True)
    return [(t.strip(), int((row != pad).sum())) for t, row in zip(texts, new)]


def run_condition(args, model, tok, rows: list[dict], name: str, extra: dict, wrapper=None) -> None:
    path = args.out_dir / "generations" / f"{name}.jsonl"
    done = {r["uid"] for r in read_jsonl(path)}
    todo = [r for r in rows if r["uid"] not in done]
    if not todo:
        log.info("[%s] already complete (%d rows), skipping", name, len(done))
        return
    log.info("[%s] generating %d prompts (%d already done)", name, len(todo), len(done))
    todo.sort(key=lambda r: len(r["prompt"]))  # similar lengths per batch -> less padding
    t0 = time.time()
    for i in range(0, len(todo), args.batch_size):
        batch = todo[i:i + args.batch_size]
        tb = time.time()
        if wrapper is None:
            outs = generate_batch(model, tok, [r["prompt"] for r in batch], args.max_new_tokens)
        else:
            with wrapper:  # hooks are active only inside this block
                outs = generate_batch(model, tok, [r["prompt"] for r in batch], args.max_new_tokens)
        per = (time.time() - tb) / len(batch)
        for r, (text, n_tok) in zip(batch, outs):
            append_jsonl(path, {**{k: r[k] for k in ("uid", "id", "lang", "set", "category", "prompt") if k in r},
                                "answers": r.get("answers"), "condition": name, **extra,
                                "response": text, "n_new_tokens": n_tok, "gen_seconds": round(per, 2)})
        n = min(i + args.batch_size, len(todo))
        log.info("[%s] %d/%d  (%.1fs/prompt, ~%.0fs left)", name, n, len(todo), per,
                 (time.time() - t0) / n * (len(todo) - n))


def main() -> int:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(args.out_dir / "run.log", encoding="utf-8")])
    rows = load_eval_prompts(args.prompt_dir, args.langs, args.limit)
    log.info("model=%s langs=%s alphas=%s prompts=%d out=%s dry_run=%s", args.model, args.langs, args.alphas,
             len(rows), args.out_dir, args.dry_run)

    model, tok = load(args)
    bundle = calibrate(args, model, tok)
    layers = steer_layer_list(args.steer_layers, bundle["layer"], bundle["n_layers"])
    write_json(args.out_dir / "run_config.json", {
        "model": args.model, "alphas": args.alphas, "langs": args.langs, "limit": args.limit,
        "max_new_tokens": args.max_new_tokens, "batch_size": args.batch_size, "pick_layer": args.pick_layer,
        "steer_layers": args.steer_layers, "calib_source": args.calib_source, "calib_langs": args.calib_langs,
        "with_ablation": args.with_ablation, "dry_run": args.dry_run, "seed": args.seed,
        "n_prompts": len(rows), "layer": bundle["layer"], "n_layers": bundle["n_layers"],
        "device": str(next(model.parameters()).device), "dtype": str(next(model.parameters()).dtype),
    })
    if args.stage == "calibrate":
        return 0

    from safetune.steer import RefusalDirectionModel

    base = {"layer": bundle["layer"], "model": args.model}
    run_condition(args, model, tok, rows, "baseline", {**base, "alpha": 0.0, "mode": "none"})
    for alpha in args.alphas:
        wrapped = RefusalDirectionModel(model, direction=bundle["direction"], mode="steer",
                                        strength=alpha, layers=layers)
        run_condition(args, model, tok, rows, f"alpha_{alpha:.2f}", {**base, "alpha": alpha, "mode": "steer"}, wrapped)
    if args.with_ablation:
        wrapped = RefusalDirectionModel(model, direction=bundle["direction"], mode="ablate", layers=layers)
        run_condition(args, model, tok, rows, "ablate", {**base, "alpha": None, "mode": "ablate"}, wrapped)
    log.info("done. Next: python src/score.py --run-dir %s", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
