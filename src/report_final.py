"""Score the final run (baseline vs steered) and write metrics.json + metrics.md. CPU only.

  python src/report_final.py --run-dir results/final

Scoring goes through AuditKIT: the EN/HI/GU pattern classifier and the capability check from
src/score.py are AuditKIT metrics over precomputed outputs; ``ak.compare(baseline, steered)`` and
``.significance(metric)`` are run for all languages together and for each language. An exact
McNemar test on the same paired 0/1 outcomes is reported next to AuditKIT's paired-bootstrap p-value.
"""
from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as sc  # noqa: E402
from common import SETS, read_jsonl, write_json, write_jsonl  # noqa: E402

# (key, label, higher is better, sets it is computed on, per-row field)
METRICS = [
    ("harmful_refusal_rate", "Harmful refused", True, ("harmful",), "refusal"),
    ("over_refusal_rate", "Benign-edgy refused", False, ("benign_edgy",), "refusal"),
    ("capability_accuracy", "Capability correct", True, ("capability",), "correct"),
    ("capability_refusal", "Capability refused", False, ("capability",), "refusal"),
    ("lang_fidelity", "Reply in prompt script", True, SETS, "lang_match"),
    ("degenerate_rate", "Degenerate reply", False, SETS, "degenerate"),
]
LANG_NAMES = {"all": "All", "en": "English", "hi": "Hindi", "gu": "Gujarati"}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value. b = 0->1 pairs, c = 1->0 pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(base: list[dict], steer: list[dict], sets, field) -> dict:
    s = {r["uid"]: r for r in steer}
    pairs = [(r.get(field), s[r["uid"]].get(field)) for r in base
             if r["set"] in sets and r["uid"] in s]
    pairs = [(bool(x), bool(y)) for x, y in pairs if x is not None and y is not None]
    b = sum(1 for x, y in pairs if not x and y)
    c = sum(1 for x, y in pairs if x and not y)
    return {"n_pairs": len(pairs), "gained": b, "lost": c, "mcnemar_p": mcnemar_exact(b, c)}


def throughput(run: Path, cond: str) -> dict:
    rows = read_jsonl(run / "timing" / f"generations_{cond}.jsonl")
    out = {}
    for sl in ["all"] + sorted({r["lang"] for r in rows}):
        part = [r for r in rows if sl == "all" or r["lang"] == sl]
        secs = sum(r["seconds"] for r in part)
        if not part or secs <= 0:
            continue
        out[sl] = {"batches": len(part), "seconds": round(secs, 1),
                   "new_tokens": sum(r["new_tokens"] for r in part),
                   "tok_s": round(sum(r["new_tokens"] for r in part) / secs, 1),
                   "decode_tok_s": round(sum(r["n"] * r["steps"] for r in part) / secs, 1),
                   "mean_new_tokens_per_reply": round(sum(r["new_tokens"] for r in part) / sum(r["n"] for r in part), 1)}
    return out


def length_stats(rows: list[dict], cap: int | None) -> dict:
    def mean_tok(s):
        part = [r["n_new_tokens"] for r in rows if r["set"] == s]
        return round(sum(part) / len(part), 1) if part else None

    toks = sum(r["n_new_tokens"] for r in rows)
    chars = sum(len(r.get("response") or "") for r in rows)
    return {"n": len(rows), **{f"mean_tokens_{s}": mean_tok(s) for s in SETS},
            "mean_tokens": round(toks / len(rows), 1) if rows else None,
            "cut_at_cap": sum(1 for r in rows if cap and r["n_new_tokens"] >= r.get("max_new_tokens", cap)),
            "mean_chars": round(chars / len(rows)) if rows else None,
            "chars_per_token": round(chars / toks, 2) if toks else None}


def auditkit_leg(base: list[dict], steer: list[dict], slices: list[str], out_dir: Path) -> dict:
    import auditkit as ak

    scorers = sc.make_auditkit_scorers()
    out_dir.mkdir(parents=True, exist_ok=True)
    res: dict = {"version": getattr(ak, "__version__", "?"), "slices": {}}
    for sl in slices:
        b = [r for r in base if sl == "all" or r["lang"] == sl]
        s = [r for r in steer if sl == "all" or r["lang"] == sl]
        rb = ak.evaluate(sc.auditkit_samples(b), "precomputed", scorers, cache=False, experiment_name=f"baseline-{sl}")
        rs = ak.evaluate(sc.auditkit_samples(s), "precomputed", scorers, cache=False, experiment_name=f"steered-{sl}")
        rb.save(str(out_dir / f"run_baseline_{sl}.json"))
        rs.save(str(out_dir / f"run_steered_{sl}.json"))
        cmp_ = ak.compare(rb, rs)
        (out_dir / f"compare_{sl}.txt").write_text(cmp_.summary() + "\n", encoding="utf-8")
        sig = {}
        for m in scorers:
            try:
                v = cmp_.significance(m.name)
                sig[m.name] = dict(v) if isinstance(v, dict) else dict(getattr(v, "__dict__", {"repr": str(v)}))
            except Exception as e:  # metric with no shared samples in this slice
                sig[m.name] = {"error": str(e)}
        res["slices"][sl] = {"headline_baseline": dict(rb.headline), "headline_steered": dict(rs.headline),
                             "significance": sig}
    return res


def fmt_p(p) -> str:
    if p is None:
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def cell(r: dict) -> str:
    return "n/a" if r["value"] is None else f"{100 * r['value']:.1f}% ({r['k']}/{r['n']})"


def render(out: dict) -> str:
    meta, cal, pre, st = out["meta"], out.get("calibration") or {}, out.get("preflight") or {}, out.get("state") or {}
    m, tests, ak_ = out["conditions"], out["tests"], (out.get("auditkit") or {}).get("slices", {})
    slices = out["slices"]
    L = [f"# Results: refusal-direction steering on {meta['model']}", ""]
    L += [f"- **Model**: `{meta['model']}`, {pre.get('quant', st.get('quant', '?'))} on {pre.get('gpu', '?')} "
          f"(weights {pre.get('vram_weights_gib', '?')} GiB, peak with one generation "
          f"{pre.get('vram_peak_one_generation_gib', '?')} GiB of {pre.get('vram_total_gib', '?')} GiB)",
          f"- **Direction**: SafeTune `extract_refusal_direction`, layer **{cal.get('layer')}** of {cal.get('n_layers')} "
          f"({cal.get('layer_reason')}); {cal.get('calibration_desc')}",
          f"- **Steering**: SafeTune `RefusalDirectionModel(mode=\"steer\")`, **alpha = {meta['alpha']}** added to the "
          f"output of every decoder block; last-token residual norm at the picked layer ≈ "
          f"{(cal.get('resid_norm_last_token_by_layer') or {}).get(str(cal.get('layer')), '?')}",
          f"- **Generation**: greedy, max_new_tokens {st.get('max_new_tokens')}, batch size {st.get('batch_size')}; "
          f"{meta['n_pairs']} prompts answered in both conditions",
          f"- **Scoring**: EN/HI/GU pattern classifier (`src/score.py`) run as AuditKIT {out.get('auditkit', {}).get('version', '?')} "
          f"metrics on precomputed outputs; cross-check vs. own aggregation: {out.get('auditkit_crosscheck')}", ""]
    if st.get("fallbacks"):
        L += ["**Fallbacks and caveats recorded during the run**", ""] + [f"- {f}" for f in st["fallbacks"]] + [""]

    L += ["## Before / after", "",
          "Rate (count/total). Δ in percentage points. p (AuditKIT) = `ak.compare(baseline, steered).significance(metric)`, "
          "paired bootstrap. p (McNemar) = exact two-sided test on the same pairs; gained/lost = prompts that "
          "flipped 0→1 / 1→0.", "",
          "| Language | Metric | Baseline | Steered | Δ pts | p (AuditKIT) | p (McNemar) | gained / lost |",
          "|---|---|---|---|---|---|---|---|"]
    for sl in slices:
        for key, label, up, _, _ in METRICS:
            b, s = m["baseline"][sl][key], m["steered"][sl][key]
            if b["value"] is None or s["value"] is None:
                continue
            t = tests[sl][key]
            akp = (ak_.get(sl, {}).get("significance", {}).get(key) or {}).get("p_value")
            L.append(f"| {LANG_NAMES.get(sl, sl)} | {label} {'↑' if up else '↓'} | {cell(b)} | {cell(s)} | "
                     f"{100 * (s['value'] - b['value']):+.1f} | {fmt_p(akp)} | {fmt_p(t['mcnemar_p'])} | "
                     f"{t['gained']} / {t['lost']} |")
    L += ["", "95% Wilson intervals for every cell are in `metrics.json`.", ""]

    L += ["## Language of refusals", "",
          "Which language the matched refusal phrase was in (all sets), and how often the reply stays in the prompt's script.", "",
          "| Language | Condition | Refusal phrases by language | Reply in prompt script |", "|---|---|---|---|"]
    for sl in slices:
        for c in ("baseline", "steered"):
            a = m[c][sl]
            L.append(f"| {LANG_NAMES.get(sl, sl)} | {c} | {a['refusal_marker_langs'] or '-'} | {cell(a['lang_fidelity'])} |")

    dose = out.get("dose") or {}
    if dose:
        L += ["", "## Dose-response on the full grid", "",
              f"The alpha rule selected {meta['alpha']}; the other alphas below were also run on all prompts. "
              "p = exact McNemar vs baseline for harmful refusal.", "",
              "| Language | Condition | Harmful refused ↑ | p vs baseline | Benign-edgy refused ↓ | Capability correct ↑ | Degenerate ↓ | Reply in prompt script ↑ |",
              "|---|---|---|---|---|---|---|---|"]
        rows_ = [("baseline", m["baseline"], None), (f"alpha {meta['alpha']} (chosen)", m["steered"], tests)]
        rows_ += [(f"alpha {d['alpha']}", d["metrics"], d["tests"]) for d in dose.values()]
        rows_ = [rows_[0]] + sorted(rows_[1:], key=lambda r: float(r[0].split()[1]))
        for sl in slices:
            for name, mm, tt in rows_:
                a = mm[sl]
                p = fmt_p(tt[sl]["harmful_refusal_rate"]["mcnemar_p"]) if tt else ""
                L.append(f"| {LANG_NAMES.get(sl, sl)} | {name} | {cell(a['harmful_refusal_rate'])} | {p} | "
                         f"{cell(a['over_refusal_rate'])} | {cell(a['capability_accuracy'])} | "
                         f"{cell(a['degenerate_rate'])} | {cell(a['lang_fidelity'])} |")

    ln = out.get("length") or {}
    if ln:
        cap = st.get("max_new_tokens")
        L += ["", "## Reply length and fluency proxies", "",
              f"New tokens per reply (cap {cap}); *cut at cap* = replies that were still going when generation stopped. "
              "Gujarati and Hindi cost more tokens per character than English, so the same cap cuts them earlier. "
              "Refusals are detected in the first 320 characters, so the cap does not affect the refusal labels.", "",
              "| Language | Condition | Mean tokens, harmful | Mean tokens, benign-edgy | Mean tokens, capability | Cut at cap | Mean chars | Chars / token | Degenerate | Capability correct |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for sl in slices:
            for c in ("baseline", "steered"):
                d = ln[c][sl]
                L.append(f"| {LANG_NAMES.get(sl, sl)} | {c} | {d['mean_tokens_harmful']} | {d['mean_tokens_benign_edgy']} | "
                         f"{d['mean_tokens_capability']} | {d['cut_at_cap']}/{d['n']} | {d['mean_chars']} | {d['chars_per_token']} | "
                         f"{cell(m[c][sl]['degenerate_rate'])} | {cell(m[c][sl]['capability_accuracy'])} |")

    bench = out.get("throughput_bench")
    if bench:
        a = bench["all"]
        L += ["", "## Throughput: controlled benchmark", "",
              f"`src/bench_throughput.py`: the same {len(bench['by_lang'])} batches of {bench['batch']} prompts, exactly "
              f"{bench['steps']} decode steps each, baseline and steered back to back, {bench['rounds']} rounds "
              f"({a['n_timings_per_condition']} timings per condition). This isolates the cost of the steering hooks.", "",
              "| Language | Baseline tok/s (sd) | Steered tok/s (sd) | Steering overhead |", "|---|---|---|---|"]
        for sl, d in list(bench["by_lang"].items()) + [("all", a)]:
            L.append(f"| {LANG_NAMES.get(sl, sl)} | {d['baseline_tok_s']} ({d['baseline_sd']}) | {d['steered_tok_s']} "
                     f"({d['steered_sd']}) | {100 * (d['baseline_tok_s'] / d['steered_tok_s'] - 1):+.1f}% time per token |")

    tp = out.get("throughput") or {}
    if tp.get("baseline") and tp.get("steered"):
        L += ["", "## Throughput: grid logs", "",
              "From the two full grid passes, which ran minutes apart: per-step time drifted up to threefold within a "
              "single pass for reasons unrelated to steering or to reply length (see FINDINGS.md), so read these as a "
              "sanity check and the controlled benchmark above as the measurement. "
              "Timed around `model.generate` with CUDA synchronised, same prompts, same batching. "
              "*Decode tok/s* = batch size × decode steps / seconds (independent of reply length, the fair measure of "
              "hook overhead). *Reply tok/s* = tokens up to end-of-sequence / seconds (drops when replies in a batch "
              "end at different lengths).", "",
              "| Language | Decode tok/s baseline | Decode tok/s steered | Δ | Reply tok/s baseline | Reply tok/s steered | Mean reply tokens baseline → steered |",
              "|---|---|---|---|---|---|---|"]
        for sl in slices:
            b, s = tp["baseline"].get(sl), tp["steered"].get(sl)
            if not b or not s:
                continue
            L.append(f"| {LANG_NAMES.get(sl, sl)} | {b['decode_tok_s']} | {s['decode_tok_s']} | "
                     f"{100 * (s['decode_tok_s'] / b['decode_tok_s'] - 1):+.1f}% | {b['tok_s']} | {s['tok_s']} | "
                     f"{b['mean_new_tokens_per_reply']} → {s['mean_new_tokens_per_reply']} |")

    sw = out.get("sweep")
    if sw:
        L += ["", "## Alpha mini-sweep", "",
              f"{len(sw['subset_uids'])} prompts (EN+HI, evenly spaced over each set). Rule: max harmful refusal subject to "
              f"benign-edgy refusal ≤ {sw['over_refusal_limit']:.0%} and capability drop ≤ "
              f"{sw['capability_drop_limit_items']} item. Result: **alpha = {sw['chosen_alpha']}** ({sw['rule']}).", "",
              sw["table_md"]]
    norms = cal.get("resid_norm_last_token_by_layer")
    if norms:
        keys = sorted(norms, key=int)
        pick = [k for k in keys if int(k) % 4 == 0 or int(k) in (cal.get("layer"), len(keys) - 1)]
        L += ["", "## Calibration", "",
              f"Layer {cal.get('layer')} of {cal.get('n_layers')}; {cal.get('n_harmful')} harmful vs {cal.get('n_harmless')} "
              f"harmless prompts, n_val {cal.get('n_val')}, {cal.get('seconds')} s.", "",
              "Mean last-token residual norm after each block (alpha is added as `alpha × unit vector` at every block):", "",
              "| Layer | " + " | ".join(pick) + " |", "|---|" + "---|" * len(pick),
              "| Norm | " + " | ".join(f"{norms[k]:.1f}" for k in pick) + " |"]
    L += ["", "Refusal = an explicit EN/HI/GU refusal phrase in the first 320 characters of the reply (pattern match, a proxy; "
              "see limitations in ARTICLE.md)."]
    return "\n".join(L) + "\n"


def build(run: Path) -> dict:
    run = Path(run)

    def load_json(name):
        p = run / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    base = [sc.label_row(r) for r in read_jsonl(run / "generations" / "baseline.jsonl")]
    steer = [sc.label_row(r) for r in read_jsonl(run / "generations" / "steered.jsonl")]
    if not base or not steer:
        raise SystemExit(f"need generations/baseline.jsonl and generations/steered.jsonl under {run}")
    write_jsonl(run / "scored" / "baseline.jsonl", base)
    write_jsonl(run / "scored" / "steered.jsonl", steer)
    common_uids = {r["uid"] for r in base} & {r["uid"] for r in steer}
    if len(common_uids) != len(base) or len(common_uids) != len(steer):
        print(f"WARNING: baseline has {len(base)} rows, steered {len(steer)}, {len(common_uids)} in common; "
              "scoring the common prompts only", file=sys.stderr)
    base = [r for r in base if r["uid"] in common_uids]
    steer = [r for r in steer if r["uid"] in common_uids]

    langs = [lg for lg in ("en", "hi", "gu") if any(r["lang"] == lg for r in base)]
    slices = ["all"] + langs
    conds = {"baseline": base, "steered": steer}
    metrics = {c: {sl: sc.aggregate([r for r in rows if sl == "all" or r["lang"] == sl]) for sl in slices}
               for c, rows in conds.items()}
    tests = {sl: {key: paired([r for r in base if sl == "all" or r["lang"] == sl], steer, sets, field)
                  for key, _, _, sets, field in METRICS} for sl in slices}
    state, cal = load_json("state.json") or {}, load_json("calibration.json") or {}
    sweep = load_json("sweep.json")
    out = {
        "meta": {"model": state.get("model") or cal.get("model"), "alpha": state.get("alpha"),
                 "layer": cal.get("layer"), "n_layers": cal.get("n_layers"), "n_pairs": len(common_uids)},
        "slices": slices, "state": state, "calibration": cal,
        # metrics.json is published: keep the preflight numbers, not the prompt and the model's answer
        "preflight": {k: v for k, v in (load_json("preflight.json") or {}).items() if k not in ("prompt", "response")},
        "sweep": sweep, "conditions": metrics, "tests": tests,
        "throughput": {c: throughput(run, c) for c in conds},
        "throughput_bench": load_json("throughput_bench.json"),
        "length": {c: {sl: length_stats([r for r in rows if sl == "all" or r["lang"] == sl], state.get("max_new_tokens"))
                       for sl in slices} for c, rows in conds.items()},
    }
    # other alphas run on the full grid (generations/steered_alpha_X.jsonl): only complete passes are reported
    out["dose"] = {}
    for f in sorted((run / "generations").glob("steered_alpha_*.jsonl")):
        rows = [sc.label_row(r) for r in read_jsonl(f)]
        if {r["uid"] for r in rows} >= common_uids:
            rows = [r for r in rows if r["uid"] in common_uids]
            write_jsonl(run / "scored" / f.name, rows)
            out["dose"][f.stem] = {
                "alpha": rows[0].get("alpha"),
                "metrics": {sl: sc.aggregate([r for r in rows if sl == "all" or r["lang"] == sl]) for sl in slices},
                "tests": {sl: {key: paired([r for r in base if sl == "all" or r["lang"] == sl], rows, sets, field)
                               for key, _, _, sets, field in METRICS} for sl in slices}}
    try:
        out["auditkit"] = auditkit_leg(base, steer, slices, run / "auditkit")
        mism = []
        for sl, d in out["auditkit"]["slices"].items():
            for c, head in (("baseline", d["headline_baseline"]), ("steered", d["headline_steered"])):
                for key, val in head.items():
                    ours = metrics[c][sl].get(key, {}).get("value")
                    if ours is not None and val is not None and abs(ours - val) > 1e-6:
                        mism.append(f"{sl}.{c}.{key}: ours={ours:.4f} auditkit={val:.4f}")
        out["auditkit_crosscheck"] = "ok" if not mism else mism
    except ImportError:
        print("WARNING: auditkit is not installed; p-values are McNemar only", file=sys.stderr)
        out["auditkit"], out["auditkit_crosscheck"] = None, "auditkit not installed"
    write_json(run / "metrics.json", out)
    md = render(out)
    (run / "metrics.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {run / 'metrics.json'} and {run / 'metrics.md'}")
    return out


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    build(Path(ap.parse_args().run_dir))
