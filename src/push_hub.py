"""Publish the final run to the Hugging Face Hub (needs a token with write access: `hf auth login`).

  python src/push_hub.py --run-dir results/final            # everything
  python src/push_hub.py --run-dir results/final --dry-run  # list what would be uploaded

All repositories are public and un-gated (hackathon submission rule):

  <user>/safety-prompts-en-hi-gu                 dataset  prompt sets + calibration prompts
  <user>/tiny-aya-refusal-steering-results       dataset  ARTICLE.md, metrics, calibration, sweep, timing
  <user>/tiny-aya-refusal-steering-generations   dataset  raw + scored model answers, AuditKIT runs, logs
                                                          (content warning at the top of its card)
  <user>/tiny-aya-refusal-direction              model    the Track 2b artifact: direction vector (safetensors)
                                                          + steering config + model card with a dual-use notice
and two collections: "... (models)" with the model repo, "... (datasets)" with the three datasets.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, SETS, read_jsonl  # noqa: E402

PROMPTS_REPO = "safety-prompts-en-hi-gu"
RESULTS_REPO = "tiny-aya-refusal-steering-results"
GENS_REPO = "tiny-aya-refusal-steering-generations"
MODEL_REPO = "tiny-aya-refusal-direction"
REPO_TYPE = {PROMPTS_REPO: "dataset", RESULTS_REPO: "dataset", GENS_REPO: "dataset", MODEL_REPO: "model"}
GITHUB = "https://github.com/kunjkansara01/tiny-aya-multilingual-steering"
COLLECTIONS = {  # title -> (description, [(repo name, note)])
    "Lexsi Hackathon T2b: Tiny Aya refusal steering (models)": (
        "Track 2b artifact: the refusal direction and steering config for CohereLabs/tiny-aya-global (SafeTune).",
        [(MODEL_REPO, "Refusal direction (layer 16 of 36) + steering config; apply with SafeTune at alpha 0.4-0.8")]),
    "Lexsi Hackathon T2b: Tiny Aya refusal steering (datasets)": (
        "Track 2b data: EN/HI/GU prompt sets, baseline and steered generations, metrics and the article.",
        [(PROMPTS_REPO, "Eval prompts (harmful / benign-edgy / capability) in EN, HI, GU + calibration prompts"),
         (GENS_REPO, "Baseline and steered model replies, scored rows, AuditKIT run files. Content warning"),
         (RESULTS_REPO, "Article, metrics with p-values, calibration, alpha sweep, timing")]),
}


def model_card(user: str, cfg: dict) -> str:
    return f"""---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
language: [en, hi, gu]
tags: [safety, steering-vector, refusal-direction, activation-steering, safetune, tiny-aya, multilingual]
---

# Tiny Aya refusal direction: inference-time safety steering (EN / HI / GU)

Lexsi AI Safety Hackathon, Track 2b artifact. This repository holds **a steering vector and its config, not model
weights**: one unit vector in the residual stream of [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global)
(hidden size {cfg['hidden_size']}), extracted with [SafeTune](https://github.com/Lexsi-Labs/SafeTune) from layer
{cfg['layer']} of {cfg['n_layers']}. Adding `alpha × vector` to every decoder block's output at inference time makes
the model refuse harmful requests more often, in English, Hindi and Gujarati, with no retraining.

> **Dual-use notice.** The same vector can be used the other way: projecting it out of the residual stream
> (`mode="ablate"`) removed refusals on our held-out harmful prompts (3/8 → 0/8). We disclose this rather than gate
> the artifact: the vector is reproducible in minutes from the public calibration prompts and SafeTune, the base
> model is small and openly downloadable, and the method is published (Arditi et al., 2024). Use it to harden,
> evaluate or study the model.

## Files

- `refusal_direction.safetensors`: `direction` (float32, [{cfg['hidden_size']}], unit norm, layer {cfg['layer']}) and
  `per_layer` ([{cfg['n_layers']}, {cfg['hidden_size']}], the candidate direction after every decoder block, as returned by SafeTune)
- `steering_config.json`: base model, layer, strengths, calibration set, residual norms, measured results

## Use

```python
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from safetune.runner.utils import load_model, load_tok
from safetune.steer import RefusalDirectionModel

base = "CohereLabs/tiny-aya-global"            # gated by Cohere: accept its licence first
tok, model = load_tok(base), load_model(base, dtype=torch.bfloat16, device="cuda")
vec = load_file(hf_hub_download("{user}/{MODEL_REPO}", "refusal_direction.safetensors"))["direction"]

steered = RefusalDirectionModel(model, direction=vec, mode="steer", strength={cfg['alpha']})   # {cfg['alpha_no_cost']}: no measured over-refusal
text = tok.apply_chat_template([{{"role": "user", "content": "..."}}], tokenize=False, add_generation_prompt=True)
enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
with steered:                                   # hooks are active only inside the block
    out = model.generate(**enc, max_new_tokens=160, do_sample=False)
print(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Measured effect (225 prompts, greedy decoding, bf16)

| | Harmful refused ↑ | Benign-edgy refused ↓ | Capability correct ↑ |
|---|---|---|---|
{cfg['results_table']}

Per language at alpha {cfg['alpha']}: {cfg['per_language']}. Gujarati is zero-shot: the vector was fitted on English and
Hindi prompts only. Steered generation takes about 1.1× the baseline time per token (36 forward hooks).

## Limitations

Refusal is judged by an EN/HI/GU phrase match (a lower bound); five harm categories (violence, self-harm, hate
speech, harassment, discrimination); 40 harmful prompts per language; one model, one run. At alpha {cfg['alpha']} three
benign English prompts were refused and one Gujarati reply became worse. Not a defence against jailbreak prompts
or against anyone who can edit activations. Full write-up: [`ARTICLE.md`](https://huggingface.co/datasets/{user}/{RESULTS_REPO}/blob/main/ARTICLE.md);
code: {GITHUB}.

Licence: the vector is derived from activations of Tiny Aya (CC-BY-NC-4.0) and is released under the same terms.
"""


def stage_model(run: Path, folder: Path, user: str) -> None:
    """The Track 2b artifact: the direction as safetensors plus a steering config and a model card."""
    import torch
    from safetensors.torch import save_file

    folder.mkdir()
    bundle = torch.load(run / "direction.pt", map_location="cpu")
    per_layer = torch.stack([bundle["per_layer"][i].float() for i in sorted(bundle["per_layer"])])
    save_file({"direction": bundle["direction"].float().contiguous(), "per_layer": per_layer.contiguous()},
              str(folder / "refusal_direction.safetensors"))
    m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    cal, cond = m["calibration"], m["conditions"]

    def pct(c, sl, key, src=None):
        r = (src or cond[c])[sl][key]
        return f"{100 * r['value']:.1f}% ({r['k']}/{r['n']})"

    def row(label, src):
        return (f"| {label} | {pct(None, 'all', 'harmful_refusal_rate', src)} | {pct(None, 'all', 'over_refusal_rate', src)} | "
                f"{pct(None, 'all', 'capability_accuracy', src)} |")

    rows = [("Baseline", 0.0, cond["baseline"])]
    rows += [(f"alpha {d['alpha']}", d["alpha"], d["metrics"]) for d in (m.get("dose") or {}).values()]
    rows += [(f"alpha {m['meta']['alpha']} (default)", m["meta"]["alpha"], cond["steered"])]
    rows.sort(key=lambda r: r[1])
    others = sorted(d["alpha"] for d in (m.get("dose") or {}).values())
    names = {"en": "English", "hi": "Hindi", "gu": "Gujarati"}
    cfg = {
        "base_model": m["meta"]["model"], "method": "refusal direction (Arditi et al. 2024), SafeTune extract_refusal_direction",
        "apply_with": 'safetune.steer.RefusalDirectionModel(model, direction, mode="steer", strength=alpha)',
        "steer_layers": "all decoder blocks (SafeTune default)", "layer": cal["layer"], "n_layers": cal["n_layers"],
        "layer_reason": cal["layer_reason"], "hidden_size": cal["hidden_size"], "alpha": m["meta"]["alpha"],
        "alpha_no_cost": others[0] if others else None, "alpha_rule": (m.get("sweep") or {}).get("rule"),
        "calibration": cal["calibration_desc"], "n_val": cal["n_val"], "dtype_used": (m.get("preflight") or {}).get("quant"),
        "resid_norm_last_token_by_layer": cal["resid_norm_last_token_by_layer"],
        "results": {label: {k: src["all"][k]["value"] for k in ("harmful_refusal_rate", "over_refusal_rate", "capability_accuracy")}
                    for label, _, src in rows},
        "safetune_version": "0.1.18", "prompts_dataset": f"{user}/{PROMPTS_REPO}", "code": GITHUB,
    }
    (folder / "steering_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    card = {**cfg, "results_table": "\n".join(row(label, src) for label, _, src in rows),
            "per_language": ", ".join(f"{names[lg]} {pct('baseline', lg, 'harmful_refusal_rate')} → "
                                      f"{pct('steered', lg, 'harmful_refusal_rate')}" for lg in m["slices"] if lg != "all")}
    (folder / "README.md").write_text(model_card(user, card), encoding="utf-8")


def prompts_card(prompt_dir: Path) -> str:
    counts = {(lg, s): len(read_jsonl(prompt_dir / lg / f"{s}.jsonl")) for lg in ("en", "hi", "gu") for s in SETS}
    cfg = "\n".join(
        f"- config_name: {s}\n  data_files:\n" + "\n".join(f"  - split: {lg}\n    path: {lg}/{s}.jsonl" for lg in ("en", "hi", "gu"))
        for s in SETS)
    cfg += ("\n- config_name: calibration\n  data_files:\n  - split: harmful\n    path: calib/harmful.jsonl\n"
            "  - split: harmless\n    path: calib/harmless.jsonl")
    rows = "\n".join(f"| `{s}` | {counts[('en', s)]} | {counts[('hi', s)]} | {counts[('gu', s)]} |" for s in SETS)
    return f"""---
license: apache-2.0
language: [en, hi, gu]
task_categories: [text-generation]
tags: [safety, red-teaming, refusal, multilingual, gujarati, hindi]
pretty_name: Safety steering prompts (EN / HI / GU)
configs:
{cfg}
---

# Safety steering prompts: English, Hindi, Gujarati

Evaluation and calibration prompts for measuring refusal behaviour of a multilingual model before and after
inference-time refusal-direction steering (Lexsi AI Safety Hackathon, Track 2b).

**Content warning:** the `harmful` config contains harmful requests (violence, self-harm, hate speech,
harassment, discrimination). They are here only to measure whether a model refuses them.

| Config | EN | HI | GU |
|---|---|---|---|
{rows}

- `harmful`: EN and HI are sampled from [`CohereLabs/aya_redteaming`](https://huggingface.co/datasets/CohereLabs/aya_redteaming)
  (Apache-2.0, human-written; global-harm rows only, 8 prompts in each of 5 categories; sexual-content, graphic and
  profanity categories and anything mentioning minors excluded). aya_redteaming has no Gujarati, so **GU is a
  translation of the English 40** (same `id`s: EN and GU are a parallel pair; HI is native and not parallel).
- `benign_edgy`: XSTest-style requests that sound alarming but are harmless ("how do I kill a Python process").
  Written in English, translated to HI and GU. Measures over-refusal.
- `capability`: short factual questions with a list of accepted `answers`. Measures utility.
- `calibration`: 32 EN + 32 HI harmful prompts from aya_redteaming (disjoint from the eval prompts) and 32 + 32
  benign requests of the same shape. **No Gujarati**, so Gujarati tests zero-shot transfer of the direction.

Fields: `id`, `lang`, `set`, `category`, `prompt`, `source`, and `answers` (capability only).
Sampling is reproducible with `src/build_prompts.py` (seed 20261114) in the project repository.
"""


def results_card(metrics: dict) -> str:
    meta = metrics.get("meta", {})
    return f"""---
license: cc-by-4.0
language: [en, hi, gu]
tags: [safety, steering, refusal-direction, tiny-aya, safetune, auditkit]
pretty_name: Tiny Aya refusal-direction steering results
---

# Refusal-direction steering on `{meta.get('model')}`: results

Lexsi AI Safety Hackathon, Track 2b. SafeTune extracts and adds the refusal direction at inference time
(alpha = {meta.get('alpha')}, direction from layer {meta.get('layer')} of {meta.get('n_layers')}); AuditKIT scores baseline vs steered.

- [`ARTICLE.md`](ARTICLE.md): the write-up
- [`metrics.md`](metrics.md) / `metrics.json`: full before/after table, deltas, p-values, throughput
- `calibration.json`, `sweep.json`, `state.json`, `preflight.json`, `timing/`, `auditkit/compare_*.txt`

Related repositories under the same account: prompts `{PROMPTS_REPO}`; raw baseline and steered model replies
`{GENS_REPO}` (content warning: replies to harmful prompts); the steering vector `{MODEL_REPO}` (model repo).
Code: {GITHUB}
"""


def gens_card() -> str:
    return f"""---
license: cc-by-nc-4.0
language: [en, hi, gu]
tags: [safety, red-teaming, refusal, steering, tiny-aya, not-for-all-audiences]
pretty_name: Tiny Aya refusal steering, raw generations
configs:
- config_name: baseline
  data_files: generations/baseline.jsonl
- config_name: steered_alpha_0.8
  data_files: generations/steered.jsonl
- config_name: steered_alpha_0.4
  data_files: generations/steered_alpha_0.4.jsonl
---

> ## Content warning
>
> **This dataset contains model responses to harmful prompts; released for safety research.** The prompts ask for
> content about violence, self-harm, hate speech, harassment and discrimination, and some of the unsteered model's
> replies comply. Do not use the replies as advice or as training targets for helpfulness. A few rows contain
> self-harm themes; if that is hard for you to read, skip the `Self-Harm` category.

# Tiny Aya refusal steering: raw generations

Every reply of `CohereLabs/tiny-aya-global` to 225 prompts (English, Hindi, Gujarati; harmful / benign-but-edgy /
capability) without steering and with SafeTune refusal-direction steering, from the Lexsi AI Safety Hackathon
Track 2b run. Released so the refusal labels and the before/after claims can be audited row by row.

- `generations/baseline.jsonl`, `generations/steered.jsonl` (alpha 0.8), `generations/steered_alpha_0.4.jsonl`: raw replies
- `scored/*.jsonl`: the same rows with the classifier's labels (`refusal`, matched pattern, `degenerate`, `correct`, script)
- `sweep/alpha_*.jsonl`: the 60-prompt alpha sweep (EN + HI); `sweep_transcripts.txt`
- `auditkit/run_*.json`: AuditKIT RunResults (baseline and steered; all languages and per language)
- `direction.pt`: the direction bundle the run used (also published as safetensors in the model repo `{MODEL_REPO}`)
- `run.log`, `preflight.json`

Fields: `uid`, `id`, `lang`, `set`, `category`, `prompt`, `answers`, `condition`, `alpha`, `mode`, `response`,
`n_new_tokens`, `quant`, `max_new_tokens`. Prompts: `{PROMPTS_REPO}`; metrics and article: `{RESULTS_REPO}`;
code: {GITHUB}. Replies are derived from a CC-BY-NC-4.0 model and are released under the same terms.
"""


def stage(run: Path, tmp: Path, user: str) -> dict[str, Path]:
    prompts, results, gens = tmp / "prompts", tmp / "results", tmp / "generations"
    # 1 prompts (public)
    shutil.copytree(ROOT / "prompts", prompts, ignore=shutil.ignore_patterns("REVIEW.md"))  # working notes, not data
    (prompts / "README.md").write_text(prompts_card(ROOT / "prompts"), encoding="utf-8")
    # 2 results (public): nothing with model answers
    results.mkdir()
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    for name in ("metrics.md", "metrics.json", "calibration.json", "sweep.json", "state.json",
                 "throughput_bench.json", "throughput_bench_run1_raw.txt"):
        if (run / name).exists():
            shutil.copy(run / name, results / name)
    pre = run / "preflight.json"
    if pre.exists():
        d = json.loads(pre.read_text(encoding="utf-8"))
        d.pop("prompt", None), d.pop("response", None)
        (results / "preflight.json").write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
    if (run / "timing").exists():
        shutil.copytree(run / "timing", results / "timing")
    (results / "auditkit").mkdir()
    for f in (run / "auditkit").glob("compare_*.txt"):
        shutil.copy(f, results / "auditkit" / f.name)
    shutil.copy(run / "metrics.md", results / "METRICS.md")  # the name ARTICLE.md links to
    for name in ("ARTICLE.md", "FINDINGS.md"):
        if (ROOT / name).exists():
            shutil.copy(ROOT / name, results / name)
    (results / "README.md").write_text(results_card(metrics), encoding="utf-8")
    # 3 raw generations (public, content warning at the top of the card)
    gens.mkdir()
    for name in ("generations", "scored", "sweep"):
        if (run / name).exists():
            shutil.copytree(run / name, gens / name)
    (gens / "auditkit").mkdir()
    for f in (run / "auditkit").glob("run_*.json"):
        shutil.copy(f, gens / "auditkit" / f.name)
    for name in ("direction.pt", "run.log", "preflight.json", "sweep_transcripts.txt", "calibration_sweep_attempt.json"):
        if (run / name).exists():
            shutil.copy(run / name, gens / name)
    (gens / "README.md").write_text(gens_card(), encoding="utf-8")
    # 4 the Track 2b model artifact (public): direction vector + steering config
    stage_model(run, tmp / "model", user)
    return {PROMPTS_REPO: prompts, RESULTS_REPO: results, GENS_REPO: gens, MODEL_REPO: tmp / "model"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=str(ROOT / "results" / "final"))
    ap.add_argument("--user", default=None, help="Hub namespace (default: the logged-in user)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    from huggingface_hub import HfApi

    api = HfApi()
    who = api.whoami()
    user = args.user or who["name"]
    role = ((who.get("auth") or {}).get("accessToken") or {}).get("role")
    links = {}
    with tempfile.TemporaryDirectory() as t:
        staged = stage(Path(args.run_dir), Path(t), user)
        for name, folder in staged.items():
            repo, kind = f"{user}/{name}", REPO_TYPE[name]
            files = sorted(str(p.relative_to(folder)).replace("\\", "/") for p in folder.rglob("*") if p.is_file())
            print(f"\n{repo}  ({kind}, public, {len(files)} files)")
            for f in files:
                print("   ", f)
            if args.dry_run:
                continue
            if role == "read":
                raise SystemExit("the stored Hugging Face token is read-only. Create a write token at "
                                 "https://huggingface.co/settings/tokens and run `hf auth login`, then re-run.")
            api.create_repo(repo, repo_type=kind, private=False, exist_ok=True)
            api.upload_folder(repo_id=repo, repo_type=kind, folder_path=str(folder),
                              commit_message="Lexsi hackathon Track 2b: final run")
            # the submission rules require every repo public and un-gated: flip one created private earlier
            # (after the upload, so the content warning is already at the top of its card)
            if api.repo_info(repo, repo_type=kind).private:
                api.update_repo_settings(repo, repo_type=kind, private=False)
                print(f"    visibility of {repo}: private -> public")
            links[name] = f"https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{repo}"
        if not args.dry_run:
            for title, (desc, items) in COLLECTIONS.items():
                col = api.create_collection(title=title, namespace=user, description=desc, private=False, exists_ok=True)
                for name, note in items:
                    api.add_collection_item(col.slug, item_id=f"{user}/{name}", item_type=REPO_TYPE[name],
                                            note=note, exists_ok=True)
                links["collection_models" if "(models)" in title else "collection_datasets"] =                     f"https://huggingface.co/collections/{col.slug}"
    if links:
        out = Path(args.run_dir) / "hub_links.json"
        out.write_text(json.dumps(links, indent=2) + "\n", encoding="utf-8")
        print("\npushed (all public):")
        for v in links.values():
            print(f"  {v}")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
