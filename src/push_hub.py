"""Publish the final run to the Hugging Face Hub (needs a token with write access: `hf auth login`).

  python src/push_hub.py --run-dir results/final            # all three repos
  python src/push_hub.py --run-dir results/final --dry-run  # list what would be uploaded

  <user>/safety-prompts-en-hi-gu                 dataset, PUBLIC   prompt sets + calibration prompts
  <user>/tiny-aya-refusal-steering-results       dataset, PUBLIC   ARTICLE.md, metrics, calibration, sweep, timing
  <user>/tiny-aya-refusal-steering-generations   dataset, PRIVATE  raw + scored model answers, AuditKIT runs,
                                                                   direction.pt, logs

Model answers to harmful prompts and the refusal direction itself (ablating it removes refusals) stay private.
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

Prompts: the public dataset `{PROMPTS_REPO}` under the same account. Raw model answers (they include replies to
harmful prompts) and the direction vector are in a private repository; available to the judges on request.
"""


def stage(run: Path, tmp: Path) -> dict[str, Path]:
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
    # 3 raw generations (private)
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
    (gens / "README.md").write_text(
        "---\nlicense: other\ntags: [safety, not-for-all-audiences]\n---\n\n# Raw generations (private)\n\n"
        "Baseline and steered answers of Tiny Aya to every prompt, the scored rows, AuditKIT RunResults, the alpha-sweep "
        "answers, the refusal direction (`direction.pt`) and the run log. Contains model replies to harmful prompts. "
        "Shared with the hackathon judges on request.\n", encoding="utf-8")
    return {PROMPTS_REPO: prompts, RESULTS_REPO: results, GENS_REPO: gens}


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
    private = {PROMPTS_REPO: False, RESULTS_REPO: False, GENS_REPO: True}
    links = {}
    with tempfile.TemporaryDirectory() as t:
        staged = stage(Path(args.run_dir), Path(t))
        for name, folder in staged.items():
            repo = f"{user}/{name}"
            files = sorted(str(p.relative_to(folder)).replace("\\", "/") for p in folder.rglob("*") if p.is_file())
            print(f"\n{repo}  ({'PRIVATE' if private[name] else 'public'}, {len(files)} files)")
            for f in files:
                print("   ", f)
            if args.dry_run:
                continue
            if role == "read":
                raise SystemExit("the stored Hugging Face token is read-only. Create a write token at "
                                 "https://huggingface.co/settings/tokens and run `hf auth login`, then re-run.")
            api.create_repo(repo, repo_type="dataset", private=private[name], exist_ok=True)
            api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(folder),
                              commit_message="Lexsi hackathon Track 2b: final run")
            links[name] = f"https://huggingface.co/datasets/{repo}"
    if links:
        out = Path(args.run_dir) / "hub_links.json"
        out.write_text(json.dumps(links, indent=2) + "\n", encoding="utf-8")
        print("\npushed:")
        for k, v in links.items():
            print(f"  {v}  ({'private' if private[k] else 'public'})")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
