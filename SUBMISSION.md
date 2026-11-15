# Submission: Lexsi AI Safety Hackathon, Track 2b (inference-time safety steering)

## Title

One direction, three languages: hardening Tiny Aya against harmful requests at inference time

## Headline finding (two sentences)

A single refusal direction extracted with SafeTune from English and Hindi prompts, added to Tiny Aya (3.35B) at
inference time, raises explicit refusal of harmful requests from 48.3% to 80.8% (p < 0.001, AuditKIT paired bootstrap)
across English, Hindi and Gujarati, with the largest gain in Gujarati (30.0% → 67.5%), a language the direction never
saw. The cost at that strength is 3 of 60 benign-but-edgy prompts refused and 1 of 45 factual answers lost, and at
half the strength the gain is still +21.7 points with no over-refusal and no capability loss.

## Links

Everything is public and un-gated (checked from an unauthenticated client).

**Collections (the two the form asks for)**

| Collection | Link | Contents |
|---|---|---|
| Models (Track 2b) | https://huggingface.co/collections/kpk11/lexsi-hackathon-t2b-tiny-aya-refusal-steering-models-6af8db51be94db1572c81848 | `kpk11/tiny-aya-refusal-direction` |
| Datasets | https://huggingface.co/collections/kpk11/lexsi-hackathon-t2b-tiny-aya-refusal-steering-datasets-6af8db52a45f2879a96003cc | prompts, generations, metrics + article |

**Individual repositories**

| What | Link | Type |
|---|---|---|
| Code | https://github.com/kunjkansara01/tiny-aya-multilingual-steering | GitHub |
| Track 2b artifact: refusal direction (safetensors) + steering config + model card with dual-use notice | https://huggingface.co/kpk11/tiny-aya-refusal-direction | HF model |
| Prompt sets (EN / HI / GU eval + calibration) | https://huggingface.co/datasets/kpk11/safety-prompts-en-hi-gu | HF dataset |
| Raw baseline and steered generations, scored rows, AuditKIT run files (content warning at the top of the card) | https://huggingface.co/datasets/kpk11/tiny-aya-refusal-steering-generations | HF dataset |
| Metrics, calibration, sweep, timing, article | https://huggingface.co/datasets/kpk11/tiny-aya-refusal-steering-results | HF dataset |
| Article | https://github.com/kunjkansara01/tiny-aya-multilingual-steering#readme (the GitHub README is the article; copy at https://huggingface.co/datasets/kpk11/tiny-aya-refusal-steering-results/blob/main/ARTICLE.md) | Markdown |

The base model, `CohereLabs/tiny-aya-global`, is gated by Cohere (licence click); none of our repositories are.

## What the form is likely to ask

| Field | Answer |
|---|---|
| Track | 2b, inference-time safety steering |
| Team | solo |
| Model | `CohereLabs/tiny-aya-global` (Cohere, 3.35B, 36 layers, CC-BY-NC-4.0, gated), bf16 |
| Lexsi repos chained | **SafeTune** 0.1.18 (`extract_refusal_direction`, `RefusalDirectionModel(mode="steer")`) → **AuditKIT** 1.1.12 (`evaluate(..., "precomputed")`, `compare`, `significance`) |
| Method in one line | add `alpha × refusal direction` (layer 16 of 36, picked by SafeTune's validation sweep) to every decoder block's output; no retraining, weights untouched |
| Data | 225 eval prompts: 40 harmful + 20 benign-but-edgy + 15 capability in each of EN, HI, GU. Harmful EN/HI sampled from `CohereLabs/aya_redteaming` (Apache-2.0); GU translated from the EN set (LLM-assisted, native-speaker spot-checked); benign-edgy and capability sets are LLM-assisted drafts curated by the author. 64 + 64 calibration prompts (EN + HI), disjoint from eval |
| Compute | one RTX 5050 Laptop GPU, 8 GB; weights 6.24 GiB, peak 6.32 GiB |
| AI usage statement | Portions of the code and prompt drafts were produced with AI assistance; experimental design, native-language review, and final analysis are the author's. |
| Reproduce | `python src/run_final.py`, then `python src/bench_throughput.py --run-dir results/final`, then `python src/report_final.py --run-dir results/final` |

## Before / after metrics (baseline → steered, alpha 0.8)

| Metric | All | English | Hindi | Gujarati (zero-shot) |
|---|---|---|---|---|
| Harmful refused ↑ | 48.3% → 80.8% (58 → 97 of 120) | 60.0% → 90.0% | 55.0% → 85.0% | 30.0% → 67.5% |
| p, AuditKIT paired bootstrap / exact McNemar | <0.001 / <0.001 | <0.001 / <0.001 | <0.001 / <0.001 | <0.001 / <0.001 |
| Benign-edgy refused ↓ | 0.0% → 5.0% (0 → 3 of 60) | 0.0% → 15.0% | 0.0% → 0.0% | 0.0% → 0.0% |
| Capability correct ↑ | 100% → 97.8% (45 → 44 of 45) | 100% → 93.3% | 100% → 100% | 100% → 100% |
| Degenerate replies ↓ | 0 → 0 of 225 | 0 → 0 | 0 → 0 | 0 → 0 |
| Reply in the prompt's script ↑ | 100% → 100% | 100% → 100% | 100% → 100% | 100% → 100% |

Second operating point, alpha 0.4: harmful refused 70.0% (EN 80.0%, HI 72.5%, GU 57.5%), benign-edgy refused 0/60,
capability 45/45.

Latency: steered generation takes a median 1.1× the baseline time per token in paired trials (about 1.3× in an
earlier run's undisturbed stretches); no extra model and no extra pass. Absolute tok/s on this machine varied more
than tenfold with other GPU load, so only the ratio is reported as a result.

## Caveats stated in the article

- Refusal is judged by an EN/HI/GU phrase match on the first 320 characters: a lower bound, and part of the gain
  is soft deflection becoming explicit refusal.
- Gujarati comprehension is a confound for the Gujarati numbers; the Gujarati harmful set is a translation.
- Five harm categories, n = 40 harmful prompts per language, one model, one run, greedy decoding.
- Alpha was chosen on a 60-prompt EN + HI subset of the evaluation set; the sweep extended beyond the requested
  0.05–0.4 grid.
- Steering made one Gujarati reply worse (`harm-35`).
- The direction is dual-use (ablating it removed refusals on held-out prompts, 3/8 → 0/8); it is published with
  that disclosed on the model card and in the article's limitations, not gated.

## Before you submit

1. The article is the top of `README.md`, with the content warning first; `ARTICLE.md` is a pointer to it.
2. The repository contains `prompts/`, which includes the harmful prompt texts (the same content as the public
   prompt dataset).
3. `results/` is git-ignored; the final numbers are in `METRICS.md` at the repository root.
