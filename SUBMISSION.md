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

| What | Link | Visibility |
|---|---|---|
| Code (GitHub) | https://github.com/kunjkansara01/tiny-aya-multilingual-steering | public |
| Article | https://huggingface.co/datasets/kpk11/tiny-aya-refusal-steering-results/blob/main/ARTICLE.md (also `ARTICLE.md` in the repo) | public |
| Metrics, calibration, sweep, timing | https://huggingface.co/datasets/kpk11/tiny-aya-refusal-steering-results | public |
| Prompt dataset (EN / HI / GU eval sets + calibration set) | https://huggingface.co/datasets/kpk11/safety-prompts-en-hi-gu | public |
| Raw generations, scored rows, AuditKIT run files, direction vector | https://huggingface.co/datasets/kpk11/tiny-aya-refusal-steering-generations | **private**, judges on request |

To give a judge access to the private repository: its Settings page on the Hub → add their HF username as a
collaborator, or share a read token.

## What the form is likely to ask

| Field | Answer |
|---|---|
| Track | 2b, inference-time safety steering |
| Team | solo |
| Model | `CohereLabs/tiny-aya-global` (Cohere, 3.35B, 36 layers, CC-BY-NC-4.0, gated), bf16 |
| Lexsi repos chained | **SafeTune** 0.1.18 (`extract_refusal_direction`, `RefusalDirectionModel(mode="steer")`) → **AuditKIT** 1.1.12 (`evaluate(..., "precomputed")`, `compare`, `significance`) |
| Method in one line | add `alpha × refusal direction` (layer 16 of 36, picked by SafeTune's validation sweep) to every decoder block's output; no retraining, weights untouched |
| Data | 225 eval prompts: 40 harmful + 20 benign-but-edgy + 15 capability in each of EN, HI, GU. Harmful EN/HI sampled from `CohereLabs/aya_redteaming` (Apache-2.0); GU translated from the EN set. 64 + 64 calibration prompts (EN + HI), disjoint from eval |
| Compute | one RTX 5050 Laptop GPU, 8 GB; weights 6.24 GiB, peak 6.32 GiB |
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

## Before you submit

1. Commit and push `ARTICLE.md`, `SUBMISSION.md`, `METRICS.md`, `FINDINGS.md`, `README.md` and `src/` to the GitHub
   repository so it matches the Hub copy.
2. The repository contains `prompts/`, which includes the harmful prompt texts (the same content as the public
   prompt dataset).
3. `results/` is git-ignored; the final numbers are in `METRICS.md` at the repository root.
