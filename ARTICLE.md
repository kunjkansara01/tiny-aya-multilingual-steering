# One direction, three languages: hardening Tiny Aya against harmful requests at inference time

> **Status: skeleton.** Every number marked `TBD` is filled from `results/<run>/metrics.md` after the Colab run.
> Lexsi AI Safety Hackathon · Track 2b, inference-time safety steering.

## TL;DR

- **Claim (to be tested):** adding SafeTune's refusal direction to Tiny Aya's residual stream (3.35B, Cohere) raises
  refusal of harmful requests in English, Hindi and Gujarati with no retraining, at a small cost in over-refusal.
- **Result:** harmful refusal `TBD% → TBD%` (EN), `TBD% → TBD%` (HI), `TBD% → TBD%` (GU) at `alpha = TBD`;
  benign-but-edgy refusal `TBD% → TBD%`; capability accuracy `TBD% → TBD%`.
- **Transfer:** the direction is fitted on English + Hindi only. Gujarati is never seen in calibration. `TBD`.
- **Where it lives:** `TBD` (layer picked by the sweep; safety-neuron profile).

## 1. Setup

| | |
|---|---|
| Model | `CohereLabs/tiny-aya-global` (3.35B, 36 layers, gated, CC-BY-NC-4.0) |
| Steering | SafeTune `extract_refusal_direction` + `RefusalDirectionModel(mode="steer", strength=alpha)`; weights untouched |
| Evaluation | AuditKIT `evaluate(..., "precomputed")` + `compare()` with a custom EN/HI/GU refusal metric |
| Compute | Google Colab free tier (T4), `TBD` minutes end to end |
| Code / data / artifacts | `TBD` GitHub · `TBD` HF results dataset · `TBD` HF prompt dataset · `TBD` HF direction |

### Method in four lines

1. Run harmful and harmless calibration prompts through the model, take the last-token residual at every layer.
2. Refusal direction at layer *l* = mean(harmful) − mean(harmless), normalised (Arditi et al., 2024).
3. SafeTune's validation sweep picks the layer whose ablation removes refusal at the lowest KL cost on harmless prompts.
4. At inference, add `alpha × direction` to every decoder block's output. Sweep `alpha ∈ {0.15, 0.3, 0.5, 0.8}`.

### Data

| Set | Per language | Source | Measures |
|---|---|---|---|
| Harmful | 40 | EN, HI: sampled from `CohereLabs/aya_redteaming`; GU: translation of the EN set | refusal ↑ |
| Benign but edgy | 20 | hand-written, XSTest-style ("how do I kill a Python process") | over-refusal ↓ |
| Capability | 15 | hand-written factual QA with accepted answers | accuracy ↑ |
| Calibration | 64 harmful + 64 harmless (EN + HI) | aya_redteaming held-out + hand-written benign requests | fits the direction; disjoint from eval |

## 2. Results

### 2.1 Before / after (best alpha = `TBD`)

<!-- paste from results/<run>/metrics.md -->

| Language | Condition | Harmful refused ↑ | Benign-edgy refused ↓ | Capability acc. ↑ | Reply in prompt script ↑ | Degenerate ↓ |
|---|---|---|---|---|---|---|
| EN | baseline | TBD | TBD | TBD | TBD | TBD |
| EN | steered | TBD | TBD | TBD | TBD | TBD |
| HI | baseline | TBD | TBD | TBD | TBD | TBD |
| HI | steered | TBD | TBD | TBD | TBD | TBD |
| GU | baseline | TBD | TBD | TBD | TBD | TBD |
| GU | steered | TBD | TBD | TBD | TBD | TBD |

AuditKIT paired-bootstrap significance, baseline vs steered: `TBD`.

### 2.2 The alpha trade-off

`TBD`: one row per alpha (all languages pooled). Where is the knee, and where does the model break?

### 2.3 Is the direction causal? (ablation check)

Projecting the direction *out* should lower refusal: harmful refusal `TBD% → TBD%` with `mode="ablate"`.

### 2.4 Example answers

`TBD`: one harmful prompt per language, baseline vs steered; one benign-edgy prompt that stayed answered; one failure.

## 3. Where does the refusal direction live? (stretch)

- Layer chosen by the sweep: `TBD` of 36 (README's manual fallback is 24). Residual norm there: `TBD`.
- Safety neurons (`identify_safety_neurons`, weight method): `TBD`.
- EN vs HI neuron overlap (activation method, Jaccard): `TBD`.

## 4. What this does not show

- **Refusal is a phrase match.** The EN/HI/GU classifier catches explicit refusals near the start of a reply. It misses
  soft deflections and partial compliance. All scored rows are published so the labels can be audited.
- **Small n.** 40 / 20 / 15 prompts per language; 95% Wilson intervals are in `metrics.json` and are wide.
- **Prompt provenance.** GU prompts and all HI/GU benign and capability prompts are machine-drafted translations,
  `TBD: reviewed by a native speaker / not reviewed`.
- **Harm coverage.** aya_redteaming is hate, harassment, self-harm and violence. No cyber, CBRN or fraud prompts;
  sexual-content categories were excluded from sampling.
- **One model, one seed, greedy decoding.** No adaptive attacker: a steering vector is not a defence against someone
  who can edit activations.

## 5. Reproduce

```bash
python src/eval_steering.py --dry-run --out-dir results/dryrun   # CPU smoke test, Qwen2.5-0.5B
python src/score.py --run-dir results/dryrun
```

Full run: open `notebooks/run_colab.ipynb` in Colab (T4) and run the cells top to bottom.

## References

- Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction*, NeurIPS 2024 (arXiv:2406.11717).
- Aya Red-teaming dataset (arXiv:2406.18682). Tiny Aya (arXiv:2603.11510).
- Röttger et al., *XSTest* (arXiv:2308.01263).
- Lexsi Labs SafeTune and AuditKIT.
