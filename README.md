# Refusal-direction steering on Tiny Aya (EN / HI / GU)

Lexsi AI Safety Hackathon, Track 2b. SafeTune steers, AuditKIT scores. Write-up: [ARTICLE.md](ARTICLE.md). Numbers: [METRICS.md](METRICS.md). Submission summary: [SUBMISSION.md](SUBMISSION.md). Lab notes: [FINDINGS.md](FINDINGS.md).

| Path | What |
|---|---|
| `src/run_final.py` | **the submitted run**: preflight → baseline grid → calibration → alpha mini-sweep → steered grid, checkpointed to `results/final/` |
| `src/report_final.py` | scores baseline vs steered through AuditKIT (`compare` + `significance` per language) → `metrics.md` |
| `src/bench_throughput.py` | paired baseline-vs-steered timing on identical batches |
| `src/push_hub.py` | publishes prompts, metrics, raw generations and the direction vector (all public) and the two HF collections |
| `src/eval_steering.py` | calibrate the refusal direction, generate baseline + one pass per alpha |
| `src/score.py` | EN/HI/GU refusal classifier → `metrics.json`, `metrics.md`, AuditKIT runs and comparisons |
| `src/interpret_run.py` | Day-2 stretch: safety-neuron localisation |
| `src/build_prompts.py` | rebuilds the aya_redteaming-derived prompt files |
| `prompts/{en,hi,gu}/` | `harmful.jsonl` (40), `benign_edgy.jsonl` (20), `capability.jsonl` (15) |
| `prompts/calib/` | calibration prompts, disjoint from the eval sets |
| `notebooks/run_colab.ipynb` | Colab T4 variant of the sweep (not used for the submitted numbers) |

```bash
python -m venv .venv                                                # Python >= 3.12
# no NVIDIA GPU (Windows especially)? install the CPU wheel first, or `import torch` fails on c10_cuda.dll:
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
python src/score.py --self-test
python src/eval_steering.py --dry-run --out-dir results/dryrun      # CPU, Qwen2.5-0.5B, ~minutes
python src/score.py --run-dir results/dryrun
python src/interpret_run.py --run-dir results/dryrun --dry-run
```

Submitted run (one 8 GB GPU, CUDA build of torch, access to the gated `CohereLabs/tiny-aya-global`):

```bash
python src/run_final.py                          # ~1 h on an RTX 5050; resumes if interrupted
python src/bench_throughput.py --run-dir results/final
python src/report_final.py --run-dir results/final
python src/push_hub.py --run-dir results/final   # needs a write token
```

Final numbers: [METRICS.md](METRICS.md). The prompt files contain harmful requests used only to measure refusal.
