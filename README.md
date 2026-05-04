# Friction-Guided Inference

Code, data, and supplementary materials for:

**Pødenphant Lund, T. (2026).** *Friction-Guided Inference: A Free Signal That Improves Any LLM.*
Zenodo. DOI: [pending Zenodo deposit]

ORCID: [0009-0000-4724-2427](https://orcid.org/0009-0000-4724-2427)

---

## TL;DR

Large language models often *know* the right answer but commit to the wrong one. This repository contains the pipeline, data, and analysis for **friction-guided inference** — a method that uses the model's own logprob distribution to:

1. **Calibrate** which correction strategies help a given model (offline, 50–200 questions per cell)
2. **Detect** per-question uncertainty via the **CR signal** (competing routes count)
3. **Commit** to a re-prompted answer or **abstain** based on calibrated thresholds

**Headline result**: combined strategy + calibrated abstention yields **+12 to +21 pp accuracy improvements** across four of five evaluated cells spanning four architectures (dense transformer, mixture-of-experts, Liquid Neural Networks) and four benchmarks (MATH-500, SimpleQA, MMLU-Pro, GPQA Diamond).

| Cell | Model | Architecture | Benchmark | Vanilla → Adaptive | Lift |
|------|-------|--------------|-----------|---------------------|------|
| 1 | Qwen2.5-7B | Dense transformer | MATH-500 L4-5 | 45.8% → 66.5% | **+20.8 pp** |
| 2 | Qwen3-235B | MoE | SimpleQA | 41.1% → 51.8% | **+10.6 pp** |
| 3 | Qwen3-235B | MoE | MMLU-Pro STEM | 55.2% → 62.9% | **+7.7 pp** |
| 4 | LiquidAI LFM2 | Liquid Neural Network | MMLU-Pro STEM | 33.8% → 41.9% | **+8.1 pp** |
| 5 | GPT-oss-20B | Dense transformer | GPQA Diamond | 27.0% → 30.4% | +3.4 pp (n.s.) |

Calibration cost: **~$1.50 per cell + 2 hours** of API calls. The signal itself is free.

---

## Repository structure

```
friction-theory-p3-friction-guided-inference/
├── README.md                              (this file)
├── LICENSE                                (CC-BY-4.0)
├── manuscript/
│   ├── paper3_friction_guided_inference.md  (paper master, ~10,500 words)
│   └── Paper3_Friction_Guided_Inference.pdf (submission PDF, 45 pages)
├── figures/
│   └── fig1-fig6.png                      (6 figures referenced in paper)
├── data/
│   ├── _audit_*.json                      (calibration audit reports)
│   ├── _b0_adaptive_policy*.json          (adaptive policy configs)
│   ├── _calibration_*.json                (calibration data)
│   └── paper3_posthoc_*.json              (post-hoc analysis outputs)
├── scripts/
│   ├── PIPELINE_AUDIT_RULES.md            (23-rule quality assurance framework)
│   ├── make_paper3_figures.py             (reproduces fig1-6)
│   └── paper3_posthoc_cross_model_mmlu.py (post-hoc analysis script)
└── src/friktionsllm/                      (the pipeline library)
    ├── friction/                          (CR signal, competing-routes utils)
    ├── steering/                          (adaptive pipeline engine)
    ├── eval/                              (calibration + evaluation harness)
    ├── ollama_client.py                   (local Ollama API client)
    ├── together_client.py                 (Together.ai API client)
    ├── config.py                          (model + steering configs)
    └── utils/                             (shared helpers)
```

---

## How to reproduce

### 1. Install

```bash
git clone https://github.com/tplund/friction-theory-p3-friction-guided-inference.git
cd friction-theory-p3-friction-guided-inference
pip install -r requirements.txt    # see paper §3 for dependencies
```

### 2. Configure API keys

Create a `.env` file in the repo root:

```
TOGETHER_API_KEY=...      # for Qwen, Mistral, Llama models
FIREWORKS_API_KEY=...     # alternative API (model availability varies)
OPENAI_API_KEY=...        # for judge AND for GPT-oss
ANTHROPIC_API_KEY=...     # for Claude judge
GOOGLE_API_KEY=...        # for Gemini judge
```

API keys are NEVER committed. The `.gitignore` excludes `.env` files.

### 3. Reproduce figures from existing data

```bash
python scripts/make_paper3_figures.py
```

This generates the 6 figures from `data/*.json` files into `figures/`. No API calls needed.

### 4. Run calibration on a new cell

Example: Qwen2.5-7B × MATH-500:

```bash
python -m friktionsllm.eval.calibrate \
    --model qwen25_7b \
    --benchmark math500 \
    --n 200
```

Then inspect the produced `_audit_*.json` for strategy recommendations.

### 5. Run the adaptive pipeline on held-out questions

```bash
python -m friktionsllm.steering.run \
    --model qwen25_7b \
    --benchmark math500 \
    --holdout-only
```

---

## Quality assurance framework

The 23-rule QA framework in [`scripts/PIPELINE_AUDIT_RULES.md`](scripts/PIPELINE_AUDIT_RULES.md) is a contribution of this paper. Every experiment must pass an automated pre-flight check covering:

- **Oracle-cheat detection** (R23e) — commit logic must not reference ground truth
- **Judge-bias detection** (R13d) — per-strategy cross-validation between LLM-judge verdicts and substring matching
- **Depth plateau enforcement** (R7) — calibration is blocked until the oracle plateau is confirmed with 3+ consecutive zero-rescue rounds
- **Logprob format validation** — ensures `top_logprobs > 1`, else CR is undefined
- **Git clean state** — prevents uncommitted calibration drift

These rules emerged from real bugs discovered during development (oracle cheat, 18× judge bias on step-by-step, data loss from file overwrites). The full post-mortem for each rule is in the markdown file.

---

## Signal caveats

The CR signal has a principled limit: it measures **cost of choosing** between competing routes, **not which route is correct**. Our ablation (paper §5.1) shows that CR-weighted answer selection does *not* outperform simply trusting the strategy response. CR's value is in **calibration** (finding which strategy works) and **abstention** (identifying questions where the model should not commit), not in per-question answer selection.

This is theoretically predicted from the friction ceiling (Pødenphant Lund 2026, Paper 1, §9.1b): friction measures computational difficulty, not truth. The cleanest empirical consequence is the **confident-wrong failure mode** — the model has *low* CR but the *wrong* answer, structurally invisible to any friction-based selector.

---

## Related papers

- **Paper 0** (BFT, foundational): Pødenphant Lund, T. (2026). *Behavioural Friction Theory: Toward a Common Currency for Behavioural Science.* Zenodo. DOI: [10.5281/zenodo.19462500](https://doi.org/10.5281/zenodo.19462500)

- **Paper 1** (theoretical foundation, companion): Pødenphant Lund, T. (2026). *Friction as the Cost of Probabilistic Computation: A Generalised Substrate Theory.* Zenodo. DOI: [10.5281/zenodo.20012655](https://doi.org/10.5281/zenodo.20012655)

- **Paper 2** (empirical companion, capacity scaling): Pødenphant Lund, T. (2026). *Capacity Scaling of Encoding-Through-Loading: Application vs. Cloze Asymmetry Across Three Orders of Magnitude.* Zenodo. DOI: [10.5281/zenodo.20013491](https://doi.org/10.5281/zenodo.20013491)

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{pødenphantlund2026frictionguided,
  author       = {P{\o}denphant Lund, Tomas},
  title        = {Friction-Guided Inference: A Free Signal That Improves Any LLM},
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {[pending]},
  url          = {https://github.com/tplund/friction-theory-p3-friction-guided-inference}
}
```

---

## License

This work is released under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](LICENSE).

You are free to share, adapt, and build on this work for any purpose, including commercially, as long as appropriate credit is given.

---

## Contact

Tomas Pødenphant Lund — Independent Research, Aarhus, Denmark
- Web: https://frictiontheory.org
- Correspondence: tomas.lund@frictiontheory.org
- ORCID: https://orcid.org/0009-0000-4724-2427
