# PIPA-LLM: A Reproducible Formal-Rule Benchmark for Auditing LLM Personal-Data Compliance Decisions

This repository accompanies the paper:

> **Auditing LLM Compliance Decisions under the Korean PIPA: A Reproducible Formal-Rule Benchmark and Cross-Lingual Analysis**
> Yongjun Kim and Eun Jung Sun. *Submitted to IEEE Access, 2026.*

We audit eight open large language models (LLMs) as personal-data **compliance reasoners** under Korea's
Personal Information Protection Act (PIPA). Ground truth is produced by a **deterministic formal-rule engine**
(`decide()`), so every label is objective and reproducible without human annotation. The benchmark is
**bilingual (Korean/English)** and scores models not only on the final outcome but at the level of individual
legal decision nodes.

## Key findings

- **Capability–safety inversion.** Stronger English performance coincides with a *larger* Korean deficit
  (Pearson `r = -0.65`; up to a 24.3-point KO–EN gap for Gemma2-9B, McNemar `p ≈ 2e-16`).
- **Language composition, not scale, drives the gap.** Within the Gemma 2 family the gap *reverses* with
  size; the heavily multilingual Qwen2.5-7B shows no significant gap.
- **The deficit is in reasoning integration, not comprehension.** Per-node fact recognition is nearly
  language-invariant (91.5% KO vs. 92.5% EN), while the integrated verdict diverges by 24.3 points.
- **Jurisdiction framing contaminates shared requirements.** Reframing identical scenarios from PIPA to GDPR
  flips judgments on requirements common to both regimes (sensitive-consent 65%, purpose 43%).
- **Bit-reproducible.** Greedy decoding yields sd = 0.00%; `T=0.7` yields sd ≤ 0.61% with a persistent gap.

## Repository layout

Scripts and data live in the repository root so every script runs as-is (no path configuration).

**Code**
```
rule_engine.py        # ⭐ deterministic PIPA ground-truth engine: decide(predicates) -> outcome + nodes
generate_dataset.py   # builds the balanced bilingual scenarios -> dataset_v1.json
run_experiments.py    # runs the 8 models x 2 languages via Ollama
evaluate.py           # outcome accuracy / cross-lingual gap
node_evaluate.py      # per-node (fact-recognition) accuracy
stability.py          # reproducibility: T=0 (greedy) and T=0.7 repetitions
mcnemar.py            # paired McNemar significance test
gdpr_symmetric.py     # RQ3: symmetric PIPA-vs-GDPR jurisdiction test
analyze.py            # figures 1-5
analyze_nodes.py      # figures 6-7
analyze_gdpr.py       # figure 8
make_table.py         # main results table
verify_numbers.py     # recomputes every number reported in the paper
```

**Data**
```
dataset_v1.json       # the benchmark: N=350 balanced bilingual scenarios with gold labels
results_v1.json       # main experiment outputs (8 models x 2 languages)
node_result.json      # per-node accuracy
stability_result.json # reproducibility results
mcnemar_result.json   # McNemar test results
gdpr_pred.json        # GDPR-framing predictions (RQ3)
```

## Reproducing the numbers

Every statistic in the paper is recomputed directly from the released result files by a single script:

```bash
python3 verify_numbers.py        # prints accuracy range, r, McNemar p, node gaps, RQ3, reproducibility
```

To regenerate the benchmark or re-run the models you need [Ollama](https://ollama.com) with the eight model
tags pulled locally:

```bash
python3 generate_dataset.py      # -> dataset_v1.json
python3 run_experiments.py       # queries each model at temperature=0 (JSON-constrained)
python3 evaluate.py
```

## Models evaluated

Gemma 2 (9B q4, 2B q8/q4), Qwen 2.5 (7B q4), Llama 3.2 (3B q8/q4), and Phi-3 mini (3.8B q8/q4),
all instruction-tuned and served locally via Ollama at `temperature=0`.

## Ethics

All scenarios are **synthetic** and contain no real personal data. The rule engine encodes statutory
*structure*, not contested legal interpretation.

## Citation

```bibtex
@article{kim2026pipa,
  title   = {Auditing LLM Compliance Decisions under the Korean PIPA:
             A Reproducible Formal-Rule Benchmark and Cross-Lingual Analysis},
  author  = {Kim, Yongjun and Sun, Eun Jung},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Under review}
}
```

## License

Code is released under the MIT License (see `LICENSE`). The dataset and result files are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
