# LLM Usage Detection in Scientific Peer Reviews

**Population-level estimation of LLM-modified content in scientific peer reviews using an MLE framework, multi-prompt reference corpora, and bootstrap uncertainty quantification.**

## Project Snapshot

This repository explores whether a **population-level statistical framework** originally developed for measuring LLM-modified scientific writing can be adapted to **scientific peer reviews**.

The project builds on the methodology and open-source implementation from:

> Liang, W., Zhang, Y., Wu, Z., Lepp, H., Ji, W., Zhao, X., Cao, H., Liu, S., He, S., Huang, Z., Yang, D., Potts, C., Manning, C. D., & Zou, J. Y. (2024). *Mapping the Increasing Use of LLMs in Scientific Papers*. arXiv:2404.01268.

**My extension focuses on:**

- adapting the workflow from scientific-paper text to **open peer-review text**;
- building an **eLife preprocessing and inference pipeline**;
- generating four AI reference corpora with progressively stronger rewriting prompts (A/B/C/D);
- comparing MLE estimates across prompt specifications and time periods;
- adding end-to-end orchestration, documentation, validation logs, and visualization for the adapted workflow.

> **Important:** the core MLE and text-distribution modules under `scripts/src/` are derived from the original Liang et al. implementation and are not claimed as original code in this repository. See [Third-Party Attribution](#third-party-attribution) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

---

## Research Question

Large language models may influence peer-review writing in ways that are difficult to measure using document-level detectors. This project asks:

> **Can a corpus-level mixture-model framework be adapted to estimate patterns of LLM-modified language in scientific peer reviews, and how sensitive are the estimates to the way the AI reference corpus is constructed?**

The goal is **population-level inference**, not classification of individual reviewers or individual review reports.

---

## Method Overview

The underlying framework treats a target corpus as a mixture of human-written and LLM-modified language distributions. Let:

- `P_H(w)` = probability of word occurrence in the human reference corpus;
- `P_Q(w)` = probability of word occurrence in the AI-modified reference corpus;
- `α` = mixture proportion estimated for the target corpus.

The project uses maximum-likelihood estimation (MLE) to estimate `α`, together with bootstrap resampling for uncertainty quantification.

### Multi-Prompt Reference Design

Rather than relying on one rewriting prompt, this adaptation creates four AI reference corpora:

| Level | Prompt strategy | Intended transformation |
| --- | --- | --- |
| **A** | Extract factual key points | Minimal structural intervention |
| **B** | Rewrite as bullet points | Light restructuring |
| **C** | Rewrite into natural paragraphs | Moderate rewriting |
| **D** | Rewrite from scratch in a professional tone | Strong rewriting |

These levels are used as a **sensitivity-analysis device** for reference-corpus construction. They should not be interpreted as a validated clinical-style scale of “AI intervention depth.”

---

## What I Built

### 1. eLife peer-review preprocessing

`scripts/preprocess_elife.py`

Transforms open peer-review data into the corpus structure required for downstream distribution estimation and inference.

### 2. Multi-prompt AI corpus generation

`scripts/generate_ai_multi_prompt.py`

Generates four prompt-specific AI reference corpora from human review samples so the downstream MLE estimates can be compared across alternative reference definitions.

### 3. Distribution-building workflow

`scripts/build_distribution.py`

Constructs the word-occurrence distributions used by the mixture-model framework.

### 4. End-to-end inference pipeline

`scripts/run_multi_prompt_pipeline.py`

Runs preprocessing outputs through the prompt-specific distribution files and MLE workflow, then aggregates results for comparison.

### 5. Documentation and reproducibility support

- `docs/pipeline-sop.md` — operation guide
- `NOTE/` — method notes and code-reading notes
- `validation/` — validation scripts and logs
- `output/` — generated reports and visualizations

---

## Repository Structure

```text
mle-llm-detection/
├── scripts/
│   ├── src/
│   │   ├── MLE.py
│   │   └── estimation.py
│   ├── preprocess_elife.py
│   ├── generate_ai_corpus.py
│   ├── generate_ai_multi_prompt.py
│   ├── build_distribution.py
│   ├── run_elife_pipeline.py
│   └── run_multi_prompt_pipeline.py
├── processed_data/        # generated outputs; mostly gitignored
├── NOTE/                  # learning and method notes
├── Reference/             # reference material
├── docs/
│   └── pipeline-sop.md
├── output/
├── validation/
├── THIRD_PARTY_NOTICES.md
└── README.md
```

---

## Pilot Analysis

The current repository contains an **exploratory eLife pilot** using open peer-review data and four alternative AI reference corpora.

In the pilot outputs, the estimated mixture proportion varies across prompt specifications, and the prompt-specific estimates show an ordered pattern in the current experiment.

This should be treated as an **exploratory sensitivity result**, not as evidence that the four prompt levels form a validated measure of real-world AI-assistance intensity. The estimates depend on the construction of both the human and AI reference corpora, sample size, preprocessing decisions, vocabulary filtering, and model assumptions.

### Why this matters

The main methodological lesson is that **reference-corpus design is itself part of the statistical model**. A population-level estimate of LLM-modified content can change depending on how “AI-modified text” is operationalized, so robustness across alternative prompt constructions should be examined rather than assuming one prompt defines the ground truth.

---

## Limitations

This is a research and learning project, not a production detector.

Key limitations include:

- the current analysis is based on a limited pilot rather than a comprehensive multi-journal study;
- prompt-generated AI reference corpora may not represent how researchers actually use LLMs during peer review;
- `α` is a corpus-level statistical estimate and must not be interpreted as the probability that any individual review was AI-written;
- estimates may be sensitive to preprocessing, vocabulary thresholds, reference-corpus composition, and model assumptions;
- the current multi-prompt design is exploratory and requires broader validation before stronger claims can be made.

---

## Quick Start

### Environment

Python 3.8+ with core dependencies including:

- `pandas`
- `numpy`
- `scipy`
- `swifter`
- `matplotlib`

AI reference-corpus generation additionally requires an API credential configured through the environment rather than hard-coded in source.

```bash
export DEEPSEEK_API_KEY="your-key-here"
```

### Run the multi-prompt workflow

```bash
python scripts/generate_ai_multi_prompt.py
python scripts/build_distribution.py
python scripts/run_multi_prompt_pipeline.py
```

Generated analysis files are written under `processed_data/` and `output/` according to the pipeline configuration.

---

## Third-Party Attribution

The statistical framework and core implementation in:

- `scripts/src/MLE.py`
- `scripts/src/estimation.py`

are derived from the open-source repository:

[Weixin-Liang/Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers](https://github.com/Weixin-Liang/Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers)

The upstream code is distributed under the **MIT License**. The original copyright and permission notice are reproduced in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

The preprocessing, peer-review adaptation, multi-prompt reference-corpus workflow, orchestration, documentation, and project-specific analysis in this repository are extensions built for this project.

---

## Reference

Liang, W., Zhang, Y., Wu, Z., Lepp, H., Ji, W., Zhao, X., Cao, H., Liu, S., He, S., Huang, Z., Yang, D., Potts, C., Manning, C. D., & Zou, J. Y. (2024). **Mapping the Increasing Use of LLMs in Scientific Papers.** arXiv:2404.01268.

- Paper: https://arxiv.org/abs/2404.01268
- Original implementation: https://github.com/Weixin-Liang/Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers

---

## Contact

**Yixin Chen (Cathy)**  
Applied Statistics / Biostatistics, Southern Medical University  
GitHub: [@Cathy-chenyx](https://github.com/Cathy-chenyx)
