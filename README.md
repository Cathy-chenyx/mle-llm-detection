# LLM Usage Detection in Scientific Peer Reviews

**Population-level estimation of LLM-modified language in scientific peer reviews using an MLE framework, alternative AI reference corpora, and bootstrap uncertainty quantification.**

## Project Snapshot

This repository explores whether a **population-level statistical framework** originally developed for measuring LLM-modified scientific writing can be adapted to **scientific peer reviews**.

The project builds on the methodology and open-source implementation from:

> Liang, W., Zhang, Y., Wu, Z., Lepp, H., Ji, W., Zhao, X., Cao, H., Liu, S., He, S., Huang, Z., Yang, D., Potts, C., Manning, C. D., & Zou, J. Y. (2024). *Mapping the Increasing Use of LLMs in Scientific Papers*. arXiv:2404.01268.

**My extension focuses on:**

- adapting the workflow from scientific-paper text to open peer-review text;
- building an eLife preprocessing and inference pipeline;
- implementing alternative prompt specifications for AI reference-corpus construction;
- comparing how reference-corpus design can affect downstream MLE estimates;
- adding project-specific orchestration, documentation, validation logs, and analysis outputs.

> **Attribution:** the core MLE and text-distribution modules under `scripts/src/` are derived from the original Liang et al. implementation and are not claimed as original code in this repository. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

---

## Research Question

Large language models may influence peer-review writing in ways that are difficult to measure using document-level detectors. This project asks:

> **Can a corpus-level mixture-model framework be adapted to estimate patterns of LLM-modified language in scientific peer reviews, and how sensitive are the estimates to the construction of the AI reference corpus?**

The goal is **population-level inference**, not classification of individual reviewers or individual review reports.

---

## Method Overview

The underlying framework treats a target corpus as a mixture of human-written and LLM-modified language distributions. Let:

- `P_H(w)` = probability of word occurrence in the human reference corpus;
- `P_Q(w)` = probability of word occurrence in the AI-modified reference corpus;
- `α` = mixture proportion estimated for the target corpus.

Maximum-likelihood estimation is used to estimate `α`, with bootstrap resampling for uncertainty quantification.

### Alternative prompt specifications

The current multi-prompt script implements four reference-corpus transformations:

| Level | Implemented prompt strategy | Intended use |
| --- | --- | --- |
| **A** | Proofread spelling / grammar / punctuation only | Very light language intervention |
| **B** | Rewrite wording and sentence structure while preserving factual content | Moderate stylistic rewriting |
| **C** | Two-stage extraction of factual points followed by reconstruction as a formal peer review | Stronger structured rewriting |
| **D** | Generate an independent reviewer-style critique using the human review only as contextual reference | Strong generative transformation |

These are **alternative reference-corpus specifications for sensitivity analysis**. They are not a validated ordinal scale of real-world AI-assistance intensity.

---

## What I Built

### 1. eLife peer-review preprocessing

`scripts/preprocess_elife.py`

Transforms peer-review text into the corpus structure required for downstream distribution estimation and inference.

### 2. AI reference-corpus workflows

- `scripts/generate_ai_corpus.py` — two-stage reference-corpus generation used in the original eLife pilot.
- `scripts/generate_ai_multi_prompt.py` — A/B/C/D alternative prompt specifications for sensitivity analysis.
- `scripts/generate_ai_journal.py` — generalized journal-level corpus generation for later multi-journal work.

### 3. Distribution building and MLE inference

- `scripts/build_distribution.py`
- `scripts/run_elife_pipeline.py`
- `scripts/run_multi_prompt_pipeline.py`

### 4. Documentation and validation

- `docs/pipeline-sop.md` — operation guide
- `NOTE/` — method notes and code-reading notes
- `validation/` — validation scripts and logs
- `output/` — project summaries / selected analysis outputs

---

## Evidence Currently Committed

The repository currently contains an **exploratory eLife pilot** based on the earlier two-stage reference-corpus workflow. The committed report documents:

- 218 usable eLife review records after preprocessing;
- 40 pre-ChatGPT human reference reviews;
- an eLife-specific word-distribution dictionary;
- month-level MLE estimates with bootstrap uncertainty;
- substantial instability in months with very small sample sizes.

The largest month in that pilot was December 2024 (`n = 95` review units), with an estimated `α` around 4% under that specific reference-corpus specification. This number should be interpreted as a **pipeline-specific exploratory estimate**, not a validated prevalence estimate for eLife or peer review generally.

The A/B/C/D framework is implemented in code, but this public repository does **not currently contain a complete, independently verified multi-prompt result package**. Until that is added, this project should be read primarily as a methodological adaptation and reproducible research workflow rather than a definitive empirical finding.

---

## Limitations

This is a research and learning project, not a production detector.

Key limitations include:

- a small human reference corpus in the initial eLife pilot;
- highly uneven month-level sample sizes;
- dependence on prompt-generated AI reference corpora that may not reflect real-world LLM usage patterns;
- sensitivity to preprocessing, vocabulary thresholds, reference-corpus composition, and model assumptions;
- `α` is a **corpus-level mixture estimate**, not a probability that any individual review was AI-written;
- the multi-prompt framework requires broader validation before stronger substantive claims are warranted.

---

## Reproducibility

### Environment

Python 3.8+ with packages including `pandas`, `numpy`, `scipy`, `swifter`, `matplotlib`, `spacy`, and either `anthropic` or `requests` depending on the generation script.

API credentials and endpoints should be configured through environment variables rather than hard-coded source values.

```bash
export DEEPSEEK_API_KEY="your-key-here"
export DEEPSEEK_BASE_URL="https://your-api-endpoint.example"
export DEEPSEEK_MODEL="your-model-name"
```

See `.env.example` for the expected configuration shape.

### Run the eLife workflow

```bash
python scripts/generate_ai_corpus.py
python scripts/build_distribution.py
python scripts/run_elife_pipeline.py
```

### Run the alternative-prompt workflow

```bash
python scripts/generate_ai_multi_prompt.py
python scripts/run_multi_prompt_pipeline.py
```

Large raw and processed datasets are intentionally excluded from version control.

---

## Data & Publication Boundary

This repository is intended to contain **code, documentation, derived summaries, and reproducibility instructions**, not a redistribution archive for third-party papers or large source datasets.

Reference papers should be accessed from their original publishers / preprint repositories. Raw peer-review data should be used according to the terms of the original data source.

---

## Third-Party Attribution

The statistical framework and core implementation in:

- `scripts/src/MLE.py`
- `scripts/src/estimation.py`

are derived from:

[Weixin-Liang/Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers](https://github.com/Weixin-Liang/Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers)

The upstream code is distributed under the **MIT License**. The original copyright and permission notice are reproduced in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

The peer-review adaptation, preprocessing, prompt experiments, orchestration, documentation, and project-specific analysis in this repository are project-specific extensions.

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
