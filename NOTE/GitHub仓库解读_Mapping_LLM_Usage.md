# Upstream Method Notes — Mapping the Increasing Use of LLMs in Scientific Papers

This note records the relationship between this repository and the upstream work by Liang et al.

## Upstream resources

- Paper: Liang et al. (2024), *Mapping the Increasing Use of LLMs in Scientific Papers*, arXiv:2404.01268
- Repository: https://github.com/Weixin-Liang/Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers
- License: MIT

## What is reused here

The core MLE and text-distribution logic in:

```text
scripts/src/MLE.py
scripts/src/estimation.py
```

is derived from the upstream implementation. The original MIT notice is reproduced in `THIRD_PARTY_NOTICES.md`.

## What this project adds

This repository focuses on adapting the population-level mixture-model workflow to scientific peer-review text, including:

- peer-review preprocessing;
- eLife pilot data preparation;
- generation of alternative AI reference corpora;
- A/B/C/D prompt-based sensitivity analysis;
- multi-journal orchestration;
- project-specific validation, documentation, and visualization.

## Reproduction setup

Clone the upstream repository separately if you want to compare implementations:

```bash
git clone https://github.com/Weixin-Liang/Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers.git
```

No local absolute paths, credentials, private endpoints, or copied paper PDFs are required for this repository.
