# eLife Peer-Review MLE Pipeline — Exploratory Pilot Summary

**Date:** 2026-07-13  
**Scope:** Early eLife pilot using a two-stage AI reference-corpus workflow.

> This document is a public-safe summary of an exploratory pilot. Internal API endpoints, infrastructure details, raw-data locations, and operational notes have been removed.

## Pipeline status

| Stage | Status | Public summary |
| --- | --- | --- |
| Core-code validation | Complete | Validation notebook completed across a grid of simulated mixture settings. |
| eLife preprocessing | Complete | 218 usable review records after cleaning. |
| Human reference corpus | Complete | 40 pre-ChatGPT reviews used as the initial human reference set. |
| AI reference corpus | Complete | 40 AI-modified reference reviews generated with a two-stage workflow. |
| Distribution building | Complete | eLife-specific word-occurrence distributions constructed. |
| MLE inference | Complete | Month-level exploratory estimates produced with bootstrap uncertainty. |
| Multi-journal extension | In progress | Requires a larger, consistently structured data source and additional validation. |

## Data summary

The pilot used open peer-review text from eLife. After cleaning, 218 review records were retained. The initial human reference corpus contained 40 reviews from the pre-ChatGPT period.

The AI reference corpus was created using a two-stage process:

1. extract factual / methodological points from each human review;
2. reconstruct those points as a formal academic peer-review paragraph.

The resulting human and AI corpora were used to estimate word-occurrence distributions for downstream mixture-model inference.

## Exploratory result

The pilot generated month-level MLE estimates with bootstrap uncertainty. Several months contained very few review records and produced unstable estimates, so they should not be interpreted substantively.

The largest month in the pilot was **December 2024 (`n = 95`)**, where the estimated mixture parameter was approximately **4%** under this specific reference-corpus definition.

This value should be interpreted cautiously. It is **not** a validated prevalence estimate of AI use in eLife peer review. The estimate depends on the construction of the human and AI reference corpora, sample size, preprocessing rules, vocabulary thresholds, and the assumptions of the mixture model.

## Main methodological lesson

The pilot suggests that **reference-corpus construction is part of the model specification**. A corpus-level estimate of LLM-modified language can change when the operational definition of “AI-modified text” changes.

This motivated the later A/B/C/D prompt framework implemented in `scripts/generate_ai_multi_prompt.py`. That framework is intended for sensitivity analysis and has not yet been established as a validated ordinal measure of real-world AI-assistance intensity.

## Key limitations

- only 40 human reference reviews in the initial pilot;
- highly uneven month-level sample sizes;
- a single-journal pilot cannot support cross-journal generalization;
- generated AI reference reviews may not reflect how real reviewers use LLMs;
- `alpha` is a corpus-level mixture estimate and must not be interpreted at the individual-review level;
- broader validation is required before making substantive claims about prevalence or trends.

## Reproducibility boundary

The public repository contains code, documentation, and derived summaries. Credentials, private infrastructure endpoints, raw-data locations, and large source datasets should remain outside version control.

For current configuration guidance, see the root `README.md` and `.env.example`.
