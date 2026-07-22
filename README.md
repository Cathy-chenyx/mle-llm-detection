# LLM Usage Detection in Scientific Peer Reviews via Multi-Prompt MLE

**MLE-based estimation of LLM-modified content proportions in scientific peer reviews, powered by a multi-prompt (A/B/C/D) pipeline with bootstrap confidence intervals.**

## Overview

The rapid adoption of large language models (LLMs) raises a critical question: to what extent has AI-generated or AI-assisted content permeated the academic peer review process? This project implements and extends the methodology from [Liang et al. (2025)](https://arxiv.org/abs/2503.xxxxx) — a monitoring framework that uses Maximum Likelihood Estimation (MLE) on word-frequency distributions to infer the proportion of LLM-modified text in large corpora, without requiring access to proprietary LLM detection tools.

Our adaptation introduces a **multi-prompt gradient pipeline**: human peer reviews are rewritten at four progressively stronger AI-intervention levels (A → D), producing a spectrum of AI corpora. The MLE model is then fit on real-world review data at each level, enabling not just detection but also **profiling of AI intervention depth** across journals and time periods.

### Key Features

- **Four-level multi-prompt pipeline**: Level A (factual extraction) → B (bullet-point rewrite) → C (paragraph rewrite) → D (full rewriting from scratch), capturing a gradient from light AI assistance to full generation.
- **Self-contained architecture**: Core modules (`MLE.py`, `estimation.py`) are embedded under `scripts/src/`, with no external repository dependencies.
- **Bootstrap confidence intervals**: Each α estimate comes with 1,000 bootstrap iterations for uncertainty quantification.
- **Comparative visualization**: Combined plots of estimated LLM-usage rates across prompt levels, time periods, and journals.
- **eLife pilot study**: Full pipeline validated on the [eLife](https://elifesciences.org/) open peer review dataset (40 human reviews + 12 months of inference data).

---

## Repository Structure

```
AI_project/
├── scripts/                          # Pipeline scripts (self-contained)
│   ├── src/                          # Core inference modules
│   │   ├── MLE.py                    # MLE estimator with bootstrap
│   │   └── estimation.py             # Word-frequency distribution builder
│   ├── preprocess_elife.py           # Raw review data → structured format
│   ├── generate_ai_corpus.py         # Single-prompt AI corpus generation
│   ├── generate_ai_multi_prompt.py   # Multi-prompt (A/B/C/D) AI corpus generation
│   ├── build_distribution.py         # Build log-probability dictionaries
│   ├── run_elife_pipeline.py         # End-to-end single-prompt pipeline
│   └── run_multi_prompt_pipeline.py  # End-to-end multi-prompt pipeline
├── processed_data/                   # Pipeline outputs (gitignored, generated at runtime)
│   └── elife/
│       ├── human_corpus/             # Human-written review samples
│       ├── ai_corpus/                # AI-generated review variants (parquet)
│       ├── inference_data/           # Monthly review data for inference
│       └── distribution/             # Log-probability dictionaries
├── NOTE/                             # Study notes & learning materials
│   ├── AI生成内容检测论文学习笔记.md
│   ├── GitHub仓库解读_Mapping_LLM_Usage.md
│   ├── MLE_estimation_逐行解读.md
│   └── 统计理论基础_极大似然估计_混合模型_群体推断.md
├── Reference/                        # Reference papers (PDF)
├── docs/                             # Documentation
│   └── pipeline-sop.md               # Full pipeline SOP
├── output/                           # Generated reports & visualizations
├── validation/                       # Validation scripts & logs
└── README.md
```

---

## Methodology

### The Mixture Model

Given a corpus of peer reviews, each text can be modeled as a mixture of *human-written* and *LLM-modified* components. Following Liang et al. (2025), the probability that a word *w* appears in a random review is:

**P(w) = (1 − α) · P_H(w) + α · P_Q(w)**

where:

| Parameter | Description |
|-----------|-------------|
| α ∈ [0, 1] | Proportion of LLM-modified content (the target of inference) |
| P_H(w) | Word occurrence probability in the **human** reference corpus |
| P_Q(w) | Word occurrence probability in the **AI** reference corpus |

### Multi-Prompt Gradient Design

To capture the *depth* of AI intervention (not just its presence), we generate four AI reference corpora at escalating levels:

| Level | Prompt Strategy | AI Intervention Depth |
|-------|----------------|----------------------|
| **A** | Extract factual key points as a numbered list | Minimal (information preservation) |
| **B** | Bullet-point rewrite preserving all content | Light (structural changes) |
| **C** | Paragraph-form rewrite, natural flow | Moderate (compositional changes) |
| **D** | Full rewrite from scratch in professional tone | Heavy (creative reformulation) |

Each level produces a distinct P_Q(w) distribution. Running MLE inference at all four levels on the same real-world data reveals not just *if* AI was used, but *how deeply*.

### Estimation

1. **Build distributions**: Compute binary word occurrence probabilities P_H(w) and P_Q(w) from reference corpora, filtered by vocabulary intersection and frequency thresholds.
2. **MLE with bootstrap**: For each monthly batch of real reviews, fit α that maximizes the binomial mixture log-likelihood. Repeat with 1,000 bootstrap resamples per batch to obtain 95% confidence intervals.

---

## Quick Start

### Prerequisites

- **Python 3.8+** with conda environment `llm-detection`
- Core dependencies: `pandas`, `numpy`, `scipy`, `swifter`, `matplotlib`
- AI corpus generation requires Anthropic-compatible API access (`DEEPSEEK_API_KEY` environment variable)

### Setup

```bash
# Clone the repository
git clone https://github.com/Cathy-chenyx/mle-llm-detection.git
cd mle-llm-detection

# Activate environment
conda activate llm-detection
```

### Run the Multi-Prompt Pipeline

```bash
# Step 1: Generate multi-prompt AI corpora (A/B/C/D)
python scripts/generate_ai_multi_prompt.py

# Step 2: Build log-probability dictionaries for each level
python scripts/build_distribution.py

# Step 3: Run full MLE inference pipeline
python scripts/run_multi_prompt_pipeline.py
```

Output is written to `processed_data/elife/`, including `.csv` alpha estimates, `.parquet` distribution files, and a combined comparison plot.

### API Key Configuration

AI corpus generation (`generate_ai_multi_prompt.py`) calls a DeepSeek API endpoint and reads the key from the environment:

```bash
export DEEPSEEK_API_KEY="your-key-here"
```

No API keys are hardcoded in the source.

---

## Key Results (eLife Pilot)

The pipeline was validated on the [eLife](https://elifesciences.org/) open peer review dataset covering 12 months of review data. The multi-prompt (A/B/C/D) pipeline successfully:

- Generated four distinct AI reference corpora with clearly differentiated word-frequency profiles
- Produced stable α estimates with tight bootstrap confidence intervals
- Revealed a monotonic gradient: **α_A < α_B < α_C < α_D**, confirming that more aggressive prompts produce systematically distinguishable signals

Detailed progress reports and comparison plots are available in the `output/` directory.

---

## Documentation

- [pipeline-sop.md](docs/pipeline-sop.md) — Full step-by-step pipeline operation manual
- [output/](output/) — Progress reports and MLE results
- [NOTE/](NOTE/) — Study notes covering MLE theory, paper analyses, and code walkthroughs

---

## Citation

If you use this code or methodology in your research, please cite:

> Liang, W., Izzo, Z., Zhang, Y., Lepp, H., Cao, H., Zhao, X., Ye, C., Liu, S., Huang, Z., McFarland, D.A., & Zou, J.Y. (2025). Quantifying large language model usage in scientific papers. *arXiv preprint*.

```bibtex
@article{liang2025quantifying,
  title={Quantifying Large Language Model Usage in Scientific Papers},
  author={Liang, Weixin and Izzo, Zachary and Zhang, Yaohui and Lepp, Haley and Cao, Hancheng and Zhao, Xuandong and Ye, Chen and Liu, Sheng and Huang, Zhi and McFarland, Daniel A and Zou, James Y},
  journal={arXiv preprint arXiv:2501.xxxxx},
  year={2025}
}
```

---

## License

This project is for academic research purposes. Please refer to the original [Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers](https://github.com/) repository for licensing terms on the base methodology.

---

## Contact

**Chen Yuxin (Cathy)** — Biostatistics graduate student, Southern Medical University  
GitHub: [@Cathy-chenyx](https://github.com/Cathy-chenyx)
