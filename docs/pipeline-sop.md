# Pipeline SOP — Peer-Review LLM Usage Estimation

This document describes the portable workflow used in this repository. All paths are repository-relative by default; private API endpoints and credentials must be supplied through environment variables.

## Environment

Recommended Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scipy matplotlib spacy anthropic requests pyarrow tqdm
python -m spacy download en_core_web_lg
```

Core modules live under:

```text
scripts/src/
├── MLE.py
└── estimation.py
```

The core MLE implementation is adapted from Liang et al.; see `THIRD_PARTY_NOTICES.md`.

## Configuration

Copy `.env.example` to a local `.env` or export variables in your shell. Do not commit real credentials or private endpoints.

```bash
export PROJECT_DIR="$PWD"
export DEEPSEEK_API_KEY="..."
export DEEPSEEK_BASE_URL="https://your-compatible-endpoint.example/v1"
export DEEPSEEK_MODEL="deepseek-v4-flash"
```

For the multi-journal workflow, also provide the local review-unit database:

```bash
export REVIEW_DB_PATH="/path/to/review_units.sqlite"
```

## eLife Pilot Workflow

### 1. Preprocess

```bash
python scripts/preprocess_elife.py
```

Expected outputs:

```text
processed_data/elife/inference_data/
processed_data/elife/human_corpus/elife_human.parquet
```

### 2. Generate AI reference corpus

Single two-stage reference corpus:

```bash
python scripts/generate_ai_corpus.py
```

Sensitivity-analysis A/B/C/D corpora:

```bash
python scripts/generate_ai_multi_prompt.py
```

### 3. Build distribution

```bash
python scripts/build_distribution.py
```

Expected output:

```text
processed_data/elife/distribution/elife.parquet
```

### 4. Run inference

Single reference corpus:

```bash
python scripts/run_elife_pipeline.py
```

Multi-prompt sensitivity analysis:

```bash
python scripts/run_multi_prompt_pipeline.py
```

## Multi-Journal Workflow

The generic entrypoint is:

```bash
python scripts/run_all_journals.py
```

It requires `REVIEW_DB_PATH` and calls the journal-level generation/inference scripts using portable environment-based configuration.

## Data Boundaries

Raw or restricted review datasets should not be committed. Large processed datasets are also excluded through `.gitignore`. Public repository content should be limited to code, documentation, small derived outputs, and non-sensitive figures.

## Interpretation Boundary

The estimated `alpha` is a corpus-level model quantity under a specific human/AI reference-corpus construction. It must not be interpreted as the probability that an individual review was AI-written, and estimates should be reported with uncertainty and sensitivity to reference-corpus design.
