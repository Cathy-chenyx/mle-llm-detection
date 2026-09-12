"""Generate an AI reference corpus for a named journal.

Required environment variables:
  DEEPSEEK_API_KEY
  DEEPSEEK_BASE_URL   OpenAI-compatible endpoint base URL, e.g. https://host/v1

Optional:
  DEEPSEEK_MODEL      default: deepseek-v4-flash
  PROJECT_DIR         default: repository root
"""

import argparse
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import spacy
import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Generate AI peer-review reference corpus")
    parser.add_argument("--journal_name", required=True)
    parser.add_argument("--project_dir", default=None)
    parser.add_argument("--api_model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--temperature", default=1.5, type=float)
    parser.add_argument("--max_tokens", default=4096, type=int)
    parser.add_argument("--n_ai_samples", default=100, type=int)
    parser.add_argument("--random_state", default=42, type=int)
    parser.add_argument("--n_fewshot", default=10, type=int)
    return parser.parse_args()


API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
API_BASE = os.environ.get("DEEPSEEK_BASE_URL", "").rstrip("/")
if not API_KEY:
    raise RuntimeError("Set DEEPSEEK_API_KEY before running this script.")
if not API_BASE:
    raise RuntimeError("Set DEEPSEEK_BASE_URL before running this script.")
if not API_BASE.endswith("/chat/completions"):
    API_BASE = f"{API_BASE}/chat/completions"

HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def chat_model(prompt, model, temperature, max_tokens):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a peer reviewer for an academic journal. Write in fluent academic English."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = requests.post(API_BASE, headers=HEADERS, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def create_tokenizer():
    nlp = spacy.load("en_core_web_lg")
    nlp.max_length = 80_000_000

    def tokenize(text):
        text = str(text).replace("\n", " ")[: nlp.max_length]
        try:
            doc = nlp(text)
            chunks = [sent.text for sent in doc.sents]
        except Exception:
            chunks = re.split(r"(?<=[.!?])\s+", text)
        out = []
        for chunk in chunks:
            words = [w for w in re.findall(r"\b\w+\b", chunk.lower()) if not w.isdigit()]
            if words:
                out.append(words)
        return out

    return tokenize


STAGE1_PROMPT = """You are simulating a peer reviewer for an academic journal. Below are example human peer-review excerpts. Write ONE new review for a different imaginary paper in a similar field. Do not copy content from the examples. Include a brief summary, major critiques, minor issues, and an overall recommendation.\n\nExamples:\n---\n{examples}\n---\n\nNew review:"""
STAGE2_PROMPT = """Condense the following AI-generated peer review into 5-7 distinct paragraphs while preserving academic vocabulary, hedging, and formal tone. Output only the condensed review.\n\n---\n{review}\n---"""


def main():
    args = parse_args()
    project_dir = Path(
        args.project_dir
        or os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[1])
    ).expanduser().resolve()
    journal = args.journal_name

    human_path = project_dir / "processed_data" / journal / "human_corpus" / f"{journal}_human.parquet"
    ai_dir = project_dir / "processed_data" / journal / "ai_corpus"
    ai_dir.mkdir(parents=True, exist_ok=True)
    ai_path = ai_dir / f"{journal}_ai.parquet"

    human_df = pd.read_parquet(human_path)
    n_human = len(human_df)
    rng = np.random.default_rng(args.random_state)
    tokenize = create_tokenizer()
    records = []
    n_fewshot = min(args.n_fewshot, n_human)

    for _ in tqdm.tqdm(range(args.n_ai_samples), desc="Generating"):
        sample_idx = rng.choice(n_human, size=n_fewshot, replace=False)
        examples = "\n\n".join(
            f"[Example {i + 1}]\n{human_df.iloc[idx]['human_sentence']}"
            for i, idx in enumerate(sample_idx)
        )
        try:
            raw_review = chat_model(
                STAGE1_PROMPT.format(examples=examples),
                args.api_model,
                args.temperature,
                args.max_tokens,
            )
            condensed = chat_model(
                STAGE2_PROMPT.format(review=raw_review),
                args.api_model,
                0.7,
                args.max_tokens,
            )
        except Exception as exc:
            print(f"API call failed: {exc}")
            time.sleep(5)
            continue

        sentences = tokenize(condensed)
        if sentences:
            records.append({"ai_sentence": sentences})
        time.sleep(1)

    pd.DataFrame(records).to_parquet(ai_path, index=False)
    print(f"Saved {len(records)}/{args.n_ai_samples} generated reviews to: {ai_path}")


if __name__ == "__main__":
    main()
