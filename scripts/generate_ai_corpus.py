"""Generate an AI reference corpus (Q) for the eLife pilot.

Configuration is read from environment variables so the script is portable and
no private infrastructure details are committed to the repository.

Required:
  DEEPSEEK_API_KEY

Optional:
  DEEPSEEK_BASE_URL   Anthropic-compatible base URL
  DEEPSEEK_MODEL      model name (default: deepseek-v4-flash)
  PROJECT_DIR         repository root (default: auto-detected)
"""

import os
import time
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd


PROJECT_DIR = Path(
    os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[1])
).expanduser().resolve()

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")

if not API_KEY:
    raise RuntimeError("Set DEEPSEEK_API_KEY before running this script.")
if not BASE_URL:
    raise RuntimeError("Set DEEPSEEK_BASE_URL before running this script.")

client = anthropic.Anthropic(api_key=API_KEY, base_url=BASE_URL, timeout=120.0)

HUMAN_PATH = PROJECT_DIR / "processed_data/elife/human_corpus/elife_human.parquet"
AI_OUT_DIR = PROJECT_DIR / "processed_data/elife/ai_corpus"
AI_OUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_SIZE = None


def reconstruct_text(sentence_lists):
    """Reconstruct plain text from tokenized sentence lists."""
    sentences = []
    for sent in sentence_lists:
        if isinstance(sent, (list, np.ndarray)):
            sentences.append(" ".join(str(w) for w in sent))
        elif isinstance(sent, str):
            sentences.append(sent)
    return " ".join(sentences)


def generate_ai_review(human_text, idx, total):
    """Generate one two-stage AI reference review."""
    text_truncated = human_text[:8000]
    stage1_prompt = (
        "Extract the key factual points from this peer review as a numbered bullet list. "
        "Output ONLY the bullet list, no preamble.\n\n"
        f"{text_truncated}"
    )
    try:
        resp1 = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            temperature=0.3,
            messages=[{"role": "user", "content": stage1_prompt}],
        )
        outline = resp1.content[0].text.strip()
    except Exception as exc:
        print(f"[{idx}/{total}] Stage 1 failed: {exc}")
        return None

    stage2_prompt = (
        "Expand the following bullet outline into a full peer review paragraph. "
        "Use formal academic language and complete sentences. Cover all bullets in order. "
        "Do not add new factual content or fabricate citations or data. "
        "Output ONLY the review paragraph, no preamble.\n\n"
        f"{outline}"
    )
    try:
        resp2 = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            temperature=0.7,
            messages=[{"role": "user", "content": stage2_prompt}],
        )
        return resp2.content[0].text.strip()
    except Exception as exc:
        print(f"[{idx}/{total}] Stage 2 failed: {exc}")
        return None


def main():
    print(f"Model: {MODEL}")
    print(f"Project: {PROJECT_DIR}")
    print(f"Loading H corpus: {HUMAN_PATH}")

    df_h = pd.read_parquet(HUMAN_PATH)
    if SAMPLE_SIZE and SAMPLE_SIZE < len(df_h):
        df_h = df_h.sample(n=SAMPLE_SIZE, random_state=42)

    df_h["human_text"] = df_h["human_sentence"].apply(reconstruct_text)
    total = len(df_h)
    ai_texts = []

    for _, row in df_h.iterrows():
        idx = len(ai_texts) + 1
        ai_text = generate_ai_review(row["human_text"], idx, total)
        if ai_text:
            ai_texts.append(ai_text)
        if idx < total:
            time.sleep(0.5)

    if not ai_texts:
        raise RuntimeError("No AI reference texts were generated.")

    out_path = AI_OUT_DIR / "elife_ai.parquet"
    pd.DataFrame({"ai_review_text": ai_texts}).to_parquet(out_path, index=False)
    print(f"Saved {len(ai_texts)}/{total} AI reference reviews to: {out_path}")


if __name__ == "__main__":
    main()
