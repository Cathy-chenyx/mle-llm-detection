"""Generate A/B/C/D AI reference corpora for sensitivity analysis.

Required environment variables:
  DEEPSEEK_API_KEY
  DEEPSEEK_BASE_URL

Optional:
  DEEPSEEK_MODEL (default: deepseek-v4-flash)
  PROJECT_DIR    (default: repository root)
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

PROMPTS = {
    "A": {
        "name": "proofread",
        "temperature": 0.3,
        "max_tokens": 16384,
        "prompt": lambda text: (
            "Proofread the following peer review. Fix ONLY spelling, grammar, and punctuation errors. "
            "Do not change wording, structure, or content. Output only the corrected review.\n\n"
            f"{text[:8000]}"
        ),
    },
    "B": {
        "name": "rewrite",
        "temperature": 0.5,
        "max_tokens": 4096,
        "prompt": lambda text: (
            "Rewrite the following peer review while preserving all factual content and criticisms. "
            "Change wording and sentence structure, use formal academic language, and add no new facts. "
            "Output only the rewritten review.\n\n"
            f"{text[:8000]}"
        ),
    },
    "D": {
        "name": "act_as_reviewer",
        "temperature": 0.7,
        "max_tokens": 4096,
        "prompt": lambda text: (
            "Write an independent academic peer review using the following review only as contextual reference. "
            "Do not copy its structure or wording. Cover methodological, statistical, and interpretive issues. "
            "Output only the new review.\n\n"
            f"{text[:8000]}"
        ),
    },
}


def reconstruct_text(sentence_lists):
    sentences = []
    for sent in sentence_lists:
        if isinstance(sent, (list, np.ndarray)):
            sentences.append(" ".join(str(w) for w in sent))
        elif isinstance(sent, str):
            sentences.append(sent)
    return " ".join(sentences)


def call_model(prompt, temperature, max_tokens):
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def generate_level_c(text):
    outline = call_model(
        "Extract the key factual points from this peer review as a numbered bullet list. "
        "Strip stylistic language and add no new information. Output only the bullet list.\n\n"
        f"{text[:8000]}",
        0.3,
        4096,
    )
    return call_model(
        "Expand the following bullet outline into a full formal peer-review paragraph. "
        "Cover every bullet, add no new factual content, and output only the review.\n\n"
        f"{outline}",
        0.7,
        4096,
    )


def main():
    df_h = pd.read_parquet(HUMAN_PATH)
    df_h["human_text"] = df_h["human_sentence"].apply(reconstruct_text)
    texts = df_h["human_text"].tolist()

    for level in ["A", "B", "C", "D"]:
        outputs = []
        for idx, text in enumerate(texts, start=1):
            try:
                if level == "C":
                    generated = generate_level_c(text)
                else:
                    cfg = PROMPTS[level]
                    generated = call_model(
                        cfg["prompt"](text), cfg["temperature"], cfg["max_tokens"]
                    )
                outputs.append(generated)
            except Exception as exc:
                print(f"[{level} {idx}/{len(texts)}] failed: {exc}")
            if idx < len(texts):
                time.sleep(0.5)

        out_path = AI_OUT_DIR / f"elife_ai_{level}.parquet"
        pd.DataFrame({"ai_review_text": outputs}).to_parquet(out_path, index=False)
        print(f"Level {level}: saved {len(outputs)}/{len(texts)} to {out_path}")


if __name__ == "__main__":
    main()
