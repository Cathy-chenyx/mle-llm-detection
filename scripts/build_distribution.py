"""Build the eLife human/AI word-occurrence distribution used by the MLE model."""

import os
import re
import sys
from pathlib import Path

import pandas as pd
import spacy

sys.path.insert(0, str(Path(__file__).parent))
from src.estimation import (
    calculate_log_probability,
    count_ai_binary_word_occurrences,
    count_human_binary_word_occurrences,
    estimate_log_probabilities,
    filter_frequent_words,
    get_vocabulary_intersection,
)

PROJECT_DIR = Path(
    os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[1])
).expanduser().resolve()
ELIFE_DIR = PROJECT_DIR / "processed_data" / "elife"
H_PATH = ELIFE_DIR / "human_corpus" / "elife_human.parquet"
Q_RAW_PATH = ELIFE_DIR / "ai_corpus" / "elife_ai.parquet"
DIST_OUT_DIR = ELIFE_DIR / "distribution"
DIST_OUT_DIR.mkdir(parents=True, exist_ok=True)

nlp = spacy.load("en_core_web_lg")


def tokenize(text):
    out = []
    for sent in nlp(str(text).replace("\n", " ")).sents:
        words = [w for w in re.findall(r"\b\w+\b", sent.text.lower()) if not w.isdigit()]
        if words:
            out.append(words)
    return out


def to_list_of_lists(series):
    result = []
    for item in series:
        sentences = [list(s) for s in item]
        result.append([s for s in sentences if len(s) > 1])
    return result


def main():
    df_h = pd.read_parquet(H_PATH)
    df_q = pd.read_parquet(Q_RAW_PATH)
    df_q["ai_sentence"] = df_q["ai_review_text"].apply(tokenize)

    n = min(len(df_h), len(df_q))
    h_docs = to_list_of_lists(df_h["human_sentence"].iloc[:n])
    q_docs = to_list_of_lists(df_q["ai_sentence"].iloc[:n])

    h_flat = pd.DataFrame({"human_sentence": [s for doc in h_docs for s in doc]}).dropna()
    q_flat = pd.DataFrame({"ai_sentence": [s for doc in q_docs for s in doc]}).dropna()

    human_counts = count_human_binary_word_occurrences(h_flat)
    ai_counts = count_ai_binary_word_occurrences(q_flat)
    human_log = estimate_log_probabilities(human_counts, len(h_flat))
    ai_log = estimate_log_probabilities(ai_counts, len(q_flat))

    common = get_vocabulary_intersection(human_counts, ai_counts)
    frequent_h = filter_frequent_words(human_counts, 5)
    frequent_q = filter_frequent_words(ai_counts, 3)
    vocab = common.intersection(frequent_h.keys(), frequent_q.keys())

    dist_df = calculate_log_probability(human_log, ai_log, vocab)
    out_path = DIST_OUT_DIR / "elife.parquet"
    dist_df.to_parquet(out_path, index=False)
    print(f"Saved {len(dist_df)} words to: {out_path}")


if __name__ == "__main__":
    main()
