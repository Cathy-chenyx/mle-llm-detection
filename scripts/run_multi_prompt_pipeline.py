"""Run A/B/C/D reference-corpus sensitivity analyses with portable paths."""

import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
from src.MLE import MLE

PROJECT_DIR = Path(
    os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[1])
).expanduser().resolve()
ELIFE_DIR = PROJECT_DIR / "processed_data" / "elife"
H_PATH = ELIFE_DIR / "human_corpus" / "elife_human.parquet"
AI_DIR = ELIFE_DIR / "ai_corpus"
INFERENCE_DIR = ELIFE_DIR / "inference_data"
DIST_DIR = ELIFE_DIR / "distribution"
OUTPUT_DIR = ELIFE_DIR
DIST_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LEVELS = ["A", "B", "C", "D"]

nlp = spacy.load("en_core_web_lg")


def tokenize(text):
    out = []
    for sent in nlp(str(text).replace("\n", " ")).sents:
        words = [w for w in re.findall(r"\b\w+\b", sent.text.lower()) if not w.isdigit()]
        if words:
            out.append(words)
    return out


def to_docs(series):
    result = []
    for item in series:
        if isinstance(item, (list, np.ndarray)):
            result.append([list(s) for s in item])
        else:
            result.append([])
    return result


def build_distribution(h_docs, q_texts, level):
    q_docs = [tokenize(text) for text in q_texts]
    n = min(len(h_docs), len(q_docs))
    h_flat = pd.DataFrame({"human_sentence": [s for doc in h_docs[:n] for s in doc]}).dropna()
    q_flat = pd.DataFrame({"ai_sentence": [s for doc in q_docs[:n] for s in doc]}).dropna()

    h_counts = count_human_binary_word_occurrences(h_flat)
    q_counts = count_ai_binary_word_occurrences(q_flat)
    h_log = estimate_log_probabilities(h_counts, len(h_flat))
    q_log = estimate_log_probabilities(q_counts, len(q_flat))
    common = get_vocabulary_intersection(h_counts, q_counts)
    vocab = common.intersection(
        filter_frequent_words(h_counts, 5).keys(),
        filter_frequent_words(q_counts, 3).keys(),
    )
    dist_df = calculate_log_probability(h_log, q_log, vocab)
    path = DIST_DIR / f"elife_{level}.parquet"
    dist_df.to_parquet(path, index=False)
    return path, len(dist_df)


def run_mle(dist_path, level):
    model = MLE(str(dist_path))
    results = []
    for parquet_file in sorted(INFERENCE_DIR.glob("*.parquet")):
        try:
            year, month = parquet_file.stem.split("_")
            alpha, ci = model.inference(str(parquet_file), exploded_data=False)
            results.append(
                {"year": int(year), "month": int(month), "alpha": alpha, "ci": ci}
            )
        except Exception as exc:
            print(f"[{level}] {parquet_file.name}: failed - {exc}")
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / f"elife_alpha_{level}.csv", index=False)
    return df


def plot_comparison(all_results):
    labels = {
        "A": "A: Proofread",
        "B": "B: Rewrite",
        "C": "C: Outline → Review",
        "D": "D: Independent reviewer-style",
    }
    fig, ax = plt.subplots(figsize=(12, 6))
    for level, df in all_results.items():
        if df.empty:
            continue
        df = df.sort_values(["year", "month"]).copy()
        x = np.arange(len(df))
        alpha = df["alpha"].to_numpy() * 100
        ci = df["ci"].to_numpy() * 100
        ax.plot(x, alpha, "o-", label=labels[level])
        ax.fill_between(x, alpha - ci, alpha + ci, alpha=0.1)
    ax.set_xlabel("Observed month index")
    ax.set_ylabel("Estimated alpha (%)")
    ax.set_title("eLife — Multi-Prompt Sensitivity Analysis")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = OUTPUT_DIR / "elife_multi_prompt_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    df_h = pd.read_parquet(H_PATH)
    h_docs = to_docs(df_h["human_sentence"])
    all_results = {}

    for level in LEVELS:
        q_path = AI_DIR / f"elife_ai_{level}.parquet"
        if not q_path.exists():
            print(f"[{level}] missing reference corpus: {q_path}")
            continue
        q_texts = pd.read_parquet(q_path)["ai_review_text"].tolist()
        dist_path, vocab_size = build_distribution(h_docs, q_texts, level)
        print(f"[{level}] vocabulary size: {vocab_size}")
        all_results[level] = run_mle(dist_path, level)

    if all_results:
        print(f"Comparison plot: {plot_comparison(all_results)}")


if __name__ == "__main__":
    main()
