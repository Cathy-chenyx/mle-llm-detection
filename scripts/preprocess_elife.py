"""Preprocess eLife peer reviews for the MLE pipeline.

PROJECT_DIR may be set explicitly; otherwise the repository root is inferred
from this script's location.
"""

import os
import re
from pathlib import Path

import pandas as pd
import spacy

PROJECT_DIR = Path(
    os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[1])
).expanduser().resolve()
DATA_DIR = PROJECT_DIR / "review_text_export"
OUT_DIR = PROJECT_DIR / "processed_data" / "elife"
INPUT_FILE = Path(
    os.environ.get("ELIFE_INPUT_FILE", DATA_DIR / "eLife_reviews_2021_2024.csv")
).expanduser().resolve()
INFERENCE_DIR = OUT_DIR / "inference_data"
HUMAN_CORPUS_DIR = OUT_DIR / "human_corpus"
INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
HUMAN_CORPUS_DIR.mkdir(parents=True, exist_ok=True)

CHATGPT_CUTOFF = (2022, 11)


def clean_html(text):
    text = re.sub(r"<[^>]+>", "", str(text))
    return (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&nbsp;", " ")
    )


def clean_text(text):
    text = clean_html(text)
    text = re.sub(r"\[Editors.? note:.*?\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[#.*?\]", "", text)
    for header in [
        r"^Acceptance summary:\s*",
        r"^Decision letter after peer review:\s*",
        r"^Summary:\s*",
        r"^Peer Review File\s*",
    ]:
        text = re.sub(header, "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


nlp = spacy.load("en_core_web_lg")


def tokenize(text):
    sentence_list = []
    for sent in nlp(text.replace("\n", " ")).sents:
        words = [w for w in re.findall(r"\b\w+\b", sent.text.lower()) if not w.isdigit()]
        if words:
            sentence_list.append(words)
    return sentence_list


def main():
    print(f"Project: {PROJECT_DIR}")
    print(f"Reading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    df["cleaned_text"] = df["review_text"].apply(clean_text)
    df = df[df["cleaned_text"].str.len() > 50].copy()
    df["inference_sentence"] = df["cleaned_text"].apply(tokenize)

    for (year, month), group in df.groupby(["year", "month"]):
        group[["inference_sentence"]].to_parquet(
            INFERENCE_DIR / f"{year}_{month}.parquet", index=False
        )

    human_mask = (
        (df["year"] < CHATGPT_CUTOFF[0])
        | ((df["year"] == CHATGPT_CUTOFF[0]) & (df["month"] < CHATGPT_CUTOFF[1]))
    )
    human_df = df[human_mask].rename(columns={"inference_sentence": "human_sentence"})
    human_path = HUMAN_CORPUS_DIR / "elife_human.parquet"
    human_df[["human_sentence"]].to_parquet(human_path, index=False)

    print(f"Processed reviews: {len(df)}")
    print(f"Human reference reviews: {len(human_df)}")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
