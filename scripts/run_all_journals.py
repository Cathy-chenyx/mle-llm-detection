"""Portable multi-journal orchestration for the peer-review MLE pipeline.

This script intentionally avoids private infrastructure paths. It expects a local
SQLite database path to be supplied through REVIEW_DB_PATH and uses repository-
relative output paths by default.

Required environment variables:
  REVIEW_DB_PATH       path to the review-unit SQLite database

For AI-corpus generation also set:
  DEEPSEEK_API_KEY
  DEEPSEEK_BASE_URL

Optional:
  PROJECT_DIR          repository root (default: auto-detected)
  DEEPSEEK_MODEL       default: deepseek-v4-flash
"""

import csv
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
import spacy

PROJECT_DIR = Path(
    os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[1])
).expanduser().resolve()
REVIEW_DB_PATH = os.environ.get("REVIEW_DB_PATH", "")
CHATGPT_CUTOFF = "2022-11-01"

JOURNALS = [
    {"dir": "bmc_med", "db_journal": "BMC Med", "n": 40},
    {"dir": "peerj", "db_journal": "PeerJ", "n": 40},
    {"dir": "f1000research", "db_journal": "F1000Research", "n": 40},
    {"dir": "elife", "db_journal": "eLife", "n": 40},
    {"dir": "nat_commun", "db_journal": "Nat Commun", "n": 40},
]


def require_database():
    if not REVIEW_DB_PATH:
        raise RuntimeError("Set REVIEW_DB_PATH to the local review-unit SQLite database.")
    path = Path(REVIEW_DB_PATH).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def clean_text(text):
    text = re.sub(r"<[^>]+>", "", str(text))
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = re.sub(r"\[Editors.? note:.*?\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[#.*?\]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def make_tokenizer():
    nlp = spacy.load("en_core_web_lg")

    def tokenize(text):
        result = []
        for sent in nlp(str(text).replace("\n", " ")).sents:
            words = [w for w in re.findall(r"\b\w+\b", sent.text.lower()) if not w.isdigit()]
            if words:
                result.append(words)
        return result

    return tokenize


def query_rows(db_path, sql, params):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def export_human_corpus(db_path, cfg, tokenize):
    rows = query_rows(
        db_path,
        """
        SELECT unit_id, content, publication_date, review_date
        FROM derived_review_units_v1
        WHERE journal = ?
          AND content IS NOT NULL
          AND LENGTH(content) > 50
          AND publication_date >= '2021-01-01'
          AND publication_date < ?
        ORDER BY LENGTH(content) DESC
        LIMIT ?
        """,
        (cfg["db_journal"], CHATGPT_CUTOFF, cfg["n"]),
    )
    if not rows:
        return 0

    out_dir = PROJECT_DIR / "review_text_export"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{cfg['dir']}_human_{cfg['n']}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["unit_id", "review_text", "publication_date", "review_date"])
        writer.writerows(rows)

    df = pd.DataFrame(rows, columns=["unit_id", "review_text", "publication_date", "review_date"])
    df["cleaned_text"] = df["review_text"].apply(clean_text)
    df = df[df["cleaned_text"].str.len() > 50].copy()
    df["human_sentence"] = df["cleaned_text"].apply(tokenize)

    human_dir = PROJECT_DIR / "processed_data" / cfg["dir"] / "human_corpus"
    human_dir.mkdir(parents=True, exist_ok=True)
    df[["human_sentence"]].to_parquet(human_dir / f"{cfg['dir']}_human.parquet", index=False)
    return len(df)


def export_inference_data(db_path, cfg, tokenize):
    rows = query_rows(
        db_path,
        """
        SELECT unit_id, content, publication_date, review_date
        FROM derived_review_units_v1
        WHERE journal = ? AND content IS NOT NULL
        ORDER BY publication_date
        """,
        (cfg["db_journal"],),
    )
    if not rows:
        return 0

    df = pd.DataFrame(rows, columns=["unit_id", "review_text", "publication_date", "review_date"])
    df["month_key"] = df.apply(
        lambda row: (
            str(row["publication_date"])[:7]
            if pd.notna(row["publication_date"])
            else (str(row["review_date"])[:7] if pd.notna(row["review_date"]) else None)
        ),
        axis=1,
    )
    df = df.dropna(subset=["month_key"])
    df = df[df["month_key"].str.match(r"^20(2[1-6])-\d{2}$")].copy()
    df["cleaned_text"] = df["review_text"].apply(clean_text)
    df = df[df["cleaned_text"].str.len() > 50].copy()
    df["inference_sentence"] = df["cleaned_text"].apply(tokenize)

    out_dir = PROJECT_DIR / "processed_data" / cfg["dir"] / "inference_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    for month_key, group in df.groupby("month_key"):
        year, month = month_key.split("-")
        group[["inference_sentence"]].to_parquet(out_dir / f"{year}_{month}.parquet", index=False)
    return len(df)


def generate_ai_corpus(cfg):
    env = os.environ.copy()
    env["PROJECT_DIR"] = str(PROJECT_DIR)
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "generate_ai_journal.py"),
            "--journal_name",
            cfg["dir"],
            "--n_ai_samples",
            str(cfg["n"]),
        ],
        env=env,
        check=False,
    )
    return result.returncode == 0


def run_journal_pipeline(cfg):
    env = os.environ.copy()
    env["PROJECT_DIR"] = str(PROJECT_DIR)
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "run_journal_pipeline.py"),
            "--journal_name",
            cfg["dir"],
            "--project_dir",
            str(PROJECT_DIR),
        ],
        env=env,
        check=False,
    )
    return result.returncode == 0


def main():
    db_path = require_database()
    tokenize = make_tokenizer()
    print(f"Project: {PROJECT_DIR}")
    print(f"Review database: {db_path}")

    for cfg in JOURNALS:
        print(f"\n=== {cfg['db_journal']} ===")
        n_h = export_human_corpus(db_path, cfg, tokenize)
        n_i = export_inference_data(db_path, cfg, tokenize)
        print(f"Human reference records: {n_h}; inference records: {n_i}")
        if n_h == 0 or n_i == 0:
            print("Skipping downstream steps because required input is missing.")
            continue
        if not generate_ai_corpus(cfg):
            print("AI corpus generation failed; skipping inference.")
            continue
        if not run_journal_pipeline(cfg):
            print("Journal inference failed.")


if __name__ == "__main__":
    main()
