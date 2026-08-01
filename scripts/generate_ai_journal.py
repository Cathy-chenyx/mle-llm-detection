"""
DeepSeek AI 审稿意见生成脚本（通用版）
=======================================
功能：读入期刊 human_corpus 的 H 语料，用两阶段 API 生成"伪 AI 审稿意见"的 tokenized 版本

用法：
  python generate_ai_journal.py --journal_name elife
  python generate_ai_journal.py --journal_name plos_one --project_dir /mnt/data/hermes-workspace/mle-llm-detection
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import requests
import time
import tqdm
import hashlib
import json
import re
import spacy


def parse_args():
    parser = argparse.ArgumentParser(description="AI 审稿意见生成")
    parser.add_argument("--journal_name", required=True,
                        help="期刊名称，如 elife")
    parser.add_argument("--project_dir", default=None,
                        help="项目根目录（默认自动检测）")
    parser.add_argument("--api_model", default="deepseek-v4-flash",
                        help="DeepSeek 模型（默认 deepseek-v4-flash）")
    parser.add_argument("--temperature", default=1.5, type=float,
                        help="API temperature（默认 1.5，高变异性）")
    parser.add_argument("--max_tokens", default=4096, type=int,
                        help="API max_tokens（默认 8192）")
    parser.add_argument("--n_ai_samples", default=100, type=int,
                        help="AI 语料目标条数（默认 100，每轮 few-shot 随机抽取 10 条 H）")
    parser.add_argument("--random_state", default=42, type=int,
                        help="随机种子（默认 42）")
    parser.add_argument("--n_fewshot", default=10, type=int,
                        help="每轮 few-shot 示例数（默认 10）")
    return parser.parse_args()


# ============================================================
# DeepSeek API 调用
# ============================================================
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not API_KEY:
    raise RuntimeError("请设置环境变量 DEEPSEEK_API_KEY")

API_BASE = "http://115.190.192.101:3000/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


def chat_deepseek(prompt, model="deepseek-v4-flash", temperature=1.5, max_tokens=8192):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a peer reviewer for an academic journal. Write in fluent academic English."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    resp = requests.post(API_BASE, headers=HEADERS, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ============================================================
# spaCy 分词
# ============================================================
AI_MAX_TEXT_LENGTH = 80_000_000


def create_tokenizer():
    nlp = spacy.load("en_core_web_lg")
    nlp.max_length = AI_MAX_TEXT_LENGTH

    def tokenize(text):
        text = text.replace('\n', ' ')
        if len(text) > AI_MAX_TEXT_LENGTH:
            text = text[:AI_MAX_TEXT_LENGTH]

        sentence_list = []
        try:
            doc = nlp(text)
            for sent in doc.sents:
                words = re.findall(r'\b\w+\b', sent.text.lower())
                words = [w for w in words if not w.isdigit()]
                if len(words) > 0:
                    sentence_list.append(words)
        except Exception:
            for chunk in re.split(r'(?<=[.!?])\s+', text):
                words = re.findall(r'\b\w+\b', chunk.lower())
                words = [w for w in words if not w.isdigit()]
                if len(words) > 0:
                    sentence_list.append(words)
        return sentence_list

    return tokenize


# ============================================================
# Prompt 模板（两级生成，与 generate_ai_corpus.py 一致）
# ============================================================
STAGE1_PROMPT = """You are simulating a peer reviewer for an academic journal. Below are some example human peer review excerpts. Your task is to write ONE new peer review that mimics the style, tone, and format of these examples, but for a DIFFERENT (imaginary) paper in a similar field. Do NOT copy content from the examples — invent a new review with its own observations, critiques, and suggestions.

The review should be 5-8 paragraphs long and include:
1. A brief summary of the paper
2. Major critique points (both strengths and weaknesses)
3. Minor issues (technical details, formatting, figures)
4. Overall recommendation

Example human reviews:
---
{examples}
---

Now write ONE new peer review in the same style:"""


STAGE2_CONDENSE = """Below is a detailed AI-generated peer review. Please condense it into exactly 5-7 distinct paragraphs, preserving all academic vocabulary, hedging language, and formal tone. Keep each paragraph self-contained with its own topic. Do NOT merge paragraphs — each paragraph should be separated by two newlines.

Original review:
---
{review}
---

Condensed review (5-7 paragraphs, each separated by two newlines):"""


# ============================================================
# 主流程
# ============================================================
def main():
    args = parse_args()
    journal = args.journal_name

    if args.project_dir:
        project_dir = Path(args.project_dir)
    else:
        project_dir = Path(__file__).resolve().parent.parent

    human_path = project_dir / "processed_data" / journal / "human_corpus" / f"{journal}_human.parquet"
    ai_dir = project_dir / "processed_data" / journal / "ai_corpus"
    ai_path = ai_dir / f"{journal}_ai.parquet"
    ai_dir.mkdir(parents=True, exist_ok=True)

    # ---- 读取 H 语料 ----
    print(f"读取 H 语料: {human_path}")
    human_df = pd.read_parquet(human_path)
    n_human = len(human_df)
    print(f"  H 语料条目数: {n_human}")

    # ---- 生成 AI 审稿 ----
    print(f"\n目标: {args.n_ai_samples} 条 AI 审稿 (seed={args.random_state})")
    rng = np.random.default_rng(args.random_state)
    tokenize = create_tokenizer()
    ai_records = []

    # 打散 H 语料索引，循环使用
    h_indices = rng.permutation(n_human)
    n_fewshot = min(args.n_fewshot, n_human)

    for i in tqdm.tqdm(range(args.n_ai_samples), desc="Generating"):
        # 每轮随机抽 n_fewshot 条 H 作为 few-shot 示例
        fewshot_mask = rng.choice(n_human, size=n_fewshot, replace=False)
        examples = ""
        for j, mask_idx in enumerate(fewshot_mask):
            row = human_df.iloc[mask_idx]
            examples += f"[Example {j+1}]\n{row['human_sentence']}\n\n"

        prompt = STAGE1_PROMPT.replace("{examples}", examples)

        # Stage 1
        try:
            raw_review = chat_deepseek(prompt, model=args.api_model,
                                       temperature=args.temperature,
                                       max_tokens=args.max_tokens)
        except Exception as e:
            print(f"\n  [第 {i+1}/{args.n_ai_samples} 条] Stage1 API 失败: {e}")
            time.sleep(5)
            continue

        # Stage 2
        try:
            condense_prompt = STAGE2_CONDENSE.replace("{review}", raw_review)
            condensed = chat_deepseek(condense_prompt, model=args.api_model,
                                      temperature=0.7, max_tokens=args.max_tokens)
        except Exception as e:
            print(f"\n  [第 {i+1}/{args.n_ai_samples} 条] Stage2 API 失败: {e}")
            time.sleep(5)
            continue

        ai_sentences = tokenize(condensed)
        if len(ai_sentences) > 0:
            ai_records.append({"ai_sentence": ai_sentences})
        else:
            print(f"\n  [警告 第 {i+1}/{args.n_ai_samples} 条] 分词结果为空，跳过")

        time.sleep(1)

        # 每 20 条打印进度
        if (i + 1) % 20 == 0:
            total_sents = sum(len(r['ai_sentence']) for r in ai_records)
            print(f"\n  [进度] {i+1}/{args.n_ai_samples} 条, "
                  f"成功 {len(ai_records)} 条, {total_sents} 句")

    # ---- 保存 ----
    print(f"\n生成 AI 条目: {len(ai_records)}/{args.n_ai_samples}")
    ai_df = pd.DataFrame(ai_records)
    ai_df.to_parquet(ai_path, index=False)
    print(f"✅ AI 语料保存至: {ai_path}")


if __name__ == "__main__":
    main()
