"""
期刊审稿意见通用预处理脚本
===========================
功能：读取期刊 CSV → 清洗 HTML → spaCy 分句分词 → 按月输出 parquet

用法：
  python preprocess_journal.py --journal_name elife --input_csv /path/to/export.csv
  python preprocess_journal.py --journal_name plos_one --input_csv /path/to/export.csv --date_col review_date

输出：
  1. processed_data/{journal}/inference_data/{year}_{month}.parquet  用于 MLE 推断
  2. processed_data/{journal}/human_corpus/{journal}_human.parquet   ChatGPT 前人类参考语料
"""

import pandas as pd
import numpy as np
import re
import os
import argparse
import spacy
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="期刊审稿意见通用预处理")
    parser.add_argument("--journal_name", required=True,
                        help="期刊名称，如 elife、plos_one（决定输出子目录名）")
    parser.add_argument("--input_csv", required=True,
                        help="从 SQLite 导出的 CSV 文件路径，需包含 content 列")
    parser.add_argument("--project_dir", default=None,
                        help="项目根目录（默认自动检测）")
    parser.add_argument("--text_col", default="content",
                        help="正文列名（默认 content）")
    parser.add_argument("--date_col", default="publication_date",
                        help="用于截取年月的日期列名（默认 publication_date，即发表日期）")
    parser.add_argument("--min_year", default=2021, type=int,
                        help="最小年份（默认 2021，过滤更早数据）")
    parser.add_argument("--max_year", default=2026, type=int,
                        help="最大年份（默认 2026，过滤更晚数据）")
    parser.add_argument("--cutoff_year", default=2022, type=int,
                        help="ChatGPT 截止年份（默认 2022）")
    parser.add_argument("--cutoff_month", default=11, type=int,
                        help="ChatGPT 截止月份（默认 11，即 ≤2022.10 的为 H 语料）")
    parser.add_argument("--h_sample_size", default=40, type=int,
                        help="H 语料采样条数（默认 40）")
    parser.add_argument("--random_state", default=42, type=int,
                        help="随机种子（默认 42，保证可复现）")
    return parser.parse_args()


# ============================================================
# 分层随机抽样（按年份）
# ============================================================
def stratified_sample_by_year(df, n_total, random_state=42):
    """按年份分层随机抽样，各年分配比例与其条目数成正比，至少 1 条/年。"""
    year_counts = df['year'].value_counts().sort_index()
    years = year_counts.index.tolist()
    counts = year_counts.values

    # Step 1: 每层至少分配 1 条（如果该层有数据）
    allocation = np.minimum(counts, 1)  # 每层 1 条
    remaining = n_total - allocation.sum()

    if remaining < 0:
        # 年份数超过 n_total，随机选 n_total 个年份各取 1 条
        sampled_years = np.random.default_rng(random_state).choice(years, n_total, replace=False)
        sampled = []
        for yr in sampled_years:
            yr_group = df[df['year'] == yr]
            sampled.append(yr_group.sample(n=1, random_state=random_state))
        return pd.concat(sampled, ignore_index=True)

    # Step 2: 按比例分配剩余名额（largest remainder method）
    proportions = np.where(counts > 0, counts / counts.sum() * remaining, 0)
    floors = np.floor(proportions).astype(int)
    remainders = proportions - floors
    allocation = allocation + floors

    still_remaining = n_total - allocation.sum()
    # 按余数降序补齐
    order = np.argsort(-remainders)
    for i in range(still_remaining):
        idx = order[i]
        if allocation[idx] < counts[idx]:
            allocation[idx] += 1

    # Step 3: 每层随机抽样
    sampled = []
    for i, yr in enumerate(years):
        if allocation[i] > 0:
            yr_group = df[df['year'] == yr]
            n_yr = min(allocation[i], len(yr_group))
            sampled.append(yr_group.sample(n=n_yr, random_state=random_state))

    result = pd.concat(sampled, ignore_index=True)
    return result


# ============================================================
# 1. 文本清洗（与 preprocess_elife.py 完全一致）
# ============================================================
def clean_html(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&nbsp;', ' ', text)
    return text


def clean_editor_notes(text):
    text = re.sub(r'\[Editors.? note:.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[#.*?\]', '', text)
    return text


def clean_section_headers(text):
    headers = [
        r'^Acceptance summary:\s*',
        r'^Decision letter after peer review:\s*',
        r'^Summary:\s*',
        r'^Peer Review File\s*',
    ]
    for h in headers:
        text = re.sub(h, '', text, flags=re.IGNORECASE | re.MULTILINE)
    return text


def clean_text(text):
    text = str(text)
    text = clean_html(text)
    text = clean_editor_notes(text)
    text = clean_section_headers(text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = text.strip()
    return text


# ============================================================
# 2. spaCy 分词（与 tokenize_demo.ipynb 完全一致）
# ============================================================
MAX_TEXT_LENGTH = 80_000_000  # 80MB 上限（Nat Commun review_bundle 最大 ~48MB）


def create_tokenizer():
    nlp = spacy.load("en_core_web_lg")
    nlp.max_length = MAX_TEXT_LENGTH  # 默认 1M，放开到 80M

    def tokenize(text):
        text = text.replace('\n', ' ')

        # 超长文本分块处理
        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH]

        sentence_list = []
        try:
            doc = nlp(text)
            for sent in doc.sents:
                words = re.findall(r'\b\w+\b', sent.text.lower())
                words = [w for w in words if not w.isdigit()]
                if len(words) > 0:
                    sentence_list.append(words)
        except Exception:
            # 极端情况回退：按句子标点简单切分
            for chunk in re.split(r'(?<=[.!?])\s+', text):
                words = re.findall(r'\b\w+\b', chunk.lower())
                words = [w for w in words if not w.isdigit()]
                if len(words) > 0:
                    sentence_list.append(words)
        return sentence_list

    return tokenize, nlp


# ============================================================
# 3. 日期提取
# ============================================================
def extract_year_month(df, date_col):
    """从日期列提取 year 和 month"""
    dates = pd.to_datetime(df[date_col], errors='coerce')
    df = df.copy()
    df['year'] = dates.dt.year
    df['month'] = dates.dt.month
    before = len(df)
    df = df.dropna(subset=['year', 'month'])
    if len(df) < before:
        print(f"  警告：{before - len(df)} 行无法解析日期，已丢弃")
    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].astype(int)
    return df


# ============================================================
# 主流程
# ============================================================
def main():
    args = parse_args()
    journal = args.journal_name

    # 项目目录
    if args.project_dir:
        project_dir = Path(args.project_dir)
    else:
        # 自动检测：脚本所在目录的上两级
        project_dir = Path(__file__).resolve().parent.parent

    data_dir = project_dir / "processed_data" / journal
    inference_dir = data_dir / "inference_data"
    human_dir = data_dir / "human_corpus"
    inference_dir.mkdir(parents=True, exist_ok=True)
    human_dir.mkdir(parents=True, exist_ok=True)

    # ---- 读取数据 ----
    print(f"读取数据: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    print(f"原始条目数: {len(df)}")
    print(f"列名: {df.columns.tolist()}")

    # 检查必要列
    if args.text_col not in df.columns:
        raise ValueError(f"未找到正文列 '{args.text_col}'，可用列: {df.columns.tolist()}")
    if args.date_col not in df.columns:
        raise ValueError(f"未找到日期列 '{args.date_col}'，可用列: {df.columns.tolist()}")

    # ---- 提取年月 ----
    print(f"从 '{args.date_col}' 提取年月...")
    df = extract_year_month(df, args.date_col)
    print(f"有效日期条目: {len(df)}，年份范围: {df['year'].min()}-{df['year'].max()}")

    # ---- 时间范围过滤（会议决定：限定 2021-2026）----
    before_range = len(df)
    df = df[(df['year'] >= args.min_year) & (df['year'] <= args.max_year)].copy()
    print(f"时间范围限定 {args.min_year}-{args.max_year}: {before_range} → {len(df)} (移除 {before_range - len(df)} 条)")
    if len(df) == 0:
        raise ValueError(f"时间范围 {args.min_year}-{args.max_year} 内无数据，请检查过滤条件")

    # ---- 清洗文本 ----
    print("清洗 HTML 标签和编辑器注释...")
    df['cleaned_text'] = df[args.text_col].apply(clean_text)

    before = len(df)
    df = df[df['cleaned_text'].str.len() > 50].copy()
    print(f"清洗后过滤空条目: {before} → {len(df)} (移除 {before - len(df)} 条)")

    # ---- 分词 ----
    print("加载 spaCy 模型并分词...（可能需要几分钟）")
    tokenize, _ = create_tokenizer()
    df['inference_sentence'] = df['cleaned_text'].apply(tokenize)

    total_sentences = df['inference_sentence'].apply(len).sum()
    total_tokens = df['inference_sentence'].apply(lambda x: sum(len(s) for s in x)).sum()
    print(f"总计: {total_sentences} 句, {total_tokens} 词")

    # ---- 保存推断数据：按年月分组 ----
    print("\n--- 保存月度推断数据 ---")
    n_months = 0
    for (year, month), group in df.groupby(['year', 'month']):
        out_path = inference_dir / f"{year}_{month}.parquet"
        group[['inference_sentence']].to_parquet(out_path, index=False)
        n_months += 1
        print(f"  {year}_{month}: {len(group)} 条 → {out_path.name}")
    print(f"共 {n_months} 个月份")

    # ---- 保存人类参考语料 ----
    print(f"\n--- 生成人类参考语料 (≤{args.cutoff_year}.{args.cutoff_month:02d}) ---")
    human_mask = (
        (df['year'] < args.cutoff_year) |
        ((df['year'] == args.cutoff_year) & (df['month'] < args.cutoff_month))
    )
    human_df = df[human_mask].copy()
    print(f"  pre-ChatGPT 审稿总数: {len(human_df)} 条")

    if len(human_df) > args.h_sample_size:
        # 按年份分层随机抽样（替代原来的"按字符数降序取前 N"）
        human_df = stratified_sample_by_year(human_df, args.h_sample_size, random_state=args.random_state)
        print(f"  分层随机抽样 {args.h_sample_size} 条（按年份比例分配，seed={args.random_state}）")
        year_alloc = human_df['year'].value_counts().sort_index()
        print(f"  各年分配: {dict(year_alloc)}")

    human_df = human_df.rename(columns={'inference_sentence': 'human_sentence'})
    human_path = human_dir / f"{journal}_human.parquet"
    human_df[['human_sentence']].to_parquet(human_path, index=False)
    print(f"  H 语料: {len(human_df)} 条 → {human_path}")

    # ---- 年份分布总览 ----
    print(f"\n--- {journal} 年份分布 ---")
    summary = df.groupby('year').agg(
        条目数=('cleaned_text', 'count'),
        平均字数=('cleaned_text', lambda x: int(x.str.len().mean())),
        总句子数=('inference_sentence', lambda x: x.apply(len).sum())
    )
    print(summary.to_string())

    print(f"\n✅ {journal} 预处理完成。输出文件：")
    print(f"  月度推断数据: {inference_dir}/")
    print(f"  人类参考语料: {human_path}")


if __name__ == "__main__":
    main()
