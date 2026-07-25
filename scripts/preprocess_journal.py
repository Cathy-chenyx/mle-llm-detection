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
    parser.add_argument("--date_col", default="review_date",
                        help="用于截取年月的日期列名（默认 review_date）")
    parser.add_argument("--cutoff_year", default=2022, type=int,
                        help="ChatGPT 截止年份（默认 2022）")
    parser.add_argument("--cutoff_month", default=11, type=int,
                        help="ChatGPT 截止月份（默认 11，即 ≤2022.10 的为 H 语料）")
    parser.add_argument("--h_sample_size", default=40, type=int,
                        help="H 语料采样条数（默认 40）")
    return parser.parse_args()


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
def create_tokenizer():
    nlp = spacy.load("en_core_web_lg")

    def tokenize(text):
        text = text.replace('\n', ' ')
        sentence_list = []
        doc = nlp(text)
        for sent in doc.sents:
            words = re.findall(r'\b\w+\b', sent.text.lower())
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
        # 按字符数降序取前 N（获取内容最丰富的审稿）
        human_df['_char_len'] = human_df['cleaned_text'].str.len()
        human_df = human_df.sort_values('_char_len', ascending=False).head(args.h_sample_size)
        human_df = human_df.drop(columns=['_char_len'])
        print(f"  采样 {args.h_sample_size} 条（按字符数降序）")

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
